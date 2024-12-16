# PyZefir
# Copyright (C) 2024 Narodowe Centrum Badań Jądrowych
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import logging

import numpy as np
import xarray as xr
from linopy import LinearExpression, Variable

from pyzefir.optimization.linopy.constraints_builder.builder import (
    PartialConstraintsBuilder,
)

_logger = logging.getLogger(__name__)


class BalancingConstraintsBuilder(PartialConstraintsBuilder):
    """
    A class responsible for building energy balancing and load shifting constraints
    in a power network optimization model.

    This class generates constraints to ensure that energy supply matches demand
    at each bus in the network, taking into account various inflows, outflows,
    generation, storage, and load-shifting mechanisms. It supports balancing for
    buses, demand-side resources (DSR), and demand chunks.
    """

    def build_constraints(self) -> None:
        """
        Builds constraints including:
        - balancing constraints
        - demand chunk balancing constraints
        - load shifting constraints
        """
        _logger.info("Balancing constraints builder is working...")
        self.balancing_constraint()
        self.demand_chunk_balancing_constraint()
        self.load_shifting_constraints()
        _logger.info("Balancing constraints builder is finished!")

    def balancing_constraint(self) -> None:
        """
        Add balancing constraints for each bus in the model.

        Calculates net load, outflow, injection, and inflow for each bus,
        then adds a constraint ensuring that above values form the equation.
        """
        _logger.debug("Building balancing constraints...")
        for bus_idx, bus_name in self.indices.BUS.mapping.items():
            net_load = self._bus_net_load(bus_idx)
            net_outflow = self._bus_outflow(bus_idx)
            net_injection = self._bus_net_injection(bus_idx)
            net_inflow = self._bus_net_inflow(bus_idx)
            ens = self.variables.bus.bus_ens.isel(bus=bus_idx)
            shift = self._shift(bus_idx)

            self.model.add_constraints(
                shift + net_load + net_outflow == ens + net_inflow + net_injection,
                name=f"{bus_name}_BALANCING_CONSTRAINT",
            )
        _logger.debug("Build balancing constraints: Done")

    def load_shifting_constraints(self) -> None:
        """
        Load and add shifting constraints for demand-side resources (DSR).

        For each bus with a DSR type, applies constraints related to
        shifting, including relative and absolute limits, and generates
        compensation constraints for defined balancing periods.
        """
        _logger.debug("Loading shifting constraints...")
        if len(self.parameters.bus.dsr_type):
            balancing_periods = self.parameters.dsr.balancing_periods
            for bus_idx, dsr_idx in self.parameters.bus.dsr_type.items():
                if bus_idx in self.parameters.bus.lbs_mapping:
                    self._shift_plus_relative_hourly_limit_constraint(bus_idx, dsr_idx)
                    self._shift_minus_relative_hourly_limit_constraint(bus_idx, dsr_idx)
                    intervals = balancing_periods[dsr_idx]
                    for interval in intervals:
                        self._gen_compensation_constraint(interval, bus_idx, dsr_idx)
                        self._gen_relative_shift_limit(interval, bus_idx, dsr_idx)
                        self._gen_abs_shift_limit(interval, bus_idx, dsr_idx)
        _logger.debug("Load shifting constraints: Done")

    def _shift(self, bus_idx: int) -> LinearExpression | float:
        """
        Shift values for buses.

        Args:
            - bus_idx: index of the bus to shift

        Returns:
            - LinearExpression or float: linear expression of the shifted bus
        """
        _logger.debug("Shifting values for bus_idx: %i", bus_idx)
        if bus_idx in self.parameters.bus.dsr_type:
            return (
                self.variables.bus.shift_plus[bus_idx]
                - self.variables.bus.shift_minus[bus_idx]
            )
        return 0.0

    def _bus_net_inflow(self, bus_idx: int) -> LinearExpression | float:
        """
        Calculate net inflow for bus.

        Args:
            - bus_idx: index of the bus

        Returns:
            - LinearExpression or float: linear expression of the bus net inflow
        """
        _logger.debug("Get bus net inflow for bus_idx: %i", bus_idx)
        result = 0.0
        for line_idx in self.parameters.bus.lines_in[bus_idx]:
            result += self.expr.netto_flow_l(line_idx)
        return result

    def _bus_outflow(self, bus_idx: int) -> LinearExpression | float:
        """
        Calculate net outflow for bus.

        Args:
            - bus_idx: index of the bus

        Returns:
            - LinearExpression or float: linear expression of the bus outflow
        """
        _logger.debug("Get bus outflow for bus_idx: %i", bus_idx)
        result = 0.0
        for line_idx in self.parameters.bus.lines_out[bus_idx]:
            result += self.variables.line.flow.isel(line=line_idx)
        return result

    def _bus_net_load(self, bus_idx: int) -> LinearExpression:
        """
        Calculate net load for the bus based on demand.

        Args:
            - bus_idx: index of the bus

        Returns:
            - LinearExpression: net load for the bus
        """
        _logger.debug("Get bus net load for bus_idx: %i", bus_idx)
        return self.expr.fraction_dem(bus_idx) + self._converters_demand(bus_idx)

    def _bus_net_injection(self, bus_idx: int) -> LinearExpression:
        """
        Calculate net injection for the bus of each generator and storage.

        Args:
            - bus_idx: index of the bus

        Returns:
            - LinearExpression: net injection for the bus
        """
        _logger.debug("Get bus net injection for bus_idx: %i", bus_idx)
        return self._storages_net_injection(bus_idx) + self._generators_net_injection(
            bus_idx
        )

    def _generators_net_injection(self, bus_idx: int) -> LinearExpression | float:
        """
        Calculate net injection for each generator in the bus.

        Args:
            - bus_idx: index of the bus

        Returns:
            - LinearExpression or float: generators net injection for the bus
        """
        result, bus_et = 0.0, self.parameters.bus.et[bus_idx]
        for gen_idx in self.parameters.bus.generators[bus_idx]:
            if bus_et in self.parameters.gen.ett[gen_idx]:
                result += self.variables.gen.gen_et[gen_idx][bus_et]

        return result

    def _storages_net_injection(self, bus_idx: int) -> LinearExpression:
        """
        Calculate net injection for each storage in the bus.

        Args:
            - bus_idx: index of the bus

        Returns:
            - LinearExpression: storages net injection for the bus
        """
        return self._storages_net_bus_balance(
            bus_idx
        ) - self._storages_net_demand_chunk_generation(bus_idx)

    def _storages_net_bus_balance(self, bus_idx: int) -> LinearExpression | float:
        """
        Calculate net balance for each storage in the bus.

        Args:
            - bus_idx: index of the bus

        Returns:
            - LinearExpression or float: storage net balance for the bus
        """
        result = 0.0
        load = self.variables.stor.load
        for st_idx in self.parameters.bus.storages[bus_idx]:
            result += self.expr.gen_netto_st(st_idx) - load.isel(stor=st_idx)
        return result

    def _storages_net_demand_chunk_generation(
        self, bus_idx: int
    ) -> LinearExpression | float:
        """
        Calculate net demand chunk for each storage in the bus.

        Args:
            - bus_idx: index of the bus

        Returns:
            - LinearExpression or float: storage net demand chunk for the bus
        """
        result = 0.0
        dch_gen = self.variables.stor.gen_dch
        for storage_idx in self.parameters.bus.storages[bus_idx]:
            for dch_idx in self.parameters.stor.demand_chunks.get(storage_idx, []):
                result += dch_gen[dch_idx][storage_idx]
        return result

    def _converters_demand(self, bus_idx: int) -> LinearExpression | float:
        """
        Calculate converter demand for the bus.

        Args:
            - bus_idx: index of the bus

        Returns:
            - LinearExpression or float: converter demand for the bus
        """
        bus_et, result = self.parameters.bus.et[bus_idx], 0.0
        for gen_idx in self.parameters.bus.generators[bus_idx]:
            if bus_et in self.parameters.gen.conv_rate[gen_idx]:
                result += self.variables.gen.gen.isel(gen=gen_idx) / xr.DataArray(
                    self.parameters.gen.conv_rate[gen_idx][bus_et],
                    dims=["hour"],
                    coords=dict(hour=self.indices.H.ii),
                )
        return result

    def demand_chunk_balancing_constraint(self) -> None:
        """
        Create balancing constraints for demand chunks.

        For each demand chunk, calculates total generation from generators
        and storages over specified periods, and adds constraints ensuring
        that the energy injection matches the demand for each period.
        """
        _logger.debug("Building demand chunk balancing constraints...")
        dch_params = self.parameters.demand_chunks_parameters
        for dch_idx, dem_val in dch_params.demand.items():
            generators_generation = (
                sum(self.variables.gen.gen_dch[dch_idx].values()) or np.nan
            )
            storages_generation = (
                sum(self.variables.stor.gen_dch[dch_idx].values()) or np.nan
            )
            time_period_idx = 0
            for p_start, p_end in dch_params.periods[dch_idx]:
                h_range = range(p_start, p_end + 1)
                energy_injection = 0.0
                if isinstance(generators_generation, Variable):
                    energy_injection += generators_generation.sel(hour=h_range).sum(
                        "hour"
                    )
                if isinstance(storages_generation, Variable):
                    energy_injection += storages_generation.sel(hour=h_range).sum(
                        "hour"
                    )
                demand = xr.DataArray(
                    dem_val[time_period_idx][: len(self.indices.Y.ord)], dims=["year"]
                )
                if isinstance(energy_injection, LinearExpression):
                    self.model.add_constraints(
                        energy_injection == demand,
                        name=f"DEMCH_{dch_idx}_START_{p_start}_END_{p_end}_BALANCING_CONSTRAINT",
                    )
                time_period_idx += 1
        _logger.debug("Build demand chunk balancing constraints: Done")

    def _gen_compensation_constraint(
        self, interval: range, bus_idx: int, dsr_idx: int
    ) -> None:
        """
        Adds generation compensation constraint for the bus.

        Args:
            - interval (range): start and end hour
            - bus_idx (int): index of the bus
            - dsr_idx (int): index of the dsr
        """
        _logger.debug(
            "Get gen compensation constraint for bus_idx: %i, dsr_idx: %i",
            bus_idx,
            dsr_idx,
        )
        shift_minus = self.variables.bus.shift_minus
        shift_plus = self.variables.bus.shift_plus
        compensation_factor = self.parameters.dsr.compensation_factor
        self.model.add_constraints(
            shift_plus[bus_idx].isel(hour=list(interval)).sum("hour")
            == compensation_factor[dsr_idx]
            * shift_minus[bus_idx].isel(hour=list(interval)).sum("hour"),
            name=f"DSR_{dsr_idx}_BUS_{bus_idx}_RANGE_{[interval.start, interval.stop]}_COMPENSATION_CONSTRAINT",
        )

    def _gen_relative_shift_limit(
        self, interval: range, bus_idx: int, dsr_idx: int
    ) -> None:
        """
        Adds generation relative shift limit constraint to the bus.

        Args:
            - interval (range): start and end hour
            - bus_idx (int): index of the bus
            - dsr_idx (int): index of the dsr
        """
        _logger.debug(
            "Get gen relative shift limit for bus_idx: %i, dsr_idx: %i",
            bus_idx,
            dsr_idx,
        )
        relative_shift_limit = self.parameters.dsr.relative_shift_limit
        shift_minus = self.variables.bus.shift_minus
        net_load = self._bus_net_load(bus_idx)
        if dsr_idx in relative_shift_limit:
            self.model.add_constraints(
                shift_minus[bus_idx].isel(hour=list(interval)).sum("hour")
                <= relative_shift_limit[dsr_idx]
                * net_load.isel(hour=list(interval)).sum("hour"),
                name=f"DSR_{dsr_idx}_BUS_{bus_idx}_RANGE_{[interval.start, interval.stop]}_RELATIVE_SHIFT_CONSTRAINT",
            )

    def _gen_abs_shift_limit(self, interval: range, bus_idx: int, dsr_idx: int) -> None:
        """
        Adds generation absolute shift limit constraint to the bus.

        Args:
            - interval (range): start and stop hour
            - bus_idx (int): index of the bus
            - dsr_idx (int): index of the dsr
        """
        _logger.debug(
            "Get gen absolute shift limit for bus_idx: %i, dsr_idx: %i",
            bus_idx,
            dsr_idx,
        )
        abs_shift_limit = self.parameters.dsr.abs_shift_limit
        shift_minus = self.variables.bus.shift_minus
        if dsr_idx in abs_shift_limit:
            self.model.add_constraints(
                shift_minus[bus_idx].isel(hour=list(interval)).sum("hour")
                <= abs_shift_limit[dsr_idx],
                name=f"DSR_{dsr_idx}_BUS_{bus_idx}_RANGE_{[interval.start, interval.stop]}_ABSOLUTE_SHIFT_CONSTRAINT",
            )

    def _shift_minus_relative_hourly_limit_constraint(
        self, bus_idx: int, dsr_idx: int
    ) -> None:
        """
        Hourly relative limit for decreasing demand, the default value of hourly_relative_shift_minus_limit
        parameter prevents lowering the load below the demand.

        Args:
            - bus_idx (int): index of the bus
            - dsr_idx (int): index of the dsr
        """
        self._load_shifting_relative_hourly_limit_constraint(
            bus_idx,
            dsr_idx,
            shift_var=self.variables.bus.shift_minus,
            limit_param=self.parameters.dsr.hourly_relative_shift_minus_limit,
            constr_name="LOAD_SHIFTING_RELATIVE_HOURLY_LIMIT_CONSTRAINT",
        )

    def _shift_plus_relative_hourly_limit_constraint(
        self, bus_idx: int, dsr_idx: int
    ) -> None:
        """
        Hourly relative limit for increasing demand, by default 100% increase in any hour at max.

        Args:
            - bus_idx (int): index of the bus
            - dsr_idx (int): index of the dsr
        """
        self._load_shifting_relative_hourly_limit_constraint(
            bus_idx,
            dsr_idx,
            shift_var=self.variables.bus.shift_plus,
            limit_param=self.parameters.dsr.hourly_relative_shift_plus_limit,
            constr_name="LOAD_COMPENSATION_RELATIVE_HOURLY_LIMIT_CONSTRAINT",
        )

    def _load_shifting_relative_hourly_limit_constraint(
        self,
        bus_idx: int,
        dsr_idx: int,
        shift_var: dict[int, Variable],
        limit_param: dict[int, float],
        constr_name: str,
    ) -> None:
        """
        Add constraint for shifting relative hourly limit.

        Args:
            - bus_idx (int): index of the bus
            - dsr_idx (int): index of the dsr
            - shift_var (dict[int, Variable]): shift variables
            - limit_param (dict[int, float]): limit parameter
            - constr_name (str): constraint name
        """
        if dsr_idx in limit_param:
            _logger.debug(
                f"Adding {constr_name} for bus_idx: {bus_idx}, dsr_idx: {dsr_idx}"
            )
            self.model.add_constraints(
                shift_var[bus_idx]
                <= self._bus_net_load(bus_idx) * limit_param[dsr_idx],
                name=f"{constr_name}_BUS_{bus_idx}_DSR{dsr_idx}",
            )
