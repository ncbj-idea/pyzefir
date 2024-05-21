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

import xarray as xr
from linopy import LinearExpression

from pyzefir.optimization.linopy.constraints_builder.builder import (
    PartialConstraintsBuilder,
)
from pyzefir.utils.functions import demand_chunk_unit_indices

_logger = logging.getLogger(__name__)


class BalancingConstraintsBuilder(PartialConstraintsBuilder):
    def build_constraints(self) -> None:
        _logger.info("Balancing constraints builder is working...")
        self.balancing_constraint()
        self.demand_chunk_balancing_constraint()
        self.load_shifting_constraints()
        _logger.info("Balancing constraints builder is finished!")

    def balancing_constraint(self) -> None:
        _logger.debug("Building balancing constraints...")
        for bus_idx, bus_name in self.indices.BUS.mapping.items():
            net_load = self._bus_net_load(bus_idx)
            net_outflow = self._bus_outflow(bus_idx)
            net_injection = self._bus_net_injection(bus_idx)
            net_inflow = self._bus_net_inflow(bus_idx)
            ens = self.variables.bus.bus_ens.isel(bus=bus_idx)
            shift = self._shift(bus_idx)

            self.model.add_constraints(
                shift + net_load + net_outflow == net_inflow + net_injection + ens,
                name=f"{bus_name}_BALANCING_CONSTRAINT",
            )
        _logger.debug("Build balancing constraints: Done")

    def load_shifting_constraints(self) -> None:
        _logger.debug("Loading shifting constraints...")
        if len(self.parameters.bus.dsr_type):
            balancing_periods = self.parameters.dsr.balancing_periods
            for bus_idx, dsr_idx in self.parameters.bus.dsr_type.items():
                if bus_idx in self.parameters.bus.lbs_mapping:
                    intervals = balancing_periods[dsr_idx]
                    for interval in intervals:
                        self._gen_compensation_constraint(interval, bus_idx, dsr_idx)
                        self._gen_relative_shift_limit(interval, bus_idx, dsr_idx)
                        self._gen_abs_shift_limit(interval, bus_idx, dsr_idx)
        _logger.debug("Load shifting constraints: Done")

    def _shift(self, bus_idx: int) -> LinearExpression | float:
        _logger.debug("Shifting values for bus_idx: %i", bus_idx)
        if bus_idx in self.parameters.bus.dsr_type:
            return self.variables.bus.shift_plus.isel(
                bus=bus_idx
            ) - self.variables.bus.shift_minus.isel(bus=bus_idx)
        return 0.0

    def _bus_net_inflow(self, bus_idx: int) -> LinearExpression | float:
        _logger.debug("Get bus net inflow for bus_idx: %i", bus_idx)
        result = 0.0
        for line_idx in self.parameters.bus.lines_in[bus_idx]:
            result += self.expr.netto_flow_l(line_idx)
        return result

    def _bus_outflow(self, bus_idx: int) -> LinearExpression | float:
        _logger.debug("Get bus outflow for bus_idx: %i", bus_idx)
        result = 0.0
        for line_idx in self.parameters.bus.lines_out[bus_idx]:
            result += self.variables.line.flow.isel(line=line_idx)
        return result

    def _bus_net_load(self, bus_idx: int) -> LinearExpression:
        _logger.debug("Get bus net load for bus_idx: %i", bus_idx)
        return self.expr.fraction_dem(bus_idx) + self._converters_demand(bus_idx)

    def _bus_net_injection(self, bus_idx: int) -> LinearExpression:
        _logger.debug("Get bus net injection for bus_idx: %i", bus_idx)
        return self._storages_net_injection(bus_idx) + self._generators_net_injection(
            bus_idx
        )

    def _storages_net_injection(self, bus_idx: int) -> LinearExpression:
        storages_net_bus_balance_expr = self._storages_net_bus_balance(bus_idx)
        storages_net_demand_chunk_generation_expr = (
            self._storages_net_demand_chunk_generation(bus_idx)
        )
        storage_net_bus_injection = (
            storages_net_bus_balance_expr - storages_net_demand_chunk_generation_expr
        )
        return storage_net_bus_injection

    def _generators_net_injection(self, bus_idx: int) -> LinearExpression | float:
        result, bus_et = 0.0, self.parameters.bus.et[bus_idx]
        for gen_idx in self.parameters.bus.generators[bus_idx]:
            if bus_et in self.parameters.gen.ett[gen_idx]:
                result += self.variables.gen.gen_et.isel(
                    gen=gen_idx, et=self.indices.ET.inverse[bus_et]
                )

        return result

    def _converters_demand(self, bus_idx: int) -> LinearExpression | float:
        bus_et, result = self.parameters.bus.et[bus_idx], 0.0
        for gen_idx in self.parameters.bus.generators[bus_idx]:
            if bus_et in self.parameters.gen.conv_rate[gen_idx]:
                result += self.variables.gen.gen.isel(gen=gen_idx) / xr.DataArray(
                    self.parameters.gen.conv_rate[gen_idx][bus_et],
                    dims=["hour"],
                    coords=dict(hour=self.indices.H.ii),
                )
        return result

    def _storages_net_demand_chunk_generation(
        self, bus_idx: int
    ) -> LinearExpression | float:
        result = 0.0
        for dem_idx, tag_idx in self.parameters.demand_chunks_parameters.tag.items():
            demand_chunk_units = demand_chunk_unit_indices(
                demand_chunk_idx=dem_idx,
                demand_chunk_tags=self.parameters.demand_chunks_parameters.tag,
                unit_tags=self.parameters.stor.tags,
            )
            connected_storages = self.parameters.bus.storages[bus_idx]
            for st_idx in set(connected_storages) & set(demand_chunk_units):
                result += self.variables.stor.gen_dch.isel(demch=dem_idx, stor=st_idx)

        return result

    def _storages_net_bus_balance(self, bus_idx: int) -> LinearExpression | float:
        result = 0.0
        for st_idx in self.parameters.bus.storages[bus_idx]:
            result += self.expr.gen_netto_st(st_idx) - self.variables.stor.load.isel(
                stor=st_idx
            )
        return result

    def demand_chunk_balancing_constraint(self) -> None:
        _logger.debug("Building demand chunk balancing constraints...")
        dch_params = self.parameters.demand_chunks_parameters
        generators_energy_type = self.parameters.gen.ett
        gen_dch_var = self.variables.gen.gen_dch
        stor_dch_var = self.variables.stor.gen_dch
        for dem_idx, dem_val in dch_params.demand.items():
            dch_et = dch_params.energy_type[dem_idx]
            dch_et_idx = self.indices.ET.inverse[dch_et]
            gen_idxs = {
                idx
                for idx in demand_chunk_unit_indices(
                    dem_idx, dch_params.tag, self.parameters.gen.tags
                )
                if dch_et in generators_energy_type[idx]
            }
            stor_idxs = demand_chunk_unit_indices(
                dem_idx, dch_params.tag, self.parameters.stor.tags
            )
            time_period_idx = 0
            for p_start, p_end in dch_params.periods[dem_idx]:
                h_range = range(p_start, p_end + 1)
                from_gen = (
                    gen_dch_var.isel(
                        et=dch_et_idx,
                        demch=dem_idx,
                        gen=list(gen_idxs),
                    )
                    .sel(hour=h_range)
                    .sum(["gen", "hour"])
                )

                from_stor = (
                    stor_dch_var.isel(
                        demch=dem_idx,
                        stor=list(stor_idxs),
                    )
                    .sel(hour=h_range)
                    .sum(["stor", "hour"])
                )

                demand = dem_val[time_period_idx][: len(self.indices.Y.ord)]
                self.model.add_constraints(
                    from_gen + from_stor == xr.DataArray(demand, dims=["year"]),
                    name=f"DEMCH_{dem_idx}_START_{p_start}_END_{p_end}_BALANCING_CONSTRAINT",
                )
                time_period_idx += 1
        _logger.debug("Build demand chunk balancing constraints: Done")

    def _gen_compensation_constraint(
        self, interval: range, bus_idx: int, dsr_idx: int
    ) -> None:
        _logger.debug(
            "Get gen compensation constraint for bus_idx: %i, dsr_idx: %i",
            bus_idx,
            dsr_idx,
        )
        shift_minus = self.variables.bus.shift_minus
        shift_plus = self.variables.bus.shift_plus
        compensation_factor = self.parameters.dsr.compensation_factor
        self.model.add_constraints(
            shift_plus.isel(bus=bus_idx, hour=list(interval))
            == compensation_factor[dsr_idx]
            * shift_minus.isel(bus=bus_idx, hour=list(interval)),
            name=f"DSR_{dsr_idx}_BUS_{bus_idx}_RANGE_{[interval.start, interval.stop]}_COMPENSATION_CONSTRAINT",
        )

    def _gen_relative_shift_limit(
        self, interval: range, bus_idx: int, dsr_idx: int
    ) -> None:
        _logger.debug(
            "Get gen relative shift limit for bus_idx: %i, dsr_idx: %i",
            bus_idx,
            dsr_idx,
        )
        relative_shift_limit = self.parameters.dsr.relative_shift_limit
        shift_minus = self.variables.bus.shift_minus
        net_load = self._bus_net_load(bus_idx)
        if dsr_idx in relative_shift_limit.keys():
            self.model.add_constraints(
                shift_minus.isel(bus=bus_idx, hour=list(interval)).sum(["hour"])
                <= relative_shift_limit[dsr_idx]
                * net_load.isel(hour=list(interval)).sum(["hour"]),
                name=f"DSR_{dsr_idx}_BUS_{bus_idx}_RANGE_{[interval.start, interval.stop]}_RELATIVE_SHIFT_CONSTRAINT",
            )

    def _gen_abs_shift_limit(self, interval: range, bus_idx: int, dsr_idx: int) -> None:
        _logger.debug(
            "Get gen absolute shift limit for bus_idx: %i, dsr_idx: %i",
            bus_idx,
            dsr_idx,
        )
        abs_shift_limit = self.parameters.dsr.abs_shift_limit
        shift_minus = self.variables.bus.shift_minus
        if dsr_idx in abs_shift_limit.keys():
            self.model.add_constraints(
                shift_minus.isel(bus=bus_idx, hour=list(interval)).sum(["hour"])
                <= abs_shift_limit[dsr_idx],
                name=f"DSR_{dsr_idx}_BUS_{bus_idx}_RANGE_{[interval.start, interval.stop]}_ABSOLUTE_SHIFT_CONSTRAINT",
            )
