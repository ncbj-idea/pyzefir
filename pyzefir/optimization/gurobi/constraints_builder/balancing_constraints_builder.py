# PyZefir
# Copyright (C) 2023-2024 Narodowe Centrum Badań Jądrowych
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

from gurobipy import MLinExpr, quicksum

from pyzefir.optimization.gurobi.constraints_builder.builder import (
    PartialConstraintsBuilder,
)
from pyzefir.utils.functions import _demch_unit


class BalancingConstraintsBuilder(PartialConstraintsBuilder):
    def build_constraints(self) -> None:
        self.balancing_constraint()
        self.demand_chunk_balancing_constraint()

    def balancing_constraint(self) -> None:
        for bus_idx, bus_name in self.indices.BUS.mapping.items():
            net_load = self._bus_net_load(bus_idx)
            net_outflow = self._bus_outflow(bus_idx)
            net_injection = self._bus_net_injection(bus_idx)
            net_inflow = self._bus_net_inflow(bus_idx)
            ens = self.variables.bus.bus_ens[bus_idx]
            self.model.addConstr(
                net_load + net_outflow == net_inflow + net_injection + ens,
                name=f"{bus_name}_BALANCING_CONSTRAINT",
            )

    def _bus_net_inflow(self, bus_idx: int) -> MLinExpr:
        return quicksum(
            self.expr.netto_flow_l(line_idx)
            for line_idx in self.parameters.bus.lines_in[bus_idx]
        )

    def _bus_outflow(self, bus_idx: int) -> MLinExpr:
        return quicksum(
            self.variables.line.flow[line_idx, :, :]
            for line_idx in self.parameters.bus.lines_out[bus_idx]
        )

    def _bus_net_load(self, bus_idx: int) -> MLinExpr:
        return self.expr.fraction_dem(bus_idx) + self._conversion_rate(bus_idx)

    def _bus_net_injection(self, bus_idx: int) -> MLinExpr:
        return self._storages_net_injection(bus_idx) + self._generators_net_injection(
            bus_idx
        )

    def _storages_net_injection(self, bus_idx: int) -> MLinExpr:
        demch_params = self.parameters.demand_chunks_parameters
        gen_dch = self.variables.stor.gen_dch
        dmch_tag = self.parameters.demand_chunks_parameters.tag
        demand_chunks_generation = quicksum(
            gen_dch[dem_idx, st_idx, :, :]
            for dem_idx, tag in demch_params.tag.items()
            for st_idx in self.parameters.bus.storages[bus_idx]
            if st_idx in _demch_unit(dem_idx, dmch_tag, self.parameters.stor.tags)
        )
        storage_net_balance = quicksum(
            self.expr.gen_netto_st(st_idx) - self.variables.stor.load[st_idx, :, :]
            for st_idx in self.parameters.bus.storages[bus_idx]
        )
        storage_net_bus_injection = storage_net_balance - demand_chunks_generation
        return storage_net_bus_injection

    def _generators_net_injection(self, bus_idx: int) -> MLinExpr | float:
        result, bus_et = 0.0, self.parameters.bus.et[bus_idx]
        for gen_idx in self.parameters.bus.generators[bus_idx]:
            if bus_et in self.parameters.gen.ett[gen_idx]:
                result += self.variables.gen.gen_et[
                    gen_idx, self.indices.ET.inverse[bus_et], :, :
                ]

        return result

    def _conversion_rate(self, bus_idx: int) -> MLinExpr | float:
        bus_et, result = self.parameters.bus.et[bus_idx], 0.0
        for gen_idx in self.parameters.bus.generators[bus_idx]:
            if bus_et in self.parameters.gen.conv_rate[gen_idx]:
                result += self.variables.gen.gen[
                    gen_idx, :, :
                ] / self.parameters.gen.conv_rate[gen_idx][bus_et].reshape(-1, 1)
        return result

    def demand_chunk_balancing_constraint(self) -> None:
        dch_params = self.parameters.demand_chunks_parameters
        generators_energy_type = self.parameters.gen.ett
        gen_dch_var = self.variables.gen.gen_dch
        stor_dch_var = self.variables.stor.gen_dch
        for dem_idx, dem_val in dch_params.demand.items():
            dch_et = dch_params.energy_type[dem_idx]
            dch_et_idx = self.indices.ET.inverse[dch_et]
            gen_idxs = {
                idx
                for idx in _demch_unit(
                    dem_idx, dch_params.tag, self.parameters.gen.tags
                )
                if dch_et in generators_energy_type[idx]
            }
            stor_idxs = _demch_unit(dem_idx, dch_params.tag, self.parameters.stor.tags)
            time_period_idx = 0
            for p_start, p_end in dch_params.periods[dem_idx]:
                h_range = range(p_start, p_end + 1)
                from_gen = quicksum(
                    gen_dch_var[dch_et_idx, dem_idx, gen_idx, h, :]
                    for gen_idx in gen_idxs
                    for h in h_range
                )
                from_stor = quicksum(
                    stor_dch_var[dem_idx, stor_idx, h, :]
                    for stor_idx in stor_idxs
                    for h in h_range
                )
                demand = dem_val[time_period_idx][: len(self.indices.Y.ord)]
                self.model.addConstr(
                    from_gen + from_stor == demand,
                    name=f"DEMCH_{dem_idx}_START_{p_start}_END_{p_end}_BALANCING_CONSTRAINT",
                )
                time_period_idx += 1
