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

from gurobipy import quicksum

from pyzefir.optimization.gurobi.constraints_builder.builder import (
    PartialConstraintsBuilder,
)
from pyzefir.utils.functions import _demch_unit


class GenerationConstraintsBuilder(PartialConstraintsBuilder):
    def build_constraints(self) -> None:
        self.generation_vs_capacity_constraints()
        self.generation_and_dump_energy()
        self.total_dump_energy()

    def generation_vs_capacity_constraints(self) -> None:
        for gen_idx, gen_name in self.indices.GEN.mapping.items():
            generation_brutto = self.variables.gen.gen[gen_idx, :, :]
            capacity = self.variables.gen.cap[gen_idx, :]
            capacity_factor_id = self.parameters.gen.capacity_factors[gen_idx]
            gen_to_tgen = self.parameters.gen.tgen
            if capacity_factor_id is not None:
                capacity_factor = self.parameters.cf.profile[capacity_factor_id]
                self.model.addConstr(
                    generation_brutto
                    == capacity_factor.reshape(-1, 1)
                    * capacity.reshape(1, -1)
                    * self.parameters.tgen.power_utilization[
                        gen_to_tgen[gen_idx]
                    ].reshape(-1, 1),
                    name=f"{gen_name}_NON_DISPATCHABLE_GEN_CAP_CONSTRAINT",
                )
            else:
                self.model.addConstr(
                    generation_brutto
                    <= capacity.reshape(1, -1)
                    * self.parameters.tgen.power_utilization[
                        gen_to_tgen[gen_idx]
                    ].reshape(-1, 1),
                    name=f"{gen_name}_DISPATCHABLE_GEN_CAP_CONSTRAINT",
                )

    def generation_and_dump_energy(self) -> None:
        for gen_idx, gen_name in self.indices.GEN.mapping.items():
            for energy_type_idx, energy_type_name in self.indices.ET.mapping.items():
                energy_type_generation = self.variables.gen.gen_et[
                    gen_idx, energy_type_idx, :, :
                ]
                energy_type_dump_energy = self.variables.gen.dump_et[
                    gen_idx, energy_type_idx, :, :
                ]
                gen_dch = self.variables.gen.gen_dch
                if energy_type_name in self.parameters.gen.ett[gen_idx]:
                    generation_brutto = self.variables.gen.gen[gen_idx, :, :]
                    efficiency = self.parameters.gen.eff[gen_idx][energy_type_name]
                    demch_params = self.parameters.demand_chunks_parameters
                    DCH = quicksum(
                        gen_dch[energy_type_idx, dem_idx, gen_idx, :, :]
                        for dem_idx, tag in demch_params.tag.items()
                        if gen_idx
                        in _demch_unit(
                            dem_idx, demch_params.tag, self.parameters.gen.tags
                        )
                        and demch_params.energy_type[dem_idx] == energy_type_name
                    )

                    self.model.addConstr(
                        generation_brutto * efficiency
                        == energy_type_generation + energy_type_dump_energy + DCH,
                        name=f"{gen_name}_{energy_type_idx}_GENERATION_DUMP_CONSTRAINT",
                    )
                else:
                    self.model.addConstr(
                        energy_type_dump_energy + energy_type_generation == 0,
                        name=f"{gen_name}_{energy_type_idx}_GENERATION_DUMP_CONSTRAINT",
                    )

    def total_dump_energy(self) -> None:
        for gen_idx, gen_name in self.indices.GEN.mapping.items():
            dump_energy, de_et, eff = (
                self.variables.gen.dump,
                self.variables.gen.dump_et,
                self.parameters.gen.eff,
            )
            de_et_sum = quicksum(
                de_et[gen_idx, self.indices.ET.inverse[et_name], :, :]
                / eff[gen_idx][et_name]
                for et_name in self.parameters.gen.ett[gen_idx]
            )
            self.model.addConstr(
                de_et_sum == dump_energy[gen_idx, :, :],
                name=f"{gen_name}_TOTAL_DUMP_ENERGY",
            )
