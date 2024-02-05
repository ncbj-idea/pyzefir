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

from pyzefir.optimization.gurobi.constraints_builder.builder import (
    PartialConstraintsBuilder,
)


class StorageConstraintsBuilder(PartialConstraintsBuilder):
    def build_constraints(self) -> None:
        self.state_of_charge_upper_bound()
        self.generation_upper_bound()
        self.balance_upper_bound()
        self.boundary_state_of_charge_values()
        self.state_of_charge_definition()
        self.loading_cycles()

    def state_of_charge_upper_bound(self) -> None:
        for st_idx, st_name in self.indices.STOR.mapping.items():
            state_of_charge = self.variables.stor.soc[st_idx, :, :]
            capacity = self.variables.stor.cap[st_idx, :]
            power_utilization = self.parameters.tstor.power_utilization[
                self.parameters.stor.tstor[st_idx]
            ]
            self.model.addConstr(
                state_of_charge <= capacity * power_utilization,
                name=f"{st_name}_STATE_OF_CHARGE_UPPER_BOUND_CONSTRAINT",
            )

    def generation_upper_bound(self) -> None:
        for st_idx, st_name in self.indices.STOR.mapping.items():
            generation = self.variables.stor.gen[st_idx, :, :]
            state_of_charge = self.variables.stor.soc[st_idx, :, :]
            self.model.addConstr(
                generation <= state_of_charge,
                name=f"{st_name}_GENERATION_UPPER_BOUND_CONSTRAINT",
            )

    def balance_upper_bound(self) -> None:
        for st_idx, st_name in self.indices.STOR.mapping.items():
            generation = self.variables.stor.gen[st_idx, :, :]
            load = self.variables.stor.load[st_idx, :, :]
            nom_p = self.expr.p_inst_st(st_idx)
            self.model.addConstr(
                generation + load <= nom_p,
                name=f"{st_name}_BALANCE_UPPER_BOUND_CONSTRAINT",
            )

    def boundary_state_of_charge_values(self) -> None:
        for st_idx, st_name in self.indices.STOR.mapping.items():
            soc = self.variables.stor.soc[st_idx, :, :]
            self.model.addConstr(
                soc[0, 0] == 0, name=f"{st_name}_INITIAL_STATE_OF_CHARGE_CONSTRAINT"
            )
            self.model.addConstr(
                soc[-1, -1] == 0, name=f"{st_name}_END_STATE_OF_CHARGE_CONSTRAINT"
            )

    def loading_cycles(self) -> None:
        for st_idx, st_name in self.indices.STOR.mapping.items():
            cycle_len = self.parameters.stor.cycle_len[st_idx]
            end_cycle_soc = self.variables.stor.soc[st_idx, :, :].T.reshape(-1)[
                ::cycle_len
            ]
            self.model.addConstr(
                end_cycle_soc == 0, name=f"{st_name}_LOADING_CYCLES_CONSTRAINT"
            )

    def state_of_charge_definition(self) -> None:
        # FIXME: below for-loop approach is 3-4 times slower than matrix implementation
        for st_idx, st_name in self.indices.STOR.mapping.items():
            soc = self.variables.stor.soc[st_idx, :, :]
            gen = self.variables.stor.gen[st_idx, :, :]
            load_netto_expr = self.expr.load_netto_st(st_idx)
            type_dict = self.parameters.stor.tstor
            e_loss = self.parameters.tstor.energy_loss[type_dict[st_idx]]
            for h in self.indices.H.ord:
                for y in self.indices.Y.ord:
                    if h != 0:
                        self.model.addConstr(
                            soc[h, y]
                            == (1 - e_loss) * soc[h - 1, y]
                            - gen[h - 1, y]
                            + load_netto_expr[h - 1, y],
                        )
                    if y > 0 and h == 0:
                        self.model.addConstr(
                            soc[h, y]
                            == (1 - e_loss) * soc[-1, y - 1]
                            - gen[-1, y - 1]
                            + load_netto_expr[-1, y - 1]
                        )
