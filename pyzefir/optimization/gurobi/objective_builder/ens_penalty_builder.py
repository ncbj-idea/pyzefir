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

import numpy as np
import pandas as pd
from gurobipy import MLinExpr, quicksum

from pyzefir.optimization.gurobi.objective_builder import ObjectiveBuilder
from pyzefir.optimization.gurobi.preprocessing.opt_parameters import (
    OptimizationParameters,
)


class EnsPenaltyCostObjectiveBuilder(ObjectiveBuilder):
    def build_expression(self) -> MLinExpr | float:
        penalty_cost = self._get_fuel_penalty_cost(self.parameters)
        if np.isinf(penalty_cost):
            penalty_cost = 1e16
        return quicksum(
            self.variables.bus.bus_ens[bus_idx, h, y] * penalty_cost
            for bus_idx in self.indices.BUS.ord
            for h in self.indices.H.ord
            for y in self.indices.Y.ord
        )

    @staticmethod
    def _nan_free(elem: float) -> float:
        return 0.0 if np.isnan(elem) else elem

    def _get_fuel_penalty_cost(self, parameters: OptimizationParameters) -> float:
        """estimate penalty cost based on fuels. If impossible or the result is not big, use a big number"""
        ens_penalty = 1e2
        res_fuel = ens_penalty
        fuel = parameters.fuel
        res_em_fee = self._get_res_emission_fee()
        effs = min(
            value
            for el in parameters.gen.eff.values()
            for value in el.values()
            if value > 0
        )
        if len(fuel.availability.keys()):
            res_fuel_arr = np.asarray(
                [
                    fuel.unit_cost[ke] / fuel.energy_per_unit[ke] / effs
                    for ke in fuel.availability.keys()
                ]
            )
            res_fuel = (
                max(res_fuel_arr.max(axis=0) * res_fuel_arr.shape[0]) * ens_penalty
            ) + res_em_fee
        res_fuel = (
            self._nan_free(res_fuel) * self.parameters.scenario_parameters.hourly_scale
        )
        res_capex = self._nan_free(
            max([el.max() for el in [*parameters.tgen.capex.values()]])
        )
        res_opex = self._nan_free(
            max([el.max() for el in [*parameters.tgen.opex.values()]])
        )
        res_transmission_fee = (
            self._nan_free(
                max([fee_array.max() for fee_array in self.parameters.tf.fee.values()])
            )
            * self.parameters.scenario_parameters.hourly_scale
            if len(self.parameters.tf.fee.values()) != 0
            else 0.0
        )
        return max(
            max(
                res_fuel * ens_penalty,
                res_capex,
                res_opex * ens_penalty,
                res_transmission_fee * ens_penalty,
            ),
            ens_penalty,
        )

    def _get_max_em_fee(self) -> dict[str, float]:
        df_emission_type = pd.DataFrame.from_dict(
            self.parameters.emf.emission_type, orient="index", columns=["emission_type"]
        )
        df_price = pd.DataFrame.from_dict(self.parameters.emf.price, orient="index")
        concated_df = pd.concat([df_emission_type, df_price], axis=1)
        return {
            emission: concated_df[concated_df["emission_type"] == emission]
            .max()
            .iloc[1:]
            .sum()
            for emission in concated_df["emission_type"].unique()
        }

    def _get_res_emission_fee(self) -> float:
        res_em_fee = {fuel_idx: 0.0 for fuel_idx in self.parameters.fuel.availability}
        max_em_fee = self._get_max_em_fee()
        for emission_type, val in max_em_fee.items():
            for fuel_idx in res_em_fee:
                res_em_fee[fuel_idx] += (
                    val * self.parameters.fuel.u_emission[fuel_idx][emission_type]
                )
        return max(res_em_fee.values())
