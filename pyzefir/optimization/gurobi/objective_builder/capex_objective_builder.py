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
from gurobipy import LinExpr, quicksum, tupledict

from pyzefir.optimization.gurobi.objective_builder import ObjectiveBuilder
from pyzefir.optimization.gurobi.preprocessing.indices import IndexingSet
from pyzefir.optimization.gurobi.preprocessing.parameters.generator_type_parameters import (
    GeneratorTypeParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.storage_type_parameters import (
    StorageTypeParameters,
)
from pyzefir.utils.functions import get_dict_vals, invert_dict_of_sets


class CapexObjectiveBuilder(ObjectiveBuilder):
    def build_expression(self) -> LinExpr:
        return self.global_capex() + self.local_capex()

    def local_capex(self) -> LinExpr:
        generator_capex = self._local_capex(
            tcap_plus=self.variables.tgen.tcap_plus,
            unit_type_param=self.parameters.tgen,
            aggr_map=self.indices.aggr_tgen_map,
        )
        storage_capex = self._local_capex(
            tcap_plus=self.variables.tstor.tcap_plus,
            unit_type_param=self.parameters.tstor,
            aggr_map=self.indices.aggr_tstor_map,
        )
        return generator_capex + storage_capex

    def global_capex(self) -> LinExpr:
        generator_capex = self._global_capex(
            cap_plus=self.variables.gen.cap_plus,
            unit_type_param=self.parameters.tgen,
            unit_type_idx=self.parameters.gen.tgen,
            non_lbs_unit_idxs=get_dict_vals(self.parameters.bus.generators).difference(
                get_dict_vals(self.indices.aggr_gen_map)
            ),
        )
        storage_capex = self._global_capex(
            cap_plus=self.variables.stor.cap_plus,
            unit_type_param=self.parameters.tstor,
            unit_type_idx=self.parameters.stor.tstor,
            non_lbs_unit_idxs=get_dict_vals(self.parameters.bus.storages).difference(
                get_dict_vals(self.indices.aggr_stor_map)
            ),
        )
        return generator_capex + storage_capex

    def _global_capex(
        self,
        cap_plus: tupledict,
        unit_type_param: GeneratorTypeParameters | StorageTypeParameters,
        unit_type_idx: dict,
        non_lbs_unit_idxs: set,
    ) -> LinExpr:
        disc_rate = self.expr.discount_rate(
            self.parameters.scenario_parameters.discount_rate
        )
        y_idxs = self.indices.Y
        unit_capex = LinExpr(0.0)
        for u_idx in non_lbs_unit_idxs:
            ut_idx = unit_type_idx[u_idx]
            capex = unit_type_param.capex[ut_idx]
            lt = unit_type_param.lt[ut_idx]
            bt = unit_type_param.bt[ut_idx]

            unit_capex += quicksum(
                self.global_capex_per_unit_per_year(
                    capex=capex,
                    cap_plus=cap_plus,
                    disc_rate=disc_rate,
                    lt=lt,
                    bt=bt,
                    y_idx=y_idx,
                    u_idx=u_idx,
                    y_idxs=y_idxs,
                )
                for y_idx in y_idxs.ord
            )
        return unit_capex

    @staticmethod
    def global_capex_per_unit_per_year(
        capex: np.ndarray,
        cap_plus: tupledict,
        disc_rate: np.ndarray,
        bt: int,
        lt: int,
        y_idx: int,
        u_idx: int,
        y_idxs: IndexingSet,
    ) -> LinExpr:
        am_indicator = CapexObjectiveBuilder._amortization_matrix_indicator(
            lt=lt, bt=bt, yy=y_idxs
        )
        return quicksum(
            am_indicator[s, y_idx]
            * capex[s]
            * cap_plus[u_idx, s]
            * disc_rate[y_idx]
            / lt
            for s in y_idxs.ord
        )

    def _local_capex(
        self,
        tcap_plus: tupledict,
        unit_type_param: GeneratorTypeParameters | StorageTypeParameters,
        aggr_map: dict[..., set],
    ) -> LinExpr:
        disc_rate = self.expr.discount_rate(
            self.parameters.scenario_parameters.discount_rate
        )
        y_idxs = self.indices.Y
        unit_type_capex = LinExpr(0.0)
        inverted_aggr_map = invert_dict_of_sets(aggr_map)
        for ut_idx, aggr_idxs in inverted_aggr_map.items():
            capex = unit_type_param.capex[ut_idx]
            lt = unit_type_param.lt[ut_idx]
            bt = unit_type_param.bt[ut_idx]

            unit_type_capex += quicksum(
                self.local_capex_per_unit_per_year(
                    capex=capex,
                    tcap_plus=tcap_plus,
                    disc_rate=disc_rate,
                    bt=bt,
                    lt=lt,
                    y_idx=y_idx,
                    ut_idx=ut_idx,
                    aggr_idxs=aggr_idxs,
                    y_idxs=y_idxs,
                )
                for y_idx in y_idxs.ord
            )

        return unit_type_capex

    @staticmethod
    def local_capex_per_unit_per_year(
        capex: np.ndarray,
        tcap_plus: tupledict,
        disc_rate: np.ndarray,
        bt: int,
        lt: int,
        y_idx: int,
        ut_idx: int,
        aggr_idxs: set[int],
        y_idxs: IndexingSet,
    ) -> LinExpr:
        am_indicator = CapexObjectiveBuilder._amortization_matrix_indicator(
            lt=lt, bt=bt, yy=y_idxs
        )
        return quicksum(
            am_indicator[s, y_idx]
            * capex[s]
            * tcap_plus[aggr_idx, ut_idx, s]
            * disc_rate[y_idx]
            / lt
            for s in y_idxs.ord
            for aggr_idx in aggr_idxs
        )

    @staticmethod
    def _amortization_matrix_indicator(
        lt: int,
        bt: int,
        yy: IndexingSet,
    ) -> np.ndarray:
        """
        Indicator matrix for y-index range in capex expression.

        :param lt: unit lifetime
        :param bt: unit build time
        :param yy: year indices
        :return: np.ndarray
        """

        return np.array(
            [
                ((yy.ord >= y + bt) & (yy.ord <= min(y + bt + lt - 1, len(yy)))).astype(
                    int
                )
                for y in yy.ord
            ]
        )
