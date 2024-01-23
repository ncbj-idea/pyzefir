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
from pyzefir.optimization.gurobi.preprocessing.indices import IndexingSet
from pyzefir.optimization.gurobi.preprocessing.parameters.generator_parameters import (
    GeneratorParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.generator_type_parameters import (
    GeneratorTypeParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.storage_parameters import (
    StorageParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.storage_type_parameters import (
    StorageTypeParameters,
)
from pyzefir.optimization.gurobi.preprocessing.variables.generator_type_variables import (
    GeneratorTypeVariables,
)
from pyzefir.optimization.gurobi.preprocessing.variables.generator_variables import (
    GeneratorVariables,
)
from pyzefir.optimization.gurobi.preprocessing.variables.storage_type_variables import (
    StorageTypeVariables,
)
from pyzefir.optimization.gurobi.preprocessing.variables.storage_variables import (
    StorageVariables,
)
from pyzefir.utils.functions import get_dict_vals


class CapacityEvolutionConstrBuilder(PartialConstraintsBuilder):
    def build_constraints(self) -> None:
        self.capacity_evolution_constraints()
        self.supplementary_evolution_constraints()
        self.base_capacity_constraints()
        self.generator_n_min_max_power_constraints()

    def generator_n_min_max_power_constraints(self) -> None:
        self._build_n_min_max_power_constraints(
            self.indices.GEN, self.parameters.gen, self.variables.gen
        )
        self._build_n_min_max_power_constraints(
            self.indices.STOR, self.parameters.stor, self.variables.stor
        )

    def capacity_evolution_constraints(self) -> None:
        self._build_capacity_evolution_constraints_gen_stor()
        self._build_local_capacity_evolution_constraints_gen_stor()

    def _build_capacity_evolution_constraints_gen_stor(self) -> None:
        self._build_capacity_evolution_constraints(
            unit_ii=self.indices.GEN,
            unit_par=self.parameters.gen,
            unit_tpar=self.parameters.tgen,
            unit_tidx=self.parameters.gen.tgen,
            unit_var=self.variables.gen,
            unit_aggr_map=self.indices.aggr_gen_map,
        )
        self._build_capacity_evolution_constraints(
            unit_ii=self.indices.STOR,
            unit_par=self.parameters.stor,
            unit_tpar=self.parameters.tstor,
            unit_tidx=self.parameters.stor.tstor,
            unit_var=self.variables.stor,
            unit_aggr_map=self.indices.aggr_stor_map,
        )

    def _build_local_capacity_evolution_constraints_gen_stor(self) -> None:
        self._build_local_capacity_evolution_constraints(
            unit_par=self.parameters.gen,
            unit_tpar=self.parameters.tgen,
            unit_tidx=self.parameters.gen.tgen,
            unit_tvar=self.variables.tgen,
            unit_aggr_map=self.indices.aggr_gen_map,
            unit_aggr_tmap=self.indices.aggr_tgen_map,
        )
        self._build_local_capacity_evolution_constraints(
            unit_par=self.parameters.stor,
            unit_tpar=self.parameters.tstor,
            unit_tidx=self.parameters.stor.tstor,
            unit_tvar=self.variables.tstor,
            unit_aggr_map=self.indices.aggr_stor_map,
            unit_aggr_tmap=self.indices.aggr_tstor_map,
        )

    def supplementary_evolution_constraints(self) -> None:
        self._build_reduced_capacity_upper_bound_constraints_gen_stor()
        self._build_local_supplementary_capacity_upper_bound_constraints_gen_stor()

    def _build_reduced_capacity_upper_bound_constraints_gen_stor(self) -> None:
        self._build_reduced_capacity_upper_bound_constraints(
            unit_ii=self.indices.GEN,
            unit_tpar=self.parameters.tgen,
            unit_tidx=self.parameters.gen.tgen,
            unit_var=self.variables.gen,
            unit_aggr_map=self.indices.aggr_gen_map,
        )
        self._build_reduced_capacity_upper_bound_constraints(
            unit_ii=self.indices.STOR,
            unit_tpar=self.parameters.tstor,
            unit_tidx=self.parameters.stor.tstor,
            unit_var=self.variables.stor,
            unit_aggr_map=self.indices.aggr_stor_map,
        )

    def _build_local_supplementary_capacity_upper_bound_constraints_gen_stor(
        self,
    ) -> None:
        """Supplementary constraints specifying the cap <-> tcap relation
        and equivalent of reduced_capacity_upper_bound_constraints for local technologies
        The constraints separately for generators and storages
        """
        self._build_local_supplementary_capacity_upper_bound_constraints(
            unit_tpar=self.parameters.tgen,
            unit_tidx=self.parameters.gen.tgen,
            unit_var=self.variables.gen,
            unit_tvar=self.variables.tgen,
            unit_aggr_tmap=self.indices.aggr_tgen_map,
        )
        self._build_local_supplementary_capacity_upper_bound_constraints(
            unit_tpar=self.parameters.tstor,
            unit_tidx=self.parameters.stor.tstor,
            unit_var=self.variables.stor,
            unit_tvar=self.variables.tstor,
            unit_aggr_tmap=self.indices.aggr_tstor_map,
        )

    def base_capacity_constraints(self) -> None:
        self.model.addConstrs(
            (
                self.variables.gen.cap[idx, 0] == val
                for idx, val in self.parameters.gen.base_cap.items()
            ),
            name="GEN_Y0_CAP_CONSTRAINT",
        )
        self.model.addConstrs(
            (
                self.variables.stor.cap[idx, 0] == val
                for idx, val in self.parameters.stor.base_cap.items()
            ),
            name="STOR_Y0_CAP_CONSTRAINT",
        )

    def _build_capacity_evolution_constraints(
        self,
        unit_ii: IndexingSet,
        unit_par: GeneratorParameters | StorageParameters,
        unit_tpar: GeneratorTypeParameters | StorageTypeParameters,
        unit_tidx: dict[int, int],
        unit_var: GeneratorVariables | StorageVariables,
        unit_aggr_map: dict[int, set],
    ) -> None:
        cap, cap_base_minus = unit_var.cap, unit_var.cap_base_minus
        cap_plus, cap_minus = unit_var.cap_plus, unit_var.cap_minus
        lbs_unit_idx = get_dict_vals(unit_aggr_map)
        for u_idx, u_name in unit_ii.mapping.items():
            if u_idx in lbs_unit_idx:  # if u_idx in any lbs then skipped
                continue
            base_cap = unit_par.base_cap[u_idx]
            lt = unit_tpar.lt[unit_tidx[u_idx]]
            bt = unit_tpar.bt[unit_tidx[u_idx]]
            for y in self.indices.Y.ord:
                initial_cap = (
                    base_cap
                    - quicksum(cap_base_minus[u_idx, s] for s in range(1, y + 1))
                    if y < lt
                    else 0
                )
                incr_cap = quicksum(
                    cap_plus[u_idx, s] for s in self._s_range(y, lt, bt)
                )
                decr_cap = quicksum(
                    cap_minus[u_idx, s, t]
                    for s in self._s_range(y, lt, bt)
                    for t in self._t_range(y, s, lt, bt)
                )
                self.model.addConstr(
                    cap[u_idx, y] == initial_cap + incr_cap - decr_cap,
                    name=f"{unit_ii.name}_{u_name}_Y_{y}_CAPACITY_EVOLUTION_CONSTRAINT",
                )

    def _build_local_capacity_evolution_constraints(
        self,
        unit_par: GeneratorParameters | StorageParameters,
        unit_tpar: GeneratorTypeParameters | StorageTypeParameters,
        unit_tidx: dict[int, int],
        unit_tvar: GeneratorTypeVariables | StorageTypeVariables,
        unit_aggr_map: dict[int, set],
        unit_aggr_tmap: dict[int, set],
    ) -> None:
        tcap, tcap_base_minus = unit_tvar.tcap, unit_tvar.tcap_base_minus
        tcap_plus, tcap_minus = unit_tvar.tcap_plus, unit_tvar.tcap_minus
        for aggr_idx in unit_aggr_map.keys():
            for t_idx in unit_aggr_tmap[aggr_idx]:
                u_idxs = self._get_unit_idx_from_type(unit_tidx, t_idx)
                base_cap = quicksum(unit_par.base_cap[u_idx] for u_idx in u_idxs)
                lt = unit_tpar.lt[t_idx]
                bt = unit_tpar.bt[t_idx]
                for y in self.indices.Y.ord:
                    initial_cap = (
                        base_cap
                        - quicksum(
                            tcap_base_minus[aggr_idx, t_idx, s] for s in range(1, y + 1)
                        )
                        if y < lt
                        else 0
                    )
                    incr_cap = quicksum(
                        tcap_plus[aggr_idx, t_idx, s] for s in self._s_range(y, lt, bt)
                    )
                    decr_cap = quicksum(
                        tcap_minus[aggr_idx, t_idx, s, t]
                        for s in self._s_range(y, lt, bt)
                        for t in self._t_range(y, s, lt, bt)
                    )
                    self.model.addConstr(
                        tcap[aggr_idx, t_idx, y] == initial_cap + incr_cap - decr_cap,
                        name=f"aggr_{aggr_idx}_type_{t_idx}_Y_{y}_LOCAL_CAPACITY_EVOLUTION_CONSTRAINT",
                    )

    def _build_reduced_capacity_upper_bound_constraints(
        self,
        unit_ii: IndexingSet,
        unit_tpar: GeneratorTypeParameters | StorageTypeParameters,
        unit_tidx: dict[int, int],
        unit_var: GeneratorVariables | StorageVariables,
        unit_aggr_map: dict[int, set],
    ) -> None:
        cap_plus, cap_minus = unit_var.cap_plus, unit_var.cap_minus
        lbs_unit_idx = get_dict_vals(unit_aggr_map)
        for u_idx, u_name in unit_ii.mapping.items():
            if u_idx in lbs_unit_idx:
                continue
            lt, bt = unit_tpar.lt[unit_tidx[u_idx]], unit_tpar.bt[unit_tidx[u_idx]]
            for y in self.indices.Y.ord:
                zero_cap_minus_sum = quicksum(
                    cap_minus[u_idx, y, s] for s in self._t_range(y, y, lt, bt)
                )
                self.model.addConstr(
                    zero_cap_minus_sum == 0,
                    name=f"{unit_ii.name}_{u_name}_Y_{y}_ZERO_REDUCED_CAPACITY_CONSTRAINT",
                )
                all_cap_minus_sum = quicksum(
                    cap_minus[u_idx, y, s] for s in self.indices.Y.ord
                )
                self.model.addConstr(
                    all_cap_minus_sum <= cap_plus[u_idx, y],
                    name=f"{unit_ii.name}_{u_name}_Y_{y}_REDUCED_CAPACITY_UB_CONSTRAINT",
                )

    def _build_local_supplementary_capacity_upper_bound_constraints(
        self,
        unit_tpar: GeneratorTypeParameters | StorageTypeParameters,
        unit_tidx: dict[int, int],
        unit_var: GeneratorVariables | StorageVariables,
        unit_tvar: GeneratorTypeVariables | StorageTypeVariables,
        unit_aggr_tmap: dict[int, set],
    ) -> None:
        cap = unit_var.cap
        tcap, tcap_plus, tcap_minus = (
            unit_tvar.tcap,
            unit_tvar.tcap_plus,
            unit_tvar.tcap_minus,
        )
        for aggr_idx in unit_aggr_tmap.keys():
            for type_idx in unit_aggr_tmap[aggr_idx]:
                lt, bt = (
                    unit_tpar.lt[type_idx],
                    unit_tpar.bt[type_idx],
                )
                u_idxs = self._get_unit_idx_from_type(unit_tidx, type_idx)

                for y in self.indices.Y.ord:
                    zero_cap_minus_sum = quicksum(
                        tcap_minus[aggr_idx, type_idx, y, s]
                        for s in self._t_range(y, y, lt, bt)
                    )
                    self.model.addConstr(
                        zero_cap_minus_sum == 0,
                        name=f"aggr_idx_{aggr_idx}_t_idx_{type_idx}_Y_{y}_LOCAL_ZERO_REDUCED_CAPACITY_CONSTRAINT",
                    )

                    t_all_cap_minus_sum = quicksum(
                        tcap_minus[aggr_idx, type_idx, y, s] for s in self.indices.Y.ord
                    )
                    self.model.addConstr(
                        t_all_cap_minus_sum <= tcap_plus[aggr_idx, type_idx, y],
                        name=f"aggr_idx_{aggr_idx}_t_idx_{type_idx}_Y_{y}_REDUCED_CAPACITY_UB_CONSTRAINT",
                    )

                    # definitions of t_cap in evolution equations:

                    self.model.addConstr(
                        tcap[aggr_idx, type_idx, y]
                        == quicksum(cap[u_idx, y] for u_idx in u_idxs),
                        name=f"cap_{aggr_idx}_t_idx_{type_idx}_Y_{y}_CAP_LOCAL_SUM_CONSTRAINT",
                    )

    def _build_n_min_max_power_constraints(
        self,
        unit_ii: IndexingSet,
        unit_par: GeneratorParameters | StorageParameters,
        unit_var: GeneratorVariables | StorageVariables,
    ) -> None:
        """Slavkov problem"""
        for u_idx, u_name in unit_ii.mapping.items():
            for lbs_idx in self.parameters.lbs.buses.keys():
                lbs_buses = set().union(
                    *list(self.parameters.lbs.buses[lbs_idx].values())
                )
                unit_buses = (
                    {unit_par.bus[u_idx]}
                    if isinstance(unit_par, StorageParameters)
                    else unit_par.buses[u_idx]
                )
                if not unit_buses.isdisjoint(lbs_buses):
                    self._build_n_min_max_power_for_aggr_constraints(
                        lbs_idx,
                        u_idx,
                        u_name,
                        unit_par,
                        unit_var,
                    )

    def _build_n_min_max_power_for_aggr_constraints(
        self,
        lbs_idx: int,
        u_idx: int,
        u_name: str,
        unit_par: GeneratorParameters | StorageParameters,
        unit_var: GeneratorVariables | StorageVariables,
    ) -> None:
        aggr_idx = [
            ii
            for ii in self.indices.AGGR.ord
            if self.parameters.aggr.lbs_indicator[ii, lbs_idx] == 1
        ].pop()
        for y in self.indices.Y.ord[1:]:
            if u_idx in unit_par.min_device_nom_power:
                min_aggregated_power = (
                    self.parameters.aggr.n_consumers[aggr_idx][y]
                    * unit_par.min_device_nom_power[u_idx]
                    * self.variables.frac.fraction[aggr_idx, lbs_idx, y]
                )
                self.model.addConstr(
                    min_aggregated_power <= unit_var.cap[u_idx, y],
                    name=f"{aggr_idx}_{u_name}_{y}_DEVICE_MIN_POWER_CONSTRAINT",
                )
            if u_idx in unit_par.max_device_nom_power:
                max_aggregated_power = (
                    self.parameters.aggr.n_consumers[aggr_idx][y]
                    * unit_par.max_device_nom_power[u_idx]
                    * self.variables.frac.fraction[aggr_idx, lbs_idx, y]
                )
                self.model.addConstr(
                    max_aggregated_power >= unit_var.cap[u_idx, y],
                    name=f"{aggr_idx}_{u_name}_{y}_DEVICE_MAX_POWER_CONSTRAIN",
                )

    @staticmethod
    def _get_unit_idx_from_type(unit_t_idx: dict[int, int], type_idx: int) -> set[int]:
        return {
            u_idx for u_idx, u_type_idx in unit_t_idx.items() if u_type_idx == type_idx
        }

    @staticmethod
    def _s_range(y: int, lt: int, bt: int) -> range:
        return range(max(0, y - lt - bt + 1), y - bt + 1)

    @staticmethod
    def _t_range(y: int, s: int, lt: int, bt: int) -> range:
        return range(s + bt, min(y, s + bt + lt - 1) + 1)
