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

from pyzefir.optimization.linopy.constraints_builder.builder import (
    PartialConstraintsBuilder,
)
from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet
from pyzefir.optimization.linopy.preprocessing.parameters.generator_parameters import (
    GeneratorParameters,
)
from pyzefir.optimization.linopy.preprocessing.parameters.generator_type_parameters import (
    GeneratorTypeParameters,
)
from pyzefir.optimization.linopy.preprocessing.parameters.storage_parameters import (
    StorageParameters,
)
from pyzefir.optimization.linopy.preprocessing.parameters.storage_type_parameters import (
    StorageTypeParameters,
)
from pyzefir.optimization.linopy.preprocessing.variables.generator_type_variables import (
    GeneratorTypeVariables,
)
from pyzefir.optimization.linopy.preprocessing.variables.generator_variables import (
    GeneratorVariables,
)
from pyzefir.optimization.linopy.preprocessing.variables.storage_type_variables import (
    StorageTypeVariables,
)
from pyzefir.optimization.linopy.preprocessing.variables.storage_variables import (
    StorageVariables,
)
from pyzefir.utils.functions import get_dict_vals

_logger = logging.getLogger(__name__)


class CapacityEvolutionConstrBuilder(PartialConstraintsBuilder):
    def build_constraints(self) -> None:
        _logger.info("Capacity evolution constraints builder is working...")
        self.capacity_evolution_constraints()
        self.supplementary_evolution_constraints()
        self.base_capacity_constraints()
        self.generator_n_min_max_power_constraints()
        _logger.info("Capacity evolution constraints builder is finished!")

    def generator_n_min_max_power_constraints(self) -> None:
        _logger.debug("Building power constraints...")
        self._build_n_min_max_power_constraints(
            self.indices.GEN, self.parameters.gen, self.variables.gen
        )
        _logger.debug("Building generator power constraints: Done")
        self._build_n_min_max_power_constraints(
            self.indices.STOR, self.parameters.stor, self.variables.stor
        )
        _logger.debug("Building storage power constraints: Done")
        _logger.debug("Build power constraints: Done")

    def capacity_evolution_constraints(self) -> None:
        _logger.debug("Building capacity evolution constraints...")
        self._build_capacity_evolution_constraints_gen_stor()
        self._build_local_capacity_evolution_constraints_gen_stor()
        _logger.debug("Build capacity evolution constraints: Done")

    def _build_capacity_evolution_constraints_gen_stor(self) -> None:
        self._build_capacity_evolution_constraints(
            unit_ii=self.indices.GEN,
            unit_par=self.parameters.gen,
            unit_tpar=self.parameters.tgen,
            unit_tidx=self.parameters.gen.tgen,
            unit_var=self.variables.gen,
            unit_aggr_map=self.indices.aggr_gen_map,
        )
        _logger.debug("Build generation capacity evolution constraints: Done")
        self._build_capacity_evolution_constraints(
            unit_ii=self.indices.STOR,
            unit_par=self.parameters.stor,
            unit_tpar=self.parameters.tstor,
            unit_tidx=self.parameters.stor.tstor,
            unit_var=self.variables.stor,
            unit_aggr_map=self.indices.aggr_stor_map,
        )
        _logger.debug("Build storage capacity evolution constraints: Done")

    def _build_local_capacity_evolution_constraints_gen_stor(self) -> None:
        self._build_local_capacity_evolution_constraints(
            unit_par=self.parameters.gen,
            unit_tpar=self.parameters.tgen,
            unit_tidx=self.parameters.gen.tgen,
            unit_tvar=self.variables.tgen,
            unit_aggr_map=self.indices.aggr_gen_map,
            unit_aggr_tmap=self.indices.aggr_tgen_map,
            unit_type="GEN",
        )
        _logger.debug("Build local generation capacity evolution constraints: Done")
        self._build_local_capacity_evolution_constraints(
            unit_par=self.parameters.stor,
            unit_tpar=self.parameters.tstor,
            unit_tidx=self.parameters.stor.tstor,
            unit_tvar=self.variables.tstor,
            unit_aggr_map=self.indices.aggr_stor_map,
            unit_aggr_tmap=self.indices.aggr_tstor_map,
            unit_type="STOR",
        )
        _logger.debug("Build local storage capacity evolution constraints: Done")

    def supplementary_evolution_constraints(self) -> None:
        _logger.debug("Building supplementary evolution constraints...")
        self._build_reduced_capacity_upper_bound_constraints_gen_stor()
        self._build_local_supplementary_capacity_upper_bound_constraints_gen_stor()
        _logger.debug("Build supplementary evolution constraints: Done")

    def _build_reduced_capacity_upper_bound_constraints_gen_stor(self) -> None:
        self._build_reduced_capacity_upper_bound_constraints(
            unit_ii=self.indices.GEN,
            unit_tpar=self.parameters.tgen,
            unit_tidx=self.parameters.gen.tgen,
            unit_var=self.variables.gen,
            unit_aggr_map=self.indices.aggr_gen_map,
        )
        _logger.debug("Build generation reduced capacity upper bound constraints: Done")
        self._build_reduced_capacity_upper_bound_constraints(
            unit_ii=self.indices.STOR,
            unit_tpar=self.parameters.tstor,
            unit_tidx=self.parameters.stor.tstor,
            unit_var=self.variables.stor,
            unit_aggr_map=self.indices.aggr_stor_map,
        )
        _logger.debug("Build storage reduced capacity upper bound constraints: Done")

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
            unit_aggr_map=self.indices.aggr_gen_map,
        )
        _logger.debug(
            "Build generation local supplementary capacity upper bound constraints: Done"
        )
        self._build_local_supplementary_capacity_upper_bound_constraints(
            unit_tpar=self.parameters.tstor,
            unit_tidx=self.parameters.stor.tstor,
            unit_var=self.variables.stor,
            unit_tvar=self.variables.tstor,
            unit_aggr_tmap=self.indices.aggr_tstor_map,
            unit_aggr_map=self.indices.aggr_stor_map,
        )
        _logger.debug(
            "Build storage local supplementary capacity upper bound constraints: Done"
        )

    def base_capacity_constraints(self) -> None:
        for idx, val in self.parameters.gen.base_cap.items():
            self.model.add_constraints(
                self.variables.gen.cap.isel(gen=idx, year=0) == val,
                name=f"GEN_{idx}_Y0_CAP_CONSTRAINT",
            )

        for idx, val in self.parameters.stor.base_cap.items():
            self.model.add_constraints(
                self.variables.stor.cap.isel(stor=idx, year=0) == val,
                name=f"STOR_{idx}_Y0_CAP_CONSTRAINT",
            )
        _logger.debug("Build base capacity constraints: Done")

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
                    -cap_base_minus.sel(
                        index=[(u_idx, s) for s in range(1, y + 1)]
                    ).sum()
                    + base_cap
                    if y < lt
                    else 0
                )
                incr_cap = cap_plus.sel(
                    index=[(u_idx, s) for s in self._s_range(y, lt, bt)]
                ).sum()
                decr_cap = cap_minus.sel(
                    index=[
                        (u_idx, s, t)
                        for s in self._s_range(y, lt, bt)
                        for t in self._t_range(y, s, lt, bt)
                    ]
                ).sum()
                self.model.add_constraints(
                    cap.isel(**{cap.dims[0]: u_idx, "year": y})
                    == initial_cap + incr_cap - decr_cap,
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
        unit_type: str,
    ) -> None:
        tcap, tcap_base_minus = unit_tvar.tcap, unit_tvar.tcap_base_minus
        tcap_plus, tcap_minus = unit_tvar.tcap_plus, unit_tvar.tcap_minus
        for aggr_idx in unit_aggr_map.keys():
            for t_idx in unit_aggr_tmap[aggr_idx]:
                u_idxs = self._get_unit_idx_from_type(
                    unit_tidx, t_idx, unit_aggr_map[aggr_idx]
                )
                base_cap: np.ndarray = np.sum(
                    [unit_par.base_cap[u_idx] for u_idx in u_idxs],
                )
                lt = unit_tpar.lt[t_idx]
                bt = unit_tpar.bt[t_idx]
                for y in self.indices.Y.ord:
                    initial_cap = (
                        -tcap_base_minus.sel(
                            index=[(aggr_idx, t_idx, s) for s in range(1, y + 1)]
                        ).sum()
                        + base_cap
                        if y < lt
                        else 0
                    )
                    incr_cap = tcap_plus.sel(
                        index=[(aggr_idx, t_idx, s) for s in self._s_range(y, lt, bt)]
                    ).sum()
                    decr_cap = tcap_minus.sel(
                        index=[
                            (aggr_idx, t_idx, s, t)
                            for s in self._s_range(y, lt, bt)
                            for t in self._t_range(y, s, lt, bt)
                        ]
                    ).sum()
                    self.model.add_constraints(
                        tcap.sel(index=[(aggr_idx, t_idx, y)])
                        == initial_cap + incr_cap - decr_cap,
                        name=f"aggr_{aggr_idx}_{unit_type}_type_{t_idx}_Y_{y}_LOCAL_CAPACITY_EVOLUTION_CONSTRAINT",
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
                zero_cap_minus_sum = cap_minus.sel(
                    index=[(u_idx, y, s) for s in self._t_range(y, y, lt, bt)]
                ).sum()
                self.model.add_constraints(
                    zero_cap_minus_sum == 0,
                    name=f"{unit_ii.name}_{u_name}_Y_{y}_ZERO_REDUCED_CAPACITY_CONSTRAINT",
                )
                all_cap_minus_sum = cap_minus.sel(
                    index=[(u_idx, y, s) for s in self.indices.Y.ord]
                ).sum()
                self.model.add_constraints(
                    all_cap_minus_sum <= cap_plus.sel(index=[(u_idx, y)]),
                    name=f"{unit_ii.name}_{u_name}_Y_{y}_REDUCED_CAPACITY_UB_CONSTRAINT",
                )

    def _build_local_supplementary_capacity_upper_bound_constraints(
        self,
        unit_tpar: GeneratorTypeParameters | StorageTypeParameters,
        unit_tidx: dict[int, int],
        unit_var: GeneratorVariables | StorageVariables,
        unit_tvar: GeneratorTypeVariables | StorageTypeVariables,
        unit_aggr_tmap: dict[int, set],
        unit_aggr_map: dict[int, set],
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
                u_idxs = self._get_unit_idx_from_type(
                    unit_tidx, type_idx, unit_aggr_map[aggr_idx]
                )

                for y in self.indices.Y.ord:
                    zero_cap_minus_sum = tcap_minus.sel(
                        index=[
                            (aggr_idx, type_idx, y, s)
                            for s in self._t_range(y, y, lt, bt)
                        ]
                    )
                    self.model.add_constraints(
                        zero_cap_minus_sum == 0,
                        name=f"aggr_idx_{aggr_idx}_{cap.dims[0]}_t_idx_{type_idx}_Y_{y}"
                        "_LOCAL_ZERO_REDUCED_CAPACITY_CONSTRAINT",
                    )
                    t_all_cap_minus_sum = tcap_minus.sel(
                        index=[(aggr_idx, type_idx, y, s) for s in self.indices.Y.ord]
                    ).sum()
                    self.model.add_constraints(
                        t_all_cap_minus_sum
                        <= tcap_plus.sel(index=[(aggr_idx, type_idx, y)]),
                        name=f"aggr_idx_{aggr_idx}_{cap.dims[0]}_t_idx_{type_idx}_Y_{y}"
                        "_LOCAL_REDUCED_CAPACITY_UB_CONSTRAINT",
                    )

                    # definitions of t_cap in evolution equations:

                    self.model.add_constraints(
                        tcap.sel(index=[(aggr_idx, type_idx, y)])
                        == cap.isel(**{cap.dims[0]: u_idxs, "year": y}).sum(),
                        name=f"cap_{aggr_idx}_{cap.dims[0]}_t_idx_{type_idx}_Y_{y}_CAP_LOCAL_SUM_CONSTRAINT",
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
                    * self.variables.frac.fraction.isel(
                        aggr=aggr_idx, lbs=lbs_idx, year=y
                    )
                )
                self.model.add_constraints(
                    min_aggregated_power
                    <= unit_var.cap.isel(**{unit_var.cap.dims[0]: u_idx, "year": y}),
                    name=f"{aggr_idx}_{u_name}_{y}_DEVICE_MIN_POWER_CONSTRAINT",
                )
            if u_idx in unit_par.max_device_nom_power:
                max_aggregated_power = (
                    self.parameters.aggr.n_consumers[aggr_idx][y]
                    * unit_par.max_device_nom_power[u_idx]
                    * self.variables.frac.fraction.isel(
                        aggr=aggr_idx, lbs=lbs_idx, year=y
                    )
                )
                self.model.add_constraints(
                    max_aggregated_power
                    >= unit_var.cap.isel(**{unit_var.cap.dims[0]: u_idx, "year": y}),
                    name=f"{aggr_idx}_{u_name}_{y}_DEVICE_MAX_POWER_CONSTRAIN",
                )

    @staticmethod
    def _get_unit_idx_from_type(
        unit_t_idx: dict[int, int], type_idx: int, unit_in_aggr: set[int]
    ) -> list[int]:
        return [
            u_idx
            for u_idx, u_type_idx in unit_t_idx.items()
            if u_type_idx == type_idx and u_idx in unit_in_aggr
        ]

    @staticmethod
    def _s_range(y: int, lt: int, bt: int) -> range:
        return range(max(0, y - lt - bt + 1), y - bt + 1)

    @staticmethod
    def _t_range(y: int, s: int, lt: int, bt: int) -> range:
        return range(s + bt, min(y, s + bt + lt - 1) + 1)
