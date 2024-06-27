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

import numpy as np
import pandas as pd

from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.linopy.preprocessing.parameters.generator_type_parameters import (
    GeneratorTypeParameters,
)
from pyzefir.optimization.linopy.preprocessing.parameters.storage_type_parameters import (
    StorageTypeParameters,
)
from pyzefir.optimization.linopy.utils import get_generator_types_capacity_multipliers
from pyzefir.optimization.results import Results
from pyzefir.utils.functions import get_dict_vals


def objective_capex(
    indices: Indices, parameters: OptimizationParameters, results: Results
) -> float:
    return local_capex(indices, parameters, results) + global_capex(
        indices, parameters, results
    )


def local_capex(
    indices: Indices, parameters: OptimizationParameters, results: Results
) -> float:
    """variables- after optimization"""
    generator_capex = _local_capex(
        indices=indices,
        unit_indices=indices.TGEN,
        parameters=parameters,
        tcap_plus=results.generators_results.tcap_plus,
        unit_type_param=parameters.tgen,
        aggr_map=indices.aggr_tgen_map,
        multipliers=get_generator_types_capacity_multipliers(
            parameters.scenario_parameters.generator_capacity_cost,
            parameters.tgen,
        ),
    )
    storage_capex = _local_capex(
        indices=indices,
        unit_indices=indices.TSTOR,
        parameters=parameters,
        tcap_plus=results.storages_results.tcap_plus,
        unit_type_param=parameters.tstor,
        aggr_map=indices.aggr_tstor_map,
    )
    return generator_capex + storage_capex


def global_capex(
    indices: Indices, parameters: OptimizationParameters, results: Results
) -> float:
    generator_capex = _global_capex(
        indices=indices,
        unit_indices=indices.GEN,
        parameters=parameters,
        cap_plus=results.generators_results.cap_plus,
        unit_type_param=parameters.tgen,
        unit_type_idx=parameters.gen.tgen,
        non_lbs_unit_idxs=get_dict_vals(parameters.bus.generators).difference(
            get_dict_vals(indices.aggr_gen_map)
        ),
        multipliers=get_generator_types_capacity_multipliers(
            parameters.scenario_parameters.generator_capacity_cost,
            parameters.tgen,
        ),
    )
    storage_capex = _global_capex(
        indices=indices,
        unit_indices=indices.STOR,
        parameters=parameters,
        cap_plus=results.storages_results.cap_plus,
        unit_type_param=parameters.tstor,
        unit_type_idx=parameters.stor.tstor,
        non_lbs_unit_idxs=get_dict_vals(parameters.bus.storages).difference(
            get_dict_vals(indices.aggr_stor_map)
        ),
    )
    return generator_capex + storage_capex


def _global_capex(
    indices: Indices,
    unit_indices: IndexingSet,
    parameters: OptimizationParameters,
    cap_plus: dict[str, pd.DataFrame],
    unit_type_param: GeneratorTypeParameters | StorageTypeParameters,
    unit_type_idx: dict,
    non_lbs_unit_idxs: set,
    multipliers: dict[int, float] | None = None,
) -> float:
    disc_rate = discount_rate(parameters.scenario_parameters.discount_rate)
    y_idxs = indices.Y
    unit_capex = 0.0
    for u_idx in non_lbs_unit_idxs:
        ut_idx = unit_type_idx[u_idx]
        mul = multipliers[ut_idx] if multipliers is not None else 1.0
        capex = unit_type_param.capex[ut_idx]
        lt = unit_type_param.lt[ut_idx]

        unit_capex += sum(
            global_capex_per_unit_per_year(
                unit_indices=unit_indices,
                capex=capex,
                cap_plus=cap_plus,
                disc_rate=disc_rate,
                lt=lt,
                s_idx=s_idx,
                u_idx=u_idx,
                y_idxs=y_idxs,
            )
            * mul
            for s_idx in y_idxs.ord
        )
    return unit_capex


def global_capex_per_unit_per_year(
    unit_indices: IndexingSet,
    capex: np.ndarray,
    cap_plus: dict[str, pd.DataFrame],
    disc_rate: np.ndarray,
    lt: int,
    s_idx: int,
    u_idx: int,
    y_idxs: IndexingSet,
) -> float:
    am_indicator = _amortization_matrix_indicator(lt=lt, yy=y_idxs)
    return sum(
        am_indicator[s_idx, y_idx]
        * capex[s_idx]
        * cap_plus[unit_indices.mapping[u_idx]][0][s_idx]
        * disc_rate[y_idx]
        / lt
        for y_idx in y_idxs.ord
    )


def _local_capex(
    indices: Indices,
    unit_indices: IndexingSet,
    parameters: OptimizationParameters,
    tcap_plus: dict[str, dict[str, pd.DataFrame]],
    unit_type_param: GeneratorTypeParameters | StorageTypeParameters,
    aggr_map: dict[..., set],
    multipliers: dict[int, float] | None = None,
) -> float:
    disc_rate = discount_rate(parameters.scenario_parameters.discount_rate)
    y_idxs = indices.Y
    unit_type_capex = 0.0
    for aggr_idx, ut_idxs in aggr_map.items():
        for ut_idx in ut_idxs:
            mul = multipliers[ut_idx] if multipliers is not None else 1.0
            capex = unit_type_param.capex[ut_idx]
            lt = unit_type_param.lt[ut_idx]
            ut_name = unit_indices.mapping[ut_idx]
            unit_type_capex += (
                sum(
                    local_capex_per_unit_per_year(
                        indices=indices,
                        capex=capex,
                        tcap_plus=tcap_plus,
                        disc_rate=disc_rate,
                        lt=lt,
                        s_idx=s_idx,
                        ut_name=ut_name,
                        aggr_idx=aggr_idx,
                        y_idxs=y_idxs,
                    )
                    for s_idx in y_idxs.ord
                )
                * mul
            )
    return unit_type_capex


def local_capex_per_unit_per_year(
    indices: Indices,
    capex: np.ndarray,
    tcap_plus: dict[str, dict[str, pd.DataFrame]],
    disc_rate: np.ndarray,
    lt: int,
    s_idx: int,
    ut_name: str,
    aggr_idx: int,
    y_idxs: IndexingSet,
) -> float:
    am_indicator = _amortization_matrix_indicator(lt=lt, yy=y_idxs)
    aggr_map = indices.AGGR.mapping

    return sum(
        am_indicator[s_idx, y_idx]
        * capex[s_idx]
        * tcap_plus[aggr_map[aggr_idx]][ut_name]["solution"][s_idx]
        * disc_rate[y_idx]
        / lt
        for y_idx in y_idxs.ord
    )


def discount_rate(yearly_rate: np.ndarray) -> np.ndarray:
    """Vector of discount rates for each year.

    Returns:
        np.ndarray: discount rate
    """
    return np.cumprod((1 + yearly_rate) ** (-1))


def _amortization_matrix_indicator(
    lt: int,
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
            ((yy.ord >= y) & (yy.ord <= min(y + lt - 1, len(yy)))).astype(int)
            for y in yy.ord
        ]
    )
