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
from pyzefir.optimization.results import Results
from pyzefir.utils.functions import get_dict_vals, invert_dict_of_sets


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
) -> float:
    disc_rate = discount_rate(parameters.scenario_parameters.discount_rate)
    y_idxs = indices.Y
    unit_capex = 0.0
    for u_idx in non_lbs_unit_idxs:
        ut_idx = unit_type_idx[u_idx]
        capex = unit_type_param.capex[ut_idx]
        lt = unit_type_param.lt[ut_idx]
        bt = unit_type_param.bt[ut_idx]

        unit_capex += sum(
            global_capex_per_unit_per_year(
                unit_indices=unit_indices,
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


def global_capex_per_unit_per_year(
    unit_indices: IndexingSet,
    capex: np.ndarray,
    cap_plus: dict[str, pd.DataFrame],
    disc_rate: np.ndarray,
    bt: int,
    lt: int,
    y_idx: int,
    u_idx: int,
    y_idxs: IndexingSet,
) -> float:
    am_indicator = _amortization_matrix_indicator(lt=lt, bt=bt, yy=y_idxs)
    return sum(
        am_indicator[s, y_idx]
        * capex[s]
        * cap_plus[unit_indices.mapping[u_idx]][0][s]
        * disc_rate[y_idx]
        / lt
        for s in y_idxs.ord
    )


def _local_capex(
    indices: Indices,
    unit_indices: IndexingSet,
    parameters: OptimizationParameters,
    tcap_plus: dict[str, dict[str, pd.DataFrame]],
    unit_type_param: GeneratorTypeParameters | StorageTypeParameters,
    aggr_map: dict[..., set],
) -> float:
    disc_rate = discount_rate(parameters.scenario_parameters.discount_rate)
    y_idxs = indices.Y
    unit_type_capex = 0.0
    inverted_aggr_map = invert_dict_of_sets(aggr_map)
    for ut_idx, aggr_idxs in inverted_aggr_map.items():
        capex = unit_type_param.capex[ut_idx]
        lt = unit_type_param.lt[ut_idx]
        bt = unit_type_param.bt[ut_idx]
        ut_name = unit_indices.mapping[ut_idx]

        unit_type_capex += sum(
            local_capex_per_unit_per_year(
                indices=indices,
                capex=capex,
                tcap_plus=tcap_plus,
                disc_rate=disc_rate,
                bt=bt,
                lt=lt,
                y_idx=y_idx,
                ut_name=ut_name,
                aggr_idxs=aggr_idxs,
                y_idxs=y_idxs,
            )
            for y_idx in y_idxs.ord
        )

    return unit_type_capex


def local_capex_per_unit_per_year(
    indices: Indices,
    capex: np.ndarray,
    tcap_plus: dict[str, dict[str, pd.DataFrame]],
    disc_rate: np.ndarray,
    bt: int,
    lt: int,
    y_idx: int,
    ut_name: str,
    aggr_idxs: set[int],
    y_idxs: IndexingSet,
) -> float:
    am_indicator = _amortization_matrix_indicator(lt=lt, bt=bt, yy=y_idxs)
    aggr_map = indices.AGGR.mapping

    return sum(
        am_indicator[s, y_idx]
        * capex[s]
        * tcap_plus[aggr_map[aggr_idx]][ut_name]["solution"][s]
        * disc_rate[y_idx]
        / lt
        for s in y_idxs.ord
        for aggr_idx in aggr_idxs
    )


def discount_rate(yearly_rate: np.ndarray) -> np.ndarray:
    """Vector of discount rates for each year.

    Returns:
        np.ndarray: discount rate
    """
    return np.cumprod((1 + yearly_rate) ** (-1))


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
            ((yy.ord >= y + bt) & (yy.ord <= min(y + bt + lt - 1, len(yy)))).astype(int)
            for y in yy.ord
        ]
    )
