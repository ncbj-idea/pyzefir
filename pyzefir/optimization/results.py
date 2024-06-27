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

import abc
from dataclasses import InitVar, dataclass, field
from typing import Final

import numpy as np
import pandas as pd
import xarray as xr
from bidict import bidict
from linopy import Variable

from pyzefir.optimization.exportable_results import (
    ExportableBusResults,
    ExportableFractionsResults,
    ExportableGeneratorsResults,
    ExportableLinesResults,
    ExportableResults,
    ExportableResultsGroup,
    ExportableStorageResults,
)
from pyzefir.optimization.linopy.expression_handler import ExpressionHandler
from pyzefir.optimization.linopy.objective_builder.capex_objective_builder import (
    CapexObjectiveBuilder,
)
from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.linopy.preprocessing.opt_variables import (
    OptimizationVariables,
)
from pyzefir.optimization.linopy.preprocessing.parameters.bus_parameters import (
    BusParameters,
)
from pyzefir.optimization.linopy.preprocessing.parameters.generator_parameters import (
    GeneratorParameters,
)
from pyzefir.optimization.linopy.preprocessing.parameters.generator_type_parameters import (
    GeneratorTypeParameters,
)
from pyzefir.optimization.linopy.preprocessing.parameters.scenario_parameters import (
    ScenarioParameters,
)
from pyzefir.optimization.linopy.preprocessing.parameters.storage_parameters import (
    StorageParameters,
)
from pyzefir.optimization.linopy.preprocessing.parameters.storage_type_parameters import (
    StorageTypeParameters,
)
from pyzefir.optimization.linopy.preprocessing.variables.bus_variables import (
    BusVariables,
)
from pyzefir.optimization.linopy.preprocessing.variables.fraction_variables import (
    FractionVariables,
)
from pyzefir.optimization.linopy.preprocessing.variables.generator_type_variables import (
    GeneratorTypeVariables,
)
from pyzefir.optimization.linopy.preprocessing.variables.generator_variables import (
    GeneratorVariables,
)
from pyzefir.optimization.linopy.preprocessing.variables.line_variables import (
    LineVariables,
)
from pyzefir.optimization.linopy.preprocessing.variables.storage_type_variables import (
    StorageTypeVariables,
)
from pyzefir.optimization.linopy.preprocessing.variables.storage_variables import (
    StorageVariables,
)
from pyzefir.optimization.linopy.utils import get_generator_types_capacity_multipliers
from pyzefir.utils.functions import get_dict_vals

HOUR_LABEL: Final[str] = "Hour"
YEAR_LABEL: Final[str] = "Year"
GENERATOR_LABEL: Final[str] = "Generator"
LBS_LABEL: Final[str] = "Local Balancing Stack"
STORAGE_LABEL: Final[str] = "Storage"
ENERGY_TYPE_LABEL: Final[str] = "Energy Type"


class ResultsGroup(abc.ABC):
    """A base class for fetching and organizing variables used in result groups.

    Methods:
        fetch_2D_variable(index: IndexingSet, variable: MVar) -> dict[str, pd.DataFrame]:
            Fetches a 2D variable and returns a dictionary mapping names to 2D Pandas DataFrames.

        fetch_1D_variable(index: IndexingSet, variable: MVar) -> dict[str, pd.DataFrame]:
            Fetches a 1D variable and returns a dictionary mapping names to 1D Pandas DataFrames.
    """

    @staticmethod
    def _rename_axes(
        df: pd.DataFrame,
        row_index: IndexingSet | None = None,
        column_index: IndexingSet | None = None,
    ) -> pd.DataFrame:
        if row_index is not None:
            df = df.rename(row_index.mapping, axis=0)
        if column_index is not None:
            df = df.rename(column_index.mapping, axis=1)

        return df

    @staticmethod
    def dict_of_1d_array_to_pandas(
        data: dict[str, pd.DataFrame],
        column_name: str,
        index_name: str = YEAR_LABEL,
    ) -> pd.DataFrame:
        # if data dictionary is empty, then we cannot use pd.concat
        if not data:
            return pd.DataFrame()

        for key in data:
            if len(data[key].shape) != 2 or data[key].shape[1] != 1:
                raise ValueError(
                    f"Only 1D Pandas DataFrame can be used."
                    f" Check the value for {key}"
                )

        df = pd.concat(data, axis=1).groupby(axis=1, level=0).sum()
        df.index.name = index_name
        df.columns.name = column_name
        return df

    @staticmethod
    def dict_of_2d_array_to_pandas(
        data: dict[str, pd.DataFrame],
        index_name: str = HOUR_LABEL,
        column_name: str = YEAR_LABEL,
    ) -> dict[str, pd.DataFrame]:
        for key in data:
            if len(data[key].shape) != 2 or data[key].shape[1] == 1:
                raise ValueError(
                    f"Only 2d Pandas DataFrame can be used."
                    f" Please check value for {key}"
                )
        ret = dict()
        for key in data:
            df = data[key]
            df.index.name = index_name
            df.columns.name = column_name
            ret[key] = df
        return ret

    @staticmethod
    def dict_of_dicts_of_arrays_to_pandas(
        data: dict[str, dict[str, pd.DataFrame]],
        index_name: str = HOUR_LABEL,
        column_name: str = YEAR_LABEL,
        energy_type_label: str = ENERGY_TYPE_LABEL,
    ) -> dict[str, pd.DataFrame]:
        ret = dict()
        for key, value_dict in data.items():
            df = pd.concat(value_dict).reset_index(
                level=0, names=[energy_type_label, index_name]
            )
            df.columns.name = column_name
            df.index.name = index_name
            ret[key] = df
        return ret

    @abc.abstractmethod
    def to_exportable(self) -> ExportableResultsGroup:
        raise NotImplementedError

    @staticmethod
    def global_capex_per_unit_per_year(
        capex: np.ndarray,
        cap_plus: xr.DataArray,
        disc_rate: np.ndarray,
        lt: int,
        s_idx: int,
        u_idx: int,
        y_idxs: IndexingSet,
    ) -> float:
        am_indicator = CapexObjectiveBuilder._amortization_matrix_indicator(
            lt=lt, yy=y_idxs
        )
        res = 0.0
        for y_idx in y_idxs.ord:
            res += (
                cap_plus.sel(index=(u_idx, y_idx)).to_numpy()
                * am_indicator[y_idx, s_idx]
                * capex[y_idx]
                * disc_rate[s_idx]
                / lt
            )

        return res

    @staticmethod
    def local_capex_per_unit_per_year(
        capex: np.ndarray,
        tcap_plus: xr.DataArray,
        disc_rate: np.ndarray,
        lt: int,
        s_idx: int,
        ut_idx: int,
        aggr_idx: int,
        y_idxs: IndexingSet,
    ) -> float:
        am_indicator = CapexObjectiveBuilder._amortization_matrix_indicator(
            lt=lt, yy=y_idxs
        )

        res = 0.0
        for y_idx in y_idxs.ord:
            res += (
                tcap_plus.sel(index=(aggr_idx, ut_idx, y_idx)).to_numpy()
                * am_indicator[y_idx, s_idx]
                * capex[y_idx]
                * disc_rate[s_idx]
                / lt
            )
        return res

    @staticmethod
    def calculate_global_capex(
        discount_rate: np.ndarray,
        bus_unit_mapping: dict[int, set[int]],
        unit_index: IndexingSet,
        aggr_unit_map: dict[int, set[int]],
        indices: Indices,
        unit_type_map: dict[int, int],
        unit_type_param: GeneratorTypeParameters | StorageTypeParameters,
        money_scale: float,
        cap_plus: Variable,
        multipliers: dict[int, float] | None = None,
    ) -> dict[str, pd.DataFrame]:
        disc_rate = ExpressionHandler.discount_rate(discount_rate)
        non_lbs_unit_idxs: dict[int, int] = {
            unit_idx: unit_type_map[unit_idx]
            for unit_idx in get_dict_vals(bus_unit_mapping).difference(
                get_dict_vals(aggr_unit_map)
            )
        }
        year_idxs = indices.Y
        result = {}
        for u_idx, ut_idx in non_lbs_unit_idxs.items():
            capex = unit_type_param.capex[ut_idx]
            lt = unit_type_param.lt[ut_idx]
            year_results = dict()
            mul = multipliers[ut_idx] if multipliers is not None else 1.0
            for year_idx in year_idxs.ord:
                unit_capex = 0.0
                unit_capex += (
                    money_scale
                    * ResultsGroup.global_capex_per_unit_per_year(
                        capex=capex,
                        cap_plus=cap_plus.solution,
                        disc_rate=disc_rate,
                        lt=lt,
                        s_idx=year_idx,
                        u_idx=u_idx,
                        y_idxs=year_idxs,
                    )
                    * mul
                )
                year_results[indices.Y.mapping[year_idx]] = unit_capex
            result[unit_index.mapping[u_idx]] = pd.DataFrame.from_dict(
                year_results, orient="index"
            )
        return result

    @staticmethod
    def calculate_local_capex(
        discount_rate: np.ndarray,
        tcap_plus: Variable,
        indices: Indices,
        unit_type_param: GeneratorTypeParameters | StorageTypeParameters,
        money_scale: float,
        unit_type_map: dict[int, int],
        aggr_unit_map: dict[int, set[int]],
        gen_mapping: bidict,
        multipliers: dict[int, float] | None = None,
    ) -> dict[str, pd.DataFrame]:
        disc_rate = ExpressionHandler.discount_rate(discount_rate)
        year_idxs = indices.Y
        aggr_ut_idxs = {
            aggr_idx: {unit_type_map[ut_idx] for ut_idx in set_of_u_idxs}
            for aggr_idx, set_of_u_idxs in aggr_unit_map.items()
        }
        result = {}
        for aggr_idx, ut_idxs in aggr_ut_idxs.items():
            aggr_name = indices.AGGR.mapping.get(aggr_idx)
            aggr_result = {}
            for ut_idx in ut_idxs:
                mul = multipliers[ut_idx] if multipliers is not None else 1.0
                capex = unit_type_param.capex[ut_idx]
                lt = unit_type_param.lt[ut_idx]
                year_results = dict()
                for year_idx in year_idxs.ord:
                    unit_capex = 0.0
                    unit_capex += (
                        money_scale
                        * ResultsGroup.local_capex_per_unit_per_year(
                            capex=capex,
                            tcap_plus=tcap_plus.solution,
                            disc_rate=disc_rate,
                            lt=lt,
                            s_idx=year_idx,
                            ut_idx=ut_idx,
                            aggr_idx=aggr_idx,
                            y_idxs=year_idxs,
                        )
                        * mul
                    )
                    year_results[indices.Y.mapping[year_idx]] = unit_capex
                aggr_result[gen_mapping[ut_idx]] = pd.DataFrame.from_dict(
                    year_results, orient="index"
                )
            if aggr_result:
                df = pd.concat(aggr_result, axis=1)
                df.columns = df.columns.droplevel(1)
                result[aggr_name] = df
        return result


@dataclass
class GeneratorsResults(ResultsGroup):
    """Generators results"""

    variable_group: InitVar[GeneratorVariables]
    """Initial value hint for the GeneratorVariables object"""
    tvariable_group: InitVar[GeneratorTypeVariables]
    """Initial value hint for the GeneratorTypeVariables object"""
    tparameters: InitVar[GeneratorTypeParameters]
    """ Initial value hint for the GeneratorTypeParameters object """
    parameters: InitVar[GeneratorParameters]
    """ Initial value hint for the GeneratorParameters object """
    bus_parameters: InitVar[BusParameters]
    """ Initial value hint for the BusParameters object """
    scenario_parameters: InitVar[ScenarioParameters]
    """ Initial value hint for the ScenarioParameters object """
    indices: InitVar[Indices]
    """Initial value hint for the Indices object"""

    gen: dict[str, pd.DataFrame] = field(init=False)
    """ generation (exportable) """
    gen_et: dict[str, dict[str, pd.DataFrame]] = field(init=False)
    """ generation per energy type (exportable) """
    gen_dch: dict[str, dict[str, dict[str, pd.DataFrame]]] = field(init=False)
    """ generation per energy type for demand chunks (non-exportable) """
    dump: dict[str, pd.DataFrame] = field(init=False)
    """ dumped energy (exportable) """
    dump_et: dict[str, dict[str, pd.DataFrame]] = field(init=False)
    """ dumped energy per energy type (exportable) """
    cap: dict[str, pd.DataFrame] = field(init=False)
    """ capacity (exportable) """
    cap_plus: dict[str, pd.DataFrame] = field(init=False)
    """ capacity increase (non-exportable) """
    cap_minus: dict[str, pd.DataFrame] = field(init=False)
    """ capacity decrease (non-exportable) """
    cap_base_minus: dict[str, pd.DataFrame] = field(init=False)
    """ base capacity decrease (non-exportable) """
    global_capex: dict[str, pd.DataFrame] = field(init=False)
    """ capex of global technologies (exportable) """
    local_capex: dict[str, pd.DataFrame] = field(init=False)
    """ capex of local (in lbs) technologies (exportable) """

    def __post_init__(
        self,
        variable_group: GeneratorVariables,
        tvariable_group: GeneratorTypeVariables,
        tparameters: GeneratorTypeParameters,
        parameters: GeneratorParameters,
        bus_parameters: BusParameters,
        scenario_parameters: ScenarioParameters,
        indices: Indices,
    ) -> None:
        self.gen = self.process_gen(variable_group)
        self.gen_et = self.process_gen_et(variable_group)
        self.gen_dch = self.process_gen_dch(variable_group)
        self.dump = self.process_dump(variable_group)
        self.dump_et = self.process_dump_et(variable_group)
        self.cap = self.process_cap(variable_group)
        self.cap_plus = self.fetch_d_dataframe(
            dimension=1,
            index=indices.GEN,
            variable=variable_group.cap_plus.solution.to_dataframe(),
            row_index=indices.Y,
            column_index=indices.Y,
            filter_map=indices.aggr_gen_map,
        )
        self.cap_minus = self.fetch_d_dataframe(
            dimension=2,
            index=indices.GEN,
            variable=variable_group.cap_minus.solution.to_dataframe(),
            row_index=indices.Y,
            column_index=indices.Y,
            filter_map=indices.aggr_gen_map,
        )
        self.tcap_plus = self.fetch_d_tvariable(
            dimension=1,
            aggr_index=indices.AGGR,
            t_index=indices.TGEN,
            variable=tvariable_group.tcap_plus.solution.to_dataframe(),
            row_index=indices.Y,
            column_index=indices.Y,
            index_map=indices.aggr_tgen_map,
        )
        self.tcap_minus = self.fetch_d_tvariable(
            dimension=2,
            aggr_index=indices.AGGR,
            t_index=indices.TGEN,
            variable=tvariable_group.tcap_minus.solution.to_dataframe(),
            row_index=indices.Y,
            index_map=indices.aggr_tgen_map,
            column_index=indices.Y,
        )
        self.cap_base_minus = self.fetch_d_dataframe(
            dimension=1,
            index=indices.GEN,
            variable=variable_group.cap_base_minus.solution.to_dataframe(),
            row_index=indices.Y,
            filter_map=indices.aggr_gen_map,
            column_index=indices.Y,
        )
        self.tcap = self.fetch_d_tvariable(
            dimension=1,
            aggr_index=indices.AGGR,
            t_index=indices.TGEN,
            variable=tvariable_group.tcap.solution.to_dataframe(),
            row_index=indices.Y,
            index_map=indices.aggr_tgen_map,
            column_index=indices.Y,
        )
        self.tcap_base_minus = self.fetch_d_tvariable(
            dimension=1,
            aggr_index=indices.AGGR,
            t_index=indices.TGEN,
            variable=tvariable_group.tcap_base_minus.solution.to_dataframe(),
            row_index=indices.Y,
            index_map=indices.aggr_tgen_map,
            column_index=indices.Y,
        )
        self.global_capex = self.calculate_global_capex(
            indices=indices,
            unit_index=indices.GEN,
            unit_type_param=tparameters,
            unit_type_map=parameters.tgen,
            bus_unit_mapping=bus_parameters.generators,
            aggr_unit_map=indices.aggr_gen_map,
            discount_rate=scenario_parameters.discount_rate,
            cap_plus=variable_group.cap_plus,
            money_scale=scenario_parameters.money_scale,
            multipliers=get_generator_types_capacity_multipliers(
                scenario_parameters.generator_capacity_cost,
                tparameters,
            ),
        )
        self.local_capex = self.calculate_local_capex(
            indices=indices,
            unit_type_param=tparameters,
            unit_type_map=parameters.tgen,
            aggr_unit_map=indices.aggr_gen_map,
            discount_rate=scenario_parameters.discount_rate,
            tcap_plus=tvariable_group.tcap_plus,
            money_scale=scenario_parameters.money_scale,
            gen_mapping=indices.TGEN.mapping,
            multipliers=get_generator_types_capacity_multipliers(
                scenario_parameters.generator_capacity_cost,
                tparameters,
            ),
        )

    @staticmethod
    def process_gen(variable_group: GeneratorVariables) -> dict[str, pd.DataFrame]:
        return {
            gen_name: df.reset_index(["gen"], drop=True).unstack().droplevel(0, axis=1)
            for gen_name, df in variable_group.gen.solution.to_dataframe().groupby(
                "gen"
            )
        }

    @staticmethod
    def process_gen_et(
        variable_group: GeneratorVariables,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        return {
            gen_name: {
                energy_type: et_df.reset_index(["gen", "et"], drop=True)
                .unstack()
                .droplevel(0, axis=1)
                for energy_type, et_df in gen_df.groupby("et")
            }
            for gen_name, gen_df in variable_group.gen_et.solution.to_dataframe().groupby(
                "gen"
            )
        }

    @staticmethod
    def process_gen_dch(
        variable_group: GeneratorVariables,
    ) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
        return {
            gen_name: {
                energy_type: {
                    demand_chunk: dch_df.reset_index(["gen", "et", "demch"], drop=True)
                    .unstack()
                    .droplevel(0, axis=1)
                    for demand_chunk, dch_df in et_df.groupby("gen")
                }
                for energy_type, et_df in gen_df.groupby("demch")
            }
            for gen_name, gen_df in variable_group.gen_dch.solution.to_dataframe().groupby(
                "et"
            )
        }

    @staticmethod
    def process_dump(variable_group: GeneratorVariables) -> dict[str, pd.DataFrame]:
        return {
            gen_name: dump_df.reset_index(["gen"], drop=True)
            .unstack()
            .droplevel(0, axis=1)
            for gen_name, dump_df in variable_group.dump.solution.to_dataframe().groupby(
                "gen"
            )
        }

    @staticmethod
    def process_dump_et(
        variable_group: GeneratorVariables,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        return {
            gen_name: {
                energy_type: et_df.reset_index(["gen", "et"], drop=True)
                .unstack()
                .droplevel(0, axis=1)
                for energy_type, et_df in gen_df.groupby("et")
            }
            for gen_name, gen_df in variable_group.dump_et.solution.to_dataframe().groupby(
                "gen"
            )
        }

    @staticmethod
    def process_cap(variable_group: GeneratorVariables) -> dict[str, pd.DataFrame]:
        return {
            gen_name: cap_df.reset_index(["gen"], drop=True).rename(
                columns={"solution": "cap"}
            )
            for gen_name, cap_df in variable_group.cap.solution.to_dataframe().groupby(
                "gen"
            )
        }

    @staticmethod
    def fetch_d_dataframe(
        dimension: int,
        index: IndexingSet,
        variable: pd.DataFrame,
        row_index: IndexingSet,
        column_index: IndexingSet,
        filter_map: dict[int, set] | set | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetches a 1D variable from a DataFrame and returns a dictionary mapping names to 1D Pandas DataFrames.

        Args:
            index (IndexingSet): The indexing set for the variable.
            variable (pd.DataFrame): The DataFrame containing the 1D variable to fetch.
            row_index (IndexingSet): The indexing set for the variable's rows.
            column_index (IndexingSet | None): The indexing set for the variable's columns.
            filter_map (Dict | Set | None): dict of sets or set indices to filter from index mapping.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary mapping names to 1D Pandas DataFrames.
        """
        match filter_map:
            case dict():
                filter_idxs = get_dict_vals(filter_map)
            case set():
                filter_idxs = filter_map
            case _:
                filter_idxs = set()

        result_dict = dict()

        for idx, name in index.mapping.items():
            if idx in filter_idxs:
                continue
            if dimension == 1:
                vals = variable[
                    variable.index.isin([(idx, i) for i in row_index.ord])
                ].values
                df = pd.DataFrame(vals, index=row_index.ord)
            elif dimension == 2:
                vals = variable[
                    variable.index.isin(
                        [(idx, i, c) for i in row_index.ord for c in column_index.ord]
                    )
                ].values.reshape(len(column_index.ord), len(row_index.ord))
                df = pd.DataFrame(vals)
            # You can customize the renaming of axes based on your requirements
            ResultsGroup._rename_axes(df, row_index, column_index)

            result_dict[name] = df

        return result_dict

    @staticmethod
    def fetch_d_tvariable(
        dimension: int,
        aggr_index: IndexingSet,
        t_index: IndexingSet,
        variable: pd.DataFrame,
        row_index: IndexingSet,
        index_map: dict[int, set],
        column_index: IndexingSet,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Fetches a 2D t_variable and returns a dictionary mapping names to 1D Pandas DataFrames.

        Args:

        aggr_index (IndexingSet): aggregate index
        t_index (IndexingSet): technology type index
        variable (Var): technology type variable
        row_index (IndexingSet): The indexing set for the variable's rows.
        column_index (IndexingSet): The indexing set for the variable's columns.
        index_map (dict): technology types for a given aggregate.

        Returns:
            dict[str, dict[str, pd.DataFrame]]: A dictionary mapping aggr names, t_names into 1D Pandas DataFrames.
        """
        result_dict: dict[str, dict[str, pd.DataFrame]] = dict()
        for aggr_idx, aggr_name in aggr_index.mapping.items():
            result_dict[aggr_name] = dict()
            for t_idx in index_map[aggr_idx]:
                t_name = t_index.mapping[t_idx]
                if dimension == 1:
                    df = variable[
                        variable.index.isin(
                            [(aggr_idx, t_idx, row) for row in row_index.ord]
                        )
                    ]
                    df.index = [tup[-1] for tup in df.index]
                    df = df.sort_index()
                elif dimension == 2:
                    df = pd.DataFrame(
                        variable[
                            variable.index.isin(
                                [
                                    (aggr_idx, t_idx, row, col)
                                    for row in row_index.ord
                                    for col in column_index.ord
                                ]
                            )
                        ].values.reshape(len(column_index.ord), len(row_index.ord))
                    )
                result_dict[aggr_name][t_name] = df
        return result_dict

    def to_exportable(self) -> ExportableGeneratorsResults:
        return ExportableGeneratorsResults(
            generation=self.dict_of_2d_array_to_pandas(self.gen),
            dump_energy=self.dict_of_2d_array_to_pandas(self.dump),
            capacity=self.dict_of_1d_array_to_pandas(
                self.cap, column_name=GENERATOR_LABEL
            ),
            generation_per_energy_type=self.dict_of_dicts_of_arrays_to_pandas(
                self.gen_et
            ),
            dump_energy_per_energy_type=self.dict_of_dicts_of_arrays_to_pandas(
                self.dump_et
            ),
            global_capex=self.dict_of_1d_array_to_pandas(
                self.global_capex, column_name=GENERATOR_LABEL
            ),
            local_capex=self.local_capex,
        )


@dataclass
class StoragesResults(ResultsGroup):
    """Storages results"""

    variable_group: InitVar[StorageVariables]
    """Initial value hint for the StorageVariables object"""
    tvariable_group: InitVar[StorageTypeVariables]
    """Initial value hint for the StorageTypeVariables object"""
    tparameters: InitVar[StorageTypeParameters]
    """ Initial value hint for the StorageTypeParameters object """
    parameters: InitVar[StorageParameters]
    """ Initial value hint for the StorageParameters object """
    bus_parameters: InitVar[BusParameters]
    """ Initial value hint for the BusParameters object """
    scenario_parameters: InitVar[ScenarioParameters]
    """ Initial value hint for the ScenarioParameters object """
    indices: InitVar[Indices]
    """Initial value hint for the Indices object"""

    gen: dict[str, pd.DataFrame] = field(init=False)
    """ generation (exportable) """
    gen_dch: dict[str, dict[str, pd.DataFrame]] = field(init=False)
    """ generation for demand chunks (non-exportable) """
    load: dict[str, pd.DataFrame] = field(init=False)
    """ load (exportable) """
    soc: dict[str, pd.DataFrame] = field(init=False)
    """ state of charge (exportable) """
    cap: dict[str, pd.DataFrame] = field(init=False)
    """ capacity (exportable) """
    cap_plus: dict[str, pd.DataFrame] = field(init=False)
    """ capacity increase (non-exportable) """
    cap_minus: dict[str, pd.DataFrame] = field(init=False)
    """ capacity decrease (non-exportable) """
    cap_base_minus: dict[str, pd.DataFrame] = field(init=False)
    """ base capacity decrease (non-exportable) """
    global_capex: dict[str, pd.DataFrame] = field(init=False)
    """ capex of global technologies (exportable) """
    local_capex: dict[str, pd.DataFrame] = field(init=False)
    """ capex of local (in lbs) technologies (exportable) """

    def __post_init__(
        self,
        variable_group: StorageVariables,
        tvariable_group: StorageTypeVariables,
        tparameters: StorageTypeParameters,
        parameters: StorageParameters,
        bus_parameters: BusParameters,
        scenario_parameters: ScenarioParameters,
        indices: Indices,
    ) -> None:
        self.gen = {
            stor_name: df.reset_index(["stor"], drop=True)
            .unstack()
            .droplevel(0, axis=1)
            for stor_name, df in variable_group.gen.solution.to_dataframe().groupby(
                "stor"
            )
        }
        self.gen_dch = {
            stor_name: {
                energy_type: demch_df.reset_index(["stor", "demch"], drop=True)
                .unstack()
                .droplevel(0, axis=1)
                for energy_type, demch_df in stor_df.groupby("stor")
            }
            for stor_name, stor_df in variable_group.gen_dch.solution.to_dataframe().groupby(
                "demch"
            )
        }
        self.load = {
            stor_name: df.reset_index(["stor"], drop=True)
            .unstack()
            .droplevel(0, axis=1)
            for stor_name, df in variable_group.load.solution.to_dataframe().groupby(
                "stor"
            )
        }
        self.soc = {
            stor_name: df.reset_index(["stor"], drop=True)
            .unstack()
            .droplevel(0, axis=1)
            for stor_name, df in variable_group.soc.solution.to_dataframe().groupby(
                "stor"
            )
        }
        self.cap = {
            stor_name: cap_df.reset_index(["stor"], drop=True).rename(
                columns={"solution": "cap"}
            )
            for stor_name, cap_df in variable_group.cap.solution.to_dataframe().groupby(
                "stor"
            )
        }
        self.tcap = tvariable_group.tcap.solution.to_dataframe()
        self.tcap_plus = tvariable_group.tcap_plus.solution.to_dataframe()
        self.cap_plus = variable_group.cap_plus.solution.to_dataframe()
        self.cap_minus = variable_group.cap_minus.solution.to_dataframe()
        self.tcap_minus = tvariable_group.tcap_minus.solution.to_dataframe()
        self.cap_base_minus = variable_group.cap_base_minus.solution.to_dataframe()
        self.tcap_base_minus = tvariable_group.tcap_base_minus.solution.to_dataframe()
        self.global_capex = self.calculate_global_capex(
            indices=indices,
            unit_index=indices.STOR,
            unit_type_param=tparameters,
            unit_type_map=parameters.tstor,
            bus_unit_mapping=bus_parameters.storages,
            aggr_unit_map=indices.aggr_stor_map,
            discount_rate=scenario_parameters.discount_rate,
            cap_plus=variable_group.cap_plus,
            money_scale=scenario_parameters.money_scale,
        )
        self.local_capex = self.calculate_local_capex(
            indices=indices,
            unit_type_param=tparameters,
            unit_type_map=parameters.tstor,
            aggr_unit_map=indices.aggr_stor_map,
            discount_rate=scenario_parameters.discount_rate,
            tcap_plus=tvariable_group.tcap_plus,
            money_scale=scenario_parameters.money_scale,
            gen_mapping=indices.TSTOR.mapping,
        )

    def to_exportable(self) -> ExportableStorageResults:
        return ExportableStorageResults(
            generation=self.dict_of_2d_array_to_pandas(self.gen),
            load=self.dict_of_2d_array_to_pandas(self.load),
            state_of_charge=self.dict_of_2d_array_to_pandas(self.soc),
            capacity=self.dict_of_1d_array_to_pandas(
                self.cap, column_name=STORAGE_LABEL
            ),
            global_capex=self.dict_of_1d_array_to_pandas(
                self.global_capex, column_name=GENERATOR_LABEL
            ),
            local_capex=self.local_capex,
        )


@dataclass
class LinesResults(ResultsGroup):
    """Lines results"""

    variable_group: InitVar[LineVariables]
    """Initial value hint for the LineVariable object"""
    indices: InitVar[Indices]
    """Initial value hint for the Indices object"""

    flow: dict[str, pd.DataFrame] = field(init=False)
    """ optimal line flows (exportable) """

    def __post_init__(self, variable_group: LineVariables, indices: Indices) -> None:
        self.flow = {
            line_name: df.reset_index(["line"], drop=True)
            .unstack()
            .droplevel(0, axis=1)
            for line_name, df in variable_group.flow.solution.to_dataframe().groupby(
                "line"
            )
        }

    def to_exportable(self) -> ExportableLinesResults:
        return ExportableLinesResults(
            flow=self.dict_of_2d_array_to_pandas(
                self.flow, index_name=HOUR_LABEL, column_name=YEAR_LABEL
            )
        )


@dataclass
class FractionsResults(ResultsGroup):
    """Fraction results"""

    variable_group: InitVar[FractionVariables]
    """Initial value hint for the FractionVariables object"""
    indices: InitVar[Indices]
    """Initial value hint for the Indices object"""

    frac: dict[str, dict[str, pd.DataFrame]] = field(init=False)
    """ fraction of local balancing stack in a given aggregated consumer (exportable) """

    def __post_init__(
        self, variable_group: FractionVariables, indices: Indices
    ) -> None:
        self.frac = {
            aggr_name: {
                consumer_name: df.reset_index(["aggr", "lbs"], drop=True).rename(
                    columns={"solution": "frac"}
                )
                for consumer_name, df in aggr_df.groupby("lbs")
            }
            for aggr_name, aggr_df in variable_group.fraction.solution.to_dataframe().groupby(
                "aggr"
            )
        }

    def to_exportable(self) -> ExportableFractionsResults:
        return ExportableFractionsResults(
            fraction={
                aggr: self.dict_of_1d_array_to_pandas(
                    self.frac[aggr], index_name=YEAR_LABEL, column_name=LBS_LABEL
                )
                for aggr in self.frac
            }
        )


@dataclass
class BusResults(ResultsGroup):
    """Bus results"""

    variable_group: InitVar[BusVariables]
    """Initial value hint for the GeneratorVariables object"""
    indices: InitVar[Indices]
    """Initial value hint for the Indices object"""

    bus_ens: dict[str, pd.DataFrame] = field(init=False)
    """ ens generator per bus """

    def __post_init__(self, variable_group: BusVariables, indices: Indices) -> None:
        self.bus_ens = {
            bus_name: df.reset_index(["bus"], drop=True).unstack().droplevel(0, axis=1)
            for bus_name, df in variable_group.bus_ens.solution.to_dataframe().groupby(
                "bus"
            )
        }
        self.shift_plus = {
            bus_name: df.reset_index(["bus"], drop=True).unstack().droplevel(0, axis=1)
            for bus_name, df in variable_group.shift_plus.solution.to_dataframe().groupby(
                "bus"
            )
        }
        self.shift_minus = {
            bus_name: df.reset_index(["bus"], drop=True).unstack().droplevel(0, axis=1)
            for bus_name, df in variable_group.shift_minus.solution.to_dataframe().groupby(
                "bus"
            )
        }

    def to_exportable(self) -> ExportableBusResults:
        return ExportableBusResults(
            generation_ens=self.dict_of_2d_array_to_pandas(self.bus_ens),
            shift_plus=self.dict_of_2d_array_to_pandas(self.shift_plus),
            shift_minus=self.dict_of_2d_array_to_pandas(self.shift_minus),
        )


@dataclass
class Results:
    """Results of an optimization."""

    variables: InitVar[OptimizationVariables]
    """Initial value hint for the OptimizationVariables object"""
    indices: InitVar[Indices]
    """Initial value hint for the Indices object"""
    parameters: InitVar[OptimizationParameters]
    """ Initial value hint for the OptimizationVariables object """

    objective_value: float
    """ optimal objective function value (exportable) """
    generators_results: GeneratorsResults = field(init=False)
    """ generators variables optimal values (exportable) """
    storages_results: StoragesResults = field(init=False)
    """ storage variables optimal values (exportable) """
    lines_results: LinesResults = field(init=False)
    """ lines variables optimal values (exportable) """
    fractions_results: FractionsResults = field(init=False)
    """ fraction variables optimal values (exportable) """
    bus_results: BusResults = field(init=False)
    """ bus variables optimal values (exportable) """

    def to_exportable(self) -> ExportableResults:
        return ExportableResults(
            objective_value=pd.Series(
                self.objective_value, name="Objective_func_value"
            ),
            generators_results=self.generators_results.to_exportable(),
            storages_results=self.storages_results.to_exportable(),
            lines_results=self.lines_results.to_exportable(),
            fractions_results=self.fractions_results.to_exportable(),
            bus_results=self.bus_results.to_exportable(),
        )

    def __post_init__(
        self,
        variables: OptimizationVariables,
        indices: Indices,
        parameters: OptimizationParameters,
    ) -> None:
        self.generators_results = GeneratorsResults(
            variable_group=variables.gen,
            tvariable_group=variables.tgen,
            indices=indices,
            parameters=parameters.gen,
            tparameters=parameters.tgen,
            bus_parameters=parameters.bus,
            scenario_parameters=parameters.scenario_parameters,
        )
        self.storages_results = StoragesResults(
            variable_group=variables.stor,
            tvariable_group=variables.tstor,
            indices=indices,
            parameters=parameters.stor,
            tparameters=parameters.tstor,
            bus_parameters=parameters.bus,
            scenario_parameters=parameters.scenario_parameters,
        )
        self.lines_results = LinesResults(
            variable_group=variables.line, indices=indices
        )
        self.fractions_results = FractionsResults(
            variable_group=variables.frac, indices=indices
        )
        self.bus_results = BusResults(variable_group=variables.bus, indices=indices)
