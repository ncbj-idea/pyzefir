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
from typing import Final, Literal

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
from pyzefir.optimization.linopy.utils import (
    calculate_storage_adjusted_generation,
    get_generator_types_capacity_multipliers,
)
from pyzefir.utils.functions import get_dict_vals

HOUR_LABEL: Final[str] = "Hour"
YEAR_LABEL: Final[str] = "Year"
GENERATOR_LABEL: Final[str] = "Generator"
LBS_LABEL: Final[str] = "Local Balancing Stack"
STORAGE_LABEL: Final[str] = "Storage"
ENERGY_TYPE_LABEL: Final[str] = "Energy Type"


class ResultsGroup(abc.ABC):
    """
    A base class for fetching and organizing variables used in result groups.

    This class provides methods to retrieve and format results from a set of optimization
    variables into structured Pandas DataFrames. It serves as a foundational structure for
    implementing specific result handling strategies.

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
        """
        Renames the axes of the given DataFrame based on the provided row and column indices.

        Args:
            - df (pd.DataFrame): The DataFrame whose axes are to be renamed.
            - row_index (IndexingSet | None): Optional. The indexing set used for renaming rows.
            - column_index (IndexingSet | None): Optional. The indexing set used for renaming columns.

        Returns:
            - pd.DataFrame: The DataFrame with renamed axes.
        """
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
        """
        Converts a dictionary of 1D Pandas DataFrames into a single Pandas DataFrame.

        Each entry in the dictionary should contain a DataFrame with one column. This method
        concatenates these DataFrames horizontally and returns a single DataFrame with the
        specified column and index names.

        Args:
            - data (dict[str, pd.DataFrame]): A dictionary mapping names to 1D DataFrames.
            - column_name (str): The name to assign to the columns of the resulting DataFrame.
            - index_name (str, optional): The name to assign to the index of the resulting DataFrame.
                                          Defaults to YEAR_LABEL.

        Returns:
            - pd.DataFrame: A single DataFrame containing all input DataFrames concatenated.

        Raises:
            - ValueError: If any of the DataFrames in the dictionary is not 1D.
        """
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
        """
        Converts a dictionary of 2D Pandas DataFrames into a new dictionary with named axes.

        This method ensures that each DataFrame in the dictionary has more than one column,
        assigning the specified index and column names to the resulting DataFrames.

        Args:
            - data (dict[str, pd.DataFrame]): A dictionary mapping names to 2D DataFrames.
            - index_name (str, optional): The name to assign to the index of each resulting DataFrame.
                Defaults to HOUR_LABEL.
            - column_name (str, optional): The name to assign to the columns of each resulting DataFrame.
                Defaults to YEAR_LABEL.

        Returns:
            - dict[str, pd.DataFrame]: A dictionary mapping names to DataFrames with renamed axes.

        Raises:
            - ValueError: If any of the DataFrames in the dictionary does not meet the dimensionality requirement.

        Example:
            >>> processed_data = self.dict_of_2d_array_to_pandas(data_dict)
        """
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
        """
        Converts a nested dictionary structure of DataFrames into a flat dictionary of DataFrames.

        The method concatenates DataFrames within the inner dictionaries and renames the resulting
        DataFrames to include energy type and index names. This allows for structured access to results.

        Args:
            - data (dict[str, dict[str, pd.DataFrame]]): A dictionary mapping outer keys to
                inner dictionaries of DataFrames.
            - index_name (str, optional): The name to assign to the index of each resulting DataFrame.
                Defaults to HOUR_LABEL.
            - column_name (str, optional): The name to assign to the columns of each resulting DataFrame.
                Defaults to YEAR_LABEL.
            - energy_type_label (str, optional): The name to assign to the energy type index.
                Defaults to ENERGY_TYPE_LABEL.

        Returns:
            - dict[str, pd.DataFrame]: A dictionary mapping outer keys to DataFrames with renamed axes.
        """
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
        """
        Calculates the global capital expenditure (CAPEX) per unit per year.

        This method computes the amortized global CAPEX over the specified lifespan of a unit,
        applying the provided discount rate and capital cost. The results are summed over the
        specified years.

        Args:
            - capex (np.ndarray): An array of capital expenditures for each year.
            - cap_plus (xr.DataArray): An array containing additional capital costs per unit.
            - disc_rate (np.ndarray): An array of discount rates for each year.
            - lt (int): The lifespan of the unit in years.
            - s_idx (int): The index of the specific year being calculated.
            - u_idx (int): The index of the unit for which CAPEX is being calculated.
            - y_idxs (IndexingSet): The set of indices representing years.

        Returns:
            - float: The total global CAPEX per unit per year for the specified unit.
        """
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
        """
        Calculates the local capital expenditure (CAPEX) per unit per year.

        Similar to the global CAPEX calculation, this method computes the amortized local CAPEX
        for a given unit type over its lifespan, considering local costs and discount rates.

        Args:
            - capex (np.ndarray): An array of capital expenditures for each year.
            - tcap_plus (xr.DataArray): An array containing additional local capital costs per unit.
            - disc_rate (np.ndarray): An array of discount rates for each year.
            - lt (int): The lifespan of the unit in years.
            - s_idx (int): The index of the specific year being calculated.
            - ut_idx (int): The index of the unit type for which CAPEX is being calculated.
            - aggr_idx (int): The index of the aggregated unit being considered.
            - y_idxs (IndexingSet): The set of indices representing years.

        Returns:
            - float: The total local CAPEX per unit per year for the specified unit type.
        """
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
        """
        Calculates the global capital expenditure (CAPEX) for multiple units over a set of years.

        This method computes CAPEX for non-aggregated units based on their type, applying the
        respective discount rates, multipliers, and capital costs. Results are structured
        into a dictionary of Pandas DataFrames.

        Args:
            - discount_rate (np.ndarray): An array of discount rates for each year.
            - bus_unit_mapping (dict[int, set[int]]): A mapping of bus indices to corresponding unit indices.
            - unit_index (IndexingSet): The indexing set representing unit indices.
            - aggr_unit_map (dict[int, set[int]]): A mapping of aggregated units to their constituent units.
            - indices (Indices): An object containing various indices including years.
            - unit_type_map (dict[int, int]): A mapping from unit indices to their type indices.
            - unit_type_param (GeneratorTypeParameters | StorageTypeParameters): Parameters related to unit types.
            - money_scale (float): A scaling factor for the monetary values.
            - cap_plus (Variable): A variable representing additional capital costs.
            - multipliers (dict[int, float] | None): Optional. A mapping of unit indices to their multipliers.

        Returns:
            - dict[str, pd.DataFrame]: A dictionary mapping unit names to DataFrames representing
                                      the annual CAPEX for each unit.
        """
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
        """
        Calculates the local capital expenditure (CAPEX) for aggregated units over a set of years.

        This method computes CAPEX for aggregated unit types, applying the respective discount rates,
        multipliers, and capital costs. The results are organized into a dictionary of DataFrames.

        Args:
            - discount_rate (np.ndarray): An array of discount rates for each year.
            - tcap_plus (Variable): A variable representing additional local capital costs.
            - indices (Indices): An object containing various indices including years.
            - unit_type_param (GeneratorTypeParameters | StorageTypeParameters): Parameters related to unit types.
            - money_scale (float): A scaling factor for the monetary values.
            - unit_type_map (dict[int, int]): A mapping from unit indices to their type indices.
            - aggr_unit_map (dict[int, set[int]]): A mapping of aggregated units to their constituent units.
            - gen_mapping (bidict): A bidirectional mapping of unit indices to their corresponding generator names.
            - multipliers (dict[int, float] | None): Optional. A mapping of unit indices to their multipliers.

        Returns:
            - dict[str, pd.DataFrame]: A dictionary mapping aggregated unit names to DataFrames representing
                                      the annual CAPEX for each aggregated unit.
        """
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
    """
    A class for processing and organizing generator-related results in energy models.

    This class retrieves and formats results from various generator variables, including generation,
    capacity, and capital expenditure (CAPEX) data. It provides methods to process and structure these
    results into organized Pandas DataFrames, facilitating analysis and reporting of generator performance
    within energy system models.
    """

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
    gen_dch: dict[str, dict[str, pd.DataFrame]] = field(init=False)
    """ generation per energy type for demand chunks (non-exportable) """
    gen_reserve_et: dict[str, dict[str, dict[str, pd.DataFrame]]] = field(init=False)
    """ generation for reserves """
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
        self.gen_et = self.process_h_y_var(variable_group.gen_et, indices)
        self.dump_et = self.process_h_y_var(variable_group.dump_et, indices)
        self.gen_dch = process_gen_dch(variable_group.gen_dch, indices, "gen")
        self.gen_reserve_et = process_gen_reserve_et(
            variable_group.gen_reserve_et,
            indices,
        )
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
        """
        Processes generation data from the variable group into a dictionary of DataFrames.

        Args:
            - variable_group (GeneratorVariables): The object containing generator variables.

        Returns:
            - dict[str, pd.DataFrame]: A dictionary mapping generator names to their respective generation DataFrames.
        """
        return {
            gen_name: df.reset_index(["gen"], drop=True).unstack().droplevel(0, axis=1)
            for gen_name, df in variable_group.gen.solution.to_dataframe().groupby(
                "gen"
            )
        }

    @staticmethod
    def process_h_y_var(
        var: dict[int, dict[str, Variable]],
        indices: Indices,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Processes generation data categorized by energy types into a structured format.

        Args:
            - var (dict[int, dict[str, Variable]]): A dictionary of variables indexed by generator index,
              with energy types as keys.
            - indices (Indices): The object containing indexing information for mapping.

        Returns:
            - dict[str, dict[str, pd.DataFrame]]: A nested dictionary where each generator name maps to
              another dictionary mapping energy types to DataFrames.
        """
        result: dict[str, dict[str, pd.DataFrame]] = dict()
        for gen_idx, data in var.items():
            gen_name = indices.GEN.mapping[gen_idx]
            result[gen_name] = dict()
            for energy_type in indices.ET.ii:
                if energy_type in data:
                    gen_et = (
                        data[energy_type]
                        .solution.to_dataframe()
                        .unstack()
                        .droplevel(0, axis=1)
                    )
                else:
                    gen_et = empty_generation_dataframe(indices)
                result[gen_name][energy_type] = gen_et
        return result

    @staticmethod
    def process_cap(variable_group: GeneratorVariables) -> dict[str, pd.DataFrame]:
        """
        Processes capacity data from the variable group into a dictionary of DataFrames.

        Args:
            - variable_group (GeneratorVariables): The object containing generator variables.

        Returns:
            - dict[str, pd.DataFrame]: A dictionary mapping generator names to their respective capacity DataFrames.
        """
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
        """
        Fetches a variable from a DataFrame and returns a dictionary mapping names to Pandas DataFrames.

        Args:
            - dimension (int): The dimension of the variable to fetch (1 or 2).
            - index (IndexingSet): The indexing set for the variable.
            - variable (pd.DataFrame): The DataFrame containing the variable to fetch.
            - row_index (IndexingSet): The indexing set for the variable's rows.
            - column_index (IndexingSet | None): The indexing set for the variable's columns.
            - filter_map (dict[int, set] | set | None): Optional; dict of sets or set indices to filter from
                the index mapping.

        Returns:
            - dict[str, pd.DataFrame]: A dictionary mapping names to Pandas DataFrames containing the fetched
                variable data.
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
        """
        Fetches a technology variable and returns a dictionary mapping names to Pandas DataFrames.

        Args:
            - dimension (int): The dimension of the variable to fetch (1 or 2).
            - aggr_index (IndexingSet): The aggregate index mapping for the variables.
            - t_index (IndexingSet): The technology type index mapping.
            - variable (pd.DataFrame): The DataFrame containing the technology variable data.
            - row_index (IndexingSet): The indexing set for the variable's rows.
            - index_map (dict[int, set]): A dictionary mapping aggregate indices to their corresponding technology
              type indices.
            - column_index (IndexingSet): The indexing set for the variable's columns.

        Returns:
            - dict[str, dict[str, pd.DataFrame]]: A nested dictionary where each aggregate name maps to another
              dictionary mapping technology type names to DataFrames containing the fetched data.
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
        """
        Converts processed generator results into a format suitable for export.

        This method organizes the internal data structures into a coherent format, specifically
        an instance of ExportableGeneratorsResults, for reporting or further analysis.

        Returns:
            - ExportableGeneratorsResults: An object containing organized generator results ready for export.
        """
        return ExportableGeneratorsResults(
            generation=self.dict_of_2d_array_to_pandas(self.gen),
            # dump_energy=self.dict_of_2d_array_to_pandas(self.dump),
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
    """
    A class for fetching and organizing storage results.

    This class processes and organizes various results related to energy storage systems,
    including generation, load, state of charge, capacity, and capital expenditures. It
    structures the data into easily accessible Pandas DataFrames, facilitating further
    analysis and reporting.
    """

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
        self.gen = calculate_storage_adjusted_generation(
            generation_result_df=variable_group.gen.solution.to_dataframe(),
            storages_generation_efficiency=parameters.gen_eff,
            storages_idxs=indices.STOR.mapping,
        )
        self.gen_dch = process_gen_dch(variable_group.gen_dch, indices, "stor")
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
        """
        Converts processed storage results into a format suitable for export.

        This method organizes the internal data structures into a coherent format, specifically
        an instance of ExportableStorageResults, for reporting or further analysis.

        Returns:
            - ExportableStorageResults: An object containing organized storage results ready for export.
        """
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
    """
    A class for fetching and organizing results related to lines.

    This class processes and organizes results associated with line flows in an energy
    system, structuring them into accessible Pandas DataFrames. It serves to facilitate
    analysis and reporting of optimal line flow results.
    """

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
        """
        Converts processed line results into a format suitable for export.

        Returns:
            - ExportableLinesResults: An object containing organized line flow results ready for export.
        """
        return ExportableLinesResults(
            flow=self.dict_of_2d_array_to_pandas(
                self.flow, index_name=HOUR_LABEL, column_name=YEAR_LABEL
            )
        )


@dataclass
class FractionsResults(ResultsGroup):
    """
    A class for fetching and organizing fraction results.

    This class processes and organizes results related to the fraction of local balancing stacks
    in aggregated consumers, structuring them into accessible Pandas DataFrames. It serves to
    facilitate analysis and reporting of fraction results across different consumer categories.
    """

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
        """
        Converts processed fraction results into a format suitable for export.

        Returns:
            - ExportableFractionsResults: An object containing organized fraction results ready for export.
        """
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
    """
    A class for fetching and organizing bus results.

    This class processes and organizes results related to the energy generation and shifts
    at each bus, structuring them into accessible Pandas DataFrames. It serves to facilitate
    analysis and reporting of bus-related results across various components of the energy system.
    """

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
        self.shift_plus = self.process_shift_variable(
            variable_group.shift_plus, indices
        )
        self.shift_minus = self.process_shift_variable(
            variable_group.shift_minus, indices
        )

    @staticmethod
    def process_shift_variable(
        var: dict[int, Variable], indices: Indices
    ) -> dict[str, pd.DataFrame]:
        """
        Processes shift variable data into structured Pandas DataFrames.

        This method collects and organizes shift variable data from the provided input object
        into structured formats for further analysis.

        Args:
            - var (dict[int, Variable]): The dictionary containing shift variable data.
            - indices (Indices): The object containing indexing information.

        Returns:
            - dict[str, pd.DataFrame]: A dictionary mapping bus names to their respective shift DataFrames.
        """
        result: dict[str, pd.DataFrame] = dict()
        for bus_idx in indices.BUS.ord:
            bus_name = indices.BUS.mapping[bus_idx]
            if bus_idx not in var:
                df = empty_generation_dataframe(indices)
            else:
                df = var[bus_idx].solution.to_dataframe().unstack().droplevel(0, axis=1)
            result[bus_name] = df
        return result

    def to_exportable(self) -> ExportableBusResults:
        """
        Converts processed bus results into a format suitable for export.

        Returns:
            - ExportableBusResults: An object containing organized bus results ready for export.
        """
        return ExportableBusResults(
            generation_ens=self.dict_of_2d_array_to_pandas(self.bus_ens),
            shift_plus=self.dict_of_2d_array_to_pandas(self.shift_plus),
            shift_minus=self.dict_of_2d_array_to_pandas(self.shift_minus),
        )


@dataclass
class Results:
    """
    Results of an optimization.

    This class aggregates and organizes the results obtained from an optimization process,
    including optimal values for various components like generators, storages, lines, fractions,
    and buses. It serves as a central repository for results, enabling easy access and export
    for reporting and analysis.
    """

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
        """
        Converts processed optimization results into a format suitable for export.

        Returns:
            - ExportableResults: An object containing organized optimization results ready for export.
        """
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


def process_gen_dch(
    gen_dch_var: dict[int, Variable],
    indices: Indices,
    energy_source_class: Literal["gen", "stor"],
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Processes generation data for demand chunks.

    This function organizes generation data from a set of demand chunk variables,
    grouping them by generation source (either generator or storage) and demand chunk.
    It returns a structured dictionary of Pandas DataFrames for further analysis.

    Args:
        - gen_dch_var (dict[int, Variable]): A dictionary mapping demand chunk indices
          to their corresponding generation variables.
        - indices (Indices): An object containing index mappings for generation and storage.
        - energy_source_class (Literal["gen", "stor"]): A string indicating the type of energy
          source to process. Must be either "gen" for generators or "stor" for storages.

    Returns:
        - dict[str, dict[str, pd.DataFrame]]: A nested dictionary where the outer keys
          represent generation source names, and the inner keys represent demand chunk names.
          Each value is a Pandas DataFrame containing generation data for the corresponding
          source and demand chunk.

    Raises:
        - ValueError: If energy_source_class is not "gen" or "stor".
    """
    if energy_source_class not in ["gen", "stor"]:
        raise ValueError(
            f"energy_source_class {energy_source_class} must be gen or stor"
        )
    energy_source_indices = (
        indices.GEN if energy_source_class == "gen" else indices.STOR
    )
    result: dict[str, dict[str, pd.DataFrame]] = dict()
    for dch_idx in indices.DEMCH.ord:
        dch_name = indices.DEMCH.mapping[dch_idx]
        for gen_idx in energy_source_indices.ord:
            gen_name = energy_source_indices.mapping[gen_idx]
            if gen_name not in result:
                result[gen_name] = dict()
            if gen_idx not in gen_dch_var[dch_idx]:
                dch_gen = empty_generation_dataframe(indices)
            else:
                dch_gen = (
                    gen_dch_var[dch_idx][gen_idx]
                    .solution.to_dataframe()
                    .unstack()
                    .droplevel(0, axis=1)
                )
            result[gen_name][dch_name] = dch_gen
    return result


def process_gen_reserve_et(
    gen_reserve_et_var: dict,
    indices: Indices,
) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
    """
    Processes generation data for reserves.

    This function organizes generation data from a set of generation for reserves.
    It returns a structured dictionary of Pandas DataFrames for further analysis.

    Args:
        - gen_reserve_et (dict[int, Variable]): A dictionary mapping gen_reserve_et indices
          to their corresponding generation variables.
        - indices (Indices): An object containing index mappings for generation and storage.
        - energy_source_class (Literal["gen", "stor"]): A string indicating the type of energy
          source to process. Must be either "gen" for generators or "stor" for storages.

    Returns:
        - dict[str, dict[str, dict[str, pd.DataFrame]]]: A nested dictionary where the outer keys
          represent generation tag names, source names, and the inner keys represent demand chunk names.

    """
    result: dict[str, dict[str, dict[str, pd.DataFrame]]] = dict()
    for tag_idx in gen_reserve_et_var:
        tag_name = indices.TAGS.mapping[tag_idx]
        if tag_name not in result:
            result[tag_name] = dict()
        for gen_idx in gen_reserve_et_var[tag_idx]:
            gen_name = indices.GEN.mapping[gen_idx]
            if gen_name not in result[tag_name]:
                result[tag_name][gen_name] = dict()
                for et, v in gen_reserve_et_var[tag_idx][gen_idx].items():
                    result[tag_name][gen_name][et] = (
                        v.solution.to_dataframe().unstack().droplevel(0, axis=1)
                    )
    return result


def empty_generation_dataframe(indices: Indices) -> pd.DataFrame:
    """
    Creates an empty generation DataFrame with specified dimensions.

    Args:
        - indices (Indices): An object containing index mappings for hours and years.

    Returns:
        - pd.DataFrame: An empty DataFrame with shape (number of hours, number of years),
            where the index represents hours and the columns represent years. The DataFrame
            is filled with zeros, indicating no generation data.
    """
    result = pd.DataFrame(
        data=np.zeros((len(indices.H), len(indices.Y))),
        columns=indices.Y.ii,
        index=indices.H.ii,
    )
    result.columns.name = "year"
    result.index.name = "hour"
    return result
