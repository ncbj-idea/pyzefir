import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pyzefir.structure_creator.data_loader.input_data import ScenarioData
from pyzefir.structure_creator.excel_writer import write_to_excel
from pyzefir.structure_creator.scenario.constants_enums import (
    ScenarioSheetName,
    ScenarioSheetsColumnName,
)
from pyzefir.structure_creator.scenario.utils import (
    get_lbs_name,
    interpolate_missing_df_values,
)

_logger = logging.getLogger(__name__)


def create_interpolated_attribute_dataframe(
    df: pd.DataFrame, index_name: str, n_years: int
) -> pd.DataFrame:
    """
    Create a DataFrame with interpolated attribute values over the specified number of years.

    This function takes an input DataFrame, interpolates missing values for the specified number
    of years, and returns a DataFrame where the attributes have been interpolated across the time range.

    Args:
        - df (pd.DataFrame): DataFrame containing the original values to interpolate from.
        - index_name (str): Name of the column to use as the index.
        - n_years (int): Number of years to interpolate over.

    Returns:
        - pd.DataFrame: A DataFrame with interpolated values and a new index.
    """
    interpolated_values = interpolate_missing_df_values(
        df.iloc[:, 1:].T, expected_idx=np.arange(n_years)
    ).T
    result = (
        pd.concat([interpolated_values, df.iloc[:, 0]], axis=1).set_index(index_name).T
    )
    return result.reset_index().rename(
        columns={"index": ScenarioSheetsColumnName.YEAR_IDX}
    )


def create_cost_parameters_df(
    cost_parameters: dict[str, pd.DataFrame], n_years: int
) -> pd.DataFrame:
    """
    Create a DataFrame for cost parameters with interpolated values if any data is missing.

    This function takes a dictionary of cost parameters (CAPEX and OPEX), interpolates missing
    values for the given number of years, and returns a consolidated DataFrame with the
    interpolated CAPEX and OPEX values for each technology type.

    Args:
        - cost_parameters (dict[str, pd.DataFrame]): Dictionary containing CAPEX and OPEX cost parameters
            for each technology type.
        - n_years (int): Number of years to interpolate over.

    Returns:
        - pd.DataFrame: DataFrame with interpolated CAPEX and OPEX values, indexed by technology type and year.
    """

    def interpolate_and_transform(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """Interpolates missing values and formats the DataFrame."""

        interpolated = interpolate_missing_df_values(
            df.set_index(ScenarioSheetsColumnName.TECHNOLOGY_TYPE).T,
            expected_idx=np.arange(n_years),
        ).T

        interpolated = interpolated.stack().reset_index()
        interpolated.columns = [
            ScenarioSheetsColumnName.TECHNOLOGY_TYPE,
            ScenarioSheetsColumnName.YEAR_IDX,
            column_name,
        ]

        return interpolated[
            [
                ScenarioSheetsColumnName.YEAR_IDX,
                ScenarioSheetsColumnName.TECHNOLOGY_TYPE,
                column_name,
            ]
        ]

    capex_column = interpolate_and_transform(
        cost_parameters[ScenarioSheetsColumnName.CAPEX],
        ScenarioSheetsColumnName.CAPEX,
    )
    opex_column = interpolate_and_transform(
        cost_parameters[ScenarioSheetsColumnName.OPEX], ScenarioSheetsColumnName.OPEX
    )
    return pd.merge(
        capex_column,
        opex_column,
        on=[
            ScenarioSheetsColumnName.YEAR_IDX,
            ScenarioSheetsColumnName.TECHNOLOGY_TYPE,
        ],
        how="outer",
    )


def create_yearly_demand_df(
    yearly_demand: dict[str, pd.DataFrame], n_years: int
) -> pd.DataFrame:
    """
    Create a DataFrame with yearly demand values for each energy type, interpolating missing values if needed.

    This function processes yearly demand data for various energy types, interpolates missing values for
    the given number of years, and aggregates the demand values into a single DataFrame, indexed by
    energy type, year, and aggregate.

    Args:
        - yearly_demand (dict[str, pd.DataFrame]): Dictionary containing yearly demand data for each energy type.
        - n_years (int): Number of years to interpolate over.

    Returns:
        - pd.DataFrame: DataFrame containing yearly demand values, organized by energy type and year.
    """
    dfs = []
    for energy_type, df in yearly_demand.items():
        interpolated_demand = interpolate_missing_df_values(
            df.iloc[:, 1:].T, expected_idx=np.arange(n_years)
        )
        aggregated_demand = (
            pd.concat([df.iloc[:, 0], interpolated_demand.T], axis=1)
            .set_index(ScenarioSheetsColumnName.AGGREGATE)
            .stack()
            .reset_index(level=1)
            .rename(
                columns={
                    "level_1": ScenarioSheetsColumnName.YEAR_IDX,
                    0: ScenarioSheetsColumnName.VALUE,
                }
            )
        )
        dfs.append(
            aggregated_demand.assign(energy_type=energy_type)[
                [
                    ScenarioSheetsColumnName.ENERGY_TYPE,
                    ScenarioSheetsColumnName.YEAR_IDX,
                    ScenarioSheetsColumnName.VALUE,
                ]
            ].reset_index()
        )

    return pd.concat(dfs)


def create_capacity_limits_df(
    dfs: dict[str, pd.DataFrame], column_name: str, n_years: int
) -> pd.DataFrame:
    """
    Create a DataFrame for capacity limits with interpolated values for the given number of years.

    This function takes a dictionary of capacity limit DataFrames, interpolates any missing values
    for each capacity parameter, and aggregates the data into a single DataFrame. The final DataFrame
    includes columns for various capacity metrics (e.g., max/min capacity, capacity increases) and is
    indexed by year and the specified column name.

    Args:
        - dfs (dict[str, pd.DataFrame]): Dictionary containing capacity limit data for different parameters.
        - column_name (str): The name of the column to index the data by (e.g., technology or region).
        - n_years (int): Number of years for which the data should be interpolated.

    Returns:
        - pd.DataFrame: DataFrame containing interpolated capacity limits, organized by year and the given column.
    """
    dfs_list = [
        interpolate_missing_df_values(
            df.set_index(column_name).T, np.arange(1, n_years)
        )
        .T.stack()
        .rename(param)
        for param, df in dfs.items()
    ]
    result = (
        pd.concat(dfs_list, axis=1)
        .reset_index()
        .rename(columns={"level_1": ScenarioSheetsColumnName.YEAR_IDX})
    )
    result_columns_order = [
        ScenarioSheetsColumnName.YEAR_IDX,
        column_name,
        ScenarioSheetsColumnName.MAX_CAPACITY,
        ScenarioSheetsColumnName.MIN_CAPACITY,
        ScenarioSheetsColumnName.MAX_CAPACITY_INCREASE,
        ScenarioSheetsColumnName.MIN_CAPACITY_INCREASE,
    ]

    return result[result_columns_order]


def create_fractions_df(fractions: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Create a DataFrame for technology fractions with interpolated values.

    This function processes a dictionary of fractions, where each fraction represents data associated
    with a specific type of technology (or other classifications). It interpolates missing values in
    the DataFrames, reshapes the data, and assigns a corresponding technology stack name. The final
    output is a unified DataFrame containing key fraction metrics for each technology stack over time.

    Args:
        - fractions (dict[str, dict[str, pd.DataFrame]]): Dictionary containing fraction data,
            categorized by technology type.

    Returns:
        - pd.DataFrame: DataFrame containing interpolated fraction data, organized by technology stack and year.
    """
    result = []

    for lbs_type, data in fractions.items():
        df_list = [
            parameter_df.set_index(ScenarioSheetsColumnName.AGGREGATE)
            .T.interpolate(method="linear", axis=0)
            .stack()
            .rename(parameter_name)
            for parameter_name, parameter_df in data.items()
        ]

        df = (
            pd.concat(df_list, axis=1)
            .reset_index()
            .rename(columns={"level_0": ScenarioSheetsColumnName.YEAR})
        )
        df[ScenarioSheetsColumnName.TECHNOLOGY_STACK] = df[
            ScenarioSheetsColumnName.AGGREGATE
        ].apply(lambda aggr_name: get_lbs_name(lbs_type=lbs_type, aggr_name=aggr_name))
        result.append(df)

    result_columns_order = [
        ScenarioSheetsColumnName.TECHNOLOGY_STACK,
        ScenarioSheetsColumnName.AGGREGATE,
        ScenarioSheetsColumnName.YEAR,
        ScenarioSheetsColumnName.MIN_FRACTION,
        ScenarioSheetsColumnName.MAX_FRACTION,
        ScenarioSheetsColumnName.MAX_FRACTION_INCREASE,
        ScenarioSheetsColumnName.MAX_FRACTION_DECREASE,
    ]
    return pd.concat(result, axis=0)[result_columns_order]


def create_constants_df(n_hours: int, n_years: int) -> pd.DataFrame:
    """
    Create a DataFrame containing constant values for the model.

    This function generates a DataFrame with predefined constant values
    that are used throughout the model. Specifically, it includes the
    total number of hours and the total number of years, providing a
    convenient reference for these values.

    Args:
        - n_hours (int): Total number of hours in the modeling period.
        - n_years (int): Total number of years in the modeling period.

    Returns:
        - pd.DataFrame: DataFrame containing constants with their names and values.
    """
    data = [
        {
            ScenarioSheetsColumnName.CONSTANTS_NAME: "N_HOURS",
            ScenarioSheetsColumnName.CONSTANTS_VALUE: n_hours,
        },
        {
            ScenarioSheetsColumnName.CONSTANTS_NAME: "N_YEARS",
            ScenarioSheetsColumnName.CONSTANTS_VALUE: n_years,
        },
    ]
    return pd.DataFrame(data)


def create_relative_emission_limits_df(
    emission_types: list[str], n_years: int
) -> pd.DataFrame:
    """
    Create a DataFrame for relative emission limits based on specified emission types.

    This function initializes a DataFrame that contains columns for each type of
    emission specified in the input list. The DataFrame is indexed by year, allowing
    for easy management and analysis of emission limits over the given time period.

    Args:
        - emission_types (list[str]): List of emission types to be included as columns.
        - n_years (int): Total number of years for which emission limits are defined.

    Returns:
        - pd.DataFrame: DataFrame with relative emission limits indexed by year.
    """
    df = pd.DataFrame(
        index=pd.RangeIndex(
            start=0, stop=n_years, name=ScenarioSheetsColumnName.YEAR_IDX
        ),
        columns=emission_types,
    )
    return df.reset_index()


def create_emission_fees_df(
    emission_fees_df: pd.DataFrame, n_years: int
) -> pd.DataFrame:
    """
    Create a DataFrame for emission fees, interpolating missing values if necessary.

    This function processes the provided emission fees DataFrame by interpolating any
    missing values across the specified number of years. The resulting DataFrame is
    structured to facilitate further analysis and modeling of emission fees over time.

    Args:
        - emission_fees_df (pd.DataFrame): The initial DataFrame containing emission fees data.
        - n_years (int): The total number of years for which the emission fees are defined.

    Returns:
        - pd.DataFrame: A DataFrame with interpolated emission fees, indexed by year.
    """
    result = (
        interpolate_missing_df_values(
            values=emission_fees_df.set_index(ScenarioSheetsColumnName.EMISSION_FEE).T,
            expected_idx=np.arange(n_years),
        )
        .reset_index()
        .rename(columns={"index": ScenarioSheetsColumnName.YEAR_IDX})
    )
    return result


def create_generation_compensation_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame for generation compensation by restructuring the provided data.

    This function transforms the input generation compensation DataFrame by
    renaming the technology type column to serve as the year index. The data is
    transposed to arrange it correctly for further analysis, and the first row
    is used as the new header.

    Args:
        - df (pd.DataFrame): The initial DataFrame containing generation compensation data.

    Returns:
        - pd.DataFrame: A restructured DataFrame with generation compensation values
          indexed by year and columns representing various compensation metrics.
    """
    df = df.rename(
        columns={
            ScenarioSheetsColumnName.TECHNOLOGY_TYPE: ScenarioSheetsColumnName.YEAR_IDX
        }
    ).T.reset_index()
    return df.set_axis(df.iloc[0], axis=1).drop(df.index[0])


def create_scenario_data_dict(
    scenario_data: ScenarioData,
    capacity_bounds_df: pd.DataFrame,
    n_years: int,
    n_hours: int,
) -> dict[ScenarioSheetName, pd.DataFrame]:
    """
    Create a dictionary of scenario data and associated DataFrames for analysis.

    This function aggregates various pieces of scenario data into a structured
    dictionary format. It includes capacity bounds, cost parameters, demand
    projections, emission limits, fuel parameters, and other relevant metrics
    for the given scenario. Each entry in the dictionary corresponds to a
    specific aspect of the scenario data.

    Args:
        - scenario_data (ScenarioData): An instance of the ScenarioData class
          containing input data for the scenario.
        - capacity_bounds_df (pd.DataFrame): A DataFrame specifying the capacity
          bounds for the scenario.
        - n_years (int): The total number of years for which the scenario is
          being analyzed.
        - n_hours (int): The total number of hours for which the scenario is
          being analyzed.

    Returns:
        - dict[ScenarioSheetName, pd.DataFrame]: A dictionary where keys are
          ScenarioSheetName enumerations and values are DataFrames containing the
          relevant scenario data.
    """
    dfs_dict = {
        ScenarioSheetName.COST_PARAMETERS: create_cost_parameters_df(
            cost_parameters=scenario_data.cost_parameters, n_years=n_years
        ),
        ScenarioSheetName.YEARLY_ENERGY_USAGE: create_yearly_demand_df(
            yearly_demand=scenario_data.yearly_demand, n_years=n_years
        ),
        ScenarioSheetName.N_CONSUMERS: create_interpolated_attribute_dataframe(
            df=scenario_data.n_consumers,
            index_name=ScenarioSheetsColumnName.AGGREGATE,
            n_years=n_years,
        ),
        ScenarioSheetName.RELATIVE_EMISSION_LIMITS: create_interpolated_attribute_dataframe(
            df=scenario_data.relative_emission_limits,
            index_name=ScenarioSheetsColumnName.EMISSION_TYPE,
            n_years=n_years,
        ),
        ScenarioSheetName.FUEL_PRICES: create_interpolated_attribute_dataframe(
            df=scenario_data.fuel_parameters[ScenarioSheetsColumnName.FUEL_PRICE],
            index_name=ScenarioSheetsColumnName.FUEL,
            n_years=n_years,
        ),
        ScenarioSheetName.FUEL_AVAILABILITY: create_interpolated_attribute_dataframe(
            df=scenario_data.fuel_parameters[
                ScenarioSheetsColumnName.FUEL_AVAILABILITY
            ],
            index_name=ScenarioSheetsColumnName.FUEL,
            n_years=n_years,
        ),
        ScenarioSheetName.ELEMENT_ENERGY_EVOLUTION_LIMITS: create_capacity_limits_df(
            dfs=scenario_data.technology_cap_limits,
            column_name=ScenarioSheetsColumnName.TECHNOLOGY_NAME,
            n_years=n_years,
        ),
        ScenarioSheetName.ENERGY_SOURCE_EVOLUTION_LIMITS: create_capacity_limits_df(
            dfs=scenario_data.technology_type_cap_limits,
            column_name=ScenarioSheetsColumnName.TECHNOLOGY_TYPE,
            n_years=n_years,
        ),
        ScenarioSheetName.FRACTIONS: create_fractions_df(
            fractions=scenario_data.fractions
        ),
        ScenarioSheetName.CONSTANTS: create_constants_df(
            n_hours=n_hours, n_years=n_years
        ),
        ScenarioSheetName.EMISSION_FEES: create_emission_fees_df(
            emission_fees_df=scenario_data.cost_parameters[
                ScenarioSheetsColumnName.EMISSION_FEES
            ],
            n_years=n_years,
        ),
        ScenarioSheetName.GENERATION_FRACTION: scenario_data.generation_fraction,
        ScenarioSheetName.CURTAILMENT_COST: create_interpolated_attribute_dataframe(
            df=scenario_data.cost_parameters[ScenarioSheetsColumnName.CURTAILMENT],
            index_name=ScenarioSheetsColumnName.TECHNOLOGY_TYPE,
            n_years=n_years,
        ),
        ScenarioSheetName.GENERATION_COMPENSATION: create_generation_compensation_df(
            scenario_data.generation_compensation
        ),
        ScenarioSheetName.ENS_PENALIZATION: scenario_data.ens_penalization,
    }
    if not scenario_data.yearly_emission_reduction.empty:
        dfs_dict[ScenarioSheetName.YEARLY_EMISSION_REDUCTION] = (
            scenario_data.yearly_emission_reduction
        )
    if not capacity_bounds_df.empty:
        dfs_dict[ScenarioSheetName.CAPACITY_BOUNDS] = capacity_bounds_df
    return dfs_dict


def create_scenario(
    scenario_data: ScenarioData,
    output_path: Path,
    capacity_bounds_df: pd.DataFrame,
    scenario_name: str,
    n_years: int,
    n_hours: int,
) -> None:
    """
    Create a scenario by processing specified data and saving it to an Excel file.

    This function compiles various elements of scenario data into a structured
    format, organizes them into a dictionary, and then writes that data to an
    Excel file. This is useful for analyzing and sharing scenario-based
    projections in an easily accessible format.

    Args:
        - scenario_data (ScenarioData): An instance of the ScenarioData class
          containing the input data for the scenario.
        - output_path (Path): The directory path where the output Excel file
          will be saved.
        - capacity_bounds_df (pd.DataFrame): A DataFrame containing capacity
          bounds relevant to the scenario.
        - scenario_name (str): A descriptive name for the scenario, used as
          the filename for the saved Excel file.
        - n_years (int): The total number of years over which the scenario is
          analyzed.
        - n_hours (int): The total number of hours considered in the scenario
          analysis.
    """
    _logger.debug("Creating scenario data objects ...")
    scenario_data_dict = create_scenario_data_dict(
        scenario_data=scenario_data,
        capacity_bounds_df=capacity_bounds_df,
        n_years=n_years,
        n_hours=n_hours,
    )
    _logger.debug("Saving %s.xlsx ...", scenario_name)
    write_to_excel(
        data=scenario_data_dict,
        output_path=output_path,
        filename=f"{scenario_name}.xlsx",
    )
