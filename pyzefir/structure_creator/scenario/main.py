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

from pathlib import Path

import numpy as np
import pandas as pd

from pyzefir.structure_creator.input_data import ScenarioData
from pyzefir.structure_creator.scenario.constants_enums import (
    ScenarioSheetName,
    ScenarioSheetsColumnName,
)
from pyzefir.structure_creator.utils import (
    get_lbs_name,
    interpolate_missing_df_values,
    write_to_excel,
)


def create_interpolated_attribute_dataframe(
    df: pd.DataFrame, index_name: str, n_years: int
) -> pd.DataFrame:
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
    def interpolate_and_transform(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        interpolated_df = (
            interpolate_missing_df_values(
                df.set_index(ScenarioSheetsColumnName.TECHNOLOGY_TYPE).T,
                expected_idx=np.arange(n_years),
            )
            .T.stack()
            .reset_index(level=1, drop=True)
            .reset_index()
            .rename(columns={0: column_name})
        )
        return interpolated_df

    capex_column = interpolate_and_transform(
        cost_parameters[ScenarioSheetsColumnName.CAPEX],
        ScenarioSheetsColumnName.CAPEX,
    )
    opex_column = interpolate_and_transform(
        cost_parameters[ScenarioSheetsColumnName.OPEX], ScenarioSheetsColumnName.OPEX
    )

    result = pd.concat([capex_column, opex_column], axis=1)
    result[ScenarioSheetsColumnName.YEAR_IDX] = result.index % n_years
    result_columns_order = [
        ScenarioSheetsColumnName.YEAR_IDX,
        ScenarioSheetsColumnName.TECHNOLOGY_TYPE,
        ScenarioSheetsColumnName.CAPEX,
        ScenarioSheetsColumnName.OPEX,
    ]

    return result.loc[:, ~result.columns.duplicated()][result_columns_order]


def create_yearly_demand_df(
    yearly_demand: dict[str, pd.DataFrame], n_years: int
) -> pd.DataFrame:
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
    result = (
        interpolate_missing_df_values(
            values=emission_fees_df.set_index(ScenarioSheetsColumnName.EMISSION_FEE).T,
            expected_idx=np.arange(n_years),
        )
        .reset_index()
        .rename(columns={"index": ScenarioSheetsColumnName.YEAR_IDX})
    )
    return result


def create_scenario_data_dict(
    scenario_data: ScenarioData,
    n_years: int,
    n_hours: int,
) -> dict[ScenarioSheetName, pd.DataFrame]:
    return {
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
    }


def create_scenario(
    scenario_data: ScenarioData,
    output_path: Path,
    scenario_name: str,
    n_years: int,
    n_hours: int,
) -> None:
    scenario_data_dict = create_scenario_data_dict(
        scenario_data=scenario_data,
        n_years=n_years,
        n_hours=n_hours,
    )
    write_to_excel(
        data=scenario_data_dict,
        output_path=output_path,
        filename=f"{scenario_name}.xlsx",
    )
