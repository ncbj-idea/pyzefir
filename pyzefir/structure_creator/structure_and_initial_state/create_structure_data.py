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

import pandas as pd

from pyzefir.structure_creator.structure_and_initial_state.constants_enums import (
    InputFileFieldName,
    StructureSheetName,
    StructureSheetsColumnName,
    StructureTemporaryColumnName,
)


def create_structure_dfs(
    global_buses_df: pd.DataFrame,
    local_buses_df: pd.DataFrame,
    aggregate_types_dict: dict,
    emission_fees_dict: dict,
    unit_emission_fee_map: dict,
    tags_techs_mapping: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate_types_df = (
        pd.DataFrame.from_dict(aggregate_types_dict, orient="index")
        .reset_index()
        .rename(columns={"index": StructureSheetsColumnName.name})
    )[
        [
            StructureSheetsColumnName.name,
            StructureSheetsColumnName.demand_type,
            StructureSheetsColumnName.n_consumers_base,
            StructureSheetsColumnName.average_area,
        ]
    ]

    emission_types_df = pd.DataFrame(
        [
            (
                emission_type,
                emission_values_dict[StructureSheetsColumnName.base_total_emission],
            )
            for emission_type, emission_values_dict in emission_fees_dict.items()
        ],
        columns=[
            StructureSheetsColumnName.name,
            StructureSheetsColumnName.base_total_emission,
        ],
    )

    emission_fees_emission_types_df = pd.DataFrame(
        [
            (emission_type, emission_fees)
            for emission_type, emission_values_dict in emission_fees_dict.items()
            for emission_fees in emission_values_dict[InputFileFieldName.emission_fees]
        ],
        columns=[
            StructureSheetsColumnName.emission_type,
            StructureSheetsColumnName.emission_fee,
        ],
    )

    unit_emission_fee_df = pd.DataFrame(
        [
            (generator_name, ets)
            for generator_name, ets_list in unit_emission_fee_map.items()
            for ets in ets_list
        ],
        columns=[
            StructureSheetsColumnName.generator,
            StructureSheetsColumnName.emission_fee,
        ],
    )

    local_unit_df = local_buses_df[
        [
            StructureSheetsColumnName.technology,
            StructureSheetsColumnName.technology_type,
            StructureTemporaryColumnName.min_capacity,
            StructureTemporaryColumnName.max_capacity,
            StructureSheetsColumnName.bus,
            StructureTemporaryColumnName.unit_class,
        ]
    ].rename(
        {
            StructureSheetsColumnName.technology: StructureSheetsColumnName.name,
            StructureTemporaryColumnName.min_capacity: StructureSheetsColumnName.min_device_nom_power,
            StructureTemporaryColumnName.max_capacity: StructureSheetsColumnName.max_device_nom_power,
        },
        axis=1,
    )

    global_unit_df = global_buses_df[
        [
            StructureSheetsColumnName.technology,
            StructureSheetsColumnName.technology_type,
            StructureTemporaryColumnName.unit_class,
            StructureSheetsColumnName.bus,
        ]
    ].rename(
        {StructureSheetsColumnName.technology: StructureSheetsColumnName.name},
        axis=1,
    )

    unit_df = pd.concat([local_unit_df, global_unit_df])

    for tag_name, techs_list in tags_techs_mapping.items():
        unit_df[tag_name] = unit_df[StructureSheetsColumnName.name].apply(
            lambda technology_name: "YES" if technology_name in techs_list else ""
        )

    return (
        aggregate_types_df,
        emission_types_df,
        emission_fees_emission_types_df,
        unit_emission_fee_df,
        unit_df,
    )


def create_structure_dict(
    global_buses_df: pd.DataFrame,
    local_buses_df: pd.DataFrame,
    energy_type_list: list[str],
    lines_df: pd.DataFrame,
    buses_out_df: pd.DataFrame,
    aggregate_types_dict: dict,
    emission_fees_dict: dict,
    unit_emission_fee_map: dict,
    tags_techs_mapping: dict,
    transmission_fee_cost_df: pd.DataFrame,
    dsr_df: pd.DataFrame,
    power_reserve_df: pd.DataFrame,
) -> dict[StructureSheetName, pd.DataFrame]:
    (
        aggregate_types_df,
        emission_types_df,
        emission_fees_emission_types_df,
        unit_emission_fee_df,
        unit_df,
    ) = create_structure_dfs(
        global_buses_df=global_buses_df,
        local_buses_df=local_buses_df,
        aggregate_types_dict=aggregate_types_dict,
        emission_fees_dict=emission_fees_dict,
        unit_emission_fee_map=unit_emission_fee_map,
        tags_techs_mapping=tags_techs_mapping,
    )

    structure_data = {
        StructureSheetName.ENERGY_TYPES: pd.DataFrame(
            energy_type_list, columns=[StructureSheetsColumnName.name]
        ).drop_duplicates(),
        StructureSheetName.AGGREGATES: aggregate_types_df,
        StructureSheetName.LINES: lines_df,
        StructureSheetName.BUSES: pd.concat(
            [
                global_buses_df[
                    [
                        StructureSheetsColumnName.bus,
                        StructureSheetsColumnName.energy_type,
                        StructureSheetsColumnName.dsr_type,
                    ]
                ],
                local_buses_df[
                    [
                        StructureSheetsColumnName.bus,
                        StructureSheetsColumnName.energy_type,
                        StructureSheetsColumnName.dsr_type,
                    ]
                ],
            ]
        )
        .drop_duplicates(ignore_index=True)
        .rename(
            {StructureSheetsColumnName.bus: StructureSheetsColumnName.name}, axis=1
        ),
        StructureSheetName.TECHNOLOGYSTACK_BUSES: local_buses_df[
            [StructureTemporaryColumnName.lbs, StructureSheetsColumnName.bus]
        ]
        .drop_duplicates(ignore_index=True)
        .rename(
            {
                StructureTemporaryColumnName.lbs: StructureSheetsColumnName.technology_stack
            },
            axis=1,
        ),
        StructureSheetName.TECHNOLOGYSTACK_AGGREGATE: local_buses_df[
            [StructureTemporaryColumnName.lbs, StructureSheetsColumnName.aggregate]
        ]
        .drop_duplicates(ignore_index=True)
        .rename(
            {
                StructureTemporaryColumnName.lbs: StructureSheetsColumnName.technology_stack
            },
            axis=1,
        ),
        StructureSheetName.GENERATORS: unit_df[
            unit_df[StructureTemporaryColumnName.unit_class] == "GENERATOR"
        ][
            [
                StructureSheetsColumnName.name,
                StructureSheetsColumnName.technology_type,
                StructureSheetsColumnName.min_device_nom_power,
                StructureSheetsColumnName.max_device_nom_power,
                *list(tags_techs_mapping.keys()),
            ]
        ]
        .rename(
            {
                StructureSheetsColumnName.technology_type: StructureSheetsColumnName.generator_type
            },
            axis=1,
        )
        .drop_duplicates(ignore_index=True),
        StructureSheetName.STORAGES: unit_df[
            unit_df[StructureTemporaryColumnName.unit_class] == "STORAGE"
        ][
            [
                StructureSheetsColumnName.name,
                StructureSheetsColumnName.technology_type,
                StructureSheetsColumnName.min_device_nom_power,
                StructureSheetsColumnName.max_device_nom_power,
                *list(tags_techs_mapping.keys()),
            ]
        ]
        .rename(
            {
                StructureSheetsColumnName.technology_type: StructureSheetsColumnName.storage_type
            },
            axis=1,
        )
        .drop_duplicates(ignore_index=True),
        StructureSheetName.TECHNOLOGY_BUS: unit_df[
            [
                StructureSheetsColumnName.name,
                StructureTemporaryColumnName.unit_class,
                StructureSheetsColumnName.bus,
            ]
        ]
        .rename(
            {
                StructureSheetsColumnName.name: StructureSheetsColumnName.technology,
                StructureTemporaryColumnName.unit_class: StructureSheetsColumnName.type,
            },
            axis=1,
        )
        .drop_duplicates(ignore_index=True),
        StructureSheetName.TECHNOLOGYSTACKS_BUSES_OUT: buses_out_df.pivot_table(
            values=StructureSheetsColumnName.bus,
            columns=[StructureSheetsColumnName.energy_type],
            aggfunc="first",
            index=StructureTemporaryColumnName.lbs,
        )
        .reset_index()
        .rename(
            {StructureTemporaryColumnName.lbs: StructureSheetsColumnName.name}, axis=1
        ),
        StructureSheetName.EMISSION_TYPES: emission_types_df,
        StructureSheetName.EMISSION_FEES_EMISSION_TYPES: emission_fees_emission_types_df,
        StructureSheetName.GENERATOR_EMISSION_FEES: unit_emission_fee_df,
        StructureSheetName.TRANSMISSION_FEES: transmission_fee_cost_df,
        StructureSheetName.DSR: dsr_df,
        StructureSheetName.POWER_RESERVE: power_reserve_df,
    }
    return structure_data
