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

from pyzefir.structure_creator.constants_enums import XlsxFileName
from pyzefir.structure_creator.input_data import StructureData
from pyzefir.structure_creator.structure_and_initial_state.constants_enums import (
    InputFileFieldName,
    InputXlsxColumnName,
    StructureSheetsColumnName,
    StructureTemporaryColumnName,
)
from pyzefir.structure_creator.structure_and_initial_state.create_initial_state_data import (
    create_initial_state_dict,
)
from pyzefir.structure_creator.structure_and_initial_state.create_structure_data import (
    create_structure_dict,
)
from pyzefir.structure_creator.utils import invert_dict, merge_dicts, write_to_excel


def get_element_name(
    aggregate: str,
    lbs: str | None = None,
    energy_type: str | None = None,
    unit_type: str | None = None,
) -> str:
    return "__".join([el for el in [aggregate, lbs, energy_type, unit_type] if el])


def get_lbs_tech_cap_parameters(
    aggregate: str, tech_type: str, cap_range: list[dict[str, pd.DataFrame]]
) -> tuple[float, ...]:
    result: list[float] = []
    for dfs in cap_range:
        aggregate_type_key = next((key for key in dfs if key in aggregate), None)

        if aggregate_type_key is not None:
            aggregate_type = dfs[aggregate_type_key]
            energy_class = aggregate.split(f"{aggregate_type_key}_")[-1]

            tech_mask = (
                aggregate_type[StructureSheetsColumnName.technology_type] == tech_type
            )
            result.append(aggregate_type.loc[tech_mask, energy_class].squeeze())

    return tuple(result)


def get_lbs_tech_config(
    tech_config_dict: dict, aggregate_name: str, lbs_type: str
) -> dict | None:
    candidates = [k for k in tech_config_dict if k in aggregate_name]
    if len(candidates) > 1:
        raise KeyError(
            f"{aggregate_name} type is ambigius in {lbs_type} configuration file"
        )
    if len(candidates) == 0:
        return None
    aggregate_type = candidates[0]
    return tech_config_dict[aggregate_type]


def create_global_data(
    global_tech_df: pd.DataFrame,
    subsystem_config: dict,
    subsystem_configuration: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, dict]:
    global_buses = subsystem_configuration[
        (subsystem_configuration.iloc[:, 1:] == 1).any(axis=1)
    ][InputXlsxColumnName.SUBSYSTEM]
    unit_emission_fee_map = dict()
    buses_info = []
    global_tech_tags_mapping = dict()
    for global_bus in global_buses:
        bus_config = subsystem_config.get(global_bus)
        if bus_config is None:
            continue

        for tech in bus_config[InputFileFieldName.technologies]:
            tech_conf = global_tech_df[tech]
            buses_info.append(
                [
                    bus_config[InputFileFieldName.energy_type],
                    bus_config[InputFileFieldName.subsystem_name],
                    tech,
                    tech_conf[InputFileFieldName.base_cap],
                    tech_conf[InputFileFieldName.type],
                    tech_conf["class"],
                    bus_config[InputFileFieldName.transmission_loss],
                    bus_config.get(InputFileFieldName.dsr_type, None),
                ]
            )

            if isinstance(tech_conf[InputFileFieldName.tags], list):
                for tag in tech_conf[InputFileFieldName.tags]:
                    if tag not in global_tech_tags_mapping:
                        global_tech_tags_mapping[tag] = {tech}
                    else:
                        global_tech_tags_mapping[tag].add(tech)
            if InputFileFieldName.emission_fees in global_tech_df.index and isinstance(
                global_tech_df[tech][InputFileFieldName.emission_fees], list
            ):
                unit_emission_fee_map[tech] = global_tech_df[tech][
                    InputFileFieldName.emission_fees
                ]

    global_buses_df = pd.DataFrame(
        buses_info,
        columns=[
            StructureSheetsColumnName.energy_type,
            StructureSheetsColumnName.bus,
            StructureSheetsColumnName.technology,
            StructureTemporaryColumnName.base_cap,
            StructureSheetsColumnName.technology_type,
            StructureTemporaryColumnName.unit_class,
            StructureSheetsColumnName.transmission_loss,
            StructureSheetsColumnName.dsr_type,
        ],
    )

    return global_buses_df, unit_emission_fee_map, global_tech_tags_mapping


def local_buses_process_energy_types(
    energy_types: list[str],
    lbs_type: str,
    energy_to_bus: dict[str, str],
    buses_list: list[list[str]],
) -> None:
    for energy in energy_types:
        buses_list.append([lbs_type, energy_to_bus[energy], energy])


def local_buses_process_technology(
    aggregate: str,
    lbs: str,
    lbs_type: str,
    tech: str,
    tech_config: dict,
    tech_energy_mapping: dict,
    energy_to_bus: dict[str, str],
    structure_data: StructureData,
    results: list[list[str | float | None]],
    unit_emission_fees_map: dict,
    unit_tech_tags: dict,
    dsr_types: dict | None,
) -> None:
    tech_class = tech_config[InputFileFieldName.TECH_CLASS]
    tech_aggr_config = get_lbs_tech_config(
        tech_config_dict=tech_config,
        aggregate_name=aggregate,
        lbs_type=lbs_type,
    )

    base_capacity, max_capacity, min_capacity = get_lbs_tech_cap_parameters(
        aggregate,
        tech,
        [structure_data.cap_base, structure_data.cap_max, structure_data.cap_min],
    )

    tech_energy_types = tech_energy_mapping.get(tech)
    if tech_energy_types is None:
        raise ValueError(
            f"there is a problem with technology {tech} in lbs {lbs_type} "
            f"- energy_type is {tech_energy_types}"
        )

    tech_name = get_element_name(aggregate=aggregate, lbs=lbs_type, unit_type=tech)
    base_fractions = structure_data.configuration[InputFileFieldName.lbs_type]
    base_n_consumers = structure_data.aggregate_types[aggregate][
        InputFileFieldName.n_consumers_base
    ]
    base_fraction = base_fractions[
        base_fractions[InputXlsxColumnName.AGGREGATE] == aggregate
    ][lbs_type].squeeze()
    base_capacity = base_capacity * base_fraction * base_n_consumers

    if tech_aggr_config is not None and InputFileFieldName.tags in tech_aggr_config:
        for tag in tech_aggr_config[InputFileFieldName.tags]:
            unit_tech_tags.setdefault(tag, set()).add(tech_name)

    results.extend(
        [
            [
                tech_name,
                tech,
                min_capacity,
                max_capacity,
                tech_class,
                energy_to_bus[et],
                lbs,
                lbs_type,
                aggregate,
                et,
                base_capacity,
                dsr_types.get(et) if dsr_types is not None else None,
            ]
            for et in tech_energy_types
            if et in energy_to_bus
        ]
    )

    emission_fees_list = tech_config.get(InputXlsxColumnName.EMISSION_FEES)
    if emission_fees_list is not None:
        unit_emission_fees_map[tech_name] = emission_fees_list


def local_buses_process_lbs_type(
    aggregate: str,
    lbs_type: str,
    lbs_config: dict,
    structure_data: StructureData,
    results: list[list[str | float | None]],
    buses_out: list[list[str]],
    buses_in: list[list[str]],
    unit_emission_fees_map: dict,
    unit_tech_tags: dict,
) -> None:
    lbs_conf = lbs_config.get(lbs_type)
    if lbs_conf is None:
        return

    lbs = get_element_name(aggregate=aggregate, lbs=lbs_type)
    energy_to_bus = {
        energy: get_element_name(aggregate=aggregate, lbs=lbs_type, energy_type=energy)
        for energy in lbs_conf[InputFileFieldName.energy_types_out]
        + lbs_conf[InputFileFieldName.energy_types_in]
    }

    local_buses_process_energy_types(
        lbs_conf[InputFileFieldName.energy_types_out], lbs, energy_to_bus, buses_out
    )
    local_buses_process_energy_types(
        lbs_conf[InputFileFieldName.energy_types_in], lbs_type, energy_to_bus, buses_in
    )

    tech_energy_mapping = invert_dict(lbs_conf[InputFileFieldName.energy_tech_mapping])
    dsr_types = lbs_conf.get(InputFileFieldName.dsr_types)
    for tech, tech_config in lbs_conf[InputFileFieldName.device_capacity_range].items():
        local_buses_process_technology(
            aggregate,
            lbs,
            lbs_type,
            tech,
            tech_config,
            tech_energy_mapping,
            energy_to_bus,
            structure_data,
            results,
            unit_emission_fees_map,
            unit_tech_tags,
            dsr_types,
        )


def local_buses_process_aggregate_apply(
    row: pd.Series,
    lbs_config: dict,
    structure_data: StructureData,
) -> tuple[
    list[list[str | float | None]], list[list[str]], list[list[str]], dict, dict
]:
    results: list[list[str | float | None]] = []
    buses_out: list[list[str]] = []
    buses_in: list[list[str]] = []
    unit_emission_fees_map: dict = {}
    unit_tech_tags: dict = {}

    aggregate = row.name

    active_lbses = row.dropna().index

    for lbs_type in active_lbses:
        local_buses_process_lbs_type(
            aggregate,
            lbs_type,
            lbs_config,
            structure_data,
            results,
            buses_out,
            buses_in,
            unit_emission_fees_map,
            unit_tech_tags,
        )

    return results, buses_out, buses_in, unit_emission_fees_map, unit_tech_tags


def create_local_buses_data(
    lbs_config_df: pd.DataFrame, lbs_config: dict, structure_data: StructureData
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    results = []
    buses_out = []
    buses_in = []
    unit_emission_fees_map = {}
    local_tech_tags = {}

    aggregates = lbs_config_df.loc[lbs_config_df.iloc[:, 1:].sum(axis=1) > 0].set_index(
        InputXlsxColumnName.AGGREGATE
    )

    processed_data = aggregates.apply(
        lambda row: local_buses_process_aggregate_apply(
            row, lbs_config, structure_data
        ),
        axis=1,
        result_type="expand",
    )

    for (
        row_results,
        row_buses_out,
        row_buses_in,
        row_emission_fees,
        row_tech_tags,
    ) in processed_data.values:
        results.extend(row_results)
        buses_out.extend(row_buses_out)
        buses_in.extend(row_buses_in)
        unit_emission_fees_map.update(row_emission_fees)
        local_tech_tags.update(row_tech_tags)

    columns = [
        StructureSheetsColumnName.technology,
        StructureSheetsColumnName.technology_type,
        StructureTemporaryColumnName.min_capacity,
        StructureTemporaryColumnName.max_capacity,
        StructureTemporaryColumnName.unit_class,
        StructureSheetsColumnName.bus,
        StructureTemporaryColumnName.lbs,
        StructureSheetsColumnName.lbs_type,
        StructureSheetsColumnName.aggregate,
        StructureSheetsColumnName.energy_type,
        StructureSheetsColumnName.base_capacity,
        StructureSheetsColumnName.dsr_type,
    ]

    local_buses_df = pd.DataFrame(results, columns=columns)

    return (
        pd.DataFrame(
            buses_out,
            columns=[
                StructureTemporaryColumnName.lbs,
                StructureSheetsColumnName.bus,
                StructureSheetsColumnName.energy_type,
            ],
        ),
        local_buses_df,
        pd.DataFrame(
            buses_in,
            columns=[
                StructureSheetsColumnName.lbs_type,
                StructureSheetsColumnName.bus,
                StructureSheetsColumnName.energy_type,
            ],
        ),
        unit_emission_fees_map,
        local_tech_tags,
    )


def create_line_connections(
    subsystems_config_df: pd.DataFrame,
    global_buses_df: pd.DataFrame,
    local_buses_in_df: pd.DataFrame,
    transmission_fee_df: pd.DataFrame,
    local_buses_df: pd.DataFrame,
) -> pd.DataFrame:
    results = []
    for _, row in subsystems_config_df.set_index(
        InputXlsxColumnName.SUBSYSTEM
    ).iterrows():
        subsystem = row.name
        subsystem_energy = global_buses_df.loc[
            global_buses_df[StructureSheetsColumnName.bus] == subsystem,
            StructureSheetsColumnName.energy_type,
        ].iloc[0]
        transmission_loss = global_buses_df.loc[
            global_buses_df[StructureSheetsColumnName.bus] == subsystem,
            StructureSheetsColumnName.transmission_loss,
        ].iloc[0]

        for lbs_type in [x for x in row.index if row[x] == 1]:
            filtered_df = local_buses_in_df[
                (local_buses_in_df[StructureSheetsColumnName.lbs_type] == lbs_type)
                & (
                    local_buses_in_df[StructureSheetsColumnName.energy_type]
                    == subsystem_energy
                )
            ]

            for bus in filtered_df[StructureSheetsColumnName.bus]:
                line_name = f"{subsystem} -> {bus}"
                aggregate = local_buses_df.loc[
                    local_buses_df[StructureSheetsColumnName.bus] == bus,
                    StructureSheetsColumnName.aggregate,
                ]
                aggregate = aggregate.iloc[0] if not aggregate.empty else None

                transmission_fee = (
                    transmission_fee_df.loc[
                        (
                            transmission_fee_df[InputXlsxColumnName.AGGREGATE]
                            == aggregate
                        )
                        & (subsystem in transmission_fee_df.columns),
                        subsystem,
                    ].iloc[0]
                    if aggregate
                    in transmission_fee_df[InputXlsxColumnName.AGGREGATE].values
                    else np.nan
                )

                results.append(
                    [
                        line_name,
                        subsystem_energy,
                        subsystem,
                        bus,
                        transmission_loss,
                        np.nan,
                        transmission_fee,
                    ]
                )

    columns = [
        StructureSheetsColumnName.name,
        StructureSheetsColumnName.energy_type,
        StructureSheetsColumnName.bus_from,
        StructureSheetsColumnName.bus_to,
        StructureSheetsColumnName.transmission_loss,
        StructureTemporaryColumnName.max_capacity,
        StructureSheetsColumnName.transmission_fee,
    ]

    return pd.DataFrame(results, columns=columns).reindex(columns=columns)


def create_structure_and_initial(
    structure_data: StructureData,
    output_path: Path,
) -> None:
    subsystem_config_df = structure_data.configuration[InputFileFieldName.subsystems]
    transmission_fee_cost_df = structure_data.configuration[
        InputFileFieldName.transmission_fee_cost
    ]
    lbs_config_df = structure_data.configuration[InputFileFieldName.lbs_type]
    subsystem_config = structure_data.subsystem_types
    lbs_config = structure_data.lbs_types
    global_tech_df = structure_data.global_technologies
    transmission_fee_df = structure_data.configuration[
        InputFileFieldName.transmission_fee
    ]
    aggregate_types_dict = structure_data.aggregate_types
    emission_fees_dict = structure_data.emission_fees
    dsr_df = structure_data.configuration[InputFileFieldName.dsr]
    power_reserve_df = structure_data.power_reserve
    (
        global_buses_df,
        global_unit_emission_fee_map,
        global_tech_tags_mapping,
    ) = create_global_data(global_tech_df, subsystem_config, subsystem_config_df)
    (
        buses_out_df,
        local_buses_df,
        buses_in_df,
        local_unit_emission_fee_map,
        local_techs_tags_mapping,
    ) = create_local_buses_data(lbs_config_df, lbs_config, structure_data)
    lines_df = create_line_connections(
        subsystem_config_df,
        global_buses_df,
        buses_in_df,
        transmission_fee_df,
        local_buses_df,
    )
    energy_types = pd.concat(
        [
            buses_in_df[StructureSheetsColumnName.energy_type],
            buses_out_df[StructureSheetsColumnName.energy_type],
        ],
        ignore_index=True,
    ).tolist()

    structure_data_created = create_structure_dict(
        global_buses_df=global_buses_df,
        local_buses_df=local_buses_df,
        lines_df=lines_df,
        energy_type_list=energy_types,
        buses_out_df=buses_out_df,
        aggregate_types_dict=aggregate_types_dict,
        emission_fees_dict=emission_fees_dict,
        unit_emission_fee_map=global_unit_emission_fee_map
        | local_unit_emission_fee_map,
        tags_techs_mapping=merge_dicts(
            local_techs_tags_mapping, global_tech_tags_mapping
        ),
        transmission_fee_cost_df=transmission_fee_cost_df,
        dsr_df=dsr_df,
        power_reserve_df=power_reserve_df,
    )
    initial_state_data = create_initial_state_dict(
        global_buses_df=global_buses_df,
        local_buses_df=local_buses_df,
        fraction_config=lbs_config_df,
    )

    write_to_excel(
        data=structure_data_created,
        output_path=output_path,
        filename=XlsxFileName.structure,
    )
    write_to_excel(
        data=initial_state_data,
        output_path=output_path,
        filename=XlsxFileName.initial_state,
    )
