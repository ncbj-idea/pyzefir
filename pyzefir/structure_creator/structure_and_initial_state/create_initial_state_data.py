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
    StructureSheetName,
    StructureSheetsColumnName,
    StructureTemporaryColumnName,
)


def create_initial_state_dict(
    global_buses_df: pd.DataFrame,
    local_buses_df: pd.DataFrame,
    fraction_config: pd.DataFrame,
) -> dict[StructureSheetName, pd.DataFrame]:
    tech_stack_data = []
    for row in fraction_config.iterrows():
        data = row[1]
        aggregate = data[0]
        for lbs_type in data.index[1:]:
            if pd.isna(data[lbs_type]):
                continue

            fraction_df = pd.DataFrame(
                columns=[
                    StructureSheetsColumnName.technology_stack,
                    StructureSheetsColumnName.aggregate,
                    StructureSheetsColumnName.base_fraction,
                ]
            )
            fraction_df[StructureSheetsColumnName.technology_stack] = local_buses_df[
                (local_buses_df[StructureSheetsColumnName.lbs_type] == lbs_type)
                & (local_buses_df[StructureSheetsColumnName.aggregate] == aggregate)
            ][StructureTemporaryColumnName.lbs]
            fraction_df[StructureSheetsColumnName.aggregate] = aggregate
            fraction_df[StructureSheetsColumnName.base_fraction] = data[lbs_type]
            tech_stack_data.append(fraction_df)
    base_capacity_global = (
        global_buses_df[
            [
                StructureSheetsColumnName.technology,
                StructureTemporaryColumnName.base_cap,
            ]
        ]
        .drop_duplicates(ignore_index=True)
        .rename(
            {
                StructureTemporaryColumnName.base_cap: StructureSheetsColumnName.base_capacity
            },
            axis=1,
        )
    )
    base_capacity_local = (
        local_buses_df[
            [
                StructureSheetsColumnName.technology,
                StructureSheetsColumnName.base_capacity,
            ]
        ]
        .drop_duplicates(ignore_index=True)
        .dropna()
    )
    initial_state_data = {
        StructureSheetName.TECHNOLOGY: pd.concat(
            (base_capacity_global, base_capacity_local)
        ),
        StructureSheetName.TECHNOLOGYSTACK: pd.concat(tech_stack_data).drop_duplicates(
            ignore_index=True
        ),
    }
    return initial_state_data
