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

from dataclasses import dataclass, field, fields

import pandas as pd


@dataclass
class BaseData:
    def convert_to_dict_of_dfs(self) -> dict[str, pd.DataFrame]:
        dfs_dict = {}
        for field_info in fields(self):
            field_name = field_info.name
            df_list = getattr(self, field_name)
            if isinstance(df_list, list) and df_list:
                dfs_dict[self._handle_field_name(field_name)] = pd.concat(df_list)
        return dfs_dict

    @staticmethod
    def _handle_field_name(name: str) -> str:
        if name == "TechnologyStack_Buses_out":
            return name
        return name.replace("__", " - ").replace("_", " ")


@dataclass
class StructureData(BaseData):
    Energy_Types: list[pd.DataFrame] = field(default_factory=list)
    Emission_Types: list[pd.DataFrame] = field(default_factory=list)
    Aggregates: list[pd.DataFrame] = field(default_factory=list)
    Lines: list[pd.DataFrame] = field(default_factory=list)
    Transmission_Fees: list[pd.DataFrame] = field(default_factory=list)
    Buses: list[pd.DataFrame] = field(default_factory=list)
    Generators: list[pd.DataFrame] = field(default_factory=list)
    Emission_Fees__Emission_Types: list[pd.DataFrame] = field(default_factory=list)
    Generator__Emission_Fees: list[pd.DataFrame] = field(default_factory=list)
    Storages: list[pd.DataFrame] = field(default_factory=list)
    DSR: list[pd.DataFrame] = field(default_factory=list)
    Technology__Bus: list[pd.DataFrame] = field(default_factory=list)
    TechnologyStack_Buses_out: list[pd.DataFrame] = field(default_factory=list)
    TechnologyStack_Buses: list[pd.DataFrame] = field(default_factory=list)
    TechnologyStack__Aggregate: list[pd.DataFrame] = field(default_factory=list)
    Power_Reserve: list[pd.DataFrame] = field(default_factory=list)
    Generator_Binding: list[pd.DataFrame] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.DSR = [
            pd.DataFrame(
                columns=[
                    "name",
                    "compensation_factor",
                    "balancing_period_len",
                    "penalization",
                    "relative_shift_limit",
                    "abs_shift_limit",
                ]
            )
        ]
        self.Power_Reserve = [
            pd.DataFrame(columns=["tag_name", "energy_type", "power_reserve_value"])
        ]


@dataclass
class InitialStateData(BaseData):
    Technology: list[pd.DataFrame] = field(default_factory=list)
    TechnologyStack: list[pd.DataFrame] = field(default_factory=list)
