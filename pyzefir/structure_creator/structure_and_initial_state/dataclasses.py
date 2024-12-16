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
    """
    A base data class for handling multiple DataFrames as attributes and converting them
    into a dictionary format.

    This class provides functionality to store lists of DataFrames as attributes and
    convert those attributes into a dictionary where each key corresponds to a specific
    attribute name (formatted appropriately) and each value is a concatenated DataFrame
    of the respective attribute.
    """

    def convert_to_dict_of_dfs(self) -> dict[str, pd.DataFrame]:
        """
        Convert object attributes to a dictionary of concatenated DataFrames.

        This method iterates over the fields of the object, checking for attributes that
        are lists of DataFrames. If a list is found, it concatenates those DataFrames and
        adds them to a dictionary using a formatted version of the attribute name as the key.

        Returns:
            - dict[str, pd.DataFrame]: A dictionary where each key is a formatted attribute
                name and each value is a concatenated DataFrame of the respective attribute.
        """
        dfs_dict = {}
        for field_info in fields(self):
            field_name = field_info.name
            df_list = getattr(self, field_name)
            if isinstance(df_list, list) and df_list:
                dfs_dict[self._handle_field_name(field_name)] = pd.concat(df_list)
        return dfs_dict

    @staticmethod
    def _handle_field_name(name: str) -> str:
        """
        Format field name for use as a dictionary key.

        Args:
            - name (str): The original field name to be formatted.

        Returns:
            - str: The formatted field name suitable for use as a dictionary key.
        """
        if name == "TechnologyStack_Buses_out":
            return name
        return name.replace("__", " - ").replace("_", " ")


@dataclass
class StructureData(BaseData):
    """
    A data class that encapsulates various structured data related to energy systems.

    This class extends the BaseData class to include multiple lists of DataFrames that
    represent different aspects of energy systems, such as energy types, emission types,
    aggregates, and more. Each attribute is initialized to an empty list of DataFrames by default.
    """

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
        """Initialize empty dataframes with specific column names."""
        self.DSR = [
            pd.DataFrame(
                columns=[
                    "name",
                    "compensation_factor",
                    "balancing_period_len",
                    "penalization_minus",
                    "penalization_plus",
                    "relative_shift_limit",
                    "abs_shift_limit",
                    "hourly_relative_shift_plus_limit",
                    "hourly_relative_shift_minus_limit",
                ]
            )
        ]
        self.Power_Reserve = [
            pd.DataFrame(columns=["tag_name", "energy_type", "power_reserve_value"])
        ]


@dataclass
class InitialStateData(BaseData):
    """
    A data class that holds initial state data for energy system modeling.

    This class extends the BaseData class and includes lists of DataFrames that represent
    the initial states of technologies and technology stacks. It serves as a structured
    way to manage and access this data for further processing in energy modeling workflows.
    """

    Technology: list[pd.DataFrame] = field(default_factory=list)
    TechnologyStack: list[pd.DataFrame] = field(default_factory=list)
