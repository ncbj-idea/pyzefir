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

from pyzefir.model.network_elements import LocalBalancingStack
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class LocalBalancingStackParser(AbstractElementParser):
    def __init__(
        self, stack_df: pd.DataFrame, bus_df: pd.DataFrame, stack_bus_df: pd.DataFrame
    ) -> None:
        self.stack_df = stack_df
        self.stack_buses_mapping = self._prepare_stack_buses_mapping(
            bus_df, stack_bus_df
        )

    def create(self) -> tuple[LocalBalancingStack, ...]:
        stacks = self.stack_df.apply(
            self._create_stack,
            axis=1,
        )
        return tuple(stacks)

    def _create_stack(
        self,
        df_row: pd.Series,
    ) -> LocalBalancingStack:
        return LocalBalancingStack(
            name=df_row["name"],
            buses_out={col: bus_name for col, bus_name in df_row[1:].items()},
            buses=self.stack_buses_mapping[df_row["name"]],
        )

    @staticmethod
    def _prepare_stack_buses_mapping(
        bus_df: pd.DataFrame, stack_bus_df: pd.DataFrame
    ) -> dict[str, dict[str, set[str]]]:
        """
        Creates dict containing mapping (lbs, energy_type) -> set[buses]
        """
        bus_df = bus_df.rename(columns={"name": "bus"})
        stack_busses_df = pd.merge(bus_df, stack_bus_df, on="bus", how="inner")
        return (
            stack_busses_df.groupby("technology_stack")
            .apply(
                lambda group: group.groupby("energy_type")["bus"].apply(set).to_dict()
            )
            .to_dict()
        )
