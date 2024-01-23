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

from pyzefir.model.network_elements import DemandProfile
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class DemandProfileParser(AbstractElementParser):
    def __init__(
        self,
        demand_dict: dict[str, pd.DataFrame],
    ) -> None:
        self.demand_dict = demand_dict

    def create(self) -> tuple[DemandProfile, ...]:
        demand_profiles: list[DemandProfile] = list()
        for name, demand_df in self.demand_dict.items():
            demand_profile = DemandProfile(
                name=name,
                normalized_profile=demand_df.set_index("hour_idx").to_dict("series"),
            )
            demand_profiles.append(demand_profile)

        return tuple(demand_profiles)
