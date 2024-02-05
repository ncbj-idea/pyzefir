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

from enum import StrEnum, auto
from typing import LiteralString


class XlsxFileName(StrEnum):
    cost_parameters = auto()
    fuel_parameters = auto()
    n_consumers = auto()
    technology_cap_limits = auto()
    technology_type_cap_limits = auto()
    yearly_demand = auto()
    configuration = auto()
    cap_min = auto()
    cap_max = auto()
    cap_base = auto()
    demand_chunks_periods = auto()
    structure = auto()
    initial_state = auto()
    demand_chunks = auto()
    relative_emission_limits = auto()
    generation_fraction = auto()
    power_reserve = auto()

    def __str__(self) -> LiteralString:
        return f"{self.name}.xlsx"


class JsonFileName(StrEnum):
    global_techs = auto()
    aggr_types = auto()
    emission_fees = auto()

    def __str__(self) -> LiteralString:
        return f"{self.name}.json"


class SubDirectory(StrEnum):
    fractions = auto()
    subsystems = auto()
    scenarios = auto()
    cap_range = auto()
    lbs = auto()
    structure_creator_resources = auto()
