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

from enum import StrEnum, auto, unique


@unique
class XlsxFileName(StrEnum):
    cost_parameters = auto()
    fuel_parameters = auto()
    n_consumers = auto()
    relative_emission_limits = auto()
    technology_cap_limits = auto()
    technology_type_cap_limits = auto()
    yearly_demand = auto()
    generation_fraction = auto()
    configuration = auto()
    aggregates = auto()
    subsystems = auto()
    emissions = auto()
    transmission_fees = auto()
    generation_compensation = auto()
    yearly_emission_reduction = auto()

    def __str__(self) -> str:
        return f"{self.name}.xlsx"


@unique
class SubDirectory(StrEnum):
    fractions = auto()
    scenarios = auto()
    lbs = auto()
