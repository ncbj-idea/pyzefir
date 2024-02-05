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

from abc import ABC
from dataclasses import dataclass

import pandas as pd


@dataclass
class ExportableResultsGroup(ABC):
    pass


@dataclass
class ExportableGeneratorsResults(ExportableResultsGroup):
    generation: dict[str, pd.DataFrame]
    dump_energy: dict[str, pd.DataFrame]
    capacity: pd.DataFrame
    generation_per_energy_type: dict[str, pd.DataFrame]
    dump_energy_per_energy_type: dict[str, pd.DataFrame]
    capex: pd.DataFrame


@dataclass
class ExportableStorageResults(ExportableResultsGroup):
    generation: dict[str, pd.DataFrame]
    load: dict[str, pd.DataFrame]
    state_of_charge: dict[str, pd.DataFrame]
    capacity: pd.DataFrame
    capex: pd.DataFrame


@dataclass
class ExportableLinesResults(ExportableResultsGroup):
    flow: dict[str, pd.DataFrame]


@dataclass
class ExportableFractionsResults(ExportableResultsGroup):
    fraction: dict[str, pd.DataFrame]


@dataclass
class ExportableBusResults(ExportableResultsGroup):
    generation_ens: dict[str, pd.DataFrame]
    shift_minus: dict[str, pd.DataFrame]
    shift_plus: dict[str, pd.DataFrame]


@dataclass
class ExportableResults:
    objective_value: pd.Series
    generators_results: ExportableGeneratorsResults
    storages_results: ExportableStorageResults
    lines_results: ExportableLinesResults
    fractions_results: ExportableFractionsResults
    bus_results: ExportableBusResults
