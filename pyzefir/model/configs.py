"""
PyZefir
Copyright (C) 2023 Narodowe Centrum Badań Jądrowych

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from abc import ABC
from dataclasses import dataclass, field

import pandas as pd

from pyzefir.model.enum import EnergyType, EmissionType


@dataclass(kw_only=True)
class EnergySourceConfig(ABC):
    life_time: int
    build_time: int
    capex: pd.Series
    opex: pd.Series
    energy_type: set[EnergyType]
    input_energy_type: set[EnergyType] = field(default_factory=set)


@dataclass(kw_only=True)
class GeneratorConfig(EnergySourceConfig):
    is_dispatchable: bool
    energy_carriers: set[str]
    efficiency: dict[EnergyType, float]
    cap_min: pd.Series
    """
    Minimal amount of installed capacity of that unit for a given year
    """
    cap_max: pd.Series
    """
    Maximal amount of installed capacity of that unit for a given year
    """
    delta_cap_min: pd.Series
    """
    Maximal decrease of installed capacity of that unit for a given year
    """
    delta_cap_max: pd.Series
    """
    Maximal increase of installed capacity of that unit for a given year
    """
    generation_demand: dict[EnergyType, dict[EnergyType, pd.Series]]
    """
    Dict showing how much energy of given type is needed to produce one unit of energy
    for specific energy type
    """
    emission_reduction: dict[EmissionType, float]
    """
    Dict describing the reduction of emission for specific emission type applied
    by the generator.
    """


@dataclass(kw_only=True)
class StorageConfig(EnergySourceConfig):
    generation_loss: pd.Series
    load_loss: pd.Series
    capacity: float
    loading_time: float
