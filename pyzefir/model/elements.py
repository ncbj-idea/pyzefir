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
from dataclasses import InitVar, dataclass, field

import pandas as pd

from pyzefir.model.configs import (
    EnergySourceConfig,
    GeneratorConfig,
    StorageConfig,
)

from pyzefir.model.enum import EnergyType, EmissionType


@dataclass
class Bus:
    name: str
    energy_type: EnergyType
    generators: set[str] = field(default_factory=set)
    storages: set[str] = field(default_factory=set)
    lines_in: set[str] = field(default_factory=set)
    lines_out: set[str] = field(default_factory=set)


@dataclass(kw_only=True)
class EnergySource:
    name: str
    config: type[EnergySourceConfig]


@dataclass(kw_only=True)
class Generator(EnergySource):
    config: GeneratorConfig
    bus: InitVar[str | set[str]]
    buses: set[str] = field(init=False)

    def __post_init__(self, bus: str | set[str]):
        if bus is None:
            self.buses = set()
        else:
            self.buses = {bus} if isinstance(bus, str) else bus


@dataclass(kw_only=True)
class Storage(EnergySource):
    config: StorageConfig
    bus: str


@dataclass(kw_only=True)
class Line:
    name: str
    energy_type: EnergyType
    fr: str
    to: str
    transmission_loss: float
    max_capacity: float


@dataclass
class Load:
    name: str
    demand: pd.DataFrame


@dataclass
class LocalBalancingStack:
    name: str
    buses: set[str] = field(default_factory=set)
    outlets: dict[EnergyType, str] = field(default_factory=dict)
    """
    Dictionary mapping energy type to bus to which Aggregated load is attached.
    For every energy type there must be at least one bus in LocalBalancingStack,
    which servers as outlet.
    """


@dataclass(kw_only=True)
class Carrier(ABC):
    name: str
    emission: dict[EmissionType, float]

    class Config:
        arbitrary_types_allowed = True


@dataclass
class ProfiledCarrier(Carrier):
    profile: pd.DataFrame


@dataclass
class FreeCarrier(Carrier):
    availabilities: pd.DataFrame
    cost: pd.Series
