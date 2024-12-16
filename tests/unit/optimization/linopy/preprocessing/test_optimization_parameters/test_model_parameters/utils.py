from dataclasses import dataclass, field

from pandas import Series

from pyzefir.model.network import Network
from pyzefir.model.network_elements import (
    EnergySource,
    EnergySourceType,
    NetworkElement,
)


@dataclass
class NetworkElementTestImplementation(NetworkElement):
    """
    Class to test the ModelParameters.fetch_element_prop
    """

    name: str
    scalar_prop: int = 0
    vector_prop: Series = field(default_factory=Series)
    dict_prop: dict[str, int] = field(default_factory=dict)

    def validate(self, network: Network) -> None:
        pass


@dataclass(kw_only=True)
class EnergySourceTestImplementation(EnergySource):
    """
    Class to test the ModelParameters.fetch_element_type_prop
    """

    energy_source_type: str
    unit_base_cap: float = 0.0

    def validate(self, network: Network) -> None:
        pass


@dataclass(kw_only=True)
class EnergySourceTypeTestImplementation(EnergySourceType):
    """
    Class to test the ModelParameters.fetch_element_type_prop
    """

    name: str
    scalar_prop: float | None = None
    vector_prop: Series | None = None
    dict_prop: dict[str, int] | None = None
    build_time: int = field(init=False, default=0)
    life_time: int = field(init=False, default=0)
    capex: Series = field(init=False, default_factory=Series)
    opex: Series = field(init=False, default_factory=Series)
    power_utilization: float = field(init=False, default=0.9)

    def validate(self, network: Network) -> None:
        pass
