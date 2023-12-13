import pytest

from pyzefir.model.network import Network
from tests.unit.defaults import ELECTRICITY, HEATING, default_network_constants


@pytest.fixture
def network_fixture() -> Network:
    network = Network(
        energy_types=[ELECTRICITY, HEATING],
        network_constants=default_network_constants,
        emission_types=["CO2", "PM10"],
    )
    return network
