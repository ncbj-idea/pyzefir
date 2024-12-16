import pytest

from pyzefir.model.network import Network
from tests.unit.defaults import (
    CO2_EMISSION,
    ELECTRICITY,
    HEATING,
    PM10_EMISSION,
    default_network_constants,
)


@pytest.fixture
def network() -> Network:
    return Network(
        energy_types=[ELECTRICITY, HEATING],
        network_constants=default_network_constants,
        emission_types=[CO2_EMISSION, PM10_EMISSION],
    )
