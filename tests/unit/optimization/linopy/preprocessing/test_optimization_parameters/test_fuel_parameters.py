import pytest
from numpy import all, arange, array, ndarray

from pyzefir.model.network import Network
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.conftest import N_YEARS


@pytest.mark.parametrize(
    "sample", [arange(N_YEARS), array([0, 3]), array([0]), array([N_YEARS - 1])]
)
def test_create(
    sample: ndarray, complete_network: Network, opt_config: OptConfig
) -> None:
    opt_config.year_sample = sample
    indices = Indices(complete_network, opt_config)
    result = OptimizationParameters(complete_network, indices, opt_config).fuel
    fuels = complete_network.fuels

    for fuel_id, fuel_name in indices.FUEL.mapping.items():
        fuel = fuels[fuel_name]
        assert result.u_emission[fuel_id] == fuel.emission
        assert result.energy_per_unit[fuel_id] == fuel.energy_per_unit
        assert all(result.unit_cost[fuel_id] == fuel.cost[sample])
        assert all(result.availability[fuel_id] == fuel.availability[sample])
