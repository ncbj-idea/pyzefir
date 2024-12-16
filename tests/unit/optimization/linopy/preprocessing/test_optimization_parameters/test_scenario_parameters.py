import pytest
from numpy import all, arange, ndarray, ones

from pyzefir.model.network import Network
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.conftest import N_YEARS


@pytest.mark.parametrize(
    "discount", [ones(N_YEARS) * 0.05, ones(N_YEARS) * 0, arange(N_YEARS)]
)
def test_create(
    discount: ndarray, complete_network: Network, opt_config: OptConfig
) -> None:
    opt_config.discount_rate = discount
    indices = Indices(complete_network, opt_config)
    result = OptimizationParameters(
        complete_network, indices, opt_config
    ).scenario_parameters

    assert all(result.discount_rate == discount)
