import pytest
from numpy import all, arange, array, ndarray

from pyzefir.model.network import Network
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.conftest import N_HOURS


@pytest.mark.parametrize(
    "sample", [arange(N_HOURS), array([1, 2, 10, 15]), array([0]), arange(100)]
)
def test_create(
    sample: ndarray, complete_network: Network, opt_config: OptConfig
) -> None:
    opt_config.hour_sample = sample
    indices = Indices(complete_network, opt_config)
    params = OptimizationParameters(complete_network, indices, opt_config).tf

    for tf_id, tf_name in indices.TF.mapping.items():
        transmission_fees = complete_network.transmission_fees[tf_name]
        assert all(params.fee[tf_id] == transmission_fees.fee[sample])
