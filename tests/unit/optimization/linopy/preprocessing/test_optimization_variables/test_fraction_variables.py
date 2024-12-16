import pytest
from linopy import Model
from numpy import arange, array, ndarray

from pyzefir.model.network import Network
from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.linopy.preprocessing.variables.fraction_variables import (
    FractionVariables,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.conftest import N_YEARS


@pytest.mark.parametrize(
    ("y_sample", "n_aggr", "n_lbs"),
    [
        (array([0, 2]), 2, 3),
        (arange(N_YEARS), 1, 10),
        (array([0, 2, 4]), 1, 5),
        (array([0]), 5, 1),
    ],
)
def test_fraction_variables(
    y_sample: ndarray,
    n_aggr: int,
    n_lbs: int,
    opt_config: OptConfig,
    empty_network: Network,
) -> None:
    opt_config.year_sample = y_sample
    model, indices = Model(), Indices(empty_network, opt_config)
    indices.AGGR, indices.LBS = IndexingSet(arange(n_aggr)), IndexingSet(arange(n_lbs))
    fraction_variables = FractionVariables(model, indices)

    assert fraction_variables.fraction.shape == (n_aggr, n_lbs, y_sample.shape[0])
