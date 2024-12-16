import pytest
from linopy import Model
from numpy import arange, array, ndarray

from pyzefir.model.network import Network
from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.linopy.preprocessing.variables.line_variables import (
    LineVariables,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.conftest import N_YEARS


@pytest.mark.parametrize(
    ("y_sample", "h_sample", "n_line"),
    [
        (arange(N_YEARS), arange(50), 1),
        (arange(N_YEARS), arange(500), 10),
        (array([1]), array([1]), 10),
        (arange(2), arange(100), 4),
    ],
)
def test_line_variables(
    y_sample: ndarray,
    h_sample: ndarray,
    n_line: int,
    opt_config: OptConfig,
    empty_network: Network,
) -> None:
    opt_config.year_sample = y_sample
    opt_config.hour_sample = h_sample
    model, indices = Model(), Indices(empty_network, opt_config)
    indices.LINE = IndexingSet(arange(n_line))
    line_variables = LineVariables(model, indices)

    assert line_variables.flow.shape == (n_line, h_sample.shape[0], y_sample.shape[0])
