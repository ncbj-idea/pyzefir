import pytest
from gurobipy import Model
from numpy import arange, array, ndarray

from pyzefir.model.network import Network
from pyzefir.optimization.gurobi.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.gurobi.preprocessing.opt_variables import (
    OptimizationVariables,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.gurobi.conftest import N_YEARS


@pytest.mark.parametrize(
    ("y_sample", "h_sample", "n_stor"),
    [
        (arange(N_YEARS), arange(100), 4),
        (array([0, 3]), arange(100), 1),
        (array([1, 4]), arange(2000), 10),
    ],
)
def test_storage_variables(
    y_sample: ndarray,
    h_sample: ndarray,
    n_stor: int,
    opt_config: OptConfig,
    empty_network: Network,
) -> None:
    opt_config.year_sample = y_sample
    opt_config.hour_sample = h_sample
    indices = Indices(empty_network, opt_config)
    indices.STOR = IndexingSet(arange(n_stor))
    model = Model()

    variables = OptimizationVariables(model, indices, opt_config).stor
    model.update()

    yy, hh = y_sample.shape[0], h_sample.shape[0]
    assert variables.gen.shape == (n_stor, hh, yy)
    assert variables.load.shape == (n_stor, hh, yy)
    assert variables.soc.shape == (n_stor, hh, yy)
    assert variables.cap.shape == (n_stor, yy)
    assert len(variables.cap_plus) == n_stor * yy
    assert all(len(variables.cap_plus.keys()[i]) == 2 for i in range(n_stor * yy))
    assert len(variables.cap_minus) == n_stor * yy * yy
    assert all(len(variables.cap_minus.keys()[i]) == 3 for i in range(n_stor * yy))
    assert len(variables.cap_base_minus) == n_stor * yy
    assert all(len(variables.cap_base_minus.keys()[i]) == 2 for i in range(n_stor * yy))
