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
    ("y_sample", "h_sample", "n_tgens", "energy_types"),
    [
        (arange(N_YEARS), arange(100), 4, {"heat"}),
        (array([0, 3]), arange(100), 1, {"heat", "ee", "transport"}),
        (array([1, 4]), arange(2000), 10, {"heat", "ee"}),
    ],
)
def test_generator_type_variables(
    y_sample: ndarray,
    h_sample: ndarray,
    n_tgens: int,
    energy_types: set[str],
    opt_config: OptConfig,
    empty_network: Network,
) -> None:
    opt_config.year_sample = y_sample
    opt_config.hour_sample = h_sample
    empty_network._energy_types = energy_types
    indices = Indices(empty_network, opt_config)
    indices.TGEN = IndexingSet(arange(n_tgens))
    setattr(
        indices, "_AGGR_TGENS", {0: arange(n_tgens)}
    )  # to simulate that all tgens are connected to aggregator
    model = Model()

    variables = OptimizationVariables(model, indices, opt_config).tgen
    model.update()

    yy = y_sample.shape[0]
    assert len(variables.tcap) == n_tgens * yy
    assert all(len(variables.tcap.keys()[i]) == 3 for i in range(n_tgens * yy))
    assert len(variables.tcap_plus) == n_tgens * yy
    assert all(len(variables.tcap_plus.keys()[i]) == 3 for i in range(n_tgens * yy))
    assert len(variables.tcap_minus) == n_tgens * yy * yy
    assert all(len(variables.tcap_minus.keys()[i]) == 4 for i in range(n_tgens * yy))
    assert len(variables.tcap_base_minus) == n_tgens * yy
    assert all(
        len(variables.tcap_base_minus.keys()[i]) == 3 for i in range(n_tgens * yy)
    )
