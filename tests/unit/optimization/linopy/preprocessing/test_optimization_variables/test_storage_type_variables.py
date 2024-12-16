import pytest
from linopy import Model
from numpy import arange, array, ndarray

from pyzefir.model.network import Network
from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.linopy.preprocessing.variables.storage_type_variables import (
    StorageTypeVariables,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.conftest import N_YEARS


@pytest.mark.parametrize(
    ("y_sample", "h_sample", "n_tstor"),
    [
        (arange(N_YEARS), arange(100), 4),
        (array([0, 3]), arange(100), 1),
        (array([1, 4]), arange(2000), 10),
    ],
)
def test_storage_type_variables(
    y_sample: ndarray,
    h_sample: ndarray,
    n_tstor: int,
    opt_config: OptConfig,
    empty_network: Network,
) -> None:
    opt_config.year_sample = y_sample
    opt_config.hour_sample = h_sample
    indices = Indices(empty_network, opt_config)
    indices.STOR = IndexingSet(arange(n_tstor))
    setattr(
        indices, "_AGGR_TSTORS", {0: arange(n_tstor)}
    )  # to simulate that all tstors are connected to aggregator
    model = Model()

    storage_type_variables = StorageTypeVariables(model, indices)

    yy = y_sample.shape[0]
    assert storage_type_variables.tcap.size == n_tstor * yy
    assert all(
        len(storage_type_variables.tcap.indexes["index"][i]) == 3
        for i in range(n_tstor * yy)
    )
    assert storage_type_variables.tcap_plus.size == n_tstor * yy
    assert all(
        len(storage_type_variables.tcap_plus.indexes["index"][i]) == 3
        for i in range(n_tstor * yy)
    )
    assert storage_type_variables.tcap_minus.size == n_tstor * yy * yy
    assert all(
        len(storage_type_variables.tcap_minus.indexes["index"][i]) == 4
        for i in range(n_tstor * yy)
    )
    assert storage_type_variables.tcap_base_minus.size == n_tstor * yy
    assert all(
        len(storage_type_variables.tcap_base_minus.indexes["index"][i]) == 3
        for i in range(n_tstor * yy)
    )
