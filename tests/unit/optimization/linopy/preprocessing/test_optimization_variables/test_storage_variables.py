from unittest.mock import MagicMock

import pytest
from linopy import Model
from numpy import arange, array, ndarray

from pyzefir.model.network import Network
from pyzefir.model.network_elements import Storage
from pyzefir.model.utils import AllowedStorageGenerationLoadMethods
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.variables.storage_variables import (
    StorageVariables,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.conftest import N_YEARS


def _test_h_y_variable(
    indices: Indices,
    network: Network,
    var_name: str,
) -> None:
    """Test if a given time series variable Var[st, h, y] is correct."""
    var = getattr(
        StorageVariables(Model(), indices, network),
        var_name,
    )
    assert var.shape == (
        len(indices.STOR),
        len(indices.H),
        len(indices.Y),
    ), f"var shape {var_name}: {var.shape} is incorrect, should be {len(indices.STOR), len(indices.Y), len(indices.H)}"


def _test_cap_plus_variable(
    indices: Indices,
    network: Network,
    n_global_storages: int,
) -> None:
    """Test if cap_plus variable is correct."""
    cap_plus = StorageVariables(Model(), indices, network).cap_plus
    assert cap_plus.size == n_global_storages * len(
        indices.Y
    ), f"cap_plus size: {cap_plus.size} is incorrect, should be {n_global_storages * len(indices.Y)}"
    assert all(
        len(x) == 2 for x in cap_plus.indexes["index"]
    ), "cap_plus variable shape is not correct"


def _test_cap_minus_variable(
    indices: Indices,
    network: Network,
    n_global_storages: int,
) -> None:
    """Test if cap_minus variable is correct."""
    cap_minus = StorageVariables(Model(), indices, network).cap_minus
    assert (
        cap_minus.size == n_global_storages * len(indices.Y) ** 2
    ), f"cap_minus variable size is {cap_minus.size} but expected {n_global_storages * len(indices.Y) ** 2}"
    assert all(
        len(x) == 3 for x in cap_minus.indexes["index"]
    ), "cap_minus variable shape is not correct"


def _test_base_cap_minus_variable(
    indices: Indices,
    network: Network,
    n_global_storages: int,
) -> None:
    """Test if base_cap_minus variable is correct."""
    cap_base_minus = StorageVariables(Model(), indices, network).cap_base_minus
    assert cap_base_minus.size == n_global_storages * len(
        indices.Y
    ), f"cap_minus variable size is {cap_base_minus.size} but expected {n_global_storages * len(indices.Y) ** 2}"
    assert all(
        len(x) == 2 for x in cap_base_minus.indexes["index"]
    ), "cap_base_minus variable shape is not correct"


@pytest.mark.parametrize(
    ("y_sample", "h_sample"),
    [
        (arange(N_YEARS), arange(100)),
        (array([0, 3]), arange(100)),
        (array([1, 4]), arange(2000)),
    ],
)
def test_storage_variables(
    y_sample: ndarray,
    h_sample: ndarray,
    opt_config: OptConfig,
    global_storages: dict[str, Storage],
    complete_network: Network,
) -> None:
    opt_config.year_sample, opt_config.hour_sample = y_sample, h_sample
    indices = Indices(complete_network, opt_config)

    _test_h_y_variable(indices, complete_network, var_name="gen")
    _test_h_y_variable(indices, complete_network, var_name="gen")
    _test_h_y_variable(indices, complete_network, var_name="load")
    _test_h_y_variable(indices, complete_network, var_name="soc")
    _test_cap_plus_variable(
        indices, complete_network, n_global_storages=len(global_storages)
    )
    _test_cap_minus_variable(
        indices, complete_network, n_global_storages=len(global_storages)
    )
    _test_base_cap_minus_variable(
        indices, complete_network, n_global_storages=len(global_storages)
    )


@pytest.mark.parametrize(
    "storage_mapping, storage_type_inverse, storages, storage_types, expected_output",
    [
        pytest.param(
            {0: "storage1", 1: "storage2"},
            {"source1": 0, "source2": 1},
            {
                "storage1": MagicMock(energy_source_type="source1"),
                "storage2": MagicMock(energy_source_type="source2"),
                "storage3": MagicMock(energy_source_type="source2"),
            },
            {
                "source1": MagicMock(generation_load_method="milp"),
                "source2": MagicMock(generation_load_method="milp"),
            },
            {
                (0, 0): "binary_var_MILP_bin",
                (1, 1): "binary_var_MILP_bin",
            },
            id="happy_path",
        ),
        pytest.param(
            {0: "storage1", 1: "storage2"},
            {"source1": 0, "source2": 1},
            {
                "storage1": MagicMock(energy_source_type="source1"),
                "storage2": MagicMock(energy_source_type="source2"),
                "storage3": MagicMock(energy_source_type="source2"),
            },
            {
                "source1": MagicMock(generation_load_method="other"),
                "source2": MagicMock(generation_load_method="other"),
            },
            {},
            id="no_method_matching",
        ),
        pytest.param(
            {},
            {"source1": 0, "source2": 1},
            {
                "storage1": MagicMock(energy_source_type="source1"),
                "storage2": MagicMock(energy_source_type="source2"),
                "storage3": MagicMock(energy_source_type="source2"),
            },
            {
                "source1": MagicMock(generation_load_method="milp"),
                "source2": MagicMock(generation_load_method="milp"),
            },
            {},
            id="no_storage_mapping_dict",
        ),
        pytest.param(
            {0: "storage1", 1: "storage2", 2: "storage3"},
            {"source1": 0, "source2": 1},
            {
                "storage1": MagicMock(energy_source_type="source1"),
                "storage2": MagicMock(energy_source_type="source2"),
                "storage3": MagicMock(energy_source_type="source2"),
            },
            {
                "source1": MagicMock(generation_load_method="other"),
                "source2": MagicMock(generation_load_method="milp"),
            },
            {
                (1, 1): "binary_var_MILP_bin",
                (2, 1): "binary_var_MILP_bin",
            },
            id="one_storage_not_matching",
        ),
    ],
)
def test_create_storage_binary_variables(
    storage_mapping: dict[int, str],
    storage_type_inverse: dict[str, int],
    storages: dict[str, MagicMock],
    storage_types: dict[str, MagicMock],
    expected_output: dict[tuple[int, int], str],
) -> None:
    model = MagicMock()
    model.add_variables.return_value = "binary_var_MILP_bin"
    indices = MagicMock()
    indices.STOR.mapping = storage_mapping
    indices.TSTOR.inverse = storage_type_inverse

    network = MagicMock()
    network.storages = storages
    for name, mock in storage_types.items():
        mock.configure_mock(name=name)
    network.storage_types = storage_types

    st_variables = StorageVariables.__new__(StorageVariables)
    result = st_variables._create_storage_binary_variables(
        model, indices, network, AllowedStorageGenerationLoadMethods.milp
    )

    assert result == expected_output


@pytest.mark.parametrize(
    "storages, storage_types",
    [
        pytest.param(
            {},
            {
                "source1": MagicMock(generation_load_method="milp"),
                "source2": MagicMock(generation_load_method="milp"),
            },
            id="no_storage_elements",
        ),
        pytest.param(
            {
                "storage1": MagicMock(energy_source_type="source1"),
                "storage2": MagicMock(energy_source_type="source2"),
                "storage3": MagicMock(energy_source_type="source2"),
            },
            {},
            id="no_storage_type_elements",
        ),
    ],
)
def test_create_storage_binary_variables_empty_network_elements(
    storages: dict[str, MagicMock],
    storage_types: dict[str, MagicMock],
) -> None:
    model = MagicMock()
    model.add_variables.return_value = "binary_var_MILP_bin"
    indices = MagicMock()
    indices.STOR.mapping = {0: "storage1", 1: "storage2"}
    indices.TSTOR.inverse = {"source1": 0, "source2": 1}

    network = MagicMock()
    network.storages = storages
    for name, mock in storage_types.items():
        mock.configure_mock(name=name)
    network.storage_types = storage_types

    st_variables = StorageVariables.__new__(StorageVariables)
    with pytest.raises(KeyError):
        st_variables._create_storage_binary_variables(
            model, indices, network, AllowedStorageGenerationLoadMethods.milp
        )
