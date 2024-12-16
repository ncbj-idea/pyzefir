from typing import Any

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.network import Network
from pyzefir.model.network_elements.energy_source_types.storage_type import StorageType
from pyzefir.model.network_elements.energy_sources.storage import Storage
from tests.unit.defaults import default_network_constants, get_default_storage_type
from tests.unit.model.test_network_elements.helpers import assert_same_exception_list


@pytest.mark.parametrize(
    "storage_type, exception_value",
    [
        pytest.param(
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                generation_efficiency="string",
            ),
            "StorageType attribute 'generation_efficiency' for default must be an instance "
            "of float | int, but it is an instance of <class 'str'> instead",
            id="generation_efficiency_wrong_type",
        ),
        pytest.param(
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                load_efficiency="string",
            ),
            "StorageType attribute 'load_efficiency' for default must be an instance "
            "of float | int, but it is an instance of <class 'str'> instead",
            id="load_efficiency_wrong_type",
        ),
        pytest.param(
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                cycle_length="string",
            ),
            "StorageType attribute 'cycle_length' for default must be an instance "
            "of int | None, but it is an instance of <class 'str'> instead",
            id="cycle_length_wrong_type",
        ),
        pytest.param(
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                power_to_capacity="string",
            ),
            "StorageType attribute 'power_to_capacity' for default must be an instance "
            "of float | int, but it is an instance of <class 'str'> instead",
            id="power_to_capacity_wrong_type",
        ),
        pytest.param(
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                energy_type=1,
            ),
            "StorageType attribute 'energy_type' for default must be an instance "
            "of <class 'str'>, but it is an instance of <class 'int'> instead",
            id="energy_type_wrong_type",
        ),
        pytest.param(
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                generation_load_method=1.12,
            ),
            "StorageType attribute 'generation_load_method' for default must be an instance "
            "of str | None, but it is an instance of <class 'float'> instead",
            id="generation_load_method_wrong_type",
        ),
    ],
)
def test_attribute_types_validators(
    storage_type: StorageType,
    exception_value: str,
    network_fixture: Network,
) -> None:
    with pytest.raises(NetworkValidatorException) as e_info:
        storage_type.validate(network_fixture)
    assert str(e_info.value) == exception_value


@pytest.mark.parametrize(
    "storage_type, storage, expected_exception_list",
    [
        pytest.param(
            None,
            Storage(
                name="storage_a",
                bus="bus_A",
                energy_source_type="test_storage_type",
                unit_base_cap=15,
                unit_min_capacity=pd.Series(
                    index=range(default_network_constants.n_years), data=np.nan
                ),
                unit_max_capacity=pd.Series(
                    index=range(default_network_constants.n_years), data=np.nan
                ),
                unit_min_capacity_increase=pd.Series(
                    index=range(default_network_constants.n_years), data=np.nan
                ),
                unit_max_capacity_increase=pd.Series(
                    index=range(default_network_constants.n_years), data=np.nan
                ),
            ),
            [
                NetworkValidatorException(
                    "Storage type test_storage_type not found in the network"
                ),
            ],
            id="test_storage_type not in the network",
        ),
        pytest.param(
            "string",
            Storage(
                name="storage_a",
                bus="bus_A",
                energy_source_type="test_storage_type",
                unit_base_cap=15,
                unit_min_capacity=pd.Series(
                    index=range(default_network_constants.n_years), data=np.nan
                ),
                unit_max_capacity=pd.Series(
                    index=range(default_network_constants.n_years), data=np.nan
                ),
                unit_min_capacity_increase=pd.Series(
                    index=range(default_network_constants.n_years), data=np.nan
                ),
                unit_max_capacity_increase=pd.Series(
                    index=range(default_network_constants.n_years), data=np.nan
                ),
            ),
            [
                NetworkValidatorException(
                    "Storage type must be of type StorageType, but it is <class 'str'> instead."
                )
            ],
            id="storage_type not instance of StorageType",
        ),
    ],
)
def test_storage_type_instance(
    storage_type: Any,
    storage: Storage,
    expected_exception_list: list[NetworkValidatorException],
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    storage._validate_storage_type(storage_type, actual_exception_list)
    assert_same_exception_list(actual_exception_list, expected_exception_list)


@pytest.mark.parametrize(
    "storage_type, exception_value",
    [
        pytest.param(
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                energy_type="COLD",
            ),
            "Energy type COLD is not compliant with the network "
            "energy types: ['ELECTRICITY', 'HEATING']",
            id="energy_type_not_in_network_COLD",
        ),
        pytest.param(
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                energy_type="HEAT_USAGE",
            ),
            "Energy type HEAT_USAGE is not compliant with the network "
            "energy types: ['ELECTRICITY', 'HEATING']",
            id="energy_type_not_in_network_HEAT_USAGE",
        ),
        pytest.param(
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                energy_type="ELECT",
            ),
            "Energy type ELECT is not compliant with the network "
            "energy types: ['ELECTRICITY', 'HEATING']",
            id="energy_type_not_in_network_ELECT",
        ),
    ],
)
def test_storage_type_energy_type(
    storage_type: StorageType,
    exception_value: str,
    network_fixture: Network,
) -> None:
    with pytest.raises(NetworkValidatorException) as e_info:
        storage_type.validate(network_fixture)
    assert str(e_info.value.args[1][0]) == exception_value


@pytest.mark.parametrize(
    "storage_type, exception_value",
    [
        pytest.param(
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                generation_efficiency=1.01,
            ),
            "The value of the generation_efficiency is inconsistent with th expected bounds of the "
            "interval: 0 <= 1.01 <= 1",
            id="generation_efficiency_above_upper_bound",
        ),
        pytest.param(
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                load_efficiency=-0.23,
            ),
            "The value of the load_efficiency is inconsistent with th expected bounds of the "
            "interval: 0 <= -0.23 <= 1",
            id="load_efficiency_below_lower_bound",
        ),
        pytest.param(
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                power_utilization=1.45,
            ),
            "The value of the power_utilization is inconsistent with th expected bounds of the "
            "interval: 0 <= 1.45 <= 1",
            id="power_utilization_above_upper_bound",
        ),
        pytest.param(
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                energy_loss=-0.01,
            ),
            "The value of the energy_loss is inconsistent with th expected bounds of the "
            "interval: 0 <= -0.01 <= 1",
            id="energy_loss_below_lower_bound",
        ),
    ],
)
def test_storage_type_attributes_values_check_interval(
    storage_type: StorageType,
    exception_value: str,
    network_fixture: Network,
) -> None:
    with pytest.raises(NetworkValidatorException) as e_info:
        storage_type.validate(network_fixture)
    assert str(e_info.value.args[1][0]) == exception_value


def test_storage_type_happy_path(network_fixture: Network) -> None:
    storage_type: StorageType = get_default_storage_type(
        series_length=default_network_constants.n_years
    )
    storage_type.validate(network_fixture)


@pytest.mark.parametrize(
    "storage_type, exception_value",
    [
        pytest.param(
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                generation_load_method="multi",
            ),
            "The value of the generation_load_method multi is inconsistent with allowed values: ['milp']",
            id="generation_load_method_multi_value",
        ),
        pytest.param(
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                generation_load_method="None",
            ),
            "The value of the generation_load_method None is inconsistent with allowed values: ['milp']",
            id="generation_load_method_None_value",
        ),
        pytest.param(
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                generation_load_method="mipl",
            ),
            "The value of the generation_load_method mipl is inconsistent with allowed values: ['milp']",
            id="generation_load_method_mipl_value",
        ),
    ],
)
def test_storage_type_generation_load_method_value(
    storage_type: StorageType,
    exception_value: str,
    network_fixture: Network,
) -> None:
    with pytest.raises(NetworkValidatorException) as e_info:
        storage_type.validate(network_fixture)
    assert str(e_info.value.args[1][0]) == exception_value
