import numpy as np
import pandas as pd
import pytest

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.network import Network
from pyzefir.model.network_elements import EnergySourceType
from tests.unit.defaults import ELECTRICITY, HEATING, default_network_constants
from tests.unit.model.test_network_elements.helpers import assert_same_exception_list


class EnergySourceTypeTest(EnergySourceType):
    def validate(self, network: Network) -> None:
        pass


@pytest.fixture
def network() -> Network:
    network = Network(
        energy_types=[ELECTRICITY, HEATING],
        network_constants=default_network_constants,
        emission_types=["CO2", "PM10"],
    )
    return network


@pytest.fixture()
def test_energy_source_type() -> EnergySourceTypeTest:
    return EnergySourceTypeTest(
        name="test",
        life_time=5,
        capex=pd.Series([0] * default_network_constants.n_years),
        opex=pd.Series([0] * default_network_constants.n_years),
        build_time=123,
        min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        min_capacity_increase=pd.Series([np.nan] * default_network_constants.n_years),
        max_capacity_increase=pd.Series([np.nan] * default_network_constants.n_years),
    )


@pytest.mark.parametrize(
    "params, exception_list",
    [
        pytest.param(
            {},
            [],
            id="valid",
        ),
        pytest.param(
            {
                "build_time": "123",
            },
            [
                NetworkValidatorException(
                    "Invalid build_time. Build_time must be an integer, not str"
                )
            ],
            id="invalid build_time",
        ),
        pytest.param(
            {
                "min_capacity": pd.Series([1] * default_network_constants.n_years),
                "max_capacity": pd.Series([1] * default_network_constants.n_years),
                "min_capacity_increase": pd.Series(
                    [1] * default_network_constants.n_years
                ),
                "max_capacity_increase": pd.Series(
                    [1] * default_network_constants.n_years
                ),
            },
            [
                NetworkValidatorException(
                    "Min_capacity must have a NaN value for the base year"
                ),
                NetworkValidatorException(
                    "Max_capacity must have a NaN value for the base year"
                ),
                NetworkValidatorException(
                    "Min_capacity_increase must have a NaN value for the base year"
                ),
                NetworkValidatorException(
                    "Max_capacity_increase must have a NaN value for the base year"
                ),
            ],
            id="not nan for first year",
        ),
        pytest.param(
            {
                "tags": ["abc", 12, False],
            },
            [NetworkValidatorException("Invalid tags: ['abc', 12, False]. ")],
            id="incorrect tags values",
        ),
        pytest.param(
            {
                "tags": ("example_t_tag_1", "example_t_tag_2"),
            },
            [
                NetworkValidatorException(
                    "Invalid tags: ('example_t_tag_1', 'example_t_tag_2'). "
                )
            ],
            id="incorrect tags type",
        ),
    ],
)
def test_validate_base_energy_source(
    network: Network,
    test_energy_source_type: EnergySourceTypeTest,
    params: dict,
    exception_list: list[NetworkValidatorException],
) -> None:
    for key, value in params.items():
        setattr(test_energy_source_type, key, value)

    actual_exception_list: list[NetworkValidatorException] = []
    test_energy_source_type._validate_energy_source_type_base(
        network, actual_exception_list
    )
    assert_same_exception_list(actual_exception_list, exception_list)
