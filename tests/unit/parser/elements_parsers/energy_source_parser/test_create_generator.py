from typing import Any

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.network_elements import Generator
from pyzefir.parser.elements_parsers.energy_source_unit_parser import (
    EnergySourceUnitParser,
)
from tests.unit.defaults import default_network_constants


@pytest.mark.parametrize(
    ("gen_name", "buses"),
    [
        pytest.param("GENERATOR_2", {"BLE", "BLE_2"}, id="two_buses_case"),
    ],
)
def test_get_set_of_buses_from_dataframe(
    gen_name: str,
    buses: set[str],
    energy_source_unit_parser: EnergySourceUnitParser,
) -> None:
    """Test if buses are extracted correctly from dataframe mock."""
    assert energy_source_unit_parser._get_set_of_buses_from_dataframe(gen_name) == buses


@pytest.mark.parametrize(
    ("gen_name", "bus"),
    [
        pytest.param("GENERATOR_1", "BLE", id="single_bus_case"),
    ],
)
def test_get_bus_from_dataframe(
    gen_name: str,
    bus: str,
    energy_source_unit_parser: EnergySourceUnitParser,
) -> None:
    """Test if bus has extracted correctly from dataframe mock."""
    assert energy_source_unit_parser._get_bus_from_dataframe(gen_name) == bus


@pytest.mark.parametrize(
    ("gen_idx", "expected_emission_fee"),
    [
        pytest.param(
            0, {"EF_3", "EF_1", "EF_2"}, id="check_if_EF_3_EF_1_EF_2_in_gen_1"
        ),
        pytest.param(1, {"EF_1"}, id="check_if_EF_1_in_gen_2"),
        pytest.param(2, set(), id="check_if_gen_3_has_no_EF"),
    ],
)
def test_create_generator_emission_fees(
    energy_source_unit_parser: EnergySourceUnitParser,
    gen_idx: int,
    expected_emission_fee: set,
) -> None:
    gen = energy_source_unit_parser._create_generator(
        energy_source_unit_parser.df_generators.iloc[gen_idx, :]
    )
    assert gen.emission_fee == expected_emission_fee


def test_create_generator_no_optional_params(
    energy_source_unit_parser: EnergySourceUnitParser,
) -> None:
    """Test if generator will be created correctly with no optional parameters provided."""
    gen = energy_source_unit_parser._create_generator(
        energy_source_unit_parser.df_generators.iloc[0, :]
    )

    assert isinstance(gen, Generator)
    assert gen.name == "GENERATOR_1"
    assert gen.buses == {"BLE"}
    assert gen.energy_source_type == "GEN_TYPE_1"
    assert gen.emission_fee == {"EF_3", "EF_1", "EF_2"}
    assert gen.unit_base_cap == 35000
    assert pd.Series([None] * default_network_constants.n_years).equals(
        gen.unit_min_capacity_increase
    )
    assert pd.Series([None] * default_network_constants.n_years).equals(
        gen.unit_max_capacity_increase
    )
    assert pd.Series([None] * default_network_constants.n_years).equals(
        gen.unit_max_capacity
    )
    assert pd.Series([None] * default_network_constants.n_years).equals(
        gen.unit_min_capacity
    )


@pytest.mark.parametrize(
    ("gen_name", "expected_params"),
    [
        pytest.param(
            "GENERATOR_1",
            {
                "unit_min_capacity": pd.Series([np.nan] * 4),
                "unit_max_capacity": pd.Series([np.nan] * 4),
                "unit_min_capacity_increase": pd.Series([3, 4, 5, 2]),
                "unit_max_capacity_increase": pd.Series([np.nan] * 4),
                "min_device_nom_power": 1.0,
                "max_device_nom_power": 10.0,
            },
            id="min_capacity_increase_only",
        ),
        pytest.param(
            "GENERATOR_2",
            {
                "unit_min_capacity": pd.Series([1, 1, 1, 1]),
                "unit_max_capacity": pd.Series([1, 1, 1, 1]),
                "unit_min_capacity_increase": pd.Series([np.nan] * 4),
                "unit_max_capacity_increase": pd.Series([2, 2, 2, 2]),
                "min_device_nom_power": 5,
                "max_device_nom_power": 15,
            },
            id="all_but_unit_delta_cap_max",
        ),
        pytest.param(
            "GENERATOR_3",
            {
                "unit_min_capacity": pd.Series([np.nan] * 4),
                "unit_max_capacity": pd.Series([np.nan] * 4),
                "unit_min_capacity_increase": pd.Series([np.nan] * 4),
                "unit_max_capacity_increase": pd.Series([np.nan] * 4),
                "min_device_nom_power": None,
                "max_device_nom_power": None,
            },
            id="all_empty",
        ),
    ],
)
def test_create_generator_with_optional_parameters(
    gen_name: str,
    expected_params: dict[str, pd.Series | None],
    energy_source_unit_parser: EnergySourceUnitParser,
    technology_evolution_mock_df: pd.DataFrame,
) -> None:
    """Test if generator will be created correctly for varius configuration of given / non given optional params."""
    energy_source_unit_parser.df_element_energy_evolution = technology_evolution_mock_df
    gen_df = energy_source_unit_parser.df_generators
    gen = energy_source_unit_parser._create_generator(
        gen_df[gen_df["name"] == gen_name].squeeze()
    )

    assert gen.name == gen_name
    for param_name, expected_value in expected_params.items():
        compare_values(value=getattr(gen, param_name), expected=expected_value)


def compare_values(value: Any, expected: Any) -> None:
    if expected is None:
        assert value is None
    elif isinstance(expected, (pd.Series, pd.DataFrame, np.ndarray)):
        if pd.isna(expected).all():
            assert pd.isna(value).all()
        else:
            assert np.all(value == expected)
    else:
        assert value == expected
