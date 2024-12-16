from pathlib import Path

import pandas as pd
import pytest

from pyzefir.model.network_elements import Fuel
from pyzefir.parser.elements_parsers.fuel_parser import FuelParser
from pyzefir.utils.path_manager import DataCategories, DataSubCategories


@pytest.fixture
def emission_per_unit_df(csv_root_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_root_path
        / f"{DataCategories.FUELS}"
        / f"{DataSubCategories.EMISSION_PER_UNIT}.csv"
    )


@pytest.fixture
def energy_per_unit_df(csv_root_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_root_path
        / f"{DataCategories.FUELS}"
        / f"{DataSubCategories.ENERGY_PER_UNIT}.csv"
    )


@pytest.fixture
def fuel_prices_df(csv_root_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_root_path
        / f"{DataCategories.SCENARIO}/scenario_1"
        / f"{DataSubCategories.FUEL_PRICES}.csv"
    )


@pytest.fixture
def fuel_availability_df(csv_root_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_root_path
        / f"{DataCategories.SCENARIO}/scenario_1"
        / f"{DataSubCategories.FUEL_AVAILABILITY}.csv"
    )


@pytest.fixture
def emission_types() -> set[str]:
    return {"CO2", "SO2"}


@pytest.fixture
def fuel_parser(
    emission_per_unit_df: pd.DataFrame,
    energy_per_unit_df: pd.DataFrame,
    fuel_prices_df: pd.DataFrame,
    fuel_availability_df: pd.DataFrame,
    emission_types: set[str],
) -> FuelParser:
    return FuelParser(
        emission_per_unit_df=emission_per_unit_df,
        energy_per_unit_df=energy_per_unit_df,
        fuel_prices_df=fuel_prices_df,
        fuel_availability_df=fuel_availability_df,
    )


def test_fuel_parser_init(fuel_parser: FuelParser) -> None:
    assert isinstance(fuel_parser.fuel_availability_df, pd.DataFrame)
    assert isinstance(fuel_parser.fuel_prices_df, pd.DataFrame)
    assert isinstance(fuel_parser.energy_per_unit_df, pd.DataFrame)
    assert isinstance(fuel_parser.emission_per_unit_df, pd.DataFrame)


def test_create(
    fuel_parser: FuelParser,
    energy_per_unit_df: pd.DataFrame,
    emission_types: set[str],
    emission_per_unit_df: pd.DataFrame,
    fuel_availability_df: pd.DataFrame,
    fuel_prices_df: pd.DataFrame,
) -> None:
    fuels = fuel_parser.create()

    assert all(isinstance(fuel, Fuel) for fuel in fuels)
    assert all(fuel.emission.keys() == emission_types for fuel in fuels)
    assert all(isinstance(fuel.availability, pd.Series) for fuel in fuels)
    assert all(isinstance(fuel.cost, pd.Series) for fuel in fuels)
    assert all(isinstance(fuel.energy_per_unit, float) for fuel in fuels)
    # check if number of fuel elements is equal to number of rows in source DataFrames
    assert (
        len(
            {
                len(fuels),
                len(energy_per_unit_df),
                len(emission_per_unit_df),
                len(fuel_prices_df.columns) - 1,
                len(fuel_availability_df.columns) - 1,
            }
        )
        == 1
    )


def test_fuel_parser_create_from_df(
    emission_types: set[str],
    fuel_prices_df: pd.DataFrame,
    fuel_availability_df: pd.DataFrame,
    energy_per_unit_df: pd.DataFrame,
    emission_per_unit_df: pd.DataFrame,
) -> None:
    row = FuelParser._merge_energy_emission_data(
        energy_per_unit_df, emission_per_unit_df
    ).iloc[1]
    name = row.name
    emission = {e_type: row[e_type] for e_type in emission_types}
    availability = fuel_availability_df[name]
    cost = fuel_prices_df[name]
    energy_per_unit = row["energy_per_unit"]

    fuel = FuelParser._create_fuel(row, fuel_prices_df, fuel_availability_df)

    assert isinstance(fuel, Fuel)
    assert fuel.name == name
    assert fuel.emission == emission
    assert fuel.availability.equals(availability)
    assert fuel.cost.equals(cost)
    assert fuel.energy_per_unit == energy_per_unit


def test_fuel_parser_create(fuel_parser: FuelParser) -> None:
    fuels = fuel_parser.create()
    assert isinstance(fuels, tuple)
    assert all(isinstance(fuel, Fuel) for fuel in fuels)


@pytest.mark.parametrize(
    "energy_df, emission_df, raise_exception",
    (
        (
            pd.DataFrame({"name": [], "energy_per_unit": []}),
            pd.DataFrame({"name": [], "emission_per_unit": []}),
            False,
        ),
        (
            pd.DataFrame({"name": ["a", "b"], "energy_per_unit": [1, 2]}),
            pd.DataFrame({"name": ["a", "c"], "emission_per_unit": [3, 4]}),
            True,
        ),
        (
            pd.DataFrame({"name": ["a", "b"], "energy_per_unit": [1, 2]}),
            pd.DataFrame({"name": ["c"], "emission_per_unit": [3]}),
            True,
        ),
        (
            pd.DataFrame({"name": ["a", "b"], "energy_per_unit": [1, 2]}),
            pd.DataFrame({"name": ["a", "b"], "emission_per_unit": [3, 4]}),
            False,
        ),
    ),
)
def test_merge_incomplete_data(
    energy_df: pd.DataFrame, emission_df: pd.DataFrame, raise_exception: bool
) -> None:
    if raise_exception:
        with pytest.raises(ValueError):
            FuelParser._merge_energy_emission_data(energy_df, emission_df)
    else:
        fuel_df = FuelParser._merge_energy_emission_data(energy_df, emission_df)
        assert fuel_df.isna().sum().sum() == 0
