# PyZefir
# Copyright (C) 2023-2024 Narodowe Centrum Badań Jądrowych
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from pathlib import Path
from typing import Type
from unittest import mock

import pytest

from pyzefir.utils.path_manager import (
    CsvPathManager,
    CsvPathManagerException,
    DataCategories,
    DataCategoriesException,
    DataSubCategories,
    XlsxPathManager,
    get_datasets_from_categories,
)


@pytest.fixture
def csv_path_manager() -> CsvPathManager:
    return CsvPathManager(Path("tmp/data"))


@pytest.fixture
def xlsx_path_manager() -> XlsxPathManager:
    return XlsxPathManager(Path("tmp/input_data"), Path("tmp/output_data"))


def test_DataCategories_get_main_categories() -> None:
    expected = [
        "initial_state",
        "structure",
        "capacity_factors",
        "demand_chunks",
        "fuels",
        "generator_types",
        "storage_types",
        "demand_types",
        "scenarios",
        "conversion_rate",
    ]
    assert set(DataCategories.get_main_categories()) == set(expected)


def test_DataCategories_get_dynamic_categories() -> None:
    expected = [
        "demand_types",
        "conversion_rate",
        "demand_chunks",
    ]
    assert DataCategories.get_dynamic_categories() == expected


@pytest.mark.parametrize(
    "data_category, expected",
    [
        (
            "fuels",
            [
                DataSubCategories.EMISSION_PER_UNIT,
                DataSubCategories.ENERGY_PER_UNIT,
            ],
        ),
        (
            "capacity_factors",
            [
                DataSubCategories.PROFILES,
            ],
        ),
        (
            "structure",
            [
                DataSubCategories.ENERGY_TYPES,
                DataSubCategories.EMISSION_TYPES,
                DataSubCategories.AGGREGATES,
                DataSubCategories.LINES,
                DataSubCategories.BUSES,
                DataSubCategories.GENERATORS,
                DataSubCategories.STORAGES,
                DataSubCategories.TECHNOLOGYSTACKS_BUSES_OUT,
                DataSubCategories.TECHNOLOGY_BUS,
                DataSubCategories.TECHNOLOGYSTACK_BUSES,
                DataSubCategories.TECHNOLOGYSTACK_AGGREGATE,
                DataSubCategories.TRANSMISSION_FEES,
                DataSubCategories.EMISSION_FEES_EMISSION_TYPES,
                DataSubCategories.GENERATOR_EMISSION_FEES,
            ],
        ),
        (
            "generator_types",
            [
                DataSubCategories.GENERATOR_TYPES,
                DataSubCategories.EFFICIENCY,
                DataSubCategories.EMISSION_REDUCTION,
                DataSubCategories.GENERATOR_TYPE_ENERGY_CARRIER,
                DataSubCategories.GENERATOR_TYPE_ENERGY_TYPE,
            ],
        ),
        (
            "initial_state",
            [DataSubCategories.TECHNOLOGY, DataSubCategories.TECHNOLOGYSTACK],
        ),
    ],
)
def test_get_datasets_from_categories(data_category: str, expected: str) -> None:
    assert get_datasets_from_categories(data_category) == expected


@pytest.mark.parametrize(
    "data_category", ["initial_states", "restructure", "generator_type"]
)
def test_get_datasets_from_categories_wrong_categories(data_category: str) -> None:
    with mock.patch("logging.Logger.warning") as mock_logger:
        with pytest.raises(KeyError):
            get_datasets_from_categories(data_category)
        mock_logger.assert_called_with(
            f"{data_category=} not in datasets_in_categories keys"
        )


@pytest.mark.parametrize(
    "data_category", ["initial_state", "structure", "generator_types"]
)
def test_data_categories_get_directory(data_category: str) -> None:
    DataCategories.check_directory_name(data_category)


@pytest.mark.parametrize(
    "data_category", ["invalid_category", "structures_value", "hvdc_types"]
)
def test_data_categories_get_directory_raises_value_error(data_category: str) -> None:
    with pytest.raises(DataCategoriesException):
        DataCategories.check_directory_name(data_category)


def test_csv_path_manager_dir_path_logger(csv_path_manager: CsvPathManager) -> None:
    expected_path = Path("tmp/data")
    with mock.patch("logging.Logger.debug") as mock_logger:
        assert csv_path_manager.dir_path == expected_path
        mock_logger.assert_called_with(f"Csv root dir path is {expected_path}")


@pytest.mark.parametrize(
    "data_category, dataset_name",
    [
        ("fuels", "Emission_Per_Unit"),
        ("capacity_factors", "Profiles"),
        ("structure", "Lines"),
        ("generator_types", "Emission_Reduction"),
        ("initial_state", "Technology"),
        ("structure", "Emission_Fees_Emission_Types"),
    ],
)
def test_csv_path_manager_get_file_name_from_dict(
    data_category: str, dataset_name: str
) -> None:
    CsvPathManager._get_file_name_from_dict(data_category, dataset_name)
    file_name = CsvPathManager._get_file_name_from_dict(data_category, dataset_name)
    assert file_name == f"{dataset_name}.csv"


@pytest.mark.parametrize(
    "data_category, dataset_name",
    [
        ("demand_types", "small_house"),
        ("invalid_category", "invalid_file"),
        ("fuels_types", "invalid_file"),
        ("fuelss_types", "fuelinos"),
        ("structure", "EmissionFees_EmissionTypes"),
    ],
)
def test_csv_path_manager_get_file_name_from_dict_errors(
    data_category: str, dataset_name: str
) -> None:
    with pytest.raises(CsvPathManagerException):
        CsvPathManager._get_file_name_from_dict(data_category, dataset_name)


@pytest.mark.parametrize(
    "data_category, dataset_name, expected",
    [
        ("initial_state", "", "tmp/data/initial_state"),
        ("initial_state", "Technology", "tmp/data/initial_state/Technology.csv"),
        ("structure", "", "tmp/data/structure"),
        ("structure", "Lines", "tmp/data/structure/Lines.csv"),
        ("structure", "Technology_Bus", "tmp/data/structure/Technology_Bus.csv"),
        ("scenarios", "Cost_Parameters", "tmp/data/scenarios/Cost_Parameters.csv"),
        ("scenarios", "Fractions", "tmp/data/scenarios/Fractions.csv"),
        ("scenarios", "Emission_Fees", "tmp/data/scenarios/Emission_Fees.csv"),
    ],
)
def test_csv_path_manager_get_path_(
    csv_path_manager: CsvPathManager,
    data_category: str,
    dataset_name: str,
    expected: str,
) -> None:
    if dataset_name:
        path = csv_path_manager.get_path(data_category, dataset_name)
    else:
        path = csv_path_manager.get_path(data_category)
    assert path == Path(expected)


@pytest.mark.parametrize(
    "dataset_name, scenario_name, expected",
    [
        (
            "Emission_Fees",
            "scenario_3",
            "tmp/data/scenarios/scenario_3/Emission_Fees.csv",
        ),
        ("Fuel_Prices", "scenario_2", "tmp/data/scenarios/scenario_2/Fuel_Prices.csv"),
        ("Cost_Parameters", "sc_414", "tmp/data/scenarios/sc_414/Cost_Parameters.csv"),
    ],
)
def test_csv_path_manager_get_path_scenario_name(
    csv_path_manager: CsvPathManager,
    scenario_name: str,
    dataset_name: str,
    expected: str,
) -> None:
    csv_path_manager._scenario_name = scenario_name
    path = csv_path_manager.get_path(
        data_category="scenarios", dataset_name=dataset_name
    )
    assert path == Path(expected)


@pytest.mark.parametrize(
    "data_category, dataset_name, error",
    [
        ("demand_types", "small_flat", CsvPathManagerException),
        ("initial_state", "lines", CsvPathManagerException),
        ("generator_types", "Emission_Per_Element", CsvPathManagerException),
        ("fuels", "emission per unit", CsvPathManagerException),
        ("structure_types", "", DataCategoriesException),
        ("invalid_category", "", DataCategoriesException),
    ],
)
def test_csv_path_manager_get_path_errors(
    csv_path_manager: CsvPathManager,
    data_category: str,
    dataset_name: str,
    error: Type[Exception],
) -> None:
    with pytest.raises(error):
        if dataset_name:
            csv_path_manager.get_path(data_category, dataset_name)
        else:
            csv_path_manager.get_path(data_category)


@pytest.mark.parametrize(
    "category, dataset_name, expected",
    [
        ("demand_types", "small_house", "tmp/data/demand_types/small_house.csv"),
        ("demand_types", "flat", "tmp/data/demand_types/flat.csv"),
        ("demand_types", "bedsit", "tmp/data/demand_types/bedsit.csv"),
        ("conversion_rate", "HEAT_PUMP", "tmp/data/conversion_rate/HEAT_PUMP.csv"),
        ("conversion_rate", "FAST_PUMP", "tmp/data/conversion_rate/FAST_PUMP.csv"),
    ],
)
def test_csv_path_manager_concatenate_path_for_dynamic_dataset_name(
    csv_path_manager: CsvPathManager,
    category: str,
    dataset_name: str,
    expected: str,
) -> None:
    file_path = csv_path_manager.concatenate_path_for_dynamic_dataset_name(
        category, dataset_name
    )
    assert file_path == Path(expected)


def test_xlsx_path_manager_input_path(xlsx_path_manager: XlsxPathManager) -> None:
    expected_path = Path("tmp/input_data")
    with mock.patch("logging.Logger.debug") as mock_logger:
        assert xlsx_path_manager.input_path == expected_path
        mock_logger.assert_called_with(f"Input path is {expected_path}")


def test_xlsx_path_manager_output_path(xlsx_path_manager: XlsxPathManager) -> None:
    expected_path = Path("tmp/output_data")
    with mock.patch("logging.Logger.debug") as mock_logger:
        assert xlsx_path_manager.output_path == expected_path
        mock_logger.assert_called_with(f"Output path is {expected_path}")


@pytest.mark.parametrize(
    "data_category, expected",
    [
        ("initial_state", "tmp/input_data/initial_state.xlsx"),
        ("structure", "tmp/input_data/structure.xlsx"),
        ("fuels", "tmp/input_data/fuels.xlsx"),
        ("capacity_factors", "tmp/input_data/capacity_factors.xlsx"),
        ("generator_types", "tmp/input_data/generator_types.xlsx"),
        ("storage_types", "tmp/input_data/storage_types.xlsx"),
        ("demand_types", "tmp/input_data/demand_types.xlsx"),
    ],
)
def test_xlsx_path_manager_get_input_file_path(
    xlsx_path_manager: XlsxPathManager, data_category: str, expected: str
) -> None:
    path = xlsx_path_manager.get_input_file_path(data_category)
    assert path == Path(expected)


@pytest.mark.parametrize(
    "data_category",
    ["initial_states", "generator_fuel", "destructure", "demand_typies"],
)
def test_xlsx_path_manager_get_input_file_path_errors(
    xlsx_path_manager: XlsxPathManager, data_category: str
) -> None:
    with pytest.raises(DataCategoriesException):
        xlsx_path_manager.get_input_file_path(data_category)


def test_xlsx_path_manager_get_path(xlsx_path_manager: XlsxPathManager) -> None:
    expected_path = Path("tmp/output_data/initial_state/Technology.csv")
    with mock.patch("logging.Logger.debug") as mock_logger:
        path = xlsx_path_manager.get_path("initial_state", "Technology")
        assert path == expected_path
        mock_logger.assert_called_with(
            f"File Technology is at the path: {expected_path}"
        )


def test_xlsx_path_manager_get_path_folder_only(
    xlsx_path_manager: XlsxPathManager,
) -> None:
    expected_path = Path("tmp/output_data/initial_state")
    with mock.patch("logging.Logger.debug") as mock_logger:
        path = xlsx_path_manager.get_path("initial_state")
        assert path == expected_path
        mock_logger.assert_called_with(f"Given folder is at the path: {expected_path}")
