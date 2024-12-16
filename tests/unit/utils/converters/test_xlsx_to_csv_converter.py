import tempfile
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
import pytest

from pyzefir.parser.validator.dataframe_validator import DataFrameValidatorException
from pyzefir.parser.validator.valid_structure import (
    DatasetConfig,
    InvalidStructureException,
)
from pyzefir.utils.converters.xlsx_to_csv_converter import (
    ExcelToCsvConverter,
    ExcelToCsvConverterException,
)
from pyzefir.utils.path_manager import DataCategories
from tests.utils import get_resources


@pytest.fixture
def converter() -> ExcelToCsvConverter:
    return ExcelToCsvConverter(
        input_files_path=Path("/tmp/input/"), output_files_path=Path("/tmp/output/")
    )


def test_converter_init() -> None:
    converter = ExcelToCsvConverter(
        input_files_path=Path("/input"), output_files_path=Path("/output")
    )
    assert converter.path_manager.input_path == Path("/input")
    assert converter.path_manager.output_path == Path("/output")
    assert converter._scenario_path is None


@pytest.mark.parametrize(
    "xlsx_df_dict, expected",
    [
        (
            {
                "Sheet - 1__test": pd.DataFrame({"A": [1, 2]}),
                "Sheet1": pd.DataFrame({"A": [1, 2]}),
            },
            {
                "Sheet_1_test": pd.DataFrame({"A": [1, 2]}),
                "Sheet1": pd.DataFrame({"A": [1, 2]}),
            },
        ),
        (
            {
                "Generator Types": pd.DataFrame({"A": [1, 2]}),
                "Generator Type - Energy Carrier": pd.DataFrame({"A": [1, 2]}),
                "Generator Type - Energy Type": pd.DataFrame({"A": [1, 2]}),
                "Energy Source Evolution Limits": pd.DataFrame({"A": [1, 2]}),
            },
            {
                "Generator_Types": pd.DataFrame({"A": [1, 2]}),
                "Generator_Type_Energy_Carrier": pd.DataFrame({"A": [1, 2]}),
                "Generator_Type_Energy_Type": pd.DataFrame({"A": [1, 2]}),
                "Energy_Source_Evolution_Limits": pd.DataFrame({"A": [1, 2]}),
            },
        ),
    ],
)
def test_handle_sheet_name_replaces_characters(
    converter: ExcelToCsvConverter,
    xlsx_df_dict: dict[str, pd.DataFrame],
    expected: dict[str, pd.DataFrame],
) -> None:
    new_xlsx_df_dict = converter._sanitize_spreadsheets_names(xlsx_df_dict)
    assert new_xlsx_df_dict.keys() == expected.keys()


@pytest.mark.parametrize(
    "xlsx_sheets_names, category, error",
    [
        (["Emission_Per_Unit", "Energy_Per_Unit"], "fuels", None),
        (
            [
                "Profiles",
            ],
            "capacity_factors",
            None,
        ),
        (
            ["Generator_Types", "Efficiency", "Emission_Reduction"],
            "generator_types",
            True,
        ),
        (
            [
                "Energy_Source_Evolution_Limits",
                "Element_Energy_Evolution_Limits",
                "Cost_Parameters",
                "Fuel_Availability",
                "Relative_Emission_Limits",
                "Fuel_Prices",
                "Constants",
                "Technology_Evolution",
                "Yearly_Demand",
                "Fractions",
                "N_Consumers",
                "Emission_Fees",
                "Generation_Fraction",
                "Curtailment_Cost",
                "Generation_Compensation",
            ],
            "scenarios",
            None,
        ),
        (["Technology", "TechnologyStack"], "initial_state", None),
        (["Technology", "Energy_Types", "Aggregates"], "initial_state", True),
        (
            ["Fuel_Availability", "Generators", "TechnologyStack_Buses_out"],
            "fuels",
            True,
        ),
    ],
)
def test_xlsx_to_csv_converter_validate_xlsx_structure(
    converter: ExcelToCsvConverter,
    xlsx_sheets_names: list[str],
    category: str,
    error: bool | None,
) -> None:
    if error:
        with pytest.raises(ExcelToCsvConverterException):
            converter._validate_xlsx_structure(
                xlsx_sheets_names=xlsx_sheets_names, category=category
            )
    else:
        assert (
            converter._validate_xlsx_structure(
                xlsx_sheets_names=xlsx_sheets_names, category=category
            )
            is None
        )


@pytest.mark.parametrize(
    "xlsx_df_dict, category, exception",
    [
        (
            {
                "Emission_Per_Unit": pd.DataFrame(
                    {"name": ["GAS", "COAL"], "CO2": [0.1, 0.5], "PM10": [0.0, 0.7]}
                )
            },
            "fuels",
            None,
        ),
        (
            {"Energy_Types": pd.DataFrame({"name": ["solar", "electric", "heat"]})},
            "structure",
            None,
        ),
        (
            {
                "Fuel_Prices": pd.DataFrame(
                    {"year_idx": [0, 1, 2], "PV": [100, 200, 100], "BOILER": [3, 1, 52]}
                )
            },
            "scenarios",
            None,
        ),
        (
            {
                "Aggregates": pd.DataFrame(
                    {
                        "name": ["Aggr1"],
                        "demand_type": ["HEAT"],
                        "n_consumers_base": 1000,
                        "average_area": 50,
                    }
                )
            },
            "structure",
            None,
        ),
        (
            {
                "Fuel_Prices": pd.DataFrame(
                    {"year_idx": [0, 1, 2], "PV": [100, 200, 100], "BOILER": [3, 1, 52]}
                )
            },
            "scenario_name",
            InvalidStructureException,
        ),
        (
            {"Aggregates": pd.DataFrame({"name": ["Aggr1"], "demand": ["HEAT"]})},
            "structure",
            DataFrameValidatorException,
        ),
        (
            {
                "Fractions": pd.DataFrame(
                    {
                        "technology_stack": ["LKT1"],
                        "aggregate": ["DOMKI"],
                        "year": [0],
                        "min_fraction": [1],
                        "max_fraction": [1],
                        "max_fraction_increase": [1],
                        "max_fraction_decrease": [1],
                    }
                )
            },
            "scenarios",
            None,
        ),
        (
            {
                "Fractions": pd.DataFrame(
                    {
                        "technology_stack": ["LKT1"],
                        "aggregate": ["DOMKI"],
                        "year": ["12"],
                        "min_fraction": [1],
                        "max_fraction": [1],
                        "max_fraction_increase": [1],
                        "max_fraction_decrease": [1],
                    }
                )
            },
            "scenarios",
            DataFrameValidatorException,
        ),
    ],
)
def test_xlsx_to_csv_converter_validate_dataframes_structure(
    converter: ExcelToCsvConverter,
    xlsx_df_dict: dict[str, pd.DataFrame],
    category: str,
    exception: Type[Exception] | None,
) -> None:
    if exception:
        with pytest.raises(exception):
            converter._validate_dataframes_structure(
                xlsx_df_dict=xlsx_df_dict, category=category
            )
    else:
        assert (
            converter._validate_dataframes_structure(
                xlsx_df_dict=xlsx_df_dict, category=category
            )
            is None
        )


@pytest.mark.parametrize(
    "category, xlsx_path_scratch",
    [
        ("structure", Path("convert_input_structure/structure.xlsx")),
        ("fuels", Path("convert_input_structure/fuels.xlsx")),
        ("capacity_factors", Path("convert_input_structure/capacity_factors.xlsx")),
        ("demand_types", Path("convert_input_structure/demand_types.xlsx")),
        ("generator_types", Path("convert_input_structure/generator_types.xlsx")),
        ("storage_types", Path("convert_input_structure/storage_types.xlsx")),
        ("conversion_rate", Path("convert_input_structure/conversion_rate.xlsx")),
    ],
)
def test_xlsx_to_csv_converter_validate_lfs_files(
    converter: ExcelToCsvConverter, category: str, xlsx_path_scratch: Path
) -> None:
    xlsx_path = get_resources(xlsx_path_scratch)
    xlsx_df_dict = pd.read_excel(xlsx_path, sheet_name=None)
    xlsx_df_dict = converter._sanitize_spreadsheets_names(xlsx_df_dict)

    assert converter._validate(category=category, xlsx_df_dict=xlsx_df_dict) is None


@pytest.mark.parametrize(
    "category, xlsx_path, expected_csv_file_names",
    [
        (
            "structure",
            "convert_input_structure/structure.xlsx",
            [
                "Energy_Types.csv",
                "Emission_Types.csv",
                "Aggregates.csv",
                "Lines.csv",
                "Buses.csv",
                "Generators.csv",
                "Storages.csv",
                "TechnologyStack_Buses_out.csv",
                "Technology_Bus.csv",
                "TechnologyStack_Buses.csv",
                "TechnologyStack_Aggregate.csv",
                "Emission_Fees_Emission_Types.csv",
                "Generator_Emission_Fees.csv",
            ],
        ),
        (
            "fuels",
            "convert_input_structure/fuels.xlsx",
            [
                "Emission_Per_Unit.csv",
                "Energy_Per_Unit.csv",
            ],
        ),
        (
            "capacity_factors",
            "convert_input_structure/capacity_factors.xlsx",
            ["Profiles.csv"],
        ),
        (
            "demand_types",
            "convert_input_structure/demand_types.xlsx",
            [
                "mieszkanie_w_bloku.csv",
                "jednorodzinny_mały.csv",
            ],
        ),
        (
            "scenarios",
            "convert_input_structure/scenarios/scenario_1.xlsx",
            [
                "Energy_Source_Evolution_Limits.csv",
                "Cost_Parameters.csv",
                "Fuel_Availability.csv",
                "Relative_Emission_Limits.csv",
                "Fuel_Prices.csv",
                "Fractions.csv",
                "Emission_Fees.csv",
            ],
        ),
        (
            "generator_types",
            "convert_input_structure/generator_types.xlsx",
            [
                "Generator_Types.csv",
                "Efficiency.csv",
                "Emission_Reduction.csv",
                "Generator_Type_Energy_Carrier.csv",
                "Generator_Type_Energy_Type.csv",
            ],
        ),
        (
            "storage_types",
            "convert_input_structure/storage_types.xlsx",
            [
                "Parameters.csv",
            ],
        ),
        (
            "initial_state",
            "convert_input_structure/initial_state.xlsx",
            [
                "TechnologyStack.csv",
                "Technology.csv",
            ],
        ),
        (
            "conversion_rate",
            "convert_input_structure/conversion_rate.xlsx",
            [
                "HEAT_PUMP.csv",
            ],
        ),
    ],
)
def test_xlsx_to_csv_convert_xlsx_to_csv(
    category: str, xlsx_path: Path, expected_csv_file_names: list[str]
) -> None:
    with tempfile.TemporaryDirectory() as tmp_output:
        tmp_output_path = Path(tmp_output)
        input_file = get_resources("convert_input_structure")
        converter = ExcelToCsvConverter(
            output_files_path=tmp_output_path, input_files_path=input_file
        )
        xlsx_path = get_resources(xlsx_path)
        xlsx_df_dict = pd.read_excel(xlsx_path, sheet_name=None)
        xlsx_df_dict = converter._sanitize_spreadsheets_names(xlsx_df_dict)
        converter._convert_xlsx_to_csv(category=category, xlsx_df_dict=xlsx_df_dict)

        expected_output_dir = tmp_output_path / category
        assert expected_output_dir.is_dir()
        for csv_name in expected_csv_file_names:
            expected_csv_file = expected_output_dir / csv_name
            assert expected_csv_file.is_file()


@pytest.mark.parametrize(
    "scenario_path_str", ["convert_input_structure/scenarios/scenario_1.xlsx", None]
)
def test_xlsx_to_csv_convert(scenario_path_str: str | None) -> None:
    amount_of_csv_files = {
        "initial_state": 2,
        "structure": 17,
        "fuels": 2,
        "capacity_factors": 1,
        "demand_chunks": 2,
        "generator_types": 7,
        "storage_types": 2,
        "demand_types": 2,
        "scenarios": 17,
        "conversion_rate": 1,
        "generator_type_efficiency": 2,
    }
    input_files_path = get_resources("convert_input_structure")
    scenario_path = get_resources(scenario_path_str) if scenario_path_str else None
    with tempfile.TemporaryDirectory() as tmp_output:
        tmp_output_path = Path(tmp_output)
        converter = ExcelToCsvConverter(
            output_files_path=tmp_output_path,
            input_files_path=input_files_path,
            scenario_path=scenario_path,
        )

        converter.convert()

        for dir_name in DataCategories.get_main_categories():
            if dir_name == DataCategories.SCENARIO or scenario_path is None:
                continue
            else:
                expected_output_dir = tmp_output_path / (
                    dir_name
                    if dir_name != DataCategories.SCENARIO
                    else dir_name + "/" + scenario_path.stem
                )
                assert expected_output_dir.is_dir()
                assert (
                    len(list(expected_output_dir.glob("*.csv")))
                    == amount_of_csv_files[dir_name]
                )


def test_xlsx_converter_wrong_path() -> None:
    wrong_input_files_path = Path("/tmp/csv_files_dir/")
    with pytest.raises(ExcelToCsvConverterException) as error:
        ExcelToCsvConverter(
            output_files_path=Path("/tmp/"),
            input_files_path=wrong_input_files_path,
            scenario_path=None,
        ).convert()
    assert "File cannot be found in given path" in str(error.value)


@pytest.mark.parametrize(
    "df, valid_struct, expected_structure",
    (
        (
            pd.DataFrame(
                {
                    "name": ["abc", "def", "ghi"],
                    "value": [1, 2, 3],
                    "flag": [True, None, np.nan],
                }
            ),
            DatasetConfig(
                dataset_name="test_dataset",
                columns={"name": str, "value": int},
                default_type={bool},
            ),
            {"flag": "bool", "name": "object", "value": "int64"},
        ),
    ),
)
def test_get_dataframe_structure(
    df: pd.DataFrame, valid_struct: DatasetConfig, expected_structure: dict[str, str]
) -> None:
    assert expected_structure == ExcelToCsvConverter._get_dataframe_structure(
        df, valid_struct
    )
