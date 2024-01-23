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

from typing import Type
from unittest import mock

import pandas as pd
import pytest

from pyzefir.parser.utils import TRUE_VALUES
from pyzefir.parser.validator.dataframe_validator import (
    DataFrameValidator,
    DataFrameValidatorException,
)
from pyzefir.parser.validator.valid_structure import (
    DataFramesColumnsType,
    DatasetConfig,
    InvalidStructureException,
    get_dataset_config_from_categories,
)
from pyzefir.utils.converters.xlsx_to_csv_converter import ExcelToCsvConverter
from tests.utils import get_resources


@pytest.mark.parametrize(
    "data_category, dataset_name, logger_msg",
    [
        ("initial_state", "Technology", None),
        (
            "structure",
            "cycle_length",
            "No dataset config found for category"
            " structure and dataset cycle_length",
        ),
        (
            "fuel_types",
            "EmissionPer_Unit",
            "data_category=fuel_types not in dataset_validation_structure keys",
        ),
        ("generator_types", "Efficiency", None),
        ("storage_types", "Parameters", None),
        ("demand_types", "Dom", None),
        (
            "demandtypes",
            "Dom",
            "data_category=demandtypes not in dataset_validation_structure keys",
        ),  # noqa typo
    ],
)
def test_valid_structure_get_dataset_config_from_categories(
    data_category: str, dataset_name: str, logger_msg: str | None
) -> None:
    if logger_msg:
        with mock.patch("logging.Logger.warning") as mock_logger, pytest.raises(
            InvalidStructureException
        ):
            get_dataset_config_from_categories(data_category, dataset_name)
            mock_logger.assert_called_with(logger_msg)
    else:
        tested = get_dataset_config_from_categories(data_category, dataset_name)
        assert isinstance(tested, DatasetConfig)
        assert tested.dataset_name == dataset_name


@pytest.mark.parametrize(
    "dataframe_structure, expected, error",
    [
        (
            {"name": "object", "float_column": "float64"},
            {"name": str, "float_column": float},
            None,
        ),
        (
            {"year_idx": "int64", "is_bool": "bool"},
            {"year_idx": int, "is_bool": bool},
            None,
        ),
        (
            {"year_idx": "int64", "round_value": "int32"},
            {"year_idx": int, "round_value": int},
            DataFrameValidatorException,
        ),
    ],
)
def test_csv_validator_translate_pandas_type_to_python_type(
    dataframe_structure: dict[str, str],
    expected: dict[str, str],
    error: Type[Exception] | None,
) -> None:
    if error:
        with pytest.raises(error):
            DataFrameValidator._translate_pandas_type_to_python_type(
                dataframe_structure, dataset_reference=""
            )
    else:
        assert (
            DataFrameValidator._translate_pandas_type_to_python_type(
                dataframe_structure, dataset_reference=""
            )
            == expected
        )


@pytest.mark.parametrize(
    "valid_structure, dataframe_structure, error_msg",
    [
        (
            DatasetConfig(
                dataset_name="", columns={"hour_idx": int}, default_type={float}
            ),
            {"hour_idx": "int64", "CO2": "float64", "SO2": "float64"},
            "",
        ),
        (
            DatasetConfig(
                dataset_name="", columns={"hour_idx": int}, default_type={float}
            ),
            {"hour_idx": "float64", "CO2": "float64", "SO2": "float64"},
            "Dataframe column hour_idx type <class 'float'> is different "
            "as in required structure <class 'int'>",
        ),
        (
            DatasetConfig(
                dataset_name="", columns={"hour_idx": int}, default_type={float}
            ),
            {"year_idx": "int64", "CO2": "float64", "SO2": "float64"},
            "Dataframe column name year_idx not found in required structure "
            "['hour_idx']",
        ),
        (
            DatasetConfig(
                dataset_name="", columns={"hour_idx": int}, default_type={float}
            ),
            {"hour_idx": "int64", "CO2": "int64", "SO2": "float64"},
            "Type of dynamic column CO2: <class 'int'> not found in "
            "allowed types {<class 'float'>}",
        ),
        (
            DatasetConfig(
                dataset_name="", columns={"hour_idx": int}, default_type={float}
            ),
            {"hour_idx": "int64", "CO2": "int64", "SO2": "object"},
            "Type of dynamic column SO2: <class 'str'> not found in "
            "allowed types {<class 'float'>}",
        ),
    ],
)
def test_csv_validator_check_dataframe_dynamic_structure(
    valid_structure: DatasetConfig,
    dataframe_structure: dict[str, DataFramesColumnsType],
    error_msg: str | set[str],
) -> None:
    exception_list: list[DataFrameValidatorException] = []

    DataFrameValidator(
        valid_structure=valid_structure,
        dataframe_structure=dataframe_structure,
        dataset_reference="test name",
    )._check_dataframe_structure(exception_list)
    if len(exception_list) == 1:
        exception = exception_list[0]
        assert isinstance(exception, DataFrameValidatorException)
        assert str(exception) == error_msg
    elif len(exception_list) > 1:
        assert all(
            isinstance(exception, DataFrameValidatorException)
            for exception in exception_list
        )
        assert set([str(e) for e in exception_list]) == set(error_msg)


@pytest.mark.parametrize(
    "valid_columns_dict, dataframe_structure, error_msg",
    [
        (
            {"generator_name": str, "conversion_rate": float, "efficiency": int},
            {
                "generator_name": "object",
                "conversion_rate": "float64",
                "efficiency": "int64",
            },
            "",
        ),
        (
            {"generator_name": str, "conversion_rate": float, "efficiency": int},
            {
                "generator_name": "object",
                "generator_diamond": "float64",
                "efficiency": "int64",
            },
            "Dataframe column name generator_diamond not found in required "
            "structure ['generator_name', 'conversion_rate', 'efficiency']",
        ),
        (
            {"generator_name": str, "conversion_rate": float, "efficiency": int},
            {
                "generator_name": "object",
                "efficiency": "int64",
                "conversion_rate": "float64",
            },
            {
                "Column efficiency is misplaced. Should be on index 2, "
                "but it is on 1 instead",
                "Column conversion_rate is misplaced. Should be on index 1, "
                "but it is on 2 instead",
            },
        ),
        (
            {"generator_name": str, "conversion_rate": float, "efficiency": int},
            {
                "generator_name": "object",
                "conversion_rate": "int64",
                "efficiency": "object",
            },
            "Dataframe column efficiency type <class 'str'> is different "
            "as in required structure <class 'int'>",
        ),
        (
            {"generator_name": str, "conversion_rate": float, "efficiency": int},
            {
                "generator_name": "object",
                "efficiency": "object",
                "conversion_rate": "int64",
            },
            {
                "Column efficiency is misplaced. Should be on index 2, but "
                "it is on 1 instead",
                "Column conversion_rate is misplaced. Should be on index 1, "
                "but it is on 2 instead",
                "Dataframe column efficiency type <class 'str'> is different "
                "as in required structure <class 'int'>",
            },
        ),
    ],
)
def test_csv_validator_check_dataframe_static_structure(
    valid_columns_dict: dict[str, DataFramesColumnsType],
    dataframe_structure: dict[str, DataFramesColumnsType],
    error_msg: str | set[str],
) -> None:
    exception_list: list[DataFrameValidatorException] = []
    valid_structure = DatasetConfig(
        dataset_name="Test_dataset", columns=valid_columns_dict
    )

    DataFrameValidator(
        valid_structure=valid_structure,
        dataframe_structure=dataframe_structure,
        dataset_reference="df_name",
    )._check_dataframe_structure(exception_list)

    if len(exception_list) == 1:
        exception = exception_list[0]
        assert isinstance(exception, DataFrameValidatorException)
        assert str(exception) == error_msg
    elif len(exception_list) > 1:
        assert all(
            isinstance(exception, DataFrameValidatorException)
            for exception in exception_list
        )
        assert set([str(e) for e in exception_list]) == set(error_msg)


@pytest.mark.parametrize(
    "dataframe_structure, valid_structure, error",
    [
        (
            {"name": "object"},
            DatasetConfig(dataset_name="Energy_Types", columns={"name": str}),
            None,
        ),
        (
            {"technology": "object", "base_capacity": "float64"},
            DatasetConfig(
                dataset_name="Technology",
                columns={"technology": str, "base_capacity": float},
            ),
            None,
        ),
        (
            {
                "name": "object",
                "energy_type": "object",
                "loss_in": "float64",
                "loss_out": "int64",
            },
            DatasetConfig(
                dataset_name="Buses",
                columns={
                    "name": str,
                    "energy_type": str,
                    "loss_in": float,
                    "loss_out": float,
                },
            ),
            None,
        ),
        (
            {"year_idx": "int64", "COAL": "int64", "PV": "int64", "SOLAR": "int64"},
            DatasetConfig(
                dataset_name="Fuel_Prices",
                columns={"year_idx": int},
                default_type={int},
            ),
            None,
        ),
        (
            {"year_idx": "int64", "COAL": "int64", "PV": "int64", "SOLAR": "object"},
            DatasetConfig(
                dataset_name="Fuel_Availability",
                columns={"year_idx": int},
                default_type={float, int},
            ),
            True,
        ),
        (
            {"name": "object", "demandtype": "object"},  # noqa typo
            DatasetConfig(
                dataset_name="Aggregates", columns={"name": str, "demand_type": str}
            ),
            True,
        ),
    ],
)
def test_csv_validator_validate(
    dataframe_structure: dict[str, str],
    valid_structure: DatasetConfig,
    error: bool | None,
) -> None:
    validator = DataFrameValidator(
        dataframe_structure=dataframe_structure,
        valid_structure=valid_structure,
        dataset_reference="dataset id",
    )
    if error:
        with pytest.raises(DataFrameValidatorException):
            validator.validate()
    else:
        validator.validate()


@pytest.mark.parametrize(
    "category, xlsx_path",
    [
        ("structure", "convert_input_structure/structure.xlsx"),
        ("fuels", "convert_input_structure/fuels.xlsx"),
        ("capacity_factors", "convert_input_structure/capacity_factors.xlsx"),
        ("demand_types", "convert_input_structure/demand_types.xlsx"),
        ("generator_types", "convert_input_structure/generator_types.xlsx"),
        ("initial_state", "convert_input_structure/initial_state.xlsx"),
        ("storage_types", "convert_input_structure/storage_types.xlsx"),
        ("scenarios", "convert_input_structure/scenarios/scenario_1.xlsx"),
    ],
)
def test_csv_validator_validate_lfs_files(category: str, xlsx_path: str) -> None:
    xlsx_df_dict = pd.read_excel(
        get_resources(xlsx_path), sheet_name=None, true_values=TRUE_VALUES
    )
    for dataset_name, df in xlsx_df_dict.items():
        dataset_name = (
            str(dataset_name).replace("-", "").replace(" ", "_").replace("__", "_")
        )
        tested = get_dataset_config_from_categories(category, dataset_name)
        columns_dict = ExcelToCsvConverter._get_dataframe_structure(df, tested)
        DataFrameValidator(
            dataframe_structure=columns_dict,
            valid_structure=tested,
            dataset_reference="dataset id",
        ).validate()


@pytest.mark.parametrize(
    "type_a, type_b, expected",
    (
        (str, str, True),
        (str, int, False),
        (int, float, True),
        (float, int, False),
        (bool, int, False),
        (int, bool, False),
    ),
)
def test_column_type_match(
    type_a: DataFramesColumnsType,
    type_b: DataFramesColumnsType,
    expected: bool,
) -> None:
    assert DataFrameValidator._check_type_match(type_a, type_b) == expected
