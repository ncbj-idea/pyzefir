import logging
from itertools import zip_longest
from typing import Type

import pandas as pd

from pyzefir.parser.validator.valid_structure import (
    DataFramesColumnsType,
    DatasetConfig,
)

_logger = logging.getLogger(__name__)


class DataFrameValidatorException(Exception):
    pass


class DataFrameValidatorGroupException(
    DataFrameValidatorException,
    ExceptionGroup,
):
    pass


class DataFrameValidator:
    """
    A class to validate the structure of a Pandas DataFrame against a defined expected structure.

    This class ensures that the DataFrame's columns conform to the expected names and types specified
    in a valid structure. It checks for both static and dynamic columns, collecting any discrepancies
    as exceptions to facilitate debugging.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dataframe_structure: dict[str, str],
        valid_structure: DatasetConfig,
        dataset_reference: str,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - df (pd.DataFrame): The DataFrame to be validated.
              dataframe_structure (dict[str, str]): A mapping of DataFrame column names to their corresponding
              pandas types.
            - valid_structure (DatasetConfig): An object that defines the expected structure of the DataFrame.
            - dataset_reference (str): A reference string for identifying the dataset being validated.
        """
        self.dataframe_structure = self._translate_pandas_type_to_python_type(
            dataframe_structure, dataset_reference
        )
        self.valid_structure = valid_structure
        self.dataset_reference = dataset_reference
        self._df = df

    def validate(self) -> None:
        """
        Validates the structure of the given DataFrame against the expected structure.

        This method checks if the DataFrame's columns match the required structure and raises an exception
        if any discrepancies are found. It gathers all validation errors and raises a grouped exception
        for easier debugging.

        Raises:
            - DataFrameValidatorGroupException: If any validation errors occur during the process.
        """
        exception_list: list[DataFrameValidatorException] = []
        self._check_dataframe_structure(exception_list=exception_list)
        if exception_list:
            raise DataFrameValidatorGroupException(
                f"Following errors occurred while processing input file "
                f"{self.dataset_reference}: ",
                exception_list,
            )

    def _check_dynamic_column(
        self,
        column_type: Type[DataFramesColumnsType] | None,
        column_name: str,
        exception_list: list[DataFrameValidatorException],
    ) -> None:
        """
        Checks the validity of dynamic columns in the DataFrame.

        Args:
            - column_type (Type[DataFramesColumnsType] | None): The detected type of the dynamic column.
            - column_name (str): The name of the dynamic column being checked.
            - exception_list (list[DataFrameValidatorException]): A list to which any validation exceptions
                will be appended.
        """
        if not self.valid_structure.default_type:
            exception_list.append(
                DataFrameValidatorException(
                    f"Dataframe column name {column_name} not found in required structure"
                    f" {list(self.valid_structure.columns)}"
                )
            )
        elif not any(
            self._check_type_match(column_type, d, self._df, column_name)
            for d in self.valid_structure.default_type
        ):
            exception_list.append(
                DataFrameValidatorException(
                    f"Type of dynamic column {column_name}: {column_type} not found"
                    f" in allowed types {self.valid_structure.default_type}"
                )
            )

    @staticmethod
    def _check_type_match(
        type_a: DataFramesColumnsType,
        type_b: DataFramesColumnsType,
        df: pd.DataFrame,
        column_name: str,
    ) -> bool:
        """
        Checks whether type_a matches type_b or if type_a is float and the column contains only NaNs.

        This function allows for the possibility of empty columns (all NaN values) when type_a is float,
        as empty columns are acceptable at this stage. It ensures that int can match float but not the
        other way around to avoid information loss.

        Args:
            - type_a (DataFramesColumnsType): The type to be matched against.
            - type_b (DataFramesColumnsType): The type to check against.
            - df (pd.DataFrame): The DataFrame containing the column data.
            - column_name (str): The name of the column being checked.

        Returns:
            - bool: True if the types match or if the conditions for empty columns are met; False otherwise.
        """
        if type_a is float and df[column_name].isnull().all():
            _logger.debug(f"Dataframe column {column_name} it's empty")
            return True
        return type_a == type_b or (type_a is int and type_b is float)

    def _check_static_column(
        self,
        column_name: str,
        column_type: DataFramesColumnsType,
        valid_column_name: str,
        valid_type: DataFramesColumnsType,
        exception_list: list[DataFrameValidatorException],
    ) -> None:
        """
        Checks the validity of static columns in the DataFrame.

        Args:
            - column_name (str): The name of the static column being checked.
            - column_type (DataFramesColumnsType): The detected type of the column.
            - valid_column_name (str): The expected name of the column.
            - valid_type (DataFramesColumnsType): The expected type of the column.
            - exception_list (list[DataFrameValidatorException]): A list to which any validation exceptions
                will be appended.
        """
        if column_name != valid_column_name:
            if column_name in self.valid_structure.columns:
                exception_list.append(
                    DataFrameValidatorException(
                        f"Column {column_name} is misplaced. Should be on index "
                        f"{list(self.valid_structure.columns).index(column_name)}, "
                        f"but it is on {list(self.dataframe_structure).index(column_name)} instead"
                    )
                )
                misplaced_column_type = self.valid_structure.columns[column_name]
                if not self._check_type_match(
                    column_type, misplaced_column_type, self._df, column_name
                ):
                    exception_list.append(
                        DataFrameValidatorException(
                            f"Dataframe column {column_name} type {column_type} "
                            f"is different as in required structure {misplaced_column_type}"
                        )
                    )
            else:
                exception_list.append(
                    DataFrameValidatorException(
                        f"Dataframe column name {column_name} not found in required structure"
                        f" {list(self.valid_structure.columns)}"
                    )
                )
        elif not self._check_type_match(column_type, valid_type, self._df, column_name):
            exception_list.append(
                DataFrameValidatorException(
                    f"Dataframe column {column_name} type {column_type} "
                    f"is different as in required structure {valid_type}"
                )
            )

    def _check_dataframe_structure(
        self,
        exception_list: list[DataFrameValidatorException],
    ) -> None:
        """
        Following conditions are checked in this method:
        1. If there are more columns in dataframe_structure than there are in valid_structure,
        if yes then valid_column_name is None.
            1a. If valid_column_name is None, then check if default type is
            defined for given dataframe (that would mean that the given column
            is "dynamic") and matches given column

        2. Otherwise, we check if given column is equal to currently checked valid_column
            2a. If no, then we check whether the given column is present in valid_structure
                2aa. If yes, that means that the column is correct, but misplaced. Then we check its type.
                2ab. If no, then the given column is not in a valid_structure.
            2b. If yes, we just have to check if its type is correct

        Args:
            - exception_list (list[DataFrameValidatorException]): A list to which any validation exceptions
                will be appended.
        """
        for column_name, valid_column_name in zip_longest(
            self.dataframe_structure, self.valid_structure.columns
        ):
            column_type = self.dataframe_structure.get(column_name)
            valid_type = self.valid_structure.columns.get(valid_column_name)
            if valid_column_name is None:
                self._check_dynamic_column(column_type, column_name, exception_list)
            else:
                self._check_static_column(
                    column_name,
                    column_type,
                    valid_column_name,
                    valid_type,
                    exception_list,
                )

    @staticmethod
    def _translate_pandas_type_to_python_type(
        dataframe_structure: dict[str, str], dataset_reference: str
    ) -> dict[str, DataFramesColumnsType]:
        """
        Translates pandas DataFrame column types to Python types.

        Args:
            - dataframe_structure (dict[str, str]): A dictionary mapping column names to their pandas types.
            - dataset_reference (str): A reference string for the dataset being processed.

        Returns:
            - dict[str, DataFramesColumnsType]: A dictionary mapping column names to their Python types.

        Raises:
            - DataFrameValidatorException: If an unknown column type is encountered.
        """
        translated_structure: dict[str, DataFramesColumnsType] = dict()
        for column_name, pandas_type_name in dataframe_structure.items():
            match pandas_type_name:
                case "int64":
                    translated_structure[column_name] = int
                case "float64":
                    translated_structure[column_name] = float
                case "object":
                    translated_structure[column_name] = str
                case "bool":
                    translated_structure[column_name] = bool
                case _:
                    _logger.warning(
                        f"Dataframe has unknown column type {pandas_type_name}"
                    )
                    raise DataFrameValidatorException(
                        f"Dataset {dataset_reference} column type error. "
                        f"Column: {column_name} type: {pandas_type_name} "
                        f"cannot be translated into python type"
                    )

        return translated_structure
