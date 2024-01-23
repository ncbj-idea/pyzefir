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

import logging
from itertools import zip_longest
from typing import Type

from pyzefir.parser.validator.valid_structure import (
    DataFramesColumnsType,
    DatasetConfig,
)

logger = logging.getLogger(__name__)


class DataFrameValidatorException(Exception):
    pass


class DataFrameValidatorGroupException(
    DataFrameValidatorException,
    ExceptionGroup,
):
    pass


class DataFrameValidator:
    def __init__(
        self,
        dataframe_structure: dict[str, str],
        valid_structure: DatasetConfig,
        dataset_reference: str,
    ) -> None:
        self.dataframe_structure = self._translate_pandas_type_to_python_type(
            dataframe_structure, dataset_reference
        )
        self.valid_structure = valid_structure
        self.dataset_reference = dataset_reference

    def validate(self) -> None:
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
        if not self.valid_structure.default_type:
            exception_list.append(
                DataFrameValidatorException(
                    f"Dataframe column name {column_name} not found in required structure"
                    f" {list(self.valid_structure.columns)}"
                )
            )
        elif not any(
            self._check_type_match(column_type, d)
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
        type_a: DataFramesColumnsType, type_b: DataFramesColumnsType
    ) -> bool:
        """
        Checks whether type b matches type a.
        Note the order of arguments i.e. int matches float,
        but float does not match int (you can't cast float to int without an information loss)
        """
        return type_a == type_b or (type_a is int and type_b is float)

    def _check_static_column(
        self,
        column_name: str,
        column_type: DataFramesColumnsType,
        valid_column_name: str,
        valid_type: DataFramesColumnsType,
        exception_list: list[DataFrameValidatorException],
    ) -> None:
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
                if not self._check_type_match(column_type, misplaced_column_type):
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
        elif not self._check_type_match(column_type, valid_type):
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
                    logger.warning(
                        f"Dataframe has unknown column type {pandas_type_name}"
                    )
                    raise DataFrameValidatorException(
                        f"Dataset {dataset_reference} column type error. "
                        f"Column: {column_name} type: {pandas_type_name} "
                        f"cannot be translated into python type"
                    )

        return translated_structure
