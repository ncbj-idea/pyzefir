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
import os.path
from pathlib import Path

import pandas as pd

from pyzefir.parser.utils import TRUE_VALUES, sanitize_dataset_name
from pyzefir.parser.validator.dataframe_validator import DataFrameValidator
from pyzefir.parser.validator.valid_structure import (
    DatasetConfig,
    get_dataset_config_from_categories,
    get_dataset_reference,
)
from pyzefir.utils.converters.converter import AbstractConverter
from pyzefir.utils.path_manager import (
    DataCategories,
    XlsxPathManager,
    get_datasets_from_categories,
)

logger = logging.getLogger(__name__)


class ExcelToCsvConverterException(Exception):
    pass


class ExcelToCsvConverter(AbstractConverter):
    def __init__(
        self,
        input_files_path: Path,
        output_files_path: Path,
        scenario_path: Path | None = None,
    ) -> None:
        self.path_manager = XlsxPathManager(
            input_path=input_files_path,
            output_path=output_files_path,
            scenario_name=scenario_path.stem if scenario_path else None,
        )
        self._scenario_path = scenario_path

    def convert(self) -> None:
        for category in DataCategories.get_main_categories():
            if category == DataCategories.SCENARIO and not self._scenario_path:
                logger.debug(
                    f"Scenario file is not passed {self._scenario_path=}, skipped"
                )
                continue
            xlsx_path = (
                self._scenario_path
                if category == DataCategories.SCENARIO
                else self.path_manager.get_input_file_path(category)
            )
            if not os.path.isfile(str(xlsx_path)):
                logger.error(f"Cannot find file in dir {xlsx_path}")
                raise ExcelToCsvConverterException(
                    f"File cannot be found in given path: {xlsx_path}"
                )

            xlsx_df_dict = pd.read_excel(
                xlsx_path, sheet_name=None, true_values=TRUE_VALUES
            )
            logger.debug(f"File {xlsx_path} found in given path")

            sanitized_df_dict = self._sanitize_spreadsheets_names(xlsx_df_dict)
            self._validate(category=category, xlsx_df_dict=sanitized_df_dict)
            self._convert_xlsx_to_csv(category, xlsx_df_dict=sanitized_df_dict)

    def _validate(self, category: str, xlsx_df_dict: dict[str, pd.DataFrame]) -> None:
        if category not in DataCategories.get_dynamic_categories():
            self._validate_xlsx_structure(
                xlsx_sheets_names=list(xlsx_df_dict.keys()), category=category
            )
        self._validate_dataframes_structure(
            xlsx_df_dict=xlsx_df_dict, category=category
        )

    @staticmethod
    def _validate_dataframes_structure(
        xlsx_df_dict: dict[str, pd.DataFrame], category: str
    ) -> None:
        for sheet_name, df in xlsx_df_dict.items():
            if df.empty:
                continue
            valid_structure = get_dataset_config_from_categories(
                data_category=category, dataset_name=sheet_name
            )
            dataframe_structure: dict[
                str, str
            ] = ExcelToCsvConverter._get_dataframe_structure(df, valid_structure)
            dataset_reference = get_dataset_reference(
                category, valid_structure.dataset_name
            )
            DataFrameValidator(
                dataframe_structure=dataframe_structure,
                valid_structure=valid_structure,
                dataset_reference=dataset_reference,
            ).validate()

    @staticmethod
    def _get_dataframe_structure(
        df: pd.DataFrame, valid_structure: DatasetConfig
    ) -> dict[str, str]:
        structure = {}
        for col, dtype in df.dtypes.items():
            if valid_structure.columns.get(col) is bool:
                df.loc[:, col] = df.loc[:, col].fillna(False)
                structure[str(col)] = "bool"
            elif (
                col not in valid_structure.columns
                and valid_structure.default_type is not None
                and bool in valid_structure.default_type
            ):
                df.loc[:, col] = df.loc[:, col].fillna(False)
                structure[str(col)] = "bool"
            else:
                structure[str(col)] = dtype.name
        return structure

    @staticmethod
    def _validate_xlsx_structure(xlsx_sheets_names: list[str], category: str) -> None:
        valid_sheets_names: list[str] = get_datasets_from_categories(category)
        missing_spreadsheets = set(valid_sheets_names).difference(xlsx_sheets_names)
        if missing_spreadsheets:
            logger.error("Required spreadsheets not in xlsx file")
            raise ExcelToCsvConverterException(
                f"Not all required spreadsheets {missing_spreadsheets} found in xlsx file spreadsheets"
                f" {xlsx_sheets_names}"
            )
        logger.debug("All required spreadsheets are in xlsx file")

    def _convert_xlsx_to_csv(
        self, category: str, xlsx_df_dict: dict[str, pd.DataFrame]
    ) -> None:
        for sheet_name, df in xlsx_df_dict.items():
            csv_path = (
                self.path_manager.concatenate_path_for_dynamic_dataset_name(
                    category, sheet_name
                )
                if category in DataCategories.get_dynamic_categories()
                else self.path_manager.get_path(category, sheet_name)
            )
            self.manage_existence_path(csv_path)
            logger.debug(f"Saving dataframe: {csv_path}")
            df.to_csv(csv_path, index=False)

    @staticmethod
    def _sanitize_spreadsheets_names(
        xlsx_df_dict: dict[str | int, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        new_xlsx_df_dict = {}
        for sheet_name in list(xlsx_df_dict.keys()):
            sheet_name_refactored = sanitize_dataset_name(str(sheet_name))
            new_xlsx_df_dict[sheet_name_refactored] = xlsx_df_dict.pop(sheet_name)
            logger.debug(f"Replace name: {sheet_name} into {sheet_name_refactored}")
        return new_xlsx_df_dict
