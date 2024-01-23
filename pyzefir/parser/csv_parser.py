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
from pathlib import Path

import pandas as pd

from pyzefir.parser.utils import TRUE_VALUES
from pyzefir.parser.validator.dataframe_validator import DataFrameValidator
from pyzefir.parser.validator.valid_structure import (
    get_dataset_config_from_categories,
    get_dataset_reference,
)
from pyzefir.utils.path_manager import (
    CsvPathManager,
    DataCategories,
    get_datasets_from_categories,
)

logger = logging.getLogger(__name__)


class CsvParserException(Exception):
    pass


class CsvParser:
    def __init__(self, path_manager: CsvPathManager) -> None:
        self._path_manager = path_manager

    def load_dfs(self) -> dict[str, dict[str, pd.DataFrame]]:
        name_df_dict: dict[str, dict[str, pd.DataFrame]] = dict()
        for category in DataCategories.get_main_categories():
            name_df_dict[category] = self._get_dfs_from_category(category=category)
        logger.debug("Entire set of dfs is valid and uploaded")
        return name_df_dict

    def _get_dfs_from_category(self, category: str) -> dict[str, pd.DataFrame]:
        category_dict = dict()
        if category in DataCategories.get_dynamic_categories():
            for csv_path in self._path_manager.get_path(category).glob("*.csv"):
                dataset_name = csv_path.stem
                df = self._read_and_validate_csv_file(
                    category=category, dataset_name=dataset_name, csv_path=csv_path
                )
                category_dict[dataset_name] = df
        else:
            for dataset_name in get_datasets_from_categories(data_category=category):
                csv_path = self._path_manager.get_path(
                    data_category=category, dataset_name=dataset_name
                )
                df = self._read_and_validate_csv_file(
                    category=category, dataset_name=dataset_name, csv_path=csv_path
                )
                category_dict[dataset_name] = df

        return category_dict

    @staticmethod
    def _read_and_validate_csv_file(
        category: str, dataset_name: str, csv_path: Path
    ) -> pd.DataFrame:
        if not csv_path.is_file():
            logger.error(f"File {dataset_name}.csv not found")
            raise CsvParserException(f"Required file: {csv_path} does not exists ")
        df = pd.read_csv(csv_path, true_values=TRUE_VALUES)
        if df.empty:
            return df
        columns_dict = {col: dtype.name for col, dtype in df.dtypes.items()}
        columns_valid_config = get_dataset_config_from_categories(
            category, dataset_name
        )
        dataset_reference = get_dataset_reference(category, dataset_name)
        DataFrameValidator(
            dataframe_structure=columns_dict,
            valid_structure=columns_valid_config,
            dataset_reference=dataset_reference,
        ).validate()
        logger.debug(f"Dataframe {dataset_name} is valid")
        return df
