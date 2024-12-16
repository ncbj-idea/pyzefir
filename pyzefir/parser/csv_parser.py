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
    get_optional_datasets_from_categories,
)

logger = logging.getLogger(__name__)


class CsvParserException(Exception):
    pass


class CsvParser:
    """
    Handles loading and validation of CSV files into DataFrames.

    This class is responsible for parsing CSV files from specified paths, validating their
    structure against predefined configurations, and loading them into a dictionary format
    for further data processing. It ensures that the data is correctly formatted and raises
    exceptions if required files are missing or the data is invalid.
    """

    def __init__(self, path_manager: CsvPathManager) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - path_manager (CsvPathManager): Manages the paths to the CSV files.
        """
        self._path_manager = path_manager

    def load_dfs(self) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Loads DataFrames from CSV files categorized under main categories.

        This method iterates over the main categories, retrieves DataFrames for each category,
        and returns them in a nested dictionary structure. It logs a debug message upon successful
        upload of all DataFrames, ensuring the entire set is valid.

        Returns:
            - dict[str, dict[str, pd.DataFrame]]: A dictionary containing DataFrames categorized
              by their respective categories and dataset names.
        """
        name_df_dict: dict[str, dict[str, pd.DataFrame]] = dict()
        for category in DataCategories.get_main_categories():
            name_df_dict[category] = self._get_dfs_from_category(category=category)
        logger.debug("Entire set of dfs is valid and uploaded")
        return name_df_dict

    def _get_dfs_from_category(self, category: str) -> dict[str, pd.DataFrame]:
        """
        Retrieves DataFrames for a specified category.

        Args:
            - category (str): The category for which to load DataFrames.

        Returns:
            - dict[str, pd.DataFrame]: A dictionary of DataFrames indexed by dataset names.
        """
        category_dict = dict()
        if category in DataCategories.get_dynamic_categories():
            for csv_path in self._path_manager.get_path(category).glob("*.csv"):
                dataset_name = csv_path.stem
                df = self._read_and_validate_csv_file(
                    category=category, dataset_name=dataset_name, csv_path=csv_path
                )
                df.columns = df.columns.astype(str)
                category_dict[dataset_name] = df
        else:
            for dataset_name in get_datasets_from_categories(data_category=category):
                csv_path = self._path_manager.get_path(
                    data_category=category, dataset_name=dataset_name
                )
                df = self._read_and_validate_csv_file(
                    category=category, dataset_name=dataset_name, csv_path=csv_path
                )
                df.columns = df.columns.astype(str)
                category_dict[dataset_name] = df

        return category_dict

    @staticmethod
    def _read_and_validate_csv_file(
        category: str, dataset_name: str, csv_path: Path
    ) -> pd.DataFrame:
        """
        Reads and validates a CSV file, returning a DataFrame.

        Args:
            - category (str): The category of the dataset.
            - dataset_name (str): The name of the dataset.
            - csv_path (Path): The path to the CSV file.

        Returns:
            - pd.DataFrame: A DataFrame containing the loaded data.
        """
        if not csv_path.is_file():
            if dataset_name in get_optional_datasets_from_categories(category):
                columns = get_dataset_config_from_categories(
                    category, dataset_name
                ).columns
                return pd.DataFrame(
                    {col: pd.Series(dtype=dtype) for col, dtype in columns.items()}
                )
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
            df=df,
            dataframe_structure=columns_dict,
            valid_structure=columns_valid_config,
            dataset_reference=dataset_reference,
        ).validate()
        logger.debug(f"Dataframe {dataset_name} is valid")
        return df
