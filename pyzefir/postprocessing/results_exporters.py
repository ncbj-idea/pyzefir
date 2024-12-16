import logging
from pathlib import Path

import pandas as pd
from sanitize_filename import sanitize

from pyzefir.optimization.exportable_results import ExportableResultsGroup
from pyzefir.postprocessing.results_handler import Exporter

_logger = logging.getLogger(__name__)


class CsvExporter(Exporter):
    """
    Class for exporting data to CSV format.

    This class provides methods to export various data structures into CSV files.
    It includes functionality to handle group results and individual objective results,
    ensuring that data is properly organized and saved in the specified directory.
    """

    @staticmethod
    def export_group_results(root_path: Path, result: ExportableResultsGroup) -> None:
        """
        Export exportable group results to CSV files.

        This method iterates over the fields of the provided results group and exports
        each field's data into CSV format. It handles both individual DataFrames and
        dictionaries of DataFrames, creating directories as needed for organized storage.

        Args:
            - root_path (Path): The root path for exporting the results.
            - result (ExportableResultsGroup): The exportable results to be parsed and exported.
        """
        if Exporter.is_results_group_empty(result):
            return
        for field_name, field_value in result.__dict__.items():
            output_path = root_path / field_name
            output_path.mkdir(parents=True, exist_ok=True)
            if isinstance(field_value, dict):
                for df_name, df in field_value.items():
                    df.to_csv(output_path / f"{sanitize(df_name)}.csv")
                    _logger.debug(
                        f"Data {df_name} saved under the path: {output_path / f'{sanitize(df_name)}.csv'}"
                    )
            else:
                field_value.to_csv(output_path / f"{field_name}.csv")
                _logger.debug(
                    f"Data {field_name} saved under the path: {output_path / f'{field_name}.csv'}"
                )

    @staticmethod
    def export_objective_result(
        root_path: Path, objective_value_series: pd.Series
    ) -> None:
        """
        Exports the given objective value series to a CSV file.

        This method saves a Pandas Series containing objective values to a CSV file
        at the specified root path, using the series name as the filename. It ensures
        that the data is accurately written and ready for further analysis.

        Args:
            - root_path (Path): The root path where the CSV file will be saved.
            - objective_value_series (pd.Series): A Pandas Series containing objective values.
        """
        objective_value_series.to_csv(root_path / f"{objective_value_series.name}.csv")


class XlsxExporter(Exporter):
    """Class for exporting data to XLSX format.

    This class provides methods to export various data structures into XLSX files.
    It ensures that data from groups and individual results are properly organized
    into separate sheets, facilitating easy access and analysis.
    """

    @staticmethod
    def export_group_results(root_path: Path, result: ExportableResultsGroup) -> None:
        """
        Export exportable group results to XLSX files.

        This method iterates over the fields of the provided results group and exports
        each field's data into XLSX format. It handles both individual DataFrames and
        dictionaries of DataFrames, creating an XLSX file for each field with appropriate data.

        Args:
            - root_path (Path): The root path for exporting the results.
            - result (ExportableResultsGroup): The exportable results to be parsed and exported.
        """
        if Exporter.is_results_group_empty(result):
            return
        root_path.mkdir(parents=True, exist_ok=True)
        for field_name, field_value in result.__dict__.items():
            with pd.ExcelWriter(
                (root_path / field_name).with_suffix(".xlsx"), engine="xlsxwriter"
            ) as writer:
                if isinstance(field_value, dict):
                    if not field_value:
                        _logger.info(
                            f"No results found for: {field_name} when saving results to xlsx."
                        )
                        continue
                    df = pd.concat(field_value, axis=1).sort_index(axis=1)
                else:
                    df = field_value
                df.to_excel(writer, sheet_name=field_name)
                _logger.debug(
                    f"Data {field_name} saved under the path: {(root_path / field_name).with_suffix('.xlsx')}"
                )

    @staticmethod
    def export_objective_result(
        root_path: Path, objective_value_series: pd.Series
    ) -> None:
        """
        Exports the given objective value series to an XLSX file.

        This method saves a Pandas Series containing objective values to an XLSX file
        at the specified root path, using the series name as the filename. It ensures
        that the data is accurately written and ready for further analysis.

        Args:
            - root_path (Path): The root path where the XLSX file will be saved.
            - objective_value_series (pd.Series): A Pandas Series containing objective values.
        """
        output_path = root_path / f"{objective_value_series.name}.xlsx"
        objective_value_series.to_excel(
            output_path, sheet_name=str(objective_value_series.name)
        )
        _logger.debug(
            f"Data {objective_value_series.name} saved under the path: {output_path}"
        )


class FeatherExporter(Exporter):
    """
    Class for exporting data to Feather format.

    This class provides functionality to export various data structures into Feather files.
    It ensures that data from groups and individual results are efficiently saved in a
    columnar format suitable for fast read and write operations.
    """

    @staticmethod
    def export_group_results(root_path: Path, result: ExportableResultsGroup) -> None:
        """
        Export exportable group results to Feather files.

        This method processes the fields of the provided results group, exporting each field's
        data into Feather format. It supports both individual DataFrames and dictionaries of
        DataFrames, ensuring each dataset is saved with appropriate naming conventions.

        Args:
            - root_path (Path): The root path for exporting the results.
            - result (ExportableResultsGroup): The exportable results to be parsed and exported.
        """
        if Exporter.is_results_group_empty(result):
            return
        for field_name, field_value in result.__dict__.items():
            output_path = root_path / field_name
            output_path.mkdir(parents=True, exist_ok=True)
            if isinstance(field_value, dict):
                for df_name, df in field_value.items():
                    df.columns = df.columns.astype(str)
                    df = df.reset_index()
                    df.to_feather(
                        output_path / f"{sanitize(df_name)}.feather", compression="lz4"
                    )
                    _logger.debug(
                        f"Data {df_name} saved under the path: {output_path / f'{sanitize(df_name)}.feather'}"
                    )
            else:
                field_value_df = field_value.reset_index()
                field_value_df.to_feather(
                    output_path / f"{field_name}.feather", compression="lz4"
                )
                _logger.debug(
                    f"Data {field_name} saved under the path: {output_path / f'{field_name}.feather'}"
                )

    @staticmethod
    def export_objective_result(
        root_path: Path, objective_value_series: pd.Series
    ) -> None:
        """
        Exports the given objective value series to a Feather file.

        This method converts a Pandas Series containing objective values into a DataFrame
        and saves it as a Feather file at the specified root path, facilitating efficient data access.

        Args:
            - root_path (Path): The root path where the Feather file will be saved.
            - objective_value_series (pd.Series): A Pandas Series containing objective values.
        """
        df = objective_value_series.to_frame().reset_index()
        df.to_feather(root_path / f"{objective_value_series.name}.feather")
