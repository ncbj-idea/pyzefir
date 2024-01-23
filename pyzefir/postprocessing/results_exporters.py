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
from sanitize_filename import sanitize

from pyzefir.optimization.exportable_results import ExportableResultsGroup
from pyzefir.postprocessing.results_handler import Exporter

_logger = logging.getLogger(__name__)


class CsvExporter(Exporter):
    """Class for exporting data to Csv format."""

    @staticmethod
    def export_group_results(root_path: Path, result: ExportableResultsGroup) -> None:
        """
        Export exportable group results to Csv files.

        Args:
            root_path (Path): The root path for exporting the results.
            result (ExportableResultsGroup): The exportable results to be parsed and exported.

        Returns:
            None
        """
        for field_name, field_value in result.__dict__.items():
            output_path = root_path / field_name
            output_path.mkdir(parents=True, exist_ok=True)
            if isinstance(field_value, dict):
                for df_name, df in field_value.items():
                    df.to_csv(output_path / f"{sanitize(df_name)}.csv")
                    _logger.debug(
                        f"Data {df_name} saved under the path: {output_path / f'{sanitize(df_name)}.csv'} "
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
        Exports the given objective value series to a Csv file.

        Parameters:
            root_path (Path): The root path where the Csv file will be saved.
            objective_value_series (pd.Series): A Pandas Series containing objective values.

        Returns:
            None
        """
        objective_value_series.to_csv(root_path / f"{objective_value_series.name}.csv")


class XlsxExporter(Exporter):
    """Class for exporting data to XLSX format."""

    @staticmethod
    def export_group_results(root_path: Path, result: ExportableResultsGroup) -> None:
        """
        Export exportable group results to Xlsx files.

        Args:
            root_path (Path): The root path for exporting the results.
            result (ExportableResultsGroup): The exportable results to be parsed and exported.

        Returns:
            None
        """
        root_path.mkdir(parents=True, exist_ok=True)
        for field_name, field_value in result.__dict__.items():
            with pd.ExcelWriter(
                (root_path / field_name).with_suffix(".xlsx"), engine="xlsxwriter"
            ) as writer:
                if isinstance(field_value, dict):
                    df = pd.concat(field_value, axis=1).sort_index(axis=1)
                else:
                    df = field_value
                df.to_excel(writer, sheet_name=field_name)
                _logger.debug(
                    f"Data {field_name} saved under the path: {(root_path / field_name).with_suffix('.xlsx')} "
                )

    @staticmethod
    def export_objective_result(
        root_path: Path, objective_value_series: pd.Series
    ) -> None:
        """
        Exports the given objective value series to a Xlsx file.

        Parameters:
            root_path (Path): The root path where the Xlsx file will be saved.
            objective_value_series (pd.Series): A Pandas Series containing objective values.

        Returns:
            None
        """
        output_path = root_path / f"{objective_value_series.name}.xlsx"
        objective_value_series.to_excel(
            output_path, sheet_name=str(objective_value_series.name)
        )
        _logger.debug(
            f"Data {objective_value_series.name} saved under the path: {output_path}"
        )
