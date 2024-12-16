from abc import ABC, abstractmethod
from enum import StrEnum, auto
from pathlib import Path

import pandas as pd

from pyzefir.optimization.exportable_results import (
    ExportableResults,
    ExportableResultsGroup,
)


class GeneralResultDirectory(StrEnum):
    GENERATORS_RESULTS = auto()
    STORAGES_RESULTS = auto()
    LINES_RESULTS = auto()
    FRACTIONS_RESULTS = auto()
    BUS_RESULTS = auto()


class Exporter(ABC):
    """
    An interface for exporting data and parsing exportable results.

    This abstract class defines the necessary methods for exporting different formats of
    data and determining the status of the results groups. Implementing classes must provide
    concrete implementations of the export methods to handle specific export formats.
    """

    @staticmethod
    def is_results_group_empty(result: ExportableResultsGroup) -> bool:
        for field_value in result.__dict__.values():
            if isinstance(field_value, dict) and field_value:
                return False
            if isinstance(field_value, pd.DataFrame) and not field_value.empty:
                return False
        return True

    @staticmethod
    @abstractmethod
    def export_group_results(root_path: Path, result: ExportableResultsGroup) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def export_objective_result(
        root_path: Path, objective_value_series: pd.Series
    ) -> None:
        raise NotImplementedError


class ResultsHandler:
    """
    Handles exporting of results using a specified exporter.

    This class is responsible for managing the export of results by delegating the actual
    export functionality to an instance of an exporter class. It provides a flexible way to
    switch between different export formats by changing the exporter.
    """

    def __init__(self, exporter: Exporter) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - exporter (Exporter): An instance of a class that implements the Exporter interface.
        """
        self._exporter = exporter

    @property
    def exporter(self) -> Exporter:
        """
        Gets the current exporter.

        Returns:
            - Exporter: The current exporter instance.
        """
        return self._exporter

    @exporter.setter
    def exporter(self, new_exporter: Exporter) -> None:
        """
        Sets a new exporter instance.

        Args:
            - new_exporter (Exporter): The new exporter instance to set.
        """
        self._exporter = new_exporter

    def export_results(
        self, export_root_path: Path, results: ExportableResults
    ) -> None:
        """
        Exports the results to the specified root path.

        This method iterates over the predefined result groups and uses the current exporter
        to export each group's results. Additionally, it exports the objective result to the
        specified path.

        Args:
            - export_root_path (Path): The root path for exporting the results.
            - results (ExportableResults): The results object containing the data to export.
        """
        for group in GeneralResultDirectory:
            self.exporter.export_group_results(
                export_root_path / group, getattr(results, group)
            )
        self.exporter.export_objective_result(export_root_path, results.objective_value)
