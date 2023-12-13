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
    """An interface for exporting data and parsing exportable results."""

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
    def __init__(self, exporter: Exporter) -> None:
        self._exporter = exporter

    @property
    def exporter(self) -> Exporter:
        return self._exporter

    @exporter.setter
    def exporter(self, new_exporter: Exporter) -> None:
        self._exporter = new_exporter

    def export_results(
        self, export_root_path: Path, results: ExportableResults
    ) -> None:
        for group in GeneralResultDirectory:
            self.exporter.export_group_results(
                export_root_path / group, getattr(results, group)
            )
        self.exporter.export_objective_result(export_root_path, results.objective_value)
