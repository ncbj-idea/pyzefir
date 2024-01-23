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
