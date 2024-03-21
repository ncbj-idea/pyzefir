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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pyzefir.structure_creator.data_loader.constants_enums import (
    SubDirectory,
    XlsxFileName,
)


@dataclass
class ScenarioData:
    """Class containing input data needed to generate scenario excel."""

    cost_parameters: dict[str, pd.DataFrame]
    fuel_parameters: dict[str, pd.DataFrame]
    relative_emission_limits: pd.DataFrame
    n_consumers: pd.DataFrame
    technology_cap_limits: dict[str, pd.DataFrame]
    technology_type_cap_limits: dict[str, pd.DataFrame]
    yearly_demand: dict[str, pd.DataFrame]
    fractions: dict[str, dict[str, pd.DataFrame]]
    generation_fraction: pd.DataFrame

    @staticmethod
    def _load_fractions(input_path: Path) -> dict[str, dict[str, pd.DataFrame]]:
        result = dict()
        fractions_directory = input_path / SubDirectory.fractions
        for element_path in fractions_directory.iterdir():
            if element_path.is_file() and element_path.suffix == ".xlsx":
                result[element_path.stem] = pd.read_excel(element_path, sheet_name=None)

        return result

    @staticmethod
    def load_scenario_data(input_path: Path) -> ScenarioData:
        return ScenarioData(
            cost_parameters=pd.read_excel(
                input_path / XlsxFileName.cost_parameters, sheet_name=None
            ),
            fuel_parameters=pd.read_excel(
                input_path / XlsxFileName.fuel_parameters, sheet_name=None
            ),
            n_consumers=pd.read_excel(input_path / XlsxFileName.n_consumers),
            relative_emission_limits=pd.read_excel(
                input_path / XlsxFileName.relative_emission_limits
            ),
            technology_cap_limits=pd.read_excel(
                input_path / XlsxFileName.technology_cap_limits, sheet_name=None
            ),
            technology_type_cap_limits=pd.read_excel(
                input_path / XlsxFileName.technology_type_cap_limits, sheet_name=None
            ),
            yearly_demand=pd.read_excel(
                input_path / XlsxFileName.yearly_demand, sheet_name=None
            ),
            fractions=ScenarioData._load_fractions(input_path),
            generation_fraction=pd.read_excel(
                input_path / XlsxFileName.generation_fraction
            ),
        )


@dataclass
class InputStructureData:
    """Class containing input data needed to generate structure.xlsx and initial_state.xlsx excels."""

    lbs_type: dict[str, dict[str, pd.DataFrame]]
    subsystem: dict[str, pd.DataFrame]
    aggregates: pd.DataFrame
    configuration: dict[str, pd.DataFrame]
    emission: dict[str, pd.DataFrame]
    transmission_fee: pd.DataFrame
    n_hours: int
    n_years: int

    @staticmethod
    def _load_lbs_files(lbs_path: Path) -> dict[str, dict[str, pd.DataFrame]]:
        return {
            element_path.stem: pd.read_excel(element_path, sheet_name=None)
            for element_path in lbs_path.iterdir()
            if element_path.is_file()
        }

    @staticmethod
    def load_structure_data(
        input_path: Path, n_hours: int, n_years: int
    ) -> InputStructureData:
        return InputStructureData(
            lbs_type=InputStructureData._load_lbs_files(input_path / SubDirectory.lbs),
            subsystem=pd.read_excel(
                input_path / XlsxFileName.subsystems, sheet_name=None
            ),
            aggregates=pd.read_excel(input_path / XlsxFileName.aggregates),
            configuration=pd.read_excel(
                input_path / XlsxFileName.configuration, sheet_name=None
            ),
            emission=pd.read_excel(
                input_path / XlsxFileName.emissions, sheet_name=None
            ),
            transmission_fee=pd.read_excel(input_path / XlsxFileName.transmission_fees),
            n_hours=n_hours,
            n_years=n_years,
        )


@dataclass
class InputData:
    """Class containing loaded input data needed to generate model input."""

    scenario_data: ScenarioData
    structure_data: InputStructureData

    @staticmethod
    def load_input_data(
        input_path: Path, scenario_name: str, n_hours: int, n_years: int
    ) -> InputData:
        return InputData(
            scenario_data=ScenarioData.load_scenario_data(
                input_path / SubDirectory.scenarios / scenario_name
            ),
            structure_data=InputStructureData.load_structure_data(
                input_path, n_hours, n_years
            ),
        )
