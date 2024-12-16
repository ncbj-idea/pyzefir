from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pyzefir.structure_creator.data_loader.constants_enums import (
    SubDirectory,
    XlsxFileName,
)

_logger = logging.getLogger(__name__)


class ScenarioDataError(Exception):
    pass


@dataclass
class ScenarioData:
    """
    Class containing input data needed to generate scenario Excel files.

    This class loads and organizes all necessary input files required to generate a scenario.
    It ensures that all required files are present and correctly formatted before processing them into data frames.
    """

    cost_parameters: dict[str, pd.DataFrame]
    """The cost parameter data for the scenario."""
    fuel_parameters: dict[str, pd.DataFrame]
    """The fuel parameter data for the scenario."""
    relative_emission_limits: pd.DataFrame
    """The emission limits data."""
    n_consumers: pd.DataFrame
    """Data about the number of consumers."""
    technology_cap_limits: dict[str, pd.DataFrame]
    """Technology capacity limit data."""
    technology_type_cap_limits: dict[str, pd.DataFrame]
    """Technology type capacity limit data."""
    yearly_demand: dict[str, pd.DataFrame]
    """Yearly demand data."""
    fractions: dict[str, dict[str, pd.DataFrame]]
    """Fractions data."""
    generation_fraction: pd.DataFrame
    """Generation fraction data."""
    generation_compensation: pd.DataFrame
    """Compensation data for generation."""
    yearly_emission_reduction: pd.DataFrame
    """Data on yearly emission reduction goals."""
    ens_penalization: pd.DataFrame
    """Data on energy-not-supplied penalization."""

    @staticmethod
    def validate_input_files(input_path: Path) -> None:
        """
        Validate scenario input files.

        This method checks if all required scenario files exist and are correctly formatted.

        Args:
            - input_path (Path): Path to the input files.

        Raises:
            - ScenarioDataError: If any required files are missing or have an incorrect format.
        """
        _logger.debug("Start validate if all required scenario input files exist")
        required_files = [
            XlsxFileName.cost_parameters,
            XlsxFileName.fuel_parameters,
            XlsxFileName.n_consumers,
            XlsxFileName.relative_emission_limits,
            XlsxFileName.technology_cap_limits,
            XlsxFileName.technology_type_cap_limits,
            XlsxFileName.yearly_demand,
            XlsxFileName.generation_fraction,
            XlsxFileName.generation_compensation,
            XlsxFileName.ens_penalization,
        ]
        fractions_directory = input_path / SubDirectory.fractions
        for element_path in fractions_directory.iterdir():
            if not element_path.is_file() and not element_path.suffix == ".xlsx":
                raise ScenarioDataError(
                    f"Given file {element_path} is not a required fraction xlsx file"
                )

        for file_name in required_files:
            file_path = input_path / file_name
            if not file_path.is_file():
                raise ScenarioDataError(
                    f"Given file {file_path} is not a required scenario xlsx file"
                )
        _logger.debug("All of required scenario input files exist")

    @staticmethod
    def _load_fractions(input_path: Path) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Load fraction from input files.

        Args:
            - input_path (Path): path to input

        Returns:
            - dict[str, dict[str, pd.DataFrame]]: loaded fractions
        """
        _logger.debug("Starting loading fractions data input files ...")
        result = dict()
        fractions_directory = input_path / SubDirectory.fractions
        for element_path in fractions_directory.iterdir():
            if element_path.is_file() and element_path.suffix == ".xlsx":
                result[element_path.stem] = pd.read_excel(element_path, sheet_name=None)
        _logger.debug("Fractions data input files loaded.")
        return result

    @staticmethod
    def load_scenario_data(input_path: Path) -> ScenarioData:
        """
        Load scenario data from input files.

        This method loads all scenario data including cost parameters, fuel parameters, emission limits,
        technology capacity limits, and other necessary components, ensuring all files are present.

        Args:
            - input_path (Path): Path to the scenario input files.

        Returns:
            - ScenarioData: Loaded scenario data class.
        """
        ScenarioData.validate_input_files(input_path)
        _logger.debug("Starting loading scenario input files ...")
        data = ScenarioData(
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
            generation_compensation=pd.read_excel(
                input_path / XlsxFileName.generation_compensation
            ),
            yearly_emission_reduction=(
                pd.read_excel(input_path / XlsxFileName.yearly_emission_reduction)
                if (input_path / XlsxFileName.yearly_emission_reduction).exists()
                else pd.DataFrame()
            ),
            ens_penalization=pd.read_excel(input_path / XlsxFileName.ens_penalization),
        )
        _logger.debug("Scenario input files loaded.")
        return data


@dataclass
class InputStructureData:
    """
    Class containing input data needed to generate structure.xlsx and initial_state.xlsx files.

    This class loads, validates, and organizes all necessary input data required to create the
    structure and initial state Excel files for the model. It ensures that all necessary files
    are present and correctly formatted before loading the data into memory.
    """

    lbs_type: dict[str, dict[str, pd.DataFrame]]
    """The loaded lbs files."""
    subsystem: dict[str, pd.DataFrame]
    """The subsystem data in DataFrame format."""
    aggregates: pd.DataFrame
    """The aggregate data."""
    configuration: dict[str, pd.DataFrame]
    """The configuration data."""
    emission: dict[str, pd.DataFrame]
    """The emission data."""
    transmission_fee: pd.DataFrame
    """The transmission fee data."""
    n_hours: int
    """Number of hours."""
    n_years: int
    """Number of years."""

    @staticmethod
    def validate_input_files(input_path: Path) -> None:
        """
        Validate input data structure.

        This method checks for the existence of required input files in the correct format
        and raises a ScenarioDataError if any of them are missing or incorrectly formatted.

        Args:
            - input_path (Path): Path to the input directory.
        """
        _logger.debug("Start validate if all required structure input files exist")
        required_files = [
            XlsxFileName.subsystems,
            XlsxFileName.aggregates,
            XlsxFileName.configuration,
            XlsxFileName.emissions,
            XlsxFileName.transmission_fees,
        ]

        lbs_path = input_path / SubDirectory.lbs
        for element_path in lbs_path.iterdir():
            if not element_path.is_file() and not element_path.suffix == ".xlsx":
                raise ScenarioDataError(
                    f"Given file {element_path} is not a required lbs xlsx file"
                )

        for file_name in required_files:
            file_path = input_path / file_name
            if not file_path.is_file():
                raise ScenarioDataError(
                    f"Given file {element_path} is not a required structure xlsx file"
                )
        _logger.debug("All of required structure input files exist")

    @staticmethod
    def _load_lbs_files(lbs_path: Path) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Load lbs files from input data.

        Args:
            - lbs_path (Path): Path to lbs files

        Returns:
            - dict[str, dict[str, pd.DataFrame]]: loaded lbs files
        """
        return {
            element_path.stem: pd.read_excel(element_path, sheet_name=None)
            for element_path in lbs_path.iterdir()
            if element_path.is_file()
        }

    @staticmethod
    def load_structure_data(
        input_path: Path, n_hours: int, n_years: int
    ) -> InputStructureData:
        """
        Load structure data from input files.

        This method loads the necessary input files for the structure, including lbs files,
        subsystems, aggregates, configuration, emissions, and transmission fee data, and
        validates their existence and format.

        Args:
            - input_path (Path): The path to the input files.
            - n_hours (int): Number of hours to consider.
            - n_years (int): Number of years to consider.

        Returns:
            - InputStructureData: The loaded structure data.
        """
        InputStructureData.validate_input_files(input_path)
        _logger.debug("Starting loading structure input files ...")
        data = InputStructureData(
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
        _logger.debug("Structure input files loaded.")
        return data


@dataclass
class InputData:
    """Class containing loaded input data needed to generate model input.

    This class manages the loading of both scenario and structure data, ensuring that
    all necessary files are present and properly formatted for model generation. It
    provides a unified way to access the data needed for further processing.
    """

    scenario_data: ScenarioData
    """The scenario data needed for the model."""
    structure_data: InputStructureData
    """The structure data needed for the model."""

    @staticmethod
    def load_input_data(
        input_path: Path, scenario_name: str, n_hours: int, n_years: int
    ) -> InputData:
        """
        Loads both scenario and structure data from input files.

        This method combines both the scenario and structure data loaders, validating
        input files and ensuring all necessary data is available for model creation.

        Args:
            - input_path (Path): The path to the input data.
            - scenario_name (str): The name of the scenario to be loaded.
            - n_hours (int): Number of hours to consider.
            - n_years (int): Number of years to consider.

        Returns:
            - InputData: The loaded input data class.
        """
        _logger.debug("Starting creating InputData class instance ...")
        input_data = InputData(
            scenario_data=ScenarioData.load_scenario_data(
                input_path / SubDirectory.scenarios / scenario_name
            ),
            structure_data=InputStructureData.load_structure_data(
                input_path, n_hours, n_years
            ),
        )
        _logger.debug("Input Data created.")
        return input_data
