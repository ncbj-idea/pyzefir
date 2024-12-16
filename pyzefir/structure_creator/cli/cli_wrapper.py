import logging
from pathlib import Path

import click

from pyzefir.cli.logger import LOG_LEVEL_MAPPING, setup_logging
from pyzefir.structure_creator.data_loader.constants_enums import SubDirectory
from pyzefir.structure_creator.data_loader.input_data import InputData
from pyzefir.structure_creator.scenario.main import create_scenario
from pyzefir.structure_creator.structure_and_initial_state.create_structures import (
    StructureCreator,
)

_logger = logging.getLogger(__name__)


def create_structure(
    input_path: str | Path,
    output_path: str | Path,
    scenario_name: str,
    n_hours: int,
    n_years: int,
) -> None:
    """
    Loads input data, creates capacity bounds, and generates a scenario.

    This function reads the input data from the specified path, creates the capacity
    bounds, and sets up the initial structure. It then creates a scenario using the
    provided scenario data, saving the results to the output path.

    Args:
        - input_path (str | Path): The path to the input data for the scenario.
        - output_path (str | Path): The path where the results will be saved.
        - scenario_name (str): The name of the scenario to be created.
        - n_hours (int): The number of hours to be considered in the scenario.
        - n_years (int): The number of years to be considered in the scenario.
    """
    _logger.info("Loading input data...")
    input_data = InputData.load_input_data(
        input_path=Path(input_path),
        scenario_name=scenario_name,
        n_hours=n_hours,
        n_years=n_years,
    )
    _logger.info("Creating structure and initial setup...")
    capacity_bounds_df = StructureCreator.create_structure_and_initial(
        input_structure=input_data.structure_data, output_path=Path(output_path)
    )
    _logger.info("Creating scenario...")
    create_scenario(
        capacity_bounds_df=capacity_bounds_df,
        scenario_data=input_data.scenario_data,
        output_path=Path(output_path) / SubDirectory.scenarios,
        scenario_name=scenario_name,
        n_hours=input_data.structure_data.n_hours,
        n_years=input_data.structure_data.n_years,
    )


@click.command()
@click.option(
    "-i",
    "--input_path",
    type=click.Path(exists=True),
    required=True,
    help="Input data for the creator.",
)
@click.option(
    "-o",
    "--output_path",
    type=click.Path(exists=False),
    required=True,
    help="Path to dump the results.",
)
@click.option(
    "-s",
    "--scenario_name",
    type=str,
    required=True,
    help="Name of the scenario.",
)
@click.option(
    "-h",
    "--n_hours",
    type=int,
    required=False,
    default=8760,
    help="N_HOURS constant.",
)
@click.option(
    "-y",
    "--n_years",
    type=int,
    required=False,
    default=20,
    help="N_YEARS constant.",
)
@click.option(
    "-l",
    "--logger_level",
    type=str,
    required=False,
    default="INFO",
    help="Logger level",
)
def run_structure_creator_cli(
    input_path: str | Path,
    output_path: str | Path,
    scenario_name: str,
    n_hours: int,
    n_years: int,
    logger_level: str,
) -> None:
    """
    Command-line interface (CLI) for running the structure creator.

    This CLI function sets up logging based on the given logger level, loads input data,
    and triggers the structure creation process. It then creates the scenario and saves
    the results to the output path.

    Args:
        - input_path (str | Path): The path to the input data required by the creator.
        - output_path (str | Path): The path where the results will be saved.
        - scenario_name (str): The name of the scenario to be created.
        - n_hours (int): The number of hours to be considered (default is 8760).
        - n_years (int): The number of years to be considered (default is 20).
        - logger_level (str): The logging level for the process (default is "INFO").
    """
    if (log_level := LOG_LEVEL_MAPPING.get(logger_level.lower())) is not None:
        setup_logging(level=log_level)
    else:
        setup_logging()
    create_structure(
        input_path=input_path,
        output_path=output_path,
        scenario_name=scenario_name,
        n_hours=n_hours,
        n_years=n_years,
    )
    _logger.info("Structure creator ended its operation ....")
