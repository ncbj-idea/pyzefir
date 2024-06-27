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
