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

from pathlib import Path

import click

from pyzefir.structure_creator.constants_enums import SubDirectory
from pyzefir.structure_creator.input_data import InputData
from pyzefir.structure_creator.scenario.main import create_scenario
from pyzefir.structure_creator.structure_and_initial_state.main import (
    create_structure_and_initial,
)


def create_structure(
    input_path: str | Path,
    output_path: str | Path,
    scenario_name: str,
    n_hours: int,
    n_years: int,
) -> None:
    input_data = InputData.load_input_data(
        input_path=Path(input_path) / SubDirectory.structure_creator_resources,
        scenario_name=scenario_name,
        n_hours=n_hours,
        n_years=n_years,
    )
    create_structure_and_initial(
        structure_data=input_data.structure_data, output_path=Path(output_path)
    )
    create_scenario(
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
def run_structure_creator_cli(
    input_path: str | Path,
    output_path: str | Path,
    scenario_name: str,
    n_hours: int,
    n_years: int,
) -> None:
    create_structure(
        input_path=input_path,
        output_path=output_path,
        scenario_name=scenario_name,
        n_hours=n_hours,
        n_years=n_years,
    )
