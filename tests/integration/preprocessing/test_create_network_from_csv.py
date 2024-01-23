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

import pandas as pd
import pytest

from pyzefir.model.network import Network, NetworkElementsDict
from pyzefir.parser.csv_parser import CsvParser
from pyzefir.parser.network_creator import NetworkCreator
from pyzefir.utils.path_manager import CsvPathManager, DataCategories, DataSubCategories


@pytest.fixture
def parsed_network(csv_root_path: Path) -> Network:
    df_dict = CsvParser(
        path_manager=CsvPathManager(csv_root_path, scenario_name="scenario_1")
    ).load_dfs()
    network = NetworkCreator.create(df_dict)
    return network


def test_random_network_elements(csv_root_path: Path, parsed_network: Network) -> None:
    df_buses = pd.read_csv(
        csv_root_path / f"{DataCategories.STRUCTURE}" / f"{DataSubCategories.BUSES}.csv"
    )
    df_lines = pd.read_csv(
        csv_root_path / f"{DataCategories.STRUCTURE}" / f"{DataSubCategories.LINES}.csv"
    )
    bus_series = df_buses.iloc[4]
    line_series = df_lines.iloc[2]

    test_bus = parsed_network.buses[bus_series["name"]]
    test_line = parsed_network.lines[line_series["name"]]

    assert isinstance(parsed_network, Network)
    assert test_bus.name == bus_series["name"]
    assert test_line.transmission_loss == line_series["transmission_loss"]
    assert test_line.max_capacity == line_series["max_capacity"]


@pytest.mark.parametrize(
    "network_attribute, df_path",
    (
        ("buses", f"{DataCategories.STRUCTURE}/{DataSubCategories.BUSES}.csv"),
        (
            "generators",
            f"{DataCategories.STRUCTURE}/{DataSubCategories.GENERATORS}.csv",
        ),
        ("lines", f"{DataCategories.STRUCTURE}/{DataSubCategories.LINES}.csv"),
        ("storages", f"{DataCategories.STRUCTURE}/{DataSubCategories.STORAGES}.csv"),
        (
            "local_balancing_stacks",
            f"{DataCategories.STRUCTURE}/{DataSubCategories.TECHNOLOGYSTACKS_BUSES_OUT}.csv",
        ),
        (
            "aggregated_consumers",
            f"{DataCategories.STRUCTURE}/{DataSubCategories.AGGREGATES}.csv",
        ),
        ("fuels", f"{DataCategories.FUELS}/{DataSubCategories.ENERGY_PER_UNIT}.csv"),
    ),
)
def test_parsed_network_structure(
    network_attribute: str, df_path: str, csv_root_path: Path, parsed_network: Network
) -> None:
    network_elements_dict = getattr(parsed_network, network_attribute)
    element_df = pd.read_csv(csv_root_path / df_path)
    assert isinstance(network_elements_dict, NetworkElementsDict)
    assert all(
        str(name) in element_df["name"].values for name in network_elements_dict.keys()
    )
    assert len(network_elements_dict) == element_df.shape[0]


def test_network_energy_types(csv_root_path: Path, parsed_network: Network) -> None:
    df_energy_types = pd.read_csv(
        csv_root_path
        / f"{DataCategories.STRUCTURE}"
        / f"{DataSubCategories.ENERGY_TYPES}.csv"
    )

    assert isinstance(parsed_network._energy_types, list)
    assert len(parsed_network.energy_types) == df_energy_types.shape[0]
    assert all(
        energy_type in df_energy_types["name"].values
        for energy_type in parsed_network._energy_types
    )
