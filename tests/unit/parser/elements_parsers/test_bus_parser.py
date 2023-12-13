from pathlib import Path

import pandas as pd
import pytest

from pyzefir.model.network_elements import Bus
from pyzefir.parser.elements_parsers.bus_parser import BusParser


def test_bus_parser_create(csv_root_path: Path) -> None:
    bus_df = pd.read_csv(csv_root_path / "structure/Buses.csv")
    buses_names = bus_df["name"].to_list()
    buses_energy_types = bus_df["energy_type"].to_list()
    bus_parser = BusParser(bus_df)

    buses = bus_parser.create()

    assert len(buses) == bus_df.shape[0]
    assert all(isinstance(bus, Bus) for bus in buses)
    for bus, excepted_name, excepted_energy_type in zip(
        buses, buses_names, buses_energy_types
    ):
        assert bus.name == excepted_name
        assert bus.energy_type == excepted_energy_type


def test_bus_parser_create_empty_dataframe() -> None:
    empty_df = pd.DataFrame(columns=["name", "energy_type"])
    bus_parser = BusParser(empty_df)
    buses = bus_parser.create()

    assert not buses
    assert len(buses) == 0


def test_bus_parser_create_invalid_df() -> None:
    invalid_df = pd.DataFrame(
        {"name": ["KSE_12", "BFA_KSE"], "type_en": ["HEAT", "HEAT"]}
    )
    bus_parser = BusParser(invalid_df)

    with pytest.raises(KeyError):
        bus_parser.create()
