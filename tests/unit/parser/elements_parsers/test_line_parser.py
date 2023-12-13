from pathlib import Path

import pandas as pd
import pytest

from pyzefir.model.network_elements import Line
from pyzefir.parser.elements_parsers.line_parser import LineParser
from pyzefir.utils.path_manager import DataCategories, DataSubCategories


@pytest.fixture
def line_dataframe(csv_root_path: Path) -> pd.DataFrame:
    lines_df = pd.read_csv(
        csv_root_path / f"{DataCategories.STRUCTURE}/{DataSubCategories.LINES}.csv"
    )
    return lines_df


@pytest.fixture
def line_parser(line_dataframe: pd.DataFrame) -> LineParser:
    return LineParser(line_df=line_dataframe)


def test_line_parser_init(line_parser: LineParser) -> None:
    assert isinstance(line_parser.line_df, pd.DataFrame)


def test_line_parser_create_line(line_dataframe: pd.DataFrame) -> None:
    energy_types = {"HEAT", "ELECTRICITY"}
    row = line_dataframe.loc[1]
    name = row["name"]
    fr = row["bus_from"]
    to = row["bus_to"]
    transmission_loss = row["transmission_loss"]
    max_capacity = row["max_capacity"]
    transmission_fee = row["transmission_fee"]

    line = LineParser._create_line(row)

    assert isinstance(line, Line)
    assert line.name == name
    assert line.energy_type in energy_types
    assert line.fr == fr
    assert line.to == to
    assert line.transmission_loss == transmission_loss
    assert line.max_capacity == max_capacity
    assert line.transmission_fee == transmission_fee


def test_line_parser_create(
    line_parser: LineParser, line_dataframe: pd.DataFrame
) -> None:
    lines = line_parser.create()
    assert isinstance(lines, tuple)
    assert all(isinstance(line, Line) for line in lines)
    assert len(lines) == line_dataframe.shape[0]
