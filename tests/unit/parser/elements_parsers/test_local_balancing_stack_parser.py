from pathlib import Path

import pandas as pd
import pytest

from pyzefir.model.network_elements import LocalBalancingStack
from pyzefir.parser.elements_parsers.local_balancing_stack_parser import (
    LocalBalancingStackParser,
)
from pyzefir.utils.path_manager import DataCategories, DataSubCategories


@pytest.fixture
def stack_dataframe(csv_root_path: Path) -> pd.DataFrame:
    stacks_df = pd.read_csv(
        csv_root_path
        / f"{DataCategories.STRUCTURE}/{DataSubCategories.TECHNOLOGYSTACKS_BUSES_OUT}.csv"
    )
    return stacks_df


@pytest.fixture
def bus_dataframe(csv_root_path: Path) -> pd.DataFrame:
    stacks_df = pd.read_csv(
        csv_root_path / f"{DataCategories.STRUCTURE}/{DataSubCategories.BUSES}.csv"
    )
    return stacks_df


@pytest.fixture
def technologystack_buses(csv_root_path: Path) -> pd.DataFrame:
    stacks_df = pd.read_csv(
        csv_root_path
        / f"{DataCategories.STRUCTURE}/{DataSubCategories.TECHNOLOGYSTACK_BUSES}.csv"
    )
    return stacks_df


@pytest.fixture
def stack_parser(
    stack_dataframe: pd.DataFrame,
    bus_dataframe: pd.DataFrame,
    technologystack_buses: pd.DataFrame,
) -> LocalBalancingStackParser:
    return LocalBalancingStackParser(
        stack_df=stack_dataframe,
        bus_df=bus_dataframe,
        stack_bus_df=technologystack_buses,
    )


@pytest.mark.parametrize(
    ("bus_df", "stack_bus_df", "expected_dict"),
    [
        pytest.param(
            pd.DataFrame(
                {"name": ["bus_1", "bus_2"], "energy_type": ["ELECTRICITY", "HEAT"]}
            ),
            pd.DataFrame(
                {
                    "technology_stack": ["LKT_1", "LKT_2"],
                    "bus": ["bus_1", "bus_2"],
                }
            ),
            {
                "LKT_1": {"ELECTRICITY": {"bus_1"}},
                "LKT_2": {"HEAT": {"bus_2"}},
            },
            id="1 bus per lbs diff et",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["bus_1", "bus_2", "bus_3", "bus_4"],
                    "energy_type": ["ELECTRICITY", "HEAT", "HEAT", "ELECTRICITY"],
                }
            ),
            pd.DataFrame(
                {
                    "technology_stack": ["LKT_1", "LKT_1", "LKT_2", "LKT_2"],
                    "bus": ["bus_1", "bus_3", "bus_2", "bus_4"],
                }
            ),
            {
                "LKT_1": {"ELECTRICITY": {"bus_1"}, "HEAT": {"bus_3"}},
                "LKT_2": {"ELECTRICITY": {"bus_4"}, "HEAT": {"bus_2"}},
            },
            id="2 bus per LKT diff et",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["bus_1", "bus_2", "bus_3", "bus_4"],
                    "energy_type": ["ELECTRICITY", "HEAT", "WIND", "ELECTRICITY"],
                }
            ),
            pd.DataFrame(
                {
                    "technology_stack": ["LKT_1", "LKT_1", "LKT_2", "LKT_2"],
                    "bus": ["bus_1", "bus_3", "bus_2", "bus_4"],
                }
            ),
            {
                "LKT_1": {"ELECTRICITY": {"bus_1"}, "WIND": {"bus_3"}},
                "LKT_2": {"ELECTRICITY": {"bus_4"}, "HEAT": {"bus_2"}},
            },
            id="2 bus per LKT wind et",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["bus_1", "bus_2", "bus_3", "bus_4"],
                    "energy_type": [
                        "ELECTRICITY",
                        "HEAT",
                        "ELECTRICITY",
                        "ELECTRICITY",
                    ],
                }
            ),
            pd.DataFrame(
                {
                    "technology_stack": ["LKT_1", "LKT_1", "LKT_2", "LKT_2"],
                    "bus": ["bus_1", "bus_3", "bus_2", "bus_4"],
                }
            ),
            {
                "LKT_1": {"ELECTRICITY": {"bus_1", "bus_3"}},
                "LKT_2": {"ELECTRICITY": {"bus_4"}, "HEAT": {"bus_2"}},
            },
            id="LKT_1 same energy",
        ),
    ],
)
def test_prepare_stack_buses_mapping(
    bus_df: pd.DataFrame,
    stack_bus_df: pd.DataFrame,
    expected_dict: dict[str, dict[str, set[str]]],
) -> None:
    result = LocalBalancingStackParser._prepare_stack_buses_mapping(
        bus_df, stack_bus_df
    )
    assert result == expected_dict


def test_stack_parser_init(stack_parser: LocalBalancingStackParser) -> None:
    assert isinstance(stack_parser.stack_df, pd.DataFrame)
    assert isinstance(stack_parser.stack_buses_mapping, dict)
    assert all(isinstance(key, str) for key in stack_parser.stack_buses_mapping.keys())
    assert all(
        isinstance(value, dict) for value in stack_parser.stack_buses_mapping.values()
    )
    for value in stack_parser.stack_buses_mapping.values():
        for en_type, buses in value.items():
            assert isinstance(en_type, str)
            assert isinstance(buses, set)
            assert all(isinstance(bus, str) for bus in buses)


def test_stack_parser_create_stack(
    stack_parser: LocalBalancingStackParser, stack_dataframe: pd.DataFrame
) -> None:
    energy_types = {"ELECTRICITY", "HEAT"}
    row = stack_dataframe.loc[1]
    name = row["name"]
    row_buses = [row["ELECTRICITY"], row["HEAT"]]

    stack = stack_parser._create_stack(row)

    assert isinstance(stack, LocalBalancingStack)
    assert stack.name == name
    assert len(stack.buses_out) == len(energy_types)
    assert len(stack.buses) == len(energy_types)
    for stack_bus_object in [stack.buses_out, stack.buses]:
        for key, value in stack_bus_object.items():
            assert isinstance(key, str)
            assert isinstance(value, (str, set))
            assert key in energy_types
            if isinstance(value, set):
                assert all(val in row_buses for val in value)
            else:
                assert value in row_buses


def test_stack_parser_create(
    stack_parser: LocalBalancingStackParser, stack_dataframe: pd.DataFrame
) -> None:
    stacks = stack_parser.create()
    assert isinstance(stacks, tuple)
    assert all(isinstance(stack, LocalBalancingStack) for stack in stacks)
    assert len(stacks) == stack_dataframe.shape[0]
