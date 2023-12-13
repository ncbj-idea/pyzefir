from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest

from pyzefir.parser.csv_parser import CsvParser, CsvParserException
from pyzefir.utils.path_manager import CsvPathManager, DataCategories


@pytest.fixture
def path_manager(csv_root_path: Path) -> CsvPathManager:
    return CsvPathManager(csv_root_path, scenario_name="scenario_1")


def read_csv(csv_name: str, csv_root_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_root_path / csv_name)


def test_csv_parser_load_dataframes_lfs(path_manager: CsvPathManager) -> None:
    data = CsvParser(path_manager=path_manager).load_dfs()
    assert data
    assert isinstance(data, dict)
    for category_name in fields(DataCategories):
        assert category_name.default in data
        for sub_category_name in data[category_name.default]:
            assert isinstance(
                data[category_name.default][sub_category_name], pd.DataFrame
            )


def test_csv_parser_load_dataframe_lack_of_df() -> None:
    path_manager = CsvPathManager(Path("/tmp/output/"))
    parser = CsvParser(path_manager=path_manager)
    with pytest.raises(CsvParserException):
        parser.load_dfs()
