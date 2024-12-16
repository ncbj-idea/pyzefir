import os
import shutil
import tempfile
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


def test_csv_parser_convert_df_columns_to_string(csv_root_path: Path) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = os.path.join(temp_dir, "files")
        shutil.copytree(csv_root_path, temp_dir)
        dc_path = os.path.join(temp_dir, "demand_chunks", "dc.csv")
        df = pd.DataFrame(
            {
                "period_start": [0, 101],
                "period_end": [100, 8760],
                0: [20, 20000],
                1: [20, 20000],
                2: [20, 20000],
                3: [20, 20000],
            }
        )
        df.to_csv(dc_path, index=False)
        path_manager = CsvPathManager(Path(temp_dir), "scenario_1")
        parser = CsvParser(path_manager=path_manager).load_dfs()
        for inner_dict in parser.values():
            for df_value in inner_dict.values():
                assert all(
                    isinstance(column_name, str) for column_name in df_value.columns
                )
