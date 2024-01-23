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

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from pyzefir.optimization.exportable_results import ExportableResultsGroup
from pyzefir.postprocessing.results_exporters import XlsxExporter
from pyzefir.postprocessing.results_handler import GeneralResultDirectory


def test_csv_exporter_export_dataframe(
    temporary_directory: Path, objective_result: pd.Series
) -> None:
    exporter = XlsxExporter()

    exporter.export_objective_result(temporary_directory, objective_result)

    assert (temporary_directory / f"{objective_result.name}.xlsx").is_file()
    exported_series = pd.read_excel(
        temporary_directory / f"{objective_result.name}.xlsx", index_col=0
    ).squeeze()
    assert_series_equal(exported_series, objective_result)


@pytest.mark.parametrize(
    "result, category",
    [
        (pytest.lazy_fixture("lines_results"), GeneralResultDirectory.LINES_RESULTS),
        (pytest.lazy_fixture("frac_results"), GeneralResultDirectory.FRACTIONS_RESULTS),
    ],
)
def test_csv_exporter_export_group_results(
    temporary_directory: Path,
    result: ExportableResultsGroup,
    category: GeneralResultDirectory,
) -> None:
    root_path = temporary_directory / category
    expected_paths_and_sheet_names = {
        root_path / f"{outer_key}.xlsx": list(inner_dict.keys())
        if isinstance(inner_dict, dict)
        else [outer_key]
        for outer_key, inner_dict in asdict(result).items()
    }

    XlsxExporter().export_group_results(root_path=root_path, result=result)

    for file_path, sheet_names in expected_paths_and_sheet_names.items():
        assert file_path.exists()
        assert file_path.is_file()
        exported_dfs_dict = pd.read_excel(
            file_path, sheet_name=None, index_col=0, header=[0, 1]
        )
        exported_df = next(iter(exported_dfs_dict.values()))
        assert exported_df.columns.levels[0].tolist() == sheet_names
        expected_dfs_dict = getattr(result, file_path.stem)
        if isinstance(expected_dfs_dict, pd.DataFrame):
            expected_dfs_dict = {file_path.stem: expected_dfs_dict}

        exported_df = {
            data_name: exported_df[data_name]
            for data_name in exported_df.columns.levels[0]
        }
        for (key1, value1), (key2, value2) in zip(
            exported_df.items(), expected_dfs_dict.items()
        ):
            assert key1 == key2
            assert_frame_equal(value1, value2)
