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
from pandas.testing import assert_frame_equal, assert_series_equal

from pyzefir.optimization.exportable_results import ExportableResultsGroup
from pyzefir.postprocessing.results_exporters import CsvExporter
from pyzefir.postprocessing.results_handler import GeneralResultDirectory


def test_csv_exporter_export_objective_result(
    temporary_directory: Path, objective_result: pd.Series
) -> None:
    exporter = CsvExporter()

    exporter.export_objective_result(temporary_directory, objective_result)

    assert (temporary_directory / f"{objective_result.name}.csv").is_file()
    exported_series = pd.read_csv(
        temporary_directory / f"{objective_result.name}.csv", index_col=0
    ).squeeze()
    assert_series_equal(exported_series, objective_result)


@pytest.mark.parametrize(
    "results, category",
    [
        (
            pytest.lazy_fixture("generators_results"),
            GeneralResultDirectory.GENERATORS_RESULTS,
        ),
        (
            pytest.lazy_fixture("storages_results"),
            GeneralResultDirectory.STORAGES_RESULTS,
        ),
        (pytest.lazy_fixture("lines_results"), GeneralResultDirectory.LINES_RESULTS),
        (pytest.lazy_fixture("frac_results"), GeneralResultDirectory.FRACTIONS_RESULTS),
    ],
)
def test_csv_exporter_export_group_results(
    temporary_directory: Path,
    results: ExportableResultsGroup,
    category: GeneralResultDirectory,
) -> None:
    root_path = temporary_directory / category
    expected_paths = [
        root_path / outer_key / f"{inner_key}.csv"
        for outer_key, inner_dict in results.__dict__.items()
        for inner_key in (
            inner_dict.keys() if isinstance(inner_dict, dict) else [outer_key]
        )
    ]

    CsvExporter().export_group_results(root_path=root_path, result=results)

    for file_path in expected_paths:
        assert file_path.exists()
        assert file_path.is_file()
        exported_df = pd.read_csv(file_path, index_col=0)
        subcategory = file_path.parts[-2]
        result = getattr(results, subcategory)
        if isinstance(result, dict):
            assert_frame_equal(exported_df, result[file_path.stem])
        else:
            assert_frame_equal(result, exported_df)
