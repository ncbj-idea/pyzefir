# PyZefir
# Copyright (C) 2024 Narodowe Centrum Badań Jądrowych
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
from pyzefir.postprocessing.results_exporters import FeatherExporter
from pyzefir.postprocessing.results_handler import GeneralResultDirectory


def test_feather_exporter_export_objective_result(
    temporary_directory: Path, objective_result: pd.Series
) -> None:
    exporter = FeatherExporter()
    exporter.export_objective_result(temporary_directory, objective_result)

    assert (temporary_directory / f"{objective_result.name}.feather").is_file()
    exported_series = pd.read_feather(
        temporary_directory / f"{objective_result.name}.feather"
    )
    exported_series = exported_series.set_index("index").squeeze()
    assert_series_equal(exported_series, objective_result, check_names=False)


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
        root_path / outer_key / f"{inner_key}.feather"
        for outer_key, inner_dict in results.__dict__.items()
        for inner_key in (
            inner_dict.keys() if isinstance(inner_dict, dict) else [outer_key]
        )
    ]

    FeatherExporter().export_group_results(root_path=root_path, result=results)

    for file_path in expected_paths:
        assert file_path.exists()
        assert file_path.is_file()
        exported_df = pd.read_feather(file_path).set_index("index")
        subcategory = file_path.parts[-2]
        result = getattr(results, subcategory)
        if isinstance(result, dict):
            assert_frame_equal(exported_df, result[file_path.stem], check_names=False)
        else:
            assert_frame_equal(result, exported_df, check_names=False)
