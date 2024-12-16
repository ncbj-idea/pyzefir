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

import pandas as pd
import pytest
from bidict import bidict

from pyzefir.optimization.linopy.utils import calculate_storage_adjusted_generation


@pytest.mark.parametrize(
    "generation_result_df, storages_generation_efficiency, storages_idxs, expected_output",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "solution": [0, 0, 1761, 0, 0, 0],
                },
                index=pd.MultiIndex.from_tuples(
                    [
                        ("storage_a", 24, 0),
                        ("storage_a", 24, 1),
                        ("storage_a", 24, 2),
                        ("storage_b", 24, 0),
                        ("storage_b", 24, 1),
                        ("storage_b", 24, 2),
                    ],
                    names=["stor", "hour", "year"],
                ),
            ),
            {0: 0.5, 1: 1.0},
            bidict({0: "storage_a", 1: "storage_b"}),
            {
                "storage_a": pd.DataFrame(
                    [[0.0, 0.0, 880.5]],
                    index=pd.Index([24], name="hour"),
                    columns=pd.Index([0, 1, 2], name="year"),
                ),
                "storage_b": pd.DataFrame(
                    [[0.0, 0.0, 0.0]],
                    index=pd.Index([24], name="hour"),
                    columns=pd.Index([0, 1, 2], name="year"),
                ),
            },
            id="happy_path",
        ),
        pytest.param(
            pd.DataFrame(
                columns=["solution"],
                index=pd.MultiIndex.from_tuples([], names=["stor", "hour", "year"]),
            ),
            {0: 0.5, 1: 1.0},
            bidict({0: "storage_a", 1: "storage_b"}),
            {},
            id="empty_dataframe",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "solution": [50, 150, 100],
                },
                index=pd.MultiIndex.from_tuples(
                    [
                        ("storage_a", 24, 0),
                        ("storage_a", 24, 1),
                        ("storage_a", 24, 2),
                    ],
                    names=["stor", "hour", "year"],
                ),
            ),
            {0: 1.0},
            bidict({0: "storage_a"}),
            {
                "storage_a": pd.DataFrame(
                    [[50.0, 150.0, 100.0]],
                    index=pd.Index([24], name="hour"),
                    columns=pd.Index([0, 1, 2], name="year"),
                ),
            },
            id="hundred_percent_efficiency",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "solution": [100],
                },
                index=pd.MultiIndex.from_tuples(
                    [
                        ("storage_a", 24, 0),
                    ],
                    names=["stor", "hour", "year"],
                ),
            ),
            {0: 0.5, 1: 1.0},
            bidict({0: "storage_a", 1: "storage_b"}),
            {
                "storage_a": pd.DataFrame(
                    [[50.0]],
                    index=pd.Index([24], name="hour"),
                    columns=pd.Index([0], name="year"),
                ),
            },
            id="more_storage_efficiency_than_data",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "solution": [0, 0, 100, 200, 300],
                },
                index=pd.MultiIndex.from_tuples(
                    [
                        ("storage_a", 24, 0),
                        ("storage_a", 24, 1),
                        ("storage_b", 25, 0),
                        ("storage_b", 25, 1),
                        ("storage_b", 25, 2),
                    ],
                    names=["stor", "hour", "year"],
                ),
            ),
            {0: 0.5, 1: 0.8},
            bidict({0: "storage_a", 1: "storage_b"}),
            {
                "storage_a": pd.DataFrame(
                    [[0.0, 0.0]],
                    index=pd.Index([24], name="hour"),
                    columns=pd.Index([0, 1], name="year"),
                ),
                "storage_b": pd.DataFrame(
                    [[80.0, 160.0, 240.0]],
                    index=pd.Index([25], name="hour"),
                    columns=pd.Index([0, 1, 2], name="year"),
                ),
            },
            id="different_hours",
        ),
    ],
)
def test_calculate_storage_adjusted_generation(
    generation_result_df: pd.DataFrame,
    storages_generation_efficiency: dict[int, float],
    storages_idxs: bidict[int, str | int],
    expected_output: dict[str, pd.DataFrame],
) -> None:
    result = calculate_storage_adjusted_generation(
        generation_result_df, storages_generation_efficiency, storages_idxs
    )
    for key in expected_output:
        pd.testing.assert_frame_equal(result[key], expected_output[key])

    for key in result:
        assert key in expected_output
