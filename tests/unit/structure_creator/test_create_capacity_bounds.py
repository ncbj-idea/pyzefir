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

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pyzefir.structure_creator.structure_and_initial_state.structure_element_creators import (
    CapacityBoundsCreator,
)


@pytest.mark.parametrize(
    "df_data, aggr_name, expected_output",
    [
        pytest.param(
            pd.DataFrame(
                {"lbs": ["lbs1", "lbs2"], "right_technology_name": ["tech1", "tech2"]}
            ),
            "aggregate1",
            pd.DataFrame(),
            id="without left_technology_name",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "lbs": ["lbs1", "lbs1", "lbs2"],
                    "left_technology_name": ["tech1", "tech2", "tech3"],
                    "right_technology_name": ["tech2", "tech1", "tech3"],
                    "gen_name": ["gen1", "gen2", "gen3"],
                    "sense": ["positive", "negative", "negative"],
                    "left_coeff": [1.5, 2.0, 2.1],
                    "aggr": ["aggregateX", "aggregateY", "aggregateZ"],
                },
                index=["tech1", "tech2", "tech3"],
            ),
            "aggregate1",
            pd.DataFrame(),
            id="no matching aggr_name",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "lbs": ["lbs1", "lbs1", "lbs2"],
                    "left_technology_name": ["tech1", "tech2", "tech3"],
                    "right_technology_name": ["tech2", "tech1", "tech3"],
                    "gen_name": ["gen1", "gen2", "gen3"],
                    "sense": ["positive", "negative", "negative"],
                    "left_coeff": [1.5, 2.0, 2.1],
                    "aggr": ["aggregateX", "aggregate1", "aggregateZ"],
                },
                index=["tech1", "tech2", "tech3"],
            ),
            "aggregate1",
            pd.DataFrame(
                {
                    "mapped_left_tech": ["gen2"],
                    "sense": ["negative"],
                    "mapped_right_tech": ["gen1"],
                    "left_coeff": [2.0],
                }
            ),
            id="match only one aggregate",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "lbs": ["lbs1", "lbs1", "lbs2"],
                    "left_technology_name": ["tech1", "tech2", "tech3"],
                    "right_technology_name": ["tech2", "tech1", "tech3"],
                    "gen_name": ["gen1", "gen2", "gen3"],
                    "sense": ["positive", "negative", "negative"],
                    "left_coeff": [1.5, 2.0, 2.1],
                    "aggr": ["all", "aggregate1", "aggregate1"],
                },
                index=["tech1", "tech2", "tech3"],
            ),
            "aggregate1",
            pd.DataFrame(
                {
                    "mapped_left_tech": ["gen1", "gen2", "gen3"],
                    "sense": ["positive", "negative", "negative"],
                    "mapped_right_tech": ["gen2", "gen1", "gen3"],
                    "left_coeff": [1.5, 2.0, 2.1],
                }
            ),
            id="all aggr are mapped, spec aggr and all",
        ),
        pytest.param(
            pd.DataFrame(),
            "aggregate1",
            pd.DataFrame(),
            id="empty df passed as argument",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "lbs": ["lbs1", "lbs1", "lbs2"],
                    "left_technology_name": ["tech1", "tech2", "tech3"],
                    "right_technology_name": [np.nan, np.nan, np.nan],
                    "gen_name": ["gen1", "gen2", "gen3"],
                    "sense": ["positive", "negative", "negative"],
                    "left_coeff": [1.5, 2.0, 2.1],
                    "aggr": ["all", "all", "all"],
                },
                index=["tech1", "tech2", "tech3"],
            ),
            "aggregate1",
            pd.DataFrame(),
            id="right_technology_name_empty",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "lbs": ["lbs1", "lbs1", "lbs2"],
                    "left_technology_name": [np.nan, np.nan, np.nan],
                    "right_technology_name": ["tech2", "tech1", "tech3"],
                    "gen_name": ["gen1", "gen2", "gen3"],
                    "sense": ["positive", "negative", "negative"],
                    "left_coeff": [1.5, 2.0, 2.1],
                    "aggr": ["all", "all", "all"],
                },
                index=["tech1", "tech2", "tech3"],
            ),
            "aggregate1",
            pd.DataFrame(),
            id="lbs_empty",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "lbs": ["lbs1", "lbs1", "lbs2"],
                    "left_technology_name": [np.nan, np.nan, np.nan],
                    "right_technology_name": ["tech2", "tech1", "tech3"],
                    "gen_name": ["gen1", "gen2", "gen3"],
                    "sense": ["positive", "negative", "negative"],
                    "left_coeff": [1.5, 2.0, 2.1],
                    "aggr": ["all", "all", "all"],
                },
                index=["tech1", "tech2", "tech3"],
            ),
            "aggregate1",
            pd.DataFrame(),
            id="left_technology_name_empty",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "lbs": ["lbs1", "lbs1", "lbs2"],
                    "left_technology_name": ["tech1", np.nan, "tech3"],
                    "right_technology_name": ["tech2", "tech1", "tech3"],
                    "gen_name": ["gen1", "gen2", "gen3"],
                    "sense": ["positive", "negative", "negative"],
                    "left_coeff": [1.5, 2.0, 2.1],
                    "aggr": ["all", "all", "aggregateX"],
                },
                index=["tech1", "tech2", "tech3"],
            ),
            "aggregate1",
            pd.DataFrame(
                {
                    "mapped_left_tech": ["gen1"],
                    "sense": ["positive"],
                    "mapped_right_tech": ["gen2"],
                    "left_coeff": [1.5],
                }
            ),
            id="left_technology_name_1_nan_and_1_no_matched_aggregate",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "lbs": ["lbs1", "lbs1", "lbs1"],
                    "left_technology_name": ["tech1", "tech2", "tech3"],
                    "right_technology_name": ["tech3", "tech2", "tech1"],
                    "gen_name": ["gen1", "gen2", "gen3"],
                    "sense": ["positive", "negative", "negative"],
                    "left_coeff": [1.5, 2.0, 2.1],
                    "aggr": ["aggregate1", "all", "aggregate1"],
                },
                index=["tech1", "tech2", "tech3"],
            ),
            "aggregate1",
            pd.DataFrame(
                {
                    "mapped_left_tech": ["gen1", "gen2", "gen3"],
                    "sense": ["positive", "negative", "negative"],
                    "mapped_right_tech": ["gen3", "gen2", "gen1"],
                    "left_coeff": [1.5, 2.0, 2.1],
                }
            ),
            id="1_lbs_all_matched",
        ),
    ],
)
def test_create_capacity_bounds_df(
    df_data: pd.DataFrame,
    aggr_name: str,
    expected_output: pd.DataFrame,
) -> None:
    with patch(
        "pyzefir.structure_creator.structure_and_initial_state.structure_element_creators."
        "CapacityBoundsCreator.handle_capacity_bounds_df_structure",
        side_effect=lambda df: df,
    ):
        result = CapacityBoundsCreator.create_capacity_bounds_df(df_data, aggr_name)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_output)
