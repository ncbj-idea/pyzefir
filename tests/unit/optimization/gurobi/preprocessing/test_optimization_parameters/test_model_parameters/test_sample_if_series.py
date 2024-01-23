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

import pytest
from numpy import all, arange, ndarray
from pandas import Series

from pyzefir.optimization.gurobi.preprocessing.parameters import ModelParameters


@pytest.mark.parametrize(
    ("data", "sample", "expected_result"),
    [
        (
            Series(arange(100)),
            arange(start=0, stop=100, step=2),
            arange(start=0, stop=100, step=2),
        ),
        (Series([1, 2, 5, 7, 10, 2, 3]), [0, 3, 4], Series([1, 7, 10])),
        (Series([]), [], Series([])),
        (Series(arange(10)), None, Series(arange(10))),
        ({"a", "b", "c"}, arange(10), {"a", "b", "c"}),
    ],
)
def test_sample_series(data: Series, sample: ndarray, expected_result: ndarray) -> None:
    assert all(ModelParameters.sample_series(data, sample) == expected_result)
