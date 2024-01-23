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

from typing import Any

import numpy as np
import pandas as pd


def assert_equal(item: Any, other: Any) -> None:
    assert isinstance(other, type(item))
    if isinstance(item, dict):
        assert set(item) == set(other)
        for key in item:
            assert_equal(item[key], other[key])
    elif isinstance(item, (pd.Series, pd.DataFrame)):
        assert item.equals(other)
    elif isinstance(item, (list, tuple, np.ndarray)):
        assert all(item == other)
    else:
        assert item == other
