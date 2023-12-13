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
