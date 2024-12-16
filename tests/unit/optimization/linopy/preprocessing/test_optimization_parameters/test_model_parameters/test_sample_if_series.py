import pytest
from numpy import all, arange, ndarray
from pandas import Series

from pyzefir.optimization.linopy.preprocessing.parameters import ModelParameters


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
