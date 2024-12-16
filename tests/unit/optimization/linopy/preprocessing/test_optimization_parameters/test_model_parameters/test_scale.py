import numpy as np
import pytest

from pyzefir.optimization.linopy.preprocessing.parameters import ModelParameters


@pytest.mark.parametrize(
    "scale, initial_dict, expected_dict",
    (
        (
            1,
            {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])},
            {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])},
        ),
        (
            2,
            {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])},
            {"a": np.array([0.5, 1, 1.5]), "b": np.array([2, 2.5, 3])},
        ),
        (
            0.4,
            {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])},
            {"a": np.array([2.5, 5, 7.5]), "b": np.array([10, 12.5, 15])},
        ),
    ),
)
def test_scale(
    scale: float,
    initial_dict: dict[str, np.ndarray],
    expected_dict: dict[str, np.ndarray],
) -> None:
    scaled_dict = ModelParameters.scale(values=initial_dict, scale=scale)
    assert expected_dict.keys() == scaled_dict.keys()
    for k in expected_dict:
        assert (expected_dict[k] == scaled_dict[k]).all()
