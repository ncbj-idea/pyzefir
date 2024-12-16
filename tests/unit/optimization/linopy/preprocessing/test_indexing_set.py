import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest

from pyzefir.model.network import NetworkElementsDict
from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet


def test_2D_array() -> None:
    with pytest.raises(ValueError, match=r"IndexingSet: 1D array required"):
        IndexingSet(np.array([[1, 2], [3, 4]]))


def test_duplicates_raises_error() -> None:
    with pytest.raises(
        ValueError, match=r"IndexingSet: provided array contains duplicates"
    ):
        IndexingSet(np.array([1, 2, 3, 4, 5, 1]))


def test_simple_use() -> None:
    sz = 100
    ii = IndexingSet(np.arange(sz))
    assert np.all(ii.ord == np.arange(sz))
    assert all(ii.mapping[i] == i for i in ii.mapping)
    assert all(ii.inverse[i] == i for i in ii.inverse)


def test_odd_numbers() -> None:
    sz = 37
    ii = IndexingSet(np.arange(1, 2 * sz, 2))
    assert np.all(ii.ord == np.arange(sz))
    assert all(ii.mapping[i] == 2 * i + 1 for i in ii.mapping)
    assert all(ii.inverse[i] == (i - 1) / 2 for i in ii.inverse)


def test_strings() -> None:
    a = np.array(["a", "xyz", "Joe", "Leokadia"])
    ii = IndexingSet(a)
    assert ii.ord.shape == a.shape
    assert np.all(ii.ord == np.arange(a.shape[0]))
    assert set(ii.inverse.keys()) == set(a)


@pytest.mark.parametrize(
    ("feature_name",),
    [
        ("fuels",),
        ("global_generators",),
        ("generator_types",),
        ("storage_types",),
    ],
)
def test_create_from_network_elements_dict(
    feature_name: str, request: FixtureRequest
) -> None:
    d = NetworkElementsDict(request.getfixturevalue(feature_name))
    idx = IndexingSet.create_from_network_elements_dict(d, feature_name)

    assert idx.name == feature_name
    assert len(idx) == len(d)
    assert np.all(idx.ord == np.arange(len(d)))
    assert set(idx.ii) == set(d)


def test_create_from_network_elements_dict_empty() -> None:
    d = NetworkElementsDict()
    idx = IndexingSet.create_from_network_elements_dict(d)

    assert idx.name == ""
    assert len(idx) == 0
    assert idx.ord.shape == (0,)
    assert idx.ii.shape == (0,)
