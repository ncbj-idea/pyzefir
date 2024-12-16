import numpy as np
import pandas as pd
import pytest
from numpy import array

from pyzefir.model.network import NetworkElementsDict
from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet
from pyzefir.optimization.linopy.preprocessing.parameters import ModelParameters
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.preprocessing.test_optimization_parameters.test_model_parameters.utils import (
    EnergySourceTestImplementation,
    EnergySourceTypeTestImplementation,
)


@pytest.mark.parametrize(
    (
        "energy_source_names",
        "energy_source_type_names",
        "connected_element_names",
        "connections",
        "expected_results",
    ),
    [
        (
            ["es1", "es2", "es3"],
            {"es1": "t1", "es2": "t1", "es3": "t2"},
            ["conn1", "conn2"],
            {"t1": "conn1", "t2": None},
            {0: 0, 1: 0, 2: None},
        ),
        (
            ["es1", "es2", "es3", "es4"],
            {"es1": "t1", "es2": "t2", "es3": "t3", "es4": "t3"},
            ["conn1", "conn2"],
            {"t1": "conn1", "t2": "conn2", "t3": "conn2"},
            {0: 0, 1: 1, 2: 1, 3: 1},
        ),
    ],
)
def test_get_index_from_type_prop(
    energy_source_names: list[str],
    energy_source_type_names: dict[str, str],
    connected_element_names: list[str],
    connections: dict[str, int],
    expected_results: dict[int, int],
) -> None:
    energy_sources = NetworkElementsDict(
        {
            name: EnergySourceTestImplementation(
                name=name,
                energy_source_type=est,
                unit_min_capacity=pd.Series([np.nan] * N_YEARS),
                unit_max_capacity=pd.Series([np.nan] * N_YEARS),
                unit_min_capacity_increase=pd.Series([np.nan] * N_YEARS),
                unit_max_capacity_increase=pd.Series([np.nan] * N_YEARS),
            )
            for name, est in energy_source_type_names.items()
        }
    )
    energy_source_types = NetworkElementsDict(
        {
            name: EnergySourceTypeTestImplementation(
                name=name,
                scalar_prop=connected_element,
                min_capacity=pd.Series([np.nan] * N_YEARS),
                max_capacity=pd.Series([np.nan] * N_YEARS),
                min_capacity_increase=pd.Series([np.nan] * N_YEARS),
                max_capacity_increase=pd.Series([np.nan] * N_YEARS),
            )
            for name, connected_element in connections.items()
        }
    )
    energy_source_idx = IndexingSet(array(energy_source_names))
    connected_element_idx = IndexingSet(array(connected_element_names))

    result = ModelParameters.get_index_from_type_prop(
        energy_sources,
        energy_source_types,
        energy_source_idx,
        connected_element_idx,
        "scalar_prop",
    )

    assert result == expected_results
