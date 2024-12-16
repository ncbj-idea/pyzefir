import numpy as np
import pandas as pd
import pytest

from pyzefir.parser.elements_parsers.energy_source_type_parser import (
    EnergySourceTypeParser,
)
from tests.unit.defaults import HEATING
from tests.unit.parser.elements_parsers.utils import assert_equal


@pytest.mark.parametrize(
    ("name", "expected_results"),
    [
        pytest.param(
            "STORAGE_TYPE_3",
            {
                "name": "STORAGE_TYPE_3",
                "life_time": 15,
                "build_time": 1,
                "capex": pd.Series([200, 150, 120]),
                "opex": pd.Series([10, 5, 3]),
                "min_capacity": pd.Series([np.nan] * 4),
                "max_capacity": pd.Series([np.nan] * 4),
                "min_capacity_increase": pd.Series([np.nan] * 4),
                "max_capacity_increase": pd.Series([np.nan] * 4),
                "energy_type": HEATING,
                "generation_efficiency": 0.88,
                "load_efficiency": 0.82,
                "cycle_length": 2190,
                "power_to_capacity": 10.0,
            },
            id="no_optional_parameters_provided",
        ),
    ],
)
def test_create_storage_type(
    name: str, expected_results: dict, energy_source_type_parser: EnergySourceTypeParser
) -> None:
    energy_source_type_df = (
        energy_source_type_parser._prepare_energy_source_parameters()
    )
    row_df = energy_source_type_parser.storage_type_df.loc[name, :]
    result = energy_source_type_parser._create_storage_type(
        row_df, energy_source_type_df
    )

    for attr, value in expected_results.items():
        to_compare = getattr(result, attr)
        assert_equal(value, to_compare)
