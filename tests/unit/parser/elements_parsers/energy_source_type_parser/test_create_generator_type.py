import numpy as np
import pandas as pd
import pytest

from pyzefir.parser.elements_parsers.energy_source_type_parser import (
    EnergySourceTypeParser,
)
from tests.unit.defaults import ELECTRICITY, HEATING
from tests.unit.parser.elements_parsers.utils import assert_equal


@pytest.mark.parametrize(
    ("name", "expected_results"),
    [
        pytest.param(
            "GEN_TYPE_3",
            {
                "name": "GEN_TYPE_3",
                "life_time": 15,
                "build_time": 0,
                "capex": pd.Series([90, 88, 85]),
                "opex": pd.Series([15, 14, 12]),
                "min_capacity": pd.Series([np.nan] * 4),
                "max_capacity": pd.Series([np.nan] * 4),
                "min_capacity_increase": pd.Series([np.nan] * 4),
                "max_capacity_increase": pd.Series([np.nan] * 4),
                "efficiency": {HEATING: 0.84},
                "energy_types": {HEATING},
                "emission_reduction": {"CO2": 0.4, "SO2": 0.25},
                "conversion_rate": {ELECTRICITY: pd.Series(np.ones(8760))},
                "fuel": None,
                "capacity_factor": None,
                "power_utilization": 0.9,
            },
            id="no_optional_parameters_provided",
        ),
    ],
)
def test_create_generator_type(
    name: str, expected_results: dict, energy_source_type_parser: EnergySourceTypeParser
) -> None:
    df_row = pd.Series(
        index=["name", "build_time", "life_time", "power_utilization"],
        data=[name, 0, 15, 0.9],
    )
    energy_source_type_df = (
        energy_source_type_parser._prepare_energy_source_parameters()
    )
    efficiency_df = energy_source_type_parser.generators_efficiency.pivot_table(
        index=energy_source_type_parser.generators_efficiency.index,
        columns="energy_type",
        values="efficiency",
    )
    demand_dict = energy_source_type_parser._prepare_conversion_rate_dict(
        energy_source_type_parser.conversion_rate
    )
    result = energy_source_type_parser._create_generator_type(
        df_row, energy_source_type_df, efficiency_df, demand_dict
    )

    for attr, value in expected_results.items():
        to_compare = getattr(result, attr)
        assert_equal(value, to_compare)
