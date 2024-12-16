import numpy as np
import pandas as pd
import pytest
from numpy import arange, array, ndarray

from pyzefir.model.network import Network
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.conftest import N_YEARS
from tests.unit.optimization.linopy.preprocessing.test_optimization_parameters.utils import (
    vectors_eq_check,
)


@pytest.mark.parametrize(
    "sample, storage_type_names, params_to_change",
    [
        pytest.param(
            arange(N_YEARS),
            ["heat_storage_type", "ee_storage_type"],
            {
                "min_capacity": pd.Series([np.nan, 0.0, 0.0, 0.0, 0.0]),
                "max_capacity": pd.Series([np.nan, 1.5, 2.0, 2.5, 3.0]),
                "min_capacity_increase": pd.Series([np.nan, 4.0, 4.2, 4.1, 4.6]),
                "max_capacity_increase": pd.Series([np.nan, 0.5, 20.0, 0.1, 30.0]),
                "energy_loss": 0.1,
                "power_utilization": 0.9,
            },
            id="full_year_range+every_storage_type+every_param",
        ),
        pytest.param(
            array([0, 3]),
            ["ee_storage_type"],
            {
                "min_capacity": pd.Series([np.nan, 0.0, 0.0, 0.0, 0.0]),
                "max_capacity": pd.Series([np.nan, 1.5, 2.0, 2.5, 3.0]),
                "energy_loss": 0.0,
            },
            id="2_years+ee_storage_type+cap_min_cap_max",
        ),
        pytest.param(
            array([2]),
            ["heat_storage_type"],
            {
                "min_capacity_increase": pd.Series([np.nan, 4.0, 4.2, 4.1, 4.6]),
                "max_capacity_increase": pd.Series([np.nan, 0.5, 20.0, 0.1, 30.0]),
                "energy_loss": 0.2,
            },
            id="single_year+heat_storage_type+delta_cap_min_delta_cap_max",
        ),
    ],
)
def test_create(
    sample: ndarray,
    storage_type_names: str,
    params_to_change: dict[str, pd.Series],
    complete_network: Network,
    opt_config: OptConfig,
) -> None:
    opt_config.year_sample = sample

    for param_name, param_value in params_to_change.items():
        for storage_type_name in storage_type_names:
            setattr(
                complete_network.storage_types[storage_type_name],
                param_name,
                param_value,
            )

    indices = Indices(complete_network, opt_config)
    storage_type_params = OptimizationParameters(
        complete_network, indices, opt_config
    ).tstor

    # assert (
    #     storage_params.power_utilization[storage_id]
    #     == storage_type.power_utilization
    # )

    for storage_type_id, storage_type_name in indices.TSTOR.mapping.items():
        storage_type = complete_network.storage_types[storage_type_name]

        for param in [
            "min_capacity",
            "max_capacity",
            "min_capacity_increase",
            "max_capacity_increase",
            "energy_loss",
            "power_utilization",
        ]:
            if param in params_to_change:
                assert vectors_eq_check(
                    getattr(storage_type_params, param)[storage_type_id],
                    getattr(storage_type, param),
                    sample,
                )
