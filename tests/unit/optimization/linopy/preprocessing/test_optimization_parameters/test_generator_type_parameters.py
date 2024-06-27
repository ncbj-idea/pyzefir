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
    "sample, generator_type_names, params_to_change",
    [
        pytest.param(
            arange(N_YEARS),
            [
                "pp_coal",
                "pp_gas",
                "heat_plant_coal",
                "heat_plant_biomass",
                "chp_coal",
                "heat_pump",
                "boiler_coal",
                "boiler_biomass",
                "pv",
                "solar",
                "wind_farm",
            ],
            {
                "min_capacity": pd.Series([np.nan, 0.0, 0.0, 0.0, 0.0]),
                "max_capacity": pd.Series([np.nan, 1.5, 2.0, 2.5, 3.0]),
                "min_capacity_increase": pd.Series([np.nan, 4.0, 4.2, 4.1, 4.6]),
                "max_capacity_increase": pd.Series([np.nan, 0.5, 20.0, 0.1, 30.0]),
            },
            id="full_year_range+every_gen_type+every_param",
        ),
        pytest.param(
            array([0, 3]),
            ["boiler_coal"],
            {
                "min_capacity": pd.Series([np.nan, 0.0, 0.0, 0.0, 0.0]),
                "max_capacity": pd.Series([np.nan, 1.5, 2.0, 2.5, 3.0]),
            },
            id="2_years+boiler_coal+cap_min_cap_max",
        ),
        pytest.param(
            array([2]),
            ["pp_gas"],
            {
                "min_capacity_increase": pd.Series([np.nan, 4.0, 4.2, 4.1, 4.6]),
                "max_capacity_increase": pd.Series([np.nan, 0.5, 20.0, 0.1, 30.0]),
            },
            id="single_year+pp_gas+delta_cap_min_delta_cap_max",
        ),
        pytest.param(
            array([2]),
            ["pp_coal", "pp_gas"],
            {
                "r": 0.4,
            },
            id="ramp",
        ),
        pytest.param(
            array([2]),
            ["pp_coal", "pp_gas"],
            {
                "energy_curtailment_cost": pd.Series([10.0, 20.0, 30.0, 40.0, 50.0]),
            },
            id="energy_curtailment_cost",
        ),
    ],
)
def test_create(
    sample: ndarray,
    generator_type_names: str,
    params_to_change: dict[str, pd.Series],
    complete_network: Network,
    opt_config: OptConfig,
) -> None:
    opt_config.year_sample = sample

    for param_name, param_value in params_to_change.items():
        for generator_type_name in generator_type_names:
            setattr(
                complete_network.generator_types[generator_type_name],
                param_name,
                param_value,
            )

    indices = Indices(complete_network, opt_config)
    generator_type_params = OptimizationParameters(
        complete_network, indices, opt_config
    ).tgen

    for generator_type_id, generator_type_name in indices.TGEN.mapping.items():
        generator_type = complete_network.generator_types[generator_type_name]

        for param in [
            "min_capacity",
            "max_capacity",
            "min_capacity_increase",
            "max_capacity_increase",
            "ramp",
            "energy_curtailment_cost",
        ]:
            if param in params_to_change:
                if generator_type.energy_curtailment_cost is not None:
                    assert vectors_eq_check(
                        getattr(generator_type_params, param)[generator_type_id],
                        getattr(generator_type, param),
                        sample,
                    )
