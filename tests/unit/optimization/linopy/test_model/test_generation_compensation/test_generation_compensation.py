# PyZefir
# Copyright (C) 2024 Narodowe Centrum Badań Jądrowych
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

from pyzefir.model.network import Network
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.results import Results
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opt_config,
    run_opt_engine,
    set_network_elements_parameters,
)
from tests.unit.optimization.linopy.test_model.utils_for_objective_testing.utils_for_obj_capex import (
    objective_capex,
)
from tests.unit.optimization.linopy.test_model.utils_for_objective_testing.utils_for_obj_dsr import (
    objective_dsr,
)
from tests.unit.optimization.linopy.test_model.utils_for_objective_testing.utils_for_obj_generation_comp import (
    objective_generation_compensation,
)
from tests.unit.optimization.linopy.test_model.utils_for_objective_testing.utils_for_obj_opex import (
    objective_opex,
)
from tests.unit.optimization.linopy.test_model.utils_for_objective_testing.utils_for_obj_varcost import (
    objective_varcost,
)
from tests.unit.optimization.linopy.utils import TOL


@pytest.mark.parametrize(
    ("generation_compensation", "n_consumers", "hour_sample"),
    [
        (
            {"pp_coal": pd.Series([9] * N_YEARS)},
            pd.Series([20, 10, 20, 10, 12]),
            np.arange(100),
        ),
        (
            {"pp_coal": pd.Series([99] * N_YEARS)},  # the objective will be negative
            pd.Series([20, 10, 20, 10, 12]),
            np.arange(50),
        ),
        (
            {
                "pp_coal": pd.Series([7] * N_YEARS),
                "local_coal_heat_plant": pd.Series([9] * N_YEARS),
            },
            pd.Series([5, 6, 7, 7, 10]),
            np.arange(80),
        ),
        (
            {
                "pp_coal": pd.Series([15] * N_YEARS),
                "local_coal_heat_plant": pd.Series([12] * N_YEARS),
                "local_coal_heat_plant2": pd.Series([4, 3, 2, 0, 2]),
            },
            pd.Series([7, 6, 7, 7, 10]),
            np.arange(50),
        ),
        (
            {
                "pp_coal": pd.Series([10, 11, 12, 13, 11] * N_YEARS),
                "local_coal_heat_plant": pd.Series([8, 10, 12, 7, 1]),
                "local_coal_heat_plant2": pd.Series([4, 3, 2, 0, 2]),
            },
            pd.Series([7, 6, 7, 7, 10]),
            np.arange(50),
        ),
        (
            {
                "pp_coal": pd.Series([0, 0, 0, 11, 11] * N_YEARS),
                "local_coal_heat_plant": pd.Series([8, 10, 11, 0, 0]),
                "local_coal_heat_plant2": pd.Series([0, 0, 2, 12, 10]),
            },
            pd.Series([7, 6, 10, 7, 12]),
            np.arange(50),
        ),
    ],
    ids=[
        "constant generation compensation for pp_coal",
        "enforce negative objective value",
        "constant generation compensation, vaiable number of n-consumers",
        "variable generation compensation in one unit",
        "variable generation compensation in all units",
        "variable generation compensation, more zeros",
    ],
)
def test_generation_compensation(
    generation_compensation: dict[str, pd.Series],
    n_consumers: pd.Series,
    hour_sample: np.ndarray,
    network: Network,
) -> None:
    """
    test generation compensation
    """

    set_network_elements_parameters(
        network.aggregated_consumers,
        {
            "aggr": {
                "stack_base_fraction": {"lbs": 0.45, "lbs2": 0.55},
                "n_consumers": n_consumers,
            }
        },
    )

    set_network_elements_parameters(
        network.generators,
        {
            "local_coal_heat_plant": {"min_device_nom_power": 10},
            "local_coal_heat_plant2": {"min_device_nom_power": 10},
        },
    )

    set_network_elements_parameters(
        network.lines,
        {
            "grid->local_ee_bus": {"transmission_fee": None},
            "grid->local_ee_bus2": {"transmission_fee": None},
        },
    )

    for gen_type, compensation in generation_compensation.items():
        set_network_elements_parameters(
            network.generator_types,
            {gen_type: {"generation_compensation": compensation}},
        ),

    opt_config = create_default_opt_config(hour_sample, np.arange(N_YEARS))
    engine = run_opt_engine(network, opt_config)

    _test_objective(engine.indices, engine.parameters, engine.results)


def _test_objective(
    indices: Indices, parameters: OptimizationParameters, results: Results
) -> None:
    expected_obj = objective_capex(indices, parameters, results)
    expected_obj += objective_opex(indices, parameters, results)
    expected_obj += objective_varcost(indices, parameters, results)
    expected_obj += objective_dsr(indices, parameters, results)
    expected_obj += objective_generation_compensation(indices, parameters, results)
    assert abs(results.objective_value - expected_obj) < TOL
