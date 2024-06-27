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
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.network import Network
from pyzefir.model.utils import NetworkConstants
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
from tests.unit.optimization.linopy.test_model.utils_for_objective_testing.utils_for_obj_opex import (
    objective_opex,
)
from tests.unit.optimization.linopy.test_model.utils_for_objective_testing.utils_for_obj_varcost import (
    objective_varcost,
)
from tests.unit.optimization.linopy.utils import TOL


@pytest.mark.parametrize(
    ("generator_capacity_cost", "gen_efficiency", "n_consumers", "hour_sample"),
    [
        (
            "brutto",
            {"pp_coal": {"electricity": 0.67}},
            pd.Series([100, 110, 120, 130, 140]),
            np.arange(100),
        ),
        (
            "netto",
            {"pp_coal": {"electricity": 0.67}},
            pd.Series([100, 110, 120, 130, 140]),
            np.arange(100),
        ),
        (
            "brutto",
            {
                "pp_coal": {"electricity": 0.8},
                "local_coal_heat_plant": {"heat": 0.3},
                "local_coal_heat_plant2": {"heat": 0.2},
            },
            pd.Series([200, 100, 200, 100, 120]),
            np.arange(90),
        ),
        (
            "netto",
            {
                "pp_coal": {"electricity": 0.8},
                "local_coal_heat_plant": {"heat": 0.3},
                "local_coal_heat_plant2": {"heat": 0.2},
            },
            pd.Series([200, 100, 200, 100, 120]),
            np.arange(90),
        ),
    ],
    ids=[
        "brutto increasing demand, low pp_coal efficiency",
        "netto increasing demand, low pp_coal efficiency",
        "brutto very low heat units",
        "netto very low heat units",
    ],
)
def test_generator_capacity_cost(
    generator_capacity_cost: str,
    gen_efficiency: dict[str, dict[str, float]],
    n_consumers: pd.Series,
    hour_sample: np.ndarray,
    network: Network,
) -> None:
    """
    test generator capacity cost
    """

    set_network_elements_parameters(
        network.aggregated_consumers,
        {
            "aggr": {
                "stack_base_fraction": {"lbs": 0.5, "lbs2": 0.5},
                "n_consumers": n_consumers,
            }
        },
    )

    set_network_elements_parameters(
        network.generators,
        {
            "local_coal_heat_plant": {"min_device_nom_power": 40},
            "local_coal_heat_plant2": {"min_device_nom_power": 50},
        },
    )

    for technology_name, et_efficiency in gen_efficiency.items():
        for energy_type, efficiency in et_efficiency.items():
            set_network_elements_parameters(
                network.generator_types,
                {
                    technology_name: {
                        "efficiency": pd.DataFrame(
                            {energy_type: [efficiency] * len(hour_sample)}
                        )
                    },
                },
            )

    set_network_elements_parameters(
        network.lines,
        {
            "grid->local_ee_bus": {"transmission_fee": None},
            "grid->local_ee_bus2": {"transmission_fee": None},
        },
    )

    new_constants_kwargs = asdict(network.constants)
    new_constants_kwargs["generator_capacity_cost"] = generator_capacity_cost
    network.constants = NetworkConstants(**new_constants_kwargs)

    opt_config = create_default_opt_config(
        hour_sample, np.arange(N_YEARS), generator_capacity_cost=generator_capacity_cost
    )
    assert opt_config.generator_capacity_cost == generator_capacity_cost
    engine = run_opt_engine(network, opt_config)

    _test_objective(engine.indices, engine.parameters, engine.results)


def _test_objective(
    indices: Indices, parameters: OptimizationParameters, results: Results
) -> None:
    expected_obj = objective_capex(indices, parameters, results)
    expected_obj += objective_opex(indices, parameters, results)
    expected_obj += objective_varcost(indices, parameters, results)
    expected_obj += objective_dsr(indices, parameters, results)
    assert abs(results.objective_value - expected_obj) < TOL
