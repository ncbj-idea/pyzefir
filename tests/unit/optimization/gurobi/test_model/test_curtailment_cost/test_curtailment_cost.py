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
from pyzefir.model.network_elements import Storage
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.results import Results
from tests.unit.optimization.gurobi.constants import N_YEARS
from tests.unit.optimization.gurobi.test_model.utils import (
    create_default_opf_config,
    run_opt_engine,
    set_network_elements_parameters,
)
from tests.unit.optimization.gurobi.test_model.utils_for_objective_testing.utils_for_obj_capex import (
    objective_capex,
)
from tests.unit.optimization.gurobi.test_model.utils_for_objective_testing.utils_for_obj_curtailment_cost import (
    objective_curtailment_cost,
)
from tests.unit.optimization.gurobi.test_model.utils_for_objective_testing.utils_for_obj_opex import (
    objective_opex,
)
from tests.unit.optimization.gurobi.test_model.utils_for_objective_testing.utils_for_obj_varcost import (
    objective_varcost,
)
from tests.unit.optimization.gurobi.utils import TOL


@pytest.mark.parametrize(
    ("hour_sample", "year_sample", "energy_curtailment_cost"),
    [
        (np.arange(50), np.arange(N_YEARS), {"pv": pd.Series([50.0] * 5)}),
        (
            np.arange(50),
            np.arange(N_YEARS),
            {"pv": pd.Series([50.0] * 5), "heat_plant_biomass": pd.Series([100.0] * 5)},
        ),
        (np.arange(50), np.arange(3), {"pv": pd.Series([50.0] * 5)}),
        (
            np.arange(100),
            np.arange(4),
            {
                "pv": pd.Series([10.0, 20.0, 30.0, 40.0] * 4),
                "heat_plant_biomass": pd.Series([100.0] * 4),
            },
        ),
    ],
)
def test_curtailment_cost(
    hour_sample: np.ndarray,
    year_sample: np.ndarray,
    energy_curtailment_cost: dict[str, pd.Series],
    network: Network,
    ee_storage: Storage,
    heat_storage: Storage,
) -> None:
    """
    Test energy curtailemet cost
    Network conf such that dump energy for PV
    Test the objective, test if dump=0 (for non-PV units), then there is no additional costs even if costs parameters
    are present
    """

    set_network_elements_parameters(
        network.generators,
        {"local_pv": {"unit_base_cap": 50.0}},
    )

    for gen_type, cost in energy_curtailment_cost.items():
        set_network_elements_parameters(
            network.generator_types,
            {gen_type: {"energy_curtailment_cost": cost}},
        )

    opt_config = create_default_opf_config(hour_sample, year_sample)
    engine = run_opt_engine(network, opt_config)

    parameters = engine.parameters
    results = engine.results
    indices = engine.indices

    _test_objective(indices, parameters, results)


def _test_objective(
    indices: Indices, parameters: OptimizationParameters, results: Results
) -> None:
    expected_obj = objective_capex(indices, parameters, results)
    expected_obj += objective_opex(indices, parameters, results)
    expected_obj += objective_varcost(indices, parameters, results)
    expected_obj += objective_curtailment_cost(indices, parameters, results)
    assert abs(results.objective_value - expected_obj) < TOL
