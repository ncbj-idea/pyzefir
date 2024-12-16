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
import pytest

from pyzefir.model.network import Network
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.results import Results
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opt_config,
    run_opt_engine,
)
from tests.unit.optimization.linopy.test_model.utils_for_objective_testing.utils_for_obj_capex import (
    objective_capex,
)
from tests.unit.optimization.linopy.test_model.utils_for_objective_testing.utils_for_obj_dsr import (
    objective_dsr,
)
from tests.unit.optimization.linopy.test_model.utils_for_objective_testing.utils_for_obj_ens import (
    objective_ens,
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


def assert_correct_objective(
    indices: Indices,
    parameters: OptimizationParameters,
    results: Results,
    tol: float = TOL,
) -> None:
    expected_obj = objective_capex(indices, parameters, results)
    expected_obj += objective_opex(indices, parameters, results)
    expected_obj += objective_varcost(indices, parameters, results)
    expected_obj += objective_dsr(indices, parameters, results)
    expected_obj += objective_ens(indices, parameters, results)
    expected_obj += objective_generation_compensation(indices, parameters, results)
    assert np.isclose(results.objective_value, expected_obj, rtol=tol)


@pytest.mark.parametrize("n_years", [1, 2, 3, 4, 5])
def test_n_years_aggregation(network: Network, n_years: int) -> None:
    res = run_opt_engine(
        network,
        create_default_opt_config(
            np.arange(100),
            np.arange(3),
            year_aggregates=np.array([1, n_years, 1]),
        ),
    )

    assert_correct_objective(res.indices, res.parameters, res.results)


@pytest.mark.parametrize("n_years", [1, 2, 3, 4, 5])
def test_n_years_aggregation_ens(network_without_grid: Network, n_years: int) -> None:
    res = run_opt_engine(
        network_without_grid,
        create_default_opt_config(
            np.arange(100),
            np.arange(3),
            year_aggregates=np.array([1, n_years, 1]),
        ),
    )

    assert_correct_objective(res.indices, res.parameters, res.results)
