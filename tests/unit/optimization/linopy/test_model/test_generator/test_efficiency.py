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
from tests.unit.optimization.linopy.names import EE, HEAT
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opf_config,
    run_opt_engine,
    set_network_elements_parameters,
)


@pytest.mark.parametrize(
    ("efficiency_heat", "efficiency_electricity"),
    [
        pytest.param(
            pd.DataFrame({HEAT: [0.5, 0.5, 0.5, 0.5, 0.5]}),
            pd.DataFrame({EE: [0.5, 0.5, 0.5, 0.5, 0.5]}),
            id="Static series",
        ),
        pytest.param(
            pd.DataFrame({HEAT: [0.8, 0.4, 0.5, 1.0, 0.2]}),
            pd.DataFrame({EE: [0.5, 0.5, 0.7, 0.2, 0.3]}),
            id="random values",
        ),
        pytest.param(
            pd.DataFrame({HEAT: [0.8, 0.7, 0.6, 0.5, 0.4]}),
            pd.DataFrame({EE: [0.1, 0.2, 0.3, 0.4, 0.5]}),
            id="rising and falling series",
        ),
        pytest.param(
            pd.DataFrame({HEAT: [1.0, 1.0, 1.0, 0.1, 0.1]}),
            pd.DataFrame({EE: [1.2, 1.1, 0.2, 0.2, 0.4]}),
            id="values ​​above 1 for the energy type",
        ),
    ],
)
def test_efficiency_series(
    network: Network,
    efficiency_heat: pd.DataFrame,
    efficiency_electricity: pd.DataFrame,
) -> None:
    set_network_elements_parameters(
        network.generator_types,
        {
            "pp_coal": {"efficiency": efficiency_electricity},
            "heat_plant_biomass": {"efficiency": efficiency_heat},
        },
    )
    opt_config = create_default_opf_config(
        hour_sample=np.array([0, 1, 2, 3, 4]), year_sample=np.array([0, 1])
    )
    run_opt_engine(network, opt_config)
