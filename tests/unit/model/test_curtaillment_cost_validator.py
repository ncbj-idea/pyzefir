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

from pyzefir.model.exceptions import NetworkValidatorExceptionGroup
from pyzefir.model.network import Network
from pyzefir.model.network_elements import GeneratorType


@pytest.mark.parametrize(
    "element_name, element_params, exception_msg",
    (
        pytest.param(
            "generator_types",
            {"CHP_COAL": {"energy_curtailment_cost": pd.Series([50.0, 50.4, 51.2])}},
            "Incorrect year indices for energy curtailment cost of generator type <CHP_COAL> "
            "The number of indexes should match the number of years",
            id="Incorrect indices for energy curtailment",
        ),
        pytest.param(
            "generator_types",
            {
                "CHP_COAL": {
                    "energy_curtailment_cost": pd.Series([50.0, 50.4, np.nan, "string"])
                }
            },
            "Incorrect values for energy curtailment cost of generator type <CHP_COAL>",
            id="Incorrect values for energy curtailment",
        ),
    ),
)
def test_network_curtailment_cost_validation(
    network: Network,
    element_name: str,
    element_params: dict,
    exception_msg: str,
) -> None:
    network.generator_types = element_params
    exception_list: list[NetworkValidatorExceptionGroup] = []

    for name, curt_cost in element_params.items():
        gen_type = GeneratorType(
            name=name,
            build_time=0,
            life_time=9,
            capex=pd.Series([]),
            opex=pd.Series([]),
            max_capacity=pd.Series([]),
            min_capacity=pd.Series([]),
            max_capacity_increase=pd.Series([]),
            min_capacity_increase=pd.Series([]),
            efficiency={"ELECTRICITY": 0.8, "HEATING": 0.9},
            energy_types=set(network.energy_types),
            emission_reduction={},
            ramp=9,
            power_utilization=0.9,
        )
        gen_type.validate_curtailment_cost(
            network, name, curt_cost["energy_curtailment_cost"], exception_list
        )

    assert len(exception_list)
    assert str(exception_list[0]) == exception_msg
