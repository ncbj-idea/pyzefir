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

from pyzefir.model.network import Network
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opt_config,
    run_opt_engine,
    set_network_elements_parameters,
)


@pytest.mark.parametrize(
    (
        "capex",
        "n_consumers",
        "min_device_nom_power",
        "max_device_nom_power",
        "life_time",
    ),
    [
        (
            pd.Series([5000.0] * N_YEARS),
            pd.Series([1000] * N_YEARS),
            10,
            100,
            7,
        ),
        (
            pd.Series([5000.0] * N_YEARS),
            pd.Series([1000] * N_YEARS),
            10,
            100,
            3,
        ),
        (
            pd.Series([5000.0, 5500.0, 6000.0, 6000.0, 5000.0] * N_YEARS),
            pd.Series([500] * N_YEARS),
            20,
            40,
            10,
        ),
        (
            pd.Series([5000.0, 5500.0, 6000.0, 6000.0, 5000.0] * N_YEARS),
            pd.Series([500] * N_YEARS),
            20,
            40,
            3,
        ),
    ],
)
def test_local_supplementary_capacity_upper_bound_constraints(
    capex: pd.Series,
    n_consumers: np.ndarray,
    min_device_nom_power: float,
    max_device_nom_power: float,
    life_time: int,
    network: Network,
) -> None:
    """
    Two LBS, two PV's, one connected to GRID
    No heating
    ens involved
    Test cap evolution, in particular cap_plus, so the contribution to yhe objective from local PV gen
    Test cap flow from one unit to another
    Test fractions
    """

    network.aggregated_consumers["aggr_ee"].n_consumers = n_consumers

    set_network_elements_parameters(
        network.generator_types,
        {
            "pv": {
                "capex": capex,
                "opex": pd.Series([0.0] * N_YEARS),
                "life_time": life_time,
            }
        },
    ),

    set_network_elements_parameters(
        network.generators,
        {
            "local_pv": {
                "min_device_nom_power": min_device_nom_power,
                "max_device_nom_power": max_device_nom_power,
            },
            "local_pv2": {
                "min_device_nom_power": min_device_nom_power,
                "max_device_nom_power": max_device_nom_power,
                "unit_base_cap": 0.0,
            },
        },
    ),

    opt_config = create_default_opt_config(np.arange(100), np.arange(N_YEARS))
    engine = run_opt_engine(network, opt_config)
    res_frac = engine.results.fractions_results.frac["aggr_ee"]
    assert np.allclose(res_frac["lbs_ee"].iloc[:, 0], np.array([1, 0, 0, 0, 0]))
    assert np.allclose(res_frac["lbs_ee2"].iloc[:, 0], np.array([0, 1, 1, 1, 1]))
    # check if cap flow per technology for lt >= N_Years, so in the second year we get the only contribution
    # to the objective from technology, not a given unit (capacity flow: two units, single PV)
    res_cap = engine.results.generators_results.cap
    cap_vals = set(res_cap["local_pv"].iloc[:, 0])
    cap2_vals = set(res_cap["local_pv2"].iloc[:, 0])
    resulted_tcap_plus = engine.results.generators_results.tcap_plus["aggr_ee"][
        "pv"
    ].values.flatten()

    test_data_check = (
        len(cap_vals) == 2
        and len(cap2_vals) == 2
        and 0.0 in cap_vals.intersection(cap2_vals)
    )
    # check if test data is correct: PV(1) decreases, PV(2) increasing such that 0 in both cases
    # plus single non-zero value:
    assert test_data_check
    # test contribution to the objective capex and its evolution:
    cap_increase = max(cap2_vals) - max(cap_vals)
    if life_time >= N_YEARS:
        expected_tcap_plus = np.array([0, cap_increase, 0, 0, 0])
    else:
        expected_tcap_plus = np.array([0, cap_increase, 0, max(cap_vals), cap_increase])
    assert np.allclose(resulted_tcap_plus, expected_tcap_plus)
