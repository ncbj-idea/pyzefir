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

import pytest

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.network import Network
from pyzefir.model.network_elements.dsr import DSR
from tests.unit.defaults import (
    CO2_EMISSION,
    ELECTRICITY,
    HEATING,
    PM10_EMISSION,
    default_network_constants,
)
from tests.unit.model.test_network_elements.helpers import assert_same_exception_list


@pytest.fixture
def network() -> Network:
    return Network(
        energy_types=[ELECTRICITY, HEATING],
        network_constants=default_network_constants,
        emission_types=[CO2_EMISSION, PM10_EMISSION],
    )


@pytest.mark.parametrize(
    "dsr, exception_list",
    (
        pytest.param(
            DSR(
                name="DSR_2",
                compensation_factor=10,
                balancing_period_len=0.1,  # noqa
                penalization_minus=10,
                penalization_plus=0,
                relative_shift_limit=1.1,
                abs_shift_limit="string",  # noqa
                hourly_relative_shift_plus_limit=1.4,
                hourly_relative_shift_minus_limit=-0.1,
            ),
            [
                NetworkValidatorException(
                    "DSR attribute 'compensation_factor' for DSR_2 must be an instance of <class 'float'>, "
                    "but it is an instance of <class 'int'> instead"
                ),
                NetworkValidatorException(
                    "DSR attribute 'balancing_period_len' for DSR_2 must be an instance of <class 'int'>, "
                    "but it is an instance of <class 'float'> instead"
                ),
                NetworkValidatorException(
                    "DSR attribute 'penalization_minus' for DSR_2 must be an instance of <class 'float'>, "
                    "but it is an instance of <class 'int'> instead"
                ),
                NetworkValidatorException(
                    "DSR attribute 'penalization_plus' for DSR_2 must be an instance of <class 'float'>, "
                    "but it is an instance of <class 'int'> instead"
                ),
                NetworkValidatorException(
                    "DSR attribute 'abs_shift_limit' for DSR_2 must be an instance of float | None, "
                    "but it is an instance of <class 'str'> instead"
                ),
                NetworkValidatorException(
                    "The value of the compensation_factor is inconsistent with th expected bounds of "
                    "the interval: 0 <= 10 <= 1"
                ),
                NetworkValidatorException(
                    "The value of the relative_shift_limit is inconsistent with th expected bounds of "
                    "the interval: 0 < 1.1 < 1"
                ),
                NetworkValidatorException(
                    "The value of the hourly_relative_shift_plus_limit is inconsistent with th expected bounds of "
                    "the interval: 0 < 1.4 < 1"
                ),
                NetworkValidatorException(
                    "The value of the hourly_relative_shift_minus_limit is inconsistent with th expected bounds of "
                    "the interval: 0 < -0.1 < 1"
                ),
            ],
            id="DSR_is_incorrect",
        ),
    ),
)
def test_dsr_validators(dsr: DSR, exception_list: list[str], network: Network) -> None:
    with pytest.raises(NetworkValidatorException) as e_info:
        dsr.validate(network)
    assert_same_exception_list(list(e_info.value.exceptions), exception_list)


def test_dsr_is_correct() -> None:
    dsr = DSR(
        name="DSR_1",
        compensation_factor=0.1,
        balancing_period_len=10,
        penalization_minus=0.6,
        penalization_plus=0.5,
        relative_shift_limit=0.9,
        abs_shift_limit=None,
    )
    actual_exception_list: list[NetworkValidatorException] = []
    dsr._validate(actual_exception_list)
    assert_same_exception_list(actual_exception_list, [])
