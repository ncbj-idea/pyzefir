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

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.network_elements import (
    Bus,
    Generator,
    GeneratorType,
    Line,
    Storage,
    StorageType,
)
from tests.unit.defaults import ELECTRICITY, HEATING, default_network_constants


def test_bus() -> None:
    bus = Bus("bus_A", energy_type=ELECTRICITY)
    assert bus.energy_type == ELECTRICITY
    assert len(bus.generators) == 0
    assert len(bus.lines_in) == 0
    assert len(bus.lines_out) == 0
    assert len(bus.storages) == 0

    with pytest.raises(TypeError):
        Bus(
            "bus",
            energy_type=ELECTRICITY,
            generators={"gen"},  # noqa
            storage={"storage"},  # noqa
            lines_out={"line"},  # noqa
            lines_in={"line"},  # noqa
        )


def test_bus_attach() -> None:
    bus = Bus("bus_A", energy_type=ELECTRICITY)
    with mock.patch("logging.Logger.debug") as mock_logger:
        bus.attach_generator("gen_1")
        mock_logger.assert_called_with(
            "Generator name: gen_1 added to bus_A generators"
        )
        bus.attach_storage("storage_1")
        mock_logger.assert_called_with(
            "Storage name: storage_1 added to bus_A storages"
        )
        bus.attach_from_line("line_fr_1")
        mock_logger.assert_called_with("Line name: line_fr_1 added to bus_A line_out")
        bus.attach_to_line("line_to_1")
        mock_logger.assert_called_with("Line name: line_to_1 added to bus_A line_in")

        bus.attach_generator("gen_1")
        mock_logger.assert_called_with(
            "Generator name: gen_1 already in bus_A generators"
        )
        bus.attach_storage("storage_1")
        mock_logger.assert_called_with(
            "Storage name: storage_1 already in bus_A storages"
        )
        bus.attach_from_line("line_fr_1")
        mock_logger.assert_called_with("Line name: line_fr_1 already in bus_A line_out")
        bus.attach_to_line("line_to_1")
        mock_logger.assert_called_with("Line name: line_to_1 already in bus_A line_in")

        assert bus.generators == {"gen_1"}
        assert bus.storages == {"storage_1"}
        assert bus.lines_out == {"line_fr_1"}
        assert bus.lines_in == {"line_to_1"}

        bus.attach_generator("gen_2")
        mock_logger.assert_called_with(
            "Generator name: gen_2 added to bus_A generators"
        )
        assert bus.generators == {"gen_1", "gen_2"}


def test_generator() -> None:
    from tests.unit.defaults import default_generator_type

    gen_type = GeneratorType(**default_generator_type)
    gen = Generator(
        name="gen_1",
        bus="bus_A",
        energy_source_type=gen_type.name,
        unit_base_cap=10,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    assert gen.buses == {"bus_A"}
    assert gen_type.inbound_energy_type == {HEATING, ELECTRICITY}

    gen_2 = Generator(
        name="gen_1",
        bus={"bus_A", "bus_B"},
        energy_source_type=gen_type.name,
        unit_base_cap=15,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    assert gen_2.buses == {"bus_A", "bus_B"}

    gen_3 = Generator(
        name="gen_3",
        bus={"bus_A", "bus_C"},
        energy_source_type=gen_type.name,
        unit_base_cap=25,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    assert gen_3.buses == {"bus_A", "bus_C"}

    gen_4 = Generator(
        name="gen_4",
        bus=None,  # noqa
        energy_source_type=gen_type.name,
        unit_base_cap=15,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    assert gen_4.buses == set()

    gen_type2 = GeneratorType(
        **default_generator_type
        | {
            "energy_types": {
                ELECTRICITY,
            }
        }
    )
    assert gen_type2.energy_types == {ELECTRICITY}

    gen_type3 = GeneratorType(**default_generator_type | {"conversion_rate": None})
    gen_6 = Generator(
        name="gen_6",
        bus="bus_A",
        energy_source_type=gen_type3.name,
        unit_base_cap=25,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    assert gen_type3.inbound_energy_type == set()
    assert gen_6.unit_base_cap == 25


def test_storage() -> None:
    from tests.unit.defaults import default_storage_type

    storage_type = StorageType(**default_storage_type)
    storage = Storage(
        name="storage_1",
        bus="bus_A",
        energy_source_type=storage_type.name,
        unit_base_cap=40,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    assert storage.bus == "bus_A"
    assert storage.unit_base_cap == 40


def test_line() -> None:
    transmission_loss = 5e-2
    max_capacity = 43.0
    line = Line(
        name="A->B",
        fr="bus_A",
        to="bus_B",
        energy_type=ELECTRICITY,
        transmission_loss=transmission_loss,
        max_capacity=max_capacity,
    )
    assert line.transmission_loss == transmission_loss
    assert line.max_capacity == max_capacity
    assert line.fr == "bus_A"
    assert line.to == "bus_B"
