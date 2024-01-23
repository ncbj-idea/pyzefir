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

from pyzefir.parser.elements_parsers.energy_source_unit_parser import (
    EnergySourceUnitParser,
)


def test_create(energy_source_unit_parser: EnergySourceUnitParser) -> None:
    """Test if create method calls _create_generator method and _create_storage method."""
    generators, storages = energy_source_unit_parser.create()

    assert isinstance(generators, tuple)
    assert len(generators) == 3
    assert set(gen.name for gen in generators) == {
        "GENERATOR_1",
        "GENERATOR_2",
        "GENERATOR_3",
    }

    assert isinstance(storages, tuple)
    assert len(storages) == 3
    assert set(strg.name for strg in storages) == {
        "STORAGE_1",
        "STORAGE_2",
        "STORAGE_3",
    }
