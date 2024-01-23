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

from pyzefir.model.network_elements.energy_source_types.generator_type import (
    GeneratorType,
)
from pyzefir.model.network_elements.energy_source_types.storage_type import StorageType
from pyzefir.model.network_elements.energy_sources.generator import Generator
from pyzefir.model.network_elements.energy_sources.storage import Storage


def create_unique_array_of_tags(
    gen_items: list[Generator | GeneratorType],
    storage_items: list[Storage | StorageType],
) -> np.ndarray:
    gen_tags = [t for ele in gen_items for t in ele.tags]
    stor_tags = [t for ele in storage_items for t in ele.tags]
    return np.unique(gen_tags + stor_tags)
