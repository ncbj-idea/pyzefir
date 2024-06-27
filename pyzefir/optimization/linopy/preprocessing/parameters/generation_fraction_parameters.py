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

from dataclasses import dataclass

from pyzefir.model.network import NetworkElementsDict
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.parameters import ModelParameters


@dataclass
class GenerationFractionParameters(ModelParameters):
    """
    Generation Fraction parameters (tag, subtag, fraction_type, energy_type, min_generation_fraction,
    max_generation_fraction)
    """

    def __init__(
        self, generation_fractions: NetworkElementsDict, indices: Indices
    ) -> None:
        self.tag = self.get_index_from_prop(
            generation_fractions, indices.GF, indices.TAGS, "tag"
        )
        self.sub_tag = self.get_index_from_prop(
            generation_fractions, indices.GF, indices.TAGS, "sub_tag"
        )
        self.et = self.get_index_from_prop(
            generation_fractions, indices.GF, indices.ET, "energy_type"
        )
        self.fraction_type = self.fetch_element_prop(
            generation_fractions, indices.GF, "fraction_type"
        )
        self.min_generation_fraction = self.fetch_element_prop(
            generation_fractions, indices.GF, "min_generation_fraction", indices.Y.ii
        )
        self.max_generation_fraction = self.fetch_element_prop(
            generation_fractions, indices.GF, "max_generation_fraction", indices.Y.ii
        )
