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
class CapacityBoundParameters(ModelParameters):
    """
    Class representing capacity bound parameters in an energy network model.

    This class encapsulates the capacity constraints between different technologies
    (generators and storages) by defining their left-hand side (LHS) and right-hand
    side (RHS) types, indices, senses, and coefficients based on the provided capacity
    bounds.
    """

    def __init__(
        self,
        capacity_bounds: NetworkElementsDict,
        indices: Indices,
        generators: NetworkElementsDict,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - capacity_bounds (NetworkElementsDict): A dictionary mapping capacity
                bound names to their respective capacity bound objects.
            - indices (Indices): An object containing indices for capacity bounds,
                generators, and storages.
            - generators (NetworkElementsDict): A dictionary mapping generator
                names to their respective Generator objects.
        """
        self.lhs_type: dict[int, str] = dict()
        self.rhs_type: dict[int, str] = dict()
        self.lhs_idx: dict[int, int] = dict()
        self.rhs_idx: dict[int, int] = dict()
        self.sense: dict[int, str] = dict()
        self.coeff: dict[int, float] = dict()
        for cap_bound in capacity_bounds.values():
            idx = indices.CAP_BOUND.inverse[cap_bound.name]
            self.lhs_type[idx] = (
                "GEN" if cap_bound.left_technology in generators else "STOR"
            )
            self.rhs_type[idx] = (
                "GEN" if cap_bound.right_technology in generators else "STOR"
            )
            self.lhs_idx[idx] = (
                indices.GEN.inverse[cap_bound.left_technology]
                if self.lhs_type[idx] == "GEN"
                else indices.STOR.inverse[cap_bound.left_technology]
            )
            self.rhs_idx[idx] = (
                indices.GEN.inverse[cap_bound.right_technology]
                if self.rhs_type[idx] == "GEN"
                else indices.STOR.inverse[cap_bound.right_technology]
            )
            self.sense[idx] = cap_bound.sense
            self.coeff[idx] = cap_bound.left_coefficient
