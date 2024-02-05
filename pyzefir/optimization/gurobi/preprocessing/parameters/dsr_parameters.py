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
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.parameters import ModelParameters


@dataclass
class DsrParameters(ModelParameters):
    """Generator parameters"""

    def __init__(
        self,
        dsr: NetworkElementsDict,
        indices: Indices,
    ) -> None:
        self.compensation_factor = self.get_prop_from_elements_if_not_none(
            dsr, indices.DSR, "compensation_factor"
        )
        """ compensation factor parameters """
        self.balancing_period_len = self.get_prop_from_elements_if_not_none(
            dsr, indices.DSR, "balancing_period_len"
        )
        """ balancing period length """
        self.penalization = self.get_prop_from_elements_if_not_none(
            dsr, indices.DSR, "penalization"
        )
        """ penalization parameters for dsr """
        self.relative_shift_limit = self.get_prop_from_elements_if_not_none(
            dsr, indices.DSR, "relative_shift_limit"
        )
        """ relative shift limit """
        self.relative_shift_limit = self.get_prop_from_elements_if_not_none(
            dsr, indices.DSR, "relative_shift_limit"
        )
        """ absolute shift limit """
        self.abs_shift_limit = self.get_prop_from_elements_if_not_none(
            dsr, indices.DSR, "abs_shift_limit"
        )
        """ absolute shift limit """
        self.balancing_periods = self.get_balancing_periods(dsr, indices)
        """balancing periods for a given dsr"""
