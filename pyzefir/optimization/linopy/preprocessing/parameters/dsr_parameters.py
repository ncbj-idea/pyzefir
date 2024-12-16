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
class DsrParameters(ModelParameters):
    """
    Class representing the demand side response (DSR) parameters.

    This class encapsulates the parameters relevant to demand side response mechanisms, which help manage energy
    demand in response to supply conditions. It retrieves important factors such as compensation factors,
    penalization rates for demand adjustments, and limits on demand shifting.
    """

    def __init__(
        self,
        dsr: NetworkElementsDict,
        indices: Indices,
        scale: float = 1.0,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - dsr (NetworkElementsDict): Dictionary containing DSR elements.
            - indices (Indices): Indices for the DSR parameters.
            - scale (float, optional): Scaling factor for penalization parameters. Defaults to 1.0.
        """
        self.compensation_factor = self.get_prop_from_elements_if_not_none(
            dsr, indices.DSR, "compensation_factor"
        )
        """ compensation factor parameters """
        self.balancing_period_len = self.get_prop_from_elements_if_not_none(
            dsr, indices.DSR, "balancing_period_len"
        )
        """ balancing period length """
        self.penalization_minus = self.scale(
            self.get_prop_from_elements_if_not_none(
                dsr, indices.DSR, "penalization_minus"
            ),
            scale=scale,
        )
        """ penalization parameters for lowering demand dsr """
        self.penalization_plus = self.scale(
            self.get_prop_from_elements_if_not_none(
                dsr, indices.DSR, "penalization_plus"
            ),
            scale=scale,
        )
        """ penalization parameters for increasing demand dsr """
        self.relative_shift_limit = self.get_prop_from_elements_if_not_none(
            dsr, indices.DSR, "relative_shift_limit"
        )
        """ relative shift limit """
        self.abs_shift_limit = self.get_prop_from_elements_if_not_none(
            dsr, indices.DSR, "abs_shift_limit"
        )
        """ absolute shift limit """
        self.balancing_periods = self.get_balancing_periods(dsr, indices)
        """balancing periods for a given dsr"""
        self.hourly_relative_shift_plus_limit = self.get_prop_from_elements_if_not_none(
            dsr, indices.DSR, "hourly_relative_shift_plus_limit"
        )
        """ relative bound for demand compensation """
        self.hourly_relative_shift_minus_limit = (
            self.get_prop_from_elements_if_not_none(
                dsr, indices.DSR, "hourly_relative_shift_minus_limit"
            )
        )
        """ relative bound for hourly demand shift """
