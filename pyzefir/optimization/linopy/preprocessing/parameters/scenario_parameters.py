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

from numpy import ndarray

from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.parameters import ModelParameters
from pyzefir.optimization.opt_config import OptConfig


class ScenarioParameters(ModelParameters):
    def __init__(
        self,
        indices: Indices,
        opt_config: OptConfig,
        rel_em_limit: dict[str, ndarray],
        base_total_emission: dict[str, float | int],
        power_reserves: dict[str, dict[str, float]],
        ens_penalty_cost: float,
        generator_capacity_cost: str,
    ) -> None:
        self.discount_rate: ndarray = opt_config.discount_rate[indices.Y.ii]
        """ discount rate included in capex formula """
        self.rel_em_limit: dict[str, ndarray] = rel_em_limit
        """ relative emission limit for each year """
        self.hourly_scale: float = opt_config.hourly_scale
        """  ratio of the hours for hours scale """
        self.base_total_emission: dict[str, float | int] = base_total_emission
        """ Total emissions for a given type for the base year """
        """Max percentage usage of the units in the tag"""
        self.power_reserves: dict[str, dict[int, float]] | None = {
            energy_type: {indices.TAGS.inverse[tag]: val for tag, val in res.items()}
            for energy_type, res in power_reserves.items()
        }
        """power reserves for energy type and a given tag"""
        self.money_scale: float = opt_config.money_scale
        """money scale parameter"""
        self.ens_penalty_cost: float = ens_penalty_cost
        """set cost for ens. Letting to be equal zero means no ens"""
        self.generator_capacity_cost: str = generator_capacity_cost
        """set generator_capacity_cost parameter; if netto then additional efficiency incorporated
        in capex/opex"""
