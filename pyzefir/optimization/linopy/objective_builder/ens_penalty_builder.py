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
import logging

import numpy as np
from linopy import LinearExpression

from pyzefir.optimization.linopy.objective_builder import ObjectiveBuilder

_logger = logging.getLogger(__name__)


class EnsPenaltyCostObjectiveBuilder(ObjectiveBuilder):
    def build_expression(self) -> LinearExpression | float:
        if self._ens_penalty_defined():
            return self.build_ens_penalty_expression()
        else:
            return 0.0

    @staticmethod
    def _get_max_or_zero(data: dict[str | int, np.ndarray | float]) -> float:
        result = np.array(list(data.values()))
        return result.max(initial=0.0)

    @property
    def _h_scale(self) -> float:
        """alias for hourly scale"""
        return self.parameters.scenario_parameters.hourly_scale

    def _ens_penalty_defined(self) -> bool:
        """true if the value of ens_penalty_cost is not np.nan (if it was specified by the user)"""
        return not np.isnan(self.parameters.scenario_parameters.ens_penalty_cost)

    def build_ens_penalty_expression(self) -> LinearExpression | float:
        _logger.info("Building ens penalty cost objective...")
        penalty_cost = self._get_ens_penalty()
        _logger.info("Ens penalty set to {}".format(penalty_cost))
        _logger.info("Ens penalty cost objective: Done")
        return (
            self.variables.bus.bus_ens
            * penalty_cost
            * self.indices.years_aggregation_array
        ).sum()

    def _get_ens_penalty(self) -> float:
        ens_penalty_multiplier = self.parameters.scenario_parameters.ens_penalty_cost
        return (
            max(
                self._get_max_var_cost(),
                self._get_max_dsr_cost(),
                self._get_max_opex_cost(),
                self._get_max_capex_cost(),
                self._get_max_transmission_fee_cost(),
            )
            * ens_penalty_multiplier
        )

    def _get_max_var_cost(self) -> float:
        """get max varying cost (fuel cost + env cost) per one energy unit produced"""
        result = 0.0
        for fuel_idx in self.indices.FUEL.mapping:
            result += self._fuel_max_emission_cost_per_energy_unit(
                fuel_idx
            ) + self._fuel_cost_per_energy_unit(fuel_idx)
        return result

    def _fuel_cost_per_energy_unit(self, fuel_idx: int) -> float:
        """get fuel cost per one unit of energy (energy in fuel)"""
        fuel_energy_per_unit = self.parameters.fuel.energy_per_unit[fuel_idx]
        fuel_cost_per_unit = self.parameters.fuel.unit_cost[fuel_idx]
        return (
            fuel_cost_per_unit.max(initial=0.0) / fuel_energy_per_unit
        ) * self._h_scale

    def _fuel_max_emission_cost_per_energy_unit(self, fuel_idx: int) -> float:
        """max amount of emission fees that needs to be paid from one unit of end energy (energy in fuel)"""
        result = 0.0
        energy_per_unit = self.parameters.fuel.energy_per_unit[fuel_idx]
        for emission_type_idx, emission_per_unit in self.parameters.fuel.u_emission[
            fuel_idx
        ].items():
            max_emission_cost = np.array(
                [
                    self.parameters.emf.price[emf_idx].max(initial=0.0)
                    for emf_idx, em_type_idx in self.parameters.emf.emission_type.items()
                    if em_type_idx == emission_type_idx
                ]
            ).max(initial=0.0)
            if emission_per_unit > 0:
                result += (
                    (energy_per_unit / emission_per_unit)
                    * max_emission_cost
                    * self._h_scale
                )

        return result

    def _get_max_capex_cost(self) -> float:
        """max capex cost taken from all generators and storage types"""
        return self._get_max_or_zero(
            self.parameters.tgen.capex | self.parameters.tstor.capex
        )

    def _get_max_opex_cost(self) -> float:
        """max opex cost taken from all generators and storage types"""
        return self._get_max_or_zero(
            self.parameters.tgen.opex | self.parameters.tstor.opex
        )

    def _get_max_dsr_cost(self) -> float:
        """max dsr cost taken from all generators and storage types (scaled by hourly_scale)"""
        return self._get_max_or_zero(self.parameters.dsr.penalization) * self._h_scale

    def _get_max_transmission_fee_cost(self) -> float:
        """max transmission fee value (scaled by hourly_scale)"""
        return self._get_max_or_zero(self.parameters.tf.fee) * self._h_scale
