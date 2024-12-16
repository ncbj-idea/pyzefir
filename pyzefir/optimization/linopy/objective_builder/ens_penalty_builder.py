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
    """
    Class for building the Energy Not Supplied (ENS) penalty cost objective.

    This class calculates the penalty cost associated with ENS in the energy system model,
    ensuring that any penalties related to unmet energy demand are properly accounted for.
    If no ENS penalty is defined in the parameters, it returns 0. The penalty is derived
    from various cost components such as variable costs, DSR costs, capex, and opex.
    """

    def build_expression(self) -> LinearExpression | float:
        """
        Builds the ENS penalty cost expression if an ENS penalty is defined.

        This method either returns the ENS penalty cost expression, built from various
        cost factors associated with ENS penalties, or returns 0 if no such penalty
        is defined in the model's parameters.

        Returns:
            - LinearExpression | float: ENS penalty cost expression or 0.0 if not defined.
        """
        if self._ens_penalty_defined():
            return self.build_ens_penalty_expression()
        else:
            return 0.0

    def _h_scale(self) -> float:
        """Alias for hourly scale from scenario parameters."""
        return self.parameters.scenario_parameters.hourly_scale

    @staticmethod
    def _get_max_or_zero(data: dict[str | int, np.ndarray | float]) -> float:
        """
        Returns the maximum value of provided data, or 0 if data is empty.

        Args:
            - data (dict[str | int, np.ndarray | float]): Input data for which to compute max.

        Returns:
            - float: Maximum value in the data, or 0.0 if empty.
        """
        result = np.array(list(data.values()))
        return result.max(initial=0.0)

    def _ens_penalty_defined(self) -> bool:
        """Returns True if the ENS penalty cost dictionary is not empty."""
        return bool(self.parameters.scenario_parameters.ens_penalty_cost)

    def build_ens_penalty_expression(self) -> LinearExpression | float:
        """
        Builds the ENS penalty cost expression based on energy types and penalty costs.

        This method computes the ENS penalty cost by identifying the energy types (ETs)
        associated with buses, applying penalty coefficients, and scaling them by various
        parameters such as the year aggregation array.

        Returns:
            - LinearExpression | float: Computed ENS penalty cost expression.
        """
        _logger.info("Building ens penalty cost objective...")
        _logger.info("Ens penalty cost objective: Done")
        et_penalty_cost = self._get_ens_penalty()
        expr = 0.0
        for et, penalty_cost in et_penalty_cost.items():
            bus_et = np.array(
                [bus for bus, e_type in self.parameters.bus.et.items() if et == e_type]
            )
            expr += (
                self.variables.bus.bus_ens.isel(bus=bus_et)
                * penalty_cost
                * self.indices.years_aggregation_array
            ).sum()
        _logger.info("Ens penalty set to {}".format(et_penalty_cost))
        return expr

    def _get_ens_penalty(self) -> dict[str, float]:
        """
        Retrieves ENS penalty costs for each energy type based on base costs.

        Returns:
            - dict[str, float]: ENS penalty cost for each energy type.
        """
        base_cost = max(
            self._get_max_var_cost(),
            self._get_max_dsr_cost(),
            self._get_max_opex_cost(),
            self._get_max_capex_cost(),
            self._get_max_transmission_fee_cost(),
        )
        return {
            energy_type: ens_penalty_coefficient * base_cost
            for energy_type, ens_penalty_coefficient in self.parameters.scenario_parameters.ens_penalty_cost.items()
        }

    def _get_max_var_cost(self) -> float:
        """
        Retrieves the maximum variable cost per unit of energy produced.

        Returns:
            - float: Maximum variable cost per unit of energy.
        """
        result = 0.0
        for fuel_idx in self.indices.FUEL.mapping:
            result += self._fuel_max_emission_cost_per_energy_unit(
                fuel_idx
            ) + self._fuel_cost_per_energy_unit(fuel_idx)
        return result

    def _fuel_cost_per_energy_unit(self, fuel_idx: int) -> float:
        """
        Retrieves fuel cost per unit of energy for the specified fuel.

        Args:
            - fuel_idx (int): Index of the fuel type.

        Returns:
            - float: Fuel cost per unit of energy.
        """
        fuel_energy_per_unit = self.parameters.fuel.energy_per_unit[fuel_idx]
        fuel_cost_per_unit = self.parameters.fuel.unit_cost[fuel_idx]
        return (
            fuel_cost_per_unit.max(initial=0.0) / fuel_energy_per_unit
        ) * self._h_scale()

    def _fuel_max_emission_cost_per_energy_unit(self, fuel_idx: int) -> float:
        """
        Retrieves the maximum emission cost per unit of energy for the specified fuel.

        Args:
            - fuel_idx (int): Index of the fuel type.

        Returns:
            - float: Maximum emission cost per unit of energy.
        """
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
                    * self._h_scale()
                )

        return result

    def _get_max_capex_cost(self) -> float:
        """
        Retrieves the maximum capital expenditure (capex) cost from generators and storage.

        Returns:
            - float: Maximum capex cost.
        """
        return max(
            self._get_max_or_zero(self.parameters.tgen.capex),
            self._get_max_or_zero(self.parameters.tstor.capex),
        )

    def _get_max_opex_cost(self) -> float:
        """
        Retrieves the maximum operational expenditure (opex) cost from generators and storage.

        Returns:
            - float: Maximum opex cost.
        """
        return max(
            self._get_max_or_zero(self.parameters.tgen.opex),
            self._get_max_or_zero(self.parameters.tstor.opex),
        )

    def _get_max_dsr_cost(self) -> float:
        """
        Retrieves the maximum demand-side response (DSR) cost scaled by hourly scale.

        Returns:
            - float: Maximum DSR cost.
        """
        return (
            self._get_max_or_zero(self.parameters.dsr.penalization_minus)
            * self._h_scale()
        )

    def _get_max_transmission_fee_cost(self) -> float:
        """
        Retrieves the maximum transmission fee cost scaled by hourly scale.

        Returns:
            - float: Maximum transmission fee cost.
        """
        return self._get_max_or_zero(self.parameters.tf.fee) * self._h_scale()
