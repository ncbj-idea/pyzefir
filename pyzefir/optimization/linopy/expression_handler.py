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

import numpy as np
import xarray as xr
from linopy import LinearExpression, Variable

from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.linopy.preprocessing.opt_variables import (
    OptimizationVariables,
)


class ExpressionHandler:
    """
    Class encapsulating simple linear expressions that describe important modeling concepts,
    such as gross-net energy conversion, fuel usage, and fuel emissions.

    This class provides methods to handle various expressions related to energy generation,
    storage, transmission, and fuel consumption, allowing for the efficient formulation of
    the optimization model.
    """

    def __init__(
        self,
        indices: Indices,
        variables: OptimizationVariables,
        parameters: OptimizationParameters,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - indices (Indices): The indices used for mapping different variables within the model.
            - variables (OptimizationVariables): The optimization variables relevant to the energy network.
            - parameters (OptimizationParameters): The optimization parameters that govern the behavior
              of the energy model.
        """
        self.indices = indices
        self.parameters = parameters
        self.variables = variables

    def fraction_dem(self, bus_idx: int) -> LinearExpression | float:
        """
        Calculates the demand in a specified bus related to fractions of local technology stacks
        in consumer aggregates.

        This method looks up the local balancing stack (LBS) mapping for the provided bus index,
        retrieves the associated demand and fraction for that LBS, and computes the overall demand
        for that bus by multiplying the demand by the fraction.

        Args:
            - bus_idx (int): The index of the bus for which to calculate the demand.

        Returns:
            - LinearExpression | float:
                - If the bus index is found in the mapping, returns a linear expression representing
                  the fraction demand based on the local technology stacks.
                - If the bus index is not found, returns 0.0.
        """
        if bus_idx not in self.parameters.bus.lbs_mapping:
            return 0.0

        lbs_idx, energy_type = (
            self.parameters.bus.lbs_mapping[bus_idx],
            self.parameters.bus.et[bus_idx],
        )
        aggr_idx = self.parameters.lbs.aggr_idx[lbs_idx]
        dem, frac = (
            self.parameters.aggr.dem[aggr_idx][energy_type],
            self.variables.frac.fraction.isel(aggr=aggr_idx, lbs=lbs_idx),
        )

        return xr.DataArray(dem, dims=["hour", "year"]) * frac

    def gen_netto_g(self, gen_idx: int, energy_type: str) -> LinearExpression:
        """
        Generator netto generation (taking into account losses).

        Args:
            - gen_idx (int): generator index
            - energy_type (str): type of produced energy

        Returns:
            - LinearExpression: linear expression for Generator netto generation
        """
        k = self.parameters.gen.eff[gen_idx][energy_type]
        v = self.variables.gen.gen.isel(gen=gen_idx)
        return self.scale(k, v)

    def gen_netto_st(self, st_idx: int) -> LinearExpression:
        """
        Storage netto generation (taking into account generation losses).

        Args:
            - st_idx (int): storage index

        Returns:
            - LinearExpression: linear expression for storage netto generation
        """
        k = self.parameters.stor.gen_eff[st_idx]
        v = self.variables.stor.gen.isel(stor=st_idx)
        return self.scale(k, v)

    def load_netto_st(self, st_idx: int) -> LinearExpression:
        """
        Storage netto energy loading (taking into account loading losses).

        Args:
            - st_idx (int): storage index

        Returns:
            - MLinExpr: Linear expression for storage netto energy loading
        """
        k = self.parameters.stor.load_eff[st_idx]
        v = self.variables.stor.load.isel(stor=st_idx)
        return self.scale(k, v)

    def netto_flow_l(self, line_idx: int) -> LinearExpression:
        """
        Line netto energy flow (taking into account losses).

        Args:
            - line_idx (int): line index

        Returns:
            - LinearExpression: Linear expression for line netto flow
        """
        k, v = (
            1 - self.parameters.line.loss[line_idx],
            self.variables.line.flow.isel(line=line_idx),
        )
        return self.scale(k, v)

    def p_inst_st(self, st_idx: int) -> LinearExpression:
        """
        Return storage installed power.

        Args:
            - st_idx (int): storage index

        Returns:
            - LinearExpression: linear expression for storage installed power
        """
        k, v = (
            self.parameters.stor.p2cap[st_idx],
            self.variables.stor.cap.isel(stor=st_idx),
        )
        return self.scale(k, v)

    @staticmethod
    def discount_rate(yearly_rate: np.ndarray) -> np.ndarray:
        """
        Vector of discount rates for each year.

        Returns:
            np.ndarray: discount rate
        """
        return np.cumprod((1 + yearly_rate) ** (-1))

    @staticmethod
    def scale(_k: float | np.ndarray, _v: Variable) -> LinearExpression:
        """
        Scales variable _v by _k.

        Args:
            - _k (float): scalar
            - _v (Variable): variable to scale

        Returns:
            - LinearExpression: scaled variable
        """
        return _k * _v

    def fuel_consumption(
        self, fuel_idx: int, gen_idx: int, hourly_scale: float
    ) -> LinearExpression:
        """
        Calculates fuel consumption for a generator.

        This method checks if the specified generator uses the provided fuel type. If it does,
        it computes the total fuel consumption by summing the generator's output over the hours
        and scaling it based on the energy content of the fuel. The result is then adjusted
        by the provided hourly scale.

        Args:
            - fuel_idx (int): The index of the fuel being consumed.
            - gen_idx (int): The index of the generator consuming the fuel.
            - hourly_scale (float): A scaling factor for converting total fuel consumption to an hourly basis.

        Returns:
            - LinearExpression: A linear expression for fuel consumption, scaled to reflect hourly usage.
              If the generator does not use the specified fuel, returns a linear expression of zeros.
        """
        if self.parameters.gen.fuel[gen_idx] != fuel_idx:
            return LinearExpression(np.zeros(len(self.indices.Y)))
        return (
            self.variables.gen.gen.isel(gen=gen_idx).sum(["hour"])
            / self.parameters.fuel.energy_per_unit[fuel_idx]
        ) * hourly_scale
