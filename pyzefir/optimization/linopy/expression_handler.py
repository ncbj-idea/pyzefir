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
    """Encapsulation of simple linear expressions

    Encapsulation of simple linear expressions, that describes important
    modelling concepts (brutto-netto energy conversion, fuel usage, fuel emission etc.)
    """

    def __init__(
        self,
        indices: Indices,
        variables: OptimizationVariables,
        parameters: OptimizationParameters,
    ) -> None:
        self.indices = indices
        self.parameters = parameters
        self.variables = variables

    def fraction_dem(self, bus_idx: int) -> LinearExpression | float:
        """Fraction demand

        Demand in bus related to fractions of local technology stacks in consumer aggregates.

        Args:
            bus_idx (int): bus index
        Returns:
            fraction_dem
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
        """Generator netto generation (taking into account losses)

        Args:
            gen_idx (int): generator index
            energy_type (str): type of produced energy
        Returns:
            MLinExpr: Linear expression for Generator netto generation
        """
        k = self.parameters.gen.eff[gen_idx][energy_type]
        v = self.variables.gen.gen.isel(gen=gen_idx)
        return self.scale(k, v)

    def gen_netto_st(self, st_idx: int) -> LinearExpression:
        """Storage netto generation (taking into account generation losses)

        Args:
            st_idx (int): storage index
        Returns:
            MLinExpr: Expression for storage netto generation
        """
        k = self.parameters.stor.gen_eff[st_idx]
        v = self.variables.stor.gen.isel(stor=st_idx)
        return self.scale(k, v)

    def load_netto_st(self, st_idx: int) -> LinearExpression:
        """Storage netto energy loading (taking into account loading losses)

        Args:
            st_idx (int): storage index
        Returns:
            MLinExpr: Linear expression for storage netto energy loading
        """
        k = self.parameters.stor.load_eff[st_idx]
        v = self.variables.stor.load.isel(stor=st_idx)
        return self.scale(k, v)

    def netto_flow_l(self, line_idx: int) -> LinearExpression:
        """Line netto energy flow (taking into account losses)

        Args:
            line_idx (int): line index
        Returns:
             MLinExpr: Linear expression for line netto flow
        : MLinExpr
        """
        k, v = (
            1 - self.parameters.line.loss[line_idx],
            self.variables.line.flow.isel(line=line_idx),
        )
        return self.scale(k, v)

    def p_inst_st(self, st_idx: int) -> LinearExpression:
        """Storage installed power

        Args:
            st_idx (int): storage index
        Returns:
            MLinExpr: Linear expression for storage installed power
        """
        k, v = (
            self.parameters.stor.p2cap[st_idx],
            self.variables.stor.cap.isel(stor=st_idx),
        )
        return self.scale(k, v)

    @staticmethod
    def discount_rate(yearly_rate: np.ndarray) -> np.ndarray:
        """Vector of discount rates for each year.

        Returns:
            np.ndarray: discount rate
        """
        return np.cumprod((1 + yearly_rate) ** (-1))

    @staticmethod
    def scale(_k: float | np.ndarray, _v: Variable) -> LinearExpression:
        return _k * _v

    def fuel_consumption(
        self, fuel_idx: int, gen_idx: int, hourly_scale: float
    ) -> LinearExpression:
        """Fuel consumption

        Args:
            fuel_idx (int): fuel index
            gen_idx (int): generator index
            hourly_scale: (float): hourly scale
        Returns:
            MLinExpr: Linear expression for fuel consumption multiply by hourly scale
        """

        if self.parameters.gen.fuel[gen_idx] != fuel_idx:
            return LinearExpression(np.zeros(len(self.indices.Y)))
        return (
            self.variables.gen.gen.isel(gen=gen_idx).sum(["hour"])
            / self.parameters.fuel.energy_per_unit[fuel_idx]
        ) * hourly_scale
