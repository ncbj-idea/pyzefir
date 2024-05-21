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

import xarray as xr
from linopy import LinearExpression

from pyzefir.optimization.linopy.objective_builder import ObjectiveBuilder

_logger = logging.getLogger(__name__)


class OpexObjectiveBuilder(ObjectiveBuilder):
    def build_expression(self) -> LinearExpression:
        _logger.info("Building opex objective...")
        return self.generator_opex() + self.storage_opex()

    def generator_opex(self) -> LinearExpression:
        opex = xr.DataArray(
            [
                self.parameters.tgen.opex[self.parameters.gen.tgen[gen_idx]]
                for gen_idx in self.indices.GEN.ord
            ],
            dims=["gen", "year"],
            coords=[self.indices.GEN.ii, self.indices.Y.ii],
            name="opex",
        )
        _logger.info("Building generator opex expression: Done")
        return (opex * self.variables.gen.cap).sum()

    def storage_opex(self) -> LinearExpression | float:
        if self.indices.STOR.ord.size:
            opex = xr.DataArray(
                [
                    self.parameters.tstor.opex[self.parameters.stor.tstor[stor_idx]]
                    for stor_idx in self.indices.STOR.ord
                ],
                dims=["stor", "year"],
                coords=[self.indices.STOR.ii, self.indices.Y.ii],
                name="opex",
            )
            _logger.info("Building generator opex expression: Done")
            return (opex * self.variables.stor.cap).sum()
        _logger.warning("Size of storage not set, returning default expression.")
        return 0
