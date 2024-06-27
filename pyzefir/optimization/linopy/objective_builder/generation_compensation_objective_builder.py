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


class GenerationCompensationObjectiveBuilder(ObjectiveBuilder):
    def build_expression(self) -> LinearExpression | float:
        _logger.info("Building generation compensation objective...")
        gen_to_type_dict = {
            k: v
            for k, v in self.parameters.gen.tgen.items()
            if v in self.parameters.tgen.generation_compensation.keys()
        }
        expr = sum(
            [
                self.generator_compensation(gen_idx, tgen_idx)
                for gen_idx, tgen_idx in gen_to_type_dict.items()
            ]
        )
        _logger.info("Variable generation compensation objective: Done")
        return expr

    def generator_compensation(
        self, gen_idx: int, tgen_idx: int
    ) -> LinearExpression | float:
        hourly_scale = self.parameters.scenario_parameters.hourly_scale
        compensation = self.parameters.tgen.generation_compensation[tgen_idx]
        generation = self.variables.gen.gen.isel(gen=gen_idx).sum(["hour"])
        return (
            -generation
            * xr.DataArray(
                compensation,
                dims=["year"],
                coords=[self.indices.Y.ii],
                name="compensation",
            )
            * self.indices.years_aggregation_array
        ).sum() * hourly_scale
