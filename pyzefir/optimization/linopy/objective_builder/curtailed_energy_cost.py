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


class CurtailedEnergyCostObjectiveBuilder(ObjectiveBuilder):
    """
    Class for building the objective function to calculate curtailed energy cost for generators.

    This class defines the construction of the objective that calculates the cost
    associated with curtailed energy, which is the energy generated but not used (curtailed)
    by generators in the energy system. The objective is computed based on generator
    curtailment variables, hourly scaling factors, and curtailment cost parameters.
    """

    def build_expression(self) -> LinearExpression | float:
        """
        Constructs the curtailed energy cost objective for all generators.

        This method creates the curtailed energy cost objective by iterating over
        generators and applying the cost of curtailed energy to each unit. The
        cost calculation is based on energy curtailment variables, the curtailment
        cost per generator type, and scaling factors such as the hourly scale.
        It sums up the curtailed energy cost across all years and generator types.

        Returns:
            - LinearExpression | float: The total curtailed energy cost expression or 0.0.
        """
        _logger.info("Building curtailed energy cost objective...")
        hourly_scale = self.parameters.scenario_parameters.hourly_scale
        curtailment_cost = self.parameters.tgen.energy_curtailment_cost
        gen_ett = {
            k: {self.indices.ET.inverse[et] for et in v}
            for k, v in self.parameters.gen.ett.items()
        }
        gen_idx_to_tgen_idx = {
            gen_idx: tgen_idx
            for gen_idx, tgen_idx in self.parameters.gen.tgen.items()
            if tgen_idx in curtailment_cost
        }
        result: LinearExpression | float = 0.0
        for gen_idx, tgen_idx in gen_idx_to_tgen_idx.items():
            curtailment_cost_per_year = xr.DataArray(
                curtailment_cost[tgen_idx],
                dims=["year"],
                coords={"year": self.indices.Y.ii},
            )
            for et in gen_ett[gen_idx]:
                result += (
                    self.variables.gen.dump_et[gen_idx][self.indices.ET.mapping[et]]
                    * self.indices.years_aggregation_array
                    * curtailment_cost_per_year
                    * hourly_scale
                )
        _logger.info("Curtailed energy cost objective: Done")
        if isinstance(result, LinearExpression):
            return result.sum()
        return result
