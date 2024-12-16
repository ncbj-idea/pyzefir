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

from pyzefir.optimization.linopy.constraints_builder.builder import (
    PartialConstraintsBuilder,
)

_logger = logging.getLogger(__name__)


class RampConstraintsBuilder(PartialConstraintsBuilder):
    """
    Class for building ramp constraints for generators in an optimization model.

    This class is responsible for constructing constraints that regulate the
    ramping behavior of generators. It ensures that the ramp up and ramp down
    rates of generators comply with specified limits based on their capacity.
    """

    def build_constraints(self) -> None:
        """
        Builds constraints including:
        - ramp up constraints
        """
        _logger.info("Ramp up constraints builder is working...")
        self.ramp_up_constraint()
        _logger.info("Ramp up constraints builder is finished!")

    def ramp_up_constraint(self) -> None:
        """
        Constructs ramp up and ramp down constraints for generators.

        For each generator, this method calculates the ramp up and ramp down constraints
        based on the generator's capacity and the specified ramp rates. It ensures that
        the changes in generation between consecutive hours do not exceed the allowed
        ramping rates.
        """
        for gen_idx, gen_name in self.indices.GEN.mapping.items():
            t_idx = self.parameters.gen.tgen[gen_idx]
            ramp_down = self.parameters.tgen.ramp_down[t_idx]
            ramp_up = self.parameters.tgen.ramp_up[t_idx]
            if not np.isnan(ramp_up) or not np.isnan(ramp_down):
                gen = self.variables.gen.gen
                cap = self.variables.gen.cap
                gen_ramp = gen.isel(gen=gen_idx, hour=slice(1, None, None)) - gen.isel(
                    gen=gen_idx, hour=slice(None, -1, None)
                )
                if not np.isnan(ramp_up):
                    self.model.add_constraints(
                        gen_ramp <= cap.isel(gen=gen_idx) * ramp_up,
                        name=f"{gen_name}_RAMP_UP_CONSTRAINT",
                    )
                if not np.isnan(ramp_down):
                    self.model.add_constraints(
                        -gen_ramp <= cap.isel(gen=gen_idx) * ramp_down,
                        name=f"{gen_name}_RAMP_DOWN_CONSTRAINT",
                    )
        _logger.debug("Build ramp up constraint: Done")
