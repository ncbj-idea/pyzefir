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
    def build_constraints(self) -> None:
        _logger.info("Ramp constraints builder is working...")
        self.ramp_up_constraint()
        _logger.info("Ramp constraints builder is finished!")

    def ramp_up_constraint(self) -> None:
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
