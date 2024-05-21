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


class LineFlowConstraintsBuilder(PartialConstraintsBuilder):
    def build_constraints(self) -> None:
        _logger.info("Line flow constraints builder is working...")
        self.build_max_flow_constraints()
        _logger.info("Line flow constraints builder is finished!")

    def build_max_flow_constraints(self) -> None:
        for line_idx, line_name in self.indices.LINE.mapping.items():
            max_capacity = self.parameters.line.cap[line_idx]
            if not np.isnan(max_capacity):
                self.model.add_constraints(
                    self.variables.line.flow.isel(line=line_idx) <= max_capacity,
                    name=f"{line_name}_LINE_FLOW_UPPER_BOUND_CONSTRAINT",
                )
        _logger.debug("Build max flow constraints: Done")
