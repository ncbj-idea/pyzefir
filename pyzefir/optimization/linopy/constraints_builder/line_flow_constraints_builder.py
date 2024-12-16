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
    """
    Class for building line flow constraints in an optimization model.

    This class is responsible for constructing constraints that limit the flow
    of energy through transmission lines based on their maximum capacities.
    It ensures that the flow does not exceed the defined maximum limits for
    each line in the network.
    """

    def build_constraints(self) -> None:
        """
        Builds constraints including:
        - maximum flow constraints
        """
        _logger.info("Line flow constraints builder is working...")
        self.build_max_flow_constraints()
        _logger.info("Line flow constraints builder is finished!")

    def build_max_flow_constraints(self) -> None:
        """
        Adds maximum flow constraints for each transmission line.

        For each line, this method ensures that the flow of energy does not exceed
        the maximum capacity defined for that line. If the maximum capacity is
        neither NaN nor infinite, a constraint is added to the model to enforce
        this limit.
        """
        for line_idx, line_name in self.indices.LINE.mapping.items():
            max_capacity = self.parameters.line.cap[line_idx]
            if not np.isnan(max_capacity) and not np.isinf(max_capacity):
                self.model.add_constraints(
                    self.variables.line.flow.isel(line=line_idx) <= max_capacity,
                    name=f"{line_name}_LINE_FLOW_UPPER_BOUND_CONSTRAINT",
                )
        _logger.debug("Build max flow constraints: Done")
