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

from pyzefir.optimization.linopy.constraints_builder.builder import (
    PartialConstraintsBuilder,
)

_logger = logging.getLogger(__name__)


class CapacityBoundsConstraintsBuilder(PartialConstraintsBuilder):
    def build_constraints(self) -> None:
        _logger.info("Capacity bounds constraints is working...")
        years = self.indices.Y.ii[1:]
        capacity_bounds = self.parameters.capacity_bounds
        gen_cap, stor_cap = self.variables.gen.cap, self.variables.stor.cap
        for idx in self.indices.CAP_BOUND.ord:
            lhs_idx, lhs_type, rhs_idx, rhs_type, sense, coeff = (
                capacity_bounds.lhs_idx[idx],
                capacity_bounds.lhs_type[idx],
                capacity_bounds.rhs_idx[idx],
                capacity_bounds.rhs_type[idx],
                capacity_bounds.sense[idx],
                capacity_bounds.coeff[idx],
            )
            left_hs = (
                gen_cap.isel(gen=lhs_idx, year=years) * coeff
                if lhs_type == "GEN"
                else stor_cap.isel(stor=lhs_idx, year=years) * coeff
            )
            right_hs = (
                gen_cap.isel(gen=rhs_idx, year=years)
                if rhs_type == "GEN"
                else stor_cap.isel(stor=rhs_idx, year=years)
            )
            self.model.add_constraints(
                left_hs == right_hs if sense == "EQ" else left_hs <= right_hs,
                name=f"{idx}_CAPACITY_BOUNDS_CONSTRAINT",
            )
