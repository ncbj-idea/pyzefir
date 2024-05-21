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
from collections import defaultdict

from pyzefir.optimization.linopy.constraints_builder.builder import (
    PartialConstraintsBuilder,
)

_logger = logging.getLogger(__name__)


class CapacityBindingBuilder(PartialConstraintsBuilder):
    def build_constraints(self) -> None:
        _logger.info("Ramp constraints builder is working...")
        self.ramp_up_constraint()
        _logger.info("Ramp constraints builder is finished!")

    def ramp_up_constraint(self) -> None:
        grouped_gens = defaultdict(list)
        for gen_idx, cb_name in self.parameters.gen.capacity_binding.items():
            grouped_gens[cb_name].append(gen_idx)
        cap = self.variables.gen.cap
        for binding_name, grouped_gen in grouped_gens.items():
            if len(grouped_gen) >= 2:
                ref_gen_idx, *other_gen_idxs = grouped_gen
                ref_cap = cap.isel(gen=ref_gen_idx, year=self.indices.Y.ii[1:])
                for gen_idx in other_gen_idxs:
                    self.model.add_constraints(
                        ref_cap == cap.isel(gen=gen_idx, year=self.indices.Y.ii[1:]),
                        name=f"{ref_gen_idx}_{gen_idx}_CAPACITY_BINDING_{binding_name}_CONSTRAINT",
                    )
        _logger.debug("Build ramp up constraint: Done")
