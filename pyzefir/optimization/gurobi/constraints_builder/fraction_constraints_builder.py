# PyZefir
# Copyright (C) 2023-2024 Narodowe Centrum Badań Jądrowych
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

from pyzefir.optimization.gurobi.constraints_builder.builder import (
    PartialConstraintsBuilder,
)


class FractionConstrBuilder(PartialConstraintsBuilder):
    def build_constraints(self) -> None:
        """
        All fractions-related constraints. Those constraints describe local balancing stacks involvement in
        aggregated consumers.
        """
        self.base_fraction_constraint()
        self.fraction_upper_bound_constraint()
        self.lbs_involvement_in_consumer_aggregates_constraint()

    def base_fraction_constraint(self) -> None:
        """
        Fixing fractions value in year y=0 of each local balancing stack in each aggregated consumer.
        """
        self.model.addConstr(
            self.variables.frac.fraction[:, :, 0] == self.parameters.aggr.fr_base,
            name="BASE_FRACTION_CONSTRAINT",
        )

    def fraction_upper_bound_constraint(self) -> None:
        """
        If given local balancing stack lbs is available for a given aggregated consumer aggr, then frac[aggr, lbs] <= 1,
        otherwise frac[aggr, lbs] <= 0.
        """
        lbs_indicator = np.repeat(
            self.parameters.aggr.lbs_indicator[:, :, np.newaxis],
            len(self.indices.Y),
            axis=2,
        )
        self.model.addConstr(
            self.variables.frac.fraction <= lbs_indicator,
            name="FRACTION_UPPER_BOUND_CONSTRAINT",
        )

    def lbs_involvement_in_consumer_aggregates_constraint(self) -> None:
        """
        Sum of all fractions of all local balancing stacks in a given aggregated consumer must be equal to 1 in every
        year.
        """
        self.model.addConstr(
            self.variables.frac.fraction.sum(axis=1) == 1,
            name="LBS_INVOLVEMENT_IN_AGGREGATES_CONSTRAINT",
        )
