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

from pyzefir.optimization.linopy.constraints_builder.builder import (
    PartialConstraintsBuilder,
)

_logger = logging.getLogger(__name__)


class FractionConstraintsBuilder(PartialConstraintsBuilder):
    def build_constraints(self) -> None:
        _logger.info("Fraction constraints builder is working...")
        self.build_base_fraction_constraint()
        self.build_fraction_upper_bound_constraint()
        self.build_lbs_involvement_in_consumer_aggregates_constraint()
        _logger.info("Fraction constraints builder is finished!")

    def build_base_fraction_constraint(self) -> None:
        """Fixing fractions value in year y=0 of each local balancing stack in each aggregated consumer."""
        base_fraction = xr.DataArray(
            data=self.parameters.aggr.fr_base,
            coords=[self.indices.AGGR.ord, self.indices.LBS.ii],
            dims=["aggr", "lbs"],
        )
        self.model.add_constraints(
            self.variables.frac.fraction.sel(year=0) == base_fraction,
            name="BASE_FRACTION_CONSTRAINT",
        )
        _logger.debug("Build base fraction constraint: Done")

    def build_fraction_upper_bound_constraint(self) -> None:
        """
        If given local balancing stack lbs is available for a given aggregated consumer aggr, then frac[aggr, lbs] <= 1,
        otherwise frac[aggr, lbs] <= 0.
        """
        lbs_indicator = xr.DataArray(
            data=self.parameters.aggr.lbs_indicator,
            coords=[self.indices.AGGR.ord, self.indices.LBS.ii],
            dims=["aggr", "lbs"],
        )
        self.model.add_constraints(
            self.variables.frac.fraction <= lbs_indicator,
            name="FRACTION_UPPER_BOUND_CONSTRAINT",
        )
        _logger.debug("Build fraction upper bound constraint: Done")

    def build_lbs_involvement_in_consumer_aggregates_constraint(self) -> None:
        """
        Sum of all fractions of all local balancing stacks in a given aggregated consumer must be equal to 1 in every
        year.
        """
        self.model.add_constraints(
            self.variables.frac.fraction.sum("lbs") == 1.0,
            name="LBS_INVOLVEMENT_IN_CONSUMER_AGGREGATES_CONSTRAINT",
        )
        _logger.debug("Build lbs involvement in consumer aggregates constraint: Done")
