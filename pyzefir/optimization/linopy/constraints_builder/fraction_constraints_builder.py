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
    """
    Class for building fraction constraints within a model.

    This class is responsible for constructing constraints that manages the
    behavior of fractions related to local balancing stacks (LBS) in aggregated
    consumers. It ensures that the fractions satisfy various conditions, such
    as base values, upper bounds, and total involvement.
    """

    def build_constraints(self) -> None:
        """
        Builds constraints including:
        - base fraction constraints
        - fraction upper bound constraints
        - lbs involvement in consumer aggregates constraints
        """
        _logger.info("Fraction constraints builder is working...")
        self.build_base_fraction_constraint()
        self.build_fraction_upper_bound_constraint()
        self.build_lbs_involvement_in_consumer_aggregates_constraint()
        _logger.info("Fraction constraints builder is finished!")

    def build_base_fraction_constraint(self) -> None:
        """
        Establishes the base fraction values for year y=0 across all
        local balancing stacks in each aggregated consumer.

        This method uses the base fraction data from the parameters to set the
        initial values of the fractions for the first year in the model.
        """
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
        Implements upper bound constraints on fractions based on local balancing stack availability.

        If a local balancing stack is available for a given aggregated consumer,
        the fraction for that stack must not exceed 1. If it is not available,
        the fraction must be 0.
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
        Ensures that the sum of fractions from all local balancing stacks
        in each aggregated consumer equals 1 for every year.

        This constraint enforces that the total fraction allocation from all
        LBSs must collectively equal 1, maintaining the integrity of the
        fraction distribution among the consumers.
        """
        self.model.add_constraints(
            self.variables.frac.fraction.sum("lbs") == 1.0,
            name="LBS_INVOLVEMENT_IN_CONSUMER_AGGREGATES_CONSTRAINT",
        )
        _logger.debug("Build lbs involvement in consumer aggregates constraint: Done")
