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
