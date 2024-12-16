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


class StorageConstraintsBuilder(PartialConstraintsBuilder):
    """
    Class for building storage constraints.

    This class constructs various constraints related to energy storage
    systems in an energy management model. It is responsible for ensuring
    that the state of charge, generation limits, balance, and loading cycles
    adhere to specified bounds and definitions.
    """

    def build_constraints(self) -> None:
        """
        Builds constraints including:
        - state of charge upper bound constraints
        - generation upper bound constraints
        - balance upper bound constraints
        - state of charge definition constraints
        - loading cycles constraints
        """
        _logger.info("Storage constraints builder is working...")
        self.state_of_charge_upper_bound()
        self.generation_upper_bound()
        self.balance_upper_bound()
        self.boundary_state_of_charge_values()
        self.state_of_charge_definition()
        self.loading_cycles()
        self.milp_correction_constraints()
        _logger.info("Storage constraints builder is finished!")

    def state_of_charge_upper_bound(self) -> None:
        """
        Adds state of charge upper bound constraints per storage unit.

        Ensures that the state of charge does not exceed the maximum capacity
        of the storage unit multiplied by its power utilization.
        """
        for st_idx, st_name in self.indices.STOR.mapping.items():
            state_of_charge = self.variables.stor.soc.isel(stor=st_idx)
            capacity = self.variables.stor.cap.isel(stor=st_idx)
            power_utilization = self.parameters.tstor.power_utilization[
                self.parameters.stor.tstor[st_idx]
            ]

            self.model.add_constraints(
                state_of_charge <= capacity * power_utilization,
                name=f"{st_name}_STATE_OF_CHARGE_UPPER_BOUND_CONSTRAINT",
            )
        _logger.debug("Build state of charge upper bound constraint: Done")

    def generation_upper_bound(self) -> None:
        """
        Adds generation upper bound constraints per generator.

        Ensures that generation from the storage unit does not exceed the
        available state of charge.
        """
        self.model.add_constraints(
            self.variables.stor.gen <= self.variables.stor.soc,
            name="STOR_GENERATION_UPPER_BOUND_CONSTRAINT",
        )
        _logger.debug("Build generation upper bound constraint: Done")

    def balance_upper_bound(self) -> None:
        """
        Adds balance upper bounds constraints per storage unit.

        Ensures that the sum of generation and load does not exceed the
        nominal power capacity of the storage unit.
        """
        for st_idx, st_name in self.indices.STOR.mapping.items():
            generation = self.expr.gen_netto_st(st_idx)
            load = self.variables.stor.load.isel(stor=st_idx)
            nom_p = self.expr.p_inst_st(st_idx)

            self.model.add_constraints(
                generation + load <= nom_p,
                name=f"{st_name}_BALANCE_UPPER_BOUND_CONSTRAINT",
            )
        _logger.debug("Build balance upper bound constraint: Done")

    def boundary_state_of_charge_values(self) -> None:
        """
        Adds boundary state of charge value constraints.

        Ensures that the state of charge is equal to 0 at the start of the
        first hour of the year and the last hour of the last year.
        """
        for st_idx, st_name in self.indices.STOR.mapping.items():
            self.model.add_constraints(
                self.variables.stor.soc.isel(stor=st_idx, hour=0, year=0) == 0,
                name=f"{st_name}_INITIAL_STATE_OF_CHARGE_CONSTRAINT",
            )
            self.model.add_constraints(
                self.variables.stor.soc.isel(stor=st_idx, hour=-1, year=-1) == 0,
                name=f"{st_name}_END_STATE_OF_CHARGE_CONSTRAINT",
            )
        _logger.debug("Build boundary state of charge values constraint: Done")

    def loading_cycles(self) -> None:
        """
        Adds loading cycles constraints.

        Ensures that the state of charge is equal to 0 at specified intervals
        defined by the loading cycles.
        """
        for st_idx, st_name in self.indices.STOR.mapping.items():
            cycle_len = self.parameters.stor.cycle_len[st_idx]
            if cycle_len is None:
                continue
            periods = np.array(
                [[h, y] for y in self.indices.Y.ord for h in self.indices.H.ord]
            )[::cycle_len]
            for period in periods:
                hour, year = period
                self.model.add_constraints(
                    self.variables.stor.soc.isel(stor=st_idx, hour=hour, year=year)
                    == 0,
                    name=f"{st_name}_HOUR={hour}_YEAR={year}_LOADING_CYCLES_CONSTRAINT",
                )
        _logger.debug("Build loading cycles constraint: Done")

    def state_of_charge_definition(self) -> None:
        """
        Adds state of charge definition constraints.

        Defines the relationship between the state of charge, generation,
        load, and energy loss over time, ensuring the correct update of
        state of charge.
        """
        for st_idx, st_name in self.indices.STOR.mapping.items():
            soc = self.variables.stor.soc.isel(stor=st_idx)
            gen = self.variables.stor.gen.isel(stor=st_idx)
            load_netto_expr = self.expr.load_netto_st(st_idx)
            type_dict = self.parameters.stor.tstor
            e_loss = self.parameters.tstor.energy_loss[type_dict[st_idx]]

            self.model.add_constraints(
                soc.isel(hour=slice(1, None, None))
                == (1 - e_loss) * soc.isel(hour=slice(None, -1, None))
                - gen.isel(hour=slice(None, -1, None))
                + load_netto_expr.isel(hour=slice(None, -1, None)),
            )

            self.model.add_constraints(
                soc.isel(hour=0, year=slice(1, None, None))
                == (1 - e_loss) * soc.isel(hour=-1, year=slice(None, -1, None))
                - gen.isel(hour=-1, year=slice(None, -1, None))
                + load_netto_expr.isel(hour=-1, year=slice(None, -1, None)),
            )
        _logger.debug("Build state of charge definition constraint: Done")

    def milp_correction_constraints(self) -> None:
        """
        Adds Mixed Integer Linear Programming (MILP) correction constraints to the model
        for storage units.

        This method iterates through binary variables associated with each storage unit
        and applies two constraints:

        1. **Generation constraint**: Ensures that the generation output of the storage unit
        is bounded by a large upper value (1e7) scaled by the corresponding binary variable.
        This effectively enforces that generation is active only when the binary variable
        is set to 1.

        2. **Load constraint**: Ensures that the sum of load and the scaled binary variable
        is also bounded by the upper value. This constrains the load behavior of the storage
        unit, depending on whether the binary variable is active (1) or inactive (0).

        Constraints are named according to the storage index (e.g., `"{storage_idx}_FIX_MILP_GEN_CONSTRAINT"`
        for generation and `"{storage_idx}_FIX_MILP_LOAD_CONSTRAINT"` for load) for better traceability.

        Returns:
            None
        """
        for (
            storage_idx,
            storage_type_idx,
        ), binary_variables in self.variables.stor.milp_bin.items():
            load = self.variables.stor.load.isel(stor=storage_idx)
            generation = self.expr.gen_netto_st(st_idx=storage_idx)
            upper_value = self.parameters.tstor.max_capacity[storage_type_idx].max()
            upper_value = upper_value if not np.isnan(upper_value) else 1e7
            self.model.add_constraints(
                generation <= upper_value * binary_variables,
                name=f"{storage_idx}__MILP_GEN_CONSTRAINT",
            )
            self.model.add_constraints(
                load + upper_value * binary_variables <= upper_value,
                name=f"{storage_idx}__MILP_LOAD_CONSTRAINT",
            )
        _logger.debug("Build milp constraints: Done")
