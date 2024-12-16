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
from linopy import LinearExpression

from pyzefir.optimization.linopy.constraints_builder.builder import (
    PartialConstraintsBuilder,
)

_logger = logging.getLogger(__name__)


class GenerationConstraintsBuilder(PartialConstraintsBuilder):
    """
    Class for building generation constraints within a model.

    This class is responsible for constructing constraints that regulate the
    behavior of generation units within the optimization model. It ensures
    that generation is correctly calculated based on various factors such as
    capacity, utilization, efficiency, and energy types.
    """

    def build_constraints(self) -> None:
        """
        Builds constraints including:
        - generation in relation to capacity constraints
        - generation and dump energy constraints
        """
        _logger.info("Generation constraints builder is working...")
        self.generation_vs_capacity_constraints()
        self.generation_and_dump_energy()
        _logger.info("Generation constraints builder is finished!")

    def generation_vs_capacity_constraints(self) -> None:
        """
        Adds generation vs capacity constraints for each generator.

        For each generator, the method establishes constraints that relate
        generation to capacity, power utilization, and capacity factor. If a
        capacity factor is available, generation is set to equal the product
        of capacity, power utilization, and the capacity factor. If not,
        generation must be lower than capacity times power utilization and
        greater than or equal to capacity times minimal power utilization.
        """
        for gen_idx, gen_name in self.indices.GEN.mapping.items():
            generation_brutto = self.variables.gen.gen.isel(gen=gen_idx)
            capacity = self.variables.gen.cap.isel(gen=gen_idx)
            capacity_factor_id = self.parameters.gen.capacity_factors[gen_idx]
            gen_to_tgen = self.parameters.gen.tgen
            power_utilization = xr.DataArray(
                self.parameters.tgen.power_utilization[gen_to_tgen[gen_idx]],
                dims=["hour"],
                coords={"hour": self.indices.H.ii},
            )
            minimal_power_utilization = xr.DataArray(
                self.parameters.tgen.minimal_power_utilization[gen_to_tgen[gen_idx]],
                dims=["hour"],
                coords={"hour": self.indices.H.ii},
            )
            if capacity_factor_id is not None:
                capacity_factor = xr.DataArray(
                    self.parameters.cf.profile[capacity_factor_id],
                    dims=["hour"],
                    coords={"hour": self.indices.H.ii},
                )
                self.model.add_constraints(
                    generation_brutto == capacity_factor * capacity * power_utilization,
                    name=f"{gen_name}_NON_DISPATCHABLE_GEN_CAP_CONSTRAINT",
                )
            else:
                self.model.add_constraints(
                    generation_brutto <= capacity * power_utilization,
                    name=f"{gen_name}_DISPATCHABLE_GEN_CAP_CONSTRAINT",
                )
                self.model.add_constraints(
                    generation_brutto >= capacity * minimal_power_utilization,
                    name=f"{gen_name}_DISPATCHABLE_MIN_POWER_UTILIZATION_CONSTRAINT",
                )
        _logger.debug("Build generation vs capacity constraints: Done")

    def generation_and_dump_energy(self) -> None:
        """
        Adds constraints for generation losses and energy dumping.

        For each generator and each energy type, this method constructs constraints
        that account for generation efficiency, energy dumping, and generation losses.
        It ensures that the relationship between generated energy, dumped energy, and
        demand chunk energy is maintained.
        """
        for gen_idx, gen_name in self.indices.GEN.mapping.items():
            for et in self.parameters.gen.ett[gen_idx]:
                gen_et = self.variables.gen.gen_et[gen_idx][et]
                disable_dump = self.parameters.tgen.disable_dump_energy[
                    self.parameters.gen.tgen[gen_idx]
                ]
                dump_et = (
                    self.variables.gen.dump_et[gen_idx][et] if not disable_dump else 0.0
                )
                gen = self.variables.gen.gen.isel(gen=gen_idx)
                eff = xr.DataArray(
                    self.parameters.tgen.eff[self.parameters.gen.tgen[gen_idx]][et],
                    dims=["hour"],
                    coords={"hour": self.indices.H.ii},
                )
                dch_gen = self.generator_demand_chunk_expr(gen_idx, et)
                gen_reserve_et = self.reserve_expr(gen_idx, et)
                self.model.add_constraints(
                    gen * eff == gen_et + gen_reserve_et + dump_et + dch_gen,
                    name=f"{gen_name}_{et}_GENERATION_ENERGY_LOSSES_CONSTRAINT",
                )

        _logger.debug("Build generation and dump energy constraints: Done")

    def generator_demand_chunk_expr(
        self, gen_idx: int, et: str
    ) -> LinearExpression | float:
        """
        Returns the expression describing the generation of a given generator
        associated with all demand chunks of a specific energy type.

        Args:
            - gen_idx (int): The index of the generator for which to compute the expression.
            - et (str): The energy type for which the demand chunk expression is calculated.

        Returns:
            - LinearExpression | float: The demand chunk expression for the generator,
                or 0.0 if no demand chunks are associated with the specified energy type.
        """
        result = 0.0
        if len(self.indices.DEMCH) == 0:
            return result
        gen_dch = self.variables.gen.gen_dch
        dch_et = self.parameters.demand_chunks_parameters.energy_type
        for dch_idx in self.parameters.gen.demand_chunks.get(gen_idx, []):
            if dch_et[dch_idx] == et:
                result += gen_dch[dch_idx][gen_idx]
        return result

    def reserve_expr(self, gen_idx: int, et: str) -> LinearExpression | float:
        """
        Returns the expression for generation blocked by power reserve

        Args:
            - gen_idx (int): The index of the generator for which to compute the expression.
            - et (str): The energy type for which the demand chunk expression is calculated.

        Returns:
            - LinearExpression | float: the reserve generator variable or zero
        """
        gen_et_reserve = self.variables.gen.gen_reserve_et
        result = 0.0
        for tag in gen_et_reserve:
            if gen_idx in gen_et_reserve[tag] and et in gen_et_reserve[tag][gen_idx]:
                result += gen_et_reserve[tag][gen_idx][et]
        return result
