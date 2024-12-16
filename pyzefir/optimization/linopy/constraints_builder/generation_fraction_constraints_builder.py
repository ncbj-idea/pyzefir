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
import xarray as xr
from linopy import LinearExpression

from pyzefir.optimization.linopy.constraints_builder.builder import (
    PartialConstraintsBuilder,
)

_logger = logging.getLogger(__name__)


class GenerationFractionConstraintsBuilder(PartialConstraintsBuilder):
    """
    Class for building generation fraction constraints within an optimization model.

    This class is responsible for constructing constraints that regulate the
    generation fractions for different energy types and tags. It establishes
    minimum and maximum generation fraction constraints based on specified
    parameters and the relationships between generators and storage units.
    """

    def build_constraints(self) -> None:
        """
        Builds constraints including:
        - minimum and maximum generation fraction constraints
        """
        _logger.info("Generation fraction constraints builder is working...")
        self.min_max_generation_fraction_constraints()
        _logger.info("Generation fraction  builder is finished!")

    def min_max_generation_fraction_constraints(
        self,
    ) -> None:
        """
        Constructs minimum and maximum generation fraction constraints.

        For each generation fraction index, this method retrieves the relevant
        parameters such as maximum and minimum generation fractions, energy type,
        and associated tags. It adds the constraints to the model based on the
        available years and multipliers for each constraint type (min or max).
        """
        for idx in self.indices.GF.ord:
            max_gen_frac = self.parameters.gf.max_generation_fraction[idx]
            min_gen_frac = self.parameters.gf.min_generation_fraction[idx]
            et = self.parameters.gf.et[idx]
            fraction_type = self.parameters.gf.fraction_type[idx]
            tag = self.parameters.gf.tag[idx]
            sub_tag = self.parameters.gf.sub_tag[idx]
            (
                tag_gen_idxs,
                subtag_gen_idxs,
                tag_stor_idxs,
                subtag_stor_idxs,
            ) = self._get_tags(tag, sub_tag)
            max_years = np.asarray(
                [
                    y
                    for y, el in zip(self.indices.Y.ii, max_gen_frac)
                    if not np.isnan(el)
                ]
            )
            if len(max_years):
                max_multiplier = xr.DataArray(
                    max_gen_frac[~np.isnan(max_gen_frac)],
                    dims=["year"],
                    coords={"year": max_years},
                )
                self._gen_constraints(
                    idx=idx,
                    et=et,
                    tag_gen_idxs=tag_gen_idxs,
                    sub_tag_gen_idxs=subtag_gen_idxs,
                    tag_stor_idxs=tag_stor_idxs,
                    sub_tag_stor_idxs=subtag_stor_idxs,
                    years=max_years,
                    fraction_type=fraction_type,
                    multiplier=max_multiplier,
                    constraint_type="MAX",
                )
            min_years = np.asarray(
                [
                    y
                    for y, el in zip(self.indices.Y.ii, min_gen_frac)
                    if not np.isnan(el)
                ]
            )
            if len(min_years):
                min_multiplier = xr.DataArray(
                    min_gen_frac[~np.isnan(min_gen_frac)],
                    dims=["year"],
                    coords={"year": min_years},
                )
                self._gen_constraints(
                    idx=idx,
                    et=et,
                    tag_gen_idxs=tag_gen_idxs,
                    sub_tag_gen_idxs=subtag_gen_idxs,
                    tag_stor_idxs=tag_stor_idxs,
                    sub_tag_stor_idxs=subtag_stor_idxs,
                    years=min_years,
                    fraction_type=fraction_type,
                    multiplier=min_multiplier,
                    constraint_type="MIN",
                )

    def _gen_constraints(
        self,
        idx: int,
        et: int,
        tag_gen_idxs: set[int],
        sub_tag_gen_idxs: set[int],
        tag_stor_idxs: set[int],
        sub_tag_stor_idxs: set[int],
        years: np.ndarray,
        fraction_type: str,
        multiplier: xr.DataArray,
        constraint_type: str,
    ) -> None:
        """
        Adds generation constraints to the model.

        Args:
            - idx (int): index for the generation fraction
            - et (int): energy type index
            - tag_gen_idxs (set[int]): set of tag generator indices
            - sub_tag_gen_idxs (set[int]): set of subtag generator indices
            - tag_stor_idxs (set[int]): set of storage indices
            - sub_tag_stor_idxs (set[int]): set of subtag storage indices
            - years (np.ndarray): array of years
            - fraction_type (str): type of fraction
            - multiplier (xr.DataArray): array of multipliers
            - constraint_type (str): type of constraint (min or max)
        """
        lhs = self._expr_gen(
            et, sub_tag_gen_idxs, sub_tag_stor_idxs, years, fraction_type
        )
        rhs = self._expr_gen(
            et, tag_gen_idxs, tag_stor_idxs, years, fraction_type, multiplier
        )
        self.model.add_constraints(
            lhs >= rhs if constraint_type == "MIN" else lhs <= rhs,
            name=f"{idx}_{constraint_type}_GENERATION_FRACTION_CONSTRAINT",
        )

    @staticmethod
    def _unit_of_given_tag(unit_tags: dict[int, set[int]], tag_idx: int) -> set[int]:
        """
        Retrieves the indices of units (generators or storages) that contain a specific tag.

        Args:
            - unit_tags (dict[int, set[int]]): Dictionary mapping unit indices to their associated tags.
            - tag_idx (int): Index of the tag to search for.

        Returns:
            - set[int]: A set of unit indices that contain the specified tag.
        """
        return {gen_idx for gen_idx, tag_set in unit_tags.items() if tag_idx in tag_set}

    def _get_tags(
        self, tag: int, sub_tag: int
    ) -> tuple[set[int], set[int], set[int], set[int]]:
        """
        Retrieves generator and storage indices associated with a specified tag and subtag.

        Args:
            - tag (int): Index of the primary tag.
            - sub_tag (int): Index of the subtag.

        Returns:
            - tuple[set[int]]: A tuple containing sets of indices for tag generators,
                subtag generators, tag storages, and subtag storages.
        """
        tag_gen_idxs = GenerationFractionConstraintsBuilder._unit_of_given_tag(
            self.parameters.gen.tags, tag
        )
        sub_tag_gen_idxs = GenerationFractionConstraintsBuilder._unit_of_given_tag(
            self.parameters.gen.tags, sub_tag
        )
        tag_stor_idxs = GenerationFractionConstraintsBuilder._unit_of_given_tag(
            self.parameters.stor.tags, tag
        )
        sub_tag_stor_idxs = GenerationFractionConstraintsBuilder._unit_of_given_tag(
            self.parameters.stor.tags, sub_tag
        )
        return tag_gen_idxs, sub_tag_gen_idxs, tag_stor_idxs, sub_tag_stor_idxs

    def _expr_gen(
        self,
        et: int,
        gen_idxs: set[int],
        stor_idxs: set[int],
        years: np.ndarray,
        fraction_type: str,
        multiplier: xr.DataArray | None = None,
    ) -> LinearExpression | float:
        """
        Calculates the total generation for generators and storages associated with a specified energy type.

        This method sums the generation contributions from both generators and storage
        units for a given energy type, applying any multipliers as necessary. It
        can also aggregate the results on a yearly basis if specified.

        Args:
            - et (int): Index of the energy type.
            - gen_idxs (set[int]): Set of generator indices to consider.
            - stor_idxs (set[int]): Set of storage indices to consider.
            - years (np.ndarray): Array of relevant years for the calculations.
            - fraction_type (str): Type of fraction, which determines how to aggregate results.
            - multiplier (xr.DataArray | None): Optional array of multipliers for scaling the results.

        Returns:
            - LinearExpression | float: The total generation expression, which may be a
                linear expression or a float value.
        """
        gen_et_var = self.variables.gen.gen_et
        stor_et_var = self.variables.stor.gen
        et_name = self.indices.ET.mapping[et]

        gen_part = sum(
            gen_et_var[gen_idx][et_name].isel(year=years) for gen_idx in gen_idxs
        )
        stor_part = sum(
            stor_et_var.isel(stor=stor_idx, year=years)
            * self.parameters.stor.gen_eff[stor_idx]
            for stor_idx in stor_idxs
        )
        final_gen_part, final_stor_part = gen_part, stor_part
        if multiplier is not None:
            final_gen_part = gen_part * multiplier
            final_stor_part = stor_part * multiplier
        if fraction_type == "yearly":
            final_gen_part = final_gen_part.sum("hour") if gen_part else 0.0
            final_stor_part = final_stor_part.sum("hour") if stor_part else 0.0

        return final_gen_part + final_stor_part
