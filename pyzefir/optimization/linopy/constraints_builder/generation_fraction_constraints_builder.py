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
    def build_constraints(self) -> None:
        _logger.info("Generation fraction constraints builder is working...")
        self.min_max_generation_fraction_constraints()
        _logger.info("Generation fraction  builder is finished!")

    def min_max_generation_fraction_constraints(
        self,
    ) -> None:
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
        """for given tag_idx and subtag_idx methods returns gen_idx | stor_idx containing provided tags"""
        return {gen_idx for gen_idx, tag_set in unit_tags.items() if tag_idx in tag_set}

    def _get_tags(
        self, tag: int, sub_tag: int
    ) -> tuple[set[int], set[int], set[int], set[int]]:
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
        """generation of generators and storages in a given tag"""
        gen_et_var = self.variables.gen.gen_et
        stor_et_var = self.variables.stor.gen

        gen_part = gen_et_var.isel(gen=list(gen_idxs), et=et, year=years).sum("gen")
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
