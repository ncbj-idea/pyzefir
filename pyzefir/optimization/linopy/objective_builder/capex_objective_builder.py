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
from linopy import LinearExpression
from linopy.variables import Variable

from pyzefir.optimization.linopy.objective_builder import ObjectiveBuilder
from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet
from pyzefir.optimization.linopy.preprocessing.parameters.generator_type_parameters import (
    GeneratorTypeParameters,
)
from pyzefir.optimization.linopy.preprocessing.parameters.storage_type_parameters import (
    StorageTypeParameters,
)
from pyzefir.optimization.linopy.utils import get_generator_types_capacity_multipliers
from pyzefir.utils.functions import get_dict_vals

_logger = logging.getLogger(__name__)


class CapexObjectiveBuilder(ObjectiveBuilder):

    def build_expression(self) -> LinearExpression:
        _logger.info("Building capex objective...")
        return self.global_capex() + self.local_capex()

    def local_capex(self) -> LinearExpression:
        generator_capex = self._local_capex(
            tcap_plus=self.variables.tgen.tcap_plus,
            unit_type_param=self.parameters.tgen,
            aggr_map=self.indices.aggr_tgen_map,
            multipliers=get_generator_types_capacity_multipliers(
                self.parameters.scenario_parameters.generator_capacity_cost,
                self.parameters.tgen,
            ),
        )
        storage_capex = self._local_capex(
            tcap_plus=self.variables.tstor.tcap_plus,
            unit_type_param=self.parameters.tstor,
            aggr_map=self.indices.aggr_tstor_map,
        )
        _logger.info("Building local capex expression: Done")
        return generator_capex + storage_capex

    def global_capex(self) -> LinearExpression:
        generator_capex = self._global_capex(
            cap_plus=self.variables.gen.cap_plus,
            unit_type_param=self.parameters.tgen,
            unit_type_idx=self.parameters.gen.tgen,
            non_lbs_unit_idxs=get_dict_vals(self.parameters.bus.generators).difference(
                get_dict_vals(self.indices.aggr_gen_map)
            ),
            multipliers=get_generator_types_capacity_multipliers(
                self.parameters.scenario_parameters.generator_capacity_cost,
                self.parameters.tgen,
            ),
        )
        storage_capex = self._global_capex(
            cap_plus=self.variables.stor.cap_plus,
            unit_type_param=self.parameters.tstor,
            unit_type_idx=self.parameters.stor.tstor,
            non_lbs_unit_idxs=get_dict_vals(self.parameters.bus.storages).difference(
                get_dict_vals(self.indices.aggr_stor_map)
            ),
        )
        _logger.info("Building global capex expression: Done")
        return generator_capex + storage_capex

    def _global_capex(
        self,
        cap_plus: Variable,
        unit_type_param: GeneratorTypeParameters | StorageTypeParameters,
        unit_type_idx: dict,
        non_lbs_unit_idxs: set,
        multipliers: dict[int, float] | None = None,
    ) -> LinearExpression | float:
        """
        Total investment cost for global (non-lbs) technologies for all years

        Args:
            cap_plus (Variable): capex increase labeled by unit index
            unit_type_param (GeneratorTypeParameters | StorageTypeParameters): technology type index
            unit_type_idx: dict[int, int]: technology index of a given unit index
            non_lbs_unit_idxs: set[int]: set of non-lbs (global) units

        Returns:
            Linear expression of capex cost

        """
        disc_rate = self.expr.discount_rate(
            self.parameters.scenario_parameters.discount_rate
        )
        y_idxs = self.indices.Y
        unit_capex = 0.0
        for u_idx in non_lbs_unit_idxs:
            ut_idx = unit_type_idx[u_idx]
            capex = unit_type_param.capex[ut_idx]
            lt = unit_type_param.lt[ut_idx]
            mul = multipliers[ut_idx] if multipliers is not None else 1.0
            for s_idx in y_idxs.ord:
                unit_capex += (
                    self.global_capex_per_unit_per_year(
                        capex=capex,
                        cap_plus=cap_plus,
                        disc_rate=disc_rate,
                        lt=lt,
                        s_idx=s_idx,
                        u_idx=u_idx,
                        y_idxs=y_idxs,
                    )
                    * mul
                )
        return unit_capex

    @staticmethod
    def global_capex_per_unit_per_year(
        capex: np.ndarray,
        cap_plus: Variable,
        disc_rate: np.ndarray,
        lt: int,
        s_idx: int,
        u_idx: int,
        y_idxs: IndexingSet,
    ) -> LinearExpression | float:
        """
        Capex for a given year for global (non-lbs) technologies
        Takes a single year index as a single argument s_idx, calculates the corresponding capex cost in this year
        The function also requires the whole set of years, given by y_idxs


        Args:
            capex (ndarray): yearly capex cost parameter
            cap_plus (Variable): capex increase labeled by unit index
            disc_rate (ndarray): yearly discount rate
            lt (int): life time of a given technology
            s_idx (int): single year index, specifying the year for capex calculation
            ut_idx (int): technology type index
            aggr_idx (int): index of a given aggregate
            y_idxs (IndexingSet): set of years

        Returns:
            Linear expression of a yearly investment cost
        """
        am_indicator = CapexObjectiveBuilder._amortization_matrix_indicator(
            lt=lt, yy=y_idxs
        )
        res = 0.0
        for y_idx in y_idxs.ord:
            res += (
                am_indicator[y_idx, s_idx]
                * capex[y_idx]
                * cap_plus.sel(index=(u_idx, y_idx))
                * disc_rate[s_idx]
                / lt
            )
        return res

    def _local_capex(
        self,
        tcap_plus: Variable,
        unit_type_param: GeneratorTypeParameters | StorageTypeParameters,
        aggr_map: dict[..., set],
        multipliers: dict[int, float] | None = None,
    ) -> LinearExpression | float:
        """
        Total investment cost for local (lbs) technologies for all years

        Args:
            tcap_plus (Variable): capex increase labeled by unit type index and aggregate index
            unit_type_param (GeneratorTypeParameters | StorageTypeParameters): technology type index
            unit_type_idx: dict[int, int]: technology index of a given unit index
            non_lbs_unit_idxs: set[int]: set of non-lbs (global) units

        Returns:
            Linear expression of capex cost

        """
        disc_rate = self.expr.discount_rate(
            self.parameters.scenario_parameters.discount_rate
        )
        y_idxs = self.indices.Y
        unit_type_capex = 0.0
        for aggr_idx, ut_idxs in aggr_map.items():
            for ut_idx in ut_idxs:
                mul = multipliers[ut_idx] if multipliers is not None else 1.0
                capex = unit_type_param.capex[ut_idx]
                lt = unit_type_param.lt[ut_idx]
                for s_idx in y_idxs.ord:
                    unit_type_capex += (
                        self.local_capex_per_unit_per_year(
                            capex=capex,
                            tcap_plus=tcap_plus,
                            disc_rate=disc_rate,
                            lt=lt,
                            s_idx=s_idx,
                            ut_idx=ut_idx,
                            aggr_idx=aggr_idx,
                            y_idxs=y_idxs,
                        )
                        * mul
                    )
        return unit_type_capex

    @staticmethod
    def local_capex_per_unit_per_year(
        capex: np.ndarray,
        tcap_plus: Variable,
        disc_rate: np.ndarray,
        lt: int,
        s_idx: int,
        ut_idx: int,
        aggr_idx: int,
        y_idxs: IndexingSet,
    ) -> LinearExpression | float:
        """
        Capex for a given year for local (lbs) technologies
        Takes a single year index as a single argument s_idx, calculates the corresponding capex cost in this year
        The function also requires the whole set of years, given by y_idxs

        Args:
            capex (ndarray): yearly capex cost parameter
            tcap_plus (Variable): capex increase labeled by unit type index and aggregate index
            disc_rate (ndarray): yearly discount rate
            lt (int): life time of a given technology
            s_idx (int): single year index, specifying the year for capex calculation
            ut_idx (int): technology type index
            aggr_idx (int): index of a given aggregate
            y_idxs (IndexingSet): set of years

        Returns:
            Linear expression of a yearly investment cost
        """
        am_indicator = CapexObjectiveBuilder._amortization_matrix_indicator(
            lt=lt, yy=y_idxs
        )
        res = 0.0
        for y_idx in y_idxs.ord:
            res += (
                am_indicator[y_idx, s_idx]
                * tcap_plus.sel(index=(aggr_idx, ut_idx, y_idx))
                * capex[y_idx]
                * disc_rate[s_idx]
                / lt
            )
        return res

    @staticmethod
    def _amortization_matrix_indicator(
        lt: int,
        yy: IndexingSet,
    ) -> np.ndarray:
        """
        Indicator matrix for y-index range in capex expression specifying the summation in capex expression
        The resulting matrix is composed of 0 and 1; zero means there is no capex contribution from the corresponding
        element

        Args:
            lt (int): unit lifetime
            yy (IndexingSet): year indices

        Returns:
            np.ndarray
        """

        return np.array(
            [
                ((yy.ord >= y) & (yy.ord <= min(y + lt - 1, len(yy)))).astype(int)
                for y in yy.ord
            ]
        )
