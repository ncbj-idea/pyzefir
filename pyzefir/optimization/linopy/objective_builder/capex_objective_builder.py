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
    """
    Class responsible for constructing the capital expenditure (capex)
    objective for energy generators and storage systems.

    This class builds the capex objective by aggregating costs associated
    with global (non-location-specific) and local (location-based) assets.
    It calculates capex for both types of assets using various parameters
    such as capacity increases, lifetime, and discount rates, ensuring that
    the correct amortization and year-based factors are applied.
    """

    def build_expression(self) -> LinearExpression:
        """
        Builds the local capital expenditure objective for generators and
        storage technologies at specific locations.

        Local capex accounts for investments in assets that are tied to
        specific locations or aggregations. This method calculates capex
        costs for each asset type, applying location-specific multipliers
        and other relevant parameters such as technology type and capacity.

        Returns:
            - LinearExpression: The total local capex for generators and storages.
        """
        _logger.info("Building capex objective...")
        return self.global_capex() + self.local_capex()

    def local_capex(self) -> LinearExpression:
        """
        Builds the local capital expenditure objective for generators and
        storage technologies at specific locations.

        Local capex accounts for investments in assets that are tied to
        specific locations or aggregations. This method calculates capex
        costs for each asset type, applying location-specific multipliers
        and other relevant parameters such as technology type and capacity.

        Returns:
            - LinearExpression: The total local capex for generators and storages.
        """
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
        """
        Builds the global capital expenditure objective for generators and
        storage systems.

        Global capex represents investments in technologies that are not
        tied to specific locations. The method calculates total investment
        costs for such global technologies by considering parameters such
        as capacity increases, lifetime, and multipliers, adjusted for
        discount rates and other factors across multiple years.

        Returns:
            - LinearExpression: The total global capex for generators and storages.
        """
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
        Computes the total global investment cost for non-location-based (non-lbs) technologies over multiple years.

        Args:
            cap_plus (Variable): Capex increase variable indexed by unit.
            unit_type_param (GeneratorTypeParameters | StorageTypeParameters):
                Parameters describing the technology type (generators or storages).
            unit_type_idx (dict[int, int]): Mapping of unit index to technology type index.
            non_lbs_unit_idxs (set[int]): Set of indices corresponding to non-lbs (global) units.
            multipliers (dict[int, float] | None): Optional multipliers to adjust capex for
                specific technology types. Defaults to None.

        Returns:
            - LinearExpression | float: Expression representing the total global capex.

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
        Computes the yearly capex for a global (non-lbs) technology.

        This method calculates the investment cost for a single year, for
        a specific technology unit, taking into account the amortization
        over the lifetime of the technology. The calculation is based on
        yearly capacity increases, discount rates, and amortization indicators.

        Args:
            - capex (np.ndarray): Yearly capex cost array.
            - cap_plus (Variable): Variable representing capex increase for each unit.
            - disc_rate (np.ndarray): Yearly discount rate array.
            - lt (int): Lifetime of the technology.
            - s_idx (int): Index of the year for which to calculate the capex.
            - u_idx (int): Index of the technology unit.
            - y_idxs (IndexingSet): Set of all year indices for which to calculate capex.

        Returns:
            - LinearExpression | float: Expression representing the yearly capex cost.
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
        Computes the total local investment cost for location-based (lbs) technologies over multiple years.

        Args:
            - tcap_plus (Variable): Capex increase variable indexed by unit type and location.
            - unit_type_param (GeneratorTypeParameters | StorageTypeParameters):
                Parameters describing the technology type (generators or storages).
            - aggr_map (dict[..., set]): Mapping of aggregate index to technology types.
            - multipliers (dict[int, float] | None): Optional multipliers to adjust capex for
                specific technology types. Defaults to None.

        Returns:
            - LinearExpression | float: Expression representing the total local capex.
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
        Computes the yearly capex for a location-based (lbs) technology.

        This method calculates the investment cost for a single year for
        location-specific assets, taking into account capacity expansions,
        discount rates, amortization, and location-based factors. The yearly
        cost is computed by amortizing the total capex over the asset's
        lifetime and applying discount rates.

        Args:
            - capex (np.ndarray): Yearly capex cost array.
            - tcap_plus (Variable): Variable representing capex increase for each unit.
            - disc_rate (np.ndarray): Yearly discount rate array.
            - lt (int): Lifetime of the technology.
            - s_idx (int): Index of the year for which to calculate the capex.
            - ut_idx (int): Index of the technology type.
            - aggr_idx (int): Aggregate index representing the location.
            - y_idxs (IndexingSet): Set of all year indices for which to calculate capex.

        Returns:
            - LinearExpression | float: Expression representing the yearly capex cost.
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
        Generates the amortization matrix indicator for use in capex calculations.

        Args:
            - lt (int): Lifetime of the asset.
            - yy (IndexingSet): Set of year indices over which to calculate amortization.

        Returns:
            - np.ndarray: Amortization matrix indicator.
        """

        return np.array(
            [
                ((yy.ord >= y) & (yy.ord <= min(y + lt - 1, len(yy)))).astype(int)
                for y in yy.ord
            ]
        )
