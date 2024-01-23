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
from gurobipy import quicksum

from pyzefir.optimization.gurobi.constraints_builder.builder import (
    PartialConstraintsBuilder,
)
from pyzefir.optimization.gurobi.preprocessing.indices import IndexingSet
from pyzefir.optimization.gurobi.preprocessing.parameters.generator_parameters import (
    GeneratorParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.generator_type_parameters import (
    GeneratorTypeParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.storage_parameters import (
    StorageParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.storage_type_parameters import (
    StorageTypeParameters,
)
from pyzefir.optimization.gurobi.preprocessing.variables.generator_variables import (
    GeneratorVariables,
)
from pyzefir.optimization.gurobi.preprocessing.variables.storage_variables import (
    StorageVariables,
)


class ScenarioConstraintsBuilder(PartialConstraintsBuilder):
    def build_constraints(self) -> None:
        self.max_fuel_consumption_constraints()
        self.max_fraction_constraints()
        self.min_fraction_constraints()
        self.max_fraction_increase_constraints()
        self.max_fraction_decrease_constraints()
        self.energy_source_type_capacity_constraints()
        self.energy_source_capacity_constraints()
        self.emission_constraints()

    def max_fuel_consumption_constraints(self) -> None:
        for fuel_idx in self.indices.FUEL.mapping.keys():
            max_fuel_availability = self.parameters.fuel.availability[fuel_idx]
            for year_idx in self.indices.Y.mapping.keys():
                if np.isnan(max_fuel_availability[year_idx]):
                    continue
                self.model.addConstr(
                    quicksum(
                        [
                            self.expr.fuel_consumption(
                                fuel_idx,
                                gen_idx,
                                self.parameters.scenario_parameters.hourly_scale,
                            )[year_idx]
                            for gen_idx in self.indices.GEN.mapping.keys()
                            if fuel_idx == self.parameters.gen.fuel[gen_idx]
                        ]
                    )
                    <= max_fuel_availability[year_idx],
                    name=f"MAX_FUEL_{fuel_idx}_AVAILABILITY_CONSTRAINT_{year_idx}",
                )

    def energy_source_type_capacity_constraints(self) -> None:
        self._generator_type_capacity_constraints()
        self._storage_type_capacity_constraints()

    def energy_source_capacity_constraints(self) -> None:
        self._storage_capacity_constraints()
        self._generator_capacity_constraints()

    def _generator_type_capacity_constraints(self) -> None:
        self._add_cap_constraints_per_energy_source_type(
            energy_source_idx=self.indices.GEN,
            energy_source_type_idx=self.indices.TGEN,
            energy_source_to_type_dict=self.parameters.gen.tgen,
            type_parameters=self.parameters.tgen,
            variables=self.variables.gen,
            element_name="TGEN",
        )

    def _storage_type_capacity_constraints(self) -> None:
        self._add_cap_constraints_per_energy_source_type(
            energy_source_idx=self.indices.STOR,
            energy_source_type_idx=self.indices.TSTOR,
            energy_source_to_type_dict=self.parameters.stor.tstor,
            type_parameters=self.parameters.tstor,
            variables=self.variables.stor,
            element_name="TSTOR",
        )

    def _generator_capacity_constraints(self) -> None:
        self._add_cap_constraints_per_energy_source(
            energy_source_idx=self.indices.GEN,
            parameters=self.parameters.gen,
            variables=self.variables.gen,
            element_name="GEN",
        )

    def _storage_capacity_constraints(self) -> None:
        self._add_cap_constraints_per_energy_source(
            energy_source_idx=self.indices.STOR,
            parameters=self.parameters.stor,
            variables=self.variables.stor,
            element_name="STOR",
        )

    def _add_cap_constraints_per_energy_source(
        self,
        energy_source_idx: IndexingSet,
        parameters: StorageParameters | GeneratorParameters,
        variables: GeneratorVariables | StorageVariables,
        element_name: str,
    ) -> None:
        for idx in energy_source_idx.mapping.keys():
            for year in self.indices.Y.mapping.keys() - [0]:
                unit_min_capacity = parameters.unit_min_capacity[idx][year]
                unit_max_capacity = parameters.unit_max_capacity[idx][year]
                unit_min_capacity_increase = parameters.unit_min_capacity_increase[idx][
                    year
                ]
                unit_max_capacity_increase = parameters.unit_max_capacity_increase[idx][
                    year
                ]

                if not np.isnan(unit_min_capacity):
                    self.model.addConstr(
                        variables.cap[idx, year] >= unit_min_capacity,
                        name=f"{idx}_{element_name}_CAP_MIN_CONSTRAINT",
                    )
                if not np.isnan(unit_max_capacity):
                    self.model.addConstr(
                        variables.cap[idx, year] <= unit_max_capacity,
                        name=f"{idx}_{element_name}_CAP_MAX_CONSTRAINT",
                    )
                if not np.isnan(unit_min_capacity_increase):
                    self.model.addConstr(
                        variables.cap[idx, year] - variables.cap[idx, year - 1]
                        >= unit_min_capacity_increase,
                        name=f"{idx}_{element_name}_DELTA_CAP_MIN_CONSTRAINT",
                    )
                if not np.isnan(unit_max_capacity_increase):
                    self.model.addConstr(
                        variables.cap[idx, year] - variables.cap[idx, year - 1]
                        <= unit_max_capacity_increase,
                        name=f"{idx}_{element_name}_DELTA_CAP_MAX_CONSTRAINT",
                    )

    def _add_cap_constraints_per_energy_source_type(
        self,
        energy_source_idx: IndexingSet,
        energy_source_type_idx: IndexingSet,
        energy_source_to_type_dict: dict[int, int],
        type_parameters: GeneratorTypeParameters | StorageTypeParameters,
        variables: GeneratorVariables | StorageVariables,
        element_name: str,
    ) -> None:
        for type_idx in energy_source_type_idx.mapping.keys():
            energy_sources_idx = [
                energy_source_idx
                for energy_source_idx in energy_source_idx.mapping.keys()
                if energy_source_to_type_dict[energy_source_idx] == type_idx
            ]

            for year_idx in self.indices.Y.mapping.keys() - [0]:
                min_capacity = type_parameters.min_capacity[type_idx][year_idx]
                max_capacity = type_parameters.max_capacity[type_idx][year_idx]
                min_capacity_increase = type_parameters.min_capacity_increase[type_idx][
                    year_idx
                ]
                max_capacity_increase = type_parameters.max_capacity_increase[type_idx][
                    year_idx
                ]

                if not np.isnan(min_capacity):
                    self.model.addConstr(
                        variables.cap[energy_sources_idx, year_idx].sum()
                        >= min_capacity,
                        name=f"{type_idx}_{element_name}_CAP_MIN_CONSTRAINT",
                    )
                if not np.isnan(max_capacity):
                    self.model.addConstr(
                        variables.cap[energy_sources_idx, year_idx].sum()
                        <= max_capacity,
                        name=f"{type_idx}_{element_name}_CAP_MAX_CONSTRAINT",
                    )
                if not np.isnan(min_capacity_increase):
                    self.model.addConstr(
                        variables.cap[energy_sources_idx, year_idx].sum()
                        - variables.cap[energy_sources_idx, year_idx - 1].sum()
                        >= min_capacity_increase,
                        name=f"{type_idx}_{element_name}_DELTA_CAP_MIN_CONSTRAINT",
                    )
                if not np.isnan(max_capacity_increase):
                    self.model.addConstr(
                        variables.cap[energy_sources_idx, year_idx].sum()
                        - variables.cap[energy_sources_idx, year_idx - 1].sum()
                        <= max_capacity_increase,
                        name=f"{type_idx}_{element_name}_DELTA_CAP_MAX_CONSTRAINT",
                    )

    def min_fraction_constraints(self) -> None:
        min_fraction = self.parameters.aggr.min_fraction
        for aggr_idx, fraction_dict in min_fraction.items():
            for lbs_idx, fraction_series in fraction_dict.items():
                not_nan_idx = ~np.isnan(fraction_series)
                if not not_nan_idx.any():
                    continue
                variable_year_frac = self.variables.frac.fraction[
                    aggr_idx, lbs_idx, not_nan_idx
                ]
                self.model.addConstr(
                    variable_year_frac >= fraction_series[not_nan_idx],
                    name=f"{aggr_idx}_{lbs_idx}_FRAC_MIN_CONSTRAINT",
                )

    def max_fraction_constraints(self) -> None:
        max_fraction = self.parameters.aggr.max_fraction
        for aggr_idx, fraction_dict in max_fraction.items():
            for lbs_idx, fraction_series in fraction_dict.items():
                not_nan_idx = ~np.isnan(fraction_series)
                if not not_nan_idx.any():
                    continue
                variable_year_frac = self.variables.frac.fraction[
                    aggr_idx, lbs_idx, not_nan_idx
                ]
                self.model.addConstr(
                    variable_year_frac <= fraction_series[not_nan_idx],
                    name=f"{aggr_idx}_{lbs_idx}_FRAC_MAX_CONSTRAINT",
                )

    def max_fraction_increase_constraints(self) -> None:
        max_fraction_increase = self.parameters.aggr.max_fraction_increase
        for aggr_idx, fraction_dict in max_fraction_increase.items():
            for lbs_idx, fraction_series in fraction_dict.items():
                not_nan_idx = ~np.isnan(fraction_series)
                if not not_nan_idx.any():
                    continue
                not_nan_idx = np.where(not_nan_idx)[0]
                variable_year_frac = self.variables.frac.fraction[aggr_idx, lbs_idx]
                self.model.addConstr(
                    variable_year_frac[not_nan_idx]
                    - variable_year_frac[not_nan_idx - 1]
                    <= fraction_series[not_nan_idx],
                    name=f"{aggr_idx}_{lbs_idx}_FRAC_MAX_INCREASE_CONSTRAINT",
                )

    def max_fraction_decrease_constraints(self) -> None:
        max_fraction_decrease = self.parameters.aggr.max_fraction_decrease
        for aggr_idx, fraction_dict in max_fraction_decrease.items():
            for lbs_idx, fraction_series in fraction_dict.items():
                not_nan_idx = ~np.isnan(fraction_series)
                if not not_nan_idx.any():
                    continue
                not_nan_idx = np.where(not_nan_idx)[0]
                variable_year_frac = self.variables.frac.fraction[aggr_idx, lbs_idx]
                self.model.addConstr(
                    variable_year_frac[not_nan_idx - 1]
                    - variable_year_frac[not_nan_idx]
                    <= fraction_series[not_nan_idx],
                    name=f"{aggr_idx}_{lbs_idx}_FRAC_MAX_DECREASE_CONSTRAINT",
                )

    def emission_constraints(self) -> None:
        for et in self.parameters.scenario_parameters.rel_em_limit.keys():
            if not np.isnan(
                self.parameters.scenario_parameters.base_total_emission[et]
            ):
                base_total_em = (
                    self.parameters.scenario_parameters.base_total_emission[et]
                    * self.parameters.scenario_parameters.hourly_scale
                )
                for y_idx in self.indices.Y.mapping.keys():
                    if not np.isnan(
                        self.parameters.scenario_parameters.rel_em_limit[et][y_idx]
                    ):
                        total_em = quicksum(
                            [
                                (
                                    self.expr.fuel_consumption(
                                        fuel_idx,
                                        gen_idx,
                                        self.parameters.scenario_parameters.hourly_scale,
                                    )[y_idx]
                                    * self.parameters.fuel.u_emission[fuel_idx][et]
                                    * (1 - self.parameters.gen.em_red[gen_idx][et])
                                )
                                for fuel_idx in self.indices.FUEL.mapping.keys()
                                for gen_idx in self.indices.GEN.mapping.keys()
                                if fuel_idx == self.parameters.gen.fuel[gen_idx]
                            ]
                        )
                        self.model.addConstr(
                            total_em
                            <= base_total_em
                            * self.parameters.scenario_parameters.rel_em_limit[et][
                                y_idx
                            ],
                            name=f"{et}_{y_idx}_EMISSIONS_CONSTRAINT",
                        )
