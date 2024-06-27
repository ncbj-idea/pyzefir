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

from pyzefir.optimization.linopy.constraints_builder.builder import (
    PartialConstraintsBuilder,
)
from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet
from pyzefir.optimization.linopy.preprocessing.parameters.generator_parameters import (
    GeneratorParameters,
)
from pyzefir.optimization.linopy.preprocessing.parameters.generator_type_parameters import (
    GeneratorTypeParameters,
)
from pyzefir.optimization.linopy.preprocessing.parameters.storage_parameters import (
    StorageParameters,
)
from pyzefir.optimization.linopy.preprocessing.parameters.storage_type_parameters import (
    StorageTypeParameters,
)
from pyzefir.optimization.linopy.preprocessing.variables.generator_variables import (
    GeneratorVariables,
)
from pyzefir.optimization.linopy.preprocessing.variables.storage_variables import (
    StorageVariables,
)
from pyzefir.utils.functions import invert_dict_of_sets

_logger = logging.getLogger(__name__)


class ScenarioConstraintsBuilder(PartialConstraintsBuilder):
    def build_constraints(self) -> None:
        _logger.info("Scenario constraints builder is working...")
        self.max_fuel_consumption_constraints()
        self.max_fraction_constraints()
        self.min_fraction_constraints()
        self.max_fraction_increase_constraints()
        self.max_fraction_decrease_constraints()
        self.energy_source_type_capacity_constraints()
        self.energy_source_capacity_constraints()
        self.emission_constraints()
        self.power_reserve_constraint()
        _logger.info("Scenario constraints builder is finished!")

    def max_fuel_consumption_constraints(self) -> None:
        for fuel_idx in self.indices.FUEL.mapping.keys():
            if fuel_idx not in self.parameters.fuel.availability or all(
                np.isnan(self.parameters.fuel.availability[fuel_idx])
            ):
                continue
            max_fuel_availability = self.parameters.fuel.availability[fuel_idx]
            for year_idx in self.indices.Y.mapping.keys():
                if np.isnan(max_fuel_availability[year_idx]):
                    continue

                total_fuel_consumption = 0.0
                for gen_idx in self.indices.GEN.mapping.keys():
                    if fuel_idx == self.parameters.gen.fuel[gen_idx]:
                        total_fuel_consumption += self.expr.fuel_consumption(
                            fuel_idx,
                            gen_idx,
                            self.parameters.scenario_parameters.hourly_scale,
                        ).isel(year=year_idx)

                if total_fuel_consumption:
                    self.model.add_constraints(
                        total_fuel_consumption <= max_fuel_availability[year_idx],
                        name=f"MAX_FUEL_{fuel_idx}_AVAILABILITY_CONSTRAINT_{year_idx}",
                    )
        _logger.debug("Build max fuel consumption constraints: Done")

    def energy_source_type_capacity_constraints(self) -> None:
        _logger.debug("Building energy source type capacity constraints...")
        self._generator_type_capacity_constraints()
        self._storage_type_capacity_constraints()
        _logger.debug("Build energy source type capacity constraints: Done")

    def energy_source_capacity_constraints(self) -> None:
        _logger.debug("Building energy source capacity constraints...")
        self._storage_capacity_constraints()
        self._generator_capacity_constraints()
        _logger.debug("Build energy source capacity constraints: Done")

    def _generator_type_capacity_constraints(self) -> None:
        self._add_cap_constraints_per_energy_source_type(
            energy_source_idx=self.indices.GEN,
            energy_source_to_type_dict=self.parameters.gen.tgen,
            type_parameters=self.parameters.tgen,
            variables=self.variables.gen,
            element_name="GEN",
        )
        _logger.debug("Build generator type capacity constraints: Done")

    def _storage_type_capacity_constraints(self) -> None:
        self._add_cap_constraints_per_energy_source_type(
            energy_source_idx=self.indices.STOR,
            energy_source_to_type_dict=self.parameters.stor.tstor,
            type_parameters=self.parameters.tstor,
            variables=self.variables.stor,
            element_name="STOR",
        )
        _logger.debug("Build storage type capacity constraints: Done")

    def _generator_capacity_constraints(self) -> None:
        self._add_cap_constraints_per_energy_source(
            energy_source_idx=self.indices.GEN,
            parameters=self.parameters.gen,
            variables=self.variables.gen,
            element_name="GEN",
        )
        _logger.debug("Build generator capacity constraints: Done")

    def _storage_capacity_constraints(self) -> None:
        self._add_cap_constraints_per_energy_source(
            energy_source_idx=self.indices.STOR,
            parameters=self.parameters.stor,
            variables=self.variables.stor,
            element_name="STOR",
        )
        _logger.debug("Build storage capacity constraints: Done")

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
                    self.model.add_constraints(
                        variables.cap.isel({element_name.lower(): idx, "year": year})
                        >= unit_min_capacity,
                        name=f"{idx}_{year}_{element_name}_CAP_MIN_CONSTRAINT",
                    )
                if not np.isnan(unit_max_capacity):
                    self.model.add_constraints(
                        variables.cap.isel({element_name.lower(): idx, "year": year})
                        <= unit_max_capacity,
                        name=f"{idx}_{year}_{element_name}_CAP_MAX_CONSTRAINT",
                    )
                if not np.isnan(unit_min_capacity_increase):
                    self.model.add_constraints(
                        variables.cap.isel({element_name.lower(): idx, "year": year})
                        - variables.cap.isel(
                            {element_name.lower(): idx, "year": year - 1}
                        )
                        >= unit_min_capacity_increase,
                        name=f"{idx}_{year}_{element_name}_DELTA_CAP_MIN_CONSTRAINT",
                    )
                if not np.isnan(unit_max_capacity_increase):
                    self.model.add_constraints(
                        variables.cap.isel({element_name.lower(): idx, "year": year})
                        - variables.cap.isel(
                            {element_name.lower(): idx, "year": year - 1}
                        )
                        <= unit_max_capacity_increase,
                        name=f"{idx}_{year}_{element_name}_DELTA_CAP_MAX_CONSTRAINT",
                    )

    def _add_cap_constraints_per_energy_source_type(
        self,
        energy_source_idx: IndexingSet,
        energy_source_to_type_dict: dict[int, int],
        type_parameters: GeneratorTypeParameters | StorageTypeParameters,
        variables: GeneratorVariables | StorageVariables,
        element_name: str,
    ) -> None:
        for type_idx in dict.fromkeys(energy_source_to_type_dict.values()):
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
                    self.model.add_constraints(
                        variables.cap.isel(
                            **{
                                element_name.lower(): energy_sources_idx,
                                "year": year_idx,
                            }
                        ).sum()
                        >= min_capacity,
                        name=f"{type_idx}_{year_idx}_T{element_name}_CAP_MIN_CONSTRAINT",
                    )
                if not np.isnan(max_capacity):
                    self.model.add_constraints(
                        variables.cap.isel(
                            **{
                                element_name.lower(): energy_sources_idx,
                                "year": year_idx,
                            }
                        ).sum()
                        <= max_capacity,
                        name=f"{type_idx}_{year_idx}_T{element_name}_CAP_MAX_CONSTRAINT",
                    )
                if not np.isnan(min_capacity_increase):
                    self.model.add_constraints(
                        variables.cap.isel(
                            **{
                                element_name.lower(): energy_sources_idx,
                                "year": year_idx,
                            }
                        ).sum()
                        - variables.cap.isel(
                            **{
                                element_name.lower(): energy_sources_idx,
                                "year": year_idx - 1,
                            }
                        ).sum()
                        >= min_capacity_increase,
                        name=f"{type_idx}_{year_idx}_T{element_name}_DELTA_CAP_MIN_CONSTRAINT",
                    )
                if not np.isnan(max_capacity_increase):
                    self.model.add_constraints(
                        variables.cap.isel(
                            **{
                                element_name.lower(): energy_sources_idx,
                                "year": year_idx,
                            }
                        ).sum()
                        - variables.cap.isel(
                            **{
                                element_name.lower(): energy_sources_idx,
                                "year": year_idx - 1,
                            }
                        ).sum()
                        <= max_capacity_increase,
                        name=f"{type_idx}_{year_idx}_T{element_name}_DELTA_CAP_MAX_CONSTRAINT",
                    )

    def min_fraction_constraints(self) -> None:
        min_fraction = self.parameters.aggr.min_fraction
        for aggr_idx, fraction_dict in min_fraction.items():
            for lbs_idx, fraction_series in fraction_dict.items():
                not_nan_idx = ~np.isnan(fraction_series)
                if not not_nan_idx.any():
                    continue
                variable_year_frac = self.variables.frac.fraction.isel(
                    aggr=aggr_idx, lbs=lbs_idx, year=not_nan_idx
                )
                lbs_min_fraction = xr.DataArray(
                    fraction_series,
                    dims="year",
                    coords={"year": self.indices.Y.ii},
                ).isel(year=not_nan_idx)

                self.model.add_constraints(
                    variable_year_frac >= lbs_min_fraction,
                    name=f"{aggr_idx}_{lbs_idx}_FRAC_MIN_CONSTRAINT",
                )
        _logger.debug("Build min fraction constraints: Done")

    def max_fraction_constraints(self) -> None:
        max_fraction = self.parameters.aggr.max_fraction
        for aggr_idx, fraction_dict in max_fraction.items():
            for lbs_idx, fraction_series in fraction_dict.items():
                not_nan_idx = ~np.isnan(fraction_series)
                if not not_nan_idx.any():
                    continue
                variable_year_frac = self.variables.frac.fraction.isel(
                    aggr=aggr_idx, lbs=lbs_idx, year=not_nan_idx
                )
                lbs_max_fraction = xr.DataArray(
                    fraction_series,
                    dims="year",
                    coords={"year": self.indices.Y.ii},
                ).isel(year=not_nan_idx)

                self.model.add_constraints(
                    variable_year_frac <= lbs_max_fraction,
                    name=f"{aggr_idx}_{lbs_idx}_FRAC_MAX_CONSTRAINT",
                )
        _logger.debug("Build max fraction constraints: Done")

    def max_fraction_increase_constraints(self) -> None:
        max_fraction_increase = self.parameters.aggr.max_fraction_increase
        for aggr_idx, fraction_dict in max_fraction_increase.items():
            for lbs_idx, fraction_series in fraction_dict.items():
                not_nan_idx = ~np.isnan(fraction_series)
                if not not_nan_idx.any():
                    continue
                not_nan_idx = np.where(not_nan_idx)[0]
                variable_year_frac = self.variables.frac.fraction.isel(
                    aggr=aggr_idx, lbs=lbs_idx
                )
                lbs_max_fraction_increase = xr.DataArray(
                    fraction_series,
                    dims="year",
                    coords={"year": self.indices.Y.ii},
                ).isel(year=not_nan_idx)

                self.model.add_constraints(
                    variable_year_frac.isel(year=not_nan_idx)
                    - variable_year_frac.isel(year=not_nan_idx - 1)
                    <= lbs_max_fraction_increase,
                    name=f"{aggr_idx}_{lbs_idx}_FRAC_MAX_INCREASE_CONSTRAINT",
                )
        _logger.debug("Build max fraction increase constraints: Done")

    def max_fraction_decrease_constraints(self) -> None:
        max_fraction_decrease = self.parameters.aggr.max_fraction_decrease
        for aggr_idx, fraction_dict in max_fraction_decrease.items():
            for lbs_idx, fraction_series in fraction_dict.items():
                not_nan_idx = ~np.isnan(fraction_series)
                if not not_nan_idx.any():
                    continue
                not_nan_idx = np.where(not_nan_idx)[0]
                variable_year_frac = self.variables.frac.fraction.isel(
                    aggr=aggr_idx, lbs=lbs_idx
                )
                lbs_max_fraction_decrease = xr.DataArray(
                    fraction_series,
                    dims="year",
                    coords={"year": self.indices.Y.ii},
                ).isel(year=not_nan_idx)

                self.model.add_constraints(
                    variable_year_frac.isel(year=not_nan_idx - 1)
                    - variable_year_frac.isel(year=not_nan_idx)
                    <= lbs_max_fraction_decrease,
                    name=f"{aggr_idx}_{lbs_idx}_FRAC_MAX_DECREASE_CONSTRAINT",
                )
        _logger.debug("Build max fraction decrease constraints: Done")

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
                        total_em = 0.0
                        for fuel_idx in self.indices.FUEL.mapping.keys():
                            for gen_idx in self.indices.GEN.mapping.keys():
                                if fuel_idx == self.parameters.gen.fuel[gen_idx]:
                                    total_em += (
                                        self.expr.fuel_consumption(
                                            fuel_idx,
                                            gen_idx,
                                            self.parameters.scenario_parameters.hourly_scale,
                                        ).isel(year=y_idx)
                                        * self.parameters.fuel.u_emission[fuel_idx][et]
                                        * (
                                            1
                                            - self.parameters.gen.em_red[gen_idx][et][
                                                y_idx
                                            ]
                                        )
                                    )

                        self.model.add_constraints(
                            total_em
                            <= base_total_em
                            * self.parameters.scenario_parameters.rel_em_limit[et][
                                y_idx
                            ],
                            name=f"{et}_{y_idx}_EMISSIONS_CONSTRAINT",
                        )
        _logger.debug("Build emission constraints: Done")

    def power_reserve_constraint(self) -> None:
        power_reserves = self.parameters.scenario_parameters.power_reserves
        if power_reserves:
            cap = self.variables.gen.cap
            gen_et = self.variables.gen.gen_et
            gens_of_tag = invert_dict_of_sets(self.parameters.gen.tags)
            for energy_type, tag_to_reserve in power_reserves.items():
                et = self.indices.ET.inverse[energy_type]
                for tag, reserve in tag_to_reserve.items():
                    self.model.add_constraints(
                        cap.isel(gen=list(gens_of_tag[tag])).sum(["gen"])
                        - gen_et.isel(gen=list(gens_of_tag[tag]), et=et).sum(["gen"])
                        >= reserve,
                        name=f"ENERGY_TYPE_{et}_TAG_{tag}_POWER_RESERVE_CONSTRAINT",
                    )
        _logger.debug("Build power reserve constraints: Done")
