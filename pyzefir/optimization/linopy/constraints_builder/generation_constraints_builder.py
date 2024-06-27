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
from pyzefir.utils.functions import demand_chunk_unit_indices

_logger = logging.getLogger(__name__)


class GenerationConstraintsBuilder(PartialConstraintsBuilder):
    def build_constraints(self) -> None:
        _logger.info("Generation constraints builder is working...")
        self.generation_vs_capacity_constraints()
        self.generation_and_dump_energy()
        self.total_dump_energy()
        _logger.info("Generation constraints builder is finished!")

    def generation_vs_capacity_constraints(self) -> None:
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
        for gen_idx, gen_name in self.indices.GEN.mapping.items():
            for energy_type_idx, energy_type_name in self.indices.ET.mapping.items():
                energy_type_generation = self.variables.gen.gen_et.isel(
                    gen=gen_idx, et=energy_type_idx
                )
                energy_type_dump_energy = self.variables.gen.dump_et.isel(
                    gen=gen_idx, et=energy_type_idx
                )
                if energy_type_name in self.parameters.gen.ett[gen_idx]:
                    generation_brutto = self.variables.gen.gen.isel(gen=gen_idx)
                    efficiency = xr.DataArray(
                        self.parameters.tgen.eff[self.parameters.gen.tgen[gen_idx]][
                            energy_type_name
                        ],
                        dims=["hour"],
                        coords={"hour": self.indices.H.ii},
                    )
                    demch_params = self.parameters.demand_chunks_parameters

                    # TODO: check if this is correct (dem_idxs always empty)
                    dem_idxs = [
                        dem_idx
                        for dem_idx, tag in demch_params.tag.items()
                        if gen_idx
                        in demand_chunk_unit_indices(
                            dem_idx, demch_params.tag, self.parameters.gen.tags
                        )
                        and demch_params.energy_type[dem_idx] == energy_type_name
                    ]
                    dch = self.variables.gen.gen_dch.isel(
                        et=energy_type_idx, demch=dem_idxs, gen=gen_idx
                    )

                    if dch.size == 0:
                        dch = 0
                    self.model.add_constraints(
                        generation_brutto * efficiency
                        == energy_type_generation + energy_type_dump_energy + dch,
                        name=f"{gen_name}_{energy_type_idx}_GENERATION_DUMP_CONSTRAINT",
                    )
                else:
                    self.model.add_constraints(
                        energy_type_dump_energy + energy_type_generation == 0,
                        name=f"{gen_name}_{energy_type_idx}_GENERATION_DUMP_CONSTRAINT",
                    )
        _logger.debug("Build generation and dump energy constraints: Done")

    def total_dump_energy(self) -> None:
        for gen_idx, gen_name in self.indices.GEN.mapping.items():
            de_et_sum = 0.0
            for et_name in self.parameters.gen.ett[gen_idx]:
                de_et_sum += self.variables.gen.dump_et.isel(
                    gen=gen_idx,
                    et=self.indices.ET.inverse[et_name],
                ) / xr.DataArray(
                    self.parameters.tgen.eff[self.parameters.gen.tgen[gen_idx]][
                        et_name
                    ],
                    dims=["hour"],
                    coords={"hour": self.indices.H.ii},
                )

            self.model.add_constraints(
                de_et_sum == self.variables.gen.dump.isel(gen=gen_idx),
                name=f"{gen_name}_TOTAL_DUMP_ENERGY",
            )
        _logger.debug("Build total dump energy constraints: Done")
