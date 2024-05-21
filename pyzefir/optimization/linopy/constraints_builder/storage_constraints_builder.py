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
    def build_constraints(self) -> None:
        _logger.info("Storage constraints builder is working...")
        self.state_of_charge_upper_bound()
        self.generation_upper_bound()
        self.balance_upper_bound()
        self.boundary_state_of_charge_values()
        self.state_of_charge_definition()
        self.loading_cycles()
        _logger.info("Storage constraints builder is finished!")

    def state_of_charge_upper_bound(self) -> None:
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
        self.model.add_constraints(
            self.variables.stor.gen <= self.variables.stor.soc,
            name="STOR_GENERATION_UPPER_BOUND_CONSTRAINT",
        )
        _logger.debug("Build generation upper bound constraint: Done")

    def balance_upper_bound(self) -> None:
        for st_idx, st_name in self.indices.STOR.mapping.items():
            generation = self.variables.stor.gen.isel(stor=st_idx)
            load = self.variables.stor.load.isel(stor=st_idx)
            nom_p = self.expr.p_inst_st(st_idx)

            self.model.add_constraints(
                generation + load <= nom_p,
                name=f"{st_name}_BALANCE_UPPER_BOUND_CONSTRAINT",
            )
        _logger.debug("Build balance upper bound constraint: Done")

    def boundary_state_of_charge_values(self) -> None:
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
