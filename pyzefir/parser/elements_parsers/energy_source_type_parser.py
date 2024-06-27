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

from collections import defaultdict
from copy import deepcopy
from typing import Literal

import numpy as np
import pandas as pd

from pyzefir.model.network_elements import GeneratorType, StorageType
from pyzefir.parser.csv_parser import CsvParserException
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser
from pyzefir.parser.elements_parsers.utils import create_tags_list


class EnergySourceTypeParserException(CsvParserException):
    pass


class EnergySourceTypeParser(AbstractElementParser):

    def __init__(
        self,
        cost_parameters_df: pd.DataFrame,
        storage_type_df: pd.DataFrame,
        energy_mix_evolution_limits_df: pd.DataFrame,
        conversion_rate: dict[str, pd.DataFrame],
        generators_efficiency: pd.DataFrame,
        generators_emission_reduction: pd.DataFrame,
        generators_energy_type: pd.DataFrame,
        generators_fuel_type: pd.DataFrame,
        generators_type: pd.DataFrame,
        generators_power_utilization: pd.DataFrame,
        n_years: int,
        n_hours: int,
        curtailment_cost: pd.DataFrame,
        generators_series_efficiency: dict[str, pd.DataFrame],
        generation_compensation: pd.DataFrame,
        yearly_emission_reduction: pd.DataFrame,
        generators_minimal_power_utilization: pd.DataFrame,
    ) -> None:
        self.generators_energy_type = generators_energy_type.copy(deep=True).set_index(
            "generator_type"
        )
        self.generators_emission_reduction = generators_emission_reduction.copy(
            deep=True
        ).set_index("generator_type")
        self.generators_efficiency = generators_efficiency.copy(deep=True).set_index(
            "generator_type", drop=True
        )
        self.conversion_rate = deepcopy(conversion_rate)
        self.energy_mix_evolution_limits_df = energy_mix_evolution_limits_df.copy(
            deep=True
        )
        self.storage_type_df = storage_type_df.copy(deep=True).set_index("storage_type")
        self.cost_parameters_df = cost_parameters_df.copy(deep=True)
        self.generators_fuel_type = generators_fuel_type.copy(deep=True)
        self.generators_type = generators_type.copy(deep=True)
        self.generators_power_utilization = generators_power_utilization.copy(
            deep=True
        ).set_index("hour_idx")
        self.n_years = n_years
        self.n_hours = n_hours
        self.curtailment_cost = curtailment_cost.copy(deep=True)
        self.generators_series_efficiency = deepcopy(generators_series_efficiency)
        self.generation_compensation = deepcopy(generation_compensation)
        self.yearly_emission_reduction = yearly_emission_reduction.copy(deep=True)
        self.generators_minimal_power_utilization = (
            generators_minimal_power_utilization.copy(deep=True)
        )

    def _prepare_energy_source_parameters(self) -> dict[str, pd.DataFrame]:
        """
        Prepare common parameters for GeneratorType and StorageType
        classes (EnergySourceType parameters).
        """
        energy_source_types = self.cost_parameters_df["technology_type"].unique()
        result_dict = dict()
        for en_source_type in energy_source_types:
            cost_df = self.cost_parameters_df[
                self.cost_parameters_df["technology_type"] == en_source_type
            ].set_index("year_idx", drop=True)
            evolution_limits_df = self.energy_mix_evolution_limits_df[
                self.energy_mix_evolution_limits_df["technology_type"] == en_source_type
            ].set_index("year_idx", drop=True)
            result_dict[en_source_type] = pd.concat(
                [cost_df, evolution_limits_df], axis=1
            )
            result_dict[en_source_type].drop("technology_type", inplace=True, axis=1)

        return result_dict

    @staticmethod
    def _prepare_generator_emission_reduction(
        df_er: pd.DataFrame,
        df_yer: pd.DataFrame,
        n_years: int,
    ) -> dict[str, dict[str, pd.Series]]:
        result_dict: dict[str, dict[str, pd.Series]] = defaultdict(dict)
        emission_dict: dict[str, dict[str, list[float]]] = defaultdict(dict)
        if not df_yer.empty:
            for emission_type, group in df_yer.groupby("emission_type"):
                for col in group.columns[2:]:
                    emission_dict[col][emission_type] = group[col].dropna().tolist()
        for generator in df_er.index:
            for emission_type in df_er.columns:
                first_value = df_er.loc[generator, emission_type]
                remaining_values = emission_dict.get(generator, {}).get(emission_type)
                if not remaining_values:
                    result_dict[generator][emission_type] = pd.Series(
                        [first_value] * n_years
                    )
                else:
                    result_dict[generator][emission_type] = pd.Series(
                        [first_value] + remaining_values
                    )
        return dict(result_dict)

    @staticmethod
    def _prepare_conversion_rate_dict(
        conversion_rate: dict[str, pd.DataFrame],
    ) -> dict[str, dict[str, pd.Series]]:
        conv_rate_dict: dict[str, dict[str, pd.Series]] = dict()
        for generator, conv_df in conversion_rate.items():
            conv_df = conv_df.set_index("hour_idx")
            series_dict = {col: conv_df[col] for col in conv_df.columns}
            conv_rate_dict[generator] = series_dict
        return conv_rate_dict

    def _create_generator_type(
        self,
        df_row: pd.Series,
        energy_source_type_df: dict[str, pd.DataFrame],
        efficiency_df: pd.DataFrame,
        conv_dict: dict[str, dict[str, pd.Series]],
        emission_reduction_dict: dict[str, dict[str, pd.Series]],
    ) -> GeneratorType:
        name = df_row["name"]
        energy_source_df = energy_source_type_df[name]
        fuel_row = self.generators_fuel_type[
            self.generators_fuel_type["generator_type"] == name
        ]
        energy_types = self.generators_energy_type.loc[name]
        if isinstance(energy_types, pd.DataFrame):
            energy_types = energy_types.squeeze()
        energy_types = set(energy_types)
        gen_type = GeneratorType(
            name=name,
            life_time=int(df_row["life_time"]),
            build_time=int(df_row["build_time"]),
            capex=energy_source_df["CAPEX"],
            opex=energy_source_df["OPEX"],
            min_capacity=energy_source_df["min_capacity"].reindex(range(self.n_years)),
            max_capacity=energy_source_df["max_capacity"].reindex(range(self.n_years)),
            min_capacity_increase=energy_source_df["min_capacity_increase"].reindex(
                range(self.n_years)
            ),
            max_capacity_increase=energy_source_df["max_capacity_increase"].reindex(
                range(self.n_years)
            ),
            efficiency=self._get_generator_efficiency(name, efficiency_df),
            energy_types=energy_types,
            emission_reduction=emission_reduction_dict[name],
            fuel=(
                None
                if pd.isna(fuel_row["fuel_name"]).all()
                else str(fuel_row["fuel_name"].iloc[0])
            ),
            capacity_factor=(
                None
                if pd.isna(fuel_row["capacity_factor_name"]).all()
                else str(fuel_row["capacity_factor_name"].iloc[0])
            ),
            power_utilization=self._get_power_utilization_boundaries(
                name, df_row, self.generators_power_utilization, "power_utilization"
            ),
            minimal_power_utilization=self._get_power_utilization_boundaries(
                name,
                df_row,
                self.generators_minimal_power_utilization,
                "minimal_power_utilization",
            ),
            ramp_down=df_row["ramp_down"] if "ramp_down" in df_row else np.nan,
            ramp_up=df_row["ramp_up"] if "ramp_up" in df_row else np.nan,
            tags=create_tags_list(df_row[7:]),
            generation_compensation=(
                self.generation_compensation[name]
                if name in self.generation_compensation.columns
                else None
            ),
        )
        if conv_rate := conv_dict.get(name):
            gen_type.conversion_rate = conv_rate
        curt_cost = self.curtailment_cost.get(name)
        if curt_cost is not None and len(curt_cost):
            gen_type.energy_curtailment_cost = curt_cost
        return gen_type

    def _create_storage_type(
        self,
        df_row: pd.Series,
        energy_source_type_df: dict[str, pd.DataFrame],
    ) -> StorageType:
        name = str(df_row.name)
        energy_source_df = energy_source_type_df[name]
        if not np.isnan(df_row["power_utilization"]):
            power_utilization = float(df_row["power_utilization"])
        else:
            power_utilization = 1.0

        energy_loss = self.storage_type_df.loc[name]["energy_loss"]
        energy_loss = 0.0 if np.isnan(energy_loss) else float(energy_loss)

        return StorageType(
            name=name,
            life_time=int(df_row["life_time"]),
            build_time=int(df_row["build_time"]),
            capex=energy_source_df["CAPEX"],
            opex=energy_source_df["OPEX"],
            min_capacity=energy_source_df["min_capacity"].reindex(range(self.n_years)),
            max_capacity=energy_source_df["max_capacity"].reindex(range(self.n_years)),
            min_capacity_increase=energy_source_df["min_capacity_increase"].reindex(
                range(self.n_years)
            ),
            max_capacity_increase=energy_source_df["max_capacity_increase"].reindex(
                range(self.n_years)
            ),
            generation_efficiency=float(
                self.storage_type_df.loc[name]["gen_efficiency"]
            ),
            load_efficiency=float(self.storage_type_df.loc[name]["load_efficiency"]),
            energy_type=self.storage_type_df.loc[name]["energy_type"],
            cycle_length=(
                int(self.storage_type_df.loc[name]["cycle_length"])
                if not np.isnan(self.storage_type_df.loc[name]["cycle_length"])
                else None
            ),
            power_to_capacity=float(
                self.storage_type_df.loc[name]["power_to_capacity"]
            ),
            power_utilization=power_utilization,
            energy_loss=energy_loss,
            tags=create_tags_list(df_row[9:]),
        )

    def create(self) -> tuple[tuple[GeneratorType, ...], tuple[StorageType, ...]]:
        energy_source_type_df = self._prepare_energy_source_parameters()
        efficiency_df = self.generators_efficiency.pivot_table(
            index=self.generators_efficiency.index,
            columns="energy_type",
            values="efficiency",
        )
        conv_dict = self._prepare_conversion_rate_dict(self.conversion_rate)
        emission_reduction_dict: dict[str, dict[str, pd.Series]] = (
            self._prepare_generator_emission_reduction(
                self.generators_emission_reduction,
                self.yearly_emission_reduction,
                self.n_years,
            )
        )

        generator_types = tuple(
            self.generators_type.apply(
                self._create_generator_type,
                axis=1,
                args=(
                    energy_source_type_df,
                    efficiency_df,
                    conv_dict,
                    emission_reduction_dict,
                ),
                result_type="reduce",
            )
        )

        storage_types = tuple(
            self.storage_type_df.apply(
                self._create_storage_type,
                axis=1,
                args=(energy_source_type_df,),
                result_type="reduce",
            )
        )

        return generator_types, storage_types

    def _get_power_utilization_boundaries(
        self,
        name: str,
        df_row: pd.Series,
        power_utilization_df: pd.DataFrame,
        type_of_utilization: Literal["power_utilization", "minimal_power_utilization"],
    ) -> pd.Series | float:
        default_value = 1.0 if type_of_utilization == "power_utilization" else 0.0
        utilization = df_row[type_of_utilization]
        if not np.isnan(utilization) and name in power_utilization_df.columns:
            raise EnergySourceTypeParserException(
                f"{type_of_utilization} for {name} must be specified by passing value "
                f"in generator_types.xlsx: Generator Types sheet or Power Utilization sheet, "
                f"but two methods were used at once"
            )
        if not np.isnan(utilization):
            utilization = pd.Series(
                data=[float(utilization)] * self.n_hours,
                index=np.arange(self.n_hours),
            )
        elif name in power_utilization_df.columns:
            utilization = power_utilization_df[name]
        else:
            utilization = pd.Series(
                data=[default_value] * self.n_hours,
                index=np.arange(self.n_hours),
            )
        return utilization

    def _get_generator_efficiency(
        self, name: str, efficiency_df: pd.DataFrame
    ) -> pd.DataFrame:
        if name in self.generators_series_efficiency:
            efficiency = self.generators_series_efficiency[name].set_index("hour_idx")
        else:
            efficiency = (
                efficiency_df.loc[name]
                .dropna()
                .to_frame()
                .T.reset_index(drop=True)
                .reindex(
                    pd.RangeIndex(start=0, stop=self.n_hours, name="hour_idx"),
                    method="ffill",
                )
                if name in efficiency_df.index
                else None
            )
        return efficiency
