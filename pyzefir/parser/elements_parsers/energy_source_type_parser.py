from copy import deepcopy

import numpy as np
import pandas as pd

from pyzefir.model.network_elements import GeneratorType, StorageType
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser
from pyzefir.parser.elements_parsers.utils import create_tags_list


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
        n_years: int,
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
        self.n_years = n_years

    def _prepare_energy_source_parameters(self) -> dict[str, pd.DataFrame]:
        """Prepare common parameters for GeneratorType and StorageType classes (EnergySourceType parameters)."""
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
    ) -> GeneratorType:
        name = df_row["name"]
        energy_source_df = energy_source_type_df[name]
        fuel_row = self.generators_fuel_type[
            self.generators_fuel_type["generator_type"] == name
        ]
        fuel = fuel_row["fuel_name"]
        capacity_factor = fuel_row["capacity_factor_name"]
        efficiency = (
            efficiency_df.loc[name].dropna().to_dict()
            if name in efficiency_df.index
            else None
        )
        ramp = df_row["ramp"] if "ramp" in df_row else np.nan
        energy_types = self.generators_energy_type.loc[name]
        if not np.isnan(df_row["power_utilization"]):
            power_utilization = float(df_row["power_utilization"])
        else:
            power_utilization = 1.0
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
            efficiency=efficiency,
            energy_types=energy_types,
            emission_reduction=self.generators_emission_reduction.loc[name].to_dict(),
            fuel=None if pd.isna(fuel).all() else fuel.iloc[0],
            capacity_factor=None
            if pd.isna(capacity_factor).all()
            else capacity_factor.iloc[0],
            power_utilization=power_utilization,
            ramp=ramp,
            tags=create_tags_list(df_row[5:]),
        )
        if conv_rate := conv_dict.get(name):
            gen_type.conversion_rate = conv_rate
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
            generation_efficiency=self.storage_type_df.loc[name]["gen_efficiency"],
            load_efficiency=self.storage_type_df.loc[name]["load_efficiency"],
            energy_type=self.storage_type_df.loc[name]["energy_type"],
            cycle_length=int(self.storage_type_df.loc[name]["cycle_length"]),
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

        generator_types = tuple(
            self.generators_type.apply(
                self._create_generator_type,
                axis=1,
                args=(energy_source_type_df, efficiency_df, conv_dict),
            )
        )

        storage_types = tuple(
            self.storage_type_df.apply(
                self._create_storage_type, axis=1, args=(energy_source_type_df,)
            )
        )

        return generator_types, storage_types
