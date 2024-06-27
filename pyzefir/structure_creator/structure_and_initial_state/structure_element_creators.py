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

import numpy as np
import pandas as pd

from pyzefir.structure_creator.structure_and_initial_state.utils import (
    join_energy_types,
)


class LineStructureCreator:
    @staticmethod
    def _create_lines_dataframe(
        first_df: pd.DataFrame, second_df: pd.DataFrame, energy_type: str
    ) -> pd.DataFrame:
        return (
            pd.merge(
                first_df.assign(key=1),
                second_df.assign(key=1),
                on="key",
            )
            .drop("key", axis=1)
            .assign(
                name=lambda x: x["bus_name_x"] + " > " + x["bus_name_y"],
                energy_type=energy_type,
                bus_from=lambda x: x["bus_name_x"],
                bus_to=lambda x: x["bus_name_y"],
                transmission_loss=lambda x: (
                    x["transmission_loss"] if "transmission_loss" in x.columns else 0.0
                ),
                max_capacity=np.nan,
                transmission_fee=lambda x: (
                    x["transmission_fee"] if "transmission_fee" in x.columns else np.nan
                ),
            )[
                [
                    "name",
                    "energy_type",
                    "bus_from",
                    "bus_to",
                    "transmission_loss",
                    "max_capacity",
                    "transmission_fee",
                ]
            ]
        )

    @staticmethod
    def create_lines(
        df_data: pd.DataFrame,
        global_subsystem_config: pd.DataFrame,
        lbs_connection_df: pd.DataFrame,
        aggr_name: str,
    ) -> pd.DataFrame:
        dfs: list[pd.DataFrame] = []
        global_subsystem_config = global_subsystem_config.rename(
            columns={f"TF_{aggr_name}": "transmission_fee"}
        )
        subsystems = global_subsystem_config["bus_name"].unique()
        energy_types_dict = (
            global_subsystem_config.groupby("bus_name")["energy_type"]
            .agg(join_energy_types)
            .to_dict()
        )
        for subsystem in subsystems:
            et: str = energy_types_dict[subsystem]
            filtered_df = df_data[
                df_data[subsystem] & (df_data["energy_type"] == et)
            ].drop_duplicates(subset="bus_name")
            global_bus = global_subsystem_config[
                global_subsystem_config["bus_name"] == subsystem
            ].drop_duplicates(subset="bus_name")
            result_df = LineStructureCreator._create_lines_dataframe(
                global_bus, filtered_df, et
            )
            if not lbs_connection_df.empty:
                lbs_to_global_tech_df = lbs_connection_df.set_index("lbs").eq(1)
                lbs_subsystems_lines_df = (
                    LineStructureCreator._create_lbs_to_subsystems_lines_dataframe(
                        lbs_to_global_tech_df[subsystem], global_bus, filtered_df, et
                    )
                )
                dfs.append(lbs_subsystems_lines_df)
            dfs.append(result_df)
        df = pd.concat(dfs)
        return df

    @staticmethod
    def _create_lbs_to_subsystems_lines_dataframe(
        connection_df: pd.Series,
        global_bus: pd.DataFrame,
        filtered_df: pd.DataFrame,
        energy_type: str,
    ) -> pd.DataFrame:
        filtered_df = filtered_df[
            filtered_df["lbs_type"].isin(connection_df.index[connection_df])
        ]
        result_df = LineStructureCreator._create_lines_dataframe(
            filtered_df, global_bus, energy_type=energy_type
        )
        return result_df

    @staticmethod
    def create_local_lbs_lines(
        df_data: pd.DataFrame,
    ) -> pd.DataFrame:
        if (
            "bus_from_id" not in df_data.columns
            or df_data["bus_from_id"].isnull().all()
        ):
            return pd.DataFrame()
        df = df_data.copy()
        local_lines_df = (
            df[["bus_from_id", "bus_to_id", "line_energy_type"]]
            .dropna()
            .drop_duplicates()
        )
        dfs: list[pd.DataFrame] = []
        for _, group_df in df.groupby("lbs"):
            for bus_fr, bus_to, et in local_lines_df.itertuples(index=False):
                bus_fr_df_filtered = (
                    group_df[
                        (group_df["bus_id"] == bus_fr) & (group_df["energy_type"] == et)
                    ]
                    .dropna(subset=["bus_from_id", "bus_to_id", "line_energy_type"])
                    .drop_duplicates(subset=["bus_name"])
                    .rename(
                        columns={
                            "local_transmission_loss": "transmission_loss",
                            "local_transmission_fee": "transmission_fee",
                        }
                    )
                )
                bus_to_df_filtered = group_df[
                    (group_df["bus_id"] == bus_to) & (group_df["energy_type"] == et)
                ].drop_duplicates(subset="bus_name")
                if bus_fr_df_filtered.empty or bus_to_df_filtered.empty:
                    continue
                result_df = LineStructureCreator._create_lines_dataframe(
                    bus_fr_df_filtered, bus_to_df_filtered, et
                )
                dfs.append(result_df)
        df = pd.concat(dfs)
        return df


class LbsStructureCreator:
    @staticmethod
    def create_technologystack_aggr_df(
        df_data: pd.DataFrame,
        aggr_name: str,
    ) -> pd.DataFrame:
        df = df_data.copy()
        df = df[["lbs"]].drop_duplicates()
        df["aggregate"] = aggr_name
        df = df.rename(columns={"lbs": "technology_stack"})
        return df


class CapacityBoundsCreator:
    @staticmethod
    def handle_capacity_bounds_df_structure(df: pd.DataFrame) -> pd.DataFrame:
        df.insert(
            0,
            "name",
            "capacity_bound__" + df["mapped_left_tech"] + "_" + df["mapped_right_tech"],
        )
        df = df.rename(
            columns={
                "mapped_left_tech": "left_technology_name",
                "mapped_right_tech": "right_technology_name",
            }
        )
        return df

    @staticmethod
    def create_capacity_bounds_df(df_data: pd.DataFrame) -> pd.DataFrame:
        if "left_technology_name" not in df_data.columns:
            return pd.DataFrame()
        dfs = []
        for _, df in df_data.groupby("lbs"):
            df = df[~df.index.duplicated(keep="first")]
            df["mapped_left_tech"] = df["left_technology_name"].map(
                lambda x: df.loc[x, "gen_name"] if pd.notna(x) else np.nan
            )
            df["mapped_right_tech"] = df["right_technology_name"].map(
                lambda x: df.loc[x, "gen_name"] if pd.notna(x) else np.nan
            )
            df = (
                df[["mapped_left_tech", "sense", "mapped_right_tech", "left_coeff"]]
                .reset_index(drop=True)
                .dropna()
            )
            if df.empty:
                continue
            df = CapacityBoundsCreator.handle_capacity_bounds_df_structure(df)
            dfs.append(df)

        return pd.concat(dfs) if dfs else pd.DataFrame()


class GeneratorStructureCreator:
    @staticmethod
    def create_generator_storage_df(
        df_data: pd.DataFrame,
        aggr_name: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = df_data.copy()
        tags_df = GeneratorStructureCreator._create_tag_df(df)
        if aggr_name is not None:
            capa_df = GeneratorStructureCreator._handle_capacity(df, aggr_name)
        else:
            capa_df = pd.DataFrame()
        df = df[["gen_name", "technology_type", "technology_class"]]
        df = df.rename(columns={"gen_name": "name"})
        df = pd.concat([df, capa_df], axis=1)

        df_generator = df[df["technology_class"] == "GENERATOR"].drop(
            columns=["technology_class"]
        )
        if "GENERATOR" in tags_df.index:
            df_tags_generator = tags_df.loc["GENERATOR", tags_df.loc["GENERATOR"].any()]
            df_generator = pd.concat([df_generator, df_tags_generator], axis=1)
        df_generator = df_generator.rename(
            columns={"technology_type": "generator_type"}
        )

        df_storage = df[df["technology_class"] == "STORAGE"].drop(
            columns=["technology_class"]
        )
        if "STORAGE" in tags_df.index:
            df_tags_storage = tags_df.loc["STORAGE", tags_df.loc["STORAGE"].any()]
            df_storage = pd.concat([df_storage, df_tags_storage], axis=1)
        df_storage = df_storage.rename(columns={"technology_type": "storage_type"})
        df_generator = df_generator.drop_duplicates(subset="name")
        df_storage = df_storage.drop_duplicates(subset="name")
        return df_generator, df_storage

    @staticmethod
    def create_generator_emission_fee_df(df_data: pd.DataFrame) -> pd.DataFrame:
        df = df_data.copy()
        df = df.dropna(subset=["emission_fee_id"])
        df = df[["gen_name", "emission_fee_id"]]
        df = df.rename(
            columns={"gen_name": "generator", "emission_fee_id": "emission_fee"}
        )
        return df

    @staticmethod
    def create_technology_to_bus_df(df_data: pd.DataFrame) -> pd.DataFrame:
        df = df_data.copy()
        df = df[["gen_name", "technology_class", "bus_name"]]
        df = df.rename(
            columns={
                "gen_name": "technology",
                "technology_class": "type",
                "bus_name": "bus",
            }
        )
        return df

    @staticmethod
    def _handle_capacity(df: pd.DataFrame, aggr_name: str) -> pd.DataFrame:
        filtered_columns = [
            col
            for col in df.columns
            if (col.startswith("MIN") or col.startswith("MAX"))
            and col.endswith(aggr_name)
        ]
        filtered_df = df[filtered_columns]
        for col in filtered_df.columns:
            if col.startswith("MIN"):
                col_name = "min_device_nom_power"
            elif col.startswith("MAX"):
                col_name = "max_device_nom_power"
            else:
                col_name = col
            filtered_df = filtered_df.rename(columns={col: col_name})
        return filtered_df

    @staticmethod
    def _create_tag_df(df: pd.DataFrame) -> pd.DataFrame:
        tags_df = pd.concat([df["technology_class"], df.filter(regex="^TAG_")], axis=1)
        tags_df = tags_df.rename(columns=lambda x: x.replace("TAG_", ""))
        tags_df = tags_df.set_index(["technology_class", df.index])
        return tags_df

    @staticmethod
    def create_generator_binding_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.loc[:, ["gen_name", "binding_name"]].dropna().drop_duplicates()
        df = df.rename(columns={"gen_name": "generator"})
        return df


class BusStructureCreator:
    @staticmethod
    def create_bus_df(df_data: pd.DataFrame) -> pd.DataFrame:
        df = df_data.copy()
        df = df[["bus_name", "energy_type"]]
        df = df.drop_duplicates(subset="bus_name")
        df = df.rename(columns={"bus_name": "name"})
        df["dsr_type"] = np.nan
        return df

    @staticmethod
    def create_technologystack_bus_df(df_data: pd.DataFrame) -> pd.DataFrame:
        df = df_data.copy()
        df = df[["lbs", "bus_name"]]
        df = df.rename(columns={"bus_name": "bus", "lbs": "technology_stack"})
        df = df.drop_duplicates()
        return df

    @staticmethod
    def create_technologystack_bout_df(df_data: pd.DataFrame) -> pd.DataFrame:
        df = df_data.copy()
        df = df[df["bus_type"] == "OUTPUT"].drop(columns=["bus_type"])
        df = df[["lbs", "energy_type", "bus_name"]].drop_duplicates()
        pivot_df = df.pivot(
            index="lbs", columns="energy_type", values="bus_name"
        ).reset_index()
        pivot_df = pivot_df.rename(columns={"lbs": "name"})
        return pivot_df


class InitStateCreator:
    @staticmethod
    def create_global_technology_df(
        df_data: pd.DataFrame, aggr_name: str | None = None
    ) -> pd.DataFrame:
        df = df_data.copy()
        df = df.rename(columns={"gen_name": "technology"})
        if aggr_name is not None:
            df = df.rename(columns={f"BASE_CAP_{aggr_name}": "base_capacity"})
        df = df[["technology", "base_capacity"]]
        return df

    @staticmethod
    def create_local_technology_df(
        df_data: pd.DataFrame, df_base_fraction: pd.DataFrame, aggr_name: str
    ) -> pd.DataFrame:
        df = df_data.copy()
        aggr_frac_row = df_base_fraction.loc[aggr_name]
        df = df.rename(
            columns={
                f"BASE_CAP_{aggr_name}": "base_capacity_row",
                "gen_name": "technology",
            }
        )
        aggr_frac_mapped = df["lbs_type"].map(aggr_frac_row)
        df["base_capacity"] = (
            df["base_capacity_row"] * aggr_frac_mapped * aggr_frac_row["n_buildings"]
        )
        df = df[["technology", "base_capacity"]]
        return df

    @staticmethod
    def create_technology_stack_df(df_data: pd.DataFrame) -> pd.DataFrame:
        df = df_data.copy()
        df = pd.melt(
            df,
            id_vars=["aggregate_id"],
            var_name="technology_stack",
            value_name="base_fraction",
        )
        df = df.dropna(subset=["base_fraction"])
        df["technology_stack"] = df["aggregate_id"] + "__" + df["technology_stack"]
        df = df.rename(columns={"aggregate_id": "aggregate"})
        df = df[["technology_stack", "aggregate", "base_fraction"]]
        return df


class StaticStructureCreator:
    @staticmethod
    def create_aggregate_df(df_data: pd.DataFrame) -> pd.DataFrame:
        df = df_data.copy()
        df = df.rename(
            columns={
                "aggregate_id": "name",
                "usable_area_per_building": "average_area",
                "n_buildings": "n_consumers_base",
                "demand_profile_type": "demand_type",
            }
        )
        df = df[["name", "demand_type", "n_consumers_base", "average_area"]]
        return df

    @staticmethod
    def create_emission_type_df(df_data: pd.DataFrame) -> pd.DataFrame:
        df = df_data.copy()
        df = df.rename(
            columns={
                "emission_type_id": "name",
                "base_emission": "base_total_emission",
            }
        )
        return df

    @staticmethod
    def create_emission_fees_emission_type_df(df_data: pd.DataFrame) -> pd.DataFrame:
        df = df_data.copy()
        df = df.rename(
            columns={
                "emission_type_id": "emission_type",
                "emission_fee_id": "emission_fee",
            }
        )
        df = df[["emission_type", "emission_fee"]]
        return df

    @staticmethod
    def create_energy_types_df(
        local_et: pd.Series, global_et: pd.Series
    ) -> pd.DataFrame:
        df = pd.concat([local_et, global_et]).drop_duplicates().to_frame()
        df = df.rename(columns={"energy_type": "name"})
        return df
