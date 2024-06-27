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

import pandas as pd

from pyzefir.structure_creator.structure_and_initial_state.utils import (
    handle_prefix_name,
)


class LocalLbsHandler:
    @staticmethod
    def create_local_lbs_data(
        lbs_type: dict[str, dict[str, pd.DataFrame]]
    ) -> pd.DataFrame:
        dfs: list[pd.DataFrame] = []
        for lbs_name, lbs_data in lbs_type.items():
            df = LocalLbsHandler._create_lbs_dataframe(lbs_name, lbs_data)
            dfs.append(df)
        return pd.concat(dfs)

    @staticmethod
    def _create_lbs_dataframe(
        lbs_name: str, lbs_data: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        bus_df = LocalLbsHandler._create_flat_bus_df(
            lbs_data["TECHNOLOGY TO BUS"], lbs_data["BUSES"]
        )
        capa_df = LocalLbsHandler._create_flat_capacity(
            {key: lbs_data[key] for key in ["BASE CAP", "MIN CAP", "MAX CAP"]}
        )
        technology_df = lbs_data["TECHNOLOGIES"].set_index("technology_id")
        lines_df = (
            LocalLbsHandler._create_flat_lines(lbs_data["LINES"], bus_df)
            if "LINES" in lbs_data
            else pd.DataFrame()
        )
        capacity_bound_df = (
            LocalLbsHandler._create_flat_capacity_bounds(lbs_data["CAPACITY_BOUNDS"])
            if "CAPACITY_BOUNDS" in lbs_data
            else pd.DataFrame()
        )
        if "TAGS" in lbs_data:
            tags_df = lbs_data["TAGS"].set_index("technology_id").add_prefix("TAG_")
            df = pd.concat(
                [technology_df, capa_df, lines_df, tags_df, capacity_bound_df], axis=1
            )
        else:
            df = pd.concat(
                [technology_df, capa_df, lines_df, capacity_bound_df], axis=1
            )
        merged_df = bus_df.merge(df, left_index=True, right_index=True)
        merged_df.insert(0, "lbs", lbs_name)
        return merged_df

    @staticmethod
    def _create_flat_bus_df(
        tech_to_bus_df: pd.DataFrame, bus_df: pd.DataFrame
    ) -> pd.DataFrame:
        tech_df = tech_to_bus_df.melt(
            id_vars=["technology_id"], var_name="bus_id", value_name="value"
        )
        tech_df = tech_df[tech_df["value"] == 1.0]
        tech_df = tech_df.drop(columns="value")
        df = pd.merge(tech_df, bus_df, on="bus_id")
        return df.set_index("technology_id")

    @staticmethod
    def _create_flat_capacity(capa_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        dfs: list[pd.DataFrame] = []
        for name, df in capa_dict.items():
            df = df.set_index("technology_id")
            df = df.add_prefix(handle_prefix_name(name))
            dfs.append(df)
        return pd.concat(dfs, axis=1)

    @staticmethod
    def _create_flat_lines(
        lines_df: pd.DataFrame, bus_df: pd.DataFrame
    ) -> pd.DataFrame:
        index_dict = dict(zip(bus_df["bus_id"], bus_df.index))
        lines_df = lines_df.rename(columns={"energy_type": "line_energy_type"})
        lines_df.index = lines_df["bus_from_id"].map(index_dict)
        return lines_df

    @staticmethod
    def _create_flat_capacity_bounds(cb_df: pd.DataFrame) -> pd.DataFrame:
        cb_df.loc[:, "left_technology_name"] = cb_df.loc[:, "left_tech_id"]
        cb_df = cb_df.set_index("left_tech_id")
        cb_df = cb_df.rename(columns={"right_tech_id": "right_technology_name"})
        return cb_df.fillna(1.0)


class GlobalSystemsHandler:
    @staticmethod
    def create_subsystem_dataframe(subsystems: dict[str, pd.DataFrame]) -> pd.DataFrame:
        connection_df = GlobalSystemsHandler._create_flat_subsystem_connection_df(
            subsystems["TECHNOLOGY TO SUBSYSTEM"], subsystems["SUBSYSTEMS"]
        )
        transmission_fee_df = GlobalSystemsHandler._create_transmission_fees_df(
            subsystems["SUBSYSTEM TRANSMISSION FEES"]
        )
        global_tech_df = subsystems["GLOBAL TECHNOLOGIES"].set_index("global_tech_id")
        merged_df = connection_df.merge(
            global_tech_df, left_index=True, right_index=True
        )
        merged_df = merged_df.merge(
            transmission_fee_df, left_on="subsystem_id", right_index=True
        )
        if "TAGS" in subsystems:
            tags_df = subsystems["TAGS"].set_index("global_tech_id").add_prefix("TAG_")
            merged_df = merged_df.merge(tags_df, left_index=True, right_index=True)
        merged_df = merged_df.reset_index()
        merged_df = merged_df.rename(
            columns={
                "global_tech_id": "gen_name",
                "subsystem_id": "bus_name",
                "binding_id": "binding_name",
            }
        )
        return merged_df.reset_index()

    @staticmethod
    def _create_flat_subsystem_connection_df(
        tech_to_sub_df: pd.DataFrame, subsystem_df: pd.DataFrame
    ) -> pd.DataFrame:
        tech_to_sub_df = tech_to_sub_df.melt(
            id_vars="global_tech_id", var_name="subsystem_id", value_name="value"
        )
        merged_df = tech_to_sub_df.merge(subsystem_df, on="subsystem_id")
        merged_df = merged_df[merged_df["value"] == 1]
        merged_df = merged_df.drop(columns="value")
        return merged_df.set_index("global_tech_id")

    @staticmethod
    def _create_transmission_fees_df(tf_df: pd.DataFrame) -> pd.DataFrame:
        tf_df = tf_df.set_index("aggregate_id").T
        tf_df = tf_df.add_prefix(handle_prefix_name("TF"))
        return tf_df
