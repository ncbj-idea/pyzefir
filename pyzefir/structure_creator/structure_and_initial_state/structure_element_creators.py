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
    """
    This class provides methods to create DataFrames representing transmission lines between
    subsystems and local bus systems (LBS).

    It includes utility methods for merging subsystem data with energy type information to
    construct transmission line data, including attributes such as transmission loss, capacity,
    and fees. The class also handles the creation of lines between LBS and global subsystems.
    """

    @staticmethod
    def _create_lines_dataframe(
        first_df: pd.DataFrame, second_df: pd.DataFrame, energy_type: str
    ) -> pd.DataFrame:
        """
        Create a DataFrame representing transmission lines by merging two input DataFrames
        and adding attributes related to energy transmission.

        Args:
            - first_df (pd.DataFrame): dataframe containing the source buses
            - second_df (pd.DataFrame): dataframe containing the destination buses
            - energy_type (str): type of energy for the lines

        Returns:
            - pd.DataFrame: merged line dataframe
        """
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
        """
        Creates a DataFrame of transmission lines by filtering and merging subsystem data
        based on energy types and bus connections.

        Args:
            - df_data (pd.DataFrame): dataframe containing subsystem and energy type information
            - global_subsystem_config (pd.DataFrame): configuration for global subsystems
            - lbs_connection_df (pd.DataFrame): connection data for local bus subsystems
            - aggr_name (str): aggregate name

        Returns:
            - pd.DataFrame: dataframe containing transmission lines
        """
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
        """
        Creates a dataframe of transmission lines connecting local bus subsystems to
        global subsystems.

        Args:
            - connection_df (pd.Series): connection status between LBS and subsystems
            - global_bus (pd.DataFrame): dataframe containing global subsystem bus information
            - filtered_df (pd.DataFrame): filtered DataFrame with LBS subsystem data
            - energy_type (str): the energy type for the transmission lines

        Returns:
            - pd.DataFrame: lbs to subsystem lines dataframe
        """
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
        """
        Creates a dataframe representing local LBS transmission lines by filtering and
        merging data based on bus ids and energy types.

        Args:
            - df_data (pd.DataFrame): dataframe containing local bus subsystem data

        Returns:
            - pd.DataFrame: dataframe with local transmission lines
        """
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
    """
    This class is responsible for creating and managing LBS (Local Bus Subsystem) structure data.
    It provides functionality to aggregate technology stacks by assigning an aggregate name to unique LBS entries.
    The result is a structured DataFrame that can be used for further processing in energy modeling
    and optimization workflows.
    """

    @staticmethod
    def create_technologystack_aggr_df(
        df_data: pd.DataFrame,
        aggr_name: str,
    ) -> pd.DataFrame:
        """
        Creates a dataframe that aggregates technology stacks by LBS and assigns an aggregate name.

        Args:
            - df_data (pd.DataFrame): dataframe containing LBS data
            - aggr_name (str): aggregate name

        Returns:
            - pd.DataFrame: A dataframe with unique technology stacks
        """
        df = df_data.copy()
        df = df[["lbs"]].drop_duplicates()
        df["aggregate"] = aggr_name
        df = df.rename(columns={"lbs": "technology_stack"})
        return df


class CapacityBoundsCreator:
    """
    A class to create and structure capacity bounds data for energy technologies.

    This class handles the mapping of technology names and the generation of DataFrames that define
    capacity bounds between different energy technologies. It processes LBS (Local Bus Subsystem) data,
    ensuring that it conforms to expected structures by filtering and renaming columns.
    """

    @staticmethod
    def handle_capacity_bounds_df_structure(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjusts the structure of the DataFrame for capacity bounds.

        This method inserts a column representing the name of the capacity bound based on
        technology mappings and renames other key columns to clarify the relationships
        between the left and right technologies.

        Args:
            - df (pd.DataFrame): The input DataFrame containing technology names and related data.

        Returns:
            - pd.DataFrame: A DataFrame with updated column names and structure for capacity bounds.
        """
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
    def create_capacity_bounds_df(
        df_data: pd.DataFrame, aggr_name: str
    ) -> pd.DataFrame:
        """
        Creates a DataFrame representing capacity bounds by mapping technology names.

        This method processes the input LBS data by grouping it according to subsystems,
        mapping the left and right technology names to their respective columns, and
        filtering the data based on the provided aggregate name. It handles missing or
        non-existent columns gracefully by returning an empty DataFrame when required.

        Args:
            - df_data (pd.DataFrame): The input data containing technology names and LBS information.
            - aggr_name (str): The aggregate name used to filter the data for relevant subsystems.

        Returns:
            - pd.DataFrame: A structured DataFrame with mapped technology names and capacity bounds,
            or an empty DataFrame if required conditions are not met.
        """
        allowed_in_all_aggregates_value = "all"
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
                df[
                    [
                        "mapped_left_tech",
                        "sense",
                        "mapped_right_tech",
                        "left_coeff",
                        "aggr",
                    ]
                ]
                .reset_index(drop=True)
                .dropna()
            )
            df = df[
                (df["aggr"] == aggr_name)
                | (df["aggr"].str.lower() == allowed_in_all_aggregates_value)
            ].drop(columns=["aggr"])
            if df.empty:
                continue
            df = CapacityBoundsCreator.handle_capacity_bounds_df_structure(df)
            dfs.append(df)

        return pd.concat(dfs) if dfs else pd.DataFrame()


class GeneratorStructureCreator:
    """
    A class for creating structured DataFrames for generators and storage technologies.

    This class processes input data related to energy generators and storage technologies,
    organizing the data into separate DataFrames for each category. It also handles the
    generation of additional data such as emission fees and capacity bounds when applicable.
    """

    @staticmethod
    def create_generator_storage_df(
        df_data: pd.DataFrame,
        aggr_name: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates separate dataframes for generators and storage technologies from the input data.

        Args:
            - df_data (pd.DataFrame): dataframe containing generator and storage technology data.
            - aggr_name (str | None, optional): Name for aggregation; if provided, capacity
              handling is applied. Defaults to None.

        Returns:
            - tuple[pd.DataFrame, pd.DataFrame]: tuple of generator and storage data.
        """
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
        """
        Creates a dataframe of generator emission fees by filtering relevant columns and
        renaming them.

        Args:
            - df_data (pd.DataFrame): DataFrame containing generator data with emission fees.

        Returns:
            - pd.DataFrame: a dataframe with generators and corresponding emission fees.
        """
        df = df_data.copy()
        df = df.dropna(subset=["emission_fee_id"])
        df = df[["gen_name", "emission_fee_id"]]
        df = df.rename(
            columns={"gen_name": "generator", "emission_fee_id": "emission_fee"}
        )
        return df

    @staticmethod
    def create_technology_to_bus_df(df_data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a dataframe mapping technologies to their associated buses by filtering
        and renaming relevant columns.

        Args:
            - df_data (pd.DataFrame): dataframe containing generator and bus information.

        Returns:
            - pd.DataFrame: a dataframe connecting technology to bus.
        """
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
        """
        Filters and renames columns related to device capacity based on aggregation name.

        Args:
            - df (pd.DataFrame): dataframe containing device capacity information.
            - aggr_name (str): aggregate name.

        Returns:
            - pd.DataFrame: dataframe with renamed columns for minimum and maximum device
              nominal power based on the specified aggregation.
        """
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
        """
        Creates a dataframe of tags by combining technology classes with tag columns.

        Args:
            - df (pd.DataFrame): dataframe containing technology class and tag data.

        Returns:
            - pd.DataFrame: dataframe with technology classes as index and tag columns.
        """
        tags_df = pd.concat([df["technology_class"], df.filter(regex="^TAG_")], axis=1)
        tags_df = tags_df.rename(columns=lambda x: x.replace("TAG_", ""))
        tags_df = tags_df.set_index(["technology_class", df.index])
        return tags_df

    @staticmethod
    def create_generator_binding_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a DataFrame mapping generators to their binding names by filtering and
        renaming relevant columns.

        Args:
            - df (pd.DataFrame): dataframe containing generator and binding information.

        Returns:
            - pd.DataFrame: a dataframe of generator bindings.
        """
        df = df.loc[:, ["gen_name", "binding_name"]].dropna().drop_duplicates()
        df = df.rename(columns={"gen_name": "generator"})
        return df


class BusStructureCreator:
    """
    A class for creating structured DataFrames related to energy buses and their technology stacks.

    This class provides methods to filter, rename, and organize data related to buses and their
    associated technology stacks, including generating unique bus lists, mapping technology stacks
    to buses, and creating pivoted structures for output buses.
    """

    @staticmethod
    def create_bus_df(df_data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a dataframe representing buses by filtering and renaming relevant columns.

        Args:
            - df_data (pd.DataFrame): dataframe containing bus names and energy types.

        Returns:
            - pd.DataFrame: a dataframe with unique buses.
        """
        df = df_data.copy()
        df = df[["bus_name", "energy_type"]]
        df = df.drop_duplicates(subset="bus_name")
        df = df.rename(columns={"bus_name": "name"})
        df["dsr_type"] = np.nan
        return df

    @staticmethod
    def create_technologystack_bus_df(df_data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a dataframe mapping technology stacks to their associated buses by
        filtering and renaming relevant columns.

        Args:
            - df_data (pd.DataFrame): dataframe containing technology stack (LBS) and bus data.

        Returns:
            - pd.DataFrame: A dataframe with unique mappings of technology_stack and bus.
        """
        df = df_data.copy()
        df = df[["lbs", "bus_name"]]
        df = df.rename(columns={"bus_name": "bus", "lbs": "technology_stack"})
        df = df.drop_duplicates()
        return df

    @staticmethod
    def create_technologystack_bout_df(df_data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a pivot dataframe mapping technology stacks to output buses by filtering
        for output bus types and pivoting on energy types.

        Args:
            - df_data (pd.DataFrame): dataframe containing bus data with technology stacks.

        Returns:
            - pd.DataFrame: a pivoted dataframe of technology stacks to bus.
        """
        df = df_data.copy()
        df = df[df["bus_type"] == "OUTPUT"].drop(columns=["bus_type"])
        df = df[["lbs", "energy_type", "bus_name"]].drop_duplicates()
        pivot_df = df.pivot(
            index="lbs", columns="energy_type", values="bus_name"
        ).reset_index()
        pivot_df = pivot_df.rename(columns={"lbs": "name"})
        return pivot_df


class InitStateCreator:
    """
    A class for creating structured DataFrames related to global and local technologies and their capacities.

    This class provides methods to process and transform data for global technologies, local technologies
    with base capacities, and technology stacks. The methods allow renaming, filtering, and calculating
    capacities based on aggregation names and fractions.
    """

    @staticmethod
    def create_global_technology_df(
        df_data: pd.DataFrame, aggr_name: str | None = None
    ) -> pd.DataFrame:
        """
        Creates a dataframe of global technologies by renaming columns and filtering relevant data.

        Args:
            - df_data (pd.DataFrame): dataframe containing generator data.
            - aggr_name (str | None, optional): Name for aggregation; if provided, the
              corresponding base capacity column is renamed. Defaults to None.

        Returns:
            - pd.DataFrame: A DataFrame with 'technology' and 'base_capacity' columns.
        """
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
        """
        Creates a DataFrame of local technologies by calculating base capacity using
        aggregation fractions.

        Args:
            - df_data (pd.DataFrame): dataframe containing technology data with base capacities.
            - df_base_fraction (pd.DataFrame): dataframe containing base fractions for aggregations.
            - aggr_name (str): Name of the aggregation used to map base capacity and fractions.

        Returns:
            - pd.DataFrame: a dataframe with of local technologies.
        """
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
        """
        Transforms a dataframe into a long format representing technology stacks and their
        base fractions.

        Args:
            - df_data (pd.DataFrame): dataframe containing technology stack data with aggregate ids.

        Returns:
            - pd.DataFrame: a dataframe of technology stacks.
        """
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
    """
    A class for creating structured DataFrames related to aggregates, emission types,
    emission fees, and energy types.

    This class provides methods for transforming input data into well-defined DataFrames,
    which are essential for further processing in energy-related applications. The methods
    focus on renaming columns, filtering relevant attributes, and consolidating energy type data.
    """

    @staticmethod
    def create_aggregate_df(df_data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a dataframe for aggregates by renaming relevant columns and selecting specific
        attributes.

        Args:
            - df_data (pd.DataFrame): dataframe containing aggregate data.

        Returns:
            - pd.DataFrame: a dataframe of aggregates.
        """
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
        """
        Creates a dataframe for emission types by renaming specific columns.

        Args:
            - df_data (pd.DataFrame): dataframe containing emission type data.

        Returns:
            - pd.DataFrame: a dataframe of emission types.
        """
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
        """
        Creates a DataFrame for emission fees linked to emission types by renaming and
        selecting relevant columns.

        Args:
            - df_data (pd.DataFrame): dataframe containing emission fee and type data.

        Returns:
            - pd.DataFrame: a dataframe with emission type and emission fee columns.
        """
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
        """
        Combines local and global energy types into a single dataframe, removing duplicates.

        Args:
            - local_et (pd.Series): series containing local energy types.
            - global_et (pd.Series): series containing global energy types.

        Returns:
            - pd.DataFrame: a dataframe with unique energy types.
        """
        df = pd.concat([local_et, global_et]).drop_duplicates().to_frame()
        df = df.rename(columns={"energy_type": "name"})
        return df
