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
from pathlib import Path

import numpy as np
import pandas as pd

from pyzefir.structure_creator.data_loader.input_data import InputStructureData
from pyzefir.structure_creator.excel_writer import write_to_excel
from pyzefir.structure_creator.structure_and_initial_state.dataclasses import (
    InitialStateData,
    StructureData,
)
from pyzefir.structure_creator.structure_and_initial_state.preprocess_handlers import (
    GlobalSystemsHandler,
    LocalLbsHandler,
)
from pyzefir.structure_creator.structure_and_initial_state.structure_element_creators import (
    BusStructureCreator,
    CapacityBoundsCreator,
    GeneratorStructureCreator,
    InitStateCreator,
    LbsStructureCreator,
    LineStructureCreator,
    StaticStructureCreator,
)

_logger = logging.getLogger(__name__)


class StructureCreator:
    @staticmethod
    def create_structure_and_initial(
        input_structure: InputStructureData, output_path: Path
    ) -> pd.DataFrame:
        _logger.debug("Creating StructureData and InitialStateData objects ...")
        structure, init, capacity_bounds_df = StructureCreator._create_structure_data(
            input_structure=input_structure
        )
        _logger.debug("Saving structure.xlsx ...")
        write_to_excel(
            data=structure.convert_to_dict_of_dfs(),
            output_path=output_path,
            filename="structure.xlsx",
        )
        _logger.debug("Saving initial_state.xlsx ...")
        write_to_excel(
            data=init.convert_to_dict_of_dfs(),
            output_path=output_path,
            filename="initial_state.xlsx",
        )
        return capacity_bounds_df

    @staticmethod
    def _create_structure_data(
        input_structure: InputStructureData,
    ) -> tuple[StructureData, InitialStateData, pd.DataFrame]:
        _logger.debug("Creating initial combined local and global dataframes  ...")
        local_lbs_config_df, global_subsystem_config_df = (
            StructureCreator._preprocess_input_data(
                lbs_type=input_structure.lbs_type, subsystem=input_structure.subsystem
            )
        )
        structure_data = StructureData()
        initial_state_data = InitialStateData()
        capacity_bounds_dfs_list: list[pd.DataFrame] = []
        _logger.debug(
            "Creating static structure (independent of individual elements)  ..."
        )
        StructureStaticCreator._create_static_structure_df(
            structure_data=structure_data,
            initial_state_data=initial_state_data,
            aggregate_df=input_structure.aggregates,
            transmission_fee_df=input_structure.transmission_fee,
            emission_df=input_structure.emission["EMISSION TYPES"],
            emission_fee_type_df=input_structure.emission["EMISSION FEES"],
            local_et=local_lbs_config_df["energy_type"],
            global_et=global_subsystem_config_df["energy_type"],
            lbs_to_aggr_df=input_structure.configuration["LBS TO AGGREGATE"],
        )
        _logger.debug("Creating local structure data  ...")
        StructureLocalCreator._create_local_structure_data(
            structure_data=structure_data,
            initial_state_data=initial_state_data,
            lbs_to_aggr_df=input_structure.configuration["LBS TO AGGREGATE"],
            local_lbs_config_df=local_lbs_config_df,
            subsystem_to_lbs_df=input_structure.configuration["SUBSYSTEMS TO LBS"],
            lbs_to_subsystem_df=input_structure.configuration.get(
                "LBS TO SUBSYSTEMS", pd.DataFrame()
            ),
            global_subsystem_config_df=global_subsystem_config_df,
            aggregate_df=input_structure.aggregates,
            capacity_bounds_dfs_list=capacity_bounds_dfs_list,
        )
        _logger.debug("Creating global structure data  ...")
        StructureGlobalCreator._create_global_structure_data(
            structure_data=structure_data,
            initial_state_data=initial_state_data,
            global_subsystem_config_df=global_subsystem_config_df,
        )

        capacity_bounds_df = pd.concat(capacity_bounds_dfs_list)

        return structure_data, initial_state_data, capacity_bounds_df

    @staticmethod
    def _preprocess_input_data(
        lbs_type: dict[str, dict[str, pd.DataFrame]], subsystem: dict[str, pd.DataFrame]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return LocalLbsHandler.create_local_lbs_data(
            lbs_type
        ), GlobalSystemsHandler.create_subsystem_dataframe(subsystem)


class StructureGlobalCreator:
    @staticmethod
    def _create_global_structure_data(
        structure_data: StructureData,
        initial_state_data: InitialStateData,
        global_subsystem_config_df: pd.DataFrame,
    ) -> None:
        _logger.debug(
            "Creating structure and initial state dataframes based on global techs ..."
        )
        global_gen_df, global_stor_df = (
            GeneratorStructureCreator.create_generator_storage_df(
                global_subsystem_config_df,
            )
        )
        global_gen_emission_fees = (
            GeneratorStructureCreator.create_generator_emission_fee_df(
                global_subsystem_config_df
            )
        )
        global_bus_df = BusStructureCreator.create_bus_df(
            global_subsystem_config_df
        ).drop_duplicates()
        global_technology_bus = GeneratorStructureCreator.create_technology_to_bus_df(
            global_subsystem_config_df
        )
        init_technology_df = InitStateCreator.create_global_technology_df(
            global_subsystem_config_df,
        )
        global_generator_binding_df = (
            GeneratorStructureCreator.create_generator_binding_df(
                global_subsystem_config_df
            )
        )
        structure_data.Storages.append(global_stor_df)
        structure_data.Generators.append(global_gen_df)
        structure_data.Generator__Emission_Fees.append(global_gen_emission_fees)
        structure_data.Buses.append(global_bus_df)
        structure_data.Technology__Bus.append(global_technology_bus)
        structure_data.Generator_Binding.append(global_generator_binding_df)

        initial_state_data.Technology.append(init_technology_df)


class StructureLocalCreator:
    @staticmethod
    def _create_local_structure_data(
        structure_data: StructureData,
        initial_state_data: InitialStateData,
        lbs_to_aggr_df: pd.DataFrame,
        local_lbs_config_df: pd.DataFrame,
        subsystem_to_lbs_df: pd.DataFrame,
        lbs_to_subsystem_df: pd.DataFrame,
        global_subsystem_config_df: pd.DataFrame,
        aggregate_df: pd.DataFrame,
        capacity_bounds_dfs_list: list[pd.DataFrame],
    ) -> None:
        lbs_to_aggr_df = lbs_to_aggr_df.set_index("aggregate_id")
        aggr_config = {
            index: lbs_to_aggr_df.columns[lbs_to_aggr_df.loc[index].notna()].tolist()
            for index in lbs_to_aggr_df.index
        }
        subsystem_to_lbs_df = subsystem_to_lbs_df.set_index("subsystem_id").eq(1)
        base_fraction_df = StructureLocalCreator._prepare_data_for_base_fraction(
            aggregate_df, lbs_to_aggr_df
        )
        for aggr_name, available_lbs in aggr_config.items():
            _logger.debug(
                "Creating structure and initial state dataframes for aggregate %s ...",
                aggr_name,
            )
            filtered_local_lbs_config_df = StructureLocalCreator._adjust_lbs_config(
                local_lbs_config_df,
                available_lbs,
                aggr_name,
                subsystem_to_lbs_df,
            )
            local_gen_df, local_stor_df = (
                GeneratorStructureCreator.create_generator_storage_df(
                    filtered_local_lbs_config_df, aggr_name
                )
            )
            local_gen_emission_fees = (
                GeneratorStructureCreator.create_generator_emission_fee_df(
                    filtered_local_lbs_config_df
                )
            )
            local_technology_bus = (
                GeneratorStructureCreator.create_technology_to_bus_df(
                    filtered_local_lbs_config_df
                )
            )
            local_bus_df = BusStructureCreator.create_bus_df(
                filtered_local_lbs_config_df
            )
            local_technology_stack_bus_df = (
                BusStructureCreator.create_technologystack_bus_df(
                    filtered_local_lbs_config_df
                )
            )
            local_technology_stack_buses_out_df = (
                BusStructureCreator.create_technologystack_bout_df(
                    filtered_local_lbs_config_df
                )
            )
            local_technology_stack_aggr_df = (
                LbsStructureCreator.create_technologystack_aggr_df(
                    filtered_local_lbs_config_df, aggr_name
                )
            )
            local_lbs_lines_df = LineStructureCreator.create_local_lbs_lines(
                filtered_local_lbs_config_df,
            )
            lines_df = LineStructureCreator.create_lines(
                filtered_local_lbs_config_df,
                global_subsystem_config_df,
                lbs_to_subsystem_df,
                aggr_name,
            )
            init_technology_df = InitStateCreator.create_local_technology_df(
                filtered_local_lbs_config_df, base_fraction_df, aggr_name
            )
            generator_binding_df = (
                GeneratorStructureCreator.create_generator_binding_df(
                    filtered_local_lbs_config_df
                )
            )
            capacity_bounds_df = CapacityBoundsCreator.create_capacity_bounds_df(
                filtered_local_lbs_config_df
            )
            structure_data.Buses.append(local_bus_df)
            structure_data.Generators.append(local_gen_df)
            structure_data.Storages.append(local_stor_df)
            structure_data.Generator__Emission_Fees.append(local_gen_emission_fees)
            structure_data.Technology__Bus.append(local_technology_bus)
            structure_data.TechnologyStack_Buses.append(local_technology_stack_bus_df)
            structure_data.TechnologyStack_Buses_out.append(
                local_technology_stack_buses_out_df
            )
            structure_data.TechnologyStack__Aggregate.append(
                local_technology_stack_aggr_df
            )
            structure_data.Lines.append(local_lbs_lines_df)
            structure_data.Lines.append(lines_df)
            structure_data.Generator_Binding.append(generator_binding_df)

            initial_state_data.Technology.append(init_technology_df)
            capacity_bounds_dfs_list.append(capacity_bounds_df)

    @staticmethod
    def _adjust_lbs_config(
        df: pd.DataFrame,
        available_lbs: list[str],
        aggr_name: str,
        subsystem_df: pd.DataFrame,
    ) -> pd.DataFrame:
        filtered_df = df[df["lbs"].isin(available_lbs)]
        filtered_df = pd.merge(
            filtered_df, subsystem_df.T, left_on="lbs", right_index=True, how="left"
        )
        filtered_df = filtered_df.assign(
            gen_name=(
                aggr_name
                + "__"
                + filtered_df["lbs"]
                + "__"
                + filtered_df["technology_type"]
                + "__"
                + filtered_df.index
            )
        )
        filtered_df = filtered_df.assign(
            bus_name=(
                aggr_name
                + "__"
                + filtered_df["lbs"]
                + "__"
                + filtered_df["energy_type"]
                + "__"
                + filtered_df["bus_id"]
            )
        )
        filtered_df = filtered_df.assign(
            binding_name=(
                aggr_name + "__" + filtered_df["lbs"] + "__" + filtered_df["binding_id"]
            ).where(~filtered_df["binding_id"].isnull(), other=np.nan)
        )
        filtered_df["lbs_type"] = filtered_df.loc[:, "lbs"]
        filtered_df.loc[:, "lbs"] = aggr_name + "__" + filtered_df["lbs"]

        return filtered_df

    @staticmethod
    def _prepare_data_for_base_fraction(
        df_aggr: pd.DataFrame,
        df_fraction: pd.DataFrame,
    ) -> pd.DataFrame:
        n_buildings_df = df_aggr.set_index("aggregate_id")[["n_buildings"]]
        df = pd.concat([df_fraction, n_buildings_df], axis=1)
        return df


class StructureStaticCreator:
    @staticmethod
    def _create_static_structure_df(
        structure_data: StructureData,
        initial_state_data: InitialStateData,
        aggregate_df: pd.DataFrame,
        transmission_fee_df: pd.DataFrame,
        emission_df: pd.DataFrame,
        emission_fee_type_df: pd.DataFrame,
        local_et: pd.Series,
        global_et: pd.Series,
        lbs_to_aggr_df: pd.DataFrame,
    ) -> None:
        aggr_df = StaticStructureCreator.create_aggregate_df(df_data=aggregate_df)
        emission_type_df = StaticStructureCreator.create_emission_type_df(
            df_data=emission_df
        )
        emission_fee_df = StaticStructureCreator.create_emission_fees_emission_type_df(
            df_data=emission_fee_type_df
        )
        et_df = StaticStructureCreator.create_energy_types_df(
            local_et=local_et, global_et=global_et
        )
        technology_stack_df = InitStateCreator.create_technology_stack_df(
            df_data=lbs_to_aggr_df
        )
        structure_data.Aggregates.append(aggr_df)
        structure_data.Emission_Types.append(emission_type_df)
        structure_data.Emission_Fees__Emission_Types.append(emission_fee_df)
        structure_data.Energy_Types.append(et_df)
        structure_data.Transmission_Fees.append(transmission_fee_df)
        initial_state_data.TechnologyStack.append(technology_stack_df)
