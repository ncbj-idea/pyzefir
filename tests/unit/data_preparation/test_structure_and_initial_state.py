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

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pyzefir.structure_creator.structure_and_initial_state.constants_enums import (
    InputFileFieldName,
)
from pyzefir.structure_creator.structure_and_initial_state.main import (
    create_global_data,
    create_line_connections,
    get_element_name,
    get_lbs_tech_cap_parameters,
    get_lbs_tech_config,
    local_buses_process_energy_types,
    local_buses_process_technology,
)


@pytest.mark.parametrize(
    "aggregate, lbs, energy_type, unit_type, expected_result",
    [
        pytest.param("A", "B", "C", "D", "A__B__C__D", id="all_param"),
        pytest.param("X", None, "Y", None, "X__Y", id="2_param_none"),
        pytest.param("P", "Q", None, "R", "P__Q__R", id="1_param_none"),
        pytest.param("M", None, None, None, "M", id="only_aggregate"),
    ],
)
def test_get_element_name(
    aggregate: str,
    lbs: str | None,
    energy_type: str | None,
    unit_type: str | None,
    expected_result: str,
) -> None:
    result = get_element_name(aggregate, lbs, energy_type, unit_type)
    assert result == expected_result


@pytest.mark.parametrize(
    "aggregate, tech_type, cap_range, expected_result",
    [
        pytest.param(
            "SINGLE_FAMILY_AB",
            "EE",
            [{"SINGLE_FAMILY": pd.DataFrame({"technology_type": ["EE"], "AB": [1.5]})}],
            (1.5,),
            id="case_dict_1_element",
        ),
        pytest.param(
            "MULTI_FAMILY_C",
            "HEAT",
            [
                {
                    "MULTI_FAMILY": pd.DataFrame(
                        {"technology_type": ["HEAT"], "C": [2.0]}
                    ),
                    "SINGLE_FAMILY": pd.DataFrame(
                        {"technology_type": ["HEAT"], "AB": [1.5]}
                    ),
                }
            ],
            (2.0,),
            id="case_dict_2_elements",
        ),
        pytest.param(
            "SINGLE_FAMILY_AB",
            "EE",
            [{"MULTI_FAMILY": pd.DataFrame({"technology_type": ["EE"], "AB": [3.0]})}],
            (),
            id="case_dict_not_match_keys",
        ),
    ],
)
def test_get_lbs_tech_cap_parameters(
    aggregate: str,
    tech_type: str,
    cap_range: list[dict[str, pd.DataFrame]],
    expected_result: tuple[float, ...],
) -> None:
    result = get_lbs_tech_cap_parameters(aggregate, tech_type, cap_range)
    assert result == expected_result


@pytest.mark.parametrize(
    "tech_config_dict, aggregate_name, lbs_type, expected_result",
    [
        pytest.param(
            {"SINGLE_FAMILY": {"boiler_coal_old_lkt": 1}},
            "SINGLE_FAMILY_AB",
            "boiler_coal_old_lkt",
            {"boiler_coal_old_lkt": 1},
            id="found_aggr_name_in_tech_config",
        ),
        pytest.param(
            {"SINGLE_FAMILY": {"MULTIFAMILY_C": 1}},
            "MULTIFAMILY_C",
            "boiler_coal_old_lkt",
            None,
            id="not_found_aggr_name_in_tech_config",
        ),
    ],
)
def test_get_lbs_tech_config(
    tech_config_dict: dict[str, dict[str, int]],
    aggregate_name: str,
    lbs_type: str,
    expected_result: dict[str, int] | None,
) -> None:
    result = get_lbs_tech_config(tech_config_dict, aggregate_name, lbs_type)
    assert result == expected_result


def test_get_lbs_tech_config_error() -> None:
    tech_config_dict = {
        "SINGLE_FAMILY_A": {"boiler_coal_old_lkt": 3},
        "SINGLE_FAMILY_A_B": {"boiler_coal_old_lkt": 2},
    }
    aggregate_name = "SINGLE_FAMILY_A_B"
    lbs_type = "boiler_coal_old_lkt"
    with pytest.raises(KeyError) as error_msg:
        get_lbs_tech_config(tech_config_dict, aggregate_name, lbs_type)
    assert (
        str(error_msg.value)
        == f"'{aggregate_name} type is ambigius in boiler_coal_old_lkt configuration file'"
    )


@pytest.mark.parametrize(
    "global_tech_df, subsystem_config, subsystem_configuration, expected_df, expected_fee_map, expected_tags_mapping",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "SUBSYSTEM_TECHNOLOGY_TYPE_2": {
                        "type": "DISTRICT_HEAT_COAL_BIG",
                        "base_cap": 6000,
                        "class": "GENERATOR",
                        "tags": ["DH"],
                        "emission_fees": ["ETS1"],
                    },
                    "SUBSYSTEM_TECHNOLOGY_TYPE_1": {
                        "type": "KSE",
                        "base_cap": 1200,
                        "class": "GENERATOR",
                        "tags": [],
                        "emission_fees": [],
                    },
                }
            ),
            {
                "boiler_coal_old_lkt": {
                    "subsystem_name": "subsystem_name_1",
                    "energy_type": "ELECTRICITY_END",
                    "technologies": ["SUBSYSTEM_TECHNOLOGY_TYPE_1"],
                    "transmission_loss": 0.04,
                },
                "heat_pump_pv_lkt": {
                    "subsystem_name": "subsystem_name_2",
                    "energy_type": "HEAT_END",
                    "technologies": ["SUBSYSTEM_TECHNOLOGY_TYPE_2"],
                    "transmission_loss": 0.13,
                },
            },
            pd.DataFrame(
                {
                    "SUBSYSTEM": [
                        "boiler_coal_old_lkt",
                        "boiler_coal_new_lkt",
                        "boiler_gas_lkt",
                        "heat_pump_pv_lkt",
                        "heating_system_lkt",
                        "electric_heating_lkt",
                    ],
                    "subsystem_name_1": [1, 1, 1, 1, 1, 1],
                    "subsystem_name_2": [0, 0, 0, 0, 1, 0],
                }
            ),
            pd.DataFrame(
                {
                    "energy_type": ["ELECTRICITY_END", "HEAT_END"],
                    "bus": [
                        "subsystem_name_1",
                        "subsystem_name_2",
                    ],
                    "technology": [
                        "SUBSYSTEM_TECHNOLOGY_TYPE_1",
                        "SUBSYSTEM_TECHNOLOGY_TYPE_2",
                    ],
                    "base_cap": [1200, 6000],
                    "technology_type": ["KSE", "DISTRICT_HEAT_COAL_BIG"],
                    "unit_class": ["GENERATOR", "GENERATOR"],
                    "transmission_loss": [0.04, 0.13],
                    "dsr_type": [None, None],
                }
            ),
            {
                "SUBSYSTEM_TECHNOLOGY_TYPE_1": [],
                "SUBSYSTEM_TECHNOLOGY_TYPE_2": ["ETS1"],
            },
            {"DH": {"SUBSYSTEM_TECHNOLOGY_TYPE_2"}},
            id="simple_case_with_2_subsystems",
        ),
    ],
)
def test_create_global_data(
    global_tech_df: pd.DataFrame,
    subsystem_config: dict[str, dict[InputFileFieldName, list[str] | float]],
    subsystem_configuration: pd.DataFrame,
    expected_df: pd.DataFrame,
    expected_fee_map: dict[str, list[str]],
    expected_tags_mapping: dict[str, set[str]],
) -> None:
    df, unit_emission_fee, tech_tags = create_global_data(
        global_tech_df, subsystem_config, subsystem_configuration
    )
    assert unit_emission_fee == expected_fee_map
    assert tech_tags == expected_tags_mapping
    assert_frame_equal(df, expected_df)


@pytest.mark.parametrize(
    "energy_types, lbs_type, energy_to_bus, initial_buses_list, expected_buses_list",
    [
        pytest.param(
            ["Energy1", "Energy2"],
            "Type1",
            {"Energy1": "Bus1", "Energy2": "Bus2"},
            [],
            [["Type1", "Bus1", "Energy1"], ["Type1", "Bus2", "Energy2"]],
            id="empty_initial_bus",
        ),
        pytest.param(
            ["Energy1"],
            "Type1",
            {"Energy1": "Bus1"},
            [["Type_init", "Bus_init", "Energy_init"]],
            [["Type_init", "Bus_init", "Energy_init"], ["Type1", "Bus1", "Energy1"]],
            id="not_empty_initial_bus",
        ),
    ],
)
def test_local_buses_process_energy_types(
    energy_types: list[str],
    lbs_type: str,
    energy_to_bus: dict[str, str],
    initial_buses_list: list[list[str]],
    expected_buses_list: list[list[str]],
) -> None:
    local_buses_process_energy_types(
        energy_types, lbs_type, energy_to_bus, initial_buses_list
    )
    assert initial_buses_list == expected_buses_list


def test_local_buses_process_technology() -> None:
    structure_data = MagicMock()
    structure_data.cap_base = {
        "SINGLE_FAMILY": pd.DataFrame(
            {
                "technology_type": ["COAL_BOILER_OLD"],
                "AB": [0.010],
                "C": [0.010],
                "D": [0.015],
                "EF": [0.015],
            }
        )
    }
    structure_data.cap_max = {
        "SINGLE_FAMILY": pd.DataFrame(
            {
                "technology_type": ["COAL_BOILER_OLD"],
                "AB": [0.010],
                "C": [0.010],
                "D": [0.015],
                "EF": [0.015],
            }
        )
    }
    structure_data.cap_min = {
        "SINGLE_FAMILY": pd.DataFrame(
            {
                "technology_type": ["COAL_BOILER_OLD"],
                "AB": [0.010],
                "C": [0.010],
                "D": [0.015],
                "EF": [0.015],
            }
        )
    }
    structure_data.configuration = {
        "lbs_type": pd.DataFrame(
            {
                "AGGREGATE": ["SINGLE_FAMILY_AB"],
                "electric_heating_lkt": [0.027981],
            }
        )
    }
    structure_data.aggregate_types = {
        "SINGLE_FAMILY_AB": {
            "n_consumers_base": 4302,
            "average_area": 180,
            "demand_type": "SINGLE_FAMILY",
        }
    }

    sample_input_data: dict = {
        "aggregate": "SINGLE_FAMILY_AB",
        "lbs": "boiler_coal_old_lkt",
        "lbs_type": "electric_heating_lkt",
        "tech": "COAL_BOILER_OLD",
        "tech_config": {
            "TECH_CLASS": "GENERATOR",
            "SINGLE_FAMILY": {},
            "MULTI_FAMILY": {},
            "SHOP_SERVICE": {},
            "OFFICE": {},
            "OTHER": {},
        },
        "tech_energy_mapping": {
            "COAL_BOILER_OLD": {"HEAT_USABLE"},
            "EE_SUBSTATION": {"ELECTRICITY_USABLE", "ELECTRICITY_END"},
            "AIR_CONDITION": {"ELECTRICITY_USABLE", "COLD"},
        },
        "energy_to_bus": {
            "HEAT_USABLE": "SINGLE_FAMILY_AB__boiler_coal_old_lkt__HEAT_USABLE",
            "ELECTRICITY_USABLE": "SINGLE_FAMILY_AB__boiler_coal_old_lkt__ELECTRICITY_USABLE",
            "COLD": "SINGLE_FAMILY_AB__boiler_coal_old_lkt__COLD",
            "HEAT_END": "SINGLE_FAMILY_AB__boiler_coal_old_lkt__HEAT_END",
            "ELECTRICITY_END": "SINGLE_FAMILY_AB__boiler_coal_old_lkt__ELECTRICITY_END",
        },
        "structure_data": structure_data,
        "results": [],
        "unit_emission_fees_map": {},
        "unit_tech_tags": {},
        "dsr_types": {},
    }

    local_buses_process_technology(**sample_input_data)

    assert sample_input_data["results"] == [
        [
            "SINGLE_FAMILY_AB__electric_heating_lkt__COAL_BOILER_OLD",
            "COAL_BOILER_OLD",
            0.01,
            0.01,
            "GENERATOR",
            "SINGLE_FAMILY_AB__boiler_coal_old_lkt__HEAT_USABLE",
            "boiler_coal_old_lkt",
            "electric_heating_lkt",
            "SINGLE_FAMILY_AB",
            "HEAT_USABLE",
            1.20374262,
            None,
        ]
    ]
    assert sample_input_data["unit_emission_fees_map"] == {}
    assert sample_input_data["unit_tech_tags"] == {}


def test_create_line_connections() -> None:
    subsystems_config_df = pd.DataFrame(
        {"SUBSYSTEM": ["Sub1", "Sub2"], "LbsType1": [1, 0], "LbsType2": [1, 0]}
    )

    global_buses_df = pd.DataFrame(
        {
            "bus": ["Sub1", "Sub2", "Bus2", "Bus3"],
            "energy_type": ["Electricity", "Heat", "Electricity", "Electricity"],
            "transmission_loss": [0.1, 0.2, 0.3, 0.4],
        }
    )

    local_buses_in_df = pd.DataFrame(
        {
            "bus": ["Bus1", "Bus2", "Bus3"],
            "lbs_type": ["LbsType1", "LbsType2", "LbsType3"],
            "energy_type": ["Electricity"] * 3,
        }
    )

    transmission_fee_df = pd.DataFrame(
        {"AGGREGATE": ["Agg1", "Agg2"], "Sub1": [0.5, 0.6], "Sub2": [0.7, 0.8]}
    )

    local_buses_df = pd.DataFrame(
        {"bus": ["Bus1", "Bus2", "Bus3"], "aggregate": ["Agg1", "Agg1", "Agg2"]}
    )

    expected_output = pd.DataFrame(
        {
            "name": ["Sub1 -> Bus1", "Sub1 -> Bus2"],
            "energy_type": ["Electricity", "Electricity"],
            "bus_from": ["Sub1", "Sub1"],
            "bus_to": ["Bus1", "Bus2"],
            "transmission_loss": [0.1, 0.1],
            "max_capacity": [np.nan, np.nan],
            "transmission_fee": [0.5, 0.5],
        }
    )
    result_df = create_line_connections(
        subsystems_config_df,
        global_buses_df,
        local_buses_in_df,
        transmission_fee_df,
        local_buses_df,
    )
    assert_frame_equal(result_df, expected_output)
