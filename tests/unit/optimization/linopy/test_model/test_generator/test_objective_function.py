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

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.network import Network
from tests.unit.optimization.linopy.constants import N_HOURS, N_YEARS
from tests.unit.optimization.linopy.names import CO2, EE, GRID, HEAT, HS, PM10
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opf_config,
    run_opt_engine,
    set_network_elements_parameters,
)
from tests.utils import get_resources


@pytest.mark.parametrize(
    (
        "opt_config_params",
        "fuels_params",
        "generator_type_params",
        "generator_params",
        "demand_params",
        "transmission_fee_params",
        "line_params",
        "aggr_params",
        "emission_fee_params",
        "expected_obj",
        "ens",
    ),
    [
        pytest.param(
            {"hour_sample": np.arange(100), "year_sample": np.arange(3)},
            {
                "biomass": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
                "coal": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {
                "heat_plant_biomass": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
                "pp_coal": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {},
            {},
            {},
            {"grid->local_ee_bus": {"transmission_fee": None}},
            {},
            {},
            0.0,
            False,
            id="zero_cost_parameters",
        ),
        pytest.param(
            {"hour_sample": np.arange(100), "year_sample": np.arange(3)},
            {
                "biomass": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
                "coal": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {
                "heat_plant_biomass": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
                "pp_coal": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {},
            {},
            {},
            {"grid->local_ee_bus": {"transmission_fee": None}},
            {},
            {},
            0.0,
            True,
            id="ens_zero_cost_parameters",
        ),
        pytest.param(
            {"hour_sample": np.arange(3), "year_sample": np.arange(5)},
            {
                "biomass": {"cost": pd.Series(np.arange(5)), "energy_per_unit": 0.5},
                "coal": {"cost": pd.Series(np.ones(5)), "energy_per_unit": 3.0},
            },
            {
                "heat_plant_biomass": {
                    "efficiency": {HEAT: 0.5, EE: 0.0},
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
                "pp_coal": {
                    "efficiency": {HEAT: 0.0, EE: 0.2},
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {
                f"pp_coal_{GRID}": {"unit_base_cap": 20},
                f"biomass_heat_plant_{HS}": {"unit_base_cap": 10},
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series(np.ones(3)),
                        EE: pd.Series([0.5, 1.0, 0.0]),
                    }
                }
            },
            {},
            {"grid->local_ee_bus": {"transmission_fee": None}},
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.ones(N_YEARS)),
                        EE: pd.Series(np.ones(N_YEARS)),
                    }
                }
            },
            {},
            (120.0 + 12.5) * 8760 / 3,  # multiply by hourly scale
            False,
            id="constant_fuel_costs_constant_yearly_energy_usage",
        ),
        pytest.param(
            {"hour_sample": np.arange(3), "year_sample": np.arange(5)},
            {
                "biomass": {"cost": pd.Series(np.arange(5)), "energy_per_unit": 0.5},
                "coal": {"cost": pd.Series(np.ones(5)), "energy_per_unit": 3.0},
            },
            {
                "heat_plant_biomass": {
                    "efficiency": {HEAT: 0.5, EE: 0.0},
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
                "pp_coal": {
                    "efficiency": {HEAT: 0.0, EE: 0.2},
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {
                f"pp_coal_{GRID}": {"unit_base_cap": 20},
                f"biomass_heat_plant_{HS}": {"unit_base_cap": 10},
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series(np.ones(3)),
                        EE: pd.Series([0.5, 1.0, 0.0]),
                    }
                }
            },
            {},
            {"grid->local_ee_bus": {"transmission_fee": None}},
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.ones(N_YEARS)),
                        EE: pd.Series(np.ones(N_YEARS)),
                    }
                }
            },
            {},
            (120.0 + 12.5) * 8760 / 3,  # multiply by hourly scale
            True,
            id="ens_constant_fuel_costs_constant_yearly_energy_usage",
        ),
        pytest.param(
            {"hour_sample": np.arange(5), "year_sample": np.arange(3)},
            {
                "biomass": {"cost": pd.Series(np.arange(5)), "energy_per_unit": 1.0},
                "coal": {"cost": pd.Series(np.ones(5)), "energy_per_unit": 1.0},
            },
            {
                "heat_plant_biomass": {
                    "efficiency": {HEAT: 0.5, EE: 0.0},
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
                "pp_coal": {
                    "efficiency": {HEAT: 0.0, EE: 1.0},
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {
                f"pp_coal_{GRID}": {"unit_base_cap": 20},
                f"biomass_heat_plant_{HS}": {"unit_base_cap": 10},
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series([1, 1, 0, 0, 1]),
                        EE: pd.Series([0.5, 1, 0, 1, 1]),
                    }
                }
            },
            {},
            {"grid->local_ee_bus": {"transmission_fee": None}},
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series([2, 1.5, 1]),
                        EE: pd.Series([1, 2, 1]),
                    },
                    "n_consumers": pd.Series([1] * 3),
                }
            },
            {},
            (21 + 14) * 8760 / 5,  # multiply by hourly scale
            False,
            id="variable_fuel_costs_variable_yearly_energy_usage",
        ),
        pytest.param(
            {"hour_sample": np.arange(5), "year_sample": np.arange(3)},
            {
                "biomass": {"cost": pd.Series(np.arange(5)), "energy_per_unit": 1.0},
                "coal": {"cost": pd.Series(np.ones(5)), "energy_per_unit": 1.0},
            },
            {
                "heat_plant_biomass": {
                    "efficiency": {HEAT: 0.5, EE: 0.0},
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
                "pp_coal": {
                    "efficiency": {HEAT: 0.0, EE: 1.0},
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {
                f"pp_coal_{GRID}": {"unit_base_cap": 20},
                f"biomass_heat_plant_{HS}": {"unit_base_cap": 10},
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series([1, 1, 0, 0, 1]),
                        EE: pd.Series([0.5, 1, 0, 1, 1]),
                    }
                }
            },
            {},
            {"grid->local_ee_bus": {"transmission_fee": None}},
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series([2, 1.5, 1]),
                        EE: pd.Series([1, 2, 1]),
                    },
                    "n_consumers": pd.Series([1] * 3),
                },
            },
            {},
            (21 + 14) * 8760 / 5,  # multiply by hourly scale
            True,
            id="ens_variable_fuel_costs_variable_yearly_energy_usage",
        ),
        pytest.param(
            {"hour_sample": np.arange(100), "year_sample": np.arange(3)},
            {
                "biomass": {
                    "cost": pd.Series(np.zeros(5)),
                },
                "coal": {
                    "cost": pd.Series(np.zeros(5)),
                },
            },
            {
                "heat_plant_biomass": {
                    "efficiency": {HEAT: 0.5, EE: 0.0},
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.ones(N_YEARS) * 1e3),
                },
                "pp_coal": {
                    "efficiency": {HEAT: 0.0, EE: 0.2},
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.ones(N_YEARS) * 1e2),
                },
            },
            {
                f"pp_coal_{GRID}": {"unit_base_cap": 0.05},
                f"biomass_heat_plant_{HS}": {"unit_base_cap": 0.2},
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series(np.ones(N_HOURS) / 1e4),
                        EE: pd.Series(np.ones(N_HOURS) / 1e4),
                    }
                }
            },
            {},
            {"grid->local_ee_bus": {"transmission_fee": None}},
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.ones(N_YEARS) * 1e3),
                        EE: pd.Series(np.ones(N_YEARS) * 1e2),
                    }
                }
            },
            {},
            615.0,
            False,
            id="constant_opex_constant_yearly_energy_usage",
        ),
        pytest.param(
            {"hour_sample": np.arange(100), "year_sample": np.arange(3)},
            {
                "biomass": {
                    "cost": pd.Series(np.zeros(5)),
                },
                "coal": {
                    "cost": pd.Series(np.zeros(5)),
                },
            },
            {
                "heat_plant_biomass": {
                    "efficiency": {HEAT: 0.5, EE: 0.0},
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.ones(N_YEARS) * 1e3),
                },
                "pp_coal": {
                    "efficiency": {HEAT: 0.0, EE: 0.2},
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.ones(N_YEARS) * 1e2),
                },
            },
            {
                f"pp_coal_{GRID}": {"unit_base_cap": 0.05},
                f"biomass_heat_plant_{HS}": {"unit_base_cap": 0.2},
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series(np.ones(N_HOURS) / 1e4),
                        EE: pd.Series(np.ones(N_HOURS) / 1e4),
                    }
                }
            },
            {},
            {"grid->local_ee_bus": {"transmission_fee": None}},
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.ones(N_YEARS) * 1e3),
                        EE: pd.Series(np.ones(N_YEARS) * 1e2),
                    }
                }
            },
            {},
            615.0,
            True,
            id="ens_constant_opex_constant_yearly_energy_usage",
        ),
        pytest.param(
            {
                "hour_sample": np.arange(100),
                "year_sample": np.arange(5),
                "discount_rate": np.array([0.5, 0.0, 1.0, 0.0, 0.5]),
            },
            {
                "biomass": {
                    "cost": pd.Series(np.zeros(5)),
                },
                "coal": {
                    "cost": pd.Series(np.zeros(5)),
                },
            },
            {
                "heat_plant_biomass": {
                    "efficiency": {HEAT: 1.0, EE: 0.0},
                    "capex": pd.Series(np.linspace(5 * 1e2, 1e2, N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                    "life_time": 1,
                    "build_time": 0,
                },
                "pp_coal": {
                    "efficiency": {HEAT: 0.0, EE: 1.0},
                    "capex": pd.Series(np.linspace(1e4, 2 * 1e3, N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                    "life_time": 1,
                    "build_time": 0,
                },
            },
            {
                f"pp_coal_{GRID}": {"unit_base_cap": 1 / 90},
                f"biomass_heat_plant_{HS}": {"unit_base_cap": 0.2},
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series(np.ones(N_HOURS) / 1e4),
                        EE: pd.Series(np.ones(N_HOURS) / 1e4),
                    }
                }
            },
            {},
            {"grid->local_ee_bus": {"transmission_fee": None}},
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.linspace(1e2, 5 * 1e2, N_YEARS)),
                        EE: pd.Series(np.linspace(1e2, 5 * 1e2, N_YEARS)),
                    }
                }
            },
            {},
            (109 / 9) + (2180 / 9),
            False,
            id="variable_capex_variable_yearly_energy_usage_non_constant_discount_rate",
        ),
        pytest.param(
            {
                "hour_sample": np.arange(100),
                "year_sample": np.arange(5),
                "discount_rate": np.array([0.5, 0.0, 1.0, 0.0, 0.5]),
            },
            {
                "biomass": {
                    "cost": pd.Series(np.zeros(5)),
                },
                "coal": {
                    "cost": pd.Series(np.zeros(5)),
                },
            },
            {
                "heat_plant_biomass": {
                    "efficiency": {HEAT: 1.0, EE: 0.0},
                    "capex": pd.Series(np.linspace(5 * 1e2, 1e2, N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                    "life_time": 1,
                    "build_time": 0,
                },
                "pp_coal": {
                    "efficiency": {HEAT: 0.0, EE: 1.0},
                    "capex": pd.Series(np.linspace(1e4, 2 * 1e3, N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                    "life_time": 1,
                    "build_time": 0,
                },
            },
            {
                f"pp_coal_{GRID}": {"unit_base_cap": 1 / 90},
                f"biomass_heat_plant_{HS}": {"unit_base_cap": 0.2},
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series(np.ones(N_HOURS) / 1e4),
                        EE: pd.Series(np.ones(N_HOURS) / 1e4),
                    }
                }
            },
            {},
            {"grid->local_ee_bus": {"transmission_fee": None}},
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.linspace(1e2, 5 * 1e2, N_YEARS)),
                        EE: pd.Series(np.linspace(1e2, 5 * 1e2, N_YEARS)),
                    }
                }
            },
            {},
            (109 / 9) + (2180 / 9),
            True,
            id="ens_variable_capex_variable_yearly_energy_usage_non_constant_discount_rate",
        ),
        pytest.param(
            {
                "hour_sample": np.arange(100),
                "year_sample": np.arange(5),
                "discount_rate": np.array([0.5, 0.0, 1.0, 0.0, 0.5]),
            },
            {
                "biomass": {
                    "cost": pd.Series(np.zeros(5)),
                },
                "coal": {
                    "cost": pd.Series(np.zeros(5)),
                },
            },
            {
                "heat_plant_biomass": {
                    "efficiency": {HEAT: 1.0, EE: 0.0},
                    "capex": pd.Series(np.ones(N_YEARS) * 1e2),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                    "life_time": 2,
                    "build_time": 1,
                },
                "pp_coal": {
                    "efficiency": {HEAT: 0.0, EE: 1.0},
                    "capex": pd.Series(np.ones(N_YEARS) * 1e3),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                    "life_time": 3,
                    "build_time": 0,
                },
            },
            {
                f"pp_coal_{GRID}": {"unit_base_cap": 0.01},
                f"biomass_heat_plant_{HS}": {"unit_base_cap": 0.01},
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series(np.ones(N_HOURS) / 1e4),
                        EE: pd.Series(np.ones(N_HOURS) / 1e4),
                    }
                }
            },
            {},
            {"grid->local_ee_bus": {"transmission_fee": None}},
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.linspace(1e2, 5 * 1e2, N_YEARS)),
                        EE: pd.Series(np.linspace(1e2, 5 * 1e2, N_YEARS)),
                    }
                }
            },
            {},
            (19 / 6) + (310 / 27),
            False,
            id="constant_capex_variable_yearly_energy_usage_non_constant_discount_rate",
        ),
        pytest.param(
            {
                "hour_sample": np.arange(100),
                "year_sample": np.arange(5),
                "discount_rate": np.array([0.5, 0.0, 1.0, 0.0, 0.5]),
            },
            {
                "biomass": {
                    "cost": pd.Series(np.zeros(5)),
                },
                "coal": {
                    "cost": pd.Series(np.zeros(5)),
                },
            },
            {
                "heat_plant_biomass": {
                    "efficiency": {HEAT: 1.0, EE: 0.0},
                    "capex": pd.Series(np.ones(N_YEARS) * 1e2),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                    "life_time": 2,
                    "build_time": 1,
                },
                "pp_coal": {
                    "efficiency": {HEAT: 0.0, EE: 1.0},
                    "capex": pd.Series(np.ones(N_YEARS) * 1e3),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                    "life_time": 3,
                    "build_time": 0,
                },
            },
            {
                f"pp_coal_{GRID}": {"unit_base_cap": 0.01},
                f"biomass_heat_plant_{HS}": {"unit_base_cap": 0.01},
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series(np.ones(N_HOURS) / 1e4),
                        EE: pd.Series(np.ones(N_HOURS) / 1e4),
                    }
                }
            },
            {},
            {"grid->local_ee_bus": {"transmission_fee": None}},
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.linspace(1e2, 5 * 1e2, N_YEARS)),
                        EE: pd.Series(np.linspace(1e2, 5 * 1e2, N_YEARS)),
                    }
                }
            },
            {},
            (19 / 6) + (310 / 27),
            True,
            id="ens_constant_capex_variable_yearly_energy_usage_non_constant_discount_rate",
        ),
        pytest.param(
            {"hour_sample": np.arange(100), "year_sample": np.arange(3)},
            {
                "biomass": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
                "coal": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {
                "heat_plant_biomass": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
                "pp_coal": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {},
            {},
            {},
            {},
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.linspace(1e2, 5 * 1e2, N_YEARS)),
                        EE: pd.Series([20000] * N_YEARS),
                    },
                    "n_consumers": pd.Series([3] * N_YEARS),
                }
            },
            {},
            20000
            * 3
            * 3
            * 0.19
            * 87.6
            * pd.read_csv(
                get_resources("test_network_assets") / "ee_profile.csv",
                sep=",",
                header=None,
                index_col=0,
            )
            .squeeze()[:100]
            .sum(),
            False,
            id="zero_cost_fuels+transmission_fee_3_consumers",
        ),
        pytest.param(
            {"hour_sample": np.arange(100), "year_sample": np.arange(3)},
            {
                "biomass": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
                "coal": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {
                "heat_plant_biomass": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
                "pp_coal": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {},
            {},
            {},
            {},
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.linspace(1e2, 5 * 1e2, N_YEARS)),
                        EE: pd.Series([20000] * N_YEARS),
                    },
                    "n_consumers": pd.Series([3] * N_YEARS),
                }
            },
            {},
            20000
            * 3
            * 3
            * 87.6
            * 0.19
            * pd.read_csv(
                get_resources("test_network_assets") / "ee_profile.csv",
                sep=",",
                header=None,
                index_col=0,
            )
            .squeeze()[:100]
            .sum(),
            True,
            id="ens_zero_cost_fuels+transmission_fee_3_consumers",
        ),
        pytest.param(
            {"hour_sample": np.arange(100), "year_sample": np.arange(3)},
            {
                "biomass": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
                "coal": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {
                "heat_plant_biomass": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
                "pp_coal": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {},
            {},
            {
                "TransmissionFee": {
                    "fee": pd.Series(
                        data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * int(N_HOURS / 10)
                    )
                }
            },
            {},
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.linspace(1e2, 5 * 1e2, N_YEARS)),
                        EE: pd.Series([20000] * N_YEARS),
                    },
                    "n_consumers": pd.Series([2] * N_YEARS),
                }
            },
            {},
            20000
            * 3
            * 2
            * 87.6
            * (
                pd.Series(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10)
                * pd.read_csv(
                    get_resources("test_network_assets") / "ee_profile.csv",
                    sep=",",
                    header=None,
                    index_col=0,
                )
                .squeeze()[:100]
                .reset_index(drop=True)
            ).sum(),
            False,
            id="zero_cost_fuels+variable_transmission_fee_2_consumers",
        ),
        pytest.param(
            {"hour_sample": np.arange(100), "year_sample": np.arange(3)},
            {
                "biomass": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
                "coal": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {
                "heat_plant_biomass": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
                "pp_coal": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {},
            {},
            {
                "TransmissionFee": {
                    "fee": pd.Series(
                        data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * int(N_HOURS / 10)
                    )
                }
            },
            {},
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.linspace(1e2, 5 * 1e2, N_YEARS)),
                        EE: pd.Series([20000] * N_YEARS),
                    },
                    "n_consumers": pd.Series([2] * N_YEARS),
                }
            },
            {},
            20000
            * 3
            * 2
            * 87.6
            * (
                pd.Series(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10)
                * pd.read_csv(
                    get_resources("test_network_assets") / "ee_profile.csv",
                    sep=",",
                    header=None,
                    index_col=0,
                )
                .squeeze()[:100]
                .reset_index(drop=True)
            ).sum(),
            True,
            id="ens_zero_cost_fuels+variable_transmission_fee_2_consumers",
        ),
        pytest.param(
            {"hour_sample": np.arange(100), "year_sample": np.arange(3)},
            {
                "biomass": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
                "coal": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {
                "heat_plant_biomass": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
                "pp_coal": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                    "emission_reduction": {CO2: 0.2, PM10: 0.0},
                },
            },
            {f"pp_coal_{GRID}": {"emission_fee": {"CO2_EMF"}}},
            {},
            {},
            {"grid->local_ee_bus": {"transmission_fee": None}},
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.linspace(1e2, 5 * 1e2, N_YEARS)),
                        EE: pd.Series([2000] * N_YEARS),
                    },
                }
            },
            {"CO2_EMF": {"price": pd.Series(data=[100.0] * N_YEARS)}},
            (
                pd.read_csv(
                    get_resources("test_network_assets") / "ee_profile.csv",
                    sep=",",
                    header=None,
                    index_col=0,
                )
                .squeeze()[:100]
                .reset_index(drop=True)
                .sum()  # demand_profil_first_100_h
                * 2000  # yearly_usage_for_aggr
                * 3  # nr_of_years
                * 13  # fuel_emissions
                * 0.8  # (1 - fuel_emission_reduc) = 1 - 0.2 = 0.8
                * 100  # emission fee price
                * 87.6  # hourly_scale = 8760 / 100
            )
            / 0.9  # generator effs
            / 15.0,  # energy_per_unit
            False,
            id="zero_cost_fuels_red_20_emission_fee_CO2_100_price",
        ),
        pytest.param(
            {"hour_sample": np.arange(100), "year_sample": np.arange(3)},
            {
                "biomass": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
                "coal": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {
                "heat_plant_biomass": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                },
                "pp_coal": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                    "emission_reduction": {CO2: 0.2, PM10: 0.0},
                },
            },
            {f"pp_coal_{GRID}": {"emission_fee": {"CO2_EMF"}}},
            {},
            {},
            {"grid->local_ee_bus": {"transmission_fee": None}},
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.linspace(1e2, 5 * 1e2, N_YEARS)),
                        EE: pd.Series([2000] * N_YEARS),
                    },
                }
            },
            {"CO2_EMF": {"price": pd.Series(data=[100.0] * N_YEARS)}},
            (
                pd.read_csv(
                    get_resources("test_network_assets") / "ee_profile.csv",
                    sep=",",
                    header=None,
                    index_col=0,
                )
                .squeeze()[:100]
                .reset_index(drop=True)
                .sum()  # demand_profil_first_100_h
                * 2000  # yearly_usage_for_aggr
                * 3  # nr_of_years
                * 13  # fuel_emissions
                * 0.8  # (1 - fuel_emission_reduc) = 1 - 0.2 = 0.8
                * 100  # emission fee price
                * 87.6  # hourly_scale = 8760 / 100
            )
            / 0.9  # generator effs
            / 15.0,  # energy_per_unit
            True,
            id="ens_zero_cost_fuels_red_20_emission_fee_CO2_100_price",
        ),
        pytest.param(
            {"hour_sample": np.arange(100), "year_sample": np.arange(3)},
            {
                "biomass": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
                "coal": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {
                "heat_plant_biomass": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                    "emission_reduction": {CO2: 0.0, PM10: 0.3},
                },
                "pp_coal": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                    "emission_reduction": {CO2: 0.2, PM10: 0.0},
                },
            },
            {
                f"pp_coal_{GRID}": {"emission_fee": {"CO2_EMF"}},
                f"biomass_heat_plant_{HS}": {"emission_fee": {"PM10_EMF"}},
            },
            {},
            {},
            {"grid->local_ee_bus": {"transmission_fee": None}},
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series([1000] * N_YEARS),
                        EE: pd.Series([2000] * N_YEARS),
                    },
                }
            },
            {
                "CO2_EMF": {"price": pd.Series(data=[100.0] * N_YEARS)},
                "PM10_EMF": {"price": pd.Series(data=[50.0] * N_YEARS)},
            },
            (
                (
                    pd.read_csv(
                        get_resources("test_network_assets") / "ee_profile.csv",
                        sep=",",
                        header=None,
                        index_col=0,
                    )
                    .squeeze()[:100]
                    .reset_index(drop=True)
                    .sum()  # demand_profil_first_100_h
                    * 2000  # yearly_usage_for_aggr
                    * 3  # nr_of_years
                    * 13  # fuel_emissions
                    * 0.8  # (1 - fuel_emission_reduc) = 1 - 0.2 = 0.8
                    * 100  # emission fee price
                    * 87.6  # hourly_scale = 8760 / 100
                )
                / 0.9  # generator effs
                / 15.0
            )  # energy_per_unit
            + (
                (
                    pd.read_csv(
                        get_resources("test_network_assets") / "heat_profile.csv",
                        sep=",",
                        header=None,
                        index_col=0,
                    )
                    .squeeze()[:100]
                    .reset_index(drop=True)
                    .sum()  # demand_profil_first_100_h
                    * 1000  # yearly_usage_for_aggr
                    * 3  # nr_of_years
                    * 0.2  # fuel_emissions
                    * 0.7  # (1 - fuel_emission_reduc) = 1 - 0.2 = 0.8
                    * 50  # emission fee price
                    * 87.6  # hourly_scale = 8760 / 100
                )
                / 0.5  # generator effs
                / 0.7
            ),  # energy_per_unit
            False,
            id="zero_cost_fuels_PM10_CO2_EMF_100_50_red_20_30",
        ),
        pytest.param(
            {"hour_sample": np.arange(100), "year_sample": np.arange(3)},
            {
                "biomass": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
                "coal": {
                    "cost": pd.Series(np.zeros(N_YEARS)),
                },
            },
            {
                "heat_plant_biomass": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                    "emission_reduction": {CO2: 0.0, PM10: 0.3},
                },
                "pp_coal": {
                    "capex": pd.Series(np.zeros(N_YEARS)),
                    "opex": pd.Series(np.zeros(N_YEARS)),
                    "emission_reduction": {CO2: 0.2, PM10: 0.0},
                },
            },
            {
                f"pp_coal_{GRID}": {"emission_fee": {"CO2_EMF"}},
                f"biomass_heat_plant_{HS}": {"emission_fee": {"PM10_EMF"}},
            },
            {},
            {},
            {"grid->local_ee_bus": {"transmission_fee": None}},
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series([1000] * N_YEARS),
                        EE: pd.Series([2000] * N_YEARS),
                    },
                }
            },
            {
                "CO2_EMF": {"price": pd.Series(data=[100.0] * N_YEARS)},
                "PM10_EMF": {"price": pd.Series(data=[50.0] * N_YEARS)},
            },
            (  # CO2
                (
                    pd.read_csv(
                        get_resources("test_network_assets") / "ee_profile.csv",
                        sep=",",
                        header=None,
                        index_col=0,
                    )
                    .squeeze()[:100]
                    .reset_index(drop=True)
                    .sum()  # demand_profil_first_100_h
                    * 2000  # yearly_usage_for_aggr
                    * 3  # nr_of_years
                    * 13  # fuel_emissions
                    * 0.8  # (1 - fuel_emission_reduc) = 1 - 0.2 = 0.8
                    * 100  # emission fee price
                    * 87.6  # hourly_scale = 8760 / 100
                )
                / 0.9  # generator effs
                / 15.0
            )  # energy_per_unit
            + (  # PM10
                (
                    pd.read_csv(
                        get_resources("test_network_assets") / "heat_profile.csv",
                        sep=",",
                        header=None,
                        index_col=0,
                    )
                    .squeeze()[:100]
                    .reset_index(drop=True)
                    .sum()  # demand_profil_first_100_h
                    * 1000  # yearly_usage_for_aggr
                    * 3  # nr_of_years
                    * 0.2  # fuel_emissions
                    * 0.7  # (1 - fuel_emission_reduc) = 1 - 0.2 = 0.8
                    * 50  # emission fee price
                    * 87.6  # hourly_scale = 8760 / 100
                )
                / 0.5  # generator effs
                / 0.7
            ),  # energy_per_unit
            False,
            id="ens_zero_cost_fuels_PM10_CO2_EMF_100_50_red_20_30",
        ),
    ],
)
def test_objective_value(
    opt_config_params: dict[str, np.ndarray],
    fuels_params: dict,
    generator_type_params: dict,
    generator_params: dict,
    demand_params: dict[str, pd.Series],
    transmission_fee_params: dict,
    line_params: dict,
    aggr_params: dict[str, ...],  # type: ignore
    emission_fee_params: dict,
    expected_obj: float,
    network: Network,
    ens: bool,
) -> None:
    """
    Test if objective function value is correct (for given parameters)
    Test ens if expected to be used and if it is switched off when not needed
    """

    set_network_elements_parameters(network.demand_profiles, demand_params)
    set_network_elements_parameters(network.aggregated_consumers, aggr_params)
    set_network_elements_parameters(network.generators, generator_params)
    set_network_elements_parameters(network.transmission_fees, transmission_fee_params)
    set_network_elements_parameters(network.lines, line_params)
    set_network_elements_parameters(network.fuels, fuels_params)
    set_network_elements_parameters(network.generator_types, generator_type_params)
    set_network_elements_parameters(network.emission_fees, emission_fee_params)

    # test if enabling ens nothing happen if no change in demand / generation

    opt_config = create_default_opf_config(**opt_config_params)
    opt_config.ens = ens
    engine = run_opt_engine(network, opt_config)

    assert np.isclose(engine.results.objective_value, expected_obj)
    assert engine.results.objective_value < 1e6

    # now enforcing ens to be used:
    if ens:
        network.aggregated_consumers["aggr"].yearly_energy_usage["heat"] *= 1e6
        network.aggregated_consumers["aggr"].yearly_energy_usage["electricity"] *= 1e6

        engine = run_opt_engine(network, opt_config)
        assert engine.results.objective_value > 1e6


@pytest.mark.parametrize(
    ("opt_config_params", "fuels_params", "generator_type_params", "should_be_equal"),
    [
        (
            {"hour_sample": np.arange(50), "year_sample": np.arange(N_YEARS)},
            {
                "coal": {"cost": pd.Series(np.zeros(N_YEARS))},
                "biomass": {"cost": pd.Series(np.zeros(N_YEARS))},
            },
            {
                "heat_plant_biomass": {"opex": pd.Series(np.zeros(N_YEARS))},
                "pp_coal": {"opex": pd.Series(np.zeros(N_YEARS))},
            },
            False,
        ),
        (
            {"hour_sample": np.arange(50), "year_sample": np.arange(N_YEARS)},
            {},
            {
                "heat_plant_biomass": {"capex": pd.Series(np.zeros(N_YEARS))},
                "pp_coal": {"capex": pd.Series(np.zeros(N_YEARS))},
            },
            True,
        ),
    ],
)
def test_discount_rate(
    opt_config_params: dict[str, np.ndarray],
    fuels_params: dict[str, pd.Series],
    generator_type_params: dict,
    should_be_equal: bool,
    network: Network,
) -> None:
    """
    Test, if only opex depends on discount_rate
    """

    set_network_elements_parameters(network.fuels, fuels_params)
    set_network_elements_parameters(network.generator_types, generator_type_params)

    opt_config_with_discount = create_default_opf_config(**opt_config_params)
    discounted_results = run_opt_engine(network, opt_config_with_discount).results

    opt_config_no_discount = create_default_opf_config(
        **(opt_config_params | {"discount_rate": np.zeros(N_YEARS)})
    )
    no_discount_results = run_opt_engine(network, opt_config_no_discount).results

    assert (
        np.isclose(
            discounted_results.objective_value, no_discount_results.objective_value
        )
        == should_be_equal
    )


@pytest.mark.parametrize(
    (
        "hour_sample_winter",
        "hour_sample_spring",
        "hour_sample_summer",
        "hour_sample_autumn",
    ),
    [
        pytest.param(
            np.arange(49, 73),
            np.arange(3313, 3337),
            np.arange(5593, 5617),
            np.arange(7465, 7489),
            id="one more day",
        ),
        pytest.param(
            np.arange(49, 97),
            np.arange(3313, 3361),
            np.arange(5593, 5641),
            np.arange(7465, 7513),
            id="two more day",
        ),
        pytest.param(
            np.arange(49, 121),
            np.arange(3313, 3385),
            np.arange(5593, 5665),
            np.arange(7465, 7537),
            id="three more day",
        ),
    ],
)
def test_hourly_scale_factor(
    network: Network,
    hour_sample_winter: np.ndarray,
    hour_sample_spring: np.ndarray,
    hour_sample_summer: np.ndarray,
    hour_sample_autumn: np.ndarray,
) -> None:
    base_hour_sample = (
        pd.read_csv(
            get_resources("integration_test/parameters/hour_sample.csv"), header=None
        )
        .squeeze()
        .values
    )
    extended_hour_sample = base_hour_sample.copy()
    for day_values in (
        hour_sample_winter,
        hour_sample_spring,
        hour_sample_summer,
        hour_sample_autumn,
    ):
        extended_hour_sample = np.insert(
            extended_hour_sample,
            np.searchsorted(extended_hour_sample, day_values[0]),
            day_values,
        )

    opt_config_base = create_default_opf_config(
        hour_sample=base_hour_sample, year_sample=np.arange(1)
    )
    opt_config_extended = create_default_opf_config(
        hour_sample=extended_hour_sample, year_sample=np.arange(1)
    )
    engine_base = run_opt_engine(network, opt_config_base)
    engine_extended = run_opt_engine(network, opt_config_extended)

    assert np.isclose(
        engine_base.results.objective_value,
        engine_extended.results.objective_value,
        rtol=1e-1,
        atol=0.0,
    )
