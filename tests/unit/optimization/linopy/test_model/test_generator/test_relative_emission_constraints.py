import numpy as np
import pandas as pd
import pytest

from pyzefir.model.network import Network
from pyzefir.model.utils import NetworkConstants
from tests.unit.optimization.linopy.constants import N_HOURS, N_YEARS
from tests.unit.optimization.linopy.names import CO2, PM10
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opt_config,
    run_opt_engine,
    set_network_elements_parameters,
)


@pytest.mark.parametrize(
    ("relative_emission_limit", "aggr_parameters", "check_fuel_emissions"),
    [
        pytest.param(
            {
                CO2: pd.Series([np.nan, 1.0, 0.95, 0.9, 0.85]),
                PM10: pd.Series([np.nan] * N_YEARS),
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        "heat": pd.Series(
                            pd.Series([1000] * N_YEARS),
                        ),
                        "electricity": pd.Series([1000, 1050, 1100, 1150, 1200]),
                    }
                }
            },
            [("coal", "CO2")],
            id="CO2 emission limits 0.05 down per year, heat yearly_usage static, ee yearly_usage up by 50",
        ),
        pytest.param(
            {
                CO2: pd.Series([np.nan, 1.0, 1.05, 1.1, 1.15]),
                PM10: pd.Series([np.nan] * N_YEARS),
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        "heat": pd.Series(
                            pd.Series([1000] * N_YEARS),
                        ),
                        "electricity": pd.Series([1000, 950, 900, 850, 800]),
                    }
                }
            },
            [("coal", "CO2")],
            id="CO2 emission limits 0.05 up per year, heat yearly_usage static, ee yearly_usage down by 50",
        ),
        pytest.param(
            {
                CO2: pd.Series([np.nan] * N_YEARS),
                PM10: pd.Series([np.nan, 1.0, 0.95, 0.9, 0.85]),
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        "heat": pd.Series([1000, 1050, 1100, 1150, 1200]),
                        "electricity": pd.Series(pd.Series([1000] * N_YEARS)),
                    }
                }
            },
            [("biomass", "PM10")],
            id="PM10 emission limits 0.05 down per year, heat yearly_usage up by 50, ee yearly_usage static",
        ),
        pytest.param(
            {
                CO2: pd.Series([np.nan] * N_YEARS),
                PM10: pd.Series([np.nan, 1.0, 1.05, 1.1, 1.15]),
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        "heat": pd.Series([1000, 950, 900, 850, 800]),
                        "electricity": pd.Series(pd.Series([1000] * N_YEARS)),
                    }
                }
            },
            [("biomass", "PM10")],
            id="PM10 emission limits 0.05 up per year, heat yearly_usage down by 50, ee yearly_usage static",
        ),
        pytest.param(
            {
                CO2: pd.Series([np.nan, 1.0, 0.9, 0.8, 0.7]),
                PM10: pd.Series([np.nan, 1.0, 0.9, 0.8, 0.7]),
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        "heat": pd.Series([1000, 1050, 1100, 1150, 1200]),
                        "electricity": pd.Series([1000, 1050, 1100, 1150, 1200]),
                    }
                }
            },
            [("biomass", "PM10"), ("coal", "CO2")],
            id="PM10 and CO2 down limits by 0.1, yearly_usage up by 50",
        ),
        pytest.param(
            {
                CO2: pd.Series([np.nan, 1.3, 1.4, 1.5, 1.6]),
                PM10: pd.Series([np.nan, 1.0, 1.05, 1.1, 1.15]),
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        "heat": pd.Series([1250, 1300, 1350, 1400, 1450]),
                        "electricity": pd.Series([1000, 990, 980, 970, 960]),
                    }
                }
            },
            [("biomass", "PM10"), ("coal", "CO2")],
            id="PM10 and CO2 up limits by 0.1, yearly_usage down by 50",
        ),
    ],
)
def test_generator_emission_limit_constraint(
    network: Network,
    relative_emission_limit: dict[str, pd.Series],
    check_fuel_emissions: list[tuple[str, str]],
    aggr_parameters: dict[str, dict],
) -> None:
    set_network_elements_parameters(network.aggregated_consumers, aggr_parameters)
    base_total_emission = {"CO2": 1.0, "PM10": 1.0}
    new_network_constants = NetworkConstants(
        n_years=N_YEARS,
        n_hours=N_HOURS,
        relative_emission_limits=relative_emission_limit,
        base_total_emission=base_total_emission,
        power_reserves={},
    )
    network.constants = new_network_constants
    opt_config = create_default_opt_config(
        hour_sample=np.arange(10), year_sample=np.arange(N_YEARS)
    )
    engine = run_opt_engine(network, opt_config)

    for fuel_name, emission_type in check_fuel_emissions:
        total_emissions = (
            (
                sum(
                    [
                        engine.results.generators_results.gen[key].sum()
                        for key in engine.results.generators_results.gen.keys()
                        if fuel_name in key
                    ]
                )
                / network.fuels[fuel_name].energy_per_unit
            )
            * opt_config.hourly_scale
            * network.fuels[fuel_name].emission[emission_type]
        ).round(2)
        base_emissions_limit = (
            relative_emission_limit[emission_type]
            * base_total_emission[emission_type]
            * opt_config.hourly_scale
        ).round(2)
        assert all(total_emissions[1:] <= base_emissions_limit[1:])
