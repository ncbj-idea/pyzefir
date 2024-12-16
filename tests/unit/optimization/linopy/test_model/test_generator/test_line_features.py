import numpy as np
import pandas as pd
import pytest

from pyzefir.model.network import Network
from pyzefir.model.network_elements import AggregatedConsumer, DemandProfile, Line
from pyzefir.optimization.model import OptimizationStatus
from tests.unit.optimization.linopy.constants import N_HOURS, N_YEARS
from tests.unit.optimization.linopy.names import EE, HEAT
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opt_config,
    run_opt_engine,
)


@pytest.mark.parametrize(
    ("year_sample", "hour_sample", "transmission_loss"),
    [
        (np.arange(5), np.arange(60), 0.1),
        (np.array([0, 1, 2, 3, 4]), np.arange(20), 0.3),
        (np.array([0, 1, 2]), np.array([1300]), 0.001),
    ],
)
def test_line_flows_and_transmission_loss(
    year_sample: np.ndarray,
    hour_sample: np.ndarray,
    transmission_loss: float,
    network: Network,
    grid_connection: Line,
    heating_system_connection: Line,
    demand_profile: DemandProfile,
    aggr: AggregatedConsumer,
) -> None:
    """
    Conditions to check:
        * grid_connection line flow == ee_demand * (1 - grid_connection.transmission_loss)
        * heating_system line flow == heat_demand * (1 - heating_system_connection.transmission_loss)
    """

    heating_system_connection.transmission_loss = transmission_loss
    grid_connection.transmission_loss = transmission_loss

    opt_config = create_default_opt_config(hour_sample, year_sample)
    engine = run_opt_engine(network, opt_config)
    energy_usage = aggr.yearly_energy_usage

    ee_energy_flow = engine.results.lines_results.flow[grid_connection.name]
    ee_demand = (
        demand_profile.normalized_profile[EE].values
        * energy_usage[EE].values.reshape(-1, 1)
    ).T

    heat_energy_flow = engine.results.lines_results.flow[heating_system_connection.name]
    heat_demand = (
        demand_profile.normalized_profile[HEAT].values
        * energy_usage[HEAT].values.reshape(-1, 1)
    ).T

    assert np.allclose(
        (1 - transmission_loss) * ee_energy_flow,
        ee_demand[hour_sample, :][:, year_sample],
    )
    assert np.allclose(
        (1 - transmission_loss) * heat_energy_flow,
        heat_demand[hour_sample, :][:, year_sample],
    )


@pytest.mark.parametrize(
    (
        "hour_sample",
        "year_sample",
        "heat_energy_usage",
        "ee_energy_usage",
        "heat_line_capacity",
        "ee_line_capacity",
        "expected_opt_status",
        "ens",
    ),
    [
        (np.arange(50), np.arange(5), 0, 0, 0, 0, [OptimizationStatus.OPTIMAL], False),
        (np.arange(50), np.arange(5), 0, 0, 0, 0, [OptimizationStatus.OPTIMAL], True),
        (
            np.arange(100),
            np.arange(5),
            1e2,
            1e2 * 0.5,
            0.01,
            0.005,
            [OptimizationStatus.OPTIMAL],
            1.0,
        ),
        (
            np.arange(100),
            np.arange(5),
            1e2,
            1e2 * 0.5,
            0.01 - 0.00001,
            0.005,
            [OptimizationStatus.WARNING],
            np.nan,
        ),
        (
            np.arange(100),
            np.arange(5),
            1e2,
            1e2 * 0.5,
            0.01 - 0.00001,
            0.005,
            [OptimizationStatus.OPTIMAL],
            1.0,
        ),
        (
            np.arange(100),
            np.arange(5),
            1e2,
            1e2 * 0.5,
            0.01,
            0.005 - 0.00001,
            [OptimizationStatus.WARNING],
            np.nan,
        ),
        (
            np.arange(100),
            np.arange(5),
            1e2,
            1e2 * 0.5,
            0.01,
            0.005 - 0.00001,
            [OptimizationStatus.OPTIMAL],
            1.0,
        ),
    ],
)
def test_line_capacity(
    hour_sample: np.ndarray,
    year_sample: np.ndarray,
    heat_energy_usage: float,
    ee_energy_usage: float,
    heat_line_capacity: float,
    ee_line_capacity: float,
    expected_opt_status: list[OptimizationStatus],
    ens: float,
    network: Network,
    grid_connection: Line,
    heating_system_connection: Line,
    aggr: AggregatedConsumer,
    demand_profile: DemandProfile,
) -> None:
    """
    Additional assumptions:
        * set demand profile to constant value == 1 / (100 * #hour_sample)
        * line transmission_losses set to 0
    Conditions to check:
        * check if for given values of line capacity model is feasible
        * check if max(flow) == line capacity
    """

    grid_connection.max_capacity = ee_line_capacity
    grid_connection.transmission_loss = 0

    heating_system_connection.max_capacity = heat_line_capacity
    heating_system_connection.transmission_loss = 0

    aggr.yearly_energy_usage[HEAT] = pd.Series(np.ones(N_YEARS) * heat_energy_usage)
    aggr.yearly_energy_usage[EE] = pd.Series(np.ones(N_YEARS) * ee_energy_usage)

    demand_profile.normalized_profile[HEAT] = pd.Series(
        np.ones(N_HOURS) / 100 / hour_sample.shape[0]
    )
    demand_profile.normalized_profile[EE] = pd.Series(
        np.ones(N_HOURS) / 100 / hour_sample.shape[0]
    )

    network.aggregated_consumers["aggr"].yearly_energy_usage["heat"] *= 1e6
    network.aggregated_consumers["aggr"].yearly_energy_usage["electricity"] *= 1e6

    opt_config = create_default_opt_config(hour_sample, year_sample)
    opt_config.ens = ens
    engine = run_opt_engine(network, opt_config)

    assert (
        engine.status in expected_opt_status if not ens else OptimizationStatus.OPTIMAL
    )

    if expected_opt_status == OptimizationStatus.OPTIMAL:
        heat_energy_flow = engine.results.lines_results.flow[
            heating_system_connection.name
        ]
        ee_energy_flow = engine.results.lines_results.flow[grid_connection.name]

        assert np.allclose(heat_line_capacity, heat_energy_flow.max(axis=0))
        assert np.allclose(ee_line_capacity, ee_energy_flow.max(axis=0))
