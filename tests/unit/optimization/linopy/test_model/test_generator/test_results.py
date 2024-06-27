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
from pyzefir.model.network_elements import Storage
from pyzefir.optimization.exportable_results import ExportableResults
from pyzefir.optimization.results import Results
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.names import HS
from tests.unit.optimization.linopy.preprocessing.utils import create_storage_type
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opt_config,
    run_opt_engine,
)


@pytest.fixture
def prepare_results(network: Network) -> tuple[Results, ExportableResults]:
    hour_sample = np.arange(50)
    year_sample = np.arange(5)

    opt_config = create_default_opt_config(hour_sample, year_sample)
    storage_type = create_storage_type(name="test_storage_type", energy_type="heat")
    network.add_storage_type(storage_type)
    storage = Storage(
        name=f"heat_storage_{HS}",
        energy_source_type="test_storage_type",
        unit_base_cap=15,
        bus=HS,
        unit_min_capacity=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity=pd.Series([np.nan] * N_YEARS),
        unit_min_capacity_increase=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity_increase=pd.Series([np.nan] * N_YEARS),
    )
    network.add_storage(storage)
    engine = run_opt_engine(network, opt_config)
    exportable_results = engine.results.to_exportable()
    return engine.results, exportable_results


def test_exportable_generator_results(
    prepare_results: tuple[Results, ExportableResults]
) -> None:
    results, exportable_results = prepare_results
    assert isinstance(exportable_results.generators_results.capacity, pd.DataFrame)
    for column in results.generators_results.cap:
        assert np.all(
            results.generators_results.cap[column].values.reshape(-1)
            == exportable_results.generators_results.capacity[column].values
        )
    for gen in results.generators_results.gen:
        assert isinstance(
            exportable_results.generators_results.generation[gen], pd.DataFrame
        )
        assert np.all(
            results.generators_results.gen[gen]
            == exportable_results.generators_results.generation[gen]
        )
    for gen in results.generators_results.dump:
        assert isinstance(
            exportable_results.generators_results.dump_energy[gen], pd.DataFrame
        )
        assert np.all(
            results.generators_results.dump[gen]
            == exportable_results.generators_results.dump_energy[gen]
        )


def test_exportable_lines_results(
    prepare_results: tuple[Results, ExportableResults]
) -> None:
    results, exportable_results = prepare_results
    for line in results.lines_results.flow:
        assert line in exportable_results.lines_results.flow
        assert isinstance(exportable_results.lines_results.flow[line], pd.DataFrame)
        assert np.all(
            results.lines_results.flow[line].values
            == exportable_results.lines_results.flow[line].values
        )


def test_exportable_fraction_results(
    prepare_results: tuple[Results, ExportableResults]
) -> None:
    results, exportable_results = prepare_results
    for aggr in results.fractions_results.frac:
        assert aggr in exportable_results.fractions_results.fraction
        assert isinstance(
            exportable_results.fractions_results.fraction[aggr], pd.DataFrame
        )
        for lbs in results.fractions_results.frac[aggr]:
            assert lbs in exportable_results.fractions_results.fraction[aggr].columns
            assert np.all(
                results.fractions_results.frac[aggr][lbs].values
                == exportable_results.fractions_results.fraction[aggr][lbs].values
            )


def test_exportable_storage_results(
    prepare_results: tuple[Results, ExportableResults]
) -> None:
    results, exportable_results = prepare_results
    for st in results.storages_results.gen:
        assert np.all(
            results.storages_results.gen[st].values
            == exportable_results.storages_results.generation[st].values
        )
    for st in results.storages_results.load:
        assert np.all(
            results.storages_results.load[st].values
            == exportable_results.storages_results.load[st].values
        )
    for st in results.storages_results.soc:
        assert np.all(
            results.storages_results.soc[st].values
            == exportable_results.storages_results.state_of_charge[st].values
        )
    for column in results.storages_results.cap:
        assert column in results.storages_results.cap
        assert np.all(
            results.storages_results.cap[column].values.reshape(-1)
            == exportable_results.storages_results.capacity[column].values
        )


def test_objective_function(prepare_results: tuple[Results, ExportableResults]) -> None:
    results, exportable_results = prepare_results
    assert results.objective_value == exportable_results.objective_value[0]
