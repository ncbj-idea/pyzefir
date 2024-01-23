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

import pandas as pd
import pytest

from pyzefir.optimization.exportable_results import (
    ExportableBusResults,
    ExportableFractionsResults,
    ExportableGeneratorsResults,
    ExportableLinesResults,
    ExportableResults,
    ExportableStorageResults,
)


@pytest.fixture()
def generators_results() -> ExportableGeneratorsResults:
    return ExportableGeneratorsResults(
        generation={
            "data1": pd.DataFrame(
                {"sample_1": [15, 30, 45, 60], "sample_2": [0.75, 0.15, 0.045, 0.006]}
            ),
            "data2": pd.DataFrame(
                {"sample_1": [0.9, 0.18, 0.054, 0.0072], "sample_2": [18, 36, 54, 72]}
            ),
        },
        dump_energy={
            "data3": pd.DataFrame(
                {"sample_1": [18, 36, 54, 72], "sample_2": [0.9, 0.18, 0.054, 0.0072]}
            ),
            "data4": pd.DataFrame(
                {"sample_1": [21, 42, 63, 84], "sample_2": [1.05, 0.21, 0.063, 0.0084]}
            ),
        },
        capacity=pd.DataFrame(
            {"sample_1": [24, 48, 72, 96], "sample_2": [1.2, 0.24, 0.072, 0.0096]}
        ),
        generation_per_energy_type={
            "data1": pd.DataFrame(
                {"sample_1": [15, 30, 45, 60], "sample_2": [0.75, 0.15, 0.045, 0.006]}
            ),
            "data2": pd.DataFrame(
                {"sample_1": [0.9, 0.18, 0.054, 0.0072], "sample_2": [18, 36, 54, 72]}
            ),
        },
        dump_energy_per_energy_type={
            "data3": pd.DataFrame(
                {"sample_1": [18, 36, 54, 72], "sample_2": [0.9, 0.18, 0.054, 0.0072]}
            ),
            "data4": pd.DataFrame(
                {"sample_1": [21, 42, 63, 84], "sample_2": [1.05, 0.21, 0.063, 0.0084]}
            ),
        },
        capex=pd.DataFrame({"sample_1": [0, 1, 2, 3], "sample_2": [0, 1, 2, 3]}),
    )


@pytest.fixture()
def storages_results() -> ExportableStorageResults:
    return ExportableStorageResults(
        generation={
            "data1": pd.DataFrame(
                {"sample_1": [27, 54, 81, 108], "sample_2": [1.35, 0.27, 0.081, 0.0108]}
            ),
            "data2": pd.DataFrame(
                {"sample_1": [30, 60, 90, 120], "sample_2": [1.5, 0.3, 0.09, 0.012]}
            ),
        },
        load={
            "data3": pd.DataFrame(
                {"sample_1": [33, 66, 99, 132], "sample_2": [1.65, 0.33, 0.099, 0.0132]}
            ),
            "data4": pd.DataFrame(
                {"sample_1": [36, 72, 108, 144], "sample_2": [1.8, 0.36, 0.108, 0.0144]}
            ),
        },
        state_of_charge={
            "data3": pd.DataFrame(
                {
                    "sample_1": [39, 78, 117, 156],
                    "sample_2": [1.95, 0.39, 0.117, 0.0156],
                }
            ),
            "data4": pd.DataFrame(
                {"sample_1": [42, 84, 126, 168], "sample_2": [2.1, 0.42, 0.126, 0.0168]}
            ),
        },
        capacity=pd.DataFrame(
            {"sample_1": [15, 30, 45, 60], "sample_2": [0.75, 0.15, 0.045, 0.006]}
        ),
        capex=pd.DataFrame({"sample_1": [0, 1, 2, 3], "sample_2": [0, 1, 2, 3]}),
    )


@pytest.fixture()
def lines_results() -> ExportableLinesResults:
    return ExportableLinesResults(
        flow={
            "data1": pd.DataFrame(
                {"sample_1": [45, 90, 135, 180], "sample_2": [2.25, 0.45, 0.135, 0.018]}
            ),
            "data2": pd.DataFrame(
                {"sample_1": [48, 96, 144, 192], "sample_2": [2.4, 0.48, 0.144, 0.0192]}
            ),
            "data3": pd.DataFrame(
                {
                    "sample_1": [51, 102, 153, 204],
                    "sample_2": [2.55, 0.51, 0.153, 0.0204],
                }
            ),
        },
    )


@pytest.fixture()
def frac_results() -> ExportableFractionsResults:
    return ExportableFractionsResults(
        fraction={
            "data1": pd.DataFrame(
                {
                    "sample_1": [54, 108, 162, 216],
                    "sample_2": [2.7, 0.54, 0.162, 0.0216],
                }
            ),
            "data2": pd.DataFrame(
                {
                    "sample_1": [57, 114, 171, 228],
                    "sample_2": [2.85, 0.57, 0.171, 0.0228],
                }
            ),
            "data3": pd.DataFrame(
                {"sample_1": [60, 120, 180, 240], "sample_2": [3.0, 0.6, 0.18, 0.024]}
            ),
        }
    )


@pytest.fixture()
def bus_results() -> ExportableBusResults:
    return ExportableBusResults(
        generation_ens={
            "data1": pd.DataFrame(
                {"sample_1": [15, 30, 45, 60], "sample_2": [0.75, 0.15, 0.045, 0.006]}
            ),
            "data2": pd.DataFrame(
                {"sample_1": [0.9, 0.18, 0.054, 0.0072], "sample_2": [18, 36, 54, 72]}
            ),
        }
    )


@pytest.fixture()
def objective_result() -> pd.Series:
    return pd.Series(
        [69, 138, 207, 276],
        index=["attr1", "attr2", "attr3", "attr4"],
        name="Objective_value",
    )


@pytest.fixture
def exportable_results(
    generators_results: ExportableGeneratorsResults,
    storages_results: ExportableStorageResults,
    lines_results: ExportableLinesResults,
    frac_results: ExportableFractionsResults,
    bus_results: ExportableBusResults,
    objective_result: pd.Series,
) -> ExportableResults:
    return ExportableResults(
        generators_results=generators_results,
        storages_results=storages_results,
        lines_results=lines_results,
        fractions_results=frac_results,
        objective_value=objective_result,
        bus_results=bus_results,
    )
