import numpy as np
import pandas as pd
import pytest

from pyzefir.model.network import Network
from pyzefir.model.network_elements import DemandChunk
from pyzefir.utils.functions import demand_chunk_unit_indices
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opt_config,
    run_opt_engine,
    set_network_elements_parameters,
)
from tests.unit.optimization.linopy.utils import TOL


@pytest.mark.parametrize(
    ("energy_type", "tag", "periods", "demand", "gen_tags", "stor_tags"),
    [
        (
            "electricity",
            "ee_tag",
            np.array([[0, 4], [45, 49]]),
            np.array([[2, 1, 1, 3, 1], [1, 1, 2, 1, 2]]),
            {},
            {"ee_storage": ["ee_tag"]},
        ),
        (
            "electricity",
            "ee_tag",
            np.array([[0, 4], [45, 49]]),
            np.array([[0, 2, 1, 1, 0], [1, 0, 0, 1, 0]]),
            {"pp_coal_grid": ["ee_tag"]},
            {},
        ),
        (
            "electricity",
            "ee_tag",
            np.array([[0, 3], [40, 49]]),
            np.array([[2, 2, 3, 3, 1], [1, 4, 2, 5, 2]]),
            {"pp_coal_grid": ["ee_tag"]},
            {"ee_storage": ["ee_tag"]},
        ),
        (
            "electricity",
            "ee_tag",
            np.array([[0, 3], [12, 23], [40, 49]]),
            np.array([[2, 2, 3, 3, 1], [2, 2, 3, 3, 1], [1, 4, 2, 5, 2]]),
            {"pp_coal_grid": ["ee_tag"]},
            {"ee_storage": ["ee_tag"]},
        ),
    ],
)
def test_demand_chunks(
    energy_type: str,
    tag: str,
    periods: np.ndarray,
    demand: np.ndarray,
    gen_tags: dict[str, list[str]],
    stor_tags: dict[str, list[str]],
    network: Network,
) -> None:
    """
    Test demand chunks for generators and storages
    """
    network.demand_chunks = {
        "dem_1": DemandChunk(
            name="dem_1",
            demand=demand,
            energy_type=energy_type,
            periods=periods,
            tag=tag,
        )
    }
    opt_config = create_default_opt_config(
        hour_sample=np.array(range(24, 63)), year_sample=np.array(range(5))
    )
    _load_tags_to_network(network, gen_tags, stor_tags)
    engine = run_opt_engine(network, opt_config)
    dch_param = engine.parameters.demand_chunks_parameters
    years = engine.indices.Y.ord
    et = engine.parameters.gen.ett
    for dem_idx, dem_val in dch_param.demand.items():
        energy_type = dch_param.energy_type[dem_idx]
        gen_names = {
            engine.indices.GEN.mapping[idx]
            for idx in demand_chunk_unit_indices(
                dem_idx, dch_param.tag, engine.parameters.gen.tags
            )
            if energy_type in et[idx]
        }
        stor_names = {
            engine.indices.STOR.mapping[idx]
            for idx in demand_chunk_unit_indices(
                dem_idx, dch_param.tag, engine.parameters.stor.tags
            )
        }

        gen_dch = engine.results.generators_results.gen_dch
        stor_dch = engine.results.storages_results.gen_dch
        dem_name = engine.indices.DEMCH.mapping[dem_idx]
        row_idx = 0
        for period in dch_param.periods[dem_idx]:
            p_start, p_end = period[0], period[-1]
            h_range = range(p_start, p_end + 1)
            from_gen = _demand_chunk_generation_sum(
                gen_dch, dem_name, gen_names, h_range, years
            )
            from_stor = _demand_chunk_generation_sum(
                stor_dch, dem_name, stor_names, h_range, years
            )
            demand = np.array(dem_val[row_idx][: len(engine.indices.Y.ord)])
            row_idx += 1
            assert np.all(abs(from_gen + from_stor - demand) < TOL)


def _load_tags_to_network(
    network: Network, gen_tags: dict[str, list[str]], stor_tags: dict[str, list[str]]
) -> None:
    for unit, unit_tag in gen_tags.items():
        set_network_elements_parameters(network.generators, {unit: {"tags": unit_tag}})
    for unit, unit_tag in stor_tags.items():
        set_network_elements_parameters(network.storages, {unit: {"tags": unit_tag}})


def _demand_chunk_generation_sum(
    gen_dch: dict[str, dict[str, pd.DataFrame]],
    dem_name: str,
    unit_names: set[str],
    h_range: range,
    years: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            sum(
                gen_dch[unit_name][dem_name][y][h]
                for unit_name in unit_names
                for h in h_range
            )
            for y in years
        ]
    )
