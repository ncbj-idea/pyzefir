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
from bidict import bidict

from pyzefir.optimization.linopy.preprocessing.parameters.generator_parameters import (
    GeneratorParameters,
)
from pyzefir.optimization.linopy.preprocessing.parameters.generator_type_parameters import (
    GeneratorTypeParameters,
)


def get_generator_types_capacity_multipliers(
    generator_capacity_cost: str,
    tgen: GeneratorTypeParameters,
) -> dict[int, float]:
    """
    Mapping capacity multiplier to a generator type

    Capacity multiplier is simply an efficiency for a given generator type if generator_capacity_cost
    is set to 'netto' and 1.0 if it is set to 'brutto'.
    """
    if generator_capacity_cost == "brutto":
        return {tgen_idx: 1.0 for tgen_idx in tgen.eff}
    elif generator_capacity_cost == "netto":
        return {
            tgen_idx: list(tgen_eff.values())[0].mean()
            for tgen_idx, tgen_eff in tgen.eff.items()
        }
    else:
        raise RuntimeError(
            f"incorrect generator_capacity_cost, given {generator_capacity_cost}, but expected 'brutto' or 'netto'"
        )


def get_generators_capacity_multipliers(
    generator_capacity_cost: str,
    tgen: GeneratorTypeParameters,
    gen: GeneratorParameters,
) -> dict[int, float]:
    """
    Mapping capacity multiplier to a generator

    Capacity multiplier is simply an efficiency for a given generator if generator_capacity_cost
    is set to 'netto' and 1.0 if it is set to 'brutto',
    """
    if generator_capacity_cost == "brutto":
        return {gen_idx: 1.0 for gen_idx in gen.tgen}
    elif generator_capacity_cost == "netto":
        return {
            gen_idx: list(tgen.eff[gen.tgen[gen_idx]].values())[0].mean()
            for gen_idx in gen.tgen
        }
    raise RuntimeError(
        f"incorrect generator_capacity_cost, given {generator_capacity_cost}, but expected 'brutto' or 'netto'"
    )


def calculate_storage_adjusted_generation(
    generation_result_df: pd.DataFrame,
    storages_generation_efficiency: dict[int, float],
    storages_idxs: bidict[int, str | int],
) -> dict[str, pd.DataFrame]:
    """
    Adjusts generation results for each storage by applying storage efficiency factors.

    This function takes a DataFrame containing generation results.
    Each storage's generation data is adjusted by multiplying it with the corresponding
    efficiency factor. The function returns a dictionary where each key is the storage name and the
    corresponding value is a DataFrame with the adjusted generation data for that storage.

    Args:
        generation_result_df (pd.DataFrame):
            A DataFrame containing generation results with an index of 'stor' (storage name)
            and other relevant data.

        storages_generation_efficiency (dict[int, float]):
            A dictionary mapping storage IDs (int) to their corresponding efficiency factors (float).

        storages_idxs (dict[int, str]):
            A dictionary mapping storage IDs (int) to human-readable storage names (str).

    Returns:
        dict[str, pd.DataFrame]:
            A dictionary where each key is a storage name (str), and each value is a DataFrame
            with the generation results adjusted by the storage's efficiency factor.
    """
    storage_name_storage_gen_eff_dict = {
        storages_idxs[idx]: storages_generation_efficiency[idx]
        for idx in storages_generation_efficiency
    }
    generation_dict = {
        stor_name: df.reset_index(["stor"], drop=True).unstack().droplevel(0, axis=1)
        * storage_name_storage_gen_eff_dict[stor_name]
        for stor_name, df in generation_result_df.groupby("stor")
    }
    return generation_dict
