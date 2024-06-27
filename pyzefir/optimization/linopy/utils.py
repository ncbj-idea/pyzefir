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
    is set to 'netto' and 1.0 if it is set to 'brutto',
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
