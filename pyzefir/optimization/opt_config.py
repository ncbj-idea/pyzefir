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

from pathlib import Path

from numpy import arange, diff, ndarray, zeros
from numpy.random import choice


class OptConfigError(Exception):
    pass


class OptConfigErrorGroup(OptConfigError, ExceptionGroup):
    pass


class OptConfig:
    """
    Class containing the configuration for an optimization engine setup and run
    """

    def __init__(
        self,
        hours: int | ndarray,
        years: int | ndarray,
        discount_rate: ndarray | None = None,
        hour_sample: int | ndarray | None = None,
        year_sample: int | ndarray | None = None,
        sol_dump_path: Path | None = None,
        opt_logs_dump_path: Path | None = None,
        money_scale: float = 1.0,
        ens: bool = True,
        use_hourly_scale: bool = True,
    ):
        self.hours: ndarray = hours if isinstance(hours, ndarray) else arange(hours)
        """ sequence of all hours in a year """
        self.years: ndarray = years if isinstance(years, ndarray) else arange(years)
        """ sequence of all years """
        self.sol_dump_path: Path | None = sol_dump_path
        """ path where *.sol file will be dumped """
        self.opt_logs_dump_path: Path | None = opt_logs_dump_path
        """ path where gurobi log file will be dumped """
        self.discount_rate: ndarray = (
            discount_rate if isinstance(discount_rate, ndarray) else zeros(years)
        )
        """ capital discount rate """
        self.hour_sample: ndarray = self.get_sample(
            self.hours, hour_sample, use_arange=False
        )
        """ subsequence of hours sequence (sample of hours that will be used in the model) """
        self.year_sample: ndarray = self.get_sample(
            self.years, year_sample, use_arange=True
        )
        """ subsequence of years sequence (sample of years that will be used in the model) """
        self.money_scale = money_scale
        """ numeric scale parameter """
        self.ens = ens
        """ use ens associated with buses if not balanced """
        self.hourly_scale: float = (
            len(self.hours) / len(self.hour_sample) if use_hourly_scale else 1.0
        )
        """ ratio of the total number of hours to the total number of hours in given sample"""
        self.validate()

    def validate(self) -> None:
        """
        validate if discount_rate, hours and years are 1D arrays
        validate if discount_rate and years have the same shape
        validate ens type
        validate if money_scale is >= 1
        validate if year_sample is consecutive
        """
        exception_list: list[OptConfigError] = []
        if (
            not len(self.discount_rate.shape)
            == len(self.hours.shape)
            == len(self.years.shape)
            == 1
        ):
            exception_list.append(
                OptConfigError("discount_rate, hours and years must be 1D arrays")
            )
        if not self.discount_rate.shape == self.years.shape:
            exception_list.append(
                OptConfigError("discount_rate shape is different than years shape")
            )
        if not isinstance(self.ens, bool):
            exception_list.append(
                OptConfigError(
                    f"ens flag must be of type bool, "
                    f"but it is {type(self.ens).__name__} instead"
                )
            )
        if self.money_scale < 1:
            exception_list.append(
                OptConfigError("money scale must be greater or equal 1")
            )

        if not (diff(self.year_sample) == 1).all() or self.year_sample[0] != 0:
            exception_list.append(
                OptConfigError("year sample must be consecutive starting from 0")
            )

        if exception_list:
            raise OptConfigErrorGroup("Errors in configuration: ", exception_list)

    @staticmethod
    def get_sample(
        idx: ndarray, sample: int | ndarray | None, use_arange: bool = False
    ) -> ndarray:
        if isinstance(sample, int):
            if use_arange:
                if sample <= len(idx):
                    result = arange(sample)
                else:
                    raise OptConfigError(
                        f"year sample {sample} must be less than or equal to year shape {len(idx)}"
                    )
            else:
                result = choice(idx.shape[0], sample, replace=False)
        elif isinstance(sample, ndarray):
            result = sample
        elif sample is None:
            result = idx
        else:
            raise OptConfigError(
                f"sample must be {int}, {ndarray} or {None}, but is of type {type(sample)}"
            )
        return result
