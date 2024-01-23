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


def assert_same_exception_list(
    actual_exception_list: list, exception_list: list
) -> None:
    assert len(actual_exception_list) == len(exception_list)

    actual_exception_list.sort(key=lambda x: str(x))
    exception_list.sort(key=lambda x: str(x))

    for actual, excepted in zip(actual_exception_list, exception_list):
        assert isinstance(actual, type(excepted))
        assert str(actual) == str(excepted)
