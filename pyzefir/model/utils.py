"""
PyZefir
Copyright (C) 2023 Narodowe Centrum Badań Jądrowych

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import itertools


class ClassCounter:
    counters = {}
    names = {}

    @classmethod
    def get_counter(cls, class_to_count):
        cls.counters.setdefault(class_to_count, itertools.count())
        return next(cls.counters[class_to_count])

    @classmethod
    def check_name_uniqueness(cls, class_to_check, name_to_check):
        if class_to_check not in cls.names:
            cls.names[class_to_check] = set()

        if name_to_check in cls.names[class_to_check]:
            raise NameError(
                f"you are trying to create an instance of a class {class_to_check} with name "
                f"{name_to_check}, however an instance of a class {class_to_check} with this name "
                f"already exist."
            )


class IdElement:

    def __init__(self, name: str):
        self.__id = ClassCounter.get_counter(self.__class__)
        ClassCounter.check_name_uniqueness(self.__class__, name)
        self.__name = name

    @property
    def id(self) -> int:
        return self.__id

    @property
    def name(self) -> str:
        return self.__name

    def __eq__(self, other):
        return isinstance(other, IdElement) and other.id == self.id and other.name == self.name

    def __hash__(self):
        return hash(self.__name)
