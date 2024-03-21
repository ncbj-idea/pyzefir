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

from collections import defaultdict
from logging import Logger


class NetworkExceptionFormatter:
    def __init__(self, exc: Exception) -> None:
        self.exc, self.msg = self.sort_exceptions(exc)

    @staticmethod
    def sort_exceptions(exc: Exception) -> tuple[dict[str, list[Exception]], str]:
        if not isinstance(exc, ExceptionGroup):
            return {exc.__class__.__name__: [exc]}, exc.args[0]

        exceptions_dict = defaultdict(list)
        for e in exc.exceptions:
            exceptions_dict[e.__class__.__name__].append(e)
        return dict(exceptions_dict), exc.message

    def format(self, logger: Logger) -> None:
        logger.error(self.msg)
        for exc_class, exc_list in self.exc.items():
            logger.error(f"{exc_class:#^100}")
            for exc in exc_list:
                if not isinstance(exc, ExceptionGroup):
                    logger.error(f"{exc.args[0]}")
                    continue
                logger.error(f"{exc.message}")
                for e_sub in exc.exceptions:
                    logger.error(f"\t\t{e_sub.args[0]}")
