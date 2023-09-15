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

import os

from setuptools import find_packages, setup

import pyzefir


def load_req(r_file: str) -> list[str]:
    with open(os.path.join(os.getcwd(), r_file)) as f:
        return [r for r in
                (line.split('#', 1)[0].strip() for line in f.readlines()) if r]


setup(
    name="pyzefir",
    packages=find_packages('.', exclude=['*tests*']),
    version=pyzefir.__version__,
    install_requires=load_req('requirements.txt'),
    python_requires=">=3.11",
    author='Narodowe Centrum Badań Jądrowych',
    author_email='office@idea.edu.pl',
    license="AGPLv3",
)
