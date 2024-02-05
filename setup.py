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

import os

from setuptools import find_packages, setup


def get_version() -> str:
    version = "unknown"
    with open("pyzefir/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                _, _, version = line.replace('"', "").split()
                break
    return version


def load_req(r_file: str) -> list[str]:
    with open(os.path.join(os.getcwd(), r_file)) as f:
        return [
            r for r in (line.split("#", 1)[0].strip() for line in f.readlines()) if r
        ]


setup(
    name="pyzefir",
    packages=find_packages(".", exclude=["*tests*"]),
    version=get_version(),
    install_requires=load_req("requirements.txt"),
    python_requires=">=3.11",
    author="Narodowe Centrum Badań Jądrowych",
    license="GNU General Public License v3 or later (GPLv3+)",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "pyzefir=pyzefir.cli.runner:cli_run",
            "structure-creator=pyzefir.structure_creator.cli.cli_wrapper:run_structure_creator_cli",
        ]
    },
)
