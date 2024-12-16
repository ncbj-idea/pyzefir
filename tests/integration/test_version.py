import re

import pyzefir


def test_version() -> None:
    assert re.match(r"^\d+\.\d+\.\d+$", pyzefir.__version__)
