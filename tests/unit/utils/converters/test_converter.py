import tempfile
from pathlib import Path

from pyzefir.utils.converters.converter import AbstractConverter


def test_abstract_converter_convert() -> None:
    class MyConverter(AbstractConverter):
        def __init__(self) -> None:
            self.status = ""

        def convert(self) -> None:
            self.status = "Conversion complete"

    converter = MyConverter()
    converter.convert()
    assert converter.status == "Conversion complete"


def test_manage_existence_path_file_exists() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        path = tmp_dir_path / "output" / "test_file.txt"
        AbstractConverter.manage_existence_path(path)
        assert tmp_dir_path.joinpath("output").is_dir()
        assert not tmp_dir_path.joinpath("input").is_dir()
        assert not path.is_file()
