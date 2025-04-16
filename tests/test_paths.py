from pathlib import PosixPath

import pytest

from esp_data.io import anypath


def test_local_path():
    path = anypath("tests/samples/file1.txt")
    assert isinstance(path, PosixPath)
    assert path.is_file()
    assert path.read_text().strip() == "hello"
    assert path.is_local
    assert path.exists()

@pytest.mark.parametrize(
    "cloud_path",
    [
        "gs://esp-ci-cd-tests/esp-data-tests/file1.txt",
        "r2://esp-ci-cd-tests/esp-data-tests/file1.txt",
    ],
)
def test_cloud_upload_path(cloud_path):
    path = anypath(cloud_path)
    path.upload_from("tests/samples/file1.txt")
    assert not path.is_local
    assert path.is_cloud
    assert path.exists()
    assert path.is_file()
    assert path.read_bytes() == b"hello\n"
    path.unlink()
    assert not path.exists()
