from pathlib import PosixPath

import pytest

from esp_data.io import anypath, filesystem_from_path


def test_local_path():
    path = anypath("tests/samples/file1.txt")
    assert isinstance(path, PosixPath)
    assert path.is_file()
    assert path.read_text().strip() == "hello"
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

    fs = filesystem_from_path(path)

    fs.put("tests/samples/file1.txt", str(path))

    info = fs.info(str(path))
    assert info["size"] == 6
    assert info["type"] == "file"
    with fs.open(str(path), "rb") as f:
        assert f.read() == b"hello\n"

    fs.rm(str(path))
    assert not fs.exists(str(path))
