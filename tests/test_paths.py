from pathlib import PosixPath

import pytest

from esp_data.io import AnyPath, is_cloud_path, is_local_path


def test_local_path():
    path = AnyPath("tests/fileio_test_folder/file1.txt")
    assert isinstance(path, PosixPath)
    assert path.is_file()
    assert path.read_text().strip() == "hello"
    assert is_local_path(path)
    assert path.exists()


@pytest.mark.parametrize(
    "cloud_path",
    [
        "gs://esp-ci-cd-tests/esp-data-tests/file1.txt",
        "r2://esp-ci-cd-tests/esp-data-tests/file1.txt",
    ],
)
def test_cloud_path(cloud_path):
    path = AnyPath(cloud_path)
    path.upload_from("tests/fileio_test_folder/file1.txt")
    assert not is_local_path(path)
    assert is_cloud_path(path)
    assert path.exists()
    assert path.is_file()
    assert path.read_bytes() == b"hello\n"
    path.unlink()
    assert not path.exists()
