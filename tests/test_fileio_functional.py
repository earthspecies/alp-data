import pytest

from esp_data import AnyPath
from esp_data.file_io.functional import (
    # create_file,
    delete_dir,
    delete_file,
    download,
    # exists,
    list_files,
    makedirs,
    open_file,
    read_bytes,
    read_text,
    upload,
    write_bytes,
    write_text,
)


@pytest.fixture
def local_test_dir(tmp_path):
    """Create a temporary directory for local tests."""
    test_dir = tmp_path / "test_fileio"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.mark.parametrize(
    "cloud_path",
    [
        AnyPath("gs://esp-ci-cd-tests/esp-data-tests/test_upload_file.bin"),
        AnyPath("r2://esp-ci-cd-tests/esp-data-tests/test_upload_file.bin"),
    ],
)
def test_upload_download_cloud(local_test_dir, cloud_path):
    """Test uploading to and downloading from cloud buckets."""
    local_file = local_test_dir / "cloud_test.bin"
    local_file.write_bytes(b"Hello Cloud")

    assert cloud_path.exists() is False
    assert local_file.exists() is True

    # Upload to remote
    assert upload(local_file, cloud_path) is True
    # Download back to a different local file
    download_target = local_test_dir / "cloud_test_download.bin"
    assert download(cloud_path, str(download_target)) is True
    assert download_target.read_bytes() == b"Hello Cloud"
    assert delete_file(cloud_path) is True


# def test_create_local_file(local_test_dir):
#     """Test creating a local file."""
#     test_file = local_test_dir / "test_create.txt"
#     assert create_file(str(test_file), data=b"Hello") is True
#     assert test_file.exists()
#     assert test_file.read_bytes() == b"Hello"


# def test_create_and_delete_file_cloud(local_test_dir):
#     """Test creating a file in a cloud bucket."""
#     test_file = "gs://esp-ci-cd-tests/esp-data-tests/test_create_cloud.txt"
#     assert create_file(test_file, data=b"Hello") is True
#     assert exists(test_file) is True
#     assert download(test_file, str(local_test_dir / "test_create_cloud.txt")) is True
#     assert (local_test_dir / "test_create_cloud.txt").read_bytes() == b"Hello"
#     # delete remote file
#     assert delete_file(test_file) is True
#     assert exists(test_file) is False


def test_open_file_read_write(local_test_dir):
    """Test opening a file for read/write."""
    test_file = local_test_dir / "test_open.txt"
    test_file.write_text("Line1")

    with open_file(str(test_file), "r") as f:
        data = f.read()
    assert data == "Line1"

    with open_file(str(test_file), "a") as f:
        f.write("Line2")

    with open_file(str(test_file), "r") as f:
        data = f.read()
    assert data == "Line1Line2"


def test_list_files(local_test_dir):
    """Test listing files in a directory."""
    (local_test_dir / "file_a.txt").write_text("A")
    (local_test_dir / "file_b.txt").write_text("B")
    sub_dir = local_test_dir / "sub"
    sub_dir.mkdir()
    (sub_dir / "file_c.txt").write_text("C")

    files = list_files(str(local_test_dir), pattern="**/*.txt")
    assert len(files) >= 3
    assert any("file_a.txt" in x for x in files)
    assert any("file_b.txt" in x for x in files)
    assert any("file_c.txt" in x for x in files)


def test_read_bytes(local_test_dir):
    """Test reading bytes from a file."""
    test_file = local_test_dir / "test_read.bin"
    test_file.write_bytes(b"\x00\xff\x10")
    data = read_bytes(str(test_file))
    assert data == b"\x00\xff\x10"


def test_read_text(local_test_dir):
    """Test reading text from a file."""
    test_file = local_test_dir / "test_read.txt"
    test_file.write_text("Hello")
    content = read_text(str(test_file))
    assert content == "Hello"


def test_write_bytes(local_test_dir):
    """Test writing bytes to a file."""
    test_file = local_test_dir / "test_write.bin"
    assert write_bytes(str(test_file), b"ABC") is True
    assert test_file.read_bytes() == b"ABC"


def test_write_text(local_test_dir):
    """Test writing text to a file."""
    test_file = local_test_dir / "test_write.txt"
    assert write_text(str(test_file), "Hello World") is True
    assert test_file.read_text() == "Hello World"


def test_delete_file(local_test_dir):
    """Test deleting a file."""
    test_file = local_test_dir / "test_delete.txt"
    test_file.write_text("To be deleted.")
    assert test_file.exists()
    assert delete_file(str(test_file)) is True
    assert not test_file.exists()


def test_makedirs(local_test_dir):
    """Test creating directories."""
    new_dir = local_test_dir / "nested" / "dir"
    assert makedirs(str(new_dir)) is True
    assert new_dir.exists()
    assert new_dir.is_dir()


@pytest.mark.parametrize(
    "cloud_dir",
    [
        AnyPath("gs://esp-ci-cd-tests/esp-data-tests/dir_tests"),
        AnyPath("r2://esp-ci-cd-tests/esp-data-tests/dir_tests"),
    ],
)
def test_makedirs_in_cloud(cloud_dir):
    """Test creating directories in the cloud."""
    assert makedirs(cloud_dir) is True
    assert cloud_dir.exists() is True
    assert delete_dir(cloud_dir) is True
    assert cloud_dir.exists() is False


@pytest.mark.parametrize(
    "cloud_dir",
    [
        AnyPath("gs://esp-ci-cd-tests/esp-data-tests/dir_tests_list"),
        AnyPath("r2://esp-ci-cd-tests/esp-data-tests/dir_tests_list"),
    ],
)
def test_list_files_in_cloud(cloud_dir, local_test_dir):
    """Test listing files in a cloud directory."""
    makedirs(cloud_dir)
    test_file = local_test_dir / "file_to_list_cloud.txt"
    test_file.write_text("Cloud content")
    remote_path = f"{cloud_dir}/file_to_list_cloud.txt"
    assert upload(str(test_file), remote_path)
    files = list_files(cloud_dir)
    assert any("file_to_list_cloud.txt" in f for f in files)
    assert delete_file(remote_path) is True


@pytest.mark.parametrize(
    "cloud_dir",
    [
        AnyPath("gs://esp-ci-cd-tests/esp-data-tests/delete_files_tests"),
        AnyPath("r2://esp-ci-cd-tests/esp-data-tests/delete_files_tests"),
    ],
)
def test_delete_files_in_cloud(cloud_dir, local_test_dir):
    """Test deleting files in a cloud directory."""
    makedirs(cloud_dir)
    test_file = local_test_dir / "file_delete_cloud.txt"
    test_file.write_text("Delete from cloud")
    remote_path = f"{cloud_dir}/file_delete_cloud.txt"
    upload(str(test_file), remote_path)
    assert delete_file(remote_path) is True
    # Try listing again to ensure file is gone
    files = list_files(cloud_dir)
    assert not any("file_delete_cloud.txt" in f for f in files)


@pytest.mark.parametrize(
    "cloud_dir",
    [
        AnyPath("gs://esp-ci-cd-tests/esp-data-tests/delete_dir_tests"),
        AnyPath("r2://esp-ci-cd-tests/esp-data-tests/delete_dir_tests"),
    ],
)
def test_delete_dir_in_cloud(cloud_dir, local_test_dir):
    """Test deleting a directory in a cloud bucket."""
    makedirs(cloud_dir)
    test_file = local_test_dir / "file_delete_dir_cloud.txt"
    test_file.write_text("Delete from cloud")
    remote_path = f"{cloud_dir}/file_delete_dir_cloud.txt"
    upload(str(test_file), remote_path)
    assert delete_file(remote_path) is True
    # Try listing again to ensure file is gone
    files = list_files(cloud_dir)
    assert not any("file_delete_dir_cloud.txt" in f for f in files)
    # Delete the directory
    assert delete_dir(cloud_dir) is True
