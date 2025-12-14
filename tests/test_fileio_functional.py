import pytest
from uuid import uuid4

from esp_data.io import anypath, filesystem_from_path, exists, rm


@pytest.fixture
def local_test_dir(tmp_path):
    """Create a temporary directory for local tests."""
    test_dir = tmp_path / "test_fileio"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.mark.parametrize(
    "cloud_path",
    [
        anypath("gs://esp-ci-cd-tests/esp-data-tests/test_upload_file.bin"),
        anypath("r2://esp-ci-cd-tests/esp-data-tests/test_upload_file.bin"),
    ],
)
def test_upload_download_cloud(local_test_dir, cloud_path):
    """Test uploading to and downloading from cloud buckets."""
    local_file = local_test_dir / "cloud_test.bin"
    local_file.write_bytes(b"Hello Cloud")

    assert exists(local_file)

    # Upload to remote

    assert not exists(cloud_path)
    filesystem_from_path(cloud_path).put(str(local_file), str(cloud_path))
    assert exists(cloud_path)

    # Download back to a different local file
    download_target = local_test_dir / "cloud_test_download.bin"
    filesystem_from_path(cloud_path).get(str(cloud_path), str(download_target))
    assert download_target.read_bytes() == b"Hello Cloud"
    rm(cloud_path)
    assert not exists(cloud_path)


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

    with test_file.open("r") as f:
        data = f.read()
    assert data == "Line1"

    with test_file.open("a") as f:
        f.write("Line2")

    with test_file.open("r") as f:
        data = f.read()
    assert data == "Line1Line2"


def test_list_files(local_test_dir):
    """Test listing files in a directory."""
    (local_test_dir / "file_a.txt").write_text("A")
    (local_test_dir / "file_b.txt").write_text("B")
    sub_dir = local_test_dir / "sub"
    sub_dir.mkdir()
    (sub_dir / "file_c.txt").write_text("C")

    files = filesystem_from_path(local_test_dir).glob(str(local_test_dir / "**/*.txt"))
    assert len(files) >= 3
    assert any("file_a.txt" in x for x in files)
    assert any("file_b.txt" in x for x in files)
    assert any("file_c.txt" in x for x in files)


def test_read_bytes(local_test_dir):
    """Test reading bytes from a file."""
    test_file = local_test_dir / "test_read.bin"
    test_file.write_bytes(b"\x00\xff\x10")
    data = test_file.read_bytes()
    assert data == b"\x00\xff\x10"


def test_read_text(local_test_dir):
    """Test reading text from a file."""
    test_file = local_test_dir / "test_read.txt"
    test_file.write_text("Hello")
    content = test_file.read_text()
    assert content == "Hello"


def test_write_bytes(local_test_dir):
    """Test writing bytes to a file."""
    test_file = local_test_dir / "test_write.bin"
    assert test_file.write_bytes(b"ABC") == 3
    assert test_file.read_bytes() == b"ABC"


def test_write_text(local_test_dir):
    """Test writing text to a file."""
    test_file = local_test_dir / "test_write.txt"
    assert test_file.write_text("Hello World") == 11
    assert test_file.read_text() == "Hello World"


def test_delete_file(local_test_dir):
    """Test deleting a file."""
    test_file = local_test_dir / "test_delete.txt"
    test_file.write_text("To be deleted.")
    assert test_file.exists()
    test_file.unlink()
    assert not test_file.exists()


def test_makedirs(local_test_dir):
    """Test creating directories."""
    new_dir = anypath(local_test_dir / "nested" / "dir")

    assert new_dir.exists() is False
    new_dir.mkdir(parents=True, exist_ok=False)
    assert new_dir.exists()
    assert new_dir.is_dir()


@pytest.mark.parametrize(
    "cloud_dir",
    [
        anypath("gs://esp-ci-cd-tests/esp-data-tests/dir_tests_list"),
        anypath("r2://esp-ci-cd-tests/esp-data-tests/dir_tests_list"),
    ],
)
def test_list_files_in_cloud(cloud_dir, local_test_dir):
    """Test listing files in a cloud directory."""

    # TODO (milad): Can't we just pre-populate a cloud directory with files and avoid
    #               creating the file and doing the put()?

    test_file = local_test_dir / "file_to_list_cloud.txt"
    test_file.write_text("Cloud content")

    remote_path = cloud_dir / "file_to_list_cloud.txt"

    fs = filesystem_from_path(remote_path)
    fs.put(str(test_file), str(remote_path))

    files = fs.ls(str(cloud_dir))
    assert any("file_to_list_cloud.txt" in f for f in files)
    rm(remote_path)
    assert not exists(anypath(remote_path))


@pytest.mark.parametrize(
    "cloud_dir",
    [
        anypath("gs://esp-ci-cd-tests/esp-data-tests/delete_files_tests"),
        anypath("r2://esp-ci-cd-tests/esp-data-tests/delete_files_tests"),
    ],
)
def test_delete_files_in_cloud(cloud_dir, local_test_dir):
    """Test deleting files in a cloud directory."""
    test_file = local_test_dir / "file_delete_cloud.txt"
    test_file.write_text("Delete from cloud")

    remote_path = cloud_dir / str(uuid4())

    assert not exists(remote_path)
    filesystem_from_path(remote_path).put(str(test_file), str(remote_path))

    assert exists(remote_path)
    rm(remote_path)
    assert not exists(remote_path)
