"""Pytest based unit tests for file_io module."""

import os
import shutil

import pytest
from cloudpathlib import GSPath

from esp_data.file_io.buckets import Bucket, GSBucket, GSBucketV2, R2Bucket
from esp_data.file_io.files import File, GSAudioFile, GSFile


def test_bucket_empty():
    # DO NOT CHANGE STUFF IN THIS FOLDER OR TESTS WILL FAIL.
    bucket = Bucket("gs://esp-ci-cd-tests/esp-data-tests/keep_empty")
    assert bucket.list_files() == []
    assert bucket.list_dirs() == []
    assert list(bucket.find_paths_with_extension(".py")) == []
    assert list(bucket.find_paths_containing("test")) == []


def test_bucket_non_empty():
    # DO NOT CHANGE STUFF IN THIS FOLDER OR TESTS WILL FAIL.
    bucket = Bucket("gs://esp-ci-cd-tests/esp-data-tests/non_empty")
    assert len(bucket.list_files(recursive=False)) == 2
    assert len(bucket.list_dirs(recursive=True)) == 0
    assert len(list(bucket.find_paths_with_extension(".txt"))) == 1
    assert len(list(bucket.find_paths_containing("random"))) == 0


def test_bucket_move_create_delete():
    # Move directories
    file = File("gs://esp-ci-cd-tests/esp-data-tests/temp_folder/temp_file.txt")
    file.create()
    bucket = Bucket("gs://esp-ci-cd-tests/esp-data-tests/")
    bucket.move_dir("temp_folder", "non_empty", overwrite=True, keep_parent=True)
    b = bucket.subdir_as_bucket("non_empty/temp_folder")
    assert b.exists
    b.delete_dir("", confirm=False)
    assert not b.exists


def test_gcs_bucket_upload():
    """Test GCSBucket class."""
    bucket = GSBucket("gs://esp-ci-cd-tests/esp-data-tests/temprandomfolderxyz")
    bucket.upload_dir("tests/fileio_test_folder")
    assert bucket.exists
    assert len(bucket.list_files(recursive=True)) > 0
    bucket.delete_dir("", confirm=False)
    assert not bucket.exists


def test_gcs_bucket_upload_data_as_str():
    b = GSBucket("gs://esp-ci-cd-tests/esp-data-tests/test_temp_subfolder")
    file_names = ["random1.txt", "random2.txt"]
    file_data = ["hello", "world"]
    b.upload_data_as_str(file_names, file_data)
    # confirm they are present
    uploaded_files = b.list_files(recursive=True)
    assert len(uploaded_files) == 2
    assert uploaded_files == [GSPath(f"{str(b)}/{file}") for file in file_names]
    b.delete_dir("", confirm=False)
    assert not b.exists


def test_gcs_bucket_rsync():
    src = GSBucket("gs://esp-ci-cd-tests/esp-data-tests/non_empty")
    dest = GSBucket("gs://esp-ci-cd-tests/esp-data-tests/some_subfolder")
    src.rsync(dest, self_is_source=True, gzip_in_flight="txt")
    f = GSFile("gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/test.csv")
    assert f.exists
    f.delete(confirm=False)


async def test_gcs_bucket_async_rsync():
    src = GSBucket("gs://esp-ci-cd-tests/esp-data-tests/non_empty")
    dest = GSBucket("gs://esp-ci-cd-tests/esp-data-tests/some_subfolder")
    await src.async_rsync(dest, self_is_source=True, gzip_in_flight="txt")
    f = GSFile("gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/test.csv")
    assert f.exists
    f.delete(confirm=False)


def test_local_file_open():
    file = File("tests/test_file_io.py")
    assert file.exists
    assert file.is_local
    assert file.read_bytes()[:5] == b'"""Py'
    fp = file.open("r")
    assert fp.readline()[:5] == '"""Py'


def test_cloud_file():
    file = File("gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/random.txt")
    assert file.exists
    assert not file.is_local
    file.download_to("tests/fileio_test_folder/random.txt")
    assert file.size() == 0


def test_cloud_file_copy():
    file = File("gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/random.txt")
    file.copy_to("gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/random_copy.txt")
    f = File("gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/random_copy.txt")
    assert f.exists
    f.delete(confirm=False)


def test_cloud_file_create():
    file = File("gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/random_new.txt")
    file.upload_from("tests/fileio_test_folder/random_local.txt")
    assert file.exists
    assert file.size() > 0
    file.delete(confirm=False)


def test_gs_file():
    with pytest.raises(ValueError):
        file = GSFile("s3://esp-ci-cd-tests/esp-data-tests/some_subfolder/random.txt")
    file = GSFile("gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/random.txt")
    assert file.exists
    assert not file.is_local
    file.download_to("tests/fileio_test_folder/random.txt")
    file = GSFile("gs://esp-ci-cd-tests/esp-data-tests/non_empty/random_for_upload_test.txt")
    file.upload_from_bytes_or_str("hello")
    assert file.exists
    assert file.size() > 0
    assert file.read_bytes() == b"hello"
    file.delete(confirm=False)
    assert not file.exists


def test_gs_audio_file():
    file = GSAudioFile("gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/nri-battlesounds.mp3")
    assert file.exists
    audio, sr = file.read_audio()
    assert len(audio) > 0
    assert sr == 44100


async def test_gs_bucket_v2():
    bucket = GSBucketV2("gs://esp-ci-cd-tests/esp-data-tests/temprandomfolder")
    bucket.upload_dir("tests/fileio_test_folder")
    assert bucket.exists
    assert len(bucket.list_files(recursive=True)) > 0
    bucket.delete_dir("tests/fileio_test_folder", confirm=False)

    await bucket.async_upload_dir("tests/fileio_test_folder")
    bucket.mkdir("tests/fileio_test_folder2")
    await bucket.async_move_dir("tests/fileio_test_folder", "tests/fileio_test_folder2")
    await bucket.async_download_to("tests/fileio_test_folder2")
    assert os.path.exists("tests/fileio_test_folder2")
    shutil.rmtree("tests/fileio_test_folder2")

    bucket.delete_dir("", confirm=False)


@pytest.mark.skip
async def test_r2_bucket():
    bucket = R2Bucket("r2://esp-ci-cd-tests/esp-data-tests/temprandomfolder")
    bucket.upload_dir("tests/fileio_test_folder")
    assert bucket.exists
    assert len(bucket.list_files(recursive=True)) > 0
    bucket.delete_dir("tests/fileio_test_folder", confirm=False)

    await bucket.async_upload_dir("tests/fileio_test_folder")
    bucket.mkdir("tests/fileio_test_folder2")
    await bucket.async_move_dir("tests/fileio_test_folder", "tests/fileio_test_folder2")
    await bucket.async_download_to("tests/fileio_test_folder2")
    assert os.path.exists("tests/fileio_test_folder2")
    shutil.rmtree("tests/fileio_test_folder2")

    bucket.delete_dir("", confirm=False)
