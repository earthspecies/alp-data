import pytest
from esp_data.utils import (
    increment_version,
    utc_now,
    validate_json_str,
    validate_path_exists,
    validate_version,
    is_local_path,
    is_gcs_path,
)


def test_increment_version():
    assert increment_version("1.0.0", "major") == "2.0.0"
    assert increment_version("1.0.0", "minor") == "1.1.0"
    assert increment_version("1.0.0", "patch") == "1.0.1"
    with pytest.raises(ValueError):
        increment_version("1.0.0", "invalid")

    with pytest.raises(ValueError):
        increment_version("1.0", "patch")


def test_utc_now():
    assert utc_now().tzinfo is not None


def test_validate_json_str():
    assert validate_json_str('{"key": "value"}') == '{"key": "value"}'
    with pytest.raises(ValueError):
        validate_json_str("not json")


def test_validate_path_exists(tmp_path):
    assert validate_path_exists(tmp_path) == str(tmp_path)
    with pytest.raises(ValueError):
        validate_path_exists("invalid_path")


def test_validate_version():
    assert validate_version("1.0.0") == "1.0.0"
    with pytest.raises(ValueError):
        validate_version("1.0")
    with pytest.raises(ValueError):
        validate_version("invalid_version")


def test_is_local_path(tmp_path):
    assert is_local_path(tmp_path)
    assert is_local_path(str(tmp_path))
    assert not is_local_path("gs://bucket/file.txt")


def test_is_gcs_path():
    assert is_gcs_path("gs://bucket/file.txt")
    assert not is_gcs_path("s3://bucket/file.txt")
    assert not is_gcs_path("file.txt")
