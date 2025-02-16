from datetime import datetime

import pytest

from esp_data.utils import (
    increment_version,
    make_id,
    run_as_async,
    utc_now,
    validate_datetime,
    validate_id,
    validate_json_str,
    validate_path_exists,
    validate_version,
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


def test_validate_datetime():
    assert validate_datetime("2021-01-01T00:00:00") == datetime(2021, 1, 1, 0, 0, 0)
    with pytest.raises(ValueError):
        validate_datetime("invalid_datetime")

    with pytest.raises(ValueError):
        validate_datetime("2021-01-1")


def test_make_id():
    id = make_id()
    assert validate_id(id) == id


async def test_run_as_async():
    def add(a, b):
        return a + b

    result = await run_as_async(add, a=1, b=2)
    assert result == 3
