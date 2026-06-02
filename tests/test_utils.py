import pytest

from alp_data.utils import (
    cached_class_property,
    increment_version,
    make_id,
    utc_now,
    validate_datetime,
    validate_id,
    validate_json_str,
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


def test_validate_version():
    assert validate_version("1.0.0") == "1.0.0"
    with pytest.raises(ValueError):
        validate_version("1.0")
    with pytest.raises(ValueError):
        validate_version("invalid_version")


def test_validate_datetime():
    with pytest.raises(ValueError):
        validate_datetime("invalid_datetime")

    with pytest.raises(ValueError):
        validate_datetime("2021-01-1")
    t = utc_now().isoformat()
    assert validate_datetime(t) == t


def test_make_id():
    id = make_id()
    assert validate_id(id) == id


def test_cached_class_property_print_output(capsys):
    class PrintTestClass:
        @cached_class_property
        def expensive_property(cls):
            print("computing expensive value")
            return "expensive_value"

    # First access should print the message
    test_instance = PrintTestClass()
    assert test_instance.expensive_property == "expensive_value"
    captured = capsys.readouterr()
    assert "computing expensive value" in captured.out

    # Second access should not print the message
    assert test_instance.expensive_property == "expensive_value"
    captured = capsys.readouterr()
    assert "computing expensive value" not in captured.out

    # Access through another instance should not print the message
    another_instance = PrintTestClass()
    assert another_instance.expensive_property == "expensive_value"
    captured = capsys.readouterr()
    assert "computing expensive value" not in captured.out

    # Access through class should not print the message
    assert PrintTestClass.expensive_property == "expensive_value"
    captured = capsys.readouterr()
    assert "computing expensive value" not in captured.out


def test_cached_class_property_without_instantiation(capsys):
    class DirectAccessClass:
        @cached_class_property
        def expensive_property(cls):
            print("computing expensive value")
            return "expensive_value"

    # Access directly through class without any instances
    assert DirectAccessClass.expensive_property == "expensive_value"
    captured = capsys.readouterr()
    assert "computing expensive value" in captured.out

    # Second access should not print the message
    assert DirectAccessClass.expensive_property == "expensive_value"
    captured = capsys.readouterr()
    assert "computing expensive value" not in captured.out

    # Now create an instance and verify it uses the cached value
    instance = DirectAccessClass()
    assert instance.expensive_property == "expensive_value"
    captured = capsys.readouterr()
    assert "computing expensive value" not in captured.out
