import pytest
import yaml
from esp_data.io.read_utils import read_text, read_yaml
from esp_data.io import AnyPathT, anypath, filesystem_from_path


@pytest.fixture
def cloud_test_dir() -> AnyPathT:
    """Create a temporary directory in cloud storage for tests."""
    test_dir = "gs://esp-ci-cd-tests/esp-data-tests/temp_test_read_utils/"
    return anypath(test_dir)


def test_read_text(tmp_path):
    """Test reading text files."""
    test_file = tmp_path / "test_read.txt"
    test_file.write_text("Hello, World!")

    content = read_text(test_file)
    assert content == "Hello, World!"


def test_read_yaml(tmp_path):
    """Test reading YAML files."""
    test_file = tmp_path / "test_read.yaml"
    test_file.write_text(
        """
        key1: value1
        key2:
          - item1
          - item2
        """
    )

    content = read_yaml(test_file)
    assert content == {"key1": "value1", "key2": ["item1", "item2"]}

    # Test when yaml only has a list, string, or number at the top level
    list_file = tmp_path / "list_read.yaml"
    list_file.write_text(
        """
        - elem1
        - elem2
        """
    )
    content = read_yaml(list_file)
    assert content == ["elem1", "elem2"]


def test_read_yaml_failure(tmp_path):
    """Test reading invalid YAML files."""
    test_file = tmp_path / "invalid_read.yaml"
    test_file.write_text(
        """
        key1 value1
        key2:
          - item1
          - item2
        """
    )
    with pytest.raises(yaml.YAMLError):
        read_yaml(test_file)

    # Also test empty file
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("")
    with pytest.raises(ValueError):
        read_yaml(empty_file)


def test_reading_yaml_from_cloud_storage(cloud_test_dir):
    """Test reading YAML files from cloud storage."""
    test_file = cloud_test_dir / "test_read.yaml"
    fs = filesystem_from_path(test_file)
    with fs.open(str(test_file), "w") as f:
        f.write(
            """
            keyA: valueA
            keyB:
              - itemA
              - itemB
            """
        )

    content = read_yaml(test_file)
    assert content == {"keyA": "valueA", "keyB": ["itemA", "itemB"]}
