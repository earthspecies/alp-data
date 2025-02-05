"""Tests for pydantic configs for database"""

import json
import os
from datetime import datetime

import pytest

from esp_data.config.db_config import DataSample, DatasetConfig, TextDataSample


def test_correct_datasample():
    """Test correct DataSample."""
    data = {
        "source_dataset": "test",
        "metadata": {"something": "else"},
    }
    sample = DataSample(**data)
    assert sample.id is not None
    assert sample.created_at is not None
    assert isinstance(sample.metadata, dict)
    assert isinstance(sample.created_at, datetime)
    assert sample.source_dataset == "test"


def test_underspecified_datasample():
    """Test underspecified DataSample."""
    # missing version, metadata
    data = {
        "source_dataset": "test",
        "creator": "test",
    }
    with pytest.raises(ValueError):
        DataSample(**data)


def test_wrongly_specified_datasample():
    """Test wrongly specified DataSample."""
    data = {
        "source_dataset": "test",
        "creator": "test",
        "metadata": ["something", "else"],
    }
    with pytest.raises(ValueError):
        DataSample(**data)


def test_wrong_version():
    """Test wrong version format"""
    data = {
        "source_dataset": "test",
        "creator": "test",
        "metadata": {"something": "else"},
        "version": "0.0",
    }
    with pytest.raises(ValueError):
        DataSample(**data)


def test_text_datasample():
    """Test TextDataSample."""
    data = {
        "source_dataset": "test",
        "creator": "test",
        "metadata": {"something": "else"},
        "version": "0.0.0",
        "text": "This is a test text",
    }
    sample = TextDataSample(**data)
    assert sample.id is not None
    assert sample.created_at is not None
    assert sample.source_dataset == "test"
    assert sample.text == "This is a test text"


def test_dataset_config():
    """Test DatasetConfig."""
    data = {
        "name": "test",
        "creator": "test",
        "version": "0.0.0",
        "description": "test",
        "sources": "test",
        "license": "CC-BY-4.0",
    }
    dataset = DatasetConfig(**data)
    assert dataset.created_at is not None
    assert dataset.name == "test"
    assert dataset.creator == "test"
    assert dataset.sources == "test"
    assert dataset.license == "CC-BY-4.0"


def test_utility_methods():
    """Test utility methods."""
    data = {
        "source_dataset": "test",
        "creator": "test",
        "metadata": {"something": "else"},
        "version": "0.0.0",
    }
    sample = DataSample(**data)
    assert sample.created_at_timestamp() == int(sample.created_at.timestamp())
    assert sample.created_at_isoformat() == sample.created_at.isoformat()
    assert sample.get_metadata_dict() == {"something": "else"}
    sample.update_metadata({"new": "metadata"})
    assert sample.metadata == {"something": "else", "new": "metadata"}
    sample.increment_version()
    assert sample.version == "0.0.1"
    assert sample.to_dict() == sample.model_dump()
    assert isinstance(sample.to_json(), str)
    sample.write_json("tests/test_data_sample.json")
    with open("tests/test_data_sample.json", "r") as f:
        assert json.load(f) == json.loads(sample.to_json())
    new_sample = DataSample.from_json("tests/test_data_sample.json")
    assert new_sample.to_dict() == sample.to_dict()
    os.remove("tests/test_data_sample.json")
