"""Tests for pydantic configs for database"""

import pytest
import json
from datetime import datetime
from esp_data.config.db_config import DataSample, TextDataSample


def test_correct_datasample():
    """Test correct DataSample."""
    data = {
        "dataset_name": "test",
        "creator": "test",
        "metadata": json.dumps({"something": "else"}),
        "version": "0.0.0",
    }
    sample = DataSample(**data)
    assert sample.id is not None
    assert sample.created_at is not None
    assert isinstance(sample.created_at, datetime)
    assert sample.dataset_name == "test"
    assert sample.creator == "test"


def test_underspecified_datasample():
    """Test underspecified DataSample."""
    # missing version, metadata
    data = {
        "dataset_name": "test",
        "creator": "test",
    }
    with pytest.raises(ValueError):
        DataSample(**data)


def test_wrongly_specified_datasample():
    """Test wrongly specified DataSample."""
    data = {
        "dataset_name": "test",
        "creator": "test",
        "metadata": "not json",
    }
    with pytest.raises(ValueError):
        DataSample(**data)


def test_wrong_version():
    """Test wrong version."""
    data = {
        "dataset_name": "test",
        "creator": "test",
        "metadata": json.dumps({"something": "else"}),
        "version": "0.0",
    }
    with pytest.raises(ValueError):
        DataSample(**data)


def test_text_datasample():
    """Test TextDataSample."""
    data = {
        "dataset_name": "test",
        "creator": "test",
        "metadata": json.dumps({"something": "else"}),
        "version": "0.0.0",
        "text": "This is a test text",
    }
    sample = TextDataSample(**data)
    assert sample.id is not None
    assert sample.created_at is not None
    assert sample.dataset_name == "test"
    assert sample.creator == "test"
    assert sample.text == "This is a test text"
    assert sample.version == "0.0.0"
