"Exporter functional api unit tests"

import pytest
import numpy as np
import io
import json
import soundfile as sf
from pathlib import Path
from typing import Any

from esp_data.io import anypath
from esp_data.exporters import _error_handler, _make_file_opener_for_wds, _write_webdataset_shard, export_as_tar
from esp_data.webdataset_utils import audio_decoder, json_encoder, audio_encoder, load_webdataset


@pytest.fixture
def audio_dataset_records() -> list[dict[str, Any]]:
    """Fixture to create a temporary audio dataset."""
    data = [
        {"audio": np.random.rand(16000).astype(np.float32), "text": "Sample text 1", "label": 0},
        {"audio": np.random.rand(16000).astype(np.float32), "text": "Sample text 2", "label": 1},
        {"audio": np.random.rand(16000).astype(np.float32), "text": "Sample text 3", "label": 2},
    ]

    return data


@pytest.fixture
def json_dataset_records() -> list[dict[str, Any]]:
    """Fixture to create a temporary JSON dataset."""
    data = [
        {"text": "Sample text 1", "label": 0},
        {"text": "Sample text 2", "label": 1},
        {"text": "Sample text 3", "label": 2},
    ]

    return data


@pytest.fixture
def webdataset_sharding_results(tmp_path: str) -> list[dict[str, Any]]:
    shard_path = tmp_path / "esp-data_webds_tests"
    data = [
        {"audio": np.random.rand(16000).astype(np.float32), "text": "Sample text 1", "label": 0},
        {"audio": np.random.rand(16000).astype(np.float32), "text": "Sample text 2", "label": 1},
        {"audio": np.random.rand(16000).astype(np.float32), "text": "Sample text 3", "label": 2},
    ]
    results = _write_webdataset_shard(data,
                                      shard_id=0,
                                      output_path=shard_path,
                                      sample_prep_function=audio_encoder,
                                      )
    return results


def test_json_encoder(json_dataset_records: list[dict[str, Any]]) -> None:
    """Test the JSON encoder function."""

    encoded_data = [json_encoder(record) for record in json_dataset_records]
    assert isinstance(encoded_data, list)
    assert len(encoded_data) == len(json_dataset_records)

    for i, item in enumerate(encoded_data):
        assert "sample.json" in item
        assert isinstance(item["sample.json"], bytes)
        # convert back
        metadata = json.loads(item["sample.json"].decode("utf-8"))
        assert metadata == json_dataset_records[i]


def test_audio_encoder(audio_dataset_records: list[dict[str, Any]]
) -> None:
    """Test the audio encoder function."""

    encoded_data = [audio_encoder(record) for record in audio_dataset_records]
    assert isinstance(encoded_data, list)
    assert len(encoded_data) == len(audio_dataset_records)

    for i, item in enumerate(encoded_data):
        assert "audio.flac" in item
        assert "metadata.json" in item
        assert isinstance(item["audio.flac"], bytes)
        # convert back
        audio_data, samplerate = sf.read(io.BytesIO(item["audio.flac"]), dtype='float32')
        assert audio_data.dtype == np.float32
        assert len(audio_data) == 16000
        assert samplerate == 16000
        # assert close to original audio data
        np.allclose(audio_data, audio_dataset_records[i]["audio"], atol=1e-5)


def test_error_handler() -> None:
    """Test the error handler function."""
    # Test with a valid exception
    try:
        raise ValueError("This is a test error")
    except Exception as e:
        with pytest.raises(ValueError):
            _error_handler(e, sample_id="test_sample", error_handling="raise")

    # Test with a non-exception type
    assert _error_handler("Not an exception", sample_id="test_sample", error_handling="ignore") is None

    # In case of error_handling=warn, logger in _error_handler will write a warning message
    # starting with `Error processing sample`


def test_make_file_opener_local(tmp_path) -> None:
    "Test the file opener function."
    test_file = tmp_path / "test_file.txt"
    with _make_file_opener_for_wds(test_file) as fp:
        fp.write(b"Hello, World!")

    # Check if the file was created and contains the expected content
    with test_file.open("rb") as f:
        content = f.read()
    assert content == b"Hello, World!"


@pytest.mark.parametrize(
    "cloud_path",
    [
        anypath("gs://esp-ci-cd-tests/esp-data-tests/test_make_file_opener.bin"),
    ],
)
def test_make_file_opener_cloud(cloud_path) -> None:
    """Test the file opener function for cloud paths."""
    with _make_file_opener_for_wds(cloud_path) as fp:
        fp.write(b"Hello, Cloud!")

    # Check if the file was created and contains the expected content
    with cloud_path.open("rb") as f:
        content = f.read()
    assert content == b"Hello, Cloud!"

    # Clean up
    cloud_path.unlink()


def test_write_webdataset_shard(webdataset_sharding_results, audio_dataset_records) -> None:
    """Test writing a webdataset shard."""
    assert len(webdataset_sharding_results["processed_ids"]) == len(audio_dataset_records)

    shard_path = webdataset_sharding_results["processed_ids"][-1]["shard_path"]
    # Check if the shard file was created
    assert Path(shard_path).exists()

    # Read the shard file to verify its content
    with open(shard_path, "rb") as f:
        content = f.read()

    assert len(content) > 0  # Ensure that the file is not empty


def test_load_webdataset(webdataset_sharding_results) -> None:
    """Test loading a webdataset."""
    parent_folder = Path(webdataset_sharding_results["processed_ids"][0]["shard_path"]).parent
    webdataset = load_webdataset(parent_folder, data_processor=audio_decoder)

    for _, sample in enumerate(webdataset):
        assert "audio" in sample
        assert isinstance(sample["audio"], np.ndarray)
        assert sample["audio"].dtype == np.float32
        assert len(sample["audio"]) == 16000
        assert "text" in sample


def test_export_as_tar(audio_dataset_records, tmp_path) -> None:
    """Test exporting a dataset as a tar file."""
    output_path = tmp_path / "test_export"
    export_as_tar(audio_dataset_records, output_path, sample_prep_function=audio_encoder)

    # Check if the tar file was created
    created_shard_path = Path(output_path) / "shard_000000.tar"
    assert created_shard_path.exists()

    # Read the tar file to verify its content
    with open(created_shard_path, "rb") as f:
        content = f.read()

    assert len(content) > 0  # Ensure that the file is not empty

    # Optionally, you can extract and check the contents of the tar file
    import tarfile
    with tarfile.open(created_shard_path, "r") as tar:
        members = tar.getmembers()
        assert len(members) == len(audio_dataset_records) * 2  # audio and metadata for each record
