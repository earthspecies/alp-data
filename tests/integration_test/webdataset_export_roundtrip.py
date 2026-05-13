"""Integration roundtrip test: BeansZero esc50 → webdataset shards → GenericDataset."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from esp_data import Dataset
from esp_data.backends.webdataset_utils import audio_decoder
from esp_data.datasets import BeansZero

SAMPLE_RATE = 16000
N_AUDIO_CHECK = 3  # Full-array comparison on first N samples only (rest check shape/sr)


@pytest.fixture(scope="module")
def esc50_ds() -> BeansZero:
    """Load BeansZero esc50 once for all tests in this module.

    Returns
    -------
    BeansZero
        Dataset instance for the esc50 split at 16 kHz.
    """
    return BeansZero(split="esc50", sample_rate=SAMPLE_RATE)


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory: pytest.TempPathFactory, esc50_ds: BeansZero) -> Path:
    """Export esc50 to webdataset shards and return the output directory.

    Returns
    -------
    Path
        Directory containing the written shard tar files.
    """
    out = tmp_path_factory.mktemp("esc50_wds")
    esc50_ds.save_to(str(out), backend="webdataset", format="webdataset")
    return out


def test_shard_files_created(export_dir: Path) -> None:
    """At least one shard tar file must exist after export."""
    shards = list(export_dir.glob("shard_*.tar"))
    assert len(shards) >= 1


def test_info_yaml_created(export_dir: Path) -> None:
    """save_to must write info.yaml alongside the shard files."""
    assert (export_dir / "info.yaml").exists()


def test_info_yaml_content(esc50_ds: BeansZero, export_dir: Path) -> None:
    """info.yaml must contain correct fields, single split entry, and sample_rate."""
    with open(export_dir / "info.yaml") as f:
        info = yaml.safe_load(f)

    assert info["name"] == esc50_ds.info.name
    assert info["version"] == esc50_ds.info.version
    assert info["owner"] == esc50_ds.info.owner

    # Only the exported split must appear, pointing at the export directory.
    assert list(info["split_paths"].keys()) == ["esc50"]
    assert info["split_paths"]["esc50"] == str(export_dir)

    # Sample rate used during export must be recorded.
    assert info["sample_rate"] == SAMPLE_RATE


def test_info_loaded_on_from_path(esc50_ds: BeansZero, export_dir: Path) -> None:
    """Dataset.from_path must populate info from info.yaml."""
    reloaded = Dataset.from_path(export_dir, data_processor=audio_decoder)
    assert reloaded.info.name == esc50_ds.info.name
    assert reloaded.info.version == esc50_ds.info.version
    assert list(reloaded.info.split_paths.keys()) == ["esc50"]


def test_sample_count_preserved(esc50_ds: BeansZero, export_dir: Path) -> None:
    """Reloaded dataset must have exactly as many samples as the source."""
    reloaded = Dataset.from_path(export_dir, data_processor=audio_decoder)
    n_reloaded = sum(1 for _ in reloaded)
    assert n_reloaded == len(esc50_ds)


def test_audio_shape_and_sample_rate(esc50_ds: BeansZero, export_dir: Path) -> None:
    """Every reloaded sample must have correct audio shape and sample rate."""
    reloaded = Dataset.from_path(export_dir, data_processor=audio_decoder)
    for i, reloaded_sample in enumerate(reloaded):
        assert "audio" in reloaded_sample
        assert isinstance(reloaded_sample["audio"], np.ndarray)
        assert reloaded_sample["sample_rate"] == SAMPLE_RATE

        orig = esc50_ds[i]
        assert reloaded_sample["audio"].shape == orig["audio"].shape


def test_audio_values_preserved(esc50_ds: BeansZero, export_dir: Path) -> None:
    """FLAC is lossless — first N samples must round-trip with tight tolerance."""
    reloaded = Dataset.from_path(export_dir, data_processor=audio_decoder)
    for i, reloaded_sample in enumerate(reloaded):
        if i >= N_AUDIO_CHECK:
            break
        orig = esc50_ds[i]
        np.testing.assert_allclose(
            reloaded_sample["audio"],
            orig["audio"],
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"Audio mismatch at sample {i}",
        )


def test_metadata_fields_preserved(esc50_ds: BeansZero, export_dir: Path) -> None:
    """Non-audio metadata fields must survive the encode/decode cycle."""
    reloaded = Dataset.from_path(export_dir, data_processor=audio_decoder)
    orig_sample = esc50_ds[0]
    reloaded_sample = next(iter(reloaded))

    metadata_keys = [k for k in orig_sample if k not in ("audio", "sample_rate")]
    for key in metadata_keys:
        assert key in reloaded_sample, f"Metadata key '{key}' missing after roundtrip"
        assert reloaded_sample[key] == orig_sample[key], (
            f"Metadata mismatch for key '{key}': "
            f"{reloaded_sample[key]!r} != {orig_sample[key]!r}"
        )
