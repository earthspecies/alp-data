"""Offline sample-rate metadata checks for annotated audio loaders."""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from alp_data.datasets.anuraset import AnuraSetStrong
from alp_data.datasets.xeno_canto_annotated_jeantet_23 import XenoCantoAnnotatedJeantet23


@pytest.fixture(params=[AnuraSetStrong, XenoCantoAnnotatedJeantet23])
def dataset_class(request):
    return request.param


@pytest.fixture(params=["pandas", "polars"])
def backend(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["indexed", "streaming"])
def streaming(request):
    return request.param


@pytest.fixture
def load_item(tmp_path, monkeypatch, dataset_class, backend, streaming):
    """Read a synthetic local manifest through the public dataset interface."""

    def load(sample_rate, presampled_rate=None, manifest_rate=None, mapping=None):
        file_rate = presampled_rate or 8000
        audio = np.tile(np.array([0.125, 0.375], dtype=np.float32), (file_rate // 4, 1))
        filename = "presampled.wav" if presampled_rate else "original.wav"
        sf.write(tmp_path / filename, audio, file_rate, subtype="FLOAT")
        selection_table = pd.DataFrame(
            {"Begin Time (s)": [0.05], "End Time (s)": [0.15], "Species": ["test_species"]}
        )
        row = {
            "audio_path": "original.wav",
            "selection_table": selection_table.to_csv(sep="\t", index=False),
        }
        if presampled_rate is not None:
            # The original is deliberately absent, so falling back to it fails.
            row[f"{sample_rate // 1000}khz_path"] = filename
        if manifest_rate is not None:
            row["sample_rate"] = manifest_rate
        manifest = tmp_path / "manifest.csv"
        pd.DataFrame([row]).to_csv(manifest, index=False)
        monkeypatch.setattr(dataset_class.info, "split_paths", {"all": str(manifest)})
        dataset = dataset_class(
            sample_rate=sample_rate,
            data_root=tmp_path,
            backend=backend,
            streaming=streaming,
            output_take_and_give=mapping,
        )
        item = next(iter(dataset)) if streaming else dataset[0]
        return item, selection_table

    return load


@pytest.mark.parametrize("manifest_rate", [None, 44100], ids=["missing", "stale"])
@pytest.mark.parametrize(
    "sample_rate,presampled_rate,expected_rate",
    [
        pytest.param(None, None, 8000, id="native"),
        pytest.param(8000, None, 8000, id="matching-native"),
        pytest.param(16000, None, 16000, id="resampled-without-presampled-path"),
        pytest.param(16000, 16000, 16000, id="presampled-16khz"),
        pytest.param(32000, 32000, 32000, id="presampled-32khz"),
        pytest.param(16000, 12000, 12000, id="presampled-actual-file-rate"),
    ],
)
def test_effective_sample_rate(
    load_item, sample_rate, presampled_rate, expected_rate, manifest_rate
):
    item, selection_table = load_item(sample_rate, presampled_rate, manifest_rate)

    assert item["sample_rate"] == expected_rate
    assert item["audio"].shape == (expected_rate // 4,)
    assert item["audio"].dtype == np.float32
    assert np.isfinite(item["audio"]).all()
    assert len(item["audio"]) / item["sample_rate"] == 0.25
    pd.testing.assert_frame_equal(item["selection_table"], selection_table)


@pytest.mark.parametrize("manifest_rate", [None, 44100], ids=["missing", "stale"])
def test_sample_rate_output_mapping(load_item, manifest_rate):
    item, selection_table = load_item(
        12000,
        manifest_rate=manifest_rate,
        mapping={"audio": "waveform", "sample_rate": "sr", "selection_table": "annotations"},
    )

    assert set(item) == {"waveform", "sr", "annotations"}
    assert item["sr"] == 12000
    assert item["waveform"].shape == (3000,)
    pd.testing.assert_frame_equal(item["annotations"], selection_table)
