#!/usr/bin/env bash
#SBATCH --job-name=datased-validate
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output="/home/%u/logs/%A_%x.log"
#SBATCH --qos=naturelm

# Validate the DataSED ingest on GCS: (1) every manifest recording has
# audio present at audio/ + audio_16k/ + audio_32k/, and (2) the DataSED
# class loads, decodes audio, and parses selection tables at 16k + 32k,
# including a windowed read. Runs on Slurm to use the node attached SA.
set -euo pipefail
cd "${HOME}/esp-data-dev"
export CLOUDSDK_CONFIG="$(mktemp -d)"   # ignore stale user ADC; use attached SA (gsutil + gcsfs)

echo "=== 1. audio-present cross-check ==="
uv run python scripts/data_preprocessing_scripts/datased/validate_audio_present.py

echo "=== 2. dataset smoke-load ==="
uv run python - <<'PY'
import numpy as np
from esp_data.datasets import DataSED

labels = None
for split in ["poly_val", "mono_val"]:
    for sr in [16000, 32000]:
        ds = DataSED(split=split, sample_rate=sr)
        assert len(ds) > 0, f"{split} empty"
        it = ds[0]
        a, st = it["audio"], it["selection_table"]
        assert isinstance(a, np.ndarray) and a.ndim == 1 and a.size > 0, "bad audio"
        assert it["sample_rate"] == sr, f"sr mismatch {it['sample_rate']} != {sr}"
        assert list(st.columns) == ["Selection", "Begin Time (s)", "End Time (s)", "Label"]
        print(f"{split}@{sr}: n={len(ds)} audio={a.shape} dur={a.size/sr:.1f}s events={len(st)}")
    labels = ds.get_available_labels()

# windowed read path (as window_annotations would drive it)
ds = DataSED(split="poly_val", sample_rate=32000)
row = dict(ds._data[0])
row["window_start_sec"], row["window_end_sec"] = 0.0, 5.0
out = ds._process(row)
assert abs(out["audio"].size / out["sample_rate"] - 5.0) < 0.2, "windowed dur wrong"
print(f"windowed 5s read -> {out['audio'].size} samples ({out['audio'].size/out['sample_rate']:.2f}s)")

print(f"labels ({len(labels)}): {labels[:5]} ...")
print("SMOKE OK")
PY
echo "Done."
