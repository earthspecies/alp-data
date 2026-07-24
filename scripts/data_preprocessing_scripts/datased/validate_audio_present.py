"""Validate that every DataSED manifest references audio present on GCS.

Checks that each recording in the mono/poly manifests has an object at
``audio/``, ``audio_16k/`` and ``audio_32k/`` on GCS, and that the
selection tables parse. Lightweight — reads only the manifest CSVs plus
three ``gsutil ls`` listings.

    uv run python scripts/data_preprocessing_scripts/datased/validate_audio_present.py
"""

from __future__ import annotations

import subprocess
import sys
from io import StringIO

import pandas as pd

GCS_ROOT = "gs://esp-data-ingestion/datased/v0.1.0"
MANIFESTS = [
    f"{GCS_ROOT}/datased_mono_all.csv",
    f"{GCS_ROOT}/datased_poly_all.csv",
]
AUDIO_DIRS = ["audio", "audio_16k", "audio_32k"]


def _basenames(prefix: str) -> set[str]:
    """Return the set of object basenames under a GCS prefix.

    Returns
    -------
    set[str]
        Basenames (e.g. ``S-0001.wav``) found directly under ``prefix``.
    """
    out = subprocess.run(
        ["gsutil", "ls", f"{prefix}/"], capture_output=True, text=True, check=True
    ).stdout
    return {line.rsplit("/", 1)[-1] for line in out.splitlines() if line.strip().endswith(".wav")}


def main() -> None:
    """Cross-check manifest references against GCS listings."""
    listings = {}
    for d in AUDIO_DIRS:
        listings[d] = _basenames(f"{GCS_ROOT}/{d}")
        print(f"{d}/: {len(listings[d])} wavs on GCS")

    problems = 0
    st_errors = 0
    for man in MANIFESTS:
        blob = subprocess.run(["gsutil", "cat", man], capture_output=True, text=True, check=True).stdout
        df = pd.read_csv(StringIO(blob), keep_default_na=False, na_values=[""])
        print(f"\n{man.rsplit('/', 1)[-1]}: {len(df)} recordings")
        for _, row in df.iterrows():
            name = str(row["sound_name"])
            for d in AUDIO_DIRS:
                if name not in listings[d]:
                    print(f"  MISSING {d}/{name}")
                    problems += 1
            try:
                st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
                assert {"Begin Time (s)", "End Time (s)", "Label"} <= set(st.columns)
            except Exception as exc:  # noqa: BLE001
                print(f"  BAD selection_table for {name}: {exc}")
                st_errors += 1

    print(f"\nmissing-audio problems: {problems} | selection-table errors: {st_errors}")
    if problems or st_errors:
        sys.exit(1)
    print("OK: all manifest audio present on GCS and selection tables parse.")


if __name__ == "__main__":
    main()
