"""
Build the DORI Phase-1 split CSVs from the ingestion manifest.

Each row is one ~15 s clip (clip-level classification, not detection). Audio
lives in gs://esp-data-ingestion/dori/v0.1.0/{recordings,audio_16k,audio_32k}/.
Writes per-split CSVs (all / train / test / onc / orcasound / ooi) plus
species_labels.csv. Clips that failed crop-download are excluded.

Usage:
    uv run python scripts/data_preprocessing_scripts/dori_build_csv.py
"""

from __future__ import annotations

import argparse
import os
import subprocess

import pandas as pd

GCS_ROOT = "gs://esp-data-ingestion/dori/v0.1.0"
COLUMNS = [
    "clip_id",
    "source",
    "audio_fp",
    "16khz_path",
    "32khz_path",
    "audio_duration",
    "species",
    "species_common",
    "ecotype",
    "call_type",
    "presence",
    "label_source",
    "license",
    "is_negative",
]


def gcs_present(gcs_root: str) -> set[tuple[str, str]]:
    """Return {(source, clip_id)} for every 32 kHz clip actually present in GCS.

    Returns
    -------
    set[tuple[str, str]]
        Source + clip_id of each ``audio_32k/<source>/<clip_id>.wav`` found.
    """
    present: set[tuple[str, str]] = set()
    for s in ("onc", "orcasound", "ooi"):
        r = subprocess.run(
            ["gsutil", "ls", f"{gcs_root}/audio_32k/{s}/"],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        for line in r.stdout.splitlines():
            if line.endswith(".wav"):
                present.add((s, line.rsplit("/", 1)[-1][:-4]))
    return present


def main() -> None:
    p = argparse.ArgumentParser()
    _default_manifest = os.path.expanduser("~/dori_staging/dori_phase1_manifest.csv")
    p.add_argument("--manifest", default=_default_manifest)
    p.add_argument("--out-dir", default=os.path.expanduser("~/dori_staging/csv"))
    p.add_argument("--gcs-root", default=GCS_ROOT)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    m = pd.read_csv(args.manifest, keep_default_na=False)
    n0 = len(m)
    # keep only clips whose 32 kHz audio actually landed in GCS
    present = gcs_present(args.gcs_root)
    print(f"clips present in GCS (audio_32k): {len(present)}")
    keys = list(zip(m["source"].astype(str), m["clip_id"].astype(str), strict=False))
    m = m[[k in present for k in keys]].reset_index(drop=True)
    # one clip per (source, clip_id): a few files had multiple segments but only
    # one window was cropped/uploaded; keep the first labelled row.
    m = m.drop_duplicates(subset=["source", "clip_id"], keep="first").reset_index(drop=True)
    print(f"manifest {n0} -> kept {len(m)} clips (dropped {n0 - len(m)})")

    src = m["source"]
    cid = m["clip_id"].astype(str)
    se = pd.to_numeric(m["segment_end"], errors="coerce")
    ss = pd.to_numeric(m["segment_start"], errors="coerce")
    dur = (se - ss).clip(lower=1.0).fillna(15.0)
    out = pd.DataFrame(
        {
            "clip_id": cid,
            "source": src,
            "audio_fp": "recordings/" + src + "/" + cid + ".flac",
            "16khz_path": "audio_16k/" + src + "/" + cid + ".wav",
            "32khz_path": "audio_32k/" + src + "/" + cid + ".wav",
            "audio_duration": dur.round(3),
            "species": m["species"],
            "species_common": m["species_common"],
            "ecotype": m["ecotype"],
            "call_type": m["call_type"],
            "presence": (~m["is_negative"].astype(bool)).astype(int),
            "label_source": m["label_source"],
            "license": m["license"],
            "is_negative": m["is_negative"].astype(bool),
        }
    )[COLUMNS]
    split = m["split"].astype(str)

    def _save(df: pd.DataFrame, name: str) -> None:
        local = os.path.join(args.out_dir, name)
        df.to_csv(local, index=False)
        os.system(f"gsutil -q cp {local} {args.gcs_root}/{name}")
        print(f"  {name}: {len(df)} rows")

    _save(out, "all.csv")
    _save(out[split == "train"].reset_index(drop=True), "train.csv")
    _save(out[split == "test"].reset_index(drop=True), "test.csv")
    for s in ["onc", "orcasound", "ooi"]:
        _save(out[out["source"] == s].reset_index(drop=True), f"{s}.csv")

    labels = sorted({x for x in out["species"] if x})
    _save(pd.DataFrame({"species": labels}), "species_labels.csv")

    print(
        f"\nTOTAL clips: {len(out)} | positives: {int(out['presence'].sum())} "
        f"| negatives: {int((~out['presence'].astype(bool)).sum())}"
    )
    print("species (canonical):\n", out["species"].replace("", "(none)").value_counts().to_string())


if __name__ == "__main__":
    main()
