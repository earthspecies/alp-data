#!/usr/bin/env python3
"""Build the PIFSC PIPAN ``humpback-detection-val`` evaluation split.

Turns the held-out ``pifsc_pipan`` validation events into a *balanced binary
humpback-presence detection* task: each example is a fixed-length audio window
and the target is ``Megaptera novaeangliae`` if humpback song is present, else
``None``.

Why a derived split (and not the raw ``val`` events)
----------------------------------------------------
- Class balance: the raw val split is ~86% humpback (3,969 ``Mn`` vs 653
  non-``Mn``), so a model that always says "humpback" scores ~0.86 accuracy.
  We balance positives and negatives 1:1.
- Window standardisation: raw ``Mn`` strong events have a median duration of
  ~1 s while weak negatives (``Background``) span up to 75 s. Left unfixed the
  model could discriminate on clip length alone. Every clip here is a fixed
  ``--window-sec`` (default 10 s) window centred on the annotated event.
- Positive cleanliness: only *strong* ``Mn`` events (tight bounds) are used as
  positives so the centred window provably contains song. Weak/subchunk-level
  ``Mn`` is dropped from positives.
- Negatives: every non-``Mn`` label (``Background``/``Other``/``Vessel``/
  ``Fish``/``Device``) is humpback-absent for the purposes of this task.

The output CSV keeps the full ``pifsc_pipan`` schema (so the registered dataset
loader reads it unchanged) with two modifications:
- ``begin_in_file_s`` / ``end_in_file_s`` overwritten to the centred window.
- a new ``det_target`` column holding the chat target string.

Upload to GCS separately with::

    gsutil cp <output_dir>/pifsc_pipan_humpback_detection_val.csv \
        gs://esp-data-ingestion/pifsc-pipan/v0.1.0/

Usage::

    uv run python scripts/data_preprocessing_scripts/pifsc_pipan/\
build_pifsc_pipan_humpback_detection.py --output-dir data/pifsc_pipan_humpback_detection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

SEED = 17
SRC_VAL_CSV = "gs://esp-data-ingestion/pifsc-pipan/v0.1.0/pifsc_pipan_val.csv"
POSITIVE_LABEL = "Mn"
POSITIVE_TARGET = "Megaptera novaeangliae"
NEGATIVE_TARGET = "None"


def _centre_window(df: pl.DataFrame, window_sec: float) -> pl.DataFrame:
    """Overwrite begin/end_in_file_s with a fixed window centred on the event."""
    half = window_sec / 2.0
    mid = (pl.col("begin_in_file_s") + pl.col("end_in_file_s")) / 2.0
    ws = (mid - half).clip(lower_bound=0.0)
    return df.with_columns(
        ws.alias("begin_in_file_s"),
        (ws + window_sec).alias("end_in_file_s"),
    )


def _cap_per_file(df: pl.DataFrame, max_per_file: int, seed: int) -> pl.DataFrame:
    """Keep at most ``max_per_file`` rows per source audio file (seeded)."""
    if max_per_file <= 0:
        return df
    return (
        df.sample(fraction=1.0, shuffle=True, seed=seed)
        .with_columns(pl.int_range(pl.len()).over("audio_path").alias("_rank"))
        .filter(pl.col("_rank") < max_per_file)
        .drop("_rank")
    )


def build(
    output_dir: Path,
    window_sec: float,
    max_pos_per_file: int,
    max_neg_per_file: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_csv(SRC_VAL_CSV, infer_schema_length=5000)

    pos = df.filter((pl.col("label") == POSITIVE_LABEL) & (pl.col("label_is_strong")))
    neg = df.filter(pl.col("label") != POSITIVE_LABEL)

    pos = _cap_per_file(pos, max_pos_per_file, seed)
    neg = _cap_per_file(neg, max_neg_per_file, seed)

    # Balance 1:1 to whichever class is scarcer (negatives, here).
    n = min(pos.height, neg.height)
    pos = pos.sample(n=n, shuffle=True, seed=seed)
    neg = neg.sample(n=n, shuffle=True, seed=seed)

    pos = pos.with_columns(pl.lit(POSITIVE_TARGET).alias("det_target"))
    neg = neg.with_columns(pl.lit(NEGATIVE_TARGET).alias("det_target"))

    out = pl.concat([pos, neg], how="vertical").sample(fraction=1.0, shuffle=True, seed=seed)
    out = _centre_window(out, window_sec)

    out_csv = output_dir / "pifsc_pipan_humpback_detection_val.csv"
    out.write_csv(out_csv)

    stats = {
        "n_clips": out.height,
        "n_positive": int((out["det_target"] == POSITIVE_TARGET).sum()),
        "n_negative": int((out["det_target"] == NEGATIVE_TARGET).sum()),
        "window_sec": window_sec,
        "max_pos_per_file": max_pos_per_file,
        "max_neg_per_file": max_neg_per_file,
        "seed": seed,
        "n_unique_files": out["audio_path"].n_unique(),
        "n_deployments": out["deployment"].n_unique(),
        "negative_label_breakdown": out.filter(pl.col("det_target") == NEGATIVE_TARGET)
        .group_by("label")
        .len()
        .sort("label")
        .to_dicts(),
        "per_deployment": out.group_by("deployment", "det_target")
        .len()
        .sort("deployment", "det_target")
        .to_dicts(),
    }
    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nWrote {out_csv} ({out.height} clips)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--window-sec", type=float, default=10.0)
    ap.add_argument("--max-pos-per-file", type=int, default=3)
    ap.add_argument("--max-neg-per-file", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    build(
        output_dir=args.output_dir,
        window_sec=args.window_sec,
        max_pos_per_file=args.max_pos_per_file,
        max_neg_per_file=args.max_neg_per_file,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
