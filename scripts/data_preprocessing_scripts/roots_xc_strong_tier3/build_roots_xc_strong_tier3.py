"""Rewrite XC-strong Tier-3 conversations into ROOTS-native form.

The synthesized XC-strong Tier-3 tasks at
``gs://foundation-model-data/synthetic/xc_strong_tier3_20260709/<task>/conversations.jsonl``
reference audio by a *relative* path (``audio_16k/XC*.wav``) and a clip id that
encodes only the window **start** (``XC*.wav#<start_sec>``); the window **end**
lives in a sidecar (``xc_strong_human_multispecies.jsonl``, keyed by that clip
id). The ``ROOTS`` loader instead expects an absolute ``audio_paths`` entry and a
crop window encoded as ``__crop_<start_ms>_<end_ms>`` in ``audio_ids``.

This script joins each task's conversations to the sidecar bounds and rewrites
every row to ROOTS-native form:

* ``audio_paths[0]`` -> absolute ``gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/<16khz_path>``
* ``audio_ids[0]``   -> ``<clip_id>__crop_<start_ms>_<end_ms>``

so ``ROOTS`` reads the exact window of the (already-on-GCS) XC recording — no
audio materialization needed. Output goes to ``<src>/roots_native/<task>.jsonl``.

Run on a host with GCS access (e.g. ``ssh slurm-login``); stdlib + ``gsutil`` only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile

SRC_ROOT = "gs://foundation-model-data/synthetic/xc_strong_tier3_20260709"
OUT_ROOT = f"{SRC_ROOT}/roots_native"
XC_AUDIO_ROOT = "gs://esp-data-ingestion/xeno-canto/v0.1.0/raw"
# Superset window table (43,716 windows); covers every task's clip ids
# (the multispecies-only file is a strict subset).
DEFAULT_SIDECAR = os.path.expanduser(
    "~/esp-research-bn/projects/NatureLM-audio-data-synth/output/xc_strong_human_windows.jsonl"
)

TASK_DIRS = (
    "highest_pitch_species_mcq_xcstrong_v1_clean",
    "highest_pitch_species_oe_xcstrong_v1_clean",
    "lowest_pitch_species_mcq_xcstrong_v1_clean",
    "lowest_pitch_species_oe_xcstrong_v1_clean",
    "longest_voc_species_mcq_xcstrong_v1_clean",
    "longest_voc_species_oe_xcstrong_v1_clean",
    "species_voc_order_mcq_xcstrong_v1_clean",
    "species_voc_order_oe_xcstrong_v1_clean",
    "tier1_structural_caption_xcstrong_v1_clean",
    "voc_cooccurrence_binary_xcstrong_v1_clean",
    "voc_count_relative_mcq_xcstrong_v1_clean",
    "vocal_dominance_mcq_xcstrong_v1_clean",
    "vocal_dominance_oe_xcstrong_v1_clean",
)


def _load_bounds(sidecar: str) -> dict[str, tuple[int, int, str]]:
    """Map ``clip_id`` -> (start_ms, end_ms, 16khz relative path).

    Returns
    -------
    dict[str, tuple[int, int, str]]
        Per clip id: window start/end in milliseconds and the 16 kHz relative path.
    """
    bounds: dict[str, tuple[int, int, str]] = {}
    with open(sidecar) as f:
        for line in f:
            r = json.loads(line)
            bounds[r["clip_id"]] = (
                round(float(r["window_start_sec"]) * 1000),
                round(float(r["window_end_sec"]) * 1000),
                str(r["16khz_path"]),
            )
    return bounds


def _rewrite_task(task: str, bounds: dict[str, tuple[int, int, str]], upload: bool) -> None:
    src = f"{SRC_ROOT}/{task}/conversations.jsonl"
    txt = subprocess.run(["gsutil", "cat", src], check=True, capture_output=True, text=True).stdout
    kept = miss = 0
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as out:
        for line in txt.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row["audio_ids"][0]
            b = bounds.get(cid)
            if b is None:
                miss += 1
                continue
            start_ms, end_ms, rel16 = b
            row["audio_paths"] = [f"{XC_AUDIO_ROOT}/{rel16}"]
            row["audio_ids"] = [f"{cid}__crop_{start_ms}_{end_ms}"]
            out.write(json.dumps(row) + "\n")
            kept += 1
        tmp = out.name
    print(f"[{task}] kept={kept:,} dropped_no_bounds={miss}", flush=True)
    if upload:
        dest = f"{OUT_ROOT}/{task}.jsonl"
        subprocess.run(["gsutil", "-q", "cp", tmp, dest], check=True)
        print(f"  -> {dest}")
    os.unlink(tmp)


def main() -> None:
    """Entry point — see module docstring."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sidecar", default=DEFAULT_SIDECAR)
    p.add_argument("--upload", action="store_true")
    p.add_argument(
        "--upload-sidecar", action="store_true", help="copy sidecar to GCS for provenance"
    )
    args = p.parse_args()

    bounds = _load_bounds(args.sidecar)
    print(f"sidecar windows: {len(bounds):,}", flush=True)
    for task in TASK_DIRS:
        _rewrite_task(task, bounds, args.upload)
    if args.upload_sidecar:
        dest = f"{SRC_ROOT}/{os.path.basename(args.sidecar)}"
        subprocess.run(["gsutil", "-q", "cp", args.sidecar, dest], check=True)
        print(f"sidecar -> {dest}")


if __name__ == "__main__":
    main()
