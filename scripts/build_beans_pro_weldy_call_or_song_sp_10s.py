#!/usr/bin/env python3
"""Build ``weldy-call-or-song-sp-10s``: 10-s context + species-conditioned prompt.

Same row set as ``weldy-call-or-song`` (read from the deployed 2-s JSONL),
each segment is now a 10-s contiguous slice — target 2-s window plus 4-s
flanks on each side, silence-padded at recording boundaries. Prompts are
species-conditioned, matching ``weldy-call-or-song-sp`` semantics:

    "Is the {species_common} making a call or a song in this recording?"

Each row's ``metadata.flank_annotations`` carries pre_4s / post_4s
co-occurrence summaries for post-eval leakage stratification.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import uuid
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path

import fsspec
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


WELDY_GCS_ROOT = "gs://esp-data-ingestion/weldy_dawn_chorus/v0.1.0"
WELDY_LABELED_CSV = f"{WELDY_GCS_ROOT}/labeled.csv"
EXISTING_TIGHT_JSONL = (
    "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/weldy_call_or_song/test.jsonl"
)
TARGET_SR = 32_000
FLANK_S = 4.0          # 4 s of real audio each side
SEGMENT_S = 10.0       # total: 4 + 2 + 4
LICENSE_STR = "CC-BY-4.0"
SOURCE_DATASET = "weldy_dawn_chorus"
DATASET_NAME = "weldy-call-or-song-sp-10s"
TASK = "call_or_song_with_species_classification"

_VARIANT_RE = re.compile(r"_(\d+)$")


def _canonical_sonotype(raw: str) -> str | None:
    if not raw:
        return None
    s = _VARIANT_RE.sub("", str(raw).strip().lower())
    return s if s in ("call", "song") else None


def _species_prompt(species_common: str, species_scientific: str) -> str:
    name = species_common.strip() if species_common else species_scientific.strip()
    if not name:
        name = "focal species"
    return f"Is the {name} making a call or a song in this recording?"


def _read_jsonl_gcs(uri: str) -> list[dict]:
    fs = fsspec.filesystem("gs")
    _, stripped = uri.split("://", 1)
    with fs.open(stripped, "r") as fh:
        return [json.loads(line) for line in fh]


def _load_weldy_audio_full(audio_32k_path: str, fs: fsspec.AbstractFileSystem) -> np.ndarray:
    full_uri = f"{WELDY_GCS_ROOT}/{audio_32k_path}"
    _, stripped = full_uri.split("://", 1)
    with fs.open(stripped, "rb") as fh:
        audio, _sr = librosa.load(io.BytesIO(fh.read()), sr=TARGET_SR, mono=True)
    return audio.astype(np.float32, copy=False)


def _cut_with_pad(audio: np.ndarray, begin_s: float, end_s: float) -> np.ndarray:
    sr = TARGET_SR
    target_len = int(round(SEGMENT_S * sr))
    a_t = int(round((begin_s - FLANK_S) * sr))
    b_t = int(round((end_s + FLANK_S) * sr))
    n = audio.shape[-1]
    a = max(0, a_t)
    b = min(n, b_t)
    seg = audio[a:b]
    out = np.zeros(target_len, dtype=np.float32)
    out_start = max(0, -a_t)
    out[out_start:out_start + seg.shape[-1]] = seg
    peak = float(np.max(np.abs(out)) or 1.0)
    return (out / peak * 0.97).astype(np.float32)


def _summarise_flanks(
    st: pd.DataFrame, *, target_species: str, target_label: str,
    target_begin: float, target_end: float,
) -> dict:
    """Per-flank counts of co-occurring annotations within ``FLANK_S`` seconds."""
    pre_a, pre_b = target_begin - FLANK_S, target_begin
    post_a, post_b = target_end, target_end + FLANK_S

    def _overlap(row, a, b) -> bool:
        return float(row["Begin Time (s)"]) < b and float(row["End Time (s)"]) > a

    def _summarise(rows: list) -> dict:
        cnt = {
            "n_total": len(rows),
            "n_species_category": 0,
            "n_same_species_same_sonotype": 0,
            "n_same_species_other_sonotype": 0,
            "n_other_species_call_or_song": 0,
            "n_drum": 0,
            "n_non_biotic": 0,
            "n_method_empty": 0,
        }
        for r in rows:
            cat = str(r.get("Category") or "")
            sp = str(r.get("Species") or "")
            canon = _canonical_sonotype(r.get("Sonotype"))
            son_raw = str(r.get("Sonotype") or "").strip().lower()
            if cat == "species":
                cnt["n_species_category"] += 1
                if sp == target_species:
                    if canon == target_label:
                        cnt["n_same_species_same_sonotype"] += 1
                    else:
                        cnt["n_same_species_other_sonotype"] += 1
                else:
                    if canon in ("call", "song"):
                        cnt["n_other_species_call_or_song"] += 1
                if "drum" in son_raw:
                    cnt["n_drum"] += 1
            elif cat == "non-biotic":
                cnt["n_non_biotic"] += 1
            elif cat == "method" and son_raw in ("empty", "complete"):
                cnt["n_method_empty"] += 1
        return cnt

    pre_rows, post_rows = [], []
    for _, r in st.iterrows():
        if (
            float(r["Begin Time (s)"]) == target_begin
            and float(r["End Time (s)"]) == target_end
            and str(r.get("Species") or "") == target_species
            and _canonical_sonotype(r.get("Sonotype")) == target_label
        ):
            continue
        if _overlap(r, pre_a, pre_b):
            pre_rows.append(r)
        if _overlap(r, post_a, post_b):
            post_rows.append(r)

    return {"pre_4s": _summarise(pre_rows), "post_4s": _summarise(post_rows)}


def make_row(*, output: str, instruction_text: str, audio_path: str, metadata: dict) -> dict:
    return {
        "source_dataset": SOURCE_DATASET,
        "dataset_name": DATASET_NAME,
        "output": output,
        "instruction_text": instruction_text,
        "instruction": f"<Audio><AudioHere></Audio> {instruction_text}",
        "task": TASK,
        "file_name": audio_path.split("/")[-1],
        "license": LICENSE_STR,
        "id": str(uuid.uuid4()),
        "metadata": json.dumps(metadata),
        "audio_path_original_sample_rate": audio_path,
    }


def build(*, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = output_dir / "audio"
    audio_out_dir.mkdir(exist_ok=True)

    logger.info("Reading existing 2-s JSONL: %s", EXISTING_TIGHT_JSONL)
    tight_rows = _read_jsonl_gcs(EXISTING_TIGHT_JSONL)
    logger.info("Rows to mirror: %d", len(tight_rows))

    logger.info("Reading Weldy labeled manifest …")
    manifest = pd.read_csv(WELDY_LABELED_CSV, keep_default_na=False, na_values=[""])
    recording_index: dict[str, tuple[str, pd.DataFrame]] = {}
    for _, row in manifest.iterrows():
        rec_id = str(row.get("fn") or Path(str(row.get("file", ""))).stem)
        raw = row.get("selection_table")
        if not isinstance(raw, str) or not raw.strip():
            continue
        st = pd.read_csv(StringIO(raw), sep="\t")
        recording_index[rec_id] = (str(row.get("32khz_path", "")), st)

    by_clip: dict[str, list[dict]] = defaultdict(list)
    for r in tight_rows:
        meta = json.loads(r["metadata"])
        by_clip[meta["recording_id"]].append({"tight_row": r, "meta": meta})

    fs = fsspec.filesystem("gs")
    n_emitted = 0
    n_pad_edge = 0
    jsonl_rows: list[dict] = []
    for rec_id, items in by_clip.items():
        entry = recording_index.get(rec_id)
        if entry is None:
            logger.warning("recording_id missing: %s — skipping %d", rec_id, len(items))
            continue
        audio_path, st = entry
        try:
            audio = _load_weldy_audio_full(audio_path, fs)
        except Exception as err:  # noqa: BLE001
            logger.warning("Audio load failed for %s: %s — skipping %d rows", audio_path, err, len(items))
            continue
        clip_dur_s = audio.shape[-1] / TARGET_SR

        for item in items:
            tight = item["tight_row"]
            meta = item["meta"]
            begin = float(meta["begin_time_s"])
            end = float(meta["end_time_s"])
            species = str(meta["species"])
            species_common = str(meta.get("species_common") or "")
            label = str(tight["output"])

            seg = _cut_with_pad(audio, begin, end)
            if (begin - FLANK_S) < 0 or (end + FLANK_S) > clip_dur_s:
                n_pad_edge += 1

            flank = _summarise_flanks(
                st, target_species=species, target_label=label,
                target_begin=begin, target_end=end,
            )

            tight_fname = Path(tight["audio_path_original_sample_rate"]).stem
            seg_filename = f"{tight_fname}_10s.wav"
            sf.write(audio_out_dir / seg_filename, seg, TARGET_SR, subtype="PCM_16")

            instruction_text = _species_prompt(species_common, species)
            new_meta = {
                **meta,
                "context_seconds": SEGMENT_S,
                "flank_seconds": FLANK_S,
                "target_window_seconds": end - begin,
                "audio_path_original_sample_rate_tight": tight["audio_path_original_sample_rate"],
                "flank_annotations": flank,
            }
            jsonl_rows.append(make_row(
                output=label,
                instruction_text=instruction_text,
                audio_path=f"audio/{seg_filename}",
                metadata=new_meta,
            ))
            n_emitted += 1
        if n_emitted % 500 == 0:
            logger.info("Cut %d / %d segments", n_emitted, len(tight_rows))

    jsonl_path = output_dir / "test.jsonl"
    with open(jsonl_path, "w") as fh:
        for row in jsonl_rows:
            fh.write(json.dumps(row) + "\n")
    logger.info("Wrote %d rows to %s; edge-padded %d", len(jsonl_rows), jsonl_path, n_pad_edge)

    # Aggregate flank-leakage stats (with 4-s flanks vs the 2-s flanks in the 6-s build).
    n_rows = len(jsonl_rows)
    rows_with_same_sp_same_sonotype = 0
    rows_with_other_call_or_song = 0
    rows_with_drum_in_flank = 0
    rows_with_clean_flanks = 0
    for r in jsonl_rows:
        fl = json.loads(r["metadata"])["flank_annotations"]
        same_sp_same = fl["pre_4s"]["n_same_species_same_sonotype"] + fl["post_4s"]["n_same_species_same_sonotype"]
        other_cs = fl["pre_4s"]["n_other_species_call_or_song"] + fl["post_4s"]["n_other_species_call_or_song"]
        drum = fl["pre_4s"]["n_drum"] + fl["post_4s"]["n_drum"]
        species_cat = fl["pre_4s"]["n_species_category"] + fl["post_4s"]["n_species_category"]
        rows_with_same_sp_same_sonotype += int(same_sp_same > 0)
        rows_with_other_call_or_song += int(other_cs > 0)
        rows_with_drum_in_flank += int(drum > 0)
        rows_with_clean_flanks += int(species_cat == 0)
    logger.info("\n=== 4-s flank-annotation aggregate ===")
    logger.info("  same-species same-sonotype in flanks: %d (%.1f%%)",
                 rows_with_same_sp_same_sonotype, 100 * rows_with_same_sp_same_sonotype / n_rows)
    logger.info("  other-species call/song in flanks:    %d (%.1f%%)",
                 rows_with_other_call_or_song, 100 * rows_with_other_call_or_song / n_rows)
    logger.info("  drum in flanks:                       %d (%.1f%%)",
                 rows_with_drum_in_flank, 100 * rows_with_drum_in_flank / n_rows)
    logger.info("  clean (no species annotations):       %d (%.1f%%)",
                 rows_with_clean_flanks, 100 * rows_with_clean_flanks / n_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "data" / "beans_pro_weldy_call_or_song_sp_10s")
    args = parser.parse_args()
    build(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
