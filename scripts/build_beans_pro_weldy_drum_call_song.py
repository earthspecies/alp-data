#!/usr/bin/env python3
"""Build the BEANS-Pro ``weldy-drum-call-song`` evaluation split.

Species-agnostic 3-way classification between drumming, calling, and singing.
Weldy's drum annotations are species-anonymous (all labelled
"Woodpecker drum" with empty Species) so this task is intentionally
species-agnostic across all three classes — for parity, we also sample the
call/song windows without per-species balancing.

Filtering / balancing:

1. Iterate ``labeled.csv`` selection_tables; keep rows with
   ``Category == "species"`` and a canonical sonotype in {drum, call, song}
   (after stripping the trailing ``_N`` variant suffix).
2. For ``drum``: keep all (Species is allowed to be empty).
   For ``call``/``song``: require non-empty Species.
3. Balance across labels to ``min(n_drum, n_call, n_song)`` rows of each
   (deterministic seed).
4. Cut 2-s mono 32-kHz PCM16 segments and emit BeansPro JSONL rows. The
   ground-truth ``output`` is the canonical ``drum``/``call``/``song``.

Prompt is the trained ``call_type`` variant 5 enumeration adapted to our
three-class space::

    "Is this drumming, a call, or a song?"

Upload with::

    gsutil -m cp test.jsonl gs://esp-data-ingestion/beans-pro/v0.1.0/raw/weldy_drum_call_song/test.jsonl
    gsutil -m cp -r audio    gs://esp-data-ingestion/beans-pro/v0.1.0/raw/weldy_drum_call_song/

Usage::

    uv run python scripts/build_beans_pro_weldy_drum_call_song.py \
        --output-dir data/beans_pro_weldy_drum_call_song \
        [--limit-clips 20]   # smoke
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import random
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


SEED = 42
WELDY_GCS_ROOT = "gs://esp-data-ingestion/weldy_dawn_chorus/v0.1.0"
WELDY_LABELED_CSV = f"{WELDY_GCS_ROOT}/labeled.csv"
TARGET_SR = 32_000
LICENSE_STR = "CC-BY-4.0"
SOURCE_DATASET = "weldy_dawn_chorus"
DATASET_NAME = "weldy-drum-call-song"
TASK = "drum_call_or_song_classification"
INSTRUCTION_TEXT = "Is this drumming, a call, or a song?"

LABELS = ("drum", "call", "song")

_VARIANT_RE = re.compile(r"_(\d+)$")


def _canonical_sonotype(raw: str) -> str | None:
    if not raw:
        return None
    s = _VARIANT_RE.sub("", str(raw).strip().lower())
    if s in LABELS:
        return s
    return None


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s or "noid")).strip("_") or "noid"


def _iter_candidate_rows(manifest_df: pd.DataFrame, limit_clips: int | None):
    """Yield candidate (clip, window, label) rows per filter rules.

    Yields
    ------
    dict
        ``file_id, recording_id, audio_32k_path, begin, end, species,
        species_common, sonotype_raw, label``. ``species`` may be empty
        when ``label == "drum"``.
    """
    n_clips = 0
    for _, row in manifest_df.iterrows():
        if limit_clips is not None and n_clips >= limit_clips:
            break
        n_clips += 1
        st_raw = row.get("selection_table")
        if not isinstance(st_raw, str) or not st_raw.strip():
            continue
        st = pd.read_csv(StringIO(st_raw), sep="\t")
        if st.empty or "Category" not in st.columns or "Sonotype" not in st.columns:
            continue
        sub = st[st["Category"] == "species"].copy()
        sub["canonical"] = sub["Sonotype"].apply(_canonical_sonotype)
        sub = sub[sub["canonical"].notna()].copy()
        if sub.empty:
            continue

        recording_id = row.get("fn") or Path(str(row.get("file", ""))).stem
        for _, srow in sub.iterrows():
            label = srow["canonical"]
            sp = str(srow.get("Species") or "").strip()
            if label in ("call", "song") and not sp:
                # species-labelled vocalisations only for call/song
                continue
            # For drum: species may be empty ("Woodpecker drum" anonymous)
            yield {
                "file_id": str(row.get("file", "")),
                "recording_id": str(recording_id),
                "audio_32k_path": str(row.get("32khz_path", "")),
                "begin": float(srow["Begin Time (s)"]),
                "end": float(srow["End Time (s)"]),
                "species": sp,
                "species_common": str(srow.get("Common Name", "")),
                "sonotype_raw": str(srow.get("Sonotype", "")),
                "label": label,
            }


def _balance(candidates: list[dict], seed: int) -> list[dict]:
    """Sample ``min(n_drum, n_call, n_song)`` of each label deterministically."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_label[c["label"]].append(c)
    counts_before = {k: len(v) for k, v in by_label.items()}
    n = min(counts_before.get(lbl, 0) for lbl in LABELS)
    logger.info("Pre-balance per-label counts: %s → balance to %d each", counts_before, n)
    rng = random.Random(seed)
    selected = []
    for lbl in LABELS:
        pool = by_label.get(lbl, [])
        selected.extend(rng.sample(pool, n))
    return selected


def _load_weldy_audio(audio_32k_path: str, fs: fsspec.AbstractFileSystem) -> tuple[np.ndarray, int]:
    full_uri = f"{WELDY_GCS_ROOT}/{audio_32k_path}"
    proto, stripped = full_uri.split("://", 1)
    assert proto == "gs"
    with fs.open(stripped, "rb") as fh:
        audio, sr = librosa.load(io.BytesIO(fh.read()), sr=TARGET_SR, mono=True)
    return audio.astype(np.float32, copy=False), sr


def _cut_and_write(audio: np.ndarray, sr: int, begin: float, end: float, out_path: Path) -> None:
    a = max(0, int(round(begin * sr)))
    b = min(audio.shape[-1], int(round(end * sr)))
    seg = audio[a:b]
    peak = float(np.max(np.abs(seg)) or 1.0)
    seg = (seg / peak * 0.97).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, seg, sr, subtype="PCM_16")


def make_beans_pro_row(*, output: str, audio_path: str, metadata: dict) -> dict:
    return {
        "source_dataset": SOURCE_DATASET,
        "dataset_name": DATASET_NAME,
        "output": output,
        "instruction_text": INSTRUCTION_TEXT,
        "instruction": f"<Audio><AudioHere></Audio> {INSTRUCTION_TEXT}",
        "task": TASK,
        "file_name": audio_path.split("/")[-1],
        "license": LICENSE_STR,
        "id": str(uuid.uuid4()),
        "metadata": json.dumps(metadata),
        "audio_path_original_sample_rate": audio_path,
    }


def build(*, output_dir: Path, limit_clips: int | None, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = output_dir / "audio"
    audio_out_dir.mkdir(exist_ok=True)

    logger.info("Reading Weldy labeled manifest: %s", WELDY_LABELED_CSV)
    manifest = pd.read_csv(WELDY_LABELED_CSV, keep_default_na=False, na_values=[""])
    logger.info("Weldy labeled: %d clips", len(manifest))

    candidates = list(_iter_candidate_rows(manifest, limit_clips))
    logger.info("Candidate (post-filter) windows: %d", len(candidates))

    selected = _balance(candidates, seed)
    label_counts = Counter(c["label"] for c in selected)
    logger.info("Selected: %d rows %s", len(selected), dict(label_counts))
    if not selected:
        logger.warning("No rows survived — nothing to write.")
        return

    fs = fsspec.filesystem("gs")
    selected_by_clip: dict[str, list[dict]] = defaultdict(list)
    for r in selected:
        selected_by_clip[r["audio_32k_path"]].append(r)
    logger.info("Source clips needed: %d", len(selected_by_clip))

    jsonl_rows: list[dict] = []
    n_done = 0
    for path, rows in selected_by_clip.items():
        try:
            audio, sr = _load_weldy_audio(path, fs)
        except Exception as err:  # noqa: BLE001
            logger.warning("Audio load failed for %s: %s — skipping %d rows.", path, err, len(rows))
            continue
        for r in rows:
            stem = (
                f"{_slug(r['recording_id'])}__"
                f"{r['begin']:.3f}_{r['end']:.3f}__"
                f"{_slug(r['species'] or 'unknown')}__{r['label']}"
            )
            seg_filename = f"{stem}.wav"
            _cut_and_write(audio, sr, r["begin"], r["end"], audio_out_dir / seg_filename)
            metadata = {
                "species": r["species"],
                "species_common": r["species_common"],
                "source_dataset": SOURCE_DATASET,
                "sonotype_raw": r["sonotype_raw"],
                "begin_time_s": r["begin"],
                "end_time_s": r["end"],
                "recording_id": r["recording_id"],
            }
            jsonl_rows.append(
                make_beans_pro_row(
                    output=r["label"],
                    audio_path=f"audio/{seg_filename}",
                    metadata=metadata,
                )
            )
            n_done += 1
        if n_done % 200 == 0:
            logger.info("Cut %d / %d segments", n_done, len(selected))

    jsonl_path = output_dir / "test.jsonl"
    with open(jsonl_path, "w") as fh:
        for row in jsonl_rows:
            fh.write(json.dumps(row) + "\n")

    logger.info("Wrote %d JSONL rows to %s", len(jsonl_rows), jsonl_path)
    logger.info("Wrote %d audio segments to %s", n_done, audio_out_dir)
    logger.info(
        "Label balance (post-write): %s",
        dict(
            Counter(
                json.loads(r["metadata"])["sonotype_raw"][:0] or r["output"] for r in jsonl_rows
            )
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "beans_pro_weldy_drum_call_song",
    )
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    build(output_dir=args.output_dir, limit_clips=args.limit_clips, seed=args.seed)


if __name__ == "__main__":
    main()
