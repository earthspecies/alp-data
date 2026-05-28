#!/usr/bin/env python3
"""Build the BEANS-Pro ``weldy-call-or-song`` evaluation split.

Turns Weldy NW Dawn Chorus per-window sonotype annotations into a balanced
per-species call-vs-song binary task in BEANS-Pro JSONL format.

Filtering / balancing pipeline:

1. Iterate ``WeldyDawnChorus(split="labeled")``'s underlying CSV; each row
   carries an embedded selection_table (TSV).
2. Within each clip, keep windows with ``Category == "species"``, non-empty
   ``Species``, and a ``Sonotype`` that canonicalises (after stripping the
   trailing ``_N`` variant suffix) to exactly ``"call"`` or ``"song"``.
3. Per (file_id, begin, end, species): drop windows whose canonical sonotype
   set is ambiguous (both call AND song for the same species in the same
   2-s window). Rare but non-trivial.
4. Per species: require ≥ ``--min-per-class`` examples of EACH label
   (default 5).
5. Per surviving species: balance to ``min(n_call, n_song)`` of each label
   (deterministic seed, no global cap — per-species macro metrics neutralise
   support imbalance downstream).
6. For each kept row, cut a 2-s mono 32-kHz PCM16 WAV segment from the source
   audio at ``[Begin Time (s), End Time (s)]`` and write to
   ``<output_dir>/audio/<filename>.wav``.
7. Emit a BEANS-Pro JSONL at ``<output_dir>/test.jsonl`` with one row per
   segment.

Upload to GCS separately with::

    gsutil -m cp -r <output_dir>/* \
        gs://esp-data-ingestion/beans-pro/v0.1.0/raw/weldy_call_or_song/

Usage::

    uv run python scripts/build_beans_pro_weldy_call_or_song.py \
        --output-dir data/beans_pro_weldy_call_or_song \
        [--limit-clips 20]   # smoke-test
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
from collections import defaultdict
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


# ── Constants ────────────────────────────────────────────────────────────

SEED = 42
WELDY_GCS_ROOT = "gs://esp-data-ingestion/weldy_dawn_chorus/v0.1.0"
WELDY_LABELED_CSV = f"{WELDY_GCS_ROOT}/labeled.csv"
TARGET_SR = 32_000
LICENSE_STR = "CC-BY-4.0"
SOURCE_DATASET = "weldy_dawn_chorus"
DATASET_NAME = "weldy-call-or-song"
TASK = "call_or_song_classification"
INSTRUCTION_TEXT = "Is this a call or a song?"

# Canonicalise the sonotype: strip a trailing "_N" variant suffix and lowercase.
_VARIANT_RE = re.compile(r"_(\d+)$")


def _canonical_sonotype(raw: str) -> str | None:
    """Return ``"call"`` or ``"song"`` if `raw` is a recognised call/song
    sonotype (with optional ``_N`` suffix); else None."""
    if not raw:
        return None
    s = str(raw).strip().lower()
    s = _VARIANT_RE.sub("", s)
    if s == "call" or s == "song":
        return s
    return None


def _slug(s: str) -> str:
    """Filesystem-safe slug from a species or file name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


# ── Pass 1: gather candidate (clip, window, species, label) rows ─────────


def _iter_candidate_rows(manifest_df: pd.DataFrame, limit_clips: int | None):
    """Yield candidate annotation rows per the filtering rules in steps 1-3.

    Yields
    ------
    dict
        Keys: ``file_id``, ``recording_id``, ``audio_32k_path`` (relative to
        Weldy data root), ``begin``, ``end``, ``species``, ``species_common``,
        ``sonotype_raw``, ``label`` (``"call"``/``"song"``).
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
        if st.empty:
            continue
        if "Category" not in st.columns or "Sonotype" not in st.columns:
            continue
        # Pre-filter: species rows with a recognised sonotype + a non-empty species.
        sub = st[
            (st["Category"] == "species")
            & st["Species"].notna()
            & (st["Species"].astype(str).str.strip() != "")
        ].copy()
        if sub.empty:
            continue
        sub["canonical"] = sub["Sonotype"].apply(_canonical_sonotype)
        sub = sub[sub["canonical"].notna()].copy()
        if sub.empty:
            continue

        # Resolve per-(window, species) ambiguity: keep only those where
        # the canonical set is exactly one of {"call", "song"}.
        # Group by (begin, end, species) → collect canonical set; drop ambiguous.
        sub["_key"] = list(
            zip(sub["Begin Time (s)"], sub["End Time (s)"], sub["Species"], strict=True)
        )
        keep_keys = set()
        per_key_canon: dict[tuple, set[str]] = defaultdict(set)
        for _, srow in sub.iterrows():
            per_key_canon[srow["_key"]].add(srow["canonical"])
        for k, vals in per_key_canon.items():
            if len(vals) == 1:
                keep_keys.add(k)
        sub = sub[sub["_key"].isin(keep_keys)].copy()
        if sub.empty:
            continue
        # Deduplicate to one row per (begin, end, species) — they all share the
        # same canonical label by construction at this point.
        sub = sub.drop_duplicates(subset=["_key"], keep="first")

        # Yield candidate rows.
        recording_id = row.get("fn") or Path(str(row.get("file", ""))).stem
        for _, srow in sub.iterrows():
            yield {
                "file_id": str(row.get("file", "")),
                "recording_id": str(recording_id),
                "audio_32k_path": str(row.get("32khz_path", "")),
                "begin": float(srow["Begin Time (s)"]),
                "end": float(srow["End Time (s)"]),
                "species": str(srow["Species"]),
                "species_common": str(srow.get("Common Name", "")),
                "sonotype_raw": str(srow.get("Sonotype", "")),
                "label": str(srow["canonical"]),
            }


# ── Pass 2: eligibility + balancing ──────────────────────────────────────


def _select_balanced(
    candidates: list[dict],
    min_per_class: int,
    seed: int,
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """Apply per-species ≥ min/min filter, then per-species balance to
    ``min(n_call, n_song)``.

    Returns
    -------
    selected : list[dict]
        Balanced rows.
    counts_before : dict[str, dict[str, int]]
        ``{species: {"call": n, "song": n}}`` BEFORE balancing — useful for
        the eligibility report logged at INFO.
    """
    by_species: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"call": [], "song": []})
    for c in candidates:
        by_species[c["species"]][c["label"]].append(c)

    counts_before = {
        sp: {"call": len(v["call"]), "song": len(v["song"])} for sp, v in by_species.items()
    }

    rng = random.Random(seed)
    selected: list[dict] = []
    for sp, buckets in sorted(by_species.items()):
        n_call, n_song = len(buckets["call"]), len(buckets["song"])
        if n_call < min_per_class or n_song < min_per_class:
            continue
        n = min(n_call, n_song)
        chosen_call = rng.sample(buckets["call"], n)
        chosen_song = rng.sample(buckets["song"], n)
        selected.extend(chosen_call)
        selected.extend(chosen_song)
    return selected, counts_before


# ── Audio cutting + JSONL emission ───────────────────────────────────────


def _load_weldy_audio(audio_32k_path: str, fs: fsspec.AbstractFileSystem) -> tuple[np.ndarray, int]:
    """Read the full 32 kHz Weldy WAV from GCS into a mono float32 ndarray."""
    full_uri = f"{WELDY_GCS_ROOT}/{audio_32k_path}"
    proto, stripped = full_uri.split("://", 1)
    assert proto == "gs"
    with fs.open(stripped, "rb") as fh:
        audio, sr = librosa.load(io.BytesIO(fh.read()), sr=TARGET_SR, mono=True)
    return audio.astype(np.float32, copy=False), sr


def _cut_and_write(audio: np.ndarray, sr: int, begin: float, end: float, out_path: Path) -> None:
    """Slice ``[begin, end]`` (exact, no padding) and write a peak-normalised PCM16 WAV."""
    a = max(0, int(round(begin * sr)))
    b = min(audio.shape[-1], int(round(end * sr)))
    seg = audio[a:b]
    peak = float(np.max(np.abs(seg)) or 1.0)
    seg = (seg / peak * 0.97).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, seg, sr, subtype="PCM_16")


def make_beans_pro_row(
    *,
    output: str,
    audio_path: str,
    metadata: dict,
) -> dict:
    """Mirror of the helper in build_beans_pro_presence.py."""
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


def _log_eligibility_table(counts_before: dict[str, dict[str, int]], min_per_class: int) -> None:
    """Pretty-print the pre-balance per-species counts and which pass eligibility."""
    rows = []
    for sp, v in sorted(counts_before.items()):
        eligible = v["call"] >= min_per_class and v["song"] >= min_per_class
        rows.append((sp, v["call"], v["song"], eligible))
    eligible_n = sum(1 for r in rows if r[3])
    logger.info("Per-species pre-balance counts (eligibility floor = %d of each):", min_per_class)
    logger.info(f"  {'species':<32s} {'n_call':>7s} {'n_song':>7s} eligible?")
    for sp, c, s, ok in rows:
        logger.info(f"  {sp:<32s} {c:>7d} {s:>7d}  {'Y' if ok else '.'}")
    logger.info(
        "Species: %d total, %d eligible (≥ %d/%d).",
        len(rows),
        eligible_n,
        min_per_class,
        min_per_class,
    )


# ── Main build ───────────────────────────────────────────────────────────


def build(
    *,
    output_dir: Path,
    min_per_class: int,
    limit_clips: int | None,
    seed: int,
) -> None:
    """Run the full build into ``output_dir/{test.jsonl,audio/*.wav}``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = output_dir / "audio"
    audio_out_dir.mkdir(exist_ok=True)

    logger.info("Reading Weldy labeled manifest: %s", WELDY_LABELED_CSV)
    manifest = pd.read_csv(WELDY_LABELED_CSV, keep_default_na=False, na_values=[""])
    logger.info("Weldy labeled: %d clips", len(manifest))

    # Pass 1: gather candidates.
    candidates = list(_iter_candidate_rows(manifest, limit_clips))
    logger.info("Candidate (post-filter, post-ambiguity) windows: %d", len(candidates))

    # Pass 2: eligibility + balance.
    selected, counts_before = _select_balanced(candidates, min_per_class, seed)
    _log_eligibility_table(counts_before, min_per_class)
    logger.info("Balanced selection: %d rows", len(selected))

    if not selected:
        logger.warning("No rows survived eligibility/balancing — nothing to write.")
        return

    # Pass 3: cut audio, emit JSONL.
    # Cache audio per recording so we only download each source clip once.
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
            # Segment filename: <recid>__<beg_s>_<end_s>__<species_slug>__<label>.wav
            stem = (
                f"{_slug(r['recording_id'])}__"
                f"{r['begin']:.3f}_{r['end']:.3f}__"
                f"{_slug(r['species'])}__{r['label']}"
            )
            seg_filename = f"{stem}.wav"
            seg_out = audio_out_dir / seg_filename
            _cut_and_write(audio, sr, r["begin"], r["end"], seg_out)
            metadata = {
                "species": r["species"],
                "species_common": r["species_common"],
                "source_dataset": SOURCE_DATASET,
                "sonotype_raw": r["sonotype_raw"],
                "begin_time_s": r["begin"],
                "end_time_s": r["end"],
                "recording_id": r["recording_id"],
            }
            row = make_beans_pro_row(
                output=r["label"],
                audio_path=f"audio/{seg_filename}",
                metadata=metadata,
            )
            jsonl_rows.append(row)
            n_done += 1
        if n_done % 200 == 0:
            logger.info("Cut %d / %d segments", n_done, len(selected))

    # Write JSONL.
    jsonl_path = output_dir / "test.jsonl"
    with open(jsonl_path, "w") as fh:
        for row in jsonl_rows:
            fh.write(json.dumps(row) + "\n")

    # Final balance summary.
    label_counts = pd.Series([r["output"] for r in jsonl_rows]).value_counts().to_dict()
    species_counts = (
        pd.Series([json.loads(r["metadata"])["species"] for r in jsonl_rows])
        .value_counts()
        .to_dict()
    )
    logger.info("Wrote %d JSONL rows to %s", len(jsonl_rows), jsonl_path)
    logger.info("Wrote %d audio segments to %s", n_done, audio_out_dir)
    logger.info("Label balance: %s", label_counts)
    logger.info(
        "Species count: %d (top 5: %s)", len(species_counts), dict(list(species_counts.items())[:5])
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "beans_pro_weldy_call_or_song",
        help="Directory for local JSONL + cut audio segments.",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=5,
        help="Per-species floor on call/song counts to be eligible.",
    )
    parser.add_argument(
        "--limit-clips",
        type=int,
        default=None,
        help="Optional cap on Weldy source clips processed (smoke).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    build(
        output_dir=args.output_dir,
        min_per_class=args.min_per_class,
        limit_clips=args.limit_clips,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
