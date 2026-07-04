#!/usr/bin/env python3
"""Build the BEANS-Pro ``dclde2013-multilabel-species`` evaluation split.

Turns the held-out ``dclde_2013_nefsc_sbnms_allbaleen`` SuperWhale component
(668 continuous 900 s HARP recordings from NEFSC Stellwagen Bank, fully
annotated for baleen whales) into a multilabel species-classification task in
BEANS-Pro JSONL format: each example is a fixed 10 s clip, and the target is
the set of baleen-whale scientific names vocalizing in that clip (or ``None``).

Because DCLDE 2013 is exhaustively annotated for baleen whales
(``all_cetaceans_labeled=True``), clips with no overlapping annotation are
*true negatives* and are included so the benchmark also measures the model's
false-positive rate.

Pipeline
--------
1. Read the canonical SuperWhale manifest from GCS, filter to
   ``dclde_2013_nefsc_sbnms_allbaleen``.
2. For each 900 s recording, slide fixed ``--window-len`` (10 s) windows at
   ``--window-hop`` (5 s). For every window compute:
   - ``present_any``: any annotated event overlaps the window at all.
   - ``labeled``: events overlapping by >= ``--min-overlap-sec`` seconds OR
     >= ``--min-overlap-frac`` of the event duration.
   Window categories:
   - **negative** (target ``None``): no event overlaps at all.
   - **ambiguous**: an event clips the window but below the inclusion
     threshold -> dropped (kept out of both positives and negatives).
   - **positive**: target = sorted ``labeled`` species intersected with the
     kept label space. Windows containing an out-of-label-space species
     (e.g. the single sei-whale file) are dropped so targets stay exhaustive.
3. Per recording, select non-overlapping windows (start gap >=
   ``--min-start-gap``) and cap to ``--max-windows-per-recording``, preferring
   multi-species windows, then single-species, then negatives (seeded).
4. Global balancing: cap how many windows any single species may appear in
   (``--per-species-cap``; multi-species windows are never dropped) and
   sub-sample negatives to ``--neg-fraction`` of the final set.
5. Cut each selected window to a mono 32 kHz PCM16 WAV from the recording's
   ``32khz_path`` source and emit a BEANS-Pro JSONL row.

Upload to GCS separately with::

    gsutil -m cp -r <output_dir>/audio \
        gs://esp-data-ingestion/beans-pro/v0.1.0/raw/dclde2013_multilabel_species/
    gsutil cp <output_dir>/test.jsonl \
        gs://esp-data-ingestion/beans-pro/v0.1.0/raw/dclde2013_multilabel_species/

Usage::

    uv run python scripts/build_beans_pro_dclde2013_multilabel_species.py \
        --output-dir data/beans_pro_dclde2013_multilabel_species \
        [--limit-clips 10]   # smoke-test
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


# ── Constants ────────────────────────────────────────────────────────────

SEED = 42
SUPERWHALE_GCS_ROOT = "gs://esp-data-ingestion/superwhale/v0.1.0/raw"
SUPERWHALE_MANIFEST = f"{SUPERWHALE_GCS_ROOT}/superwhale_detection.csv"
SOURCE_DATASET = "dclde_2013_nefsc_sbnms_allbaleen"
DATASET_NAME = "dclde2013-multilabel-species"
TASK = "multilabel_species_classification"
TARGET_SR = 32_000
LICENSE_STR = "CC0-1.0"
# Match the training-time ``multilabel_species`` prompt wording exactly.
INSTRUCTION_TEXT = "List the scientific names of all species vocalizing in this audio clip."
LABEL_SEPARATOR = ", "

# Baleen species kept in the label space. Blue whale (B. musculus) is kept but
# underpowered (4 source files); sei whale (B. borealis, 1 file) is out of the
# label space, and any window containing it is dropped to keep targets
# exhaustive within the label space.
KEEP_SPECIES = {
    "Balaenoptera physalus",  # fin whale
    "Megaptera novaeangliae",  # humpback whale
    "Eubalaena glacialis",  # North Atlantic right whale
    "Balaenoptera musculus",  # blue whale (underpowered)
}


def _slug(s: str) -> str:
    """Return a filesystem-safe slug from `s`."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


# ── Manifest loading ──────────────────────────────────────────────────────


def _load_dclde2013_rows(manifest_uri: str, limit_clips: int | None) -> pd.DataFrame:
    """Read the SuperWhale manifest and return the DCLDE 2013 rows.

    Returns
    -------
    pandas.DataFrame
        Rows whose ``source_dataset`` is ``dclde_2013_nefsc_sbnms_allbaleen``.
    """
    logger.info("Reading SuperWhale manifest: %s", manifest_uri)
    df = pd.read_csv(manifest_uri, keep_default_na=False, na_values=[])
    df = df[df["source_dataset"] == SOURCE_DATASET].reset_index(drop=True)
    logger.info("DCLDE 2013 recordings in manifest: %d", len(df))
    if limit_clips is not None:
        df = df.head(limit_clips).reset_index(drop=True)
        logger.info("Limited to first %d recordings (smoke).", len(df))
    return df


def _parse_events(selection_table: str) -> list[tuple[float, float, str]]:
    """Parse a selection-table TSV blob into ``(begin, end, species)`` events.

    Returns
    -------
    list[tuple[float, float, str]]
        One tuple per annotated event with a resolvable species label.
    """
    if not isinstance(selection_table, str) or not selection_table.strip():
        return []
    lines = selection_table.strip().split("\n")
    if len(lines) < 2:
        return []
    events: list[tuple[float, float, str]] = []
    reader = csv_dictreader(selection_table)
    for ev in reader:
        species = (ev.get("species") or ev.get("canonical_name") or "").strip()
        if not species:
            continue
        try:
            begin = float(ev["Begin Time (s)"])
            end = float(ev["End Time (s)"])
        except (KeyError, ValueError):
            continue
        if end <= begin:
            continue
        events.append((begin, end, species))
    return events


def csv_dictreader(tsv: str):
    """Yield dict rows from a tab-separated selection-table blob."""
    import csv

    csv.field_size_limit(sys.maxsize)
    return csv.DictReader(io.StringIO(tsv), delimiter="\t")


# ── Windowing + labeling ──────────────────────────────────────────────────


def _label_window(
    events: list[tuple[float, float, str]],
    win_start: float,
    win_end: float,
    min_overlap_sec: float,
    min_overlap_frac: float,
) -> tuple[str, frozenset[str]] | None:
    """Categorize a window from its overlapping events.

    Returns
    -------
    tuple[str, frozenset[str]] | None
        ``("negative", frozenset())`` for a clean negative,
        ``("positive", species_set)`` for a labeled positive, or ``None`` when
        the window is ambiguous / contains an out-of-label-space species and
        should be dropped.
    """
    present_any = False
    labeled: set[str] = set()
    for begin, end, species in events:
        overlap = min(end, win_end) - max(begin, win_start)
        if overlap <= 0:
            continue
        present_any = True
        event_dur = end - begin
        if overlap >= min_overlap_sec or overlap >= min_overlap_frac * event_dur:
            labeled.add(species)

    if not present_any:
        return ("negative", frozenset())
    if not labeled:
        # An event clips the window but below threshold: ambiguous -> drop.
        return None
    # Any out-of-label-space species present (e.g. sei): drop to keep targets
    # exhaustive within the kept label space.
    if labeled - KEEP_SPECIES:
        return None
    return ("positive", frozenset(labeled))


def _enumerate_windows(
    events: list[tuple[float, float, str]],
    duration: float,
    win_len: float,
    win_hop: float,
    min_overlap_sec: float,
    min_overlap_frac: float,
) -> list[dict]:
    """Enumerate labeled windows for one recording.

    Returns
    -------
    list[dict]
        Window dicts with ``start``, ``end``, ``category`` and ``species``.
    """
    windows: list[dict] = []
    start = 0.0
    while start + win_len <= duration + 1e-6:
        end = start + win_len
        result = _label_window(events, start, end, min_overlap_sec, min_overlap_frac)
        if result is not None:
            category, species = result
            windows.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "category": category,
                    "species": species,
                }
            )
        start += win_hop
    return windows


def _select_per_recording(
    windows: list[dict],
    min_start_gap: float,
    cap_multi: int,
    cap_single: int,
    cap_neg: int,
    rng: random.Random,
) -> list[dict]:
    """Select non-overlapping windows for one recording, capped per category.

    Greedily keeps windows whose starts are >= ``min_start_gap`` apart (so
    selected windows do not overlap), then takes up to ``cap_multi`` multi-,
    ``cap_single`` single-species, and ``cap_neg`` negative windows. Capping
    per category (rather than overall) keeps single-species and negative
    windows from being starved in this dense multi-species soundscape, and
    capping per recording limits temporal autocorrelation.

    Returns
    -------
    list[dict]
        The selected window dicts for this recording.
    """
    ordered = sorted(windows, key=lambda w: w["start"])
    non_overlapping: list[dict] = []
    last_start = -1e9
    for w in ordered:
        if w["start"] >= last_start + min_start_gap:
            non_overlapping.append(w)
            last_start = w["start"]

    multi = [w for w in non_overlapping if len(w["species"]) >= 2]
    single = [w for w in non_overlapping if len(w["species"]) == 1]
    negative = [w for w in non_overlapping if not w["species"]]
    rng.shuffle(multi)
    rng.shuffle(single)
    rng.shuffle(negative)

    return multi[:cap_multi] + single[:cap_single] + negative[:cap_neg]


# ── Audio cutting + JSONL emission ─────────────────────────────────────────


def _load_audio(path_32k: str, fs: fsspec.AbstractFileSystem) -> np.ndarray:
    """Read a full 32 kHz mono recording from GCS into a float32 array."""
    full_uri = f"{SUPERWHALE_GCS_ROOT}/{path_32k}"
    _, stripped = full_uri.split("://", 1)
    with fs.open(stripped, "rb") as fh:
        audio, _ = librosa.load(io.BytesIO(fh.read()), sr=TARGET_SR, mono=True)
    return audio.astype(np.float32, copy=False)


def _cut_and_write(audio: np.ndarray, start: float, end: float, out_path: Path) -> None:
    """Slice ``[start, end]`` and write a peak-normalised PCM16 WAV."""
    a = max(0, int(round(start * TARGET_SR)))
    b = min(audio.shape[-1], int(round(end * TARGET_SR)))
    seg = audio[a:b]
    peak = float(np.max(np.abs(seg)) or 1.0)
    seg = (seg / peak * 0.97).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, seg, TARGET_SR, subtype="PCM_16")


def make_beans_pro_row(*, output: str, audio_path: str, metadata: dict) -> dict:
    """Build one BEANS-Pro JSONL record."""
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


# ── Global balancing ───────────────────────────────────────────────────────


def _balance(
    selected: list[dict],
    single_species_cap: int,
    neg_fraction: float,
    rng: random.Random,
) -> list[dict]:
    """Cap single-species windows per species and set the negative fraction.

    Multi-species windows are always kept (they carry the co-occurrence signal
    that makes this a genuine multilabel task). Single-species windows are
    capped per species to ``single_species_cap`` so the fin whale (which
    dominates the soundscape) does not swamp per-species support. Negatives are
    sub-sampled so they make up ``neg_fraction`` of the final set.

    Returns
    -------
    list[dict]
        The balanced selection.
    """
    multi = [w for w in selected if len(w["species"]) >= 2]
    negatives = [w for w in selected if not w["species"]]
    singles_by_sp: dict[str, list[dict]] = defaultdict(list)
    for w in selected:
        if len(w["species"]) == 1:
            singles_by_sp[next(iter(w["species"]))].append(w)

    kept_positives: list[dict] = list(multi)
    for sp, rows in singles_by_sp.items():
        rng.shuffle(rows)
        kept_positives.extend(rows[:single_species_cap])

    n_pos = len(kept_positives)
    if neg_fraction >= 1.0:
        max_neg = len(negatives)
    else:
        max_neg = int(round(neg_fraction / (1.0 - neg_fraction) * n_pos))
    rng.shuffle(negatives)
    kept_negatives = negatives[:max_neg]

    final = kept_positives + kept_negatives
    rng.shuffle(final)
    return final


# ── Statistics ──────────────────────────────────────────────────────────────


def _summarize(rows: list[dict], n_recordings: int) -> dict:
    """Compute dataset statistics for reporting.

    Returns
    -------
    dict
        Summary statistics keyed for JSON serialization and logging.
    """
    species_files: Counter = Counter()
    cardinality: Counter = Counter()
    per_recording: Counter = Counter()
    n_pos = n_neg = n_multi = 0
    for w in rows:
        k = len(w["species"])
        cardinality[k] += 1
        per_recording[w["recording_id"]] += 1
        if k == 0:
            n_neg += 1
        else:
            n_pos += 1
            if k >= 2:
                n_multi += 1
            for sp in w["species"]:
                species_files[sp] += 1
    counts = sorted(per_recording.values())
    return {
        "n_windows": len(rows),
        "n_recordings_with_windows": len(per_recording),
        "n_source_recordings": n_recordings,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "negative_fraction": round(n_neg / len(rows), 4) if rows else 0.0,
        "n_multi_species": n_multi,
        "label_cardinality_histogram": dict(sorted(cardinality.items())),
        "per_species_window_counts": dict(species_files.most_common()),
        "windows_per_recording": {
            "min": counts[0] if counts else 0,
            "median": counts[len(counts) // 2] if counts else 0,
            "max": counts[-1] if counts else 0,
        },
    }


# ── Main build ───────────────────────────────────────────────────────────


def build(args: argparse.Namespace) -> None:
    """Run the full build into ``output_dir/{test.jsonl,audio/*.wav,stats.json}``."""
    output_dir: Path = args.output_dir
    audio_out_dir = output_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    df = _load_dclde2013_rows(args.manifest, args.limit_clips)

    # Pass 1: enumerate + select per recording.
    selected: list[dict] = []
    for _, row in df.iterrows():
        events = _parse_events(row["selection_table"])
        try:
            duration = float(row["duration_s"])
        except (KeyError, ValueError):
            continue
        recording_id = Path(str(row["audio_path"])).stem
        windows = _enumerate_windows(
            events,
            duration,
            args.window_len,
            args.window_hop,
            args.min_overlap_sec,
            args.min_overlap_frac,
        )
        rec_selected = _select_per_recording(
            windows,
            args.min_start_gap,
            args.cap_multi_per_recording,
            args.cap_single_per_recording,
            args.cap_neg_per_recording,
            rng,
        )
        for w in rec_selected:
            w["recording_id"] = recording_id
            w["path_32k"] = row["32khz_path"]
            selected.append(w)

    logger.info("Selected windows before balancing: %d", len(selected))

    # Pass 2: global balancing.
    final = _balance(selected, args.single_species_cap, args.neg_fraction, rng)
    logger.info("Windows after balancing: %d", len(final))

    stats = _summarize(final, len(df))
    logger.info("Dataset statistics:\n%s", json.dumps(stats, indent=2))

    if args.stats_only:
        (output_dir).mkdir(parents=True, exist_ok=True)
        with open(output_dir / "stats.json", "w") as fh:
            json.dump(stats, fh, indent=2)
        logger.info("Wrote stats only (no audio) to %s", output_dir / "stats.json")
        return

    # Pass 3: cut audio + emit JSONL, caching one recording's audio at a time.
    by_recording: dict[str, list[dict]] = defaultdict(list)
    for w in final:
        by_recording[w["path_32k"]].append(w)

    fs = fsspec.filesystem("gs")
    jsonl_rows: list[dict] = []
    n_done = 0
    for path_32k, rows in by_recording.items():
        try:
            audio = _load_audio(path_32k, fs)
        except Exception as err:  # noqa: BLE001
            logger.warning("Audio load failed for %s: %s — skipping %d windows.", path_32k, err, len(rows))
            continue
        for w in rows:
            species_sorted = sorted(w["species"])
            output = LABEL_SEPARATOR.join(species_sorted) if species_sorted else "None"
            stem = f"{_slug(w['recording_id'])}__{w['start']:.1f}_{w['end']:.1f}"
            seg_filename = f"{stem}.wav"
            _cut_and_write(audio, w["start"], w["end"], audio_out_dir / seg_filename)
            metadata = {
                "recording_id": w["recording_id"],
                "window_start_sec": w["start"],
                "window_end_sec": w["end"],
                "species_list": species_sorted,
                "n_species": len(species_sorted),
                "is_negative": not species_sorted,
                "source_dataset": SOURCE_DATASET,
            }
            jsonl_rows.append(
                make_beans_pro_row(
                    output=output,
                    audio_path=f"audio/{seg_filename}",
                    metadata=metadata,
                )
            )
            n_done += 1
        if n_done % 200 == 0:
            logger.info("Cut %d / %d windows", n_done, len(final))

    jsonl_path = output_dir / "test.jsonl"
    with open(jsonl_path, "w") as fh:
        for r in jsonl_rows:
            fh.write(json.dumps(r) + "\n")
    with open(output_dir / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    logger.info("Wrote %d JSONL rows to %s", len(jsonl_rows), jsonl_path)
    logger.info("Wrote %d audio segments to %s", n_done, audio_out_dir)


def main() -> None:
    """Parse CLI arguments and run the build."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "beans_pro_dclde2013_multilabel_species",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=SUPERWHALE_MANIFEST,
        help="SuperWhale manifest URI/path (override for local stats-only smoke).",
    )
    parser.add_argument("--window-len", type=float, default=10.0)
    parser.add_argument("--window-hop", type=float, default=5.0)
    parser.add_argument("--min-overlap-sec", type=float, default=1.0)
    parser.add_argument("--min-overlap-frac", type=float, default=0.5)
    parser.add_argument("--min-start-gap", type=float, default=10.0)
    parser.add_argument("--cap-multi-per-recording", type=int, default=2)
    parser.add_argument("--cap-single-per-recording", type=int, default=2)
    parser.add_argument("--cap-neg-per-recording", type=int, default=3)
    parser.add_argument("--single-species-cap", type=int, default=500)
    parser.add_argument("--neg-fraction", type=float, default=0.30)
    parser.add_argument("--limit-clips", type=int, default=None, help="Smoke cap on recordings.")
    parser.add_argument("--stats-only", action="store_true", help="Compute stats without cutting audio.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
