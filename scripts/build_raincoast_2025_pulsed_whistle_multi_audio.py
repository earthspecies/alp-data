#!/usr/bin/env python3
"""Build Raincoast 2025 pulsed-vs-whistle few-shot multi-audio eval data.

The source annotations are Raven selection tables with event boundaries in
``Begin Time (s)`` and ``End Time (s)``. This script crops each scored event
into a short WAV and emits BEANS-Pro multi-audio JSONL rows where fixed support
examples precede the query clip.

Usage::

    uv run python scripts/build_raincoast_2025_pulsed_whistle_multi_audio.py
    uv run python scripts/build_raincoast_2025_pulsed_whistle_multi_audio.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from esp_data.io import filesystem_from_path, read_audio  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

ANNOTATION_ROOT = (
    "gs://esp-raincoast/2025/annotation/Hydrophone_recordings/pd-pv-w-detections_evaluation-set_v0"
)
RAW_ROOT = "gs://esp-raincoast/2025/raw"
OUTPUT_ROOT = "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/raincoast_2025_pulsed_whistle_fewshot"
SPLIT_NAME = "raincoast-2025-pulsed-whistle-fewshot"

SCORED_LABELS = {"pd", "pv", "pu", "w"}
PULSED_LABELS = {"pd", "pv", "pu"}
SUPPORT_LABELS = ("pd", "pv", "w")

PROMPT = """You are classifying killer whale vocalizations.

Definitions adapted from Ford 1989:
- Pulsed calls are complex tones with multiple sound components. Their pulse repetition
  rate can extend up to 4 kHz. They often contain abrupt shifts in pulsing rate, and
  many include an overlapping narrow-band tonal component.
- Discrete pulsed calls repeat in a sequence and are usually not longer than 1.5 seconds.
- Variable pulsed calls do not repeat in a sequence and range from short squeaks and
  trills to long, raucous squawks.
- Whistles are single narrow-band tones with little or no harmonic or side-band structure,
  occurring between 1.5 and 18 kHz. They may contain modulations or abrupt shifts in
  frequency.

Labeled support examples:
Audio 1 is a discrete pulsed call.
<Audio><AudioHere></Audio>
Audio 2 is a variable pulsed call.
<Audio><AudioHere></Audio>
Audio 3 is a whistle.
<Audio><AudioHere></Audio>

Classify the query vocalization.
<Audio><AudioHere></Audio>

Answer with exactly one of: pulsed, whistle."""


@dataclass(frozen=True)
class Event:
    """Single Raincoast annotation event."""

    event_id: str
    annotation_path: str
    raw_audio_path: str
    begin_file: str
    selection: str
    begin_time: float
    end_time: float
    low_freq_hz: float | None
    high_freq_hz: float | None
    call_type: str
    binary_label: str
    boat_noise: str
    comments: str


def _strip_gs(path: str) -> str:
    """Return a GCS path without its scheme for ``gcsfs`` globbing.

    Returns
    -------
    str
        Path without the ``gs://`` prefix.
    """
    return path.removeprefix("gs://")


def _maybe_float(value: str | None) -> float | None:
    """Parse a possibly empty float field.

    Returns
    -------
    float | None
        Parsed float, or ``None`` when the value is empty or invalid.
    """
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if np.isnan(parsed):
        return None
    return parsed


def list_annotation_paths(annotation_root: str) -> list[str]:
    """List Raincoast selection tables below an annotation root.

    Parameters
    ----------
    annotation_root
        GCS or local directory containing ``*.txt`` annotation files.

    Returns
    -------
    list[str]
        Sorted annotation paths, excluding README files.
    """
    fs = filesystem_from_path(annotation_root)
    paths = [
        f"gs://{path}" if annotation_root.startswith("gs://") else path
        for path in fs.glob(f"{_strip_gs(annotation_root)}/*.txt")
    ]
    return sorted(
        path for path in paths if not path.rsplit("/", 1)[-1].lower().startswith("readme")
    )


def build_raw_audio_index(raw_root: str) -> dict[str, str]:
    """Build a case-insensitive basename to raw audio path index.

    Parameters
    ----------
    raw_root
        GCS or local raw data root.

    Returns
    -------
    dict[str, str]
        Maps lowercase WAV basenames to full paths.

    Raises
    ------
    ValueError
        If two raw audio files have the same case-insensitive basename.
    """
    fs = filesystem_from_path(raw_root)
    index: dict[str, str] = {}
    for path in fs.glob(f"{_strip_gs(raw_root)}/**/*"):
        if not path.lower().endswith(".wav"):
            continue
        full_path = f"gs://{path}" if raw_root.startswith("gs://") else path
        name = full_path.rsplit("/", 1)[-1].lower()
        if name in index and index[name] != full_path:
            raise ValueError(
                f"Duplicate raw audio basename {name!r}: {index[name]} and {full_path}"
            )
        index[name] = full_path
    return index


def load_events(annotation_root: str, raw_root: str) -> list[Event]:
    """Load scored Raincoast pulsed/whistle events.

    Parameters
    ----------
    annotation_root
        Directory containing annotation TSVs.
    raw_root
        Directory containing raw hydrophone WAVs.

    Returns
    -------
    list[Event]
        Events with call types in ``pd``, ``pv``, ``pu``, or ``w``.

    Raises
    ------
    FileNotFoundError
        If an annotation references a missing raw audio file.
    """
    raw_index = build_raw_audio_index(raw_root)
    events: list[Event] = []
    all_counts: Counter[str] = Counter()
    missing_raw: set[str] = set()

    for annotation_path in list_annotation_paths(annotation_root):
        fs = filesystem_from_path(annotation_path)
        with fs.open(annotation_path, "r") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                call_type = (row.get("Call Type") or "").strip().lower()
                all_counts[call_type] += 1
                if call_type not in SCORED_LABELS:
                    continue

                begin_file = (row.get("Begin File") or "").strip()
                raw_audio_path = raw_index.get(begin_file.lower())
                if raw_audio_path is None:
                    missing_raw.add(begin_file)
                    continue

                begin_time = _maybe_float(row.get("Begin Time (s)"))
                end_time = _maybe_float(row.get("End Time (s)"))
                if begin_time is None or end_time is None or end_time <= begin_time:
                    logger.warning("Skipping invalid time bounds in %s: %s", annotation_path, row)
                    continue

                selection = (row.get("Selection") or str(len(events) + 1)).strip()
                stem = Path(begin_file).stem.replace(" ", "_")
                event_id = f"{stem}_sel{int(float(selection)):04d}"
                binary_label = "whistle" if call_type == "w" else "pulsed"
                events.append(
                    Event(
                        event_id=event_id,
                        annotation_path=annotation_path,
                        raw_audio_path=raw_audio_path,
                        begin_file=begin_file,
                        selection=selection,
                        begin_time=begin_time,
                        end_time=end_time,
                        low_freq_hz=_maybe_float(row.get("Low Freq (Hz)")),
                        high_freq_hz=_maybe_float(row.get("High Freq (Hz)")),
                        call_type=call_type,
                        binary_label=binary_label,
                        boat_noise=(row.get("boat noise (h/m/l/n)") or "").strip(),
                        comments=(row.get("Comments") or "").strip(),
                    )
                )

    if missing_raw:
        raise FileNotFoundError(f"Missing raw audio for Begin File values: {sorted(missing_raw)}")

    logger.info("All annotation label counts: %s", dict(sorted(all_counts.items())))
    logger.info("Loaded %d scored events", len(events))
    return sorted(events, key=lambda event: (event.begin_file, event.begin_time, event.selection))


def choose_support_events(events: list[Event]) -> list[Event]:
    """Choose fixed support examples for every few-shot row.

    Parameters
    ----------
    events
        Scored Raincoast events.

    Returns
    -------
    list[Event]
        One support event each for ``pd``, ``pv`` and ``w``.

    Raises
    ------
    ValueError
        If any required support call type is absent.
    """
    support_events: list[Event] = []
    for call_type in SUPPORT_LABELS:
        matches = [event for event in events if event.call_type == call_type]
        if not matches:
            raise ValueError(f"No support event available for call type {call_type!r}")
        support_events.append(matches[0])
    return support_events


def crop_bounds(event: Event, max_seconds: float, context_seconds: float) -> tuple[float, float]:
    """Compute crop bounds around an event.

    Parameters
    ----------
    event
        Event to crop.
    max_seconds
        Maximum output clip duration.
    context_seconds
        Context to add before and after the annotation.

    Returns
    -------
    tuple[float, float]
        Start and end time in source-file seconds.
    """
    start = max(0.0, event.begin_time - context_seconds)
    end = event.end_time + context_seconds
    if end - start <= max_seconds:
        return start, end

    center = 0.5 * (event.begin_time + event.end_time)
    start = max(0.0, center - max_seconds / 2.0)
    end = start + max_seconds
    return start, end


def clip_rel_path(event: Event) -> str:
    """Return the relative output path for an event crop.

    Returns
    -------
    str
        Relative clip path below the split root.
    """
    return f"clips/{event.call_type}/{event.event_id}.wav"


def write_clip(
    event: Event,
    output_root: str,
    max_seconds: float,
    context_seconds: float,
    overwrite: bool,
) -> str:
    """Crop and write one event WAV.

    Parameters
    ----------
    event
        Event to write.
    output_root
        Root where clips are stored.
    max_seconds
        Maximum clip duration.
    context_seconds
        Context to include around the annotation.
    overwrite
        Whether to rewrite existing clips.

    Returns
    -------
    str
        Relative clip path below ``output_root``.
    """
    rel_path = clip_rel_path(event)
    out_path = f"{output_root.rstrip('/')}/{rel_path}"
    fs = filesystem_from_path(out_path)
    if not overwrite and fs.exists(out_path):
        return rel_path

    start_time, end_time = crop_bounds(
        event,
        max_seconds=max_seconds,
        context_seconds=context_seconds,
    )
    audio, sample_rate = read_audio(event.raw_audio_path, start_time=start_time, end_time=end_time)
    audio = np.asarray(audio, dtype=np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    with fs.open(out_path, "wb") as handle:
        handle.write(buffer.read())
    return rel_path


def make_row(
    event: Event,
    row_idx: int,
    support_events: list[Event],
) -> dict[str, Any]:
    """Build a BEANS-Pro multi-audio JSONL row.

    Parameters
    ----------
    event
        Query event.
    row_idx
        Row index in the output split.
    support_events
        Fixed few-shot support events.

    Returns
    -------
    dict[str, Any]
        JSONL-ready row.
    """
    row_id = f"raincoast_2025_pulsed_whistle_{row_idx:05d}"
    audio_paths = [*(clip_rel_path(support) for support in support_events), clip_rel_path(event)]
    metadata = {
        "query_event_id": event.event_id,
        "query_call_type": event.call_type,
        "query_label": event.binary_label,
        "query_begin_file": event.begin_file,
        "query_begin_time": event.begin_time,
        "query_end_time": event.end_time,
        "query_low_freq_hz": event.low_freq_hz,
        "query_high_freq_hz": event.high_freq_hz,
        "query_boat_noise": event.boat_noise,
        "query_comments": event.comments,
        "support_event_ids": [support.event_id for support in support_events],
        "support_call_types": [support.call_type for support in support_events],
    }
    return {
        "id": row_id,
        "audio_paths": audio_paths,
        "audio_ids": [row_id],
        "template_path": "raincoast/pulsed_whistle_fewshot",
        "skills": ["few_shot_call_type_classification"],
        "messages": [
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": event.binary_label},
        ],
        "task": "raincoast_pulsed_whistle_fewshot",
        "source_dataset": "Raincoast 2025 killer whale hydrophone recordings",
        "dataset_name": SPLIT_NAME,
        "license": "private",
        "metadata": json.dumps(metadata, sort_keys=True),
        "audio_path_original_sample_rate": clip_rel_path(event),
        "original_raincoast_id": event.event_id,
    }


def write_jsonl(rows: list[dict[str, Any]], output_root: str) -> str:
    """Write the output JSONL split.

    Parameters
    ----------
    rows
        Rows to serialize.
    output_root
        Output split root.

    Returns
    -------
    str
        Full JSONL path.
    """
    jsonl_path = f"{output_root.rstrip('/')}/test.jsonl"
    fs = filesystem_from_path(jsonl_path)
    with fs.open(jsonl_path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return jsonl_path


def build_dataset(
    annotation_root: str,
    raw_root: str,
    output_root: str,
    max_seconds: float,
    context_seconds: float,
    overwrite: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    """Build crops and JSONL rows for the Raincoast few-shot split.

    Parameters
    ----------
    annotation_root
        Source annotation directory.
    raw_root
        Source raw audio directory.
    output_root
        Destination split root.
    max_seconds
        Maximum crop duration.
    context_seconds
        Context to add around each annotation.
    overwrite
        Whether to overwrite existing clips.
    dry_run
        If true, only summarize rows without writing outputs.

    Returns
    -------
    list[dict[str, Any]]
        JSONL rows that were or would be written.
    """
    events = load_events(annotation_root=annotation_root, raw_root=raw_root)
    support_events = choose_support_events(events)
    support_ids = {event.event_id for event in support_events}
    query_events = [event for event in events if event.event_id not in support_ids]

    logger.info(
        "Support events: %s",
        [(event.event_id, event.call_type) for event in support_events],
    )
    logger.info(
        "Query label counts: %s",
        dict(Counter(event.binary_label for event in query_events)),
    )
    logger.info(
        "Query fine-call counts: %s",
        dict(Counter(event.call_type for event in query_events)),
    )

    rows = [make_row(event, row_idx, support_events) for row_idx, event in enumerate(query_events)]
    if dry_run:
        logger.info("Dry run: would write %d clips and %d rows", len(events), len(rows))
        return rows

    for idx, event in enumerate(events, start=1):
        write_clip(
            event=event,
            output_root=output_root,
            max_seconds=max_seconds,
            context_seconds=context_seconds,
            overwrite=overwrite,
        )
        if idx % 50 == 0:
            logger.info("Wrote or verified %d/%d clips", idx, len(events))

    jsonl_path = write_jsonl(rows, output_root=output_root)
    logger.info("Wrote %d rows to %s", len(rows), jsonl_path)
    return rows


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build Raincoast 2025 pulsed-vs-whistle few-shot multi-audio JSONL."
    )
    parser.add_argument("--annotation-root", default=ANNOTATION_ROOT)
    parser.add_argument("--raw-root", default=RAW_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--context-seconds", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build the Raincoast few-shot split."""
    args = parse_args()
    build_dataset(
        annotation_root=args.annotation_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
        max_seconds=args.max_seconds,
        context_seconds=args.context_seconds,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
