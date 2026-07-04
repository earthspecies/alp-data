#!/usr/bin/env python3
"""Build Raincoast 2025 NRKW few-shot call-type eval splits.

The source annotations are Marie Ana's Raven selection tables over Raincoast
2025 call-type classification snippets. Confident Ford-style call-type labels
are cropped into short WAV clips and used as Raincoast queries. The generated
BEANS-Pro multi-audio rows classify each query by choosing between labeled
support examples.

Usage::

    uv run python scripts/build_raincoast_2025_nrkw_calltype_fewshot_multi_audio.py --dry-run
    uv run python scripts/build_raincoast_2025_nrkw_calltype_fewshot_multi_audio.py --all-variants
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import random
import re
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

SEED = 42
LABELS = ("A", "B", "C", "D")
SUPPORT_SOURCES = ("ford", "raincoast")
NUM_SUPPORT_VARIANTS = (1, 2, 3)

ANNOTATION_ROOT = (
    "gs://esp-raincoast/2025/annotation/Hydrophone_recordings/"
    "call_type_classification_snippets/annotation/MA"
)
SNIPPET_AUDIO_ROOT = (
    "gs://esp-raincoast/2025/annotation/Hydrophone_recordings/"
    "call_type_classification_snippets/audio"
)
FORD_METADATA_PATH = "gs://esp-data-ingestion/ford-catalogue/metadata.jsonl"
FORD_AUDIO_ROOT = "gs://esp-data-ingestion/ford-catalogue"
GCS_OUTPUT_ROOT = (
    "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/"
    "raincoast_2025_nrkw_calltype_fewshot"
)


@dataclass(frozen=True)
class Clip:
    """Audio clip available as a query or support example."""

    clip_id: str
    call_type: str
    raw_call_type: str
    rel_path: str
    source_audio_path: str
    source_dataset: str
    source_annotation_path: str | None = None
    source_selection: str | None = None
    begin_time: float | None = None
    end_time: float | None = None
    low_freq_hz: float | None = None
    high_freq_hz: float | None = None
    notes: str = ""


def _strip_gs(path: str) -> str:
    """Return a GCS path without its scheme.

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


def normalize_call_type(call_type: str) -> str:
    """Normalize Ford subtype labels to base call types.

    Examples such as ``N01ii`` and ``N09i`` map to ``N1`` and ``N9``.

    Parameters
    ----------
    call_type
        Raw or parsed Ford-style call type.

    Returns
    -------
    str
        Normalized base call type.
    """
    stripped = call_type.strip()
    match = re.fullmatch(r"([NS])0*(\d+)(?:[ivx]+)?", stripped, flags=re.IGNORECASE)
    if match is None:
        return stripped
    prefix, number = match.groups()
    return f"{prefix.upper()}{int(number)}"


def parse_confident_ford_label(raw_call_type: str) -> tuple[str, str] | None:
    """Parse a confident Ford-style NRKW call label.

    Parameters
    ----------
    raw_call_type
        Raw value from the MA ``call_type`` column.

    Returns
    -------
    tuple[str, str] | None
        Parsed exact call label and normalized base call label, or ``None`` for
        uncertain/non-target labels.
    """
    label = raw_call_type.strip()
    if "?" in label or " or " in label.lower():
        return None
    match = re.search(r"N\d{1,2}(?:[ivx]+)?", label, flags=re.IGNORECASE)
    if match is None:
        return None
    exact_label = re.sub(r"^N0+(\d)", r"N\1", match.group(0).upper())
    return exact_label, normalize_call_type(exact_label)


def list_annotation_paths(annotation_root: str) -> list[str]:
    """List MA Raven selection tables.

    Parameters
    ----------
    annotation_root
        GCS or local folder containing ``*.txt`` selection tables.

    Returns
    -------
    list[str]
        Sorted annotation paths.
    """
    fs = filesystem_from_path(annotation_root)
    return [
        f"gs://{path}" if annotation_root.startswith("gs://") else path
        for path in sorted(fs.glob(f"{_strip_gs(annotation_root)}/*.txt"))
    ]


def source_audio_path_for_annotation(annotation_path: str, audio_root: str) -> str:
    """Return the snippet WAV path corresponding to one annotation table.

    Parameters
    ----------
    annotation_path
        MA Raven selection table path.
    audio_root
        Folder containing snippet WAVs.

    Returns
    -------
    str
        Full source snippet WAV path.
    """
    filename = annotation_path.rsplit("/", 1)[-1]
    stem = filename.removesuffix(".Table.1.selections.txt")
    return f"{audio_root.rstrip('/')}/{stem}.wav"


def read_text(path: str) -> str:
    """Read a local or remote text file.

    Parameters
    ----------
    path
        File path.

    Returns
    -------
    str
        Decoded text content.
    """
    fs = filesystem_from_path(path)
    with fs.open(path, "r") as handle:
        data = handle.read()
    return data.decode("utf-8") if isinstance(data, bytes) else data


def event_rel_path(event_id: str, call_type: str) -> str:
    """Return the output-relative path for a Raincoast event crop.

    Returns
    -------
    str
        Relative WAV path below the output root.
    """
    return f"clips/{call_type}/{event_id}.wav"


def load_ma_events(annotation_root: str, audio_root: str) -> list[Clip]:
    """Load confident Ford-style MA call-type events.

    Parameters
    ----------
    annotation_root
        Folder containing MA Raven selection tables.
    audio_root
        Folder containing the matching snippet WAVs.

    Returns
    -------
    list[Clip]
        Confident NRKW call-type events.

    Raises
    ------
    FileNotFoundError
        If an annotation table's matching source WAV is missing.
    """
    events: list[Clip] = []
    raw_counts: Counter[str] = Counter()
    parsed_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    missing_audio: list[str] = []

    for annotation_path in list_annotation_paths(annotation_root):
        source_audio_path = source_audio_path_for_annotation(annotation_path, audio_root)
        if not filesystem_from_path(source_audio_path).exists(source_audio_path):
            missing_audio.append(source_audio_path)
            continue
        audio_stem = Path(source_audio_path).stem.replace(" ", "_")
        text = read_text(annotation_path)
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        for row in reader:
            raw_call_type = (row.get("call_type") or "").strip()
            raw_counts[raw_call_type] += 1
            parsed = parse_confident_ford_label(raw_call_type)
            if parsed is None:
                skipped_counts[raw_call_type] += 1
                continue
            exact_call_type, base_call_type = parsed
            begin_time = _maybe_float(row.get("Begin Time (s)"))
            end_time = _maybe_float(row.get("End Time (s)"))
            if begin_time is None or end_time is None or end_time <= begin_time:
                skipped_counts[f"{raw_call_type}:invalid_time"] += 1
                continue
            selection = (row.get("Selection") or str(len(events) + 1)).strip()
            event_id = f"{audio_stem}_sel{int(float(selection)):04d}"
            parsed_counts[base_call_type] += 1
            events.append(
                Clip(
                    clip_id=event_id,
                    call_type=base_call_type,
                    raw_call_type=exact_call_type,
                    rel_path=event_rel_path(event_id, base_call_type),
                    source_audio_path=source_audio_path,
                    source_dataset="raincoast_2025_nrkw_ma",
                    source_annotation_path=annotation_path,
                    source_selection=selection,
                    begin_time=begin_time,
                    end_time=end_time,
                    low_freq_hz=_maybe_float(row.get("Low Freq (Hz)")),
                    high_freq_hz=_maybe_float(row.get("High Freq (Hz)")),
                    notes=(row.get("notes") or "").strip(),
                )
            )

    if missing_audio:
        raise FileNotFoundError(f"Missing source snippet WAVs: {missing_audio}")

    logger.info("Loaded %d confident MA NRKW events", len(events))
    logger.info("Raw MA call_type counts: %s", dict(sorted(raw_counts.items())))
    logger.info("Confident base call-type counts: %s", dict(sorted(parsed_counts.items())))
    logger.info("Skipped non-target/uncertain counts: %s", dict(sorted(skipped_counts.items())))
    return sorted(
        events,
        key=lambda clip: (clip.call_type, clip.source_audio_path, clip.begin_time),
    )


def ford_rel_path(row: dict[str, Any]) -> str:
    """Return the Ford catalogue-relative audio path for a metadata row.

    Returns
    -------
    str
        Relative Ford catalogue audio path.
    """
    return str(row.get("relative_path") or row.get("audio_file_path") or row["audio_file"])


def load_ford_supports(metadata_path: str, ford_audio_root: str) -> list[Clip]:
    """Load Ford catalogue clips as support candidates.

    Parameters
    ----------
    metadata_path
        Ford catalogue metadata JSONL path.
    ford_audio_root
        Root folder for Ford catalogue audio.

    Returns
    -------
    list[Clip]
        Ford support candidates with output-relative support paths.
    """
    fs = filesystem_from_path(metadata_path)
    supports: list[Clip] = []
    missing_audio: list[str] = []
    with fs.open(metadata_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw_call_type = str(row["call_type"])
            call_type = normalize_call_type(raw_call_type)
            rel_path = ford_rel_path(row)
            support_rel_path = f"support_ford/{rel_path}"
            source_audio_path = f"{ford_audio_root.rstrip('/')}/{rel_path}"
            if not filesystem_from_path(source_audio_path).exists(source_audio_path):
                missing_audio.append(source_audio_path)
                continue
            supports.append(
                Clip(
                    clip_id=str(row.get("filename") or rel_path),
                    call_type=call_type,
                    raw_call_type=raw_call_type,
                    rel_path=support_rel_path,
                    source_audio_path=source_audio_path,
                    source_dataset="ford_catalogue",
                    notes=json.dumps(row, sort_keys=True),
                )
            )
    if missing_audio:
        logger.info("Skipped %d missing Ford support files", len(missing_audio))
        logger.info("Missing Ford support files: %s", missing_audio)
    logger.info("Loaded %d Ford support clips", len(supports))
    logger.info(
        "Ford base call-type counts: %s",
        dict(sorted(Counter(clip.call_type for clip in supports).items())),
    )
    return sorted(supports, key=lambda clip: (clip.call_type, clip.rel_path))


def group_by_call_type(clips: list[Clip]) -> dict[str, list[Clip]]:
    """Group clips by normalized call type.

    Returns
    -------
    dict[str, list[Clip]]
        Clips keyed by normalized call type.
    """
    grouped: dict[str, list[Clip]] = {}
    for clip in clips:
        grouped.setdefault(clip.call_type, []).append(clip)
    return {key: sorted(value, key=lambda clip: clip.rel_path) for key, value in grouped.items()}


def crop_bounds(event: Clip, max_seconds: float, context_seconds: float) -> tuple[float, float]:
    """Compute crop bounds around a Raincoast event.

    Parameters
    ----------
    event
        Raincoast event clip.
    max_seconds
        Maximum crop duration.
    context_seconds
        Context to add around the annotation.

    Returns
    -------
    tuple[float, float]
        Start and end time in source snippet seconds.

    Raises
    ------
    ValueError
        If the event lacks time bounds.
    """
    if event.begin_time is None or event.end_time is None:
        raise ValueError(f"Raincoast event {event.clip_id} is missing time bounds")
    start = max(0.0, event.begin_time - context_seconds)
    end = event.end_time + context_seconds
    if end - start <= max_seconds:
        return start, end

    center = 0.5 * (event.begin_time + event.end_time)
    start = max(0.0, center - max_seconds / 2.0)
    return start, start + max_seconds


def write_event_clip(
    event: Clip,
    output_root: str,
    max_seconds: float,
    context_seconds: float,
    overwrite: bool,
) -> None:
    """Crop and write one Raincoast event WAV.

    Parameters
    ----------
    event
        Raincoast event to crop.
    output_root
        Output root for generated audio.
    max_seconds
        Maximum crop duration.
    context_seconds
        Context to include around the annotation.
    overwrite
        Whether to overwrite existing clips.
    """
    out_path = f"{output_root.rstrip('/')}/{event.rel_path}"
    fs = filesystem_from_path(out_path)
    if not overwrite and fs.exists(out_path):
        return

    start_time, end_time = crop_bounds(event, max_seconds, context_seconds)
    audio, sample_rate = read_audio(
        event.source_audio_path,
        start_time=start_time,
        end_time=end_time,
    )
    audio = np.asarray(audio, dtype=np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    with fs.open(out_path, "wb") as handle:
        handle.write(buffer.read())


def copy_support_clip(support: Clip, output_root: str, overwrite: bool) -> None:
    """Copy one external support clip into the generated audio root.

    Parameters
    ----------
    support
        Support clip to copy.
    output_root
        Output root for generated audio.
    overwrite
        Whether to overwrite existing support files.
    """
    out_path = f"{output_root.rstrip('/')}/{support.rel_path}"
    out_fs = filesystem_from_path(out_path)
    if not overwrite and out_fs.exists(out_path):
        return
    in_fs = filesystem_from_path(support.source_audio_path)
    with in_fs.open(support.source_audio_path, "rb") as source:
        data = source.read()
    with out_fs.open(out_path, "wb") as target:
        target.write(data)


def build_prompt(
    option_call_types: dict[str, str],
    option_supports: dict[str, list[Clip]],
    num_supports_per_option: int,
) -> str:
    """Build the few-shot MCQ prompt for one row.

    Parameters
    ----------
    option_call_types
        Mapping from answer label to normalized call type.
    option_supports
        Mapping from answer label to support clips.
    num_supports_per_option
        Maximum number of support clips shown per option.

    Returns
    -------
    str
        Prompt containing one ``<AudioHere>`` per support plus one query.
    """
    lines = [
        "You are classifying Northern Resident killer whale pulsed discrete call types.",
        "",
        "Each answer option gives labeled example clips from a call type:",
    ]
    for label in LABELS:
        lines.append(f"{label}. Call type {option_call_types[label]}")
        supports = option_supports[label]
        for support_idx in range(len(supports)):
            if num_supports_per_option == 1:
                lines.append("<Audio><AudioHere></Audio>")
            else:
                lines.append(f"Example {support_idx + 1}: <Audio><AudioHere></Audio>")
    lines.extend(
        [
            "",
            "Now classify the Raincoast query clip:",
            "<Audio><AudioHere></Audio>",
            "",
            "Which option has the same call type as the query? "
            f"Answer with exactly one of: {', '.join(LABELS)}.",
        ]
    )
    return "\n".join(lines)


def choose_supports(
    *,
    call_type: str,
    query: Clip,
    grouped_supports: dict[str, list[Clip]],
    support_source: str,
    rng: random.Random,
    count: int,
) -> list[Clip]:
    """Choose support clips for one answer option.

    Parameters
    ----------
    call_type
        Normalized call type for the answer option.
    query
        Query clip to avoid when supports are Raincoast clips.
    grouped_supports
        Support clips keyed by normalized call type.
    support_source
        Support source name.
    rng
        Seeded RNG.
    count
        Number of support clips to select.

    Returns
    -------
    list[Clip]
        Selected support clips.

    Raises
    ------
    ValueError
        If there are not enough support clips.
    """
    candidates = grouped_supports[call_type]
    if support_source == "raincoast":
        candidates = [clip for clip in candidates if clip.clip_id != query.clip_id]
        different_file = [
            clip for clip in candidates if clip.source_audio_path != query.source_audio_path
        ]
        if len(different_file) >= count:
            candidates = different_file
    if len(candidates) < count:
        raise ValueError(f"Need {count} supports for {call_type}, got {len(candidates)}")
    return rng.sample(candidates, k=count)


def variant_split_name(support_source: str, num_supports_per_option: int) -> str:
    """Return the registered split name for a variant.

    Returns
    -------
    str
        Split name used by ``BeansProMultiAudio``.
    """
    return f"raincoast-2025-nrkw-ma-{support_source}-support-4way-{num_supports_per_option}shot"


def variant_output_path(
    output_root: str,
    support_source: str,
    num_supports_per_option: int,
) -> str:
    """Return the JSONL output path for a variant.

    Returns
    -------
    str
        GCS or local JSONL output path.
    """
    split_dir = variant_split_name(support_source, num_supports_per_option).replace("-", "_")
    return f"{output_root.rstrip('/')}/{split_dir}/test.jsonl"


def make_row(
    *,
    row_idx: int,
    query: Clip,
    split_name: str,
    support_source: str,
    num_supports_per_option: int,
    option_call_types: dict[str, str],
    option_supports: dict[str, list[Clip]],
    correct_label: str,
) -> dict[str, Any]:
    """Build one BEANS-Pro multi-audio JSONL row.

    Returns
    -------
    dict[str, Any]
        JSONL-ready row.
    """
    row_id = f"{split_name.replace('-', '_')}_{row_idx:05d}"
    audio_paths = [
        *(support.rel_path for label in LABELS for support in option_supports[label]),
        query.rel_path,
    ]
    metadata = {
        "query_call_type": query.call_type,
        "query_raw_call_type": query.raw_call_type,
        "query_audio_path": query.rel_path,
        "query_source_audio_path": query.source_audio_path,
        "query_annotation_path": query.source_annotation_path,
        "query_selection": query.source_selection,
        "query_begin_time": query.begin_time,
        "query_end_time": query.end_time,
        "query_low_freq_hz": query.low_freq_hz,
        "query_high_freq_hz": query.high_freq_hz,
        "query_notes": query.notes,
        "support_source": support_source,
        "num_options": len(LABELS),
        "max_supports_per_option": num_supports_per_option,
        "option_support_counts": {
            label: len(option_supports[label]) for label in LABELS
        },
        "option_call_types": option_call_types,
        "option_audio_paths": {
            label: [support.rel_path for support in option_supports[label]]
            for label in LABELS
        },
        "option_source_audio_paths": {
            label: [support.source_audio_path for support in option_supports[label]]
            for label in LABELS
        },
        "correct": correct_label,
    }
    return {
        "id": row_id,
        "audio_paths": audio_paths,
        "audio_ids": [row_id],
        "template_path": (
            f"raincoast/nrkw_ma_{support_source}_support_4way_"
            f"{num_supports_per_option}shot"
        ),
        "skills": ["few_shot_call_type_classification", "audio_multiple_choice"],
        "messages": [
            {
                "role": "user",
                "content": build_prompt(
                    option_call_types=option_call_types,
                    option_supports=option_supports,
                    num_supports_per_option=num_supports_per_option,
                ),
            },
            {"role": "assistant", "content": correct_label},
        ],
        "task": split_name.replace("-", "_"),
        "source_dataset": "Raincoast 2025 NRKW call-type snippets annotated by Marie Ana",
        "dataset_name": split_name,
        "license": "private",
        "metadata": json.dumps(metadata, sort_keys=True),
        "audio_path_original_sample_rate": query.rel_path,
        "original_raincoast_id": query.clip_id,
    }


def build_rows(
    *,
    queries: list[Clip],
    supports: list[Clip],
    support_source: str,
    num_supports_per_option: int,
    split_name: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Build one Raincoast NRKW few-shot variant.

    Parameters
    ----------
    queries
        Raincoast query clips.
    supports
        Support clips.
    support_source
        Support source name.
    num_supports_per_option
        Number of supports per answer option.
    split_name
        Dataset split name.
    seed
        Random seed.

    Returns
    -------
    list[dict[str, Any]]
        Generated rows.

    Raises
    ------
    ValueError
        If fewer than four support-eligible call types are available.
    """
    grouped_supports = group_by_call_type(supports)
    grouped_queries = group_by_call_type(queries)
    support_types = sorted(
        call_type
        for call_type, clips in grouped_supports.items()
        if len(clips) >= num_supports_per_option
    )
    if len(support_types) < len(LABELS):
        raise ValueError(f"Need at least {len(LABELS)} support classes, got {len(support_types)}")

    query_types = []
    for call_type, query_clips in grouped_queries.items():
        if call_type not in support_types:
            continue
        if support_source == "raincoast" and len(query_clips) < num_supports_per_option + 1:
            continue
        query_types.append(call_type)

    rng = random.Random(seed)
    output_rows: list[dict[str, Any]] = []
    for query_call_type in sorted(query_types):
        negative_pool = [call_type for call_type in support_types if call_type != query_call_type]
        if len(negative_pool) < len(LABELS) - 1:
            raise ValueError(
                f"Need {len(LABELS) - 1} negatives for {query_call_type}, got {len(negative_pool)}"
            )
        for query in grouped_queries[query_call_type]:
            option_types = [query_call_type, *rng.sample(negative_pool, k=len(LABELS) - 1)]
            rng.shuffle(option_types)
            option_call_types = dict(zip(LABELS, option_types, strict=True))
            correct_label = next(
                label
                for label, call_type in option_call_types.items()
                if call_type == query_call_type
            )
            option_supports = {
                label: choose_supports(
                    call_type=call_type,
                    query=query,
                    grouped_supports=grouped_supports,
                    support_source=support_source,
                    rng=rng,
                    count=num_supports_per_option,
                )
                for label, call_type in option_call_types.items()
            }
            output_rows.append(
                make_row(
                    row_idx=len(output_rows),
                    query=query,
                    split_name=split_name,
                    support_source=support_source,
                    num_supports_per_option=num_supports_per_option,
                    option_call_types=option_call_types,
                    option_supports=option_supports,
                    correct_label=correct_label,
                )
            )

    logger.info("Generated %d rows for %s", len(output_rows), split_name)
    logger.info(
        "Query call-type counts: %s",
        dict(
            sorted(
                Counter(
                    json.loads(row["metadata"])["query_call_type"]
                    for row in output_rows
                ).items()
            )
        ),
    )
    logger.info(
        "Correct option label counts: %s",
        dict(sorted(Counter(row["messages"][1]["content"] for row in output_rows).items())),
    )
    return output_rows


def validate_rows(
    rows: list[dict[str, Any]],
    num_supports_per_option: int,
) -> None:
    """Validate generated row shape.

    Parameters
    ----------
    rows
        Generated rows.
    num_supports_per_option
        Number of supports per option.

    Raises
    ------
    ValueError
        If a generated row is internally inconsistent.
    """
    for row in rows:
        metadata = json.loads(row["metadata"])
        expected_audio_count = sum(metadata["option_support_counts"].values()) + 1
        if len(row["audio_paths"]) != expected_audio_count:
            raise ValueError(f"Unexpected audio count in {row['id']}")
        if row["messages"][0]["content"].count("<AudioHere>") != expected_audio_count:
            raise ValueError(f"Unexpected placeholder count in {row['id']}")
        if row["messages"][1]["content"] not in LABELS:
            raise ValueError(f"Invalid answer label in {row['id']}")
        if metadata["num_options"] != len(LABELS):
            raise ValueError(f"Unexpected option count in {row['id']}")
        if metadata["max_supports_per_option"] != num_supports_per_option:
            raise ValueError(f"Unexpected support count in {row['id']}")
        if any(
            count != num_supports_per_option
            for count in metadata["option_support_counts"].values()
        ):
            raise ValueError(f"Support counts are not exact in {row['id']}")
        if metadata["query_audio_path"] in metadata["option_audio_paths"][metadata["correct"]]:
            raise ValueError(f"Correct support includes query audio in {row['id']}")


def write_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    """Write generated rows to JSONL.

    Parameters
    ----------
    rows
        Rows to serialize.
    path
        GCS or local output path.
    """
    fs = filesystem_from_path(path)
    with fs.open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), path)


def write_audio_assets(
    *,
    rows: list[dict[str, Any]],
    raincoast_events: list[Clip],
    ford_supports: list[Clip],
    output_root: str,
    max_seconds: float,
    context_seconds: float,
    overwrite: bool,
) -> None:
    """Write all audio assets referenced by generated rows.

    Parameters
    ----------
    rows
        Generated JSONL rows.
    raincoast_events
        Raincoast event clips.
    ford_supports
        Ford support clips.
    output_root
        Output root for generated audio.
    max_seconds
        Maximum Raincoast crop duration.
    context_seconds
        Context around Raincoast events.
    overwrite
        Whether to overwrite existing audio assets.
    """
    referenced_paths = {path for row in rows for path in row["audio_paths"]}
    raincoast_by_path = {event.rel_path: event for event in raincoast_events}
    ford_by_path = {support.rel_path: support for support in ford_supports}

    raincoast_to_write = [
        raincoast_by_path[path] for path in sorted(referenced_paths) if path in raincoast_by_path
    ]
    ford_to_copy = [ford_by_path[path] for path in sorted(referenced_paths) if path in ford_by_path]

    logger.info("Writing/verifying %d Raincoast crops", len(raincoast_to_write))
    for idx, event in enumerate(raincoast_to_write, start=1):
        write_event_clip(
            event=event,
            output_root=output_root,
            max_seconds=max_seconds,
            context_seconds=context_seconds,
            overwrite=overwrite,
        )
        if idx % 25 == 0:
            logger.info("Wrote or verified %d/%d Raincoast crops", idx, len(raincoast_to_write))

    logger.info("Copying/verifying %d Ford support clips", len(ford_to_copy))
    for idx, support in enumerate(ford_to_copy, start=1):
        copy_support_clip(support=support, output_root=output_root, overwrite=overwrite)
        if idx % 25 == 0:
            logger.info("Copied or verified %d/%d Ford support clips", idx, len(ford_to_copy))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build Raincoast 2025 NRKW call-type few-shot JSONL splits."
    )
    parser.add_argument("--annotation-root", default=ANNOTATION_ROOT)
    parser.add_argument("--snippet-audio-root", default=SNIPPET_AUDIO_ROOT)
    parser.add_argument("--ford-metadata-path", default=FORD_METADATA_PATH)
    parser.add_argument("--ford-audio-root", default=FORD_AUDIO_ROOT)
    parser.add_argument("--output-root", default=GCS_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--context-seconds", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--support-source",
        choices=(*SUPPORT_SOURCES, "all"),
        default="ford",
    )
    parser.add_argument(
        "--num-supports-per-option",
        type=int,
        choices=NUM_SUPPORT_VARIANTS,
        default=1,
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Build all 4-way x 1/2/3-shot variants for Ford and Raincoast supports.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def variant_specs(args: argparse.Namespace) -> list[tuple[str, int, str, str]]:
    """Return requested variant specs.

    Parameters
    ----------
    args
        Parsed CLI arguments.

    Returns
    -------
    list[tuple[str, int, str, str]]
        Tuples of support source, support count, split name, and output path.
    """
    if args.all_variants:
        return [
            (
                support_source,
                num_supports,
                variant_split_name(support_source, num_supports),
                variant_output_path(args.output_root, support_source, num_supports),
            )
            for support_source in SUPPORT_SOURCES
            for num_supports in NUM_SUPPORT_VARIANTS
        ]
    support_sources = SUPPORT_SOURCES if args.support_source == "all" else (args.support_source,)
    return [
        (
            support_source,
            args.num_supports_per_option,
            variant_split_name(support_source, args.num_supports_per_option),
            variant_output_path(args.output_root, support_source, args.num_supports_per_option),
        )
        for support_source in support_sources
    ]


def main() -> None:
    """Build requested Raincoast NRKW few-shot splits."""
    args = parse_args()
    raincoast_events = load_ma_events(args.annotation_root, args.snippet_audio_root)
    ford_supports = load_ford_supports(args.ford_metadata_path, args.ford_audio_root)
    support_by_source = {
        "ford": ford_supports,
        "raincoast": raincoast_events,
    }

    all_built_rows: list[dict[str, Any]] = []
    for support_source, num_supports, split_name, output_path in variant_specs(args):
        built_rows = build_rows(
            queries=raincoast_events,
            supports=support_by_source[support_source],
            support_source=support_source,
            num_supports_per_option=num_supports,
            split_name=split_name,
            seed=args.seed,
        )
        validate_rows(built_rows, num_supports_per_option=num_supports)
        all_built_rows.extend(built_rows)
        if args.dry_run:
            logger.info("Dry run: would write %d rows to %s", len(built_rows), output_path)
            continue
        write_jsonl(built_rows, output_path)

    if args.dry_run:
        logger.info("Dry run: would write audio assets for %d total rows", len(all_built_rows))
        return
    write_audio_assets(
        rows=all_built_rows,
        raincoast_events=raincoast_events,
        ford_supports=ford_supports,
        output_root=args.output_root,
        max_seconds=args.max_seconds,
        context_seconds=args.context_seconds,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
