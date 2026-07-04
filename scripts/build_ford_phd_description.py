#!/usr/bin/env python3
"""Build Ford PhD 4-way acoustic-description call-type rows.

Each row contains one Ford catalogue query clip and four acoustic descriptions:
the true call type plus three random negative call types. The answer is the
letter whose description matches the query clip's base call type.

Usage::

    uv run python scripts/build_ford_phd_description.py --dry-run
    uv run python scripts/build_ford_phd_description.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from esp_data.io import filesystem_from_path  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

SEED = 42
LABELS = ("A", "B", "C", "D")
METADATA_PATH = "gs://esp-data-ingestion/ford-catalogue/metadata.jsonl"
DESCRIPTION_PATH = (
    REPO_ROOT
    / "esp-research/projects/NatureLM-audio-v1.5/config/datasets/"
    "ford_phd_call_type_descriptions.json"
)
OUTPUT_PATH = "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/ford_phd_description/test.jsonl"
SPLIT_NAME = "ford-phd-description"

PROMPT_TEMPLATE = """<Audio><AudioHere></Audio>

You are classifying Northern Resident killer whale pulsed discrete call types
from the Ford catalogue.
Which acoustic description best matches the call in the audio?

A: {A}
B: {B}
C: {C}
D: {D}

Answer with exactly one of: A, B, C, D."""


def normalize_call_type(call_type: str) -> str:
    """Normalize Ford catalogue subtype labels to base call types.

    Examples such as ``N01i`` and ``N08iii`` map to ``N1`` and ``N8`` so
    catalogue subtype clips can use the base descriptions from Ford's thesis.

    Parameters
    ----------
    call_type
        Raw Ford catalogue or description call type.

    Returns
    -------
    str
        Normalized base call type.
    """
    stripped = call_type.strip()
    match = re.fullmatch(r"([NS])(\d+)(?:[ivx]+)?", stripped, flags=re.IGNORECASE)
    if match is None:
        return stripped
    prefix, number = match.groups()
    return f"{prefix.upper()}{int(number)}"


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load JSONL records from a local or remote path.

    Parameters
    ----------
    path
        Path to a JSONL file.

    Returns
    -------
    list[dict[str, Any]]
        Parsed JSONL records.
    """
    fs = filesystem_from_path(path)
    rows: list[dict[str, Any]] = []
    with fs.open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_descriptions(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load Ford PhD acoustic descriptions keyed by call type.

    Parameters
    ----------
    path
        Local or remote JSON file containing description records.

    Returns
    -------
    dict[str, dict[str, Any]]
        Description records keyed by call type.

    Raises
    ------
    ValueError
        If a description is missing a call type.
    """
    path_str = str(path)
    fs = filesystem_from_path(path_str)
    with fs.open(path_str, "r") as handle:
        records = json.load(handle)

    descriptions: dict[str, dict[str, Any]] = {}
    empty_call_types = []
    for record in records:
        call_type = str(record.get("call_type") or "").strip()
        acoustic_description = str(record.get("acoustic_description") or "").strip()
        if not call_type:
            raise ValueError(f"Invalid Ford description record: {record}")
        normalized_call_type = normalize_call_type(call_type)
        if not acoustic_description:
            empty_call_types.append(normalized_call_type)
            continue
        if normalized_call_type in descriptions:
            raise ValueError(f"Duplicate description for call type {normalized_call_type!r}")
        descriptions[normalized_call_type] = record
    empty_without_usable = sorted(set(empty_call_types) - set(descriptions))
    if empty_without_usable:
        logger.info(
            "Skipped %d call types with empty acoustic descriptions: %s",
            len(empty_without_usable),
            ", ".join(empty_without_usable),
        )
    return descriptions


def relative_audio_path(row: dict[str, Any]) -> str:
    """Return the catalogue-relative audio path for a metadata row.

    Parameters
    ----------
    row
        Ford catalogue metadata row.

    Returns
    -------
    str
        Path relative to ``gs://esp-data-ingestion/ford-catalogue/``.
    """
    return str(row.get("relative_path") or row.get("audio_file_path") or row["audio_file"])


def audio_exists(audio_root: str, relative_path: str) -> bool:
    """Return whether a Ford catalogue audio object exists.

    Parameters
    ----------
    audio_root
        Ford catalogue audio root.
    relative_path
        Catalogue-relative audio path.

    Returns
    -------
    bool
        Whether the audio object exists.
    """
    path = f"{audio_root.rstrip('/')}/{relative_path}"
    fs = filesystem_from_path(path)
    return fs.exists(path)


def build_prompt(option_descriptions: dict[str, str]) -> str:
    """Build the user prompt for one description-matching row.

    Parameters
    ----------
    option_descriptions
        Mapping from option labels to acoustic descriptions.

    Returns
    -------
    str
        Prompt with one audio placeholder and four text options.
    """
    return PROMPT_TEMPLATE.format(**option_descriptions)


def make_row(
    *,
    row_idx: int,
    query_row: dict[str, Any],
    option_call_types: dict[str, str],
    descriptions: dict[str, dict[str, Any]],
    correct_label: str,
) -> dict[str, Any]:
    """Build one BEANS-Pro description JSONL row.

    Parameters
    ----------
    row_idx
        Row index for deterministic ID generation.
    query_row
        Ford catalogue metadata row for the query clip.
    option_call_types
        Mapping from option labels to call types.
    descriptions
        Ford PhD description records keyed by call type.
    correct_label
        Correct answer option.

    Returns
    -------
    dict[str, Any]
        JSONL-ready BEANS-Pro row.
    """
    row_id = f"ford_phd_description_{row_idx:05d}"
    query_call_type = str(query_row["call_type"])
    query_base_call_type = normalize_call_type(query_call_type)
    option_descriptions = {
        label: str(descriptions[call_type]["acoustic_description"])
        for label, call_type in option_call_types.items()
    }
    metadata = {
        "query_call_type": query_call_type,
        "query_base_call_type": query_base_call_type,
        "query_audio_path": relative_audio_path(query_row),
        "query_filename": query_row.get("filename"),
        "query_clan": query_row.get("clan"),
        "query_pod": query_row.get("pod"),
        "query_sample": query_row.get("sample"),
        "option_call_types": option_call_types,
        "option_source_pages": {
            label: descriptions[call_type].get("source_pages")
            for label, call_type in option_call_types.items()
        },
        "correct": correct_label,
        "species": "Orcinus orca",
        "species_common": "Northern Resident killer whale",
        "description_source": "Ford PhD call type descriptions",
    }
    return {
        "id": row_id,
        "instruction": build_prompt(option_descriptions),
        "output": correct_label,
        "audio_path_original_sample_rate": relative_audio_path(query_row),
        "task": "ford_phd_description",
        "source_dataset": "Ford catalogue Northern Resident killer whale pulsed discrete calls",
        "dataset_name": SPLIT_NAME,
        "license": "private",
        "metadata": json.dumps(metadata, sort_keys=True),
        "original_ford_catalogue_id": query_row.get("filename", row_id),
    }


def build_rows(
    rows: list[dict[str, Any]],
    descriptions: dict[str, dict[str, Any]],
    seed: int,
    audio_root: str,
) -> list[dict[str, Any]]:
    """Build 4-way description rows from Ford catalogue metadata.

    Parameters
    ----------
    rows
        Parsed Ford catalogue metadata rows.
    descriptions
        Ford PhD description records keyed by call type.
    seed
        Random seed for option sampling and ordering.
    audio_root
        Root path used to validate Ford catalogue audio objects.

    Returns
    -------
    list[dict[str, Any]]
        Generated JSONL rows.

    Raises
    ------
    ValueError
        If fewer than four call types have usable descriptions.
    """
    eligible_rows = []
    missing_audio_paths = []
    for row in rows:
        if normalize_call_type(str(row.get("call_type") or "")) not in descriptions:
            continue
        audio_path = relative_audio_path(row)
        if not audio_exists(audio_root, audio_path):
            missing_audio_paths.append(audio_path)
            continue
        eligible_rows.append(row)
    eligible_call_types = sorted(
        {normalize_call_type(str(row["call_type"])) for row in eligible_rows}
    )
    if len(eligible_call_types) < len(LABELS):
        raise ValueError(
            f"Need at least {len(LABELS)} described call types, got {len(eligible_call_types)}"
        )

    missing_description_counts = Counter(
        normalize_call_type(str(row.get("call_type") or ""))
        for row in rows
        if normalize_call_type(str(row.get("call_type") or "")) not in descriptions
    )
    if missing_description_counts:
        logger.info(
            "Skipped %d clips without descriptions across %d call types",
            sum(missing_description_counts.values()),
            len(missing_description_counts),
        )
    if missing_audio_paths:
        logger.info(
            "Skipped %d described clips with missing audio: %s",
            len(missing_audio_paths),
            ", ".join(sorted(missing_audio_paths)),
        )

    rng = random.Random(seed)
    output_rows: list[dict[str, Any]] = []
    for query_row in sorted(
        eligible_rows,
        key=lambda row: (
            normalize_call_type(str(row["call_type"])),
            str(row["call_type"]),
            relative_audio_path(row),
        ),
    ):
        query_call_type = normalize_call_type(str(query_row["call_type"]))
        negative_pool = [
            call_type for call_type in eligible_call_types if call_type != query_call_type
        ]
        option_types = [query_call_type, *rng.sample(negative_pool, k=3)]
        rng.shuffle(option_types)
        option_call_types = dict(zip(LABELS, option_types, strict=True))
        correct_label = next(
            label
            for label, call_type in option_call_types.items()
            if call_type == query_call_type
        )
        output_rows.append(
            make_row(
                row_idx=len(output_rows),
                query_row=query_row,
                option_call_types=option_call_types,
                descriptions=descriptions,
                correct_label=correct_label,
            )
        )

    logger.info("Loaded %d Ford catalogue clips", len(rows))
    logger.info("Eligible described clips: %d", len(eligible_rows))
    logger.info("Eligible described call types: %d", len(eligible_call_types))
    logger.info(
        "Query raw call-type counts: %s",
        dict(sorted(Counter(str(row["call_type"]) for row in eligible_rows).items())),
    )
    logger.info(
        "Query base call-type counts: %s",
        dict(
            sorted(
                Counter(
                    normalize_call_type(str(row["call_type"])) for row in eligible_rows
                ).items()
            )
        ),
    )
    logger.info(
        "Correct option label counts: %s",
        dict(sorted(Counter(row["output"] for row in output_rows).items())),
    )
    return output_rows


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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build Ford PhD 4-way acoustic-description JSONL."
    )
    parser.add_argument("--metadata-path", default=METADATA_PATH)
    parser.add_argument("--description-path", default=str(DESCRIPTION_PATH))
    parser.add_argument("--output-path", default=OUTPUT_PATH)
    parser.add_argument(
        "--audio-root",
        default="gs://esp-data-ingestion/ford-catalogue/",
        help="Ford catalogue audio root used to filter stale metadata rows.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build the Ford PhD description split."""
    args = parse_args()
    rows = build_rows(
        rows=load_jsonl(args.metadata_path),
        descriptions=load_descriptions(args.description_path),
        seed=args.seed,
        audio_root=args.audio_root,
    )
    if args.dry_run:
        logger.info("Dry run: would write %d rows to %s", len(rows), args.output_path)
        return
    write_jsonl(rows, args.output_path)


if __name__ == "__main__":
    main()
