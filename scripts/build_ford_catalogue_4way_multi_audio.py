#!/usr/bin/env python3
"""Build Ford catalogue 4-way few-shot call-type classification rows.

Each row contains four labeled support clips, one per answer option, followed by
one query clip. The correct option is the support clip with the same pulsed
discrete call type as the query; the other three options are random call-type
negatives.

Usage::

    uv run python scripts/build_ford_catalogue_4way_multi_audio.py --dry-run
    uv run python scripts/build_ford_catalogue_4way_multi_audio.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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
OUTPUT_PATH = (
    "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/ford_catalogue_pulsed_discrete_4way/test.jsonl"
)
SPLIT_NAME = "ford-catalogue-pulsed-discrete-4way"

PROMPT_TEMPLATE = """You are classifying Northern Resident killer whale pulsed discrete
call types from the Ford catalogue.

Each answer option gives one labeled example clip from a call type:
A. Call type {A}
<Audio><AudioHere></Audio>
B. Call type {B}
<Audio><AudioHere></Audio>
C. Call type {C}
<Audio><AudioHere></Audio>
D. Call type {D}
<Audio><AudioHere></Audio>

Now classify the query clip:
<Audio><AudioHere></Audio>

Which option has the same call type as the query? Answer with exactly one of: A, B, C, D."""


def load_metadata(path: str) -> list[dict[str, Any]]:
    """Load Ford catalogue metadata rows.

    Parameters
    ----------
    path
        JSONL metadata path.

    Returns
    -------
    list[dict[str, Any]]
        Parsed metadata rows with audio paths and call types.
    """
    fs = filesystem_from_path(path)
    rows = []
    with fs.open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_by_call_type(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group catalogue rows by call type.

    Parameters
    ----------
    rows
        Parsed catalogue metadata rows.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Rows keyed by call type.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        call_type = str(row["call_type"])
        grouped.setdefault(call_type, []).append(row)
    return {
        key: sorted(value, key=lambda row: str(row["relative_path"]))
        for key, value in grouped.items()
    }


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


def build_prompt(option_call_types: dict[str, str]) -> str:
    """Build the user prompt for a row.

    Parameters
    ----------
    option_call_types
        Mapping from option label to call type.

    Returns
    -------
    str
        Prompt with five ``<AudioHere>`` placeholders.
    """
    return PROMPT_TEMPLATE.format(**option_call_types)


def choose_support(
    call_type: str,
    query_row: dict[str, Any],
    grouped: dict[str, list[dict[str, Any]]],
    rng: random.Random,
) -> dict[str, Any]:
    """Choose a support clip for one option.

    Parameters
    ----------
    call_type
        Option call type.
    query_row
        Query row to avoid when the option is correct.
    grouped
        Metadata rows grouped by call type.
    rng
        Seeded random generator.

    Returns
    -------
    dict[str, Any]
        Selected support metadata row.

    Raises
    ------
    ValueError
        If no support clip is available.
    """
    query_path = relative_audio_path(query_row)
    candidates = [row for row in grouped[call_type] if relative_audio_path(row) != query_path]
    if not candidates:
        raise ValueError(f"No non-query support clip available for {call_type}")
    return rng.choice(candidates)


def make_row(
    *,
    row_idx: int,
    query_row: dict[str, Any],
    option_call_types: dict[str, str],
    option_support_rows: dict[str, dict[str, Any]],
    correct_label: str,
) -> dict[str, Any]:
    """Build one BEANS-Pro multi-audio JSONL row.

    Parameters
    ----------
    row_idx
        Row index for deterministic ID generation.
    query_row
        Query catalogue metadata row.
    option_call_types
        Mapping from option labels to call types.
    option_support_rows
        Mapping from option labels to support metadata rows.
    correct_label
        Correct answer option.

    Returns
    -------
    dict[str, Any]
        JSONL-ready row.
    """
    row_id = f"ford_catalogue_pulsed_discrete_4way_{row_idx:05d}"
    audio_paths = [
        *(relative_audio_path(option_support_rows[label]) for label in LABELS),
        relative_audio_path(query_row),
    ]
    metadata = {
        "query_call_type": query_row["call_type"],
        "query_audio_path": relative_audio_path(query_row),
        "query_filename": query_row.get("filename"),
        "query_clan": query_row.get("clan"),
        "query_pod": query_row.get("pod"),
        "query_sample": query_row.get("sample"),
        "option_call_types": option_call_types,
        "option_audio_paths": {
            label: relative_audio_path(option_support_rows[label]) for label in LABELS
        },
        "correct": correct_label,
    }
    return {
        "id": row_id,
        "audio_paths": audio_paths,
        "audio_ids": [row_id],
        "template_path": "ford_catalogue/pulsed_discrete_4way",
        "skills": ["few_shot_call_type_classification", "audio_multiple_choice"],
        "messages": [
            {"role": "user", "content": build_prompt(option_call_types)},
            {"role": "assistant", "content": correct_label},
        ],
        "task": "ford_catalogue_pulsed_discrete_4way",
        "source_dataset": "Ford catalogue Northern Resident killer whale pulsed discrete calls",
        "dataset_name": SPLIT_NAME,
        "license": "private",
        "metadata": json.dumps(metadata, sort_keys=True),
        "audio_path_original_sample_rate": relative_audio_path(query_row),
        "original_ford_catalogue_id": query_row.get("filename", row_id),
    }


def build_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Build 4-way few-shot rows from catalogue metadata.

    Parameters
    ----------
    rows
        Parsed Ford catalogue metadata rows.
    seed
        Random seed for negative options and support clips.

    Returns
    -------
    list[dict[str, Any]]
        Generated JSONL rows.
    """
    grouped = group_by_call_type(rows)
    eligible_call_types = sorted(
        call_type for call_type, group in grouped.items() if len(group) >= 2
    )
    rng = random.Random(seed)
    output_rows: list[dict[str, Any]] = []

    logger.info("Loaded %d clips across %d call types", len(rows), len(grouped))
    logger.info("Eligible call types with >=2 clips: %d", len(eligible_call_types))
    logger.info("Singleton call types skipped: %d", len(grouped) - len(eligible_call_types))

    for query_call_type in eligible_call_types:
        negative_pool = [
            call_type for call_type in eligible_call_types if call_type != query_call_type
        ]
        for query_row in grouped[query_call_type]:
            option_types = [query_call_type, *rng.sample(negative_pool, k=3)]
            rng.shuffle(option_types)
            option_call_types = dict(zip(LABELS, option_types, strict=True))
            correct_label = next(
                label
                for label, call_type in option_call_types.items()
                if call_type == query_call_type
            )
            option_support_rows = {
                label: choose_support(call_type, query_row, grouped, rng)
                for label, call_type in option_call_types.items()
            }
            output_rows.append(
                make_row(
                    row_idx=len(output_rows),
                    query_row=query_row,
                    option_call_types=option_call_types,
                    option_support_rows=option_support_rows,
                    correct_label=correct_label,
                )
            )

    label_counts = Counter(row["messages"][1]["content"] for row in output_rows)
    logger.info("Generated %d rows", len(output_rows))
    logger.info("Correct option label counts: %s", dict(sorted(label_counts.items())))
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
        description="Build Ford catalogue 4-way pulsed-discrete multi-audio JSONL."
    )
    parser.add_argument("--metadata-path", default=METADATA_PATH)
    parser.add_argument("--output-path", default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build the Ford catalogue 4-way split."""
    args = parse_args()
    rows = build_rows(load_metadata(args.metadata_path), seed=args.seed)
    if args.dry_run:
        logger.info("Dry run: would write %d rows to %s", len(rows), args.output_path)
        return
    write_jsonl(rows, args.output_path)


if __name__ == "__main__":
    main()
