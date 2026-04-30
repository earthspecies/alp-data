#!/usr/bin/env python3
"""Build 3-way few-shot gibbon detection eval splits for BEANS-Pro.

Uses the BEANS-Zero gibbons split to create fixed-label few-shot detection
examples with three support clips (A/B/C) and an optional background
environment clip. The query answer is one of ``A``, ``B``, ``C``, or
``None``.

Produces two splits:
- ``gibbon-fewshot-detection``: all queryable clips after support holdout
- ``gibbon-fewshot-detection-balanced``: all present clips + matched ``None``

Usage::

    uv run python scripts/build_beans_pro_gibbon_detection.py
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
from collections import Counter, defaultdict
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
BG_PROB = 0.5
N_SUPPORT_PER_TYPE = 2
LABEL_TO_TYPE: dict[str, str] = {
    "A": "Multiple pulse gibbon call",
    "B": "Single pulse gibbon call",
    "C": "Gibbon duet",
}
TYPE_TO_LABEL = {value: key for key, value in LABEL_TO_TYPE.items()}
GIBBONS_JSONL = "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/gibbons_test.jsonl"


def build_instruction(has_background: bool) -> str:
    """Build the few-shot detection prompt.

    Parameters
    ----------
    has_background : bool
        Whether to include a background environment clip before the query.

    Returns
    -------
    str
        Prompt with ``<AudioHere>`` placeholders matching audio order.
    """
    lines = [
        "Here are examples of 3 sounds.",
        "",
        "A: <Audio><AudioHere></Audio>",
        "B: <Audio><AudioHere></Audio>",
        "C: <Audio><AudioHere></Audio>",
    ]
    if has_background:
        lines.extend(
            [
                "",
                "Here is the background environment: <Audio><AudioHere></Audio>",
            ]
        )
    lines.extend(
        [
            "",
            "Which of the above sounds are present in this recording, if any?",
            "<Audio><AudioHere></Audio>",
        ]
    )
    return "\n".join(lines)


def load_gibbons() -> list[dict[str, Any]]:
    """Load the BEANS-Zero gibbons JSONL.

    Returns
    -------
    list[dict[str, Any]]
        Parsed JSONL records.
    """
    logger.info("Loading %s", GIBBONS_JSONL)
    fs = filesystem_from_path(GIBBONS_JSONL)
    rows: list[dict[str, Any]] = []
    with fs.open(str(GIBBONS_JSONL), "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    logger.info("Loaded %d rows", len(rows))
    return rows


def group_rows_by_label(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group rows by BEANS-Zero output label.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Gibbon source rows.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Mapping from output label to source rows.

    Raises
    ------
    ValueError
        If the source data contains unexpected output labels.
    """
    allowed = {*LABEL_TO_TYPE.values(), "None"}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output = row["output"]
        if output not in allowed:
            raise ValueError(f"Unexpected gibbon output label: {output!r}")
        grouped[output].append(row)
    return dict(grouped)


def hold_out_support(
    grouped_rows: dict[str, list[dict[str, Any]]],
    rng: random.Random,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create deterministic support pools and query rows.

    Parameters
    ----------
    grouped_rows : dict[str, list[dict[str, Any]]]
        Source rows grouped by output label.
    rng : random.Random
        Seeded random generator.

    Returns
    -------
    tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]
        Support pool by call type, present query rows, and ``None`` query rows.

    Raises
    ------
    ValueError
        If a target call type does not have enough clips for support + query.
    """
    support_pool: dict[str, list[dict[str, Any]]] = {}
    present_queries: list[dict[str, Any]] = []

    for call_type in LABEL_TO_TYPE.values():
        candidates = list(grouped_rows.get(call_type, []))
        if len(candidates) <= N_SUPPORT_PER_TYPE:
            raise ValueError(
                f"{call_type!r} has only {len(candidates)} clips; "
                f"need more than {N_SUPPORT_PER_TYPE}."
            )
        rng.shuffle(candidates)
        support_pool[call_type] = candidates[:N_SUPPORT_PER_TYPE]
        present_queries.extend(candidates[N_SUPPORT_PER_TYPE:])

    none_queries = list(grouped_rows.get("None", []))
    return support_pool, present_queries, none_queries


def build_audio_paths(
    support_pool: dict[str, list[dict[str, Any]]],
    has_background: bool,
    target_row: dict[str, Any],
    none_pool: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[list[str], dict[str, str]]:
    """Build ordered audio paths for one example.

    Parameters
    ----------
    support_pool : dict[str, list[dict[str, Any]]]
        Held-out support rows for each of the three call types.
    has_background : bool
        Whether to include a background environment clip.
    target_row : dict[str, Any]
        Source query row.
    none_pool : list[dict[str, Any]]
        Candidate ``None`` rows for background selection.
    rng : random.Random
        Seeded random generator.

    Returns
    -------
    tuple[list[str], dict[str, str]]
        Ordered audio paths and label-to-type mapping metadata.

    Raises
    ------
    ValueError
        If no background candidate is available when requested.
    """
    audio_paths = [
        rng.choice(support_pool[LABEL_TO_TYPE[label]])["audio_path_32KHz"]
        for label in ("A", "B", "C")
    ]

    if has_background:
        if (
            len(none_pool) == 1
            and none_pool[0]["audio_path_32KHz"] == target_row["audio_path_32KHz"]
        ):
            raise ValueError("No valid background environment clips available.")
        background_row = rng.choice(none_pool)
        while background_row["audio_path_32KHz"] == target_row["audio_path_32KHz"]:
            background_row = rng.choice(none_pool)
        audio_paths.append(background_row["audio_path_32KHz"])

    audio_paths.append(target_row["audio_path_32KHz"])
    return audio_paths, LABEL_TO_TYPE


def make_row(
    *,
    split_name: str,
    row_idx: int,
    target_row: dict[str, Any],
    support_pool: dict[str, list[dict[str, Any]]],
    none_pool: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, Any]:
    """Build one output JSONL row.

    Parameters
    ----------
    split_name : str
        Output split name.
    row_idx : int
        Deterministic row index.
    target_row : dict[str, Any]
        Source query row.
    support_pool : dict[str, list[dict[str, Any]]]
        Held-out support rows for each call type.
    none_pool : list[dict[str, Any]]
        Candidate background rows.
    rng : random.Random
        Seeded random generator.

    Returns
    -------
    dict[str, Any]
        JSONL-ready row.
    """
    has_background = rng.random() < BG_PROB
    audio_paths, option_types = build_audio_paths(
        support_pool=support_pool,
        has_background=has_background,
        target_row=target_row,
        none_pool=none_pool,
        rng=rng,
    )
    answer = TYPE_TO_LABEL.get(target_row["output"], "None")
    row_id = f"{split_name.replace('-', '_')}_{row_idx:05d}"
    metadata = {
        "option_types": option_types,
        "answer": answer,
        "target_output": target_row["output"],
        "has_background": has_background,
        "source_file": target_row.get("file_name", ""),
    }
    return {
        "id": row_id,
        "audio_paths": audio_paths,
        "audio_ids": [row_id],
        "template_path": "audio_synth/fewshot_detection_v2",
        "skills": ["multilabel_detection", "few_shot_detection"],
        "messages": [
            {"role": "user", "content": build_instruction(has_background)},
            {"role": "assistant", "content": answer},
        ],
        "task": "fewshot_detection",
        "source_dataset": "Hainan Gibbons",
        "dataset_name": split_name,
        "license": "CC-BY-NC-SA",
        "metadata": json.dumps(metadata),
        "audio_path_original_sample_rate": target_row["audio_path_original_sample_rate"],
        "original_beans_zero_id": target_row["id"],
    }


def build_rows(
    split_name: str,
    query_rows: list[dict[str, Any]],
    support_pool: dict[str, list[dict[str, Any]]],
    none_pool: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Generate all rows for a split.

    Parameters
    ----------
    split_name : str
        Output split name.
    query_rows : list[dict[str, Any]]
        Source query rows to convert.
    support_pool : dict[str, list[dict[str, Any]]]
        Held-out support rows for each call type.
    none_pool : list[dict[str, Any]]
        Candidate background rows.
    rng : random.Random
        Seeded random generator.

    Returns
    -------
    list[dict[str, Any]]
        Generated rows.
    """
    rows = [
        make_row(
            split_name=split_name,
            row_idx=i,
            target_row=target_row,
            support_pool=support_pool,
            none_pool=none_pool,
            rng=rng,
        )
        for i, target_row in enumerate(query_rows)
    ]
    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row_id = f"{split_name.replace('-', '_')}_{i:05d}"
        row["id"] = row_id
        row["audio_ids"] = [row_id]
    return rows


def build_balanced_rows(
    full_rows: list[dict[str, Any]],
    split_name: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Build a balanced present-vs-none subset.

    Parameters
    ----------
    full_rows : list[dict[str, Any]]
        Full generated split.
    split_name : str
        Balanced split name.
    rng : random.Random
        Seeded random generator.

    Returns
    -------
    list[dict[str, Any]]
        Balanced rows.

    Raises
    ------
    ValueError
        If there are fewer ``None`` rows than present rows.
    """
    present_rows = [row for row in full_rows if row["messages"][1]["content"] != "None"]
    none_rows = [row for row in full_rows if row["messages"][1]["content"] == "None"]
    if len(none_rows) < len(present_rows):
        raise ValueError(
            f"Need at least {len(present_rows)} None rows, found {len(none_rows)}."
        )

    balanced_rows = [
        copy.deepcopy(row)
        for row in [*present_rows, *rng.sample(none_rows, len(present_rows))]
    ]
    rng.shuffle(balanced_rows)
    for i, row in enumerate(balanced_rows):
        row["id"] = f"{split_name.replace('-', '_')}_{i:05d}"
        row["audio_ids"] = [row["id"]]
        row["dataset_name"] = split_name
    return balanced_rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute row-level summary statistics.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Generated split rows.

    Returns
    -------
    dict[str, Any]
        Summary counts for answers and background usage.
    """
    answers = Counter(row["messages"][1]["content"] for row in rows)
    background_count = 0
    for row in rows:
        metadata = json.loads(row["metadata"])
        if metadata["has_background"]:
            background_count += 1
    return {
        "n_rows": len(rows),
        "answers": dict(sorted(answers.items())),
        "background_rows": background_count,
        "background_fraction": (background_count / len(rows)) if rows else 0.0,
    }


def verify_support_query_disjoint(
    support_pool: dict[str, list[dict[str, Any]]],
    query_rows: list[dict[str, Any]],
) -> None:
    """Verify support rows are excluded from the query set.

    Parameters
    ----------
    support_pool : dict[str, list[dict[str, Any]]]
        Held-out support rows.
    query_rows : list[dict[str, Any]]
        Source query rows.

    Raises
    ------
    ValueError
        If any support clip also appears in the query set.
    """
    support_paths = {
        row["audio_path_32KHz"]
        for rows in support_pool.values()
        for row in rows
    }
    query_paths = {row["audio_path_32KHz"] for row in query_rows}
    overlap = support_paths & query_paths
    if overlap:
        raise ValueError(f"Support/query overlap detected for {len(overlap)} clips.")


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows to a JSONL file.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Output records.
    path : Path
        Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), path)


def main() -> None:
    """Build both gibbon detection splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "beans_pro_gibbon_detection",
        help="Directory for output JSONL files.",
    )
    args = parser.parse_args()

    rng = random.Random(SEED)
    source_rows = load_gibbons()
    grouped_rows = group_rows_by_label(source_rows)
    support_pool, present_queries, none_queries = hold_out_support(grouped_rows, rng)
    full_query_rows = [*present_queries, *none_queries]
    verify_support_query_disjoint(support_pool, full_query_rows)

    logger.info(
        "Support holdout: %s",
        {
            TYPE_TO_LABEL[call_type]: len(rows)
            for call_type, rows in support_pool.items()
        },
    )
    logger.info(
        "Query rows: %d present, %d none",
        len(present_queries),
        len(none_queries),
    )

    full_rows = build_rows(
        split_name="gibbon-fewshot-detection",
        query_rows=full_query_rows,
        support_pool=support_pool,
        none_pool=none_queries,
        rng=rng,
    )
    balanced_rows = build_balanced_rows(
        full_rows=full_rows,
        split_name="gibbon-fewshot-detection-balanced",
        rng=rng,
    )

    write_jsonl(full_rows, args.output_dir / "gibbon_fewshot_detection.jsonl")
    write_jsonl(
        balanced_rows,
        args.output_dir / "gibbon_fewshot_detection_balanced.jsonl",
    )

    logger.info("Full summary: %s", summarize_rows(full_rows))
    logger.info("Balanced summary: %s", summarize_rows(balanced_rows))


if __name__ == "__main__":
    main()
