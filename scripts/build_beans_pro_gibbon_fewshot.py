#!/usr/bin/env python3
"""Build few-shot gibbon detection evaluation splits for BEANS-Pro multi-audio.

Creates binary detection tasks from the BEANS-Zero gibbons dataset: given
2 support clips of a target gibbon call type, determine whether a query clip
contains that call type.

Splits produced:
- ``gibbon-fewshot-multipulse``: Multiple pulse gibbon call (~736 examples)
- ``gibbon-fewshot-singlepulse``: Single pulse gibbon call (~80 examples)
- ``gibbon-fewshot-duet``: Gibbon duet (~40 examples)
- ``gibbon-fewshot-tiny``: Small balanced mix for pipeline testing (~24 examples)

Usage::

    uv run python scripts/build_beans_pro_gibbon_fewshot.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from esp_data.io import filesystem_from_path  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

SEED = 42
N_SUPPORT = 2
GIBBONS_JSONL = "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/gibbons_test.jsonl"

INSTRUCTION_TEMPLATE = (
    "Here are example(s) of a target call type.\n"
    "{support_block}\n"
    "Does the following recording contain this target call type?\n"
    "<Audio><AudioHere></Audio>"
)

CALL_TYPES = {
    "gibbon-fewshot-multipulse": "Multiple pulse gibbon call",
    "gibbon-fewshot-singlepulse": "Single pulse gibbon call",
    "gibbon-fewshot-duet": "Gibbon duet",
}


# ── Helpers ──────────────────────────────────────────────────────────────


def load_gibbons() -> list[dict]:
    """Load all rows from the BEANS-Zero gibbons JSONL.

    Returns
    -------
    list[dict]
        List of parsed JSON rows.
    """
    logger.info("Loading %s", GIBBONS_JSONL)
    fs = filesystem_from_path(GIBBONS_JSONL)
    rows = []
    with fs.open(str(GIBBONS_JSONL), "r") as f:
        for line in f:
            rows.append(json.loads(line))
    logger.info("  Loaded %d rows", len(rows))
    return rows


def build_instruction(n_support: int) -> str:
    """Build the user instruction with the correct number of support placeholders.

    Parameters
    ----------
    n_support
        Number of support audio examples.

    Returns
    -------
    str
        Formatted instruction string with ``<AudioHere>`` placeholders.
    """
    support_lines = [
        "<Audio><AudioHere></Audio>" for _ in range(n_support)
    ]
    support_block = "\n".join(support_lines)
    return INSTRUCTION_TEMPLATE.format(support_block=support_block)


def make_row(
    *,
    split_name: str,
    call_type: str,
    support_paths: list[str],
    query_path: str,
    label: str,
    row_idx: int,
    source_row: dict,
) -> dict:
    """Build a single JSONL row in DRASDIC-compatible multi-audio format.

    Parameters
    ----------
    split_name
        The split this row belongs to.
    call_type
        The target call type string.
    support_paths
        List of audio paths for support examples.
    query_path
        Audio path for the query clip.
    label
        ``"Yes"`` or ``"No"``.
    row_idx
        Row index for deterministic ID generation.
    source_row
        Original BEANS-Zero row (for metadata).

    Returns
    -------
    dict
        A JSONL-ready row.
    """
    audio_paths = [*support_paths, query_path]
    instruction = build_instruction(len(support_paths))

    row_id = f"{split_name.replace('-', '_')}_{row_idx:05d}"
    metadata = {
        "call_type": call_type,
        "label": label,
        "source_file": source_row.get("file_name", ""),
    }

    return {
        "id": row_id,
        "audio_paths": audio_paths,
        "audio_ids": [row_id],
        "template_path": "audio_synth/binary_audio",
        "skills": ["binary_audio", "few_shot_detection"],
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": label},
        ],
        "task": "call_type_binary",
        "source_dataset": "Hainan Gibbons",
        "dataset_name": split_name,
        "license": "CC-BY-NC-SA",
        "metadata": json.dumps(metadata),
        "audio_path_original_sample_rate": query_path,
    }


def build_split(
    split_name: str,
    call_type: str,
    all_rows: list[dict],
    rng: random.Random,
) -> list[dict]:
    """Build a full balanced split for one call type.

    Parameters
    ----------
    split_name
        Name for the output split.
    call_type
        Target call type label.
    all_rows
        All BEANS-Zero gibbon rows.
    rng
        Seeded random generator.

    Returns
    -------
    list[dict]
        List of JSONL rows for this split.
    """
    positives = [r for r in all_rows if r["output"] == call_type]
    negatives = [r for r in all_rows if r["output"] == "None"]

    if len(positives) < N_SUPPORT + 1:
        logger.warning(
            "  %s: only %d positives, need at least %d (support + 1 query). Skipping.",
            split_name, len(positives), N_SUPPORT + 1,
        )
        return []

    # Shuffle positives, hold out first N_SUPPORT as support
    shuffled_pos = list(positives)
    rng.shuffle(shuffled_pos)
    support_rows = shuffled_pos[:N_SUPPORT]
    query_pos = shuffled_pos[N_SUPPORT:]

    support_paths = [r["audio_path_32KHz"] for r in support_rows]

    # Balance negatives to match positive query count
    n_neg = len(query_pos)
    shuffled_neg = list(negatives)
    rng.shuffle(shuffled_neg)
    query_neg = shuffled_neg[:n_neg]

    logger.info(
        "  %s: %d support, %d pos queries, %d neg queries",
        split_name, len(support_paths), len(query_pos), len(query_neg),
    )

    rows = []
    idx = 0
    for r in query_pos:
        rows.append(make_row(
            split_name=split_name,
            call_type=call_type,
            support_paths=support_paths,
            query_path=r["audio_path_32KHz"],
            label="Yes",
            row_idx=idx,
            source_row=r,
        ))
        idx += 1

    for r in query_neg:
        rows.append(make_row(
            split_name=split_name,
            call_type=call_type,
            support_paths=support_paths,
            query_path=r["audio_path_32KHz"],
            label="No",
            row_idx=idx,
            source_row=r,
        ))
        idx += 1

    rng.shuffle(rows)
    return rows


def build_tiny_split(
    all_splits: dict[str, list[dict]],
    rng: random.Random,
) -> list[dict]:
    """Build a small balanced split for pipeline testing.

    Takes 4 Yes + 4 No from each call type (or all if fewer available).

    Parameters
    ----------
    all_splits
        Dict mapping split name to its full rows.
    rng
        Seeded random generator.

    Returns
    -------
    list[dict]
        Small balanced split.
    """
    tiny = []
    for _split_name, rows in all_splits.items():
        yes_rows = [r for r in rows if r["messages"][1]["content"] == "Yes"]
        no_rows = [r for r in rows if r["messages"][1]["content"] == "No"]
        n = min(4, len(yes_rows), len(no_rows))
        tiny.extend(rng.sample(yes_rows, n))
        tiny.extend(rng.sample(no_rows, n))

    rng.shuffle(tiny)
    # Re-assign IDs and split name
    for i, row in enumerate(tiny):
        row["id"] = f"gibbon_fewshot_tiny_{i:05d}"
        row["audio_ids"] = [row["id"]]
        row["dataset_name"] = "gibbon-fewshot-tiny"

    return tiny


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write rows to a JSONL file.

    Parameters
    ----------
    rows
        List of dicts to write.
    path
        Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), path)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """Generate all gibbon few-shot detection JSONL splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "beans_pro_gibbon_fewshot",
        help="Directory for output JSONL files.",
    )
    args = parser.parse_args()

    all_rows = load_gibbons()
    rng = random.Random(SEED)

    splits: dict[str, list[dict]] = {}
    for split_name, call_type in CALL_TYPES.items():
        logger.info("Building %s (%s)", split_name, call_type)
        rows = build_split(split_name, call_type, all_rows, rng)
        splits[split_name] = rows
        out_path = args.output_dir / f"{split_name.replace('-', '_')}.jsonl"
        write_jsonl(rows, out_path)

    # Tiny split
    logger.info("Building gibbon-fewshot-tiny")
    tiny = build_tiny_split(splits, rng)
    write_jsonl(tiny, args.output_dir / "gibbon_fewshot_tiny.jsonl")

    logger.info("Done!")


if __name__ == "__main__":
    main()
