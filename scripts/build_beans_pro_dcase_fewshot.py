#!/usr/bin/env python3
"""Build DCASE few-shot 4-way multiple-choice eval for BEANS-Pro multi-audio.

Given 4 species/sound options (1 example clip each) + a target clip,
identify which option is present in the target recording.

Uses single-species clips from the BEANS-Zero DCASE split. 2 clips per
species are held out as a fixed support pool; the rest become queries.

Usage::

    uv run python scripts/build_beans_pro_dcase_fewshot.py
"""

from __future__ import annotations

import argparse
import collections
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
DCASE_JSONL = "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/dcase_test.jsonl"
MIN_CLIPS = 4  # need 2 support + at least 2 queries
N_SUPPORT = 2
AUDIO_PREFIX = "audio/dcase/32KHz/"

FOUR_WAY_INSTRUCTION = (
    "Here are four call types.\n\n"
    "A: <Audio><AudioHere></Audio>\n"
    "B: <Audio><AudioHere></Audio>\n"
    "C: <Audio><AudioHere></Audio>\n"
    "D: <Audio><AudioHere></Audio>\n\n"
    "Which call type best matches the following recording?\n"
    "<Audio><AudioHere></Audio>"
)


# ── Helpers ──────────────────────────────────────────────────────────────


def load_dcase() -> list[dict]:
    """Load all rows from the BEANS-Zero DCASE JSONL.

    Returns
    -------
    list[dict]
        Parsed JSONL rows.
    """
    logger.info("Loading %s", DCASE_JSONL)
    fs = filesystem_from_path(DCASE_JSONL)
    rows = []
    with fs.open(str(DCASE_JSONL), "r") as f:
        for line in f:
            rows.append(json.loads(line))
    logger.info("  %d rows", len(rows))
    return rows


def build_species_index(rows: list[dict]) -> dict[str, list[str]]:
    """Build species → list of audio paths for single-species clips.

    Returns
    -------
    dict[str, list[str]]
        Maps species label to list of audio paths relative to beans-pro root.
    """
    index: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        output = row["output"]
        if output == "None" or ", " in output:
            continue
        filename = row.get("file_name", "")
        if not filename:
            continue
        index[output].append(AUDIO_PREFIX + filename)
    return dict(index)


def make_row(
    *,
    row_idx: int,
    option_paths: list[str],
    target_path: str,
    correct_label: str,
    option_types: dict[str, str],
    correct_type: str,
) -> dict:
    """Build a single JSONL row.

    Returns
    -------
    dict
        A JSONL-ready row.
    """
    audio_paths = option_paths + [target_path]
    row_id = f"dcase_4way_{row_idx:05d}"

    return {
        "id": row_id,
        "audio_paths": audio_paths,
        "audio_ids": [row_id],
        "template_path": "audio_synth/multiple_choice",
        "skills": ["multiple_choice", "few_shot_detection"],
        "messages": [
            {"role": "user", "content": FOUR_WAY_INSTRUCTION},
            {"role": "assistant", "content": correct_label},
        ],
        "task": "call_type_multiple_choice",
        "source_dataset": "DCASE-2021-Task-5",
        "dataset_name": "dcase-4way",
        "license": "CC-BY",
        "metadata": json.dumps({
            "option_types": option_types,
            "correct": correct_label,
            "correct_type": correct_type,
        }),
        "audio_path_original_sample_rate": target_path,
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write rows to a JSONL file.

    Parameters
    ----------
    rows
        Records to write.
    path
        Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), path)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """Generate DCASE 4-way multiple-choice JSONL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "beans_pro_dcase",
    )
    args = parser.parse_args()

    all_rows = load_dcase()
    species_index = build_species_index(all_rows)

    # Filter to types with enough clips
    eligible = {
        sp: paths
        for sp, paths in species_index.items()
        if len(paths) >= MIN_CLIPS
    }
    logger.info(
        "Eligible types: %d (of %d, min %d clips)",
        len(eligible), len(species_index), MIN_CLIPS,
    )
    for sp, paths in sorted(eligible.items(), key=lambda x: -len(x[1])):
        logger.info("  %s: %d", sp, len(paths))

    rng = random.Random(SEED)
    types = list(eligible.keys())
    labels = ["A", "B", "C", "D"]

    # Hold out N_SUPPORT clips per type as fixed support
    support_pool: dict[str, list[str]] = {}
    query_pool: dict[str, list[str]] = {}
    for sp, paths in eligible.items():
        shuffled = list(paths)
        rng.shuffle(shuffled)
        support_pool[sp] = shuffled[:N_SUPPORT]
        query_pool[sp] = shuffled[N_SUPPORT:]

    # Generate examples: iterate over all query clips
    rows = []
    idx = 0
    for correct_type in types:
        for query_path in query_pool[correct_type]:
            # Pick correct answer position (cycle for balance)
            correct_idx = idx % 4

            # Pick 3 confuser types
            confusers = rng.sample(
                [t for t in types if t != correct_type], 3
            )

            # Build option list: correct type at correct_idx, confusers fill rest
            option_types_list = list(confusers)
            option_types_list.insert(correct_idx, correct_type)

            # Pick one support clip per option
            option_paths = [
                rng.choice(support_pool[t]) for t in option_types_list
            ]

            rows.append(make_row(
                row_idx=idx,
                option_paths=option_paths,
                target_path=query_path,
                correct_label=labels[correct_idx],
                option_types=dict(
                    zip(labels, option_types_list, strict=True)
                ),
                correct_type=correct_type,
            ))
            idx += 1

    rng.shuffle(rows)

    balance = collections.Counter(r["messages"][1]["content"] for r in rows)
    logger.info("Total: %d, balance: %s", len(rows), dict(balance))

    write_jsonl(rows, args.output_dir / "dcase_4way.jsonl")
    logger.info("Done!")


if __name__ == "__main__":
    main()
