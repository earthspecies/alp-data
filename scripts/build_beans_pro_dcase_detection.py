#!/usr/bin/env python3
"""Build DCASE few-shot multi-label detection eval for BEANS-Pro.

Uses ALL beans-zero DCASE clips (13,688) as targets, including None and
multi-label examples. Prompt matches the DRASDIC v2 fewshot detection
format with 4 sound-type options and optional background environment clip.

Produces two splits:
- ``dcase-fewshot-detection``: full 13,688 examples
- ``dcase-fewshot-detection-balanced``: ~3,200 examples (50/50 None vs present)

Usage::

    uv run python scripts/build_beans_pro_dcase_detection.py
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import sys
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

# ── Constants ────────────────────────────────────────────────────────────

SEED = 42
LABELS = ["A", "B", "C", "D"]
N_OPTIONS = 4
N_SUPPORT = 2  # clips per type held out as support pool
MIN_CLIPS_FOR_SUPPORT = 2  # fixed support pool size
BG_PROB = 0.5  # probability of including a background environment clip

DCASE_JSONL = "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/dcase_test.jsonl"


# ── Prompt builders ─────────────────────────────────────────────────────


def build_instruction(
    n_options: int,
    has_background: bool,
) -> str:
    """Build the fewshot detection instruction string.

    Parameters
    ----------
    n_options : int
        Number of sound-type options (1-4).
    has_background : bool
        Whether a background environment clip is included.

    Returns
    -------
    str
        Prompt with ``<AudioHere>`` placeholders.
    """
    if n_options == 1:
        header = "Here is an example of a sound."
    else:
        header = f"Here are examples of {n_options} sounds."

    lines = [header, ""]
    for i in range(n_options):
        lines.append(f"{LABELS[i]}: <Audio><AudioHere></Audio>")

    if has_background:
        lines.append("")
        lines.append("Here is the background environment: <Audio><AudioHere></Audio>")

    lines.append("")
    lines.append(
        "Which of the above sounds are present in this recording, if any?"
    )
    lines.append("<Audio><AudioHere></Audio>")

    return "\n".join(lines)


# ── Helpers ──────────────────────────────────────────────────────────────


def load_dcase() -> list[dict[str, Any]]:
    """Load beans-zero DCASE JSONL from GCS.

    Returns
    -------
    list[dict[str, Any]]
        Parsed records.
    """
    logger.info("Loading %s", DCASE_JSONL)
    fs = filesystem_from_path(DCASE_JSONL)
    rows = []
    with fs.open(str(DCASE_JSONL), "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    logger.info("  %d rows", len(rows))
    return rows


def parse_labels(output: str) -> list[str]:
    """Parse the output label string into a list of sound types.

    Parameters
    ----------
    output : str
        Ground-truth output (e.g., ``"Ovenbird"``, ``"Ovenbird, Swainson's Thrush"``,
        ``"None"``).

    Returns
    -------
    list[str]
        List of present sound types, empty for ``"None"``.
    """
    if output == "None":
        return []
    return [s.strip() for s in output.split(", ")]


def build_type_index(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Build sound_type → audio paths for all clips containing that type.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        DCASE JSONL records.

    Returns
    -------
    dict[str, list[str]]
        Maps sound type to list of 32KHz audio paths.
    """
    index: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        for label in parse_labels(row["output"]):
            index[label].append(row["audio_path_32KHz"])
    return dict(index)


def choose_support_path(
    sound_type: str,
    target_path: str,
    support_pool: dict[str, list[str]],
    rng: random.Random,
) -> str:
    """Pick a support clip that is not the target clip itself.

    Parameters
    ----------
    sound_type : str
        Sound type whose support clip should be selected.
    target_path : str
        Query clip path for the current example.
    support_pool : dict[str, list[str]]
        Fixed support pool for each sound type.
    rng : random.Random
        Seeded RNG.

    Returns
    -------
    str
        Chosen support clip path.

    Raises
    ------
    ValueError
        If no non-target support clip is available for ``sound_type``.
    """
    candidates = [path for path in support_pool[sound_type] if path != target_path]
    if not candidates:
        raise ValueError(
            f"No non-target support clip available for {sound_type!r}: {target_path!r}"
        )
    return rng.choice(candidates)


def make_row(
    *,
    split_name: str,
    row_idx: int,
    audio_paths: list[str],
    instruction: str,
    answer: str,
    option_types: dict[str, str],
    present_types: list[str],
    has_background: bool,
    original_bz_id: str,
) -> dict[str, Any]:
    """Build a single JSONL row.

    Returns
    -------
    dict[str, Any]
        JSONL-ready row.
    """
    row_id = f"dcase_fsdet_{row_idx:06d}"
    return {
        "id": row_id,
        "audio_paths": audio_paths,
        "audio_ids": [row_id],
        "template_path": "audio_synth/fewshot_detection",
        "skills": ["multilabel_detection", "few_shot_detection"],
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": answer},
        ],
        "task": "fewshot_detection",
        "source_dataset": "DCASE-2021-Task-5",
        "dataset_name": split_name,
        "license": "CC-BY",
        "metadata": json.dumps({
            "option_types": option_types,
            "present_types": present_types,
            "has_background": has_background,
        }),
        "audio_path_original_sample_rate": audio_paths[-1],
        "original_beans_zero_id": original_bz_id,
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows to a JSONL file.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Records to write.
    path : Path
        Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("  Wrote %d rows to %s", len(rows), path)


# ── Generation ──────────────────────────────────────────────────────────


def generate_detection(
    dcase_rows: list[dict[str, Any]],
    support_pool: dict[str, list[str]],
    bg_pool: list[str],
    all_types: list[str],
    split_name: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Generate fewshot detection examples from all DCASE clips.

    Parameters
    ----------
    dcase_rows : list[dict[str, Any]]
        All beans-zero DCASE records.
    support_pool : dict[str, list[str]]
        Sound type → held-out support audio paths.
    bg_pool : list[str]
        Audio paths for "None" clips (background environment candidates).
    all_types : list[str]
        All eligible sound types.
    split_name : str
        Output split name.
    rng : random.Random
        Seeded RNG.

    Returns
    -------
    list[dict[str, Any]]
        JSONL rows.

    Raises
    ------
    ValueError
        If a target has more labels than available options or lacks a
        valid non-target support clip.
    """
    rows: list[dict[str, Any]] = []

    for src in dcase_rows:
        target_path = src["audio_path_32KHz"]
        bz_id = src["id"]
        present = parse_labels(src["output"])
        if len(present) > N_OPTIONS:
            raise ValueError(
                f"Target has {len(present)} labels but only {N_OPTIONS} options: {src['id']!r}"
            )
        missing_support = [sound_type for sound_type in present if sound_type not in support_pool]
        if missing_support:
            raise ValueError(
                f"Missing support pool for labels {missing_support!r} in row {src['id']!r}"
            )

        # Select option types: include present ones, fill rest with absent types
        n_present = len(present)
        selected_present = list(present)

        absent_types = [t for t in all_types if t not in present]
        n_absent = N_OPTIONS - n_present
        if len(absent_types) < n_absent:
            n_absent = len(absent_types)
        selected_absent = rng.sample(absent_types, n_absent)

        option_types_list = selected_present + selected_absent
        # Shuffle option positions to avoid positional bias
        rng.shuffle(option_types_list)

        # Map labels and build answer
        option_map = {LABELS[i]: t for i, t in enumerate(option_types_list)}
        answer_labels = sorted(
            [lab for lab, t in option_map.items() if t in present],
            key=lambda x: LABELS.index(x),
        )
        answer = ", ".join(answer_labels) if answer_labels else "None"

        # Pick 1 support clip per option
        option_paths = [
            choose_support_path(t, target_path, support_pool, rng)
            for t in option_types_list
        ]

        # Background environment clip (50% chance)
        has_bg = rng.random() < BG_PROB
        bg_path: str | None = None
        if has_bg:
            # Pick a random "None" clip that isn't the target
            bg_candidates = [p for p in bg_pool if p != target_path]
            if bg_candidates:
                bg_path = rng.choice(bg_candidates)
            else:
                has_bg = False

        # Build audio_paths list: options [+ background] + target
        audio_paths = list(option_paths)
        if has_bg and bg_path:
            audio_paths.append(bg_path)
        audio_paths.append(target_path)

        n_opts = len(option_types_list)
        instruction = build_instruction(n_opts, has_bg)

        rows.append(make_row(
            split_name=split_name,
            row_idx=len(rows),
            audio_paths=audio_paths,
            instruction=instruction,
            answer=answer,
            option_types=option_map,
            present_types=present,
            has_background=has_bg,
            original_bz_id=bz_id,
        ))

    rng.shuffle(rows)
    # Re-index after shuffle
    for i, row in enumerate(rows):
        row["id"] = f"dcase_fsdet_{i:06d}"
        row["audio_ids"] = [row["id"]]

    return rows


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """Generate DCASE fewshot detection eval splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "beans_pro_dcase_detection",
    )
    args = parser.parse_args()

    dcase_rows = load_dcase()

    # Build type index from all clips containing each sound type
    type_index = build_type_index(dcase_rows)
    logger.info("Sound types with support candidates: %d", len(type_index))
    for t, paths in sorted(type_index.items(), key=lambda x: -len(x[1])):
        logger.info("  %s: %d clips", t, len(paths))

    # Filter to types with enough clips for support
    eligible_types = {
        t: paths for t, paths in type_index.items()
        if len(paths) >= MIN_CLIPS_FOR_SUPPORT
    }
    logger.info(
        "Eligible types (>=%d clips): %d",
        MIN_CLIPS_FOR_SUPPORT, len(eligible_types),
    )

    rng = random.Random(SEED)

    # Hold out a fixed support pool for each sound type.
    support_pool: dict[str, list[str]] = {}
    for t, paths in eligible_types.items():
        shuffled = list(paths)
        rng.shuffle(shuffled)
        support_pool[t] = shuffled[:N_SUPPORT]

    all_types = list(eligible_types.keys())

    # Build background environment pool (all "None" clips)
    bg_pool = [
        row["audio_path_32KHz"]
        for row in dcase_rows
        if row["output"] == "None"
    ]
    logger.info("Background environment pool: %d clips", len(bg_pool))

    target_rows = dcase_rows
    logger.info("Target clips: %d", len(target_rows))

    # Generate full split
    rng_full = random.Random(SEED)
    full_rows = generate_detection(
        target_rows, support_pool, bg_pool, all_types,
        split_name="dcase-fewshot-detection",
        rng=rng_full,
    )

    # Stats
    answer_types = collections.Counter()
    bg_count = 0
    for row in full_rows:
        ans = row["messages"][1]["content"]
        if ans == "None":
            answer_types["none"] += 1
        else:
            n_labels = len(ans.split(", "))
            answer_types[f"{n_labels}_label"] += 1
        meta = json.loads(row["metadata"])
        if meta["has_background"]:
            bg_count += 1

    logger.info("=== dcase-fewshot-detection ===")
    logger.info("  Total: %d", len(full_rows))
    logger.info("  Answers: %s", dict(sorted(answer_types.items())))
    logger.info("  With background: %d (%.1f%%)", bg_count, bg_count / len(full_rows) * 100)

    write_jsonl(full_rows, args.output_dir / "dcase_fewshot_detection.jsonl")

    # Generate balanced split: ~50/50 None vs present
    rng_bal = random.Random(SEED)
    present_rows = [r for r in full_rows if r["messages"][1]["content"] != "None"]
    none_rows = [r for r in full_rows if r["messages"][1]["content"] == "None"]
    n_present = len(present_rows)
    sampled_none = rng_bal.sample(none_rows, min(len(none_rows), n_present))

    balanced_rows = present_rows + sampled_none
    rng_bal.shuffle(balanced_rows)
    # Re-index
    for i, row in enumerate(balanced_rows):
        row = dict(row)  # don't mutate the full-split rows
        row["id"] = f"dcase_fsdet_bal_{i:06d}"
        row["audio_ids"] = [row["id"]]
        row["dataset_name"] = "dcase-fewshot-detection-balanced"
        balanced_rows[i] = row

    bal_answers = collections.Counter()
    for row in balanced_rows:
        ans = row["messages"][1]["content"]
        if ans == "None":
            bal_answers["none"] += 1
        else:
            bal_answers["present"] += 1

    logger.info("=== dcase-fewshot-detection-balanced ===")
    logger.info("  Total: %d", len(balanced_rows))
    logger.info("  Balance: %s", dict(sorted(bal_answers.items())))

    write_jsonl(balanced_rows, args.output_dir / "dcase_fewshot_detection_balanced.jsonl")

    logger.info("Done!")


if __name__ == "__main__":
    main()
