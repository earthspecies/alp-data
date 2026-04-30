#!/usr/bin/env python3
"""Build giant otter call-type few-shot eval splits for BEANS-Pro multi-audio.

Creates two multi-audio evaluation tasks from the giant otter vocal repertoire:
- ``giant-otter-same-different``: ~1000 same/different call-type pairs
- ``giant-otter-4way``: ~500 4-way multiple-choice call-type matching

Usage::

    uv run python scripts/build_beans_pro_giant_otter_calltype.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import pandas as pd

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
MIN_EXAMPLES = 5

CSV_PATH = "gs://esp-ml-datasets/giant_otters/v0.1.0/raw/giant_otters_annotations_test.csv"
# Audio will be copied to beans-pro; paths in JSONL are relative to beans-pro root
AUDIO_PREFIX = "audio/giant_otters/"

SAME_DIFF_INSTRUCTION = (
    "Are these two sounds the same call type?\n"
    "<Audio><AudioHere></Audio>\n"
    "<Audio><AudioHere></Audio>"
)

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


def load_data() -> pd.DataFrame:
    """Load the giant otters CSV from GCS.

    Returns
    -------
    pd.DataFrame
        The loaded annotations.
    """
    logger.info("Loading %s", CSV_PATH)
    fs = filesystem_from_path(CSV_PATH)
    with fs.open(str(CSV_PATH), "r") as f:
        df = pd.read_csv(f, keep_default_na=False, na_values=[""])
    logger.info("  %d rows, %d call types", len(df), df["vocalization"].nunique())
    return df


def build_type_index(
    df: pd.DataFrame,
    existing_audio: set[str] | None = None,
) -> dict[str, list[str]]:
    """Build call_type → list of audio paths.

    Parameters
    ----------
    df
        The giant otters annotations dataframe.
    existing_audio
        Optional set of known-existing audio filenames. Rows whose audio
        is not in this set are skipped.

    Returns
    -------
    dict[str, list[str]]
        Maps vocalization label to list of audio paths (relative to
        beans-pro root).
    """
    index: dict[str, list[str]] = {}
    skipped = 0
    for _, row in df.iterrows():
        label = row["vocalization"]
        path = row["path"]
        if not label or not path or pd.isna(label) or pd.isna(path):
            continue
        filename = Path(str(path)).name
        if existing_audio is not None and filename not in existing_audio:
            skipped += 1
            continue
        beans_path = AUDIO_PREFIX + filename
        index.setdefault(label, []).append(beans_path)
    if skipped:
        logger.warning("Skipped %d rows with missing audio files", skipped)
    return {k: v for k, v in index.items() if len(v) >= MIN_EXAMPLES}


def make_row(
    *,
    split_name: str,
    row_idx: int,
    audio_paths: list[str],
    instruction: str,
    output: str,
    task: str,
    template_path: str,
    metadata: dict,
) -> dict:
    """Build a single JSONL row.

    Returns
    -------
    dict
        A JSONL-ready row.
    """
    row_id = f"{split_name.replace('-', '_')}_{row_idx:05d}"
    return {
        "id": row_id,
        "audio_paths": audio_paths,
        "audio_ids": [row_id],
        "template_path": template_path,
        "skills": [task],
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ],
        "task": task,
        "source_dataset": "Giant Otters (Mumm & Knörnschild 2014)",
        "dataset_name": split_name,
        "license": "CC-BY-4.0",
        "metadata": json.dumps(metadata),
        "audio_path_original_sample_rate": audio_paths[-1],
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


# ── Same/Different ───────────────────────────────────────────────────────


def build_same_different(
    type_index: dict[str, list[str]],
    n_examples: int,
    rng: random.Random,
) -> list[dict]:
    """Build same/different call-type pairs.

    Parameters
    ----------
    type_index
        Call type → audio paths mapping.
    n_examples
        Total examples (half Same, half Different).
    rng
        Seeded RNG.

    Returns
    -------
    list[dict]
        JSONL rows.
    """
    types = list(type_index.keys())
    n_same = n_examples // 2
    n_diff = n_examples - n_same
    rows = []

    # Same pairs
    for i in range(n_same):
        ct = rng.choice(types)
        clips = rng.sample(type_index[ct], 2)
        rows.append(make_row(
            split_name="giant-otter-same-different",
            row_idx=i,
            audio_paths=clips,
            instruction=SAME_DIFF_INSTRUCTION,
            output="Same",
            task="call_type_same_different",
            template_path="audio_synth/same_different",
            metadata={"call_type_1": ct, "call_type_2": ct},
        ))

    # Different pairs
    for i in range(n_diff):
        ct1, ct2 = rng.sample(types, 2)
        clip1 = rng.choice(type_index[ct1])
        clip2 = rng.choice(type_index[ct2])
        rows.append(make_row(
            split_name="giant-otter-same-different",
            row_idx=n_same + i,
            audio_paths=[clip1, clip2],
            instruction=SAME_DIFF_INSTRUCTION,
            output="Different",
            task="call_type_same_different",
            template_path="audio_synth/same_different",
            metadata={"call_type_1": ct1, "call_type_2": ct2},
        ))

    rng.shuffle(rows)
    return rows


# ── 4-Way Multiple Choice ────────────────────────────────────────────────


def build_4way(
    type_index: dict[str, list[str]],
    n_examples: int,
    rng: random.Random,
) -> list[dict]:
    """Build 4-way multiple-choice call-type matching.

    Parameters
    ----------
    type_index
        Call type → audio paths mapping.
    n_examples
        Total examples.
    rng
        Seeded RNG.

    Returns
    -------
    list[dict]
        JSONL rows.
    """
    types = list(type_index.keys())
    labels = ["A", "B", "C", "D"]
    rows = []

    for i in range(n_examples):
        # Pick 4 distinct call types
        chosen_types = rng.sample(types, 4)

        # For each option, pick one example clip
        option_clips = [rng.choice(type_index[ct]) for ct in chosen_types]

        # Pick which option is correct (balanced across A/B/C/D)
        correct_idx = i % 4
        correct_type = chosen_types[correct_idx]

        # Pick a different clip from the correct type as the target
        available = [c for c in type_index[correct_type] if c != option_clips[correct_idx]]
        if not available:
            # Fallback: reuse the option clip (rare, only if type has exactly MIN_EXAMPLES)
            target_clip = option_clips[correct_idx]
        else:
            target_clip = rng.choice(available)

        audio_paths = option_clips + [target_clip]

        rows.append(make_row(
            split_name="giant-otter-4way",
            row_idx=i,
            audio_paths=audio_paths,
            instruction=FOUR_WAY_INSTRUCTION,
            output=labels[correct_idx],
            task="call_type_multiple_choice",
            template_path="audio_synth/multiple_choice",
            metadata={
                "option_types": dict(zip(labels, chosen_types, strict=True)),
                "correct": labels[correct_idx],
                "correct_type": correct_type,
            },
        ))

    rng.shuffle(rows)
    return rows


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """Generate giant otter call-type eval JSONL splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "beans_pro_giant_otter",
    )
    parser.add_argument("--n-same-diff", type=int, default=1000)
    parser.add_argument("--n-4way", type=int, default=500)
    args = parser.parse_args()

    df = load_data()

    # Check which audio files actually exist in the source
    logger.info("Checking existing audio files in source...")
    fs = filesystem_from_path(CSV_PATH)
    source_dir = "esp-ml-datasets/giant_otters/v0.1.0/raw/audio/Audio_S1/"
    existing_audio = {obj.split("/")[-1] for obj in fs.ls(source_dir)}
    logger.info("  %d audio files exist in source", len(existing_audio))

    type_index = build_type_index(df, existing_audio=existing_audio)
    logger.info("Eligible call types (>=%d examples): %d", MIN_EXAMPLES, len(type_index))
    for ct, clips in sorted(type_index.items(), key=lambda x: -len(x[1])):
        logger.info("  %s: %d", ct, len(clips))

    rng = random.Random(SEED)

    # Same/Different
    sd_rows = build_same_different(type_index, args.n_same_diff, rng)
    import collections
    sd_balance = collections.Counter(r["messages"][1]["content"] for r in sd_rows)
    logger.info("same-different: %d rows, balance: %s", len(sd_rows), dict(sd_balance))
    write_jsonl(sd_rows, args.output_dir / "giant_otter_same_different.jsonl")

    # 4-way
    fw_rows = build_4way(type_index, args.n_4way, rng)
    fw_balance = collections.Counter(r["messages"][1]["content"] for r in fw_rows)
    logger.info("4way: %d rows, balance: %s", len(fw_rows), dict(fw_balance))
    write_jsonl(fw_rows, args.output_dir / "giant_otter_4way.jsonl")

    # Print audio files that need copying
    all_paths = set()
    for row in sd_rows + fw_rows:
        all_paths.update(row["audio_paths"])
    logger.info("Unique audio files referenced: %d", len(all_paths))

    logger.info("Done!")


if __name__ == "__main__":
    main()
