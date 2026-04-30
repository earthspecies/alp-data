#!/usr/bin/env python3
"""Build multi-audio 4-way species classification for DRASDIC.

Given 4 species options (1 example clip each) + a target recording,
identify which species the target belongs to. Correct answer always present.
Mix of hard negatives (same family) and random negatives.

Uses XC + iNat train_unseen with long-tail upsampling to boost rare species.

Usage::

    uv run python scripts/build_drasdic_species_mcq.py
    uv run python scripts/build_drasdic_species_mcq.py --n-examples 1000
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
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
N_EXAMPLES = 50_000
MIN_RECORDINGS = 2  # need 1 option + 1 query
HARD_NEG_RATIO = 0.5  # fraction using same-family confusers
SUFFICIENT_THRESHOLD = 10  # long-tail upsample: target count for rare species
MAX_REPEATS = 5  # long-tail upsample: max duplication per example

XC_CSV = "gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/train_unseen_20260203.csv"
INAT_CSV = "gs://esp-ml-datasets/inaturalist/v0.1.0/raw/train_unseen_20260201.csv"

XC_32K_PREFIX = "xeno-canto/v0.1.0/raw/audio_32k/"
INAT_PREFIX = "inaturalist/v0.1.0/raw/"

INSTRUCTION = (
    "Here are four species.\n\n"
    "A: <Audio><AudioHere></Audio>\n"
    "B: <Audio><AudioHere></Audio>\n"
    "C: <Audio><AudioHere></Audio>\n"
    "D: <Audio><AudioHere></Audio>\n\n"
    "Which species best matches the following recording?\n"
    "<Audio><AudioHere></Audio>"
)

GCS_UPLOAD_PATH = (
    "gs://esp-data-ingestion/drasdic/v0.1.0/species_mcq.jsonl"
)


# ── Helpers ──────────────────────────────────────────────────────────────


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV from GCS.

    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    logger.info("Loading %s", path)
    fs = filesystem_from_path(path)
    with fs.open(str(path), "r") as f:
        df = pd.read_csv(f, low_memory=False)
    logger.info("  %d rows", len(df))
    return df


def build_species_index(
    xc: pd.DataFrame, inat: pd.DataFrame
) -> dict[str, list[tuple[str, str]]]:
    """Build species → [(audio_path, family)] from both datasets.

    Returns
    -------
    dict[str, list[tuple[str, str]]]
        Maps canonical_name to list of (audio_path, family) tuples.
    """
    index: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)

    for _, row in xc.iterrows():
        species = row.get("canonical_name", "")
        p32 = row.get("32khz_path", "")
        family = str(row.get("family", ""))
        if not species or not p32 or pd.isna(species) or pd.isna(p32):
            continue
        index[species].append((XC_32K_PREFIX + str(p32), family))

    for _, row in inat.iterrows():
        species = row.get("canonical_name", "")
        p32 = row.get("32khz_path", "")
        family = str(row.get("family", ""))
        if not species or not p32 or pd.isna(species) or pd.isna(p32):
            continue
        index[species].append((INAT_PREFIX + str(p32), family))

    return dict(index)


def apply_long_tail_balance(
    species_index: dict[str, list[tuple[str, str]]],
    sufficient_threshold: int,
    max_repeats: int,
    seed: int,
) -> dict[str, list[tuple[str, str]]]:
    """Apply long-tail upsampling to the species index.

    Species below ``sufficient_threshold`` recordings have their
    entries duplicated up to ``max_repeats`` times, compressing
    the distribution tail so rare species are sampled more often.

    Parameters
    ----------
    species_index
        Original species → recordings mapping.
    sufficient_threshold
        Target count for underrepresented species.
    max_repeats
        Maximum duplication factor per recording.
    seed
        Random seed.

    Returns
    -------
    dict[str, list[tuple[str, str]]]
        Balanced species index.
    """
    rng = random.Random(seed)
    balanced: dict[str, list[tuple[str, str]]] = {}

    for species, recs in species_index.items():
        count = len(recs)
        if count >= sufficient_threshold:
            balanced[species] = list(recs)
        else:
            target = min(sufficient_threshold, count * max_repeats)
            # Repeat recordings to reach target
            extended = list(recs)
            while len(extended) < target:
                extended.append(rng.choice(recs))
            balanced[species] = extended[:target]

    return balanced


def make_row(
    *,
    row_idx: int,
    option_paths: list[str],
    target_path: str,
    correct_label: str,
    option_species: dict[str, str],
    correct_species: str,
    hard_neg: bool,
) -> dict:
    """Build a single JSONL row.

    Returns
    -------
    dict
        JSONL-ready row.
    """
    audio_paths = option_paths + [target_path]
    row_id = f"species_mcq_{row_idx:06d}"

    return {
        "id": row_id,
        "audio_paths": audio_paths,
        "audio_ids": [row_id],
        "template_path": "species_mcq",
        "skills": ["multiple_choice", "species_classification"],
        "messages": [
            {"role": "user", "content": INSTRUCTION},
            {"role": "assistant", "content": correct_label},
        ],
        "task": "species_mcq",
        "source_dataset": "xeno-canto+inaturalist",
        "dataset_name": "species-mcq",
        "license": "mixed",
        "metadata": json.dumps({
            "option_species": option_species,
            "correct": correct_label,
            "correct_species": correct_species,
            "hard_negatives": hard_neg,
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


# ── Generation ───────────────────────────────────────────────────────────


def generate_examples(
    species_index: dict[str, list[tuple[str, str]]],
    n_examples: int,
    hard_neg_ratio: float,
    seed: int,
) -> list[dict]:
    """Generate 4-way species MCQ examples.

    Parameters
    ----------
    species_index
        Maps species → list of (audio_path, family).
    n_examples
        Number of examples.
    hard_neg_ratio
        Fraction using same-family confusers.
    seed
        Random seed.

    Returns
    -------
    list[dict]
        JSONL rows.
    """
    rng = random.Random(seed)
    labels = ["A", "B", "C", "D"]

    eligible = {
        sp: recs for sp, recs in species_index.items() if len(recs) >= MIN_RECORDINGS
    }
    species_list = list(eligible.keys())
    weights = [1.0 / math.sqrt(len(eligible[sp])) for sp in species_list]

    # Build family → species index for hard negatives
    family_to_species: dict[str, list[str]] = collections.defaultdict(list)
    for sp, recs in eligible.items():
        families = {r[1] for r in recs if r[1] and r[1] != "nan"}
        for fam in families:
            family_to_species[fam].append(sp)

    logger.info("Eligible species: %d, families: %d", len(eligible), len(family_to_species))

    rows = []
    hard_count = 0
    for i in range(n_examples):
        focal = rng.choices(species_list, weights=weights, k=1)[0]
        focal_recs = eligible[focal]

        # Sample option + query from focal (different recordings)
        sampled = rng.sample(focal_recs, min(2, len(focal_recs)))
        focal_option_path = sampled[0][0]
        focal_query_path = sampled[1][0] if len(sampled) > 1 else sampled[0][0]

        # Decide hard vs random negatives
        use_hard = rng.random() < hard_neg_ratio
        focal_families = {r[1] for r in focal_recs if r[1] and r[1] != "nan"}

        confusers: list[str] = []
        if use_hard and focal_families:
            # Gather same-family species
            candidates = set()
            for fam in focal_families:
                candidates.update(family_to_species.get(fam, []))
            candidates.discard(focal)
            if len(candidates) >= 3:
                confusers = rng.sample(list(candidates), 3)
                hard_count += 1

        if len(confusers) < 3:
            # Fall back to random
            confusers = rng.sample(
                [s for s in species_list if s != focal], 3
            )

        # Pick one clip per confuser
        confuser_paths = [rng.choice(eligible[c])[0] for c in confusers]

        # Assign correct position (balanced)
        correct_idx = i % 4
        option_species_list = list(confusers)
        option_species_list.insert(correct_idx, focal)
        option_paths = list(confuser_paths)
        option_paths.insert(correct_idx, focal_option_path)

        rows.append(make_row(
            row_idx=i,
            option_paths=option_paths,
            target_path=focal_query_path,
            correct_label=labels[correct_idx],
            option_species=dict(
                zip(labels, option_species_list, strict=True)
            ),
            correct_species=focal,
            hard_neg=(len(confusers) == 3 and use_hard),
        ))

        if (i + 1) % 10000 == 0:
            logger.info("  Generated %d / %d", i + 1, n_examples)

    logger.info("Hard negatives: %d / %d (%.1f%%)", hard_count, n_examples,
                hard_count / n_examples * 100)
    return rows


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """Generate species MCQ JSONL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "data" / "drasdic_species_mcq",
    )
    parser.add_argument("--n-examples", type=int, default=N_EXAMPLES)
    args = parser.parse_args()

    xc = load_csv(XC_CSV)
    inat = load_csv(INAT_CSV)

    species_index = build_species_index(xc, inat)
    logger.info("Species in index: %d", len(species_index))

    # Apply long-tail upsampling to boost rare species
    balanced = apply_long_tail_balance(
        species_index,
        sufficient_threshold=SUFFICIENT_THRESHOLD,
        max_repeats=MAX_REPEATS,
        seed=SEED,
    )
    before_counts = {sp: len(recs) for sp, recs in species_index.items()}
    after_counts = {sp: len(recs) for sp, recs in balanced.items()}
    boosted = sum(1 for sp in balanced if after_counts[sp] > before_counts.get(sp, 0))
    logger.info(
        "Long-tail upsample: %d species boosted (threshold=%d, max_repeats=%d)",
        boosted, SUFFICIENT_THRESHOLD, MAX_REPEATS,
    )

    rows = generate_examples(balanced, args.n_examples, HARD_NEG_RATIO, SEED)

    # Stats
    answer_dist = collections.Counter(r["messages"][1]["content"] for r in rows)
    species_used = collections.Counter(
        json.loads(r["metadata"])["correct_species"] for r in rows
    )
    logger.info("Answer balance: %s", dict(sorted(answer_dist.items())))
    logger.info("Unique focal species: %d", len(species_used))
    logger.info("Top 10: %s", species_used.most_common(10))
    logger.info("Bottom 10: %s", species_used.most_common()[-10:])

    write_jsonl(rows, args.output_dir / "species_mcq.jsonl")
    logger.info("Done!")


if __name__ == "__main__":
    main()
