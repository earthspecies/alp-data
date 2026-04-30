#!/usr/bin/env python3
"""Build same-species few-shot identification split for DRASDIC.

Given 2-5 reference clips of a species, determine if a query clip is the
same species. Uses XC + iNat train_unseen data, biased toward rarer species.

Produces ~200k examples (50/50 Yes/No) as a JSONL matching DRASDIC multi-audio
format.

Usage::

    uv run python scripts/build_drasdic_same_species.py
    uv run python scripts/build_drasdic_same_species.py --n-examples 1000  # small test
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
N_EXAMPLES = 200_000
MIN_RECORDINGS_PER_SPECIES = 6  # need support + pos query + margin

XC_CSV = "gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/train_unseen_20260203.csv"
INAT_CSV = "gs://esp-ml-datasets/inaturalist/v0.1.0/raw/train_unseen_20260201.csv"

# Audio path prefixes relative to gs://esp-ml-datasets/
XC_32K_PREFIX = "xeno-canto/v0.1.0/raw/audio_32k/"
INAT_PREFIX = "inaturalist/v0.1.0/raw/"

INSTRUCTION_TEMPLATE = (
    "Here are example(s) of a target species.\n"
    "{support_block}\n"
    "Is the following recording the same species?\n"
    "<Audio><AudioHere></Audio>"
)

GCS_UPLOAD_PATH = "gs://esp-data-ingestion/beans-pro-multi-audio/v0.1.0/raw/same_species/test.jsonl"


# ── Helpers ──────────────────────────────────────────────────────────────


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV from GCS.

    Returns
    -------
    pd.DataFrame
        The loaded data.
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
    """Build species → [(audio_path, order)] index from both datasets.

    Returns
    -------
    dict[str, list[tuple[str, str]]]
        Maps canonical_name to list of (relative_audio_path, taxonomic_order).
    """
    index: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)

    for _, row in xc.iterrows():
        species = row.get("canonical_name", "")
        p32 = row.get("32khz_path", "")
        order = str(row.get("order", ""))
        if not species or not p32 or pd.isna(species) or pd.isna(p32):
            continue
        index[species].append((XC_32K_PREFIX + str(p32), order))

    for _, row in inat.iterrows():
        species = row.get("canonical_name", "")
        p32 = row.get("32khz_path", "")
        order = str(row.get("order", ""))
        if not species or not p32 or pd.isna(species) or pd.isna(p32):
            continue
        index[species].append((INAT_PREFIX + str(p32), order))

    return dict(index)


def build_instruction(n_support: int) -> str:
    """Build instruction with the right number of support placeholders.

    Returns
    -------
    str
        Formatted instruction string.
    """
    support_lines = ["<Audio><AudioHere></Audio>" for _ in range(n_support)]
    return INSTRUCTION_TEMPLATE.format(support_block="\n".join(support_lines))


def make_row(
    *,
    row_idx: int,
    support_paths: list[str],
    query_path: str,
    label: str,
    focal_species: str,
    query_species: str,
) -> dict:
    """Build a single JSONL row.

    Returns
    -------
    dict
        DRASDIC-compatible multi-audio row.
    """
    audio_paths = [*support_paths, query_path]
    instruction = build_instruction(len(support_paths))
    row_id = f"same_species_{row_idx:06d}"

    metadata = {
        "focal_species": focal_species,
        "query_species": query_species,
        "label": label,
        "n_support": len(support_paths),
    }

    return {
        "id": row_id,
        "audio_paths": audio_paths,
        "audio_ids": [row_id],
        "template_path": "same_species",
        "skills": ["same_species", "species_identification"],
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": label},
        ],
        "task": "same_species",
        "source_dataset": "xeno-canto+inaturalist",
        "dataset_name": "same-species",
        "license": "mixed",
        "metadata": json.dumps(metadata),
        "audio_path_original_sample_rate": query_path,
    }


# ── Main generation ──────────────────────────────────────────────────────


def generate_examples(
    species_index: dict[str, list[tuple[str, str]]],
    n_examples: int,
    seed: int,
) -> list[dict]:
    """Generate same-species few-shot examples.

    Parameters
    ----------
    species_index
        Maps species name to list of (audio_path, order) tuples.
    n_examples
        Total examples to generate.
    seed
        Random seed.

    Returns
    -------
    list[dict]
        List of JSONL rows.
    """
    rng = random.Random(seed)

    # Filter to species with enough recordings
    eligible = {
        sp: recs
        for sp, recs in species_index.items()
        if len(recs) >= MIN_RECORDINGS_PER_SPECIES
    }
    logger.info(
        "Eligible species: %d (of %d total, min %d recordings)",
        len(eligible), len(species_index), MIN_RECORDINGS_PER_SPECIES,
    )

    # Compute sampling weights: 1/sqrt(count) — upweights rare species
    species_list = list(eligible.keys())
    weights = [1.0 / math.sqrt(len(eligible[sp])) for sp in species_list]

    # Build order → species index for hard negatives
    order_to_species: dict[str, list[str]] = collections.defaultdict(list)
    for sp, recs in eligible.items():
        orders = {r[1] for r in recs if r[1] and r[1] != "nan"}
        for o in orders:
            order_to_species[o].append(sp)

    rows = []
    for i in range(n_examples):
        # Pick focal species (weighted toward rare)
        focal = rng.choices(species_list, weights=weights, k=1)[0]
        focal_recs = eligible[focal]

        # Pick support count (2-5)
        n_support = rng.randint(2, min(5, len(focal_recs) - 1))

        # Sample support + one positive query
        sampled = rng.sample(focal_recs, n_support + 1)
        support_paths = [r[0] for r in sampled[:n_support]]
        pos_query_path = sampled[n_support][0]

        if rng.random() < 0.5:
            # Positive example
            rows.append(make_row(
                row_idx=i,
                support_paths=support_paths,
                query_path=pos_query_path,
                label="Yes",
                focal_species=focal,
                query_species=focal,
            ))
        else:
            # Negative example — prefer same-order confuser
            focal_orders = {r[1] for r in focal_recs if r[1] and r[1] != "nan"}
            confuser_candidates = set()
            for o in focal_orders:
                confuser_candidates.update(order_to_species.get(o, []))
            confuser_candidates.discard(focal)

            if confuser_candidates:
                confuser = rng.choice(list(confuser_candidates))
            else:
                confuser = rng.choice([s for s in species_list if s != focal])

            neg_rec = rng.choice(eligible[confuser])
            rows.append(make_row(
                row_idx=i,
                support_paths=support_paths,
                query_path=neg_rec[0],
                label="No",
                focal_species=focal,
                query_species=confuser,
            ))

        if (i + 1) % 50000 == 0:
            logger.info("  Generated %d / %d examples", i + 1, n_examples)

    return rows


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


def print_stats(rows: list[dict]) -> None:
    """Print summary statistics.

    Parameters
    ----------
    rows
        Generated JSONL rows.
    """
    labels = collections.Counter(r["messages"][1]["content"] for r in rows)
    n_support = collections.Counter(
        len(r["audio_paths"]) - 1 for r in rows
    )
    species_used = collections.Counter(
        json.loads(r["metadata"])["focal_species"] for r in rows
    )

    logger.info("Label balance: %s", dict(labels))
    logger.info("Support count distribution: %s", dict(sorted(n_support.items())))
    logger.info("Unique focal species: %d", len(species_used))
    logger.info(
        "Top 10 focal species: %s",
        species_used.most_common(10),
    )
    logger.info(
        "Bottom 10 focal species: %s",
        species_used.most_common()[-10:],
    )


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """Generate same-species few-shot JSONL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "drasdic_same_species",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=N_EXAMPLES,
    )
    args = parser.parse_args()

    xc = load_csv(XC_CSV)
    inat = load_csv(INAT_CSV)

    species_index = build_species_index(xc, inat)
    logger.info("Total species in index: %d", len(species_index))

    rows = generate_examples(species_index, args.n_examples, SEED)
    print_stats(rows)

    out_path = args.output_dir / "same_species.jsonl"
    write_jsonl(rows, out_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
