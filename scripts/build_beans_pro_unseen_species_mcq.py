#!/usr/bin/env python3
"""Build unseen-species few-shot 4-way MCQ eval splits for BEANS-Pro.

Creates multi-audio 4-way species classification benchmarks from the
BEANS-Zero unseen taxonomy holdouts. For each of three unseen levels
(species, genus, family), generates two variants:

- **random**: confusers drawn uniformly from other species in the level
- **hard**: confusers drawn from the closest taxonomic group
  (same-genus for unseen-species, same-family for unseen-genus,
  same-order for unseen-family), falling back to random when needed

Usage::

    uv run python scripts/build_beans_pro_unseen_species_mcq.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
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
MIN_RECORDINGS = 2  # focal species needs >=2 (1 option + 1 target)

INSTRUCTION = (
    "Here are four species.\n\n"
    "A: <Audio><AudioHere></Audio>\n"
    "B: <Audio><AudioHere></Audio>\n"
    "C: <Audio><AudioHere></Audio>\n"
    "D: <Audio><AudioHere></Audio>\n\n"
    "Which species best matches the following recording?\n"
    "<Audio><AudioHere></Audio>"
)

_BZ_BASE = "gs://esp-ml-datasets/beans-zero/v0.1.0/raw"

_UNSEEN_JSONL = {
    "species": f"{_BZ_BASE}/unseen-species-sci_test.jsonl",
    "genus": f"{_BZ_BASE}/unseen-genus-sci_test.jsonl",
    "family": f"{_BZ_BASE}/unseen-family-sci_test.jsonl",
}

MAPPING_CSV = f"{_BZ_BASE}/unseen_xc_mapping.csv"

# Which taxonomic rank to use for hard negatives at each unseen level.
# For unseen-family, genus is used because order/family info is missing
# for non-XC recordings, and same-genus confusers are the hardest
# (same genus implies same family).
_HARD_NEG_RANK = {
    "species": "genus",
    "genus": "family",
    "family": "genus",
}

# ── Data types ───────────────────────────────────────────────────────────

Recording = dict[str, str]  # keys: audio_path, species, genus, family, order


# ── Helpers ──────────────────────────────────────────────────────────────


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load a JSONL file from GCS.

    Parameters
    ----------
    path : str
        GCS path.

    Returns
    -------
    list[dict[str, Any]]
        Parsed records.
    """
    fs = filesystem_from_path(path)
    records = []
    with fs.open(str(path), "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d rows from %s", len(records), path)
    return records


def load_mapping_csv(
    path: str,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Load the unseen_xc_mapping.csv and build two indexes.

    Returns both a per-recording (``bz_id``) mapping and a per-species
    taxonomy lookup.  The per-species lookup is built from the *best*
    available taxonomy for each species (only XC rows have full taxonomy;
    iNat/Watkins/ASA rows are often missing genus/family/order).

    Parameters
    ----------
    path : str
        GCS path to the CSV.

    Returns
    -------
    tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]
        ``(by_id, by_species)`` where ``by_id`` maps ``bz_id`` to
        taxonomy and ``by_species`` maps ``canonical_name`` to the
        best-known taxonomy for that species.
    """
    fs = filesystem_from_path(path)
    with fs.open(str(path), "r") as f:
        content = f.read()
    reader = csv.DictReader(io.StringIO(content))
    by_id: dict[str, dict[str, str]] = {}
    by_species: dict[str, dict[str, str]] = {}
    for row in reader:
        bz_id = row.get("bz_id", "")
        sp = row.get("canonical_name", "")
        genus = row.get("genus", "")
        family = row.get("family", "")
        order = row.get("order", "")

        tax = {"species": sp, "genus": genus, "family": family, "order": order}
        if bz_id:
            by_id[bz_id] = tax

        # Keep the most complete taxonomy per species (XC rows typically
        # have genus/family/order while other sources don't).
        if sp and genus:
            existing = by_species.get(sp)
            if existing is None or (not existing["family"] and family):
                by_species[sp] = tax

    # Infer genus from binomial name for species missing from XC
    for sp, tax in list(by_species.items()):
        if not tax["genus"] and " " in sp:
            tax["genus"] = sp.split()[0]

    logger.info(
        "Loaded %d mapping entries, %d species with taxonomy from %s",
        len(by_id), len(by_species), path,
    )
    return by_id, by_species


def build_species_index(
    records: list[dict[str, Any]],
    by_id: dict[str, dict[str, str]],
    by_species: dict[str, dict[str, str]],
    level: str,
) -> dict[str, list[Recording]]:
    """Build species → list of Recording from beans-zero JSONL + mapping.

    Uses the per-recording mapping first, then falls back to the
    per-species taxonomy lookup (which propagates XC taxonomy to
    non-XC recordings of the same species).

    Parameters
    ----------
    records : list[dict[str, Any]]
        Beans-zero unseen JSONL records.
    by_id : dict[str, dict[str, str]]
        ``bz_id`` → taxonomy mapping.
    by_species : dict[str, dict[str, str]]
        ``canonical_name`` → best-known taxonomy.
    level : str
        Unseen level (``"species"``, ``"genus"``, ``"family"``).

    Returns
    -------
    dict[str, list[Recording]]
        Maps species name to list of recordings with taxonomy.
    """
    index: dict[str, list[Recording]] = collections.defaultdict(list)
    missing_tax = 0

    for row in records:
        species = row["output"]
        bz_id = row["id"]
        audio_path = row["audio_path_32KHz"]

        # Try per-recording mapping first, then species-level fallback
        tax = by_id.get(bz_id)
        if tax is None or not tax.get("genus"):
            tax = by_species.get(species)
        if tax is None:
            missing_tax += 1
            # Last resort: infer genus from binomial name
            genus = species.split()[0] if " " in species else ""
            tax = {"species": species, "genus": genus, "family": "", "order": ""}

        index[species].append({
            "audio_path": audio_path,
            "species": species,
            "genus": tax["genus"],
            "family": tax["family"],
            "order": tax["order"],
            "bz_id": bz_id,
        })

    if missing_tax:
        logger.warning("  %d recordings missing taxonomy for level %s", missing_tax, level)

    logger.info(
        "  %d species, %d total recordings",
        len(index), sum(len(v) for v in index.values()),
    )
    return dict(index)


def build_taxon_to_species(
    species_index: dict[str, list[Recording]],
    rank: str,
) -> dict[str, list[str]]:
    """Build taxon → list of species for hard negative sampling.

    Parameters
    ----------
    species_index : dict[str, list[Recording]]
        Species → recordings mapping.
    rank : str
        Taxonomic rank key (``"genus"``, ``"family"``, ``"order"``).

    Returns
    -------
    dict[str, list[str]]
        Maps taxon name to list of species.
    """
    taxon_to_sp: dict[str, set[str]] = collections.defaultdict(set)
    for sp, recs in species_index.items():
        taxa = {r[rank] for r in recs if r[rank] and r[rank] != "nan"}
        for t in taxa:
            taxon_to_sp[t].add(sp)
    return {t: list(sps) for t, sps in taxon_to_sp.items()}


def make_row(
    *,
    split_name: str,
    row_idx: int,
    audio_paths: list[str],
    output: str,
    option_species: dict[str, str],
    correct_species: str,
    hard_negatives: bool,
    original_bz_id: str,
) -> dict[str, Any]:
    """Build a single JSONL row.

    Returns
    -------
    dict[str, Any]
        JSONL-ready row.
    """
    row_id = f"{split_name.replace('-', '_')}_{row_idx:05d}"
    return {
        "id": row_id,
        "audio_paths": audio_paths,
        "audio_ids": [row_id],
        "template_path": "species_mcq",
        "skills": ["multiple_choice", "species_classification"],
        "messages": [
            {"role": "user", "content": INSTRUCTION},
            {"role": "assistant", "content": output},
        ],
        "task": "species_mcq",
        "source_dataset": "xeno-canto+inaturalist (BEANS-Zero unseen)",
        "dataset_name": split_name,
        "license": "mixed",
        "metadata": json.dumps({
            "option_species": option_species,
            "correct": output,
            "correct_species": correct_species,
            "hard_negatives": hard_negatives,
        }),
        "audio_path_original_sample_rate": audio_paths[-1],
        "original_beans_zero_id": original_bz_id,
    }


def generate_4way(
    species_index: dict[str, list[Recording]],
    split_name: str,
    hard_neg_rank: str | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Generate 4-way species MCQ from an unseen level.

    One question per target clip from species with >=2 recordings.

    Parameters
    ----------
    species_index : dict[str, list[Recording]]
        Species → recordings.
    split_name : str
        Output split name.
    hard_neg_rank : str | None
        Taxonomic rank for hard negatives (``None`` for random only).
    rng : random.Random
        Seeded RNG.

    Returns
    -------
    list[dict[str, Any]]
        JSONL rows.
    """
    # Focal species: need >=2 recordings
    focal_species = {
        sp: recs for sp, recs in species_index.items()
        if len(recs) >= MIN_RECORDINGS
    }
    all_species = list(species_index.keys())

    # Build hard negative index if needed
    taxon_to_sp: dict[str, list[str]] | None = None
    if hard_neg_rank:
        taxon_to_sp = build_taxon_to_species(species_index, hard_neg_rank)

    logger.info(
        "  Focal species (>=%d): %d / %d, hard_neg_rank: %s",
        MIN_RECORDINGS, len(focal_species), len(all_species), hard_neg_rank,
    )

    rows: list[dict[str, Any]] = []
    hard_count = 0

    for sp, recs in sorted(focal_species.items()):
        for target_rec in recs:
            target_path = target_rec["audio_path"]
            target_bz_id = target_rec["bz_id"]

            # Pick a DIFFERENT recording from the same species as the option
            other_recs = [r for r in recs if r["audio_path"] != target_path]
            if not other_recs:
                continue
            option_rec = rng.choice(other_recs)
            focal_option_path = option_rec["audio_path"]

            # Pick 3 confuser species
            confusers: list[str] = []
            is_hard = False

            if hard_neg_rank and taxon_to_sp:
                # Try hard negatives from same taxon.
                # Use as many same-taxon confusers as available
                # (up to 3), then fill the rest with random.
                focal_taxa = {
                    r[hard_neg_rank] for r in recs
                    if r[hard_neg_rank] and r[hard_neg_rank] != "nan"
                }
                candidates: set[str] = set()
                for t in focal_taxa:
                    candidates.update(taxon_to_sp.get(t, []))
                candidates.discard(sp)
                if candidates:
                    n_hard = min(3, len(candidates))
                    confusers = rng.sample(list(candidates), n_hard)
                    is_hard = True

            if len(confusers) < 3:
                # Fill remaining slots with random species
                pool = [s for s in all_species if s != sp and s not in confusers]
                n_remaining = 3 - len(confusers)
                confusers.extend(rng.sample(pool, n_remaining))

            # Pick one clip per confuser
            confuser_paths = [
                rng.choice(species_index[c])["audio_path"]
                for c in confusers
            ]

            # Assign correct position (balanced across A/B/C/D)
            correct_idx = len(rows) % 4

            option_species_list = list(confusers)
            option_species_list.insert(correct_idx, sp)
            option_paths = list(confuser_paths)
            option_paths.insert(correct_idx, focal_option_path)

            audio_paths = option_paths + [target_path]

            if is_hard:
                hard_count += 1

            rows.append(make_row(
                split_name=split_name,
                row_idx=len(rows),
                audio_paths=audio_paths,
                output=LABELS[correct_idx],
                option_species=dict(zip(LABELS, option_species_list, strict=True)),
                correct_species=sp,
                hard_negatives=is_hard,
                original_bz_id=target_bz_id,
            ))

    rng.shuffle(rows)
    # Re-index after shuffle
    for i, row in enumerate(rows):
        row["id"] = f"{split_name.replace('-', '_')}_{i:05d}"
        row["audio_ids"] = [row["id"]]

    logger.info(
        "  %s: %d rows, hard negatives: %d (%.1f%%)",
        split_name, len(rows), hard_count,
        hard_count / len(rows) * 100 if rows else 0,
    )
    return rows


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


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """Generate unseen-species 4-way MCQ eval splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "beans_pro_unseen_species_mcq",
    )
    args = parser.parse_args()

    # Load taxonomy mapping
    by_id, by_species = load_mapping_csv(MAPPING_CSV)

    for level in ["species", "genus", "family"]:
        logger.info("=== %s ===", level)

        records = load_jsonl(_UNSEEN_JSONL[level])
        species_index = build_species_index(records, by_id, by_species, level)

        hard_rank = _HARD_NEG_RANK[level]

        # Random negatives variant
        rng_rand = random.Random(SEED)
        split_rand = f"unseen-{level}-4way"
        rows_rand = generate_4way(
            species_index, split_rand, hard_neg_rank=None, rng=rng_rand,
        )
        answer_dist = collections.Counter(r["messages"][1]["content"] for r in rows_rand)
        logger.info("  answers: %s", dict(sorted(answer_dist.items())))
        write_jsonl(rows_rand, args.output_dir / f"unseen_{level}_4way.jsonl")

        # Hard negatives variant
        rng_hard = random.Random(SEED)
        split_hard = f"unseen-{level}-4way-hard"
        rows_hard = generate_4way(
            species_index, split_hard, hard_neg_rank=hard_rank, rng=rng_hard,
        )
        answer_dist_h = collections.Counter(r["messages"][1]["content"] for r in rows_hard)
        logger.info("  answers: %s", dict(sorted(answer_dist_h.items())))
        write_jsonl(rows_hard, args.output_dir / f"unseen_{level}_4way_hard.jsonl")

    logger.info("Done!")


if __name__ == "__main__":
    main()
