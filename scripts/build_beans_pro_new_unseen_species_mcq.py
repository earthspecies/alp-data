#!/usr/bin/env python3
"""Build balanced unseen-species 4-way MCQ from new unseen-taxa holdouts.

This replaces the legacy BEANS-Zero-derived ``unseen-species-4way`` split with
new Xeno-canto and iNaturalist holdout recordings. Each row presents four
species exemplar clips followed by one query clip; the model answers with the
option whose species matches the query.

Usage::

    uv run python scripts/build_beans_pro_new_unseen_species_mcq.py
"""

from __future__ import annotations

import argparse
import collections
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

SEED = 42
LABELS = ["A", "B", "C", "D"]
SPLIT_NAME = "unseen-species-4way"
MIN_RECORDINGS = 2

DEFAULT_XC_HOLDOUT = (
    "/mnt/home/data-ingestion/intermediate/xc_gbif_download/new_unseen_holdout.csv"
)
DEFAULT_INAT_HOLDOUT = (
    "/mnt/home/data-ingestion/intermediate/inaturalist_gbif_download/new_unseen_holdout.csv"
)

INSTRUCTION = (
    "Here are four species.\n\n"
    "A: <Audio><AudioHere></Audio>\n"
    "B: <Audio><AudioHere></Audio>\n"
    "C: <Audio><AudioHere></Audio>\n"
    "D: <Audio><AudioHere></Audio>\n\n"
    "Which species best matches the following recording?\n"
    "<Audio><AudioHere></Audio>"
)

Recording = dict[str, str]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xc-holdout", default=DEFAULT_XC_HOLDOUT)
    parser.add_argument("--inat-holdout", default=DEFAULT_INAT_HOLDOUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "beans_pro_new_unseen_species_mcq",
    )
    parser.add_argument(
        "--max-per-species",
        type=int,
        default=10,
        help="Maximum query rows per species.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--skip-audio-existence-check",
        action="store_true",
        help="Do not filter rows by checking GCS audio object existence.",
    )
    return parser.parse_args()


def clean_value(value: object) -> str:
    """Convert manifest values to compact strings without pandas null sentinels.

    Returns
    -------
    str
        Empty string for null-like values, otherwise stripped text.
    """

    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def load_holdout(path: str, source_dataset: str) -> list[Recording]:
    """Load one holdout CSV into normalized recording dictionaries.

    Returns
    -------
    list[Recording]
        Usable recordings with species, taxonomy, source, and relative audio path.
    """

    columns = [
        "xc_id",
        "inat_id",
        "sound_id",
        "32khz_path",
        "canonical_name",
        "species",
        "species_common",
        "vernacularName",
        "class",
        "order",
        "family",
        "genus",
        "license",
        "media_license",
        "holdout_reason",
    ]
    df0 = pd.read_csv(path, nrows=0)
    usecols = [column for column in columns if column in df0.columns]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    rows: list[Recording] = []
    for _, row in df.iterrows():
        species = clean_value(row.get("canonical_name", "")) or clean_value(row.get("species", ""))
        audio_path = clean_value(row.get("32khz_path", ""))
        if not species or not audio_path:
            continue
        if source_dataset == "xeno-canto":
            rel_audio_path = f"xeno-canto/v0.1.0/raw/audio_32k/{audio_path}"
            source_id = clean_value(row.get("xc_id", ""))
        else:
            rel_audio_path = f"inaturalist/v0.1.0/raw/{audio_path}"
            source_id = clean_value(row.get("sound_id", "")) or clean_value(row.get("inat_id", ""))
        rows.append({
            "audio_path": rel_audio_path,
            "species": species,
            "species_common": clean_value(row.get("species_common", row.get("vernacularName", ""))),
            "class": clean_value(row.get("class", "")),
            "order": clean_value(row.get("order", "")),
            "family": clean_value(row.get("family", "")),
            "genus": clean_value(row.get("genus", "")),
            "license": clean_value(row.get("license", row.get("media_license", ""))),
            "source_dataset": source_dataset,
            "source_id": source_id,
            "holdout_reason": clean_value(row.get("holdout_reason", "")),
        })
    logger.info("Loaded %d usable recordings from %s", len(rows), path)
    return rows


def build_species_index(recordings: list[Recording]) -> dict[str, list[Recording]]:
    """Build species to recordings index.

    Returns
    -------
    dict[str, list[Recording]]
        Recordings grouped by canonical species name.
    """

    species_index: dict[str, list[Recording]] = collections.defaultdict(list)
    for rec in recordings:
        species_index[rec["species"]].append(rec)
    return dict(species_index)


def filter_existing_audio(recordings: list[Recording]) -> list[Recording]:
    """Keep only recordings whose audio object exists in GCS.

    Returns
    -------
    list[Recording]
        Recordings with resolvable audio paths.
    """

    kept = []
    missing = []
    fs = filesystem_from_path("gs://esp-data-ingestion/")
    for rec in recordings:
        uri = f"gs://esp-data-ingestion/{rec['audio_path']}"
        if fs.exists(uri):
            kept.append(rec)
        else:
            missing.append(uri)
    if missing:
        logger.warning("Dropped %d recordings with missing audio", len(missing))
        for uri in missing[:10]:
            logger.warning("  missing: %s", uri)
    logger.info("Audio existence filter kept %d / %d recordings", len(kept), len(recordings))
    return kept


def build_rank_to_species(
    species_index: dict[str, list[Recording]],
    rank: str,
) -> dict[str, list[str]]:
    """Build rank value to species index.

    Returns
    -------
    dict[str, list[str]]
        Taxon value mapped to sorted species names.
    """

    rank_to_species: dict[str, set[str]] = collections.defaultdict(set)
    for species, recs in species_index.items():
        values = {rec[rank] for rec in recs if rec.get(rank)}
        for value in values:
            rank_to_species[value].add(species)
    return {rank_value: sorted(species) for rank_value, species in rank_to_species.items()}


def choose_confusers(
    species: str,
    recs: list[Recording],
    species_index: dict[str, list[Recording]],
    rank_to_species: dict[str, dict[str, list[str]]],
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    """Choose three confuser species, preferring closer taxonomy.

    Returns
    -------
    tuple[list[str], list[str]]
        Confuser species and the taxonomic strategy used for each selected species.
    """

    confusers: list[str] = []
    strategies: list[str] = []
    all_species = sorted(species_index)

    for rank in ("genus", "family", "order", "class"):
        rank_candidates: set[str] = set()
        values = {rec[rank] for rec in recs if rec.get(rank)}
        for value in values:
            rank_candidates.update(rank_to_species[rank].get(value, []))
        rank_candidates.discard(species)
        rank_candidates.difference_update(confusers)
        if not rank_candidates:
            continue
        needed = 3 - len(confusers)
        chosen = rng.sample(sorted(rank_candidates), min(needed, len(rank_candidates)))
        confusers.extend(chosen)
        strategies.extend([rank] * len(chosen))
        if len(confusers) == 3:
            return confusers, strategies

    remaining = [
        candidate
        for candidate in all_species
        if candidate != species and candidate not in confusers
    ]
    chosen = rng.sample(remaining, 3 - len(confusers))
    confusers.extend(chosen)
    strategies.extend(["random"] * len(chosen))
    return confusers, strategies


def make_row(
    *,
    row_idx: int,
    audio_paths: list[str],
    output: str,
    option_species: dict[str, str],
    correct_species: str,
    query_recording: Recording,
    support_recording: Recording,
    confuser_strategies: list[str],
) -> dict[str, object]:
    """Build one BEANS-Pro multi-audio JSONL row.

    Returns
    -------
    dict[str, object]
        JSON-serializable multi-audio row.
    """

    row_id = f"{SPLIT_NAME.replace('-', '_')}_{row_idx:05d}"
    licenses = [query_recording["license"], support_recording["license"]]
    metadata = {
        "option_species": option_species,
        "correct": output,
        "correct_species": correct_species,
        "correct_species_common": query_recording["species_common"],
        "confuser_strategies": confuser_strategies,
        "query_source_dataset": query_recording["source_dataset"],
        "query_source_id": query_recording["source_id"],
        "query_class": query_recording["class"],
        "query_order": query_recording["order"],
        "query_family": query_recording["family"],
        "query_genus": query_recording["genus"],
        "holdout_reason": query_recording["holdout_reason"],
    }
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
        "source_dataset": "xeno-canto+inaturalist new unseen holdouts",
        "dataset_name": SPLIT_NAME,
        "license": "; ".join(sorted({license_value for license_value in licenses if license_value}))
        or "mixed",
        "metadata": json.dumps(metadata, sort_keys=True),
        "audio_path_original_sample_rate": audio_paths[-1],
    }


def generate_rows(
    species_index: dict[str, list[Recording]],
    *,
    max_per_species: int,
    seed: int,
) -> list[dict[str, object]]:
    """Generate balanced unseen-species 4-way MCQ rows.

    Returns
    -------
    list[dict[str, object]]
        Generated rows.
    """

    rng = random.Random(seed)
    focal_species = {
        species: sorted(recs, key=lambda rec: rec["audio_path"])
        for species, recs in species_index.items()
        if len(recs) >= MIN_RECORDINGS
    }
    rank_to_species = {
        rank: build_rank_to_species(species_index, rank)
        for rank in ("genus", "family", "order", "class")
    }
    rows: list[dict[str, object]] = []
    strategy_counts: collections.Counter[str] = collections.Counter()
    for species, recs in sorted(focal_species.items()):
        target_recs = recs
        if len(target_recs) > max_per_species:
            target_recs = sorted(
                rng.sample(target_recs, max_per_species),
                key=lambda rec: rec["audio_path"],
            )
        for target_rec in target_recs:
            support_pool = [rec for rec in recs if rec["audio_path"] != target_rec["audio_path"]]
            support_rec = rng.choice(support_pool)
            confusers, strategies = choose_confusers(
                species,
                recs,
                species_index,
                rank_to_species,
                rng,
            )
            confuser_recs = [rng.choice(species_index[confuser]) for confuser in confusers]
            correct_idx = len(rows) % len(LABELS)
            option_species_list = list(confusers)
            option_species_list.insert(correct_idx, species)
            option_paths = [rec["audio_path"] for rec in confuser_recs]
            option_paths.insert(correct_idx, support_rec["audio_path"])
            strategy_list = list(strategies)
            strategy_list.insert(correct_idx, "correct")
            rows.append(make_row(
                row_idx=len(rows),
                audio_paths=option_paths + [target_rec["audio_path"]],
                output=LABELS[correct_idx],
                option_species=dict(zip(LABELS, option_species_list, strict=True)),
                correct_species=species,
                query_recording=target_rec,
                support_recording=support_rec,
                confuser_strategies=strategy_list,
            ))
            strategy_counts.update(strategies)

    rng.shuffle(rows)
    for idx, row in enumerate(rows):
        row_id = f"{SPLIT_NAME.replace('-', '_')}_{idx:05d}"
        row["id"] = row_id
        row["audio_ids"] = [row_id]
    logger.info("Generated %d rows; confuser strategies: %s", len(rows), dict(strategy_counts))
    return rows


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    """Write rows to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), path)


def write_manifest(
    rows: list[dict[str, object]],
    species_index: dict[str, list[Recording]],
    args: argparse.Namespace,
) -> None:
    """Write generation summary for auditing."""

    species_counts = {species: len(recs) for species, recs in species_index.items()}
    eligible_species = {
        species: count for species, count in species_counts.items() if count >= MIN_RECORDINGS
    }
    answer_counts = collections.Counter(row["messages"][1]["content"] for row in rows)
    correct_species_counts = collections.Counter(
        json.loads(str(row["metadata"]))["correct_species"] for row in rows
    )
    manifest = {
        "xc_holdout": args.xc_holdout,
        "inat_holdout": args.inat_holdout,
        "max_per_species": args.max_per_species,
        "seed": args.seed,
        "rows": len(rows),
        "species_total": len(species_index),
        "species_eligible": len(eligible_species),
        "answer_counts": dict(sorted(answer_counts.items())),
        "correct_species_min_rows": min(correct_species_counts.values()) if rows else 0,
        "correct_species_max_rows": max(correct_species_counts.values()) if rows else 0,
        "top_correct_species": correct_species_counts.most_common(20),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote manifest to %s", manifest_path)


def main() -> None:
    """Build the balanced unseen-species replacement split."""

    args = parse_args()
    recordings = load_holdout(args.xc_holdout, "xeno-canto")
    recordings.extend(load_holdout(args.inat_holdout, "inaturalist"))
    if not args.skip_audio_existence_check:
        recordings = filter_existing_audio(recordings)
    species_index = build_species_index(recordings)
    rows = generate_rows(species_index, max_per_species=args.max_per_species, seed=args.seed)
    write_jsonl(rows, args.output_dir / "unseen_species_4way.jsonl")
    write_manifest(rows, species_index, args)


if __name__ == "__main__":
    main()
