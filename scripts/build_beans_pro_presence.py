#!/usr/bin/env python3
"""Build BEANS-Pro binary presence and call-type evaluation splits.

Generates 8 pre-computed JSONL files for beans_pro:

**Taxonomic presence** (from XC + iNat val_unseen, balanced):
- ``bird-presence``: bird vs non-bird
- ``mammal-presence``: mammal vs non-mammal
- ``insect-presence``: insect vs non-insect
- ``amphibian-presence``: amphibian vs non-amphibian

**Call-type tasks** (from beans_zero_call_variants):
- ``alarm-call-presence``: alarm call binary
- ``flight-call-presence``: flight call binary
- ``begging-call-presence``: begging call binary
- ``call-type-fixed-vocab``: fixed 5-label multilabel

Usage::

    uv run python scripts/build_beans_pro_presence.py
    uv run python scripts/build_beans_pro_presence.py --output-dir /tmp/presence_splits
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import uuid
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

# GCS CSV paths
XC_VAL_UNSEEN = "gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/val_unseen_20260203.csv"
INAT_VAL_UNSEEN = "gs://esp-ml-datasets/inaturalist/v0.1.0/raw/val_unseen_20260201.csv"

# Audio path prefixes (relative to gs://esp-ml-datasets/)
XC_32K_PREFIX = "xeno-canto/v0.1.0/raw/audio_32k/"
INAT_32K_PREFIX = "inaturalist/v0.1.0/raw/"

# Beans-Zero audio root (relative to gs://esp-ml-datasets/)
BEANS_ZERO_PREFIX = "beans-zero/v0.1.0/raw/"

# Taxon presence task definitions: (split_name, task, positive_class, instruction_text)
TAXON_TASKS = [
    (
        "bird-presence",
        "bird_presence",
        "Aves",
        "Is there a bird vocalizing in this recording? Answer Yes or No.",
    ),
    (
        "mammal-presence",
        "mammal_presence",
        "Mammalia",
        "Does this recording contain mammal vocalizations? Answer Yes or No.",
    ),
    (
        "insect-presence",
        "insect_presence",
        "Insecta",
        "Does this recording contain insect sounds? Answer Yes or No.",
    ),
    (
        "amphibian-presence",
        "amphibian_presence",
        "Amphibia",
        "Is there a frog or amphibian vocalizing in this recording? Answer Yes or No.",
    ),
]

# Call-type binary presence: (split_name, source_jsonl, label_col, target_call_type)
CALL_TYPE_BINARY_TASKS = [
    ("alarm-call-presence", "alarm_call_binary.jsonl", "alarm_call_present", "alarm call"),
    ("flight-call-presence", "flight_call_binary.jsonl", "flight_call_present", "flight call"),
    ("begging-call-presence", "begging_call_binary.jsonl", "begging_call_present", "begging call"),
]

# Call-type fixed vocab
FIXED_VOCAB_INSTRUCTION = (
    "Which of the following are present in this recording? "
    "Choose all that apply: alarm call, flight call, begging call, song, call."
)


# ── Helpers ──────────────────────────────────────────────────────────────


def load_gcs_csv(path: str) -> pd.DataFrame:
    """Load a CSV from GCS into a DataFrame.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    logger.info("Loading %s", path)
    fs = filesystem_from_path(path)
    with fs.open(path, "r") as f:
        df = pd.read_csv(f, low_memory=False)
    logger.info("  %d rows, %d columns", len(df), len(df.columns))
    return df


def balance_binary(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Downsample majority class to match minority.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Balanced positive and negative DataFrames.
    """
    n = min(len(positives), len(negatives))
    rng = random.Random(seed)
    if len(positives) > n:
        idx = rng.sample(range(len(positives)), n)
        positives = positives.iloc[sorted(idx)].reset_index(drop=True)
    if len(negatives) > n:
        idx = rng.sample(range(len(negatives)), n)
        negatives = negatives.iloc[sorted(idx)].reset_index(drop=True)
    return positives, negatives


def make_beans_pro_row(
    *,
    source_dataset: str,
    dataset_name: str,
    output: str,
    instruction_text: str,
    task: str,
    audio_path: str,
    license_str: str,
    metadata: dict,
) -> dict:
    """Build a single beans_pro JSONL row.

    Returns
    -------
    dict
        A beans_pro-format row.
    """
    return {
        "source_dataset": source_dataset,
        "dataset_name": dataset_name,
        "output": output,
        "instruction_text": instruction_text,
        "instruction": f"<Audio><AudioHere></Audio> {instruction_text}",
        "task": task,
        "file_name": audio_path.split("/")[-1],
        "license": license_str,
        "id": str(uuid.uuid4()),
        "metadata": json.dumps(metadata),
        "audio_path_original_sample_rate": audio_path,
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write rows to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), path)


def print_balance(rows: list[dict], split_name: str) -> None:
    """Print class balance summary."""
    outputs = [r["output"] for r in rows]
    counts = pd.Series(outputs).value_counts()
    logger.info("  %s: %d rows, balance: %s", split_name, len(rows), dict(counts))


# ── Group A: Taxon presence ─────────────────────────────────────────────


def _xc_audio_path(row: pd.Series) -> str | None:
    """Build relative audio path for XC 32khz under gs://esp-ml-datasets/.

    Returns
    -------
    str | None
        Relative path, or None if no 32khz_path available.
    """
    p = row.get("32khz_path")
    if not p or pd.isna(p) or str(p).strip() == "":
        return None
    return XC_32K_PREFIX + str(p)


def _inat_audio_path(row: pd.Series) -> str | None:
    """Build relative audio path for iNat 32khz under gs://esp-ml-datasets/.

    Returns
    -------
    str | None
        Relative path, or None if no 32khz_path available.
    """
    p = row.get("32khz_path")
    if not p or pd.isna(p) or str(p).strip() == "":
        return None
    return INAT_32K_PREFIX + str(p)


def build_taxon_presence_splits(output_dir: Path) -> None:
    """Build the 4 taxon binary presence JSONL splits.

    Parameters
    ----------
    output_dir
        Directory for output files.
    """
    xc = load_gcs_csv(XC_VAL_UNSEEN)
    inat = load_gcs_csv(INAT_VAL_UNSEEN)

    for split_name, task, pos_class, instruction_text in TAXON_TASKS:
        logger.info("Building %s (positive=%s)", split_name, pos_class)

        # Positives: rows where class == pos_class
        xc_pos = xc[xc["class"] == pos_class].copy()
        inat_pos = inat[inat["class"] == pos_class].copy()

        # Negatives: rows where class != pos_class and class is not empty
        xc_neg = xc[(xc["class"] != pos_class) & (xc["class"].notna()) & (xc["class"] != "")].copy()
        inat_neg = inat[
            (inat["class"] != pos_class) & (inat["class"].notna()) & (inat["class"] != "")
        ].copy()

        n_pos = len(xc_pos) + len(inat_pos)
        n_neg = len(xc_neg) + len(inat_neg)
        logger.info("  Raw: %d pos (%d xc + %d inat), %d neg (%d xc + %d inat)",
                     n_pos, len(xc_pos), len(inat_pos), n_neg, len(xc_neg), len(inat_neg))

        # Build row lists with audio paths
        pos_rows = []
        for _, row in xc_pos.iterrows():
            ap = _xc_audio_path(row)
            if ap is None:
                continue
            pos_rows.append(("xeno-canto", ap, row.get("license", ""), row))

        for _, row in inat_pos.iterrows():
            ap = _inat_audio_path(row)
            if ap is None:
                continue
            pos_rows.append(("inaturalist", ap, row.get("license", ""), row))

        neg_rows = []
        for _, row in xc_neg.iterrows():
            ap = _xc_audio_path(row)
            if ap is None:
                continue
            neg_rows.append(("xeno-canto", ap, row.get("license", ""), row))

        for _, row in inat_neg.iterrows():
            ap = _inat_audio_path(row)
            if ap is None:
                continue
            neg_rows.append(("inaturalist", ap, row.get("license", ""), row))

        # Balance
        n = min(len(pos_rows), len(neg_rows))
        rng = random.Random(SEED)
        if len(pos_rows) > n:
            pos_rows = rng.sample(pos_rows, n)
        if len(neg_rows) > n:
            neg_rows = rng.sample(neg_rows, n)

        logger.info("  Balanced: %d pos + %d neg = %d total", len(pos_rows), len(neg_rows),
                     len(pos_rows) + len(neg_rows))

        # Build JSONL rows
        jsonl_rows = []
        for source, audio_path, lic, raw_row in pos_rows:
            metadata = {
                "class": str(raw_row.get("class", "")),
                "species": str(raw_row.get("species", raw_row.get("species_scientific", ""))),
                "species_common": str(
                    raw_row.get("species_common", raw_row.get("vernacularName", ""))
                ),
                "source": source,
            }
            jsonl_rows.append(make_beans_pro_row(
                source_dataset=source,
                dataset_name=split_name,
                output="Yes",
                instruction_text=instruction_text,
                task=task,
                audio_path=audio_path,
                license_str=str(lic) if not pd.isna(lic) else "",
                metadata=metadata,
            ))

        for source, audio_path, lic, raw_row in neg_rows:
            metadata = {
                "class": str(raw_row.get("class", "")),
                "species": str(raw_row.get("species", raw_row.get("species_scientific", ""))),
                "species_common": str(
                    raw_row.get("species_common", raw_row.get("vernacularName", ""))
                ),
                "source": source,
            }
            jsonl_rows.append(make_beans_pro_row(
                source_dataset=source,
                dataset_name=split_name,
                output="No",
                instruction_text=instruction_text,
                task=task,
                audio_path=audio_path,
                license_str=str(lic) if not pd.isna(lic) else "",
                metadata=metadata,
            ))

        # Shuffle so Yes/No aren't grouped
        rng.shuffle(jsonl_rows)

        print_balance(jsonl_rows, split_name)
        out_path = output_dir / f"{split_name.replace('-', '_')}.jsonl"
        write_jsonl(jsonl_rows, out_path)


# ── Group B: Call-type tasks ─────────────────────────────────────────────


def _find_call_variants_dir() -> Path:
    """Locate the beans_zero_call_variants_from_mapping directory.

    Returns
    -------
    Path
        Path to the manifest directory.

    Raises
    ------
    FileNotFoundError
        If the directory cannot be found.
    """
    candidates = [
        REPO_ROOT / "data" / "beans_zero_call_variants_from_mapping",
        Path.home() / "esp-data-dev" / "data" / "beans_zero_call_variants_from_mapping",
    ]
    for c in candidates:
        if c.exists() and (c / "alarm_call_binary.jsonl").exists():
            return c
    raise FileNotFoundError(
        "Cannot find beans_zero_call_variants_from_mapping directory. "
        f"Searched: {[str(c) for c in candidates]}"
    )


def build_call_type_splits(output_dir: Path) -> None:
    """Build the 4 call-type JSONL splits from beans_zero_call_variants.

    Parameters
    ----------
    output_dir
        Directory for output files.
    """
    variants_dir = _find_call_variants_dir()
    logger.info("Using call variants from %s", variants_dir)

    # Binary presence tasks
    for split_name, source_file, label_col, target_call_type in CALL_TYPE_BINARY_TASKS:
        logger.info("Building %s from %s", split_name, source_file)
        source_path = variants_dir / source_file

        rows_in = []
        with open(source_path) as f:
            for line in f:
                rows_in.append(json.loads(line))
        logger.info("  Loaded %d rows", len(rows_in))

        instruction_text = (
            f"Is a {target_call_type} present in this recording? Answer Yes or No."
        )

        jsonl_rows = []
        for row in rows_in:
            label = row.get(label_col, "")
            if label not in ("Yes", "No"):
                continue
            audio_path = BEANS_ZERO_PREFIX + row["audio_path_original_sample_rate"]
            metadata = {
                "target_call_type": target_call_type,
                "beans_zero_label": row.get("beans_zero_label", ""),
                "xc_species": row.get("xc_species", ""),
                "source": "beans_zero_call_variants",
            }
            jsonl_rows.append(make_beans_pro_row(
                source_dataset=row.get("source_dataset", "Xeno-canto"),
                dataset_name=split_name,
                output=label,
                instruction_text=instruction_text,
                task="call_type_presence_binary",
                audio_path=audio_path,
                license_str=row.get("license", ""),
                metadata=metadata,
            ))

        print_balance(jsonl_rows, split_name)
        out_path = output_dir / f"{split_name.replace('-', '_')}.jsonl"
        write_jsonl(jsonl_rows, out_path)

    # Fixed-vocab multilabel task
    logger.info("Building call-type-fixed-vocab from linkage_manifest.jsonl")
    linkage_path = variants_dir / "linkage_manifest.jsonl"
    rows_in = []
    with open(linkage_path) as f:
        for line in f:
            rows_in.append(json.loads(line))
    logger.info("  Loaded %d rows", len(rows_in))

    jsonl_rows = []
    for row in rows_in:
        bfv = row.get("behavior_fixed_vocab", "")
        if not bfv or not str(bfv).strip():
            continue
        audio_path = BEANS_ZERO_PREFIX + row["audio_path_original_sample_rate"]
        metadata = {
            "behavior_fixed_vocab_count": row.get("behavior_fixed_vocab_count", 0),
            "xc_species": row.get("xc_species", ""),
            "xc_behavior_raw": row.get("xc_behavior_raw", ""),
            "source": "beans_zero_call_variants",
        }
        jsonl_rows.append(make_beans_pro_row(
            source_dataset=row.get("source_dataset", "Xeno-canto"),
            dataset_name="call-type-fixed-vocab",
            output=str(bfv),
            instruction_text=FIXED_VOCAB_INSTRUCTION,
            task="call_type_fixed_vocab",
            audio_path=audio_path,
            license_str=row.get("license", ""),
            metadata=metadata,
        ))

    logger.info("  call-type-fixed-vocab: %d rows (after filtering empty)", len(jsonl_rows))
    out_path = output_dir / "call_type_fixed_vocab.jsonl"
    write_jsonl(jsonl_rows, out_path)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """Generate all presence and call-type JSONL splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "beans_pro_presence",
        help="Directory for output JSONL files.",
    )
    args = parser.parse_args()

    build_taxon_presence_splits(args.output_dir)
    build_call_type_splits(args.output_dir)

    logger.info("Done! Output in %s", args.output_dir)


if __name__ == "__main__":
    main()
