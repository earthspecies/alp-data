#!/usr/bin/env python3
"""Build BEANS-Pro-style evaluation splits from the new Xeno-canto release.

The generated JSONL files keep the frozen BEANS-Pro row format while using the
2026-04-29 ``all_unseen_new_only`` manifest as the source. This is intended as a
reviewable local reprocessing step before publishing a new BEANS-Pro version.

Usage::

    uv run python scripts/build_beans_pro_xc_new_eval.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
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
DEFAULT_SOURCE_CSV = (
    "gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/all_unseen_new_only_20260429.csv"
)
DEFAULT_AUDIO_PREFIX = "audio_32k/"

NOISY_BEHAVIOR_SUBSTRINGS = (
    "{",
    "background",
    "playback",
    " pb",
    "highway",
    "aircraft",
    "traffic",
    "people",
    "microphone",
    "http",
    "windmill",
    "tractor",
    "motor",
    "machine",
    "in training",
)

PRESENCE_TEXT_COLUMNS = (
    "caption",
    "caption2",
    "caption3",
    "description",
    "fieldNotes",
    "occurrenceRemarks",
    "media_description",
    "Associated Taxa",
    "behavior",
)

PRESENCE_TEXT_PATTERNS = {
    "Aves": re.compile(r"\b(bird|birds|avian|aves)\b", re.IGNORECASE),
    "Mammalia": re.compile(
        r"\b(mammal|mammals|bat|bats|whale|whales|dolphin|dolphins|"
        r"deer|monkey|dog|wolf|fox|seal|otter)\b",
        re.IGNORECASE,
    ),
    "Insecta": re.compile(
        r"\b(insect|insects|cricket|grasshopper|katydid|cicada|bee|beetle)\b",
        re.IGNORECASE,
    ),
    "Amphibia": re.compile(r"\b(amphibian|amphibians|frog|frogs|toad|toads)\b", re.IGNORECASE),
}

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

CALL_TYPE_BINARY_TASKS = [
    ("alarm-call-presence", "alarm call"),
    ("flight-call-presence", "flight call"),
    ("begging-call-presence", "begging call"),
]

FIXED_VOCAB_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("alarm call", re.compile(r"\balarm call\b", re.IGNORECASE)),
    ("flight call", re.compile(r"\bflight call\b", re.IGNORECASE)),
    ("begging call", re.compile(r"\bbegging call\b", re.IGNORECASE)),
    ("song", re.compile(r"\bsong\b", re.IGNORECASE)),
    ("call", re.compile(r"(^|,\s*)call(\s*,|$)", re.IGNORECASE)),
]

FIXED_VOCAB_INSTRUCTION = (
    "Which of the following are present in this recording? "
    "Choose all that apply: alarm call, flight call, begging call, song, call."
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        default=DEFAULT_SOURCE_CSV,
        help="Xeno-canto manifest CSV to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "beans_pro_xc_new_20260429",
        help="Directory for generated JSONL files.",
    )
    parser.add_argument(
        "--audio-prefix",
        default=DEFAULT_AUDIO_PREFIX,
        help="Prefix prepended to 32khz_path in generated rows.",
    )
    parser.add_argument(
        "--max-taxon-per-label",
        type=int,
        default=1000,
        help="Maximum Yes or No examples per taxon-presence split.",
    )
    parser.add_argument(
        "--max-call-type-per-label",
        type=int,
        default=1000,
        help="Maximum Yes or No examples per call-type binary split.",
    )
    parser.add_argument(
        "--max-fixed-vocab-per-output",
        type=int,
        default=500,
        help="Maximum examples per fixed-vocabulary output combination.",
    )
    parser.add_argument(
        "--max-fixed-vocab-total",
        type=int,
        default=2000,
        help="Maximum total examples for the fixed-vocabulary split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for deterministic sampling and shuffling.",
    )
    return parser.parse_args()


def load_source_csv(path: str) -> pd.DataFrame:
    """Load the source Xeno-canto manifest.

    Parameters
    ----------
    path
        GCS or local CSV path.

    Returns
    -------
    pd.DataFrame
        Source rows with the columns needed by this builder.
    """

    columns = [
        "xc_id",
        "relative_path",
        "32khz_path",
        "class",
        "species",
        "species_common",
        "vernacularName",
        "scientificName",
        "canonical_name",
        "behavior",
        "Associated Taxa",
        "caption",
        "caption2",
        "caption3",
        "description",
        "fieldNotes",
        "occurrenceRemarks",
        "media_description",
        "license",
        "license_url",
        "media_license",
        "media_license_url",
        "source_version",
        "audio_exists",
    ]
    logger.info("Loading %s", path)
    fs = filesystem_from_path(path)
    with fs.open(path, "r") as handle:
        available_columns = pd.read_csv(handle, nrows=0).columns.tolist()
    usecols = [column for column in columns if column in available_columns]
    with fs.open(path, "r") as handle:
        df = pd.read_csv(handle, usecols=usecols, low_memory=False)
    logger.info("Loaded %d rows with %d columns", len(df), len(df.columns))
    return df


def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply source-level quality filters shared by all generated splits.

    Parameters
    ----------
    df
        Raw manifest rows.

    Returns
    -------
    pd.DataFrame
        Rows with usable audio paths and taxonomic class labels.
    """

    out = df.copy()
    out["32khz_path"] = out["32khz_path"].fillna("").astype(str).str.strip()
    out["class"] = out["class"].fillna("").astype(str).str.strip()
    out = out[(out["32khz_path"] != "") & (out["class"] != "")]
    if "audio_exists" in out.columns:
        audio_exists = out["audio_exists"].astype(str).str.lower().isin({"true", "1", "yes"})
        if audio_exists.any():
            out = out[audio_exists]
        else:
            logger.info("Skipping audio_exists filter because no rows are marked true")
    subset = ["32khz_path"]
    if "xc_id" in out.columns:
        subset.append("xc_id")
    out = out.drop_duplicates(subset=subset).reset_index(drop=True)
    logger.info("After quality filter: %d rows", len(out))
    return out


def clean_behavior_mask(series: pd.Series) -> pd.Series:
    """Return a mask for rows with usable behavior annotations.

    Returns
    -------
    pd.Series
        Boolean mask aligned to ``series``.
    """

    behavior = series.fillna("").astype(str).str.strip()
    mask = (behavior != "") & (~behavior.str.lower().isin({"?", "uncertain"}))
    behavior_lower = behavior.str.lower()
    for substring in NOISY_BEHAVIOR_SUBSTRINGS:
        mask &= ~behavior_lower.str.contains(substring, regex=False, na=False)
    return mask


def fixed_vocab_labels(behavior: str) -> list[str]:
    """Extract fixed-vocabulary call-type labels from a behavior string.

    Returns
    -------
    list[str]
        Labels matched in fixed-vocabulary order.
    """

    return [label for label, pattern in FIXED_VOCAB_PATTERNS if pattern.search(behavior)]


def associated_taxon_names(value: object) -> list[str]:
    """Parse the Xeno-canto associated-taxa field into candidate names.

    Returns
    -------
    list[str]
        Parsed scientific/common names listed as background taxa.
    """

    text = clean_value(value)
    if not text:
        return []
    text = re.sub(r"^\s*has background sounds:\s*", "", text, flags=re.IGNORECASE)
    return [part.strip() for part in re.split(r"[|;,]", text) if part.strip()]


def build_name_to_classes(df: pd.DataFrame) -> dict[str, set[str]]:
    """Build a lookup from taxon names to observed classes in the manifest.

    Returns
    -------
    dict[str, set[str]]
        Lowercased taxon/common names mapped to one or more classes.
    """

    lookup: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        class_name = clean_value(row.get("class", ""))
        if not class_name:
            continue
        for column in ("canonical_name", "species", "scientificName", "species_common"):
            name = clean_value(row.get(column, ""))
            if name:
                lookup.setdefault(name.lower(), set()).add(class_name)
    return lookup


def associated_classes(value: object, name_to_classes: dict[str, set[str]]) -> set[str]:
    """Map associated taxa names to known classes.

    Returns
    -------
    set[str]
        Classes recovered for associated/background taxa.
    """

    classes: set[str] = set()
    for name in associated_taxon_names(value):
        classes.update(name_to_classes.get(name.lower(), set()))
    return classes


def text_mentions_target(row: pd.Series, target_class: str) -> bool:
    """Return whether source metadata text mentions the target taxon broadly.

    Returns
    -------
    bool
        True when a conservative keyword pattern matches any text field.
    """

    pattern = PRESENCE_TEXT_PATTERNS[target_class]
    text = " ".join(clean_value(row.get(column, "")) for column in PRESENCE_TEXT_COLUMNS)
    return bool(pattern.search(text))


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


def stable_id(split_name: str, row: pd.Series, output: str) -> str:
    """Create a stable row identifier from source keys and the assigned output.

    Returns
    -------
    str
        Deterministic identifier for the generated row.
    """

    key = "|".join(
        [
            split_name,
            str(row.get("xc_id", "")),
            str(row.get("32khz_path", "")),
            output,
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{split_name.replace('-', '_')}_{digest}"


def audio_path(row: pd.Series, audio_prefix: str) -> str:
    """Build the row audio path relative to the new release root.

    Returns
    -------
    str
        Relative audio path for the generated row.
    """

    return f"{audio_prefix.rstrip('/')}/{row['32khz_path']}"


def metadata(row: pd.Series, extra: dict[str, object] | None = None) -> dict[str, object]:
    """Build compact row metadata for provenance and auditing.

    Returns
    -------
    dict[str, object]
        JSON-serializable metadata payload.
    """

    values = {
        "xc_id": clean_value(row.get("xc_id", "")),
        "class": clean_value(row.get("class", "")),
        "species": clean_value(row.get("species", row.get("canonical_name", ""))),
        "species_common": clean_value(row.get("species_common", row.get("vernacularName", ""))),
        "canonical_name": clean_value(row.get("canonical_name", "")),
        "behavior": clean_value(row.get("behavior", "")),
        "source_version": clean_value(row.get("source_version", "")),
        "source": "xeno-canto-new-only-20260429",
    }
    if extra:
        values.update(extra)
    return values


def make_row(
    *,
    split_name: str,
    output: str,
    instruction_text: str,
    task: str,
    row: pd.Series,
    audio_prefix: str,
    extra_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one BEANS-Pro-format JSONL record.

    Returns
    -------
    dict[str, object]
        JSON-serializable BEANS-Pro row.
    """

    row_audio_path = audio_path(row, audio_prefix)
    return {
        "source_dataset": "xeno-canto",
        "dataset_name": split_name,
        "output": output,
        "instruction_text": instruction_text,
        "instruction": f"<Audio><AudioHere></Audio> {instruction_text}",
        "task": task,
        "file_name": Path(row_audio_path).name,
        "license": str(row.get("license", "")),
        "id": stable_id(split_name, row, output),
        "metadata": json.dumps(metadata(row, extra_metadata), sort_keys=True),
        "audio_path_original_sample_rate": row_audio_path,
    }


def sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample up to ``n`` rows without replacement.

    Returns
    -------
    pd.DataFrame
        Sampled rows, or all rows when ``len(df) <= n``.
    """

    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def proportional_sample_counts(counts: dict[str, int], max_total: int) -> dict[str, int]:
    """Allocate a total cap across groups while preserving proportions.

    Returns
    -------
    dict[str, int]
        Per-group sample counts summing to at most ``max_total``.
    """

    total = sum(counts.values())
    if total <= max_total:
        return dict(counts)
    nonempty_keys = [key for key, value in counts.items() if value > 0]
    if len(nonempty_keys) <= max_total:
        allocated = {key: (1 if counts[key] > 0 else 0) for key in counts}
        remaining_budget = max_total - sum(allocated.values())
    else:
        allocated = {key: 0 for key in counts}
        remaining_budget = max_total
    remaining_pool = sum(max(counts[key] - allocated[key], 0) for key in counts)
    quotas = {
        key: max(counts[key] - allocated[key], 0) * remaining_budget / remaining_pool
        for key in counts
    }
    for key in counts:
        allocated[key] += min(counts[key] - allocated[key], int(quotas[key]))
    remaining = max_total - sum(allocated.values())
    remainders = sorted(
        counts,
        key=lambda key: (quotas[key] - allocated[key], counts[key], key),
        reverse=True,
    )
    for key in remainders:
        if remaining <= 0:
            break
        if allocated[key] < counts[key]:
            allocated[key] += 1
            remaining -= 1
    return allocated


def shuffle_records(records: list[dict[str, object]], seed: int) -> list[dict[str, object]]:
    """Return deterministically shuffled records.

    Returns
    -------
    list[dict[str, object]]
        Shuffled copy of ``records``.
    """

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def write_jsonl(records: list[dict[str, object]], path: Path) -> None:
    """Write records as newline-delimited JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    logger.info("Wrote %d rows to %s", len(records), path)


def build_taxon_presence_splits(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    max_per_label: int,
    audio_prefix: str,
    seed: int,
) -> dict[str, int]:
    """Build balanced cross-taxonomic binary presence splits.

    Returns
    -------
    dict[str, int]
        Generated row counts keyed by split name.
    """

    taxon_df = df[df["behavior"].pipe(clean_behavior_mask)].copy()
    logger.info("Taxon presence rows after clean behavior filter: %d", len(taxon_df))
    name_to_classes = build_name_to_classes(taxon_df)

    sizes = {}
    audits = {}
    for split_name, task, positive_class, instruction_text in TAXON_TASKS:
        positives = taxon_df[taxon_df["class"] == positive_class]
        raw_negatives = taxon_df[taxon_df["class"] != positive_class].copy()
        associated_target = raw_negatives["Associated Taxa"].map(
            lambda value, target=positive_class: target
            in associated_classes(value, name_to_classes)
        )
        text_target = raw_negatives.apply(
            lambda row, target=positive_class: text_mentions_target(row, target),
            axis=1,
        )
        negatives = raw_negatives[~associated_target & ~text_target]
        positive_candidates = len(positives)
        strict_negative_candidates = len(negatives)
        n = min(len(positives), len(negatives), max_per_label)
        positives = sample_rows(positives, n, seed)
        negatives = sample_rows(negatives, n, seed + 1)
        audits[split_name] = {
            "target_class": positive_class,
            "positive_candidates": int(positive_candidates),
            "raw_negative_candidates": int(len(raw_negatives)),
            "strict_negative_candidates": int(strict_negative_candidates),
            "removed_associated_target": int(associated_target.sum()),
            "removed_text_target": int(text_target.sum()),
            "removed_any_target_evidence": int((associated_target | text_target).sum()),
            "sampled_per_label": int(n),
        }
        records = [
            make_row(
                split_name=split_name,
                output="Yes",
                instruction_text=instruction_text,
                task=task,
                row=row,
                audio_prefix=audio_prefix,
            )
            for _, row in positives.iterrows()
        ]
        records.extend(
            make_row(
                split_name=split_name,
                output="No",
                instruction_text=instruction_text,
                task=task,
                row=row,
                audio_prefix=audio_prefix,
            )
            for _, row in negatives.iterrows()
        )
        records = shuffle_records(records, seed)
        write_jsonl(records, output_dir / f"{split_name.replace('-', '_')}.jsonl")
        sizes[split_name] = len(records)
        logger.info(
            "%s: %d Yes + %d strict No (%d raw No candidates, %d removed)",
            split_name,
            n,
            n,
            len(raw_negatives),
            int((associated_target | text_target).sum()),
        )
    audit_path = output_dir / "presence_negative_audit.json"
    audit_path.write_text(json.dumps(audits, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote presence negative audit to %s", audit_path)
    return sizes


def build_call_type_binary_splits(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    max_per_label: int,
    audio_prefix: str,
    seed: int,
) -> dict[str, int]:
    """Build balanced binary call-type presence splits from cleaned behavior.

    Returns
    -------
    dict[str, int]
        Generated row counts keyed by split name.
    """

    behavior_df = df[clean_behavior_mask(df["behavior"])].copy()
    behavior_df["fixed_vocab_labels"] = behavior_df["behavior"].map(fixed_vocab_labels)
    behavior_df = behavior_df[behavior_df["fixed_vocab_labels"].map(bool)].copy()

    sizes = {}
    for offset, (split_name, target_call_type) in enumerate(CALL_TYPE_BINARY_TASKS):
        target = target_call_type
        positives = behavior_df[
            behavior_df["fixed_vocab_labels"].map(lambda labels, target=target: target in labels)
        ]
        negatives = behavior_df[
            behavior_df["fixed_vocab_labels"].map(
                lambda labels, target=target: target not in labels
            )
        ]
        n = min(len(positives), len(negatives), max_per_label)
        positives = sample_rows(positives, n, seed + offset)
        negatives = sample_rows(negatives, n, seed + offset + 100)
        instruction_text = (
            f"Is a {target_call_type} present in this recording? Answer Yes or No."
        )
        records = [
            make_row(
                split_name=split_name,
                output="Yes",
                instruction_text=instruction_text,
                task="call_type_presence_binary",
                row=row,
                audio_prefix=audio_prefix,
                extra_metadata={
                    "target_call_type": target_call_type,
                    "behavior_fixed_vocab": ", ".join(row["fixed_vocab_labels"]),
                },
            )
            for _, row in positives.iterrows()
        ]
        records.extend(
            make_row(
                split_name=split_name,
                output="No",
                instruction_text=instruction_text,
                task="call_type_presence_binary",
                row=row,
                audio_prefix=audio_prefix,
                extra_metadata={
                    "target_call_type": target_call_type,
                    "behavior_fixed_vocab": ", ".join(row["fixed_vocab_labels"]),
                },
            )
            for _, row in negatives.iterrows()
        )
        records = shuffle_records(records, seed + offset)
        write_jsonl(records, output_dir / f"{split_name.replace('-', '_')}.jsonl")
        sizes[split_name] = len(records)
        logger.info("%s: %d Yes + %d No", split_name, n, n)
    return sizes


def build_fixed_vocab_split(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    max_per_output: int,
    max_total: int,
    audio_prefix: str,
    seed: int,
) -> dict[str, int]:
    """Build a stratified fixed-vocabulary call-type split.

    Returns
    -------
    dict[str, int]
        Generated row count for the fixed-vocabulary split.
    """

    behavior_df = df[clean_behavior_mask(df["behavior"])].copy()
    behavior_df["fixed_vocab_labels"] = behavior_df["behavior"].map(fixed_vocab_labels)
    behavior_df = behavior_df[behavior_df["fixed_vocab_labels"].map(bool)].copy()
    behavior_df["output"] = behavior_df["fixed_vocab_labels"].map(lambda labels: ", ".join(labels))

    groups = {
        output: sample_rows(group, max_per_output, seed + offset)
        for offset, (output, group) in enumerate(sorted(behavior_df.groupby("output")))
    }
    group_counts = {output: len(group) for output, group in groups.items()}
    sample_counts = proportional_sample_counts(group_counts, max_total)

    records = []
    output_counts: dict[str, int] = {}
    for offset, (output, group) in enumerate(sorted(groups.items())):
        sampled = sample_rows(group, sample_counts[output], seed + 1000 + offset)
        output_counts[output] = len(sampled)
        records.extend(
            make_row(
                split_name="call-type-fixed-vocab",
                output=output,
                instruction_text=FIXED_VOCAB_INSTRUCTION,
                task="call_type_fixed_vocab",
                row=row,
                audio_prefix=audio_prefix,
                extra_metadata={
                    "behavior_fixed_vocab": output,
                    "behavior_fixed_vocab_count": len(row["fixed_vocab_labels"]),
                },
            )
            for _, row in sampled.iterrows()
        )

    records = shuffle_records(records, seed)
    write_jsonl(records, output_dir / "call_type_fixed_vocab.jsonl")
    logger.info("call-type-fixed-vocab output counts: %s", output_counts)
    return {"call-type-fixed-vocab": len(records)}


def write_manifest(output_dir: Path, sizes: dict[str, int], args: argparse.Namespace) -> None:
    """Write a small manifest describing the generated local dataset."""

    manifest = {
        "source_csv": args.source_csv,
        "audio_root": "gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/",
        "audio_prefix": args.audio_prefix,
        "seed": args.seed,
        "max_taxon_per_label": args.max_taxon_per_label,
        "max_call_type_per_label": args.max_call_type_per_label,
        "max_fixed_vocab_per_output": args.max_fixed_vocab_per_output,
        "max_fixed_vocab_total": args.max_fixed_vocab_total,
        "splits": sizes,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote manifest to %s", path)


def main() -> None:
    """Build all XC-new-only evaluation JSONLs."""

    args = parse_args()
    source = quality_filter(load_source_csv(args.source_csv))

    sizes = {}
    sizes.update(
        build_taxon_presence_splits(
            source,
            args.output_dir,
            max_per_label=args.max_taxon_per_label,
            audio_prefix=args.audio_prefix,
            seed=args.seed,
        )
    )
    sizes.update(
        build_call_type_binary_splits(
            source,
            args.output_dir,
            max_per_label=args.max_call_type_per_label,
            audio_prefix=args.audio_prefix,
            seed=args.seed,
        )
    )
    sizes.update(
        build_fixed_vocab_split(
            source,
            args.output_dir,
            max_per_output=args.max_fixed_vocab_per_output,
            max_total=args.max_fixed_vocab_total,
            audio_prefix=args.audio_prefix,
            seed=args.seed,
        )
    )
    write_manifest(args.output_dir, sizes, args)


if __name__ == "__main__":
    main()
