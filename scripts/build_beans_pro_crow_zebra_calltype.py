#!/usr/bin/env python3
"""Build crow & zebra multi-audio 4-way call-type eval splits for BEANS-Pro.

Reads the existing single-audio description-matching JSONL files for carrion
crow and plains zebra, then generates exactly-aligned multi-audio 4-way
multiple-choice benchmarks: same targets, same confuser call types, same
answer positions — but audio examples replace text descriptions.

Usage::

    uv run python scripts/build_beans_pro_crow_zebra_calltype.py
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import re
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

FOUR_WAY_INSTRUCTION = (
    "Here are four call types.\n\n"
    "A: <Audio><AudioHere></Audio>\n"
    "B: <Audio><AudioHere></Audio>\n"
    "C: <Audio><AudioHere></Audio>\n"
    "D: <Audio><AudioHere></Audio>\n\n"
    "Which call type best matches the following recording?\n"
    "<Audio><AudioHere></Audio>"
)

LABELS = ["A", "B", "C", "D"]

# Source JSONL paths (existing beans_pro description-matching splits)
_SOURCES = {
    "crow": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/carrion_crow_descriptions/test.jsonl",
    "zebra": "gs://esp-data-ingestion/beans-pro/v0.1.0/raw/zebra_descriptions/test.jsonl",
}

# ── Helpers ──────────────────────────────────────────────────────────────


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file from GCS.

    Parameters
    ----------
    path : str
        GCS path to the JSONL file.

    Returns
    -------
    list[dict]
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


def parse_options(instruction_text: str) -> dict[str, str]:
    """Parse the 4 option descriptions from an instruction_text field.

    Parameters
    ----------
    instruction_text : str
        The instruction text containing labelled options A-D.

    Returns
    -------
    dict[str, str]
        Maps label (A-D) to description text.

    Raises
    ------
    ValueError
        If not all 4 options are found.
    """
    # Match "A: <description text>" up to the next option or end-of-options
    pattern = r"([A-D]):\s*(.+?)(?=\n[A-D]:|(?:\n\nAnswer))"
    matches = re.findall(pattern, instruction_text, re.DOTALL)
    options = {label: text.strip() for label, text in matches}
    if len(options) != 4:
        raise ValueError(
            f"Expected 4 options, found {len(options)}: {list(options.keys())}"
        )
    return options


def build_description_to_calltype(
    records: list[dict],
) -> dict[str, str]:
    """Build a description text → call_type mapping from the correct answers.

    Parameters
    ----------
    records : list[dict]
        Beans-pro JSONL records with ``instruction_text``, ``output``, and
        ``metadata`` fields.

    Returns
    -------
    dict[str, str]
        Maps description text to call_type.
    """
    desc_to_ct: dict[str, str] = {}
    for row in records:
        meta = json.loads(row["metadata"])
        call_type = meta["call_type"]
        correct_label = row["output"]
        options = parse_options(row["instruction_text"])
        correct_desc = options[correct_label]
        if correct_desc in desc_to_ct:
            assert desc_to_ct[correct_desc] == call_type, (
                f"Conflicting mapping for description: "
                f"{desc_to_ct[correct_desc]} vs {call_type}"
            )
        desc_to_ct[correct_desc] = call_type
    logger.info("  Mapped %d unique descriptions to call types", len(desc_to_ct))
    return desc_to_ct


def build_calltype_to_paths(records: list[dict]) -> dict[str, list[str]]:
    """Build call_type → list of audio paths.

    Parameters
    ----------
    records : list[dict]
        Beans-pro JSONL records.

    Returns
    -------
    dict[str, list[str]]
        Maps call_type to list of audio paths.
    """
    index: dict[str, list[str]] = {}
    for row in records:
        meta = json.loads(row["metadata"])
        ct = meta["call_type"]
        path = row["audio_path_original_sample_rate"]
        index.setdefault(ct, []).append(path)
    return index


def make_row(
    *,
    split_name: str,
    row_idx: int,
    audio_paths: list[str],
    instruction: str,
    output: str,
    task: str,
    template_path: str,
    source_dataset: str,
    license_str: str,
    metadata: dict,
    original_id: str,
) -> dict:
    """Build a single JSONL row in the beans_pro_multi_audio format.

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
        "source_dataset": source_dataset,
        "dataset_name": split_name,
        "license": license_str,
        "metadata": json.dumps(metadata),
        "audio_path_original_sample_rate": audio_paths[-1],
        "original_beans_pro_id": original_id,
    }


def build_4way(
    records: list[dict],
    desc_to_ct: dict[str, str],
    ct_to_paths: dict[str, list[str]],
    split_name: str,
    source_dataset: str,
    license_str: str,
    rng: random.Random,
) -> list[dict]:
    """Build 4-way multi-audio rows aligned to existing beans_pro rows.

    For each source row, the same target clip and same 4 confuser call types
    are used.  Text descriptions are replaced by audio examples.

    Parameters
    ----------
    records : list[dict]
        Source beans_pro JSONL records.
    desc_to_ct : dict[str, str]
        Description text → call_type mapping.
    ct_to_paths : dict[str, list[str]]
        Call_type → audio paths mapping.
    split_name : str
        Output split name (e.g. ``"crow-4way"``).
    source_dataset : str
        Source dataset attribution string.
    license_str : str
        License string.
    rng : random.Random
        Seeded RNG.

    Returns
    -------
    list[dict]
        Multi-audio JSONL rows.
    """
    rows = []
    skipped = 0

    for i, src in enumerate(records):
        meta = json.loads(src["metadata"])
        correct_ct = meta["call_type"]
        correct_label = src["output"]
        target_path = src["audio_path_original_sample_rate"]

        # Parse all 4 option call types from the original row
        options = parse_options(src["instruction_text"])
        option_types = {}
        for label in LABELS:
            desc = options[label]
            if desc not in desc_to_ct:
                logger.warning(
                    "Row %d: description not in lookup for option %s, skipping",
                    i, label,
                )
                break
            option_types[label] = desc_to_ct[desc]
        else:
            # All 4 options resolved — proceed
            pass

        if len(option_types) != 4:
            skipped += 1
            continue

        # Sanity check: the correct label should map to the correct call type
        assert option_types[correct_label] == correct_ct, (
            f"Row {i}: correct label {correct_label} maps to "
            f"{option_types[correct_label]}, expected {correct_ct}"
        )

        # Pick an audio example for each option (not the target clip)
        option_clips = []
        ok = True
        for label in LABELS:
            ct = option_types[label]
            available = [p for p in ct_to_paths[ct] if p != target_path]
            if not available:
                logger.warning(
                    "Row %d: no non-target clips for option %s (%s), skipping",
                    i, label, ct,
                )
                ok = False
                break
            option_clips.append(rng.choice(available))

        if not ok:
            skipped += 1
            continue

        # audio_paths: 4 option clips + 1 target (matches <AudioHere> order)
        audio_paths = option_clips + [target_path]

        rows.append(make_row(
            split_name=split_name,
            row_idx=len(rows),
            audio_paths=audio_paths,
            instruction=FOUR_WAY_INSTRUCTION,
            output=correct_label,
            task="call_type_multiple_choice",
            template_path="audio_synth/multiple_choice",
            source_dataset=source_dataset,
            license_str=license_str,
            metadata={
                "option_types": option_types,
                "correct": correct_label,
                "correct_type": correct_ct,
                "species": meta.get("species", ""),
                "species_common": meta.get("species_common", ""),
            },
            original_id=src["id"],
        ))

    if skipped:
        logger.warning("Skipped %d rows for %s", skipped, split_name)

    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write rows to a JSONL file.

    Parameters
    ----------
    rows : list[dict]
        Records to write.
    path : Path
        Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), path)


# ── Main ─────────────────────────────────────────────────────────────────


def build_species(
    species_key: str,
    split_name: str,
    source_dataset: str,
    license_str: str,
    output_dir: Path,
    rng: random.Random,
) -> list[dict]:
    """Build the 4-way split for one species.

    Parameters
    ----------
    species_key : str
        Key into ``_SOURCES`` (``"crow"`` or ``"zebra"``).
    split_name : str
        Output split name.
    source_dataset : str
        Attribution string.
    license_str : str
        License string.
    output_dir : Path
        Directory for the output JSONL.
    rng : random.Random
        Seeded RNG.

    Returns
    -------
    list[dict]
        Generated rows.
    """
    logger.info("=== Building %s ===", split_name)

    records = load_jsonl(_SOURCES[species_key])

    # Step 1: description → call_type mapping
    desc_to_ct = build_description_to_calltype(records)

    # Step 2: call_type → audio paths
    ct_to_paths = build_calltype_to_paths(records)
    for ct, paths in sorted(ct_to_paths.items(), key=lambda x: -len(x[1])):
        logger.info("  %s: %d clips", ct, len(paths))

    # Step 3: build aligned 4-way rows
    rows = build_4way(
        records, desc_to_ct, ct_to_paths,
        split_name=split_name,
        source_dataset=source_dataset,
        license_str=license_str,
        rng=rng,
    )

    # Stats
    answer_dist = collections.Counter(r["messages"][1]["content"] for r in rows)
    logger.info("%s: %d rows, answers: %s", split_name, len(rows), dict(answer_dist))

    # Write
    fname = f"{split_name.replace('-', '_')}.jsonl"
    write_jsonl(rows, output_dir / fname)

    return rows


def main() -> None:
    """Generate crow and zebra 4-way call-type eval JSONL splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "beans_pro_crow_zebra_calltype",
    )
    args = parser.parse_args()

    rng = random.Random(SEED)

    crow_rows = build_species(
        species_key="crow",
        split_name="crow-4way",
        source_dataset="10.64898/2026.04.02.715916",
        license_str="CC-BY-NC-4.0",
        output_dir=args.output_dir,
        rng=rng,
    )

    zebra_rows = build_species(
        species_key="zebra",
        split_name="zebra-4way",
        source_dataset="10.5061/dryad.v9s4mw73w",
        license_str="CC0-1.0",
        output_dir=args.output_dir,
        rng=rng,
    )

    # Summary
    all_audio = set()
    for row in crow_rows + zebra_rows:
        all_audio.update(row["audio_paths"])
    logger.info("Total unique audio files referenced: %d", len(all_audio))
    logger.info("Done!")


if __name__ == "__main__":
    main()
