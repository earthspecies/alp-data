#!/usr/bin/env python3
"""Build species-conditioned variants of the weldy-call-or-song splits.

For each row in an existing weldy call-or-song JSONL (2-s or 6-s), emit a
new JSONL row whose ``instruction`` includes the focal species name —
turning the species-agnostic binary call/song question into a species-
conditioned one. The audio is reused verbatim: the new split's data_root
points at the existing split's GCS audio folder, and ``audio_path_original_sample_rate``
in each row carries the same relative path.

Prompt pattern (extends the trained ``alarm_call_presence_with_species``
shape to call vs song)::

    "Is the {species_common} making a call or a song in this recording?"

Falls back to the scientific (canonical) name when ``species_common`` is
empty.

Usage::

    uv run python scripts/build_beans_pro_weldy_call_or_song_species_conditioned.py \\
        --source-split weldy_call_or_song \\
        --output-dir data/beans_pro_weldy_call_or_song_sp

    uv run python scripts/build_beans_pro_weldy_call_or_song_species_conditioned.py \\
        --source-split weldy_call_or_song_6s \\
        --output-dir data/beans_pro_weldy_call_or_song_sp_6s
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

import fsspec

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


BEANS_PRO_RAW = "gs://esp-data-ingestion/beans-pro/v0.1.0/raw"

# Map source-split → (canonical dataset_name, BEANS-Pro split slug we're creating)
DEFAULT_SUFFIXES = {
    "weldy_call_or_song": "weldy-call-or-song-sp",
    "weldy_call_or_song_6s": "weldy-call-or-song-sp-6s",
}


def _species_prompt(species_common: str, species_scientific: str) -> str:
    """Build the species-conditioned prompt body (no audio tag)."""
    name = species_common.strip() if species_common else species_scientific.strip()
    if not name:
        name = "focal species"
    return f"Is the {name} making a call or a song in this recording?"


def _read_jsonl_gcs(uri: str) -> list[dict]:
    fs = fsspec.filesystem("gs")
    _, stripped = uri.split("://", 1)
    with fs.open(stripped, "r") as fh:
        return [json.loads(line) for line in fh]


def build(*, source_split: str, output_dir: Path, dataset_name_override: str | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    src_jsonl = f"{BEANS_PRO_RAW}/{source_split}/test.jsonl"
    logger.info("Reading source JSONL: %s", src_jsonl)
    rows = _read_jsonl_gcs(src_jsonl)
    logger.info("Source rows: %d", len(rows))

    dataset_name = dataset_name_override or DEFAULT_SUFFIXES.get(source_split)
    if dataset_name is None:
        raise ValueError(
            f"Unknown source_split={source_split!r}; pass --dataset-name explicitly."
        )

    out: list[dict] = []
    n_common = 0
    n_fallback_scientific = 0
    n_no_name = 0
    for r in rows:
        meta = json.loads(r["metadata"])
        sp_common = str(meta.get("species_common") or "").strip()
        sp_sci = str(meta.get("species") or "").strip()
        if sp_common:
            n_common += 1
        elif sp_sci:
            n_fallback_scientific += 1
        else:
            n_no_name += 1
        prompt = _species_prompt(sp_common, sp_sci)
        new_row = {
            "source_dataset": r.get("source_dataset", "weldy_dawn_chorus"),
            "dataset_name": dataset_name,
            "output": r["output"],
            "instruction_text": prompt,
            "instruction": f"<Audio><AudioHere></Audio> {prompt}",
            "task": "call_or_song_with_species_classification",
            "file_name": r["file_name"],
            "license": r.get("license", "CC-BY-4.0"),
            "id": str(uuid.uuid4()),
            "metadata": json.dumps({**meta, "source_split": source_split}),
            # Reuse the same audio: the BeansPro split's data_root will point
            # at the source-split's audio folder.
            "audio_path_original_sample_rate": r["audio_path_original_sample_rate"],
        }
        out.append(new_row)

    jsonl_path = output_dir / "test.jsonl"
    with open(jsonl_path, "w") as fh:
        for row in out:
            fh.write(json.dumps(row) + "\n")
    logger.info("Wrote %d rows to %s", len(out), jsonl_path)
    logger.info(
        "Prompt species source: %d common name / %d scientific fallback / %d no-name fallback",
        n_common, n_fallback_scientific, n_no_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-split", required=True,
                        choices=sorted(DEFAULT_SUFFIXES) + ["custom"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default=None,
                        help="Override the emitted BeansPro split slug.")
    args = parser.parse_args()
    build(
        source_split=args.source_split,
        output_dir=args.output_dir,
        dataset_name_override=args.dataset_name,
    )


if __name__ == "__main__":
    main()
