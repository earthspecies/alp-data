#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Extract balanced 5-second Flying / Not Flying clips from the test split.

This script reads `test_info.csv` from
`gs://spanish-carrion-crows/flying_with_annotations/`, generates all
non-overlapping clip candidates from `Flying` and `Not Flying` regions, selects
a balanced subset with a preference for source-file diversity, downloads the
audio from GCS, and writes a manifest CSV.
"""

from __future__ import annotations

import argparse
import io
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import soundfile as sf

from esp_data.io.read_utils import read_audio, read_text

GCS_ROOT = "gs://spanish-carrion-crows/flying_with_annotations"
TEST_INFO_CSV = f"{GCS_ROOT}/test_info.csv"

RAW_TO_OUTPUT_LABEL = {
    "Flying": "flying",
    "Not Flying": "not_flying",
}


@dataclass(frozen=True)
class ClipCandidate:
    """A single fixed-duration clip candidate from one annotated region."""

    label: str
    source_fn: str
    audio_gcs_path: str
    selection_table_gcs_path: str
    clip_start_s: float
    clip_end_s: float
    segment_start_s: float
    segment_end_s: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/spanish_carrion_crows_flight_test_clips_5s_100",
        help="Directory to write extracted audio and manifest.",
    )
    parser.add_argument(
        "--clip-duration-s",
        type=float,
        default=5.0,
        help="Fixed clip duration in seconds.",
    )
    parser.add_argument(
        "--num-per-class",
        type=int,
        default=50,
        help="Number of clips to extract for each class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for deterministic sampling.",
    )
    return parser.parse_args()


def to_gcs_path(raw_path: str) -> str:
    """Convert the notebook-era local path in the CSVs to the matching GCS path."""
    marker = "flying_with_annotations/"
    if marker not in raw_path:
        raise ValueError(f"Unexpected dataset path: {raw_path}")
    suffix = raw_path.split(marker, maxsplit=1)[1]
    return f"{GCS_ROOT}/{suffix}"


def load_test_split() -> pd.DataFrame:
    """Load the held-out test split metadata."""
    return pd.read_csv(TEST_INFO_CSV, dtype={"fn": "string"})


def load_selection_table(selection_table_gcs_path: str) -> pd.DataFrame:
    """Load one selection table from GCS."""
    raw_text = read_text(selection_table_gcs_path)
    return pd.read_csv(io.StringIO(raw_text), sep="\t")


def build_candidates(test_df: pd.DataFrame, clip_duration_s: float) -> list[ClipCandidate]:
    """Enumerate all non-overlapping fixed-length clip candidates."""
    candidates: list[ClipCandidate] = []
    for row in test_df.to_dict(orient="records"):
        source_fn = str(row["fn"])
        audio_gcs_path = to_gcs_path(str(row["audio_fp"]))
        selection_table_gcs_path = to_gcs_path(str(row["selection_table_fp"]))
        selection_table = load_selection_table(selection_table_gcs_path)

        for segment in selection_table.to_dict(orient="records"):
            raw_label = str(segment["Annotation"])
            if raw_label not in RAW_TO_OUTPUT_LABEL:
                continue

            segment_start_s = float(segment["Begin Time (s)"])
            segment_end_s = float(segment["End Time (s)"])
            segment_duration_s = segment_end_s - segment_start_s
            windows_in_segment = int(segment_duration_s // clip_duration_s)
            if windows_in_segment <= 0:
                continue

            for window_idx in range(windows_in_segment):
                clip_start_s = segment_start_s + (window_idx * clip_duration_s)
                clip_end_s = clip_start_s + clip_duration_s
                candidates.append(
                    ClipCandidate(
                        label=RAW_TO_OUTPUT_LABEL[raw_label],
                        source_fn=source_fn,
                        audio_gcs_path=audio_gcs_path,
                        selection_table_gcs_path=selection_table_gcs_path,
                        clip_start_s=clip_start_s,
                        clip_end_s=clip_end_s,
                        segment_start_s=segment_start_s,
                        segment_end_s=segment_end_s,
                    )
                )

    return candidates


def select_balanced_candidates(
    candidates: list[ClipCandidate],
    num_per_class: int,
    seed: int,
) -> list[ClipCandidate]:
    """Select a balanced subset while spreading picks across source files."""
    rng = random.Random(seed)
    selected: list[ClipCandidate] = []

    by_label: dict[str, dict[str, list[ClipCandidate]]] = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        by_label[candidate.label][candidate.source_fn].append(candidate)

    for label in sorted(by_label):
        file_to_candidates = by_label[label]
        for file_candidates in file_to_candidates.values():
            rng.shuffle(file_candidates)

        file_order = list(file_to_candidates)
        rng.shuffle(file_order)
        label_selected: list[ClipCandidate] = []

        while len(label_selected) < num_per_class:
            made_progress = False
            for source_fn in file_order:
                remaining = file_to_candidates[source_fn]
                if not remaining:
                    continue
                label_selected.append(remaining.pop())
                made_progress = True
                if len(label_selected) == num_per_class:
                    break

            if not made_progress:
                available = sum(len(items) for items in file_to_candidates.values())
                raise ValueError(
                    f"Not enough {label} clips. Requested {num_per_class}, "
                    f"but only found {len(label_selected) + available} candidates."
                )

        selected.extend(label_selected)

    rng.shuffle(selected)
    return selected


def clip_id_for(candidate: ClipCandidate) -> str:
    """Build a stable clip identifier."""
    start_ms = int(round(candidate.clip_start_s * 1000))
    end_ms = int(round(candidate.clip_end_s * 1000))
    return f"{candidate.label}__{candidate.source_fn}__{start_ms:06d}_{end_ms:06d}"


def write_outputs(
    selected: list[ClipCandidate],
    output_dir: Path,
) -> pd.DataFrame:
    """Download audio clips and write the manifest."""
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    for candidate in selected:
        clip_id = clip_id_for(candidate)
        label_dir = audio_dir / candidate.label
        label_dir.mkdir(parents=True, exist_ok=True)
        audio_path = label_dir / f"{clip_id}.wav"

        audio, sample_rate = read_audio(
            candidate.audio_gcs_path,
            start_time=candidate.clip_start_s,
            end_time=candidate.clip_end_s,
        )
        sf.write(audio_path, audio, sample_rate, format="WAV", subtype="PCM_16")

        manifest_rows.append(
            {
                "clip_id": clip_id,
                "label": candidate.label,
                "audio_path": str(audio_path.resolve()),
                "audio_relpath": str(audio_path.relative_to(output_dir)),
                "source_fn": candidate.source_fn,
                "source_audio_gcs_path": candidate.audio_gcs_path,
                "selection_table_gcs_path": candidate.selection_table_gcs_path,
                "clip_start_s": candidate.clip_start_s,
                "clip_end_s": candidate.clip_end_s,
                "segment_start_s": candidate.segment_start_s,
                "segment_end_s": candidate.segment_end_s,
            }
        )

    manifest = pd.DataFrame(manifest_rows).sort_values(["label", "source_fn", "clip_start_s"])
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    return manifest


def main() -> None:
    """Run extraction."""
    args = parse_args()
    output_dir = Path(args.output_dir)

    print(f"Loading test split from {TEST_INFO_CSV}")
    test_df = load_test_split()
    print(f"Test files: {len(test_df)}")

    print("Building candidate windows...")
    candidates = build_candidates(test_df, clip_duration_s=args.clip_duration_s)
    counts = pd.Series([candidate.label for candidate in candidates]).value_counts().to_dict()
    print(f"Candidate counts: {counts}")

    selected = select_balanced_candidates(
        candidates=candidates,
        num_per_class=args.num_per_class,
        seed=args.seed,
    )
    print(f"Selected {len(selected)} total clips")

    manifest = write_outputs(selected, output_dir=output_dir)
    print(f"Wrote manifest to {output_dir / 'manifest.csv'}")
    print("Final label counts:")
    print(manifest["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
