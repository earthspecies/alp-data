"""Build the MammalNet behavior-classification manifests.

Given a locally-extracted MammalNet release (``--src`` containing the trimmed
clips and the annotation files), this script builds per-split manifest CSVs
with absolute ``gs://`` ``video_path`` columns for the **behavior recognition**
task (12 high-level behaviors).

Downloads (public S3, done by ``jobs/build_mammalnet.sh``):
- ``https://mammalnet.s3.amazonaws.com/trimmed_videos.tar.gz``
- ``https://mammalnet.s3.amazonaws.com/annotation.tar``

Annotation layout (from the public ``annotation.tar``): ``<src>/annotation/``
holds ``behavior_to_id.txt`` (``behavior_name<TAB>id``, the 12 behaviors) and
``composition/{train,val,test}.csv`` — headerless, whitespace-separated rows
``<clip_path> <animal_id> <behavior_id>`` where ``clip_path`` is relative to
``<src>`` (e.g. ``trimmed_videos/OT88wS6FLoQ.mp4``). The behavior label is the
third column, mapped to a name via ``behavior_to_id.txt``.

Usage (see jobs/build_mammalnet.sh):
    uv run python scripts/data_preprocessing_scripts/mammalnet/build_mammalnet.py \
        --src /scratch/$USER/mammalnet --out /scratch/$USER/mammalnet/staging \
        --splits test
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

GCS_ROOT_DEFAULT = "gs://esp-data-ingestion/mammalnet/v0.1.0"

_OUT_COLUMNS = ["asset_id", "modality", "label", "behavior", "split", "video_path"]


def load_behavior_map(annot_dir: Path) -> dict[str, str]:
    """Load ``behavior_to_id.txt`` as an id -> behavior-name map.

    Parameters
    ----------
    annot_dir : Path
        The MammalNet ``annotation`` directory.

    Returns
    -------
    dict[str, str]
        Mapping from string behavior id to behavior name (12 behaviors).

    Raises
    ------
    FileNotFoundError
        If ``behavior_to_id.txt`` is not found.
    """
    fp = annot_dir / "behavior_to_id.txt"
    if not fp.exists():
        raise FileNotFoundError(f"behavior_to_id.txt not found under {annot_dir}")
    id_to_name: dict[str, str] = {}
    with open(fp) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                name, idx = parts
                id_to_name[idx.strip()] = name.strip()
    return id_to_name


def load_behavior_split(
    annot_dir: Path, split: str, id_to_name: dict[str, str]
) -> list[tuple[str, str]]:
    """Parse ``composition/{split}.csv`` into ``(clip_ref, behavior)`` rows.

    The file is headerless and whitespace-separated:
    ``<clip_path> <behavior_id> <animal_id>``. The behavior label is the second
    column (0–11), mapped to a name via ``id_to_name``; the third column is the
    genus id (0–171) and is ignored here.

    Parameters
    ----------
    annot_dir : Path
        The MammalNet ``annotation`` directory.
    split : str
        ``train`` / ``val`` / ``test``.
    id_to_name : dict[str, str]
        Behavior id -> name map.

    Returns
    -------
    list[tuple[str, str]]
        ``(clip_ref, behavior_name)`` pairs.

    Raises
    ------
    FileNotFoundError
        If ``composition/{split}.csv`` is not found.
    RuntimeError
        If a row has an unknown behavior id or the file parses to 0 rows.
    """
    fp = annot_dir / "composition" / f"{split}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"{fp} not found (expected MammalNet composition split).")
    rows: list[tuple[str, str]] = []
    with open(fp) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            clip_ref, behavior_id = parts[0], parts[1]
            if behavior_id not in id_to_name:
                raise RuntimeError(f"{fp}: unknown behavior id {behavior_id!r} (map={id_to_name}).")
            rows.append((clip_ref, id_to_name[behavior_id]))
    if not rows:
        raise RuntimeError(f"{fp}: parsed 0 rows; verify the file format.")
    return rows


def _clip_asset_id(clip_ref: str) -> str:
    """Return a stable asset id (basename without extension) for a clip ref.

    Returns
    -------
    str
        The clip basename without its extension.
    """
    return Path(str(clip_ref)).stem


def build_split(
    src: Path,
    annot_dir: Path,
    split: str,
    label_of: dict[str, int],
    id_to_name: dict[str, str],
    gcs_root: str,
) -> pd.DataFrame:
    """Build the manifest rows for one split.

    Parameters
    ----------
    src : Path
        Extracted MammalNet root (clip paths are relative to this).
    annot_dir : Path
        The MammalNet ``annotation`` directory.
    split : str
        ``train`` / ``val`` / ``test``.
    label_of : dict[str, int]
        Behavior name -> label index (the official ``behavior_to_id.txt`` id).
    id_to_name : dict[str, str]
        Behavior id -> name map.
    gcs_root : str
        GCS root for absolute ``video_path`` columns.

    Returns
    -------
    pd.DataFrame
        The manifest for the split.

    Raises
    ------
    RuntimeError
        If any referenced clip file is missing under ``src``.
    """
    pairs = load_behavior_split(annot_dir, split, id_to_name)
    rows = []
    missing = 0
    for clip_ref, behavior in pairs:
        aid = _clip_asset_id(clip_ref)
        # clip_ref is relative to src (e.g. "trimmed_videos/<id>.mp4").
        candidates = [src / clip_ref, src / "trimmed_videos" / f"{aid}.mp4"]
        local = next((c for c in candidates if c.exists()), None)
        if local is None:
            missing += 1
            continue
        rows.append(
            {
                "asset_id": aid,
                "modality": "video",
                "label": label_of[behavior],
                "behavior": behavior,
                "split": split,
                "video_path": f"{gcs_root}/video/{aid}.mp4",
            }
        )
    if missing:
        raise RuntimeError(
            f"{split}: {missing} clips referenced by the annotation are missing under "
            f"{src}; refusing to ship a manifest with missing files."
        )
    return pd.DataFrame(rows, columns=_OUT_COLUMNS)


def main() -> None:
    """Run the MammalNet behavior manifest build."""
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Extracted MammalNet root dir.")
    p.add_argument("--out", required=True, help="Staging output dir.")
    p.add_argument("--gcs-root", default=GCS_ROOT_DEFAULT)
    p.add_argument("--splits", nargs="+", default=["test"],
                   help="Splits to build/stage (default: test only, to bound size).")
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    annot_dir = src / "annotation"
    if not annot_dir.is_dir():
        annot_dir = src  # some releases untar annotation files flat

    print("=== 1. load behavior label space (behavior_to_id.txt) ===", flush=True)
    id_to_name = load_behavior_map(annot_dir)
    # Label index == the official behavior id (0–11), so labels stay canonical.
    label_of = {name: int(idx) for idx, name in id_to_name.items()}
    print(f"{len(label_of)} behaviors: {sorted(label_of)}")

    print("\n=== 2. build manifests ===", flush=True)
    frames = []
    for split in args.splits:
        df = build_split(src, annot_dir, split, label_of, id_to_name, args.gcs_root)
        df.to_csv(out / f"mammalnet_{split}.csv", index=False)
        print(f"mammalnet_{split}.csv: {len(df)} clips")
        frames.append(df)
    alldf = pd.concat(frames, ignore_index=True)
    alldf.to_csv(out / "mammalnet_all.csv", index=False)
    print(f"mammalnet_all.csv: {len(alldf)} clips")

    # Emit a manifest of clip basenames to upload (so the job stages only what
    # the manifests reference, not the whole 539 h corpus).
    with open(out / "upload_clip_ids.txt", "w", newline="") as f:
        w = csv.writer(f)
        for aid in sorted(set(alldf["asset_id"])):
            w.writerow([aid])
    print("\nDONE.")


if __name__ == "__main__":
    main()
