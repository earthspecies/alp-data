"""Build the NeWT (Natural World Tasks) image-classification manifests.

Given a locally-extracted NeWT release (``--src`` containing
``newt2021_images/`` and ``newt2021_labels.csv``), this script builds the
per-split + unified manifest CSVs with absolute ``gs://`` image-path
columns pointing at the eventual GCS layout.

All 164 binary tasks are kept (including the ``species`` sub-cluster) so the
resulting benchmark is directly comparable to the NeWT numbers reported by
BioCLIP 2. The task metadata (``task`` / ``task_cluster`` /
``task_subcluster``) is carried through so an evaluator can break results
down per cluster.

The actual image upload + manifest upload is done by the
``jobs/build_newt.sh`` wrapper via ``gsutil -m rsync`` after this script
finishes — keeping the large byte movement out of Python.

Usage (see jobs/build_newt.sh):
    uv run python scripts/data_preprocessing_scripts/newt/build_newt.py \
        --src /scratch/$USER/newt \
        --out /scratch/$USER/newt/staging
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

GCS_ROOT_DEFAULT = "gs://esp-data-ingestion/newt/v0.1.0"

# NeWT labels CSV columns (visipedia/newt, newt2021_labels.csv).
_SRC_COLUMNS = [
    "id",
    "task_cluster",
    "task_subcluster",
    "task",
    "label",
    "text_label",
    "split",
    "height",
    "width",
]

# Manifest columns (esp-data NeWT dataset).
_OUT_COLUMNS = [
    "asset_id",
    "modality",
    "label",
    "text_label",
    "task",
    "task_cluster",
    "task_subcluster",
    "split",
    "image_path",
]


def build_manifest(src: Path, gcs_root: str) -> pd.DataFrame:
    """Build the unified NeWT image manifest.

    Parameters
    ----------
    src : Path
        Extracted NeWT root (containing ``newt2021_labels.csv`` and
        ``newt2021_images/``).
    gcs_root : str
        GCS root for absolute ``image_path`` columns.

    Returns
    -------
    pd.DataFrame
        The manifest with one row per image, columns :data:`_OUT_COLUMNS`.

    Raises
    ------
    FileNotFoundError
        If ``newt2021_labels.csv`` is not found under ``src``.
    RuntimeError
        If the labels CSV is missing expected columns, or if any referenced
        image file is absent (a manifest must never point at a missing file).
    """
    labels_fp = src / "newt2021_labels.csv"
    if not labels_fp.exists():
        raise FileNotFoundError(f"NeWT labels CSV not found at {labels_fp}")

    df = pd.read_csv(labels_fp, keep_default_na=False, na_values=[])
    missing_cols = [c for c in _SRC_COLUMNS if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"NeWT labels CSV missing columns: {missing_cols}")

    images_dir = src / "newt2021_images"
    rows = []
    missing_imgs = 0
    for _, r in df.iterrows():
        aid = str(r["id"])
        if not (images_dir / f"{aid}.jpg").exists():
            missing_imgs += 1
            continue
        rows.append(
            {
                "asset_id": aid,
                "modality": "image",
                "label": int(r["label"]),
                "text_label": r["text_label"],
                "task": r["task"],
                "task_cluster": r["task_cluster"],
                "task_subcluster": r["task_subcluster"],
                "split": str(r["split"]),
                "image_path": f"{gcs_root}/images/{aid}.jpg",
            }
        )
    if missing_imgs:
        raise RuntimeError(
            f"{missing_imgs} NeWT rows reference an image file absent from "
            f"{images_dir}; refusing to ship a manifest with missing files."
        )
    return pd.DataFrame(rows, columns=_OUT_COLUMNS)


def write_splits(df: pd.DataFrame, out: Path) -> None:
    """Write ``newt_all`` + per-split manifest CSVs.

    Parameters
    ----------
    df : pd.DataFrame
        The unified manifest (with a ``split`` column).
    out : Path
        Staging output directory.
    """
    df.to_csv(out / "newt_all.csv", index=False)
    n_tasks = df["task"].nunique()
    print(f"newt_all.csv: {len(df)} rows, {n_tasks} tasks")
    for split in sorted(df["split"].unique()):
        sub = df[df["split"] == split]
        sub.to_csv(out / f"newt_{split}.csv", index=False)
        print(f"newt_{split}.csv: {len(sub)} rows")


def main() -> None:
    """Run the NeWT manifest build."""
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Extracted NeWT root dir.")
    p.add_argument("--out", required=True, help="Staging output dir.")
    p.add_argument("--gcs-root", default=GCS_ROOT_DEFAULT)
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=== build NeWT manifest ===", flush=True)
    manifest = build_manifest(src, args.gcs_root)

    # Report the cluster / sub-cluster breakdown so scope is auditable.
    print("\n=== task breakdown (cluster / subcluster: #tasks, #images) ===")
    grp = manifest.groupby(["task_cluster", "task_subcluster"])
    for (cluster, sub), g in grp:
        print(f"  {cluster} / {sub or '-'}: {g['task'].nunique()} tasks, {len(g)} images")

    write_splits(manifest, out)
    print("\nDONE.")


if __name__ == "__main__":
    main()
