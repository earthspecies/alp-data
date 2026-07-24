"""Build the IDLE-OO Camera Traps manifests + extracted images.

Downloads the HuggingFace dataset ``imageomics/IDLE-OO-Camera-Traps``
(parquet with embedded image bytes), writes each image out as a JPEG, and
builds the unified + per-split manifest CSVs with absolute ``gs://``
image-path columns.

The source already carries ``scientific_name`` + a full taxonomy, so the
canonical name space is taken directly from ``scientific_name`` (no GBIF
crosswalk). Labels are 0-based indices assigned by sorted canonical name.

Image upload + manifest upload is done by ``jobs/build_idle_oo.sh`` via
``gsutil -m rsync`` after this script finishes.

Usage (see jobs/build_idle_oo.sh):
    uv run python scripts/data_preprocessing_scripts/idle_oo_camera_traps/build_idle_oo.py \
        --out /scratch/$USER/idle_oo/staging
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd
from PIL import Image

GCS_ROOT_DEFAULT = "gs://esp-data-ingestion/idle_oo_camera_traps/v0.1.0"
REPO_ID = "imageomics/IDLE-OO-Camera-Traps"
# The repo main branch stores raw PNGs; the embedded-image parquet (with
# scientific_name + taxonomy, as served to the dataset viewer) lives on HF's
# auto-converted `refs/convert/parquet` branch, listed by the parquet API.
_HF_PARQUET_API = f"https://huggingface.co/api/datasets/{REPO_ID}/parquet"
TAXONOMY_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

_OUT_COLUMNS = [
    "asset_id",
    "modality",
    "label",
    "canonical_name",
    "species_common",
    "original_label",
    *TAXONOMY_RANKS,
    "split",
    "image_path",
]


def _image_bytes(value: object) -> bytes:
    """Extract raw image bytes from a HuggingFace parquet image cell.

    Parameters
    ----------
    value : object
        The ``image`` cell — typically a dict with a ``bytes`` key, or raw
        bytes.

    Returns
    -------
    bytes
        The encoded image bytes.

    Raises
    ------
    ValueError
        If image bytes cannot be extracted from the cell.
    """
    if isinstance(value, dict) and value.get("bytes") is not None:
        return value["bytes"]
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise ValueError(f"Unrecognised image cell type: {type(value)!r}")


def load_dataframe(cache_dir: str) -> pd.DataFrame:
    """Download the dataset parquet shard(s) from the HF Hub via HTTPS.

    Uses the public HuggingFace dataset API + resolve URLs (stdlib only, no
    ``huggingface_hub`` dependency).

    Parameters
    ----------
    cache_dir : str
        Local directory to download the parquet shard(s) into.

    Returns
    -------
    pd.DataFrame
        The concatenated dataset rows.

    Raises
    ------
    FileNotFoundError
        If the dataset lists no parquet files.
    """
    with urllib.request.urlopen(_HF_PARQUET_API, timeout=60) as resp:
        meta = json.loads(resp.read().decode())
    urls = meta.get("default", {}).get("test", [])
    if not urls:
        raise FileNotFoundError(f"No parquet files listed for dataset {REPO_ID}: {meta}")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, url in enumerate(urls):
        local = cache / f"test_{i:04d}.parquet"
        if not local.exists():
            print(f"  downloading shard {i} ...", flush=True)
            urllib.request.urlretrieve(url, local)
        frames.append(pd.read_parquet(local))
    print(f"Loaded {len(urls)} parquet shard(s)")
    return pd.concat(frames, ignore_index=True)


def build(df: pd.DataFrame, out: Path, gcs_root: str) -> pd.DataFrame:
    """Write images out and build the IDLE-OO manifest.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset rows (with an ``image`` column of embedded bytes).
    out : Path
        Staging output directory (an ``images/`` subdir is created here).
    gcs_root : str
        GCS root for absolute ``image_path`` columns.

    Returns
    -------
    pd.DataFrame
        The manifest, columns :data:`_OUT_COLUMNS`.
    """
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Drop rows without a species-level label (the source includes ~150 images
    # annotated only to a coarse level, e.g. original_label "bird", whose
    # scientific_name is null/None). A species-classification benchmark can only
    # score the 119 species-labelled images, matching BioCLIP 2's usage.
    canon_all = df["scientific_name"].astype(str).str.strip()
    keep = ~canon_all.str.lower().isin(["", "none", "nan", "<na>"])
    dropped = int((~keep).sum())
    if dropped:
        print(f"dropping {dropped} rows with no species-level scientific_name")
    df = df[keep].reset_index(drop=True)

    # 0-based label per species, assigned by sorted canonical (scientific) name.
    canon = df["scientific_name"].astype(str).str.strip().str.capitalize()
    label_of = {name: i for i, name in enumerate(sorted(canon.unique()))}

    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        aid = f"idleoo_{i:06d}"
        img = Image.open(io.BytesIO(_image_bytes(r["image"]))).convert("RGB")
        img.save(images_dir / f"{aid}.jpg", "JPEG", quality=95)
        canonical = str(r["scientific_name"]).strip().capitalize()
        row = {
            "asset_id": aid,
            "modality": "image",
            "label": label_of[canonical],
            "canonical_name": canonical,
            "species_common": r.get("common_name", ""),
            "original_label": r.get("original_label", ""),
            "split": "test",
            "image_path": f"{gcs_root}/images/{aid}.jpg",
        }
        # `class` is stored as `cls` in the source parquet.
        row["class"] = r.get("cls", r.get("class", ""))
        for rank in TAXONOMY_RANKS:
            if rank == "class":
                continue
            row[rank] = r.get(rank, "")
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"  wrote {i + 1}/{len(df)} images", flush=True)
    return pd.DataFrame(rows, columns=_OUT_COLUMNS)


def main() -> None:
    """Run the IDLE-OO Camera Traps manifest + image build."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Staging output dir.")
    p.add_argument("--gcs-root", default=GCS_ROOT_DEFAULT)
    p.add_argument("--hf-cache", default=None, help="Parquet download cache (default <out>/hf).")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hf_cache = args.hf_cache or str(out / "hf")

    print("=== 1. download + load parquet ===", flush=True)
    df = load_dataframe(hf_cache)
    print(f"{len(df)} rows, {df['scientific_name'].nunique()} species")

    print("\n=== 2. write images + build manifest ===", flush=True)
    manifest = build(df, out, args.gcs_root)

    manifest.to_csv(out / "idle_oo_all.csv", index=False)
    manifest.to_csv(out / "idle_oo_test.csv", index=False)
    print(
        f"idle_oo_all.csv / idle_oo_test.csv: {len(manifest)} rows, "
        f"{manifest['canonical_name'].nunique()} species"
    )
    print("\nDONE.")


if __name__ == "__main__":
    main()
