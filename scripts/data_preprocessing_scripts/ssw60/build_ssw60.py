"""Build the SSW60 multimodal manifests + resampled audio mirrors.

Given a locally-extracted SSW60 tarball (``--src``), this script:

1. GBIF-links the 60 ``taxa.csv`` species on ``scientific_name`` via
   ``esp_data.discover.gbif_taxonomy.GBIFConverter`` and writes
   ``taxa_gbif.csv``.
2. Resamples each ``audio_ml/<asset_id>.wav`` (22.05 kHz mono) to 16 kHz
   and 32 kHz mirrors under ``<out>/audio_16k`` / ``<out>/audio_32k``
   (parallel, bounded memory — one file per worker).
3. Builds per-modality and unified manifest CSVs with absolute ``gs://``
   path columns pointing at the eventual GCS layout.

The actual media upload (audio / video / images) and manifest upload is
done by the ``jobs/build_ssw60.sh`` wrapper via ``gsutil -m rsync`` after
this script finishes — keeping the large byte movement out of Python.

Usage (see jobs/build_ssw60.sh):
    uv run python scripts/data_preprocessing_scripts/ssw60/build_ssw60.py \
        --src /scratch/$USER/ssw60/ssw60 \
        --out /scratch/$USER/ssw60/staging \
        --workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf

from esp_data.discover.gbif_taxonomy import GBIFConverter

GCS_ROOT_DEFAULT = "gs://esp-data-ingestion/ssw60/v0.1.0"
TAXONOMY_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]
TARGET_SRS = [16000, 32000]

# GBIF-accepted scientific-name corrections for SSW60 taxa whose Clements
# binomial does not resolve directly against the GBIF animals backbone.
# Filled after the first dry-run reports unresolved names.
SCI_NAME_FIX: dict[str, str] = {}


def link_taxa(src: Path, out: Path, gbif_cache: str | None) -> pd.DataFrame:
    """GBIF-link the SSW60 taxa table and write ``taxa_gbif.csv``.

    Parameters
    ----------
    src : Path
        Path to the extracted SSW60 root (containing ``taxa.csv``).
    out : Path
        Staging output directory for ``taxa_gbif.csv``.
    gbif_cache : str or None
        Local cache path for the GBIF animals TSV, or None for the default.

    Returns
    -------
    pd.DataFrame
        The taxa table augmented with GBIF fields, indexed by ``label``.

    Raises
    ------
    RuntimeError
        If any of the 60 taxa fail to resolve against GBIF (so the
        manifest never ships partial taxonomy).
    """
    taxa = pd.read_csv(src / "taxa.csv", keep_default_na=False, na_values=[])
    converter = GBIFConverter(cache_path=gbif_cache) if gbif_cache else GBIFConverter()

    records = []
    unresolved = []
    for _, r in taxa.iterrows():
        sci = " ".join(str(r["scientific_name"]).split())
        rec = {
            "label": int(r["label"]),
            "species_code": r["species_code"],
            "species_common": r["common_name"],
            "scientific_name": sci,
            "family": r["family"],
            "order": r["order"],
            "canonical_name": "",
            "gbifID": "",
            "taxonKey": "",
        }
        for rank in TAXONOMY_RANKS:
            rec[rank] = ""

        info, ok = converter(SCI_NAME_FIX.get(sci, sci))
        if ok:
            rec["canonical_name"] = info["canonicalName"]
            rec["gbifID"] = int(info["taxonID"])
            rec["taxonKey"] = int(info["taxonID"])
            for rank in TAXONOMY_RANKS:
                rec[rank] = info.get(rank, "")
        else:
            unresolved.append((int(r["label"]), sci))
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(out / "taxa_gbif.csv", index=False)
    print(
        f"taxa_gbif.csv: {len(df)} taxa, "
        f"resolved {int((df['canonical_name'] != '').sum())}/{len(df)}"
    )
    if unresolved:
        print("UNRESOLVED (add to SCI_NAME_FIX):")
        for label, sci in unresolved:
            print(f"  {label}\t{sci}")
        raise RuntimeError(
            f"{len(unresolved)} taxa failed GBIF resolution; add SCI_NAME_FIX entries."
        )
    return df.set_index("label", drop=False)


def _resample_one(asset_id: str, src_audio: Path, out: Path) -> str | None:
    """Resample one audio asset to all target rates.

    Parameters
    ----------
    asset_id : str
        Asset identifier (the WAV stem under ``audio_ml/``).
    src_audio : Path
        Directory containing ``<asset_id>.wav``.
    out : Path
        Staging root holding ``audio_16k`` / ``audio_32k`` subdirs.

    Returns
    -------
    str or None
        ``None`` on success, or an error string ``"<asset_id>: <msg>"``.
    """
    try:
        in_fp = src_audio / f"{asset_id}.wav"
        y, sr = sf.read(str(in_fp), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        for target_sr in TARGET_SRS:
            dst = out / f"audio_{target_sr // 1000}k" / f"{asset_id}.wav"
            if dst.exists():
                continue
            if sr != target_sr:
                yr = librosa.resample(
                    y=y,
                    orig_sr=sr,
                    target_sr=target_sr,
                    scale=True,
                    res_type="kaiser_best",
                )
            else:
                yr = y
            sf.write(str(dst), yr, target_sr, subtype="PCM_16")
        return None
    except Exception as e:  # noqa: BLE001  collected and reported by caller
        return f"{asset_id}: {e}"


def resample_audio(src: Path, out: Path, asset_ids: list[str], workers: int) -> None:
    """Resample all audio assets to 16 kHz / 32 kHz mirrors in parallel.

    Parameters
    ----------
    src : Path
        Extracted SSW60 root (containing ``audio_ml/``).
    out : Path
        Staging output directory (``audio_16k`` / ``audio_32k`` created here).
    asset_ids : list[str]
        Audio asset identifiers to resample.
    workers : int
        Number of worker processes.

    Raises
    ------
    RuntimeError
        If any audio file fails to resample.
    """
    for target_sr in TARGET_SRS:
        (out / f"audio_{target_sr // 1000}k").mkdir(parents=True, exist_ok=True)

    fn = partial(_resample_one, src_audio=src / "audio_ml", out=out)
    errors = []
    with mp.Pool(workers) as pool:
        for i, err in enumerate(pool.imap_unordered(fn, asset_ids, chunksize=16), 1):
            if err:
                errors.append(err)
            if i % 500 == 0:
                print(f"  resampled {i}/{len(asset_ids)}", flush=True)
    print(f"resampled {len(asset_ids)} audio assets ({len(errors)} errors)")
    if errors:
        for e in errors[:20]:
            print(f"  ERROR {e}")
        raise RuntimeError(f"{len(errors)} audio files failed to resample.")


def _gbif_cols(taxa: pd.DataFrame, label: int) -> dict[str, object]:
    """Return the GBIF/taxonomy columns for a given 0-based label.

    Parameters
    ----------
    taxa : pd.DataFrame
        GBIF-linked taxa table indexed by ``label``.
    label : int
        SSW60 class label (0-59).

    Returns
    -------
    dict[str, object]
        The species/taxonomy fields for the label.
    """
    row = taxa.loc[label]
    cols = {
        "species_code": row["species_code"],
        "canonical_name": row["canonical_name"],
        "species_common": row["species_common"],
        "family": row["family"],
        "order": row["order"],
        "gbifID": row["gbifID"],
        "taxonKey": row["taxonKey"],
    }
    for rank in TAXONOMY_RANKS:
        cols[rank] = row[rank]
    return cols


_PATH_COLS = ["audio_path", "16khz_path", "32khz_path", "image_path", "video_path"]
_VIDEO_META = [
    "fps",
    "frame_count",
    "duration_seconds",
    "frame_height",
    "frame_width",
    "reliable_audio",
]


def _base_row(
    asset_id: str, modality: str, label: int, split: str, taxa: pd.DataFrame
) -> dict[str, object]:
    """Construct the common manifest fields for one asset.

    Parameters
    ----------
    asset_id : str
        Asset identifier.
    modality : str
        ``audio`` / ``video`` / ``image``.
    label : int
        SSW60 class label.
    split : str
        Source split.
    taxa : pd.DataFrame
        GBIF-linked taxa table indexed by ``label``.

    Returns
    -------
    dict[str, object]
        The row with id/modality/label/split + taxonomy + empty path/video
        columns (filled by the caller).
    """
    row: dict[str, object] = {
        "asset_id": asset_id,
        "modality": modality,
        "label": label,
        "split": split,
    }
    row.update(_gbif_cols(taxa, label))
    for c in _PATH_COLS:
        row[c] = ""
    for c in _VIDEO_META:
        row[c] = ""
    return row


def build_audio(src: Path, taxa: pd.DataFrame, gcs_root: str) -> pd.DataFrame:
    """Build the audio manifest.

    Parameters
    ----------
    src : Path
        Extracted SSW60 root.
    taxa : pd.DataFrame
        GBIF-linked taxa table indexed by label.
    gcs_root : str
        GCS root for absolute path columns.

    Returns
    -------
    pd.DataFrame
        The audio manifest.
    """
    df = pd.read_csv(src / "audio_ml.csv", keep_default_na=False, na_values=[])
    rows = []
    for _, r in df.iterrows():
        aid = str(r["asset_id"])
        row = _base_row(aid, "audio", int(r["label"]), str(r["split"]), taxa)
        row["audio_path"] = f"{gcs_root}/audio/{aid}.wav"
        row["16khz_path"] = f"{gcs_root}/audio_16k/{aid}.wav"
        row["32khz_path"] = f"{gcs_root}/audio_32k/{aid}.wav"
        row["duration_seconds"] = r["duration_seconds"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_video(src: Path, taxa: pd.DataFrame, gcs_root: str) -> pd.DataFrame:
    """Build the video manifest.

    Parameters
    ----------
    src : Path
        Extracted SSW60 root.
    taxa : pd.DataFrame
        GBIF-linked taxa table indexed by label.
    gcs_root : str
        GCS root for absolute path columns.

    Returns
    -------
    pd.DataFrame
        The video manifest.
    """
    df = pd.read_csv(src / "video_ml.csv", keep_default_na=False, na_values=[])
    rows = []
    for _, r in df.iterrows():
        aid = str(r["asset_id"])
        row = _base_row(aid, "video", int(r["label"]), str(r["split"]), taxa)
        row["video_path"] = f"{gcs_root}/video/{aid}.mp4"
        row["fps"] = r["fps"]
        row["frame_count"] = r["frame_count"]
        row["duration_seconds"] = r["duration_seconds"]
        row["frame_height"] = r["frame_height"]
        row["frame_width"] = r["frame_width"]
        row["reliable_audio"] = r["reliable_audio"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_images(src: Path, taxa: pd.DataFrame, gcs_root: str) -> pd.DataFrame:
    """Build the combined image manifest (iNat2021 + NABirds).

    Parameters
    ----------
    src : Path
        Extracted SSW60 root.
    taxa : pd.DataFrame
        GBIF-linked taxa table indexed by label.
    gcs_root : str
        GCS root for absolute path columns.

    Returns
    -------
    pd.DataFrame
        The image manifest spanning both image sources.
    """
    rows = []
    missing = 0
    for csv_name, subdir in [
        ("images_inat.csv", "images_inat"),
        ("images_nabirds.csv", "images_nabirds"),
    ]:
        df = pd.read_csv(src / csv_name, keep_default_na=False, na_values=[])
        for _, r in df.iterrows():
            aid = str(r["asset_id"])
            # Drop rows whose source image is absent (the upstream NABirds
            # CSV references one asset, 13e5d907…, with no shipped JPG); a
            # manifest must never point at a missing file.
            if not (src / subdir / f"{aid}.jpg").exists():
                missing += 1
                continue
            row = _base_row(aid, "image", int(r["label"]), str(r["split"]), taxa)
            row["image_path"] = f"{gcs_root}/{subdir}/{aid}.jpg"
            rows.append(row)
    if missing:
        print(f"build_images: dropped {missing} rows with no source image file")
    return pd.DataFrame(rows)


def write_splits(df: pd.DataFrame, out: Path, prefix: str, split_map: dict[str, str]) -> None:
    """Write per-split + ``_all`` CSVs for a modality manifest.

    Parameters
    ----------
    df : pd.DataFrame
        The modality manifest (with a ``split`` column).
    out : Path
        Staging output directory.
    prefix : str
        Filename prefix (e.g. ``ssw60_audio``).
    split_map : dict[str, str]
        Maps source split value -> output filename suffix
        (e.g. ``{"train": "train", "test": "test"}``).
    """
    df.to_csv(out / f"{prefix}_all.csv", index=False)
    print(f"{prefix}_all.csv: {len(df)} rows")
    for src_split, suffix in split_map.items():
        sub = df[df["split"] == src_split]
        sub.to_csv(out / f"{prefix}_{suffix}.csv", index=False)
        print(f"{prefix}_{suffix}.csv: {len(sub)} rows")


def main() -> None:
    """Run the full SSW60 manifest + audio-mirror build."""
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Extracted SSW60 root dir.")
    p.add_argument("--out", required=True, help="Staging output dir.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--gcs-root", default=GCS_ROOT_DEFAULT)
    p.add_argument("--gbif-cache", default=None, help="Local cache path for the GBIF animals TSV.")
    p.add_argument(
        "--skip-resample", action="store_true", help="Skip audio resampling (manifests only)."
    )
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=== 1. GBIF-link taxa ===", flush=True)
    taxa = link_taxa(src, out, args.gbif_cache)

    print("\n=== 2. build manifests ===", flush=True)
    audio = build_audio(src, taxa, args.gcs_root)
    video = build_video(src, taxa, args.gcs_root)
    images = build_images(src, taxa, args.gcs_root)

    unified = pd.concat([audio, video, images], ignore_index=True)
    unified.to_csv(out / "ssw60_all.csv", index=False)
    print(
        f"ssw60_all.csv: {len(unified)} rows "
        f"(audio={len(audio)}, video={len(video)}, image={len(images)})"
    )

    write_splits(audio, out, "ssw60_audio", {"train": "train", "test": "test"})
    write_splits(video, out, "ssw60_video", {"train": "train", "test": "test"})
    write_splits(images, out, "ssw60_images", {"train": "train", "test": "test", "val": "val"})

    if not args.skip_resample:
        print("\n=== 3. resample audio -> 16k / 32k ===", flush=True)
        resample_audio(src, out, audio["asset_id"].astype(str).tolist(), args.workers)
    else:
        print("\n=== 3. resample audio SKIPPED ===")

    print("\nDONE.")


if __name__ == "__main__":
    main()
