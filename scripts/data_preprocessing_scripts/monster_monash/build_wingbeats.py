"""Ingest the monster-monash MosquitoSound / InsectSound wingbeat datasets.

Both are downstream of the UCR archive (Potamitis 2018 Wingbeats /
Chen et al. 2014 InsectSound) and ship as float32 numpy arrays + integer
class labels on HuggingFace (``monster-monash/{MosquitoSound,InsectSound}``).

Pipeline (per dataset):
1. Stream ``${NAME}_X.npy`` + ``${NAME}_y.npy`` from HF.
2. For each clip:
   - Write a 6 kHz native FLAC under ``audio/<shard>/clip_<idx>.flac``
   - Resample to 16 kHz / 32 kHz, write parallel FLACs under
     ``audio_16k/<shard>/clip_<idx>.flac`` and ``audio_32k/<shard>/clip_<idx>.flac``
3. Build the manifest CSV with ``audio_path``, ``16khz_path``, ``32khz_path``,
   ``class_id``, ``species`` (best-known mapping), GBIF taxonomy, license,
   plus the 5-fold cross-validation test-indices columns.
4. Upload everything to ``gs://esp-data-ingestion/{name}/v0.1.0/``.

Memory model: ``X.npy`` is memmapped (no full load). One clip at a time
per worker. Peak RAM ~few hundred MB even for MosquitoSound's 4 GB array.

Usage:
    uv run python build_wingbeats.py --dataset MosquitoSound --workers 8 [--upload]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT_GCS_ROOT = "gs://esp-data-ingestion"

# --- Class-name mappings (best-known per source-dataset documentation) ---

# MosquitoSound = Wingbeats (Potamitis 2018 Kaggle). Class IDs are in
# alphabetical order matching the Wingbeats Kaggle folder structure.
MOSQUITO_SPECIES = [
    "Aedes aegypti",
    "Aedes albopictus",
    "Anopheles arabiensis",
    "Anopheles gambiae",
    "Culex pipiens",
    "Culex quinquefasciatus",
]
MOSQUITO_COMMON = [
    "Yellow fever mosquito",
    "Asian tiger mosquito",
    "Arabian malaria mosquito",
    "African malaria mosquito",
    "Northern house mosquito",
    "Southern house mosquito",
]
# GBIF taxonKeys (Culicidae; verified against gbif.org backbone).
MOSQUITO_GBIF = {
    "Aedes aegypti": "1651430",
    "Aedes albopictus": "1652212",
    "Anopheles arabiensis": "1657898",
    "Anopheles gambiae": "1657917",
    "Culex pipiens": "1659938",
    "Culex quinquefasciatus": "1659847",
}

# InsectSound = UCR InsectSound (Chen 2014). 10 classes, 5,000 each.
# Class names follow the canonical UCR documentation (5 taxa × 2 sexes).
INSECT_TAXA_SEX = [
    ("Aedes aegypti", "female"),
    ("Aedes aegypti", "male"),
    ("Culex stigmatosoma", "female"),
    ("Culex stigmatosoma", "male"),
    ("Culex tarsalis", "female"),
    ("Culex tarsalis", "male"),
    ("Culex quinquefasciatus", "female"),
    ("Culex quinquefasciatus", "male"),
    ("Musca domestica", "female"),
    ("Drosophila simulans", "female"),
]
INSECT_GBIF = {
    "Aedes aegypti": "1651430",
    "Culex stigmatosoma": "1659885",
    "Culex tarsalis": "1660062",
    "Culex quinquefasciatus": "1659847",
    "Musca domestica": "1497987",
    "Drosophila simulans": "1496710",
}

# GBIF higher taxonomy for the mosquito + housefly + drosophila families
# encountered in both datasets. Looked up once; pasted to avoid an
# online GBIF call per clip.
GBIF_HIGHER = {
    "Aedes aegypti": (
        "Animalia",
        "Arthropoda",
        "Insecta",
        "Diptera",
        "Culicidae",
        "Aedes",
        "Yellow fever mosquito",
    ),
    "Aedes albopictus": (
        "Animalia",
        "Arthropoda",
        "Insecta",
        "Diptera",
        "Culicidae",
        "Aedes",
        "Asian tiger mosquito",
    ),
    "Anopheles arabiensis": (
        "Animalia",
        "Arthropoda",
        "Insecta",
        "Diptera",
        "Culicidae",
        "Anopheles",
        "Arabian malaria mosquito",
    ),
    "Anopheles gambiae": (
        "Animalia",
        "Arthropoda",
        "Insecta",
        "Diptera",
        "Culicidae",
        "Anopheles",
        "African malaria mosquito",
    ),
    "Culex pipiens": (
        "Animalia",
        "Arthropoda",
        "Insecta",
        "Diptera",
        "Culicidae",
        "Culex",
        "Northern house mosquito",
    ),
    "Culex quinquefasciatus": (
        "Animalia",
        "Arthropoda",
        "Insecta",
        "Diptera",
        "Culicidae",
        "Culex",
        "Southern house mosquito",
    ),
    "Culex stigmatosoma": (
        "Animalia",
        "Arthropoda",
        "Insecta",
        "Diptera",
        "Culicidae",
        "Culex",
        "Banded foul-water mosquito",
    ),
    "Culex tarsalis": (
        "Animalia",
        "Arthropoda",
        "Insecta",
        "Diptera",
        "Culicidae",
        "Culex",
        "Western encephalitis mosquito",
    ),
    "Musca domestica": (
        "Animalia",
        "Arthropoda",
        "Insecta",
        "Diptera",
        "Muscidae",
        "Musca",
        "Common housefly",
    ),
    "Drosophila simulans": (
        "Animalia",
        "Arthropoda",
        "Insecta",
        "Diptera",
        "Drosophilidae",
        "Drosophila",
        "Simulans fruit fly",
    ),
}


def _hf_url(name: str, fname: str) -> str:
    return f"https://huggingface.co/datasets/monster-monash/{name}/resolve/main/{fname}"


def _download(name: str, out_dir: Path) -> None:
    """Fetch X.npy / y.npy / test_indices_fold_*.txt for the given dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [f"{name}_X.npy", f"{name}_y.npy"] + [f"test_indices_fold_{i}.txt" for i in range(5)]
    for f in files:
        dst = out_dir / f
        if dst.exists() and dst.stat().st_size > 0:
            print(f"  cached {f} ({dst.stat().st_size:,}B)", flush=True)
            continue
        url = _hf_url(name, f)
        print(f"  downloading {f} from {url}", flush=True)
        # curl -L follows redirects to the LFS CDN.
        rc = subprocess.run(
            ["curl", "-sSL", "--max-time", "1800", "--retry", "3", "-o", str(dst), url],
            check=False,
        ).returncode
        if rc != 0 or not dst.exists() or dst.stat().st_size == 0:
            raise RuntimeError(f"download failed for {f} (rc={rc})")
        print(f"    -> {dst.stat().st_size:,}B", flush=True)


def _load_fold_indices(in_dir: Path, n_rows: int) -> dict[int, set[int]]:
    """Return ``{fold: set_of_test_indices}`` for the 5 folds."""
    folds: dict[int, set[int]] = {}
    for k in range(5):
        p = in_dir / f"test_indices_fold_{k}.txt"
        idxs = np.loadtxt(p, dtype=np.int64).reshape(-1)
        # Defensive bounds check.
        bad = idxs[(idxs < 0) | (idxs >= n_rows)]
        if bad.size:
            raise ValueError(f"fold {k}: {bad.size} out-of-range indices")
        folds[k] = set(idxs.tolist())
    return folds


def _resample_quality() -> str:
    """Resampling backend: prefer soxr_hq (10x faster than kaiser_best, equivalent quality)."""
    return "soxr_hq"


def _write_clip(args: tuple) -> tuple[int, bool, str]:
    """Worker: write FLACs at 6/16/32 kHz for one clip.

    Parameters
    ----------
    args : tuple
        ``(idx, sample, work_dir, name, native_sr)`` where ``sample`` is
        the 1D numpy float32 audio array.
    """
    import librosa
    import soundfile as sf

    idx, sample, work_dir, name, native_sr = args
    shard = f"{idx // 1000:04d}"
    base = f"clip_{idx:06d}.flac"
    # Folder names mirror the layout used by F0Bioacoustic / DORI.
    native_dir = work_dir / "audio" / shard
    sr16_dir = work_dir / "audio_16k" / shard
    sr32_dir = work_dir / "audio_32k" / shard
    for d in (native_dir, sr16_dir, sr32_dir):
        d.mkdir(parents=True, exist_ok=True)
    try:
        sample = np.asarray(sample, dtype=np.float32).reshape(-1)
        # Soft-clip to [-1, 1] to be safe before integer encoding.
        peak = np.max(np.abs(sample)) if sample.size else 1.0
        if peak > 1.0:
            sample = sample / peak
        sf.write(str(native_dir / base), sample, native_sr, subtype="PCM_16", format="FLAC")
        a16 = librosa.resample(
            sample, orig_sr=native_sr, target_sr=16000, res_type=_resample_quality()
        ).astype(np.float32)
        sf.write(str(sr16_dir / base), a16, 16000, subtype="PCM_16", format="FLAC")
        a32 = librosa.resample(
            sample, orig_sr=native_sr, target_sr=32000, res_type=_resample_quality()
        ).astype(np.float32)
        sf.write(str(sr32_dir / base), a32, 32000, subtype="PCM_16", format="FLAC")
        return idx, True, ""
    except Exception as e:
        return idx, False, f"{type(e).__name__}: {e}"


def write_clips(
    name: str, X: np.ndarray, native_sr: int, work_dir: Path, workers: int
) -> dict[int, bool]:
    """Write FLACs for every clip; return ``{idx: success}``."""
    n = X.shape[0]
    print(f"Writing {n:,} clips × 3 sample-rate variants ({workers} workers)...", flush=True)
    t0 = time.time()
    ok: dict[int, bool] = {}
    errors: list[tuple[int, str]] = []
    # Iterate the memmapped array row-by-row; pass a copy to the worker so
    # the worker isn't dependent on the parent's mmap lifetime.

    def _iter():
        for i in range(n):
            yield (i, np.array(X[i]).astype(np.float32).reshape(-1), work_dir, name, native_sr)

    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        completed = 0
        for idx, success, err in ex.map(_write_clip, _iter(), chunksize=64):
            completed += 1
            ok[idx] = success
            if not success:
                errors.append((idx, err))
                if len(errors) <= 5:
                    print(f"  ERR clip {idx}: {err}", flush=True)
            if completed % 5000 == 0:
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-3)
                eta = (n - completed) / max(rate, 1e-3) / 60.0
                print(
                    f"  {completed:,}/{n:,}  rate={rate:.0f}/s  eta={eta:.1f} min  "
                    f"err={len(errors)}",
                    flush=True,
                )
    print(f"Done. {sum(ok.values()):,} ok, {n - sum(ok.values())} errors", flush=True)
    return ok


def build_manifest(
    name: str,
    y: np.ndarray,
    n_native_samples: int,
    native_sr: int,
    folds: dict[int, set[int]],
    ok: dict[int, bool],
    out_csv_dir: Path,
) -> dict[str, int]:
    """Build per-clip rows + write `<name>_all/train/val.csv`."""
    if name == "MosquitoSound":
        species_labels = MOSQUITO_SPECIES
        gbif_lookup = MOSQUITO_GBIF
        common_lookup = dict(zip(MOSQUITO_SPECIES, MOSQUITO_COMMON, strict=True))
        sex_per_class: list[str] = [""] * 6
    elif name == "InsectSound":
        species_labels = [sp for sp, _ in INSECT_TAXA_SEX]
        sex_per_class = [sx for _, sx in INSECT_TAXA_SEX]
        gbif_lookup = INSECT_GBIF
        common_lookup = {sp: GBIF_HIGHER[sp][-1] for sp in set(species_labels)}
    else:
        raise ValueError(f"unknown dataset {name}")

    n = len(y)
    rows = []
    val_set = folds[0]  # fold_0 test indices = val for our convention
    duration_s = n_native_samples / native_sr
    for i in range(n):
        if not ok.get(i, False):
            continue
        shard = f"{i // 1000:04d}"
        base = f"clip_{i:06d}.flac"
        cid = int(y[i])
        species = species_labels[cid]
        family, genus = GBIF_HIGHER[species][4], GBIF_HIGHER[species][5]
        kingdom, phylum, cls, order = GBIF_HIGHER[species][:4]
        sp_common = common_lookup.get(species, GBIF_HIGHER[species][-1])
        rows.append(
            {
                "clip_id": f"{name.lower()}_{i:06d}",
                "audio_path": f"audio/{shard}/{base}",
                "16khz_path": f"audio_16k/{shard}/{base}",
                "32khz_path": f"audio_32k/{shard}/{base}",
                "audio_duration": round(duration_s, 4),
                "native_sample_rate": native_sr,
                "class_id": cid,
                "species": species,
                "canonical_name": species,
                "scientific_name_unified": species,
                "species_common": sp_common,
                "sex": sex_per_class[cid],
                "gbifID": gbif_lookup.get(species, ""),
                "kingdom": kingdom,
                "phylum": phylum,
                "class": cls,
                "order": order,
                "family": family,
                "genus": genus,
                "fold_0_test": int(i in folds[0]),
                "fold_1_test": int(i in folds[1]),
                "fold_2_test": int(i in folds[2]),
                "fold_3_test": int(i in folds[3]),
                "fold_4_test": int(i in folds[4]),
                "split": "val" if i in val_set else "train",
                "license": "Public Domain",
                "source_dataset": f"monster-monash/{name}",
            }
        )
    df = pd.DataFrame(rows)
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    snake = "mosquito_sound" if name == "MosquitoSound" else "insect_sound"
    paths = {
        "all": out_csv_dir / f"{snake}_all.csv",
        "train": out_csv_dir / f"{snake}_train.csv",
        "val": out_csv_dir / f"{snake}_val.csv",
    }
    df.to_csv(paths["all"], index=False)
    df[df["split"] == "train"].drop(columns=["split"]).to_csv(paths["train"], index=False)
    df[df["split"] == "val"].drop(columns=["split"]).to_csv(paths["val"], index=False)
    print(f"Manifest: {len(df):,} rows -> {paths['all']}")
    print(f"  train: {len(df[df['split'] == 'train']):,}")
    print(f"  val:   {len(df[df['split'] == 'val']):,}")
    print(df["species"].value_counts().to_string())
    return {k: int((df["split"] == k).sum()) for k in ("train", "val")}


def upload_gcs(name: str, work_dir: Path) -> None:
    """rsync audio/ + audio_16k/ + audio_32k/ + CSVs to GCS."""
    snake = "mosquito-sound" if name == "MosquitoSound" else "insect-sound"
    gcs_root = f"{OUT_GCS_ROOT}/monster-monash-{snake}/v0.1.0"
    print(f"Uploading to {gcs_root}/ ...", flush=True)
    for sub in ("audio", "audio_16k", "audio_32k"):
        src = work_dir / sub
        if not src.exists():
            continue
        print(f"  rsync {sub}/...", flush=True)
        subprocess.run(
            ["gsutil", "-m", "rsync", "-r", str(src) + "/", f"{gcs_root}/{sub}/"],
            check=True,
        )
    for f in work_dir.glob("*.csv"):
        subprocess.run(
            ["gsutil", "-q", "cp", str(f), f"{gcs_root}/{f.name}"],
            check=True,
        )
    print("Upload done.", flush=True)


def main() -> None:
    """Run the full ingest for a single dataset."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["MosquitoSound", "InsectSound"])
    p.add_argument("--work-dir", default="/mnt/home/esp-data-dev/monster_monash_staging")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--upload", action="store_true")
    p.add_argument(
        "--skip-write",
        action="store_true",
        help="Skip clip writing (just manifest + upload from cached dir).",
    )
    p.add_argument("--clean-audio-after-upload", action="store_true")
    args = p.parse_args()

    name = args.dataset
    work_dir = Path(args.work_dir) / name
    cache_dir = Path(args.work_dir)  # X / y / fold files live here

    _download(name, cache_dir)

    Y_path = cache_dir / f"{name}_y.npy"
    X_path = cache_dir / f"{name}_X.npy"
    y = np.load(Y_path)
    X = np.load(X_path, mmap_mode="r")
    print(f"Loaded {name}: X.shape={X.shape}, y.shape={y.shape}, dtype={X.dtype}")
    native_sr = 6000
    n_native_samples = X.shape[-1]
    print(f"  native sample-rate={native_sr} Hz, clip length={n_native_samples / native_sr:.3f}s")

    folds = _load_fold_indices(cache_dir, n_rows=X.shape[0])
    print(f"  folds: {[len(folds[k]) for k in range(5)]}")

    if args.skip_write:
        # Assume previous run wrote clips already; mark all as ok.
        ok = {i: True for i in range(X.shape[0])}
    else:
        ok = write_clips(name, X, native_sr, work_dir, args.workers)

    build_manifest(name, y, n_native_samples, native_sr, folds, ok, work_dir)

    if args.upload:
        upload_gcs(name, work_dir)
        if args.clean_audio_after_upload:
            for sub in ("audio", "audio_16k", "audio_32k"):
                d = work_dir / sub
                if d.exists():
                    shutil.rmtree(d)
                    print(f"  removed local {sub}/")


if __name__ == "__main__":
    main()
