"""Build the standalone PIFSC PIPAN dataset manifest CSVs.

Source data:
- Annotations CSV: ``gs://esp-data-ingestion/superwhale/v0.1.0/raw/pifsc/annotations.csv``
  (byte-identical to the public NOAA bucket; 38,857 rows)
- Audio (originals, 10 kHz FLAC):
  ``gs://esp-data-ingestion/superwhale/v0.1.0/raw/pifsc/audio/pipan_10/...``
- Audio (pre-resampled, partial — 1,268 / 5,489 unique files):
  ``gs://esp-data-ingestion/superwhale/v0.1.0/raw/audio_{16k,32k}/pifsc/audio/pipan_10/...``

Outputs (uploaded to ``gs://esp-data-ingestion/pifsc-pipan/v0.1.0/``):
- ``pifsc_pipan_all.csv``        — every annotation event (one per row)
- ``pifsc_pipan_train.csv``      — ~90% of files, deployment-stratified
- ``pifsc_pipan_val.csv``        — ~10% of files, deployment-stratified
- ``pifsc_pipan_xwav_index.csv`` — per-FLAC subchunk table (side artifact)
- ``pifsc_pipan_labels.csv``     — label vocabulary with descriptions

Usage:
    uv run python scripts/data_preprocessing_scripts/pifsc_pipan/build_pifsc_pipan.py \
        --out-dir ~/pifsc_pipan_staging --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from xwav_parser import parse_xwav_from_gcs  # noqa: E402

ANNOTATIONS_CSV = "gs://esp-data-ingestion/superwhale/v0.1.0/raw/pifsc/annotations.csv"
# Source-of-truth audio: NOAA public bucket. The annotations.csv references
# files under ``pipan/`` (the old layout); the public bucket has since
# been reorganised to ``pipan_10/`` AND some files were moved between
# deployment-subdirectories. We do a one-time bulk-list and look up each
# file by basename rather than relying on the recorded directory.
NOAA_FLAC_PREFIX = "gs://noaa-passive-bioacoustic/pifsc/audio/pipan_10"
# Partial pre-resampled mirror under our own bucket — when present (~23%
# of files), the loader can skip the 10→{16,32} kHz upsample at load time.
MIRROR_AUDIO_ROOT = "gs://esp-data-ingestion/superwhale/v0.1.0/raw"
MIRROR_16K_PREFIX = f"{MIRROR_AUDIO_ROOT}/audio_16k/pifsc/audio/pipan_10"
MIRROR_32K_PREFIX = f"{MIRROR_AUDIO_ROOT}/audio_32k/pifsc/audio/pipan_10"
OUT_GCS_ROOT = "gs://esp-data-ingestion/pifsc-pipan/v0.1.0"


def _bulk_list_noaa(cache_path: Path) -> dict[str, str]:
    """Bulk-list every FLAC under the NOAA PIPAN bucket; cache locally.

    Returns
    -------
    dict[str, str]
        ``{filename: full_gs_uri}`` — filenames are unique across the
        whole bucket (verified) so this is an unambiguous lookup table.
    """
    if cache_path.exists():
        print(f"Reading NOAA listing cache {cache_path} ...", flush=True)
    else:
        print(f"Bulk-listing {NOAA_FLAC_PREFIX} ...", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            subprocess.run(
                ["gsutil", "ls", "-r", f"{NOAA_FLAC_PREFIX}/**"],
                stdout=f,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=600,
            )
    flacs: dict[str, str] = {}
    for line in cache_path.read_text().splitlines():
        line = line.strip()
        if line.endswith(".flac"):
            flacs[line.rsplit("/", 1)[-1]] = line
    print(f"  {len(flacs):,} unique FLACs in the NOAA bucket")
    return flacs


# Allen 2021 label vocabulary (from README at the source).
LABEL_VOCAB = {
    "Mn": {
        "description": "Humpback whale (Megaptera novaeangliae) song",
        "species": "Megaptera novaeangliae",
        "coarse_call_type": "song",
        "is_biological": True,
    },
    "Background": {
        "description": "Environmental sounds only (no biological/anthropogenic events)",
        "species": "",
        "coarse_call_type": "background",
        "is_biological": False,
    },
    "Other": {
        "description": "Acoustic event present, not in the other categories",
        "species": "",
        "coarse_call_type": "other",
        "is_biological": False,
    },
    "Vessel": {
        "description": "Vessel noise",
        "species": "",
        "coarse_call_type": "noise",
        "is_biological": False,
    },
    "Fish": {
        "description": "Fish sound not otherwise identified",
        "species": "",
        "coarse_call_type": "fish",
        "is_biological": True,
    },
    "Device": {
        "description": "Noise from the recording equipment",
        "species": "",
        "coarse_call_type": "noise",
        "is_biological": False,
    },
}

# GBIF taxonomy for the one species in the Allen 2021 vocabulary. Hard-coded
# since the only species is Megaptera novaeangliae.
GBIF_MN = {
    "canonical_name": "Megaptera novaeangliae",
    "gbifID": "2440735",
    "kingdom": "Animalia",
    "phylum": "Chordata",
    "class": "Mammalia",
    "order": "Artiodactyla",
    "family": "Balaenopteridae",
    "genus": "Megaptera",
    "species_common": "Humpback whale",
}


def _resampled_paths(noaa_uri: str) -> tuple[str, str]:
    """Return the parallel 16 kHz and 32 kHz WAV URIs for a NOAA FLAC URI.

    Source: ``gs://noaa-passive-bioacoustic/pifsc/audio/pipan_10/{deploy}/{site}/audio/{file}.flac``
    Mirror: ``gs://esp-data-ingestion/superwhale/v0.1.0/raw/audio_{16k,32k}/pifsc/audio/pipan_10/{deploy}/{site}/audio/audio/{file}.wav``

    Returns
    -------
    tuple[str, str]
        ``(16khz_uri, 32khz_uri)`` derived purely by string transform.
    """
    src_prefix = f"{NOAA_FLAC_PREFIX}/"
    if not noaa_uri.startswith(src_prefix):
        raise ValueError(f"Unexpected source URI prefix: {noaa_uri}")
    rel = noaa_uri[len(src_prefix) :]  # <deploy>/<site>/audio/<file>.flac
    parts = rel.split("/")
    if len(parts) < 4 or parts[-2] != "audio":
        raise ValueError(f"Unexpected source URI shape: {noaa_uri}")
    deploy, site, _audio, fname = parts[0], parts[1], parts[2], "/".join(parts[3:])
    wname = fname[:-5] + ".wav" if fname.endswith(".flac") else fname
    rel_mirror = f"{deploy}/{site}/audio/audio/{wname}"
    return (f"{MIRROR_16K_PREFIX}/{rel_mirror}", f"{MIRROR_32K_PREFIX}/{rel_mirror}")


def _bulk_list(gcs_prefix: str) -> set[str]:
    """Return the full set of object URIs under ``gcs_prefix`` (recursive)."""
    print(f"Listing {gcs_prefix}/ ...", flush=True)
    proc = subprocess.run(
        ["gsutil", "ls", "-r", f"{gcs_prefix}/**"],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return {line.strip() for line in proc.stdout.splitlines() if line.strip().endswith(".wav")}


def _deployment_of(src_uri: str) -> str:
    """Extract deployment slug (e.g. ``saipan``) from a NOAA FLAC URI."""
    rel = src_uri[len(NOAA_FLAC_PREFIX) + 1 :]
    return rel.split("/", 1)[0]


def _parse_one(args: tuple[str, str, set[int]]) -> tuple[str, dict | None, str | None]:
    """Parse XWAV header for one FLAC; return its per-file summary or an error.

    Parameters
    ----------
    args : tuple
        ``(src_uri, mirror_uri, needed_subchunk_indices)`` — only the listed
        subchunk indices need their cum-offset computed; we still parse the
        whole table to verify uniformity.

    Returns
    -------
    tuple[str, dict | None, str | None]
        ``(src_uri, summary, error)`` where ``summary`` is a dict with keys
        ``src_uri``, ``n_subchunks``, ``min_dur_s``, ``max_dur_s``,
        ``mean_dur_s``, ``total_dur_s``, ``is_uniform``, and ``cum_offsets``
        (a dict ``{subchunk_index: cum_offset_s_before}`` only for the
        requested indices, plus subchunk 0 with cum=0).
    """
    src_uri, mirror_uri, needed = args
    try:
        idx = parse_xwav_from_gcs(mirror_uri, head_bytes=262144)
    except ValueError:
        try:
            idx = parse_xwav_from_gcs(mirror_uri, head_bytes=1_048_576)
        except Exception as e2:
            return src_uri, None, f"retry failed: {e2}"
    except Exception as e:
        return src_uri, None, f"{type(e).__name__}: {e}"

    durations = [sc.duration_s for sc in idx.subchunks]
    cum = idx.cum_offset_s
    # Only persist the cum-offsets we actually need (annotated subchunks).
    cum_offsets = {i: round(cum[i], 6) for i in needed if 0 <= i < len(durations)}
    summary = {
        "src_uri": src_uri,
        "n_subchunks": len(durations),
        "min_dur_s": round(min(durations), 6) if durations else 0.0,
        "max_dur_s": round(max(durations), 6) if durations else 0.0,
        "mean_dur_s": round(sum(durations) / len(durations), 6) if durations else 0.0,
        "total_dur_s": round(cum[-1], 6) if cum else 0.0,
        "is_uniform": bool(durations and max(durations) - min(durations) < 1e-6),
        "cum_offsets": cum_offsets,
    }
    return src_uri, summary, None


def build_xwav_summaries(
    needed_by_src: dict[str, tuple[str, set[int]]],
    workers: int,
    cache_path: Path,
) -> dict[str, dict]:
    """Parse XWAV headers for every unique FLAC; cache per-file summaries.

    Parameters
    ----------
    needed_by_src : dict[str, tuple[str, set[int]]]
        Maps source URI to ``(mirror_uri, set_of_needed_subchunk_indices)``.
    workers : int
        Parallel header parsers.
    cache_path : Path
        Where to persist the per-file summary CSV.

    Returns
    -------
    dict[str, dict]
        Maps source URI to its summary dict (with ``cum_offsets``).
    """
    if cache_path.exists():
        print(f"Reading cached XWAV summaries from {cache_path} ...", flush=True)
        df = pd.read_csv(cache_path)
        # ``cum_offsets`` was serialised as a JSON string per row.
        import json

        result: dict[str, dict] = {}
        for r in df.itertuples(index=False):
            cum_str = r.cum_offsets_json
            cum = json.loads(cum_str) if isinstance(cum_str, str) and cum_str else {}
            cum = {int(k): float(v) for k, v in cum.items()}
            result[r.src_uri] = {
                "src_uri": r.src_uri,
                "n_subchunks": int(r.n_subchunks),
                "min_dur_s": float(r.min_dur_s),
                "max_dur_s": float(r.max_dur_s),
                "mean_dur_s": float(r.mean_dur_s),
                "total_dur_s": float(r.total_dur_s),
                "is_uniform": bool(r.is_uniform),
                "cum_offsets": cum,
            }
        return result

    print(
        f"Parsing XWAV headers for {len(needed_by_src)} FLAC files with {workers} workers ...",
        flush=True,
    )
    args_iter = [(src, mu, idxs) for src, (mu, idxs) in needed_by_src.items()]

    summaries: dict[str, dict] = {}
    errors: list[tuple[str, str]] = []
    completed = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_parse_one, a) for a in args_iter]
        for fut in cf.as_completed(futures):
            src_uri, summary, err = fut.result()
            completed += 1
            if err is not None:
                errors.append((src_uri, err))
                if len(errors) <= 10:
                    print(f"  ERR [{completed}/{len(needed_by_src)}] {src_uri}: {err}", flush=True)
                continue
            summaries[src_uri] = summary
            if completed % 100 == 0 or completed == len(needed_by_src):
                print(
                    f"  parsed {completed}/{len(needed_by_src)} ({len(errors)} errors so far)",
                    flush=True,
                )
    print(f"Parsed {len(summaries)} / {len(needed_by_src)} FLACs; {len(errors)} errors")

    # Persist compact cache.
    import json

    cache_rows = [
        {
            "src_uri": s["src_uri"],
            "n_subchunks": s["n_subchunks"],
            "min_dur_s": s["min_dur_s"],
            "max_dur_s": s["max_dur_s"],
            "mean_dur_s": s["mean_dur_s"],
            "total_dur_s": s["total_dur_s"],
            "is_uniform": s["is_uniform"],
            "cum_offsets_json": json.dumps(s["cum_offsets"]),
        }
        for s in summaries.values()
    ]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cache_rows).to_csv(cache_path, index=False)
    print(f"Cached XWAV summaries -> {cache_path}")

    if errors:
        err_path = cache_path.with_name(cache_path.stem + "_errors.csv")
        pd.DataFrame(errors, columns=["src_uri", "error"]).to_csv(err_path, index=False)
        print(f"Wrote {len(errors)} parse errors -> {err_path}")

    return summaries


def split_deployment_stratified(
    unique_flacs_by_deploy: dict[str, list[str]], val_frac: float, seed: int
) -> tuple[set[str], set[str]]:
    """Deployment-stratified file-level split.

    For each deployment, allocate ``ceil(val_frac × n_files)`` files to val
    (random with the given seed), the rest to train. A deployment with a
    single file goes entirely to train.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(train_files, val_files)`` — sets of mirror FLAC URIs.
    """
    import random

    rng = random.Random(seed)
    train: set[str] = set()
    val: set[str] = set()
    for deploy, files in unique_flacs_by_deploy.items():
        files = sorted(files)
        n_val = int(round(len(files) * val_frac))
        if len(files) > 1 and n_val == 0:
            n_val = 1  # ensure each deployment contributes at least 1 val file
        n_val = min(n_val, len(files) - 1) if len(files) > 1 else 0
        rng.shuffle(files)
        val.update(files[:n_val])
        train.update(files[n_val:])
    return train, val


def main() -> None:
    """Build the standalone PIFSC PIPAN manifest."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=os.path.expanduser("~/pifsc_pipan_staging"))
    p.add_argument("--out-gcs", default=OUT_GCS_ROOT)
    p.add_argument("--workers", type=int, default=8, help="Parallel XWAV header parsers.")
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--upload", action="store_true", help="Upload final CSVs to GCS.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load annotations.csv ----
    print(f"Fetching {ANNOTATIONS_CSV} ...", flush=True)
    raw = subprocess.run(
        ["gsutil", "cat", ANNOTATIONS_CSV],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    ).stdout
    df = pd.read_csv(
        io.StringIO(raw),
        keep_default_na=False,
        na_values=[""],
        dtype={
            "subchunk_index": "int64",
            "label_is_strong": "boolean",
            "implicit_negatives": "boolean",
            "begin_rel_subchunk": "float64",
            "end_rel_subchunk": "float64",
        },
    )
    print(f"Loaded {len(df):,} annotation rows")

    # ---- 2. Derive deployment + audio_path columns ----
    # The source `flac_compressed_xwav_object` references some files at
    # paths that no longer exist (legacy ``pipan/`` prefix + some files
    # were moved between deployment subdirectories). Bulk-list the NOAA
    # bucket once and look up each file by basename.
    noaa_listing_cache = out_dir / "_noaa_flac_listing.txt"
    noaa_flacs = _bulk_list_noaa(noaa_listing_cache)
    src_paths = df["flac_compressed_xwav_object"].to_numpy()
    resolved = []
    unresolved = 0
    for p in src_paths:
        fname = p.rsplit("/", 1)[-1]
        canonical = noaa_flacs.get(fname)
        if canonical is None:
            unresolved += 1
            resolved.append(None)
        else:
            resolved.append(canonical)
    if unresolved:
        print(
            f"WARN: {unresolved:,}/{len(df):,} annotation rows reference "
            f"FLACs absent from the NOAA bucket; dropping"
        )
    df["audio_path"] = resolved
    before = len(df)
    df = df.dropna(subset=["audio_path"]).reset_index(drop=True)
    print(f"After resolving NOAA paths: {len(df):,} rows ({before - len(df):,} dropped)")
    df["deployment"] = df["audio_path"].map(_deployment_of)

    # ---- 3. Build the per-FLAC XWAV summary + cum-offsets only for needed subchunks ----
    needed_by_src: dict[str, tuple[str, set[int]]] = {}
    for src, sc in df[["audio_path", "subchunk_index"]].itertuples(index=False, name=None):
        if src not in needed_by_src:
            needed_by_src[src] = (src, set())
        needed_by_src[src][1].add(int(sc))
    xwav_cache = out_dir / "pifsc_pipan_xwav_index.csv"
    summaries = build_xwav_summaries(needed_by_src, workers=args.workers, cache_path=xwav_cache)

    non_uniform = [s["src_uri"] for s in summaries.values() if not s["is_uniform"]]
    if non_uniform:
        print(
            f"NOTE: {len(non_uniform)} files have non-uniform subchunk durations "
            f"(first few: {non_uniform[:3]})"
        )

    # ---- 4. Compute begin_in_file_s / end_in_file_s for every annotation ----
    cum_lookup: dict[tuple[str, int], float] = {}
    for src, s in summaries.items():
        for sc_idx, off in s["cum_offsets"].items():
            cum_lookup[(src, int(sc_idx))] = float(off)

    src_col = df["audio_path"].to_numpy()
    idx_col = df["subchunk_index"].to_numpy()
    cum: list[float] = []
    missing = 0
    for s_uri, sc_i in zip(src_col, idx_col, strict=False):
        v = cum_lookup.get((s_uri, int(sc_i)))
        if v is None:
            missing += 1
            cum.append(float("nan"))
        else:
            cum.append(v)
    df["_cum_offset_before"] = cum
    if missing:
        print(
            f"WARN: {missing:,} annotation rows had no XWAV match (likely "
            "due to FLAC parse errors — see *_errors.csv)"
        )
    df["begin_in_file_s"] = (df["_cum_offset_before"] + df["begin_rel_subchunk"]).round(6)
    df["end_in_file_s"] = (df["_cum_offset_before"] + df["end_rel_subchunk"]).round(6)
    df.drop(columns=["_cum_offset_before"], inplace=True)

    # ---- 5. Bulk-list pre-resampled mirrors ----
    print("Listing pre-resampled 16 kHz mirror ...", flush=True)
    set_16k = _bulk_list(MIRROR_16K_PREFIX)
    print(f"  {len(set_16k):,} files")
    print("Listing pre-resampled 32 kHz mirror ...", flush=True)
    set_32k = _bulk_list(MIRROR_32K_PREFIX)
    print(f"  {len(set_32k):,} files")

    def _16k(u: str) -> str:
        try:
            wav, _ = _resampled_paths(u)
        except ValueError:
            return ""
        return wav if wav in set_16k else ""

    def _32k(u: str) -> str:
        try:
            _, wav = _resampled_paths(u)
        except ValueError:
            return ""
        return wav if wav in set_32k else ""

    df["16khz_path"] = df["audio_path"].map(_16k)
    df["32khz_path"] = df["audio_path"].map(_32k)

    # ---- 6. Derived label / species / call-type / GBIF columns ----
    df["coarse_call_type"] = df["label"].map(
        lambda lab: LABEL_VOCAB.get(lab, {}).get("coarse_call_type", "")
    )
    df["species"] = df["label"].map(lambda lab: LABEL_VOCAB.get(lab, {}).get("species", ""))
    for k, v in GBIF_MN.items():
        df[k] = df["label"].map(lambda lab, _v=v: _v if lab == "Mn" else "")
    df["license"] = "CC0-1.0"
    df["source_dataset"] = "pifsc_pipan"

    # ---- 7. Train/val split (deployment-stratified, file-level) ----
    by_deploy: dict[str, list[str]] = defaultdict(list)
    for u, d in (
        df[["audio_path", "deployment"]].drop_duplicates().itertuples(index=False, name=None)
    ):
        by_deploy[d].append(u)
    train_files, val_files = split_deployment_stratified(by_deploy, args.val_frac, args.seed)
    print(f"Split: {len(train_files)} train files, {len(val_files)} val files")
    df["split"] = df["audio_path"].map(lambda u: "val" if u in val_files else "train")

    # Final column order.
    cols = [
        "audio_path",
        "16khz_path",
        "32khz_path",
        "deployment",
        "xwav_subchunk_index",
        "begin_in_subchunk_s",
        "end_in_subchunk_s",
        "begin_in_file_s",
        "end_in_file_s",
        "begin_utc",
        "end_utc",
        "label",
        "label_is_strong",
        "implicit_negatives",
        "audit_name",
        "coarse_call_type",
        "species",
        "canonical_name",
        "gbifID",
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species_common",
        "license",
        "source_dataset",
    ]
    out_df = df.rename(
        columns={
            "subchunk_index": "xwav_subchunk_index",
            "begin_rel_subchunk": "begin_in_subchunk_s",
            "end_rel_subchunk": "end_in_subchunk_s",
        }
    )[cols + ["split"]]

    # ---- 8. Write CSVs ----
    def _save(d: pd.DataFrame, name: str) -> None:
        d = d.drop(columns=["split"])
        local = out_dir / name
        d.to_csv(local, index=False)
        print(f"  {name}: {len(d):,} rows -> {local}")
        if args.upload:
            subprocess.run(
                ["gsutil", "-q", "cp", str(local), f"{args.out_gcs}/{name}"],
                check=True,
                timeout=300,
            )
            print(f"    uploaded to {args.out_gcs}/{name}")

    _save(out_df, "pifsc_pipan_all.csv")
    _save(out_df[out_df["split"] == "train"].reset_index(drop=True), "pifsc_pipan_train.csv")
    _save(out_df[out_df["split"] == "val"].reset_index(drop=True), "pifsc_pipan_val.csv")

    labels_df = pd.DataFrame(
        [
            {
                "label": k,
                "description": v["description"],
                "species": v["species"],
                "coarse_call_type": v["coarse_call_type"],
                "is_biological": v["is_biological"],
            }
            for k, v in LABEL_VOCAB.items()
        ]
    )
    labels_local = out_dir / "pifsc_pipan_labels.csv"
    labels_df.to_csv(labels_local, index=False)
    print(f"  pifsc_pipan_labels.csv: {len(labels_df)} rows -> {labels_local}")
    if args.upload:
        subprocess.run(
            ["gsutil", "-q", "cp", str(labels_local), f"{args.out_gcs}/pifsc_pipan_labels.csv"],
            check=True,
            timeout=60,
        )
        subprocess.run(
            ["gsutil", "-q", "cp", str(xwav_cache), f"{args.out_gcs}/pifsc_pipan_xwav_index.csv"],
            check=True,
            timeout=120,
        )
        print(f"    uploaded labels + xwav_index to {args.out_gcs}/")

    # ---- 9. Print summary ----
    print("\n=== SUMMARY ===")
    print(f"Total events: {len(out_df):,}")
    print(f"Unique FLAC files: {out_df['audio_path'].nunique():,}")
    print(f"Pre-resampled 16 kHz coverage: {(out_df['16khz_path'] != '').mean() * 100:.1f}%")
    print(f"Pre-resampled 32 kHz coverage: {(out_df['32khz_path'] != '').mean() * 100:.1f}%")
    print("Label distribution (split=all):")
    print(out_df["label"].value_counts().to_string())
    print("Per-deployment counts:")
    print(out_df["deployment"].value_counts().to_string())


if __name__ == "__main__":
    main()
