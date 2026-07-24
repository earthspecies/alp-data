"""Build the MammAlps Benchmark I (behavior) manifests + trimmed clips.

MammAlps is open on Zenodo (record 15040901, MIT). ``jobs/build_mammalps.sh``
downloads the ``mammalps_v1.zip`` once (a single large sequential download is
far more reliable on the cluster than thousands of range requests), and this
script selectively extracts — via stdlib ``zipfile`` (instant local seeks) —
only the Benchmark I metadata CSVs and the per-clip source files it needs.

Verified Benchmark I layout (from the zip):
- ``benchmark_1/metadata/{train,val,test}.csv`` — columns
  ``video_path,start_s,end_s,activity,actions,species`` (headed). test.csv has
  1,244 clips. ``video_path`` is relative to ``benchmark_1/clips/`` and points
  at a per-camera-view mp4; each row is a temporal segment ``[start_s, end_s]``.
- ``benchmark_1/clips/<event>/<event>_c<view>.mp4`` — VIDEO-ONLY source clips.
- ``benchmark_1/audios/<event>/<event>_c<view>.wav`` — the aligned AUDIO track,
  stored as a SEPARATE parallel file (the mp4s carry no audio stream).

For each split row, the source mp4 + its parallel wav are extracted and
ffmpeg-**muxed + trimmed** to ``[start_s, end_s]`` into a single audiovisual
``<out>/video/<asset_id>.mp4`` (video + audio streams), where
``asset_id = <mp4-stem>_<start_ms>_<end_ms>`` — SSW60-style clips carrying both
modalities so vision-only / audio-only / vision+audio all derive from one file.
Clips whose wav is missing are written video-only.

Usage (see jobs/build_mammalps.sh):
    uv run python scripts/data_preprocessing_scripts/mammalps/build_mammalps.py \
        --zip /scratch/$USER/mammalps/mammalps_v1.zip \
        --labels-json /scratch/$USER/mammalps/labels_mapping_b1.json \
        --out /scratch/$USER/mammalps/staging --splits test
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

GCS_ROOT_DEFAULT = "gs://esp-data-ingestion/mammalps/v0.1.0"
_METADATA_SUFFIX = "benchmark_1/metadata/{split}.csv"
_CLIPS_KEY = "benchmark_1/clips/"
_AUDIOS_KEY = "benchmark_1/audios/"
_OUT_COLUMNS = [
    "asset_id", "modality", "activity", "activity_label",
    "species", "actions", "split", "video_path",
]


def load_activity_label(labels_json: Path) -> dict[str, int]:
    """Load the activity name -> id map from ``labels_mapping_b1.json``.

    Returns
    -------
    dict[str, int]
        Activity name -> class id (11 activities).

    Raises
    ------
    FileNotFoundError
        If the labels JSON is not found.
    """
    if not labels_json.exists():
        raise FileNotFoundError(f"labels_mapping_b1.json not found at {labels_json}")
    with open(labels_json) as f:
        return json.load(f)["activities"]


def _index(zf: zipfile.ZipFile, key: str) -> dict[str, str]:
    """Index archive members whose path contains ``key`` by their post-key tail.

    Returns
    -------
    dict[str, str]
        Mapping ``<relative path after key>`` -> full member name.
    """
    out: dict[str, str] = {}
    for n in zf.namelist():
        i = n.find(key)
        if i != -1 and not n.endswith("/"):
            out[n[i + len(key):]] = n
    return out


def _norm_actions(raw: str) -> str:
    """Normalize a ``;``-separated action string to a ``, ``-joined set.

    Returns
    -------
    str
        Actions joined by ``", "`` with ``none`` / empties dropped.
    """
    acts = [a.strip() for a in str(raw).replace(";", ",").split(",")]
    return ", ".join(sorted({a for a in acts if a and a.lower() != "none"}))


def _mux_trim(
    video_src: Path, audio_src: Path | None, start_s: float, end_s: float, dst: Path
) -> bool:
    """ffmpeg mux (video + optional wav audio) and trim to ``[start_s, end_s]``.

    Returns
    -------
    bool
        True on success, False if ffmpeg failed.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", str(video_src)]
    if audio_src is not None:
        cmd += ["-i", str(audio_src), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac"]
    cmd += ["-ss", f"{start_s}", "-to", f"{end_s}", "-c:v", "libx264", "-preset", "veryfast"]
    cmd += [str(dst)]
    return subprocess.run(cmd, capture_output=True).returncode == 0 and dst.exists()


def build_split(
    zf: zipfile.ZipFile,
    split: str,
    activity_label: dict[str, int],
    clips_idx: dict[str, str],
    audios_idx: dict[str, str],
    out: Path,
    gcs_root: str,
) -> pd.DataFrame:
    """Extract + trim the clips for one split and build its manifest.

    Returns
    -------
    pd.DataFrame
        The manifest for the split.

    Raises
    ------
    RuntimeError
        If the metadata CSV is missing columns, an activity is unmapped, or too
        many clips fail to extract/trim.
    """
    meta_member = next(
        (n for n in zf.namelist() if n.endswith(_METADATA_SUFFIX.format(split=split))), None
    )
    if meta_member is None:
        raise RuntimeError(f"{_METADATA_SUFFIX.format(split=split)} not found in archive")
    df = pd.read_csv(io.BytesIO(zf.read(meta_member)), keep_default_na=False, na_values=[])
    expected = {"video_path", "start_s", "end_s", "activity", "actions", "species"}
    missing_cols = expected - set(df.columns)
    if missing_cols:
        raise RuntimeError(f"{split}.csv missing {missing_cols}; found {list(df.columns)}")

    video_dir = out / "video"
    tmp_root = Path(tempfile.mkdtemp())
    src_cache: dict[str, tuple[Path, Path | None]] = {}
    rows = []
    failed = 0
    total = len(df)
    for i, (_, r) in enumerate(df.iterrows()):
        activity = str(r["activity"]).strip()
        if activity not in activity_label:
            raise RuntimeError(f"{split}: activity {activity!r} not in {sorted(activity_label)}")
        rel_mp4 = str(r["video_path"]).strip()
        rel_wav = rel_mp4.rsplit(".", 1)[0] + ".wav"
        start_s, end_s = float(r["start_s"]), float(r["end_s"])
        aid = f"{Path(rel_mp4).stem}_{round(start_s * 1000)}_{round(end_s * 1000)}"
        dst = video_dir / f"{aid}.mp4"
        if not dst.exists():
            if rel_mp4 not in src_cache:
                if rel_mp4 not in clips_idx:
                    failed += 1
                    continue
                v = tmp_root / Path(rel_mp4).name
                v.write_bytes(zf.read(clips_idx[rel_mp4]))
                a = None
                if rel_wav in audios_idx:
                    a = tmp_root / Path(rel_wav).name
                    a.write_bytes(zf.read(audios_idx[rel_wav]))
                src_cache[rel_mp4] = (v, a)
            v, a = src_cache[rel_mp4]
            if not _mux_trim(v, a, start_s, end_s, dst):
                failed += 1
                continue
        rows.append(
            {
                "asset_id": aid,
                "modality": "video",
                "activity": activity,
                "activity_label": activity_label[activity],
                "species": str(r["species"]).strip(),
                "actions": _norm_actions(r["actions"]),
                "split": split,
                "video_path": f"{gcs_root}/video/{aid}.mp4",
            }
        )
        if (i + 1) % 100 == 0:
            print(f"  {split}: {i + 1}/{total} ({len(rows)} ok, {failed} failed)", flush=True)
    if failed > 0.05 * max(1, total):
        raise RuntimeError(f"{split}: {failed}/{total} clips failed to extract/trim.")
    print(f"{split}: {len(rows)} clips ({failed} failed)", flush=True)
    return pd.DataFrame(rows, columns=_OUT_COLUMNS)


def main() -> None:
    """Run the MammAlps Benchmark I selective extract + manifest build."""
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True, help="Path to the downloaded mammalps_v1.zip.")
    p.add_argument("--labels-json", required=True, help="Path to labels_mapping_b1.json.")
    p.add_argument("--out", required=True, help="Staging output dir.")
    p.add_argument("--gcs-root", default=GCS_ROOT_DEFAULT)
    p.add_argument("--splits", nargs="+", default=["test"],
                   help="Splits to build/stage (default: test only, to bound size).")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    activity_label = load_activity_label(Path(args.labels_json))
    print(f"{len(activity_label)} activities: {sorted(activity_label)}", flush=True)

    frames = []
    with zipfile.ZipFile(args.zip) as zf:
        print("indexing archive members ...", flush=True)
        clips_idx = _index(zf, _CLIPS_KEY)
        audios_idx = _index(zf, _AUDIOS_KEY)
        print(f"indexed {len(clips_idx)} clips, {len(audios_idx)} audios", flush=True)
        for split in args.splits:
            df = build_split(zf, split, activity_label, clips_idx, audios_idx, out, args.gcs_root)
            df.to_csv(out / f"mammalps_{split}.csv", index=False)
            print(f"mammalps_{split}.csv: {len(df)} clips", flush=True)
            frames.append(df)
    alldf = pd.concat(frames, ignore_index=True)
    alldf.to_csv(out / "mammalps_all.csv", index=False)
    print(f"mammalps_all.csv: {len(alldf)} clips\n\nDONE.", flush=True)


if __name__ == "__main__":
    main()
