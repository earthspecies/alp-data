"""
Build a per-recording, WABAD-shape aggregation of BirdSet's SSW-test split.

Reads the existing strongly-labeled per-event manifest
    gs://esp-ml-datasets/birdset/v0.1.0/raw/SSW_test.csv      (50,760 events)
and emits a per-recording manifest
    gs://esp-data-ingestion/birdset/v0.1.0/raw/SSW_test_aggregated.csv
(285 rows) with the following WABAD-compatible columns:

    fn, audio_fp, audio_duration, subdataset, selection_table, 16khz_path,
    32khz_path, audio_path, relative_path, gcs_path

The ``selection_table`` cell is a tab-separated TSV blob with header:

    Selection  View  Channel  Begin Time (s)  End Time (s)
    Low Freq (Hz)  High Freq (Hz)  Species

(matching the WABAD ``window_annotations`` / ``annotation_features``
chain transforms exactly).

``audio_duration`` is the true full-recording duration in seconds, probed
via ``soundfile.info`` on the 16 kHz pre-resampled WAVs cached locally.
(The 16k WAVs are much smaller than the original OGGs and have an easy
header, so probing is fast.)

The relative path columns (``audio_fp``, ``audio_path``, ``16khz_path``,
``32khz_path``) are kept identical to the upstream SSW_test.csv layout, so
``BirdSet`` (which hardcodes its data_root to
``gs://esp-ml-datasets/birdset/v0.1.0/raw/``) will resolve audio correctly
even though our new manifest CSV lives in a different bucket.
"""

from __future__ import annotations

import argparse
import io
import os
import wave

import pandas as pd
from google.cloud import storage

SRC = "gs://esp-ml-datasets/birdset/v0.1.0/raw/SSW_test.csv"
DST_GCS = "gs://esp-data-ingestion/birdset/v0.1.0/raw/SSW_test_aggregated.csv"
BIRDSET_BUCKET = "esp-ml-datasets"
BIRDSET_PREFIX = "birdset/v0.1.0/raw/"

ST_COLUMNS = [
    "Selection",
    "View",
    "Channel",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Species",
]


def probe_durations(rec_ids: list[str], client: storage.Client) -> dict[str, float]:
    """Probe the true recording duration of each SSW-test soundscape.

    Reads the first 1 KiB of each 16 kHz WAV via a GCS byte-range request and
    parses the declared ``nframes`` field from the WAV header using Python's
    ``wave`` stdlib (which trusts the header rather than counting actual
    audio bytes — fast and correct without downloading the body).

    Returns
    -------
    dict[str, float]
        Mapping from recording id (``SSW_NNN_YYYYMMDD_HHMMSSZ``) to
        recording duration in seconds.
    """
    bucket = client.bucket(BIRDSET_BUCKET)
    durations: dict[str, float] = {}
    for i, rid in enumerate(rec_ids):
        key = f"{BIRDSET_PREFIX}audio_16k/SSW/test/{rid}.wav"
        blob = bucket.blob(key)
        data = blob.download_as_bytes(start=0, end=1023)
        w = wave.open(io.BytesIO(data))
        durations[rid] = w.getnframes() / float(w.getframerate())
        w.close()
        if (i + 1) % 50 == 0:
            print(f"  probed {i + 1}/{len(rec_ids)}")
    return durations


def build_selection_table_tsv(group: pd.DataFrame) -> str:
    """Build a WABAD-shape TSV selection-table blob from per-event rows.

    Returns
    -------
    str
        Tab-separated values text with ``ST_COLUMNS`` header and one event per
        row, sorted by begin time.
    """
    g = group.sort_values("start_time").reset_index(drop=True)
    out = pd.DataFrame(
        {
            "Selection": range(1, len(g) + 1),
            "View": "Spectrogram 1",
            "Channel": 1,
            "Begin Time (s)": g["start_time"].astype(float),
            "End Time (s)": g["end_time"].astype(float),
            "Low Freq (Hz)": g["low_freq"].astype(float),
            "High Freq (Hz)": g["high_freq"].astype(float),
            # Prefer canonical_name (GBIF-linked); fall back to species
            "Species": g["canonical_name"].where(
                g["canonical_name"].astype(str).str.len() > 0, g["species"]
            ),
        },
        columns=ST_COLUMNS,
    )
    return out.to_csv(sep="\t", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=SRC)
    p.add_argument(
        "--out-local",
        default="/mnt/home/birdset_ssw_agg/SSW_test_aggregated.csv",
    )
    p.add_argument("--out-gcs", default=DST_GCS)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out_local), exist_ok=True)

    print(f"[load] {args.src}")
    df = pd.read_csv(args.src, keep_default_na=False)
    n_ev = len(df)
    n_rec = df["id"].nunique()
    n_sp = df["species"].nunique()
    print(f"  {n_ev} events / {n_rec} unique recordings / {n_sp} species")

    print("\n[probe] true recording durations via 16k WAV headers")
    rec_ids = sorted(df["id"].unique())
    client = storage.Client()
    durations = probe_durations(rec_ids, client)
    dur_series = pd.Series(durations)
    print(
        f"  probed {len(durations)} recordings; median dur={dur_series.median():.0f}s "
        f"min={dur_series.min():.0f}s max={dur_series.max():.0f}s"
    )

    print("\n[aggregate] grouping events by recording id")
    rows: list[dict] = []
    for rid, group in df.groupby("id"):
        st_tsv = build_selection_table_tsv(group)
        first = group.iloc[0]
        rows.append(
            {
                "fn": rid,
                "audio_fp": first["audio_path"],
                "audio_duration": durations[rid],
                "subdataset": "SSW",
                "selection_table": st_tsv,
                "relative_path": first["relative_path"],
                "gcs_path": first["gcs_path"],
                "audio_path": first["audio_path"],
                "16khz_path": f"audio_16k/SSW/test/{rid}.wav",
                "32khz_path": f"audio_32k/SSW/test/{rid}.wav",
                "lat": first.get("lat", ""),
                "long": first.get("long", ""),
                "source": first.get("source", ""),
                "license": first.get("license", ""),
                "n_events": len(group),
            }
        )
    out_df = pd.DataFrame(rows)
    print(f"  aggregated: {len(out_df)} recordings")
    ev_med = int(out_df["n_events"].median())
    ev_mean = int(out_df["n_events"].mean())
    ev_max = int(out_df["n_events"].max())
    print(f"  events stats: median={ev_med} mean={ev_mean} max={ev_max}")

    print(f"\n[write] local: {args.out_local}")
    out_df.to_csv(args.out_local, index=False)
    print(f"  size: {os.path.getsize(args.out_local) / 1e6:.1f} MB")

    if args.dry_run:
        print("[dry-run] not uploading.")
        return

    print(f"[upload] {args.out_gcs}")
    rc = os.system(f"gsutil -q cp {args.out_local} {args.out_gcs}")
    if rc != 0:
        raise RuntimeError(f"gsutil cp failed rc={rc}")
    print("Done.")


if __name__ == "__main__":
    main()
