"""Format the Sierra Nevada soundscapes (Kahl, Wood et al.).

33 hour-long soundscapes (32 kHz FLAC) from the Lassen and Plumas National
Forests in the Sierra Nevada, California, USA, fully annotated with
time-frequency boxes for 56 bird species. Labels are 2021 eBird species codes;
unlike other collections in this series, there are no ``????`` (unidentifiable)
labels.

eBird codes are mapped to scientific names with the 2021 eBird taxonomy, then to
accepted GBIF canonical names with `GBIFConverter`. Zero-length events (end time
<= start time) are dropped.

The raw FLACs are losslessly re-encoded with ffmpeg (see `reencode_audio`) next
to the manifest so ``audio_path`` stays relative to it, then the ``all`` split
is written and QC'd.
"""

import json
import os
import subprocess
import tempfile
from io import StringIO

import numpy as np
import pandas as pd

from alp_data.discover import GBIFConverter
from alp_data.io import anypath, audio_stereo_to_mono, filesystem, read_audio

RAW_DIR = "esp-data-ingestion/western_us_soundscapes/raw"
OUT_DIR = "esp-data-ingestion/western_us_soundscapes/formatted"
EBIRD_TO_SCI_FP = "esp-data-ingestion/taxonomy/ebird_taxonomy_cached/ebird_to_sci_name_2021.json"
TARGET_FPS = {
    "all": f"gs://{OUT_DIR}/all.csv",
}

converter = GBIFConverter()
fs = filesystem("gcs")


def build_species_map(ebird_codes: list[str]) -> dict[str, str]:
    """
    Map eBird codes to GBIF canonical names.

    Parameters
    ----------
    ebird_codes : list[str]
        eBird species codes present in the annotations.

    Returns
    -------
    dict[str, str]
        eBird code to GBIF canonical name.

    Raises
    ------
    ValueError
        If a scientific name cannot be resolved by `GBIFConverter`.
    """
    with fs.open(EBIRD_TO_SCI_FP) as f:
        ebird_to_sci = json.load(f)

    species_map = {}
    for code in ebird_codes:
        species_info, matched = converter(ebird_to_sci[code])
        if not matched:
            raise ValueError(f"GBIF could not resolve {code} ({ebird_to_sci[code]})")
        species_map[code] = species_info["canonicalName"]
    return species_map


def reencode_audio(audio_fps: list[str]) -> None:
    """
    Re-encode the raw FLACs with ffmpeg and upload them to ``OUT_DIR/audio``.

    5 of the 33 recordings contain stretches of full-scale clipping where
    libsndfile (used by `read_audio`) stops decoding ("flac decoder lost
    sync", "unknown error in flac decoder"). ffmpeg decodes them in full, and
    its re-encode is lossless and readable by libsndfile. Every recording is
    re-encoded so all are handled the same way.

    A failed `gsutil` or `ffmpeg` call propagates as
    `subprocess.CalledProcessError`.

    Parameters
    ----------
    audio_fps : list[str]
        GCS paths (without ``gs://``) of the raw FLACs.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir = os.path.join(tmp_dir, "raw")
        out_dir = os.path.join(tmp_dir, "audio")
        os.makedirs(raw_dir)
        os.makedirs(out_dir)
        subprocess.run(
            ["gsutil", "-m", "cp", *[f"gs://{fp}" for fp in audio_fps], raw_dir], check=True
        )
        out_fps = []
        for fp in audio_fps:
            src = os.path.join(raw_dir, os.path.basename(fp))
            out_fps.append(os.path.join(out_dir, os.path.basename(fp)))
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", src, "-c:a", "flac", out_fps[-1]],
                check=True,
            )
            os.remove(src)
        subprocess.run(["gsutil", "-m", "cp", *out_fps, f"gs://{OUT_DIR}/audio/"], check=True)


def write_manifest(anno: pd.DataFrame, audio_fns: list[str], target_fp: str) -> None:
    """
    Write a manifest with one selection table per recording to `target_fp`.

    Recordings without events get an empty (header-only) selection table. The
    upload goes through `gsutil`; a failure propagates as
    `subprocess.CalledProcessError`.

    Parameters
    ----------
    anno : pd.DataFrame
        Formatted annotations with ``Filename`` and selection table columns.
    audio_fns : list[str]
        Audio file names, one manifest row each.
    target_fp : str
        GCS path of the manifest CSV.
    """
    info = {"audio_file_name": [], "audio_path": [], "selection_table": []}
    for audio_fn in audio_fns:
        info["audio_file_name"].append(audio_fn)
        info["audio_path"].append(os.path.join("audio", audio_fn))
        st = anno[anno["Filename"] == audio_fn].sort_values("Begin Time (s)")
        st = st[["Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)", "Species"]]
        info["selection_table"].append(st.to_csv(sep="\t", index=False))

    out_fn = os.path.basename(target_fp)
    pd.DataFrame(info).to_csv(out_fn, index=False)
    subprocess.run(["gsutil", "cp", out_fn, target_fp], check=True)
    os.remove(out_fn)
    print(f"Wrote {len(audio_fns)} recordings, {len(anno)} events to {target_fp}")


def format_dataset() -> None:
    """
    Re-encode audio and write the ``all`` manifest, without zero-length events.

    Writes go through `gsutil` (user credentials) since the default `gcsfs`
    credentials may only have read access to the bucket. A failed `gsutil`
    or `ffmpeg` call propagates as `subprocess.CalledProcessError`.
    """
    raw_anno = pd.read_csv(f"gs://{RAW_DIR}/annotations.csv")
    zero_length = raw_anno["End Time (s)"] <= raw_anno["Start Time (s)"]
    print(f"Dropping {zero_length.sum()} zero-length events")
    raw_anno = raw_anno[~zero_length].copy()
    species_map = build_species_map(sorted(raw_anno["Species eBird Code"].unique()))
    raw_anno["Species"] = raw_anno["Species eBird Code"].map(species_map)
    raw_anno["Begin Time (s)"] = raw_anno["Start Time (s)"]

    audio_fps = sorted(fs.glob(f"{RAW_DIR}/*.flac"))
    reencode_audio(audio_fps)

    audio_fns = [os.path.basename(fp) for fp in audio_fps]
    write_manifest(raw_anno, audio_fns, TARGET_FPS["all"])


def iterate_qc(
    df: pd.DataFrame,
    data_root: str | None = None,
) -> pd.DataFrame:
    """
    Iterate dataset and run basic QC. Returns a DataFrame of issues.

    Parameters
    ----------
    df : pd.DataFrame
        Manifest with ``audio_path`` and ``selection_table`` columns.
    data_root : str | None
        Directory that ``audio_path`` is relative to.

    Returns
    ---------
    pd.DataFrame
        DataFrame of issues
    """
    problems = []
    for i, row in df.iterrows():
        print(f"{i} / {len(df)}")
        audio_path = (
            anypath(data_root) / row["audio_path"] if data_root else anypath(row["audio_path"])
        )

        try:
            audio, sr = read_audio(audio_path)
        except Exception as e:
            problems.append(
                {
                    "idx": i,
                    "audio_path": row.get("audio_path", ""),
                    "issue": f"read_audio_failed: {e}",
                }
            )
            continue

        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        # selection table
        try:
            st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
        except Exception as e:
            problems.append(
                {
                    "idx": i,
                    "audio_path": row.get("audio_path", ""),
                    "issue": f"st_parse_failed: {e}",
                }
            )
            continue

        # QC checks
        if audio.size < 10:
            problems.append(
                {"idx": i, "audio_path": row.get("audio_path", ""), "issue": "too_short"}
            )

        if np.any(np.isnan(audio)):
            problems.append(
                {"idx": i, "audio_path": row.get("audio_path", ""), "issue": "nan_in_audio"}
            )

        if np.all(audio == 0):
            problems.append(
                {"idx": i, "audio_path": row.get("audio_path", ""), "issue": "all_zeros"}
            )

        audio_end = len(audio) / float(sr)
        st_end = float(st["Begin Time (s)"].max()) if not st.empty else 0.0
        if st_end > audio_end + 1e-6:
            problems.append(
                {"idx": i, "audio_path": row.get("audio_path", ""), "issue": "events_after_audio"}
            )

        species_in_st = list(st["Species"].unique())
        for species in species_in_st:
            speciesinfo, success = converter(species)
            if not success:
                problems.append(
                    {
                        "idx": i,
                        "audio_path": row.get("audio_path", ""),
                        "issue": f"unrecognized species {species}",
                    }
                )
            elif not speciesinfo["canonicalName"] == species:
                problems.append(
                    {
                        "idx": i,
                        "audio_path": row.get("audio_path", ""),
                        "issue": f"non-gbif species {species}",
                    }
                )

    return pd.DataFrame(problems)


if __name__ == "__main__":
    # 1) Format
    format_dataset()

    # 2) Iterate QC
    for split, target_fp in TARGET_FPS.items():
        print(f"QC {split}")
        df = pd.read_csv(target_fp)
        data_root = anypath(target_fp).parent
        qc_df = iterate_qc(df, data_root)
        qc_fp = f"sierra_nevada_{split}_qc_report.csv"
        qc_df.to_csv(qc_fp, index=False)
        print(f"Wrote QC report with {len(qc_df)} issues to: {qc_fp}")
