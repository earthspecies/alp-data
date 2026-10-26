"""Format NIPS4Bplus (Morfi, Stowell & Pamuła), the temporal annotations of the
NIPS4B 2013 bird challenge training set.

687 short (1-5 s) 44.1 kHz WAV recordings from France, provided by
Biotope for the NIPS4B 2013 challenge. NIPS4Bplus (Figshare version 7) adds
per-event onset/duration annotations, labelled with the challenge's 87 classes
(species x call/song/drum of 77 bird, 9 insect and 1 amphibian classes) plus
``Unknown`` and ``Human``. The challenge test recordings have no annotations
and are not used.

The extracted challenge tarball and Figshare files were uploaded unchanged to
`RAW_DIR` (without the archives themselves). Of the 687 training recordings:

- 13 have no annotation file (6 ambiguous, 7 insects only, per the README).
- 5 have a 0-byte annotation file but a species in the weak labels, so they are
  treated as unannotated and dropped.
- 100 have an annotation file holding only a line break. They are the weak
  label ``Empty`` (background noise) files, so they are kept with no events.

That leaves 669 recordings. Scientific names in the species list are mapped to
accepted GBIF canonical names with `GBIFConverter` (after `NAME_CORRECTIONS`).
``Human`` becomes ``Homo sapiens``, ``Unknown`` stays ``Unknown``, and
``Pelophylax kl. grafi`` is kept as is (see `NON_GBIF_NAMES`). The original
label is kept as ``class_name``.

The audio is copied next to the manifests so ``audio_path`` stays relative to
them, then two versions of the ``all`` split are written and QC'd:

- ``all``: bird events only.
- ``all_with_nonbirds``: bird, insect, amphibian, ``Human`` and ``Unknown``
  events. Noise (the ``Empty`` class) is omitted.
"""

import os
import subprocess
from io import StringIO

import numpy as np
import pandas as pd

from alp_data.discover import GBIFConverter
from alp_data.io import anypath, audio_stereo_to_mono, filesystem, read_audio

RAW_DIR = "esp-data-ingestion/nips4bplus/raw"
OUT_DIR = "esp-data-ingestion/nips4bplus/formatted"
AUDIO_DIR = f"{RAW_DIR}/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/train"
ANNO_DIR = f"{RAW_DIR}/temporal_annotations_nips4b"
SPECIES_LIST_FP = f"{RAW_DIR}/nips4b_birdchallenge_espece_list.csv"
TARGET_FPS = {
    "all": f"gs://{OUT_DIR}/all.csv",
    "all_with_nonbirds": f"gs://{OUT_DIR}/all_with_nonbirds.csv",
}
# Species list names that `GBIFConverter` does not resolve as written
NAME_CORRECTIONS = {
    "lullula arborea": "Lullula arborea",
    "Regulus ignicapillus": "Regulus ignicapilla",
}
# Graf's hybrid frog is a klepton (P. perezi x P. ridibundus) with no GBIF entry
NON_GBIF_NAMES = {"Pelophylax kl. grafi"}
UNKNOWN_LABEL = "Unknown"
# Labels the annotators added that are not in the species list: label -> (Species, type)
EXTRA_LABELS = {"Unknown": (UNKNOWN_LABEL, "unknown"), "Human": ("Homo sapiens", "human")}
ST_COLUMNS = ["Begin Time (s)", "End Time (s)", "Species", "class_name"]

converter = GBIFConverter()
fs = filesystem("gcs")


def build_class_map() -> pd.DataFrame:
    """
    Map each class name to a GBIF canonical name and a sound source type.

    Returns
    -------
    pd.DataFrame
        Indexed by ``class_name``, with ``Species`` and ``type`` columns. Types
        are ``bird``, ``insect``, ``amphibian`` and ``noise`` from the species
        list, plus ``unknown`` and ``human`` from `EXTRA_LABELS`.

    Raises
    ------
    ValueError
        If a scientific name cannot be resolved by `GBIFConverter`.
    """
    with fs.open(SPECIES_LIST_FP) as f:
        species_list = pd.read_csv(f)

    rows = []
    for _, r in species_list.iterrows():
        sci_name = NAME_CORRECTIONS.get(r["Scientific_name"], r["Scientific_name"])
        if r["type"] == "noise" or sci_name in NON_GBIF_NAMES:
            species = sci_name
        else:
            species_info, matched = converter(sci_name)
            if not matched:
                raise ValueError(f"GBIF could not resolve {r['class name']} ({sci_name})")
            species = species_info["canonicalName"]
        rows.append({"class_name": r["class name"], "Species": species, "type": r["type"]})
    for class_name, (species, kind) in EXTRA_LABELS.items():
        rows.append({"class_name": class_name, "Species": species, "type": kind})
    return pd.DataFrame(rows).set_index("class_name")


def load_annotations(class_map: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Read the per-recording annotation files.

    Each file is headerless ``start,duration,label`` with CRLF line endings.
    0-byte files are skipped along with their recording; files holding only a
    line break are kept as recordings without events.

    Parameters
    ----------
    class_map : pd.DataFrame
        Output of `build_class_map`.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Events with ``Filename``, `ST_COLUMNS` and ``type`` columns, and the
        sorted audio file names of the annotated recordings.

    Raises
    ------
    ValueError
        If an annotation file contains a label missing from `class_map`.
    """
    anno_fps = sorted(fs.glob(f"{ANNO_DIR}/annotation_train*.csv"))
    contents = fs.cat(anno_fps)

    events = []
    audio_fns = []
    n_unannotated = 0
    for fp in anno_fps:
        raw = contents[fp]
        if not raw:
            n_unannotated += 1
            continue
        file_id = os.path.basename(fp).removeprefix("annotation_train").removesuffix(".csv")
        audio_fn = f"nips4b_birds_trainfile{file_id}.wav"
        audio_fns.append(audio_fn)
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            begin, duration, label = line.split(",")
            # Source times have 9 decimals; rounding drops float noise from the sum
            events.append(
                {
                    "Filename": audio_fn,
                    "Begin Time (s)": float(begin),
                    "End Time (s)": round(float(begin) + float(duration), 9),
                    "class_name": label.strip(),
                }
            )
    print(f"Skipping {n_unannotated} recordings with 0-byte annotation files")

    events = pd.DataFrame(events)
    unmapped = set(events["class_name"]) - set(class_map.index)
    if unmapped:
        raise ValueError(f"Labels missing from the species list: {sorted(unmapped)}")
    events = events.join(class_map, on="class_name")
    return events, audio_fns


def write_manifest(anno: pd.DataFrame, audio_fns: list[str], target_fp: str) -> None:
    """
    Write a manifest with one selection table per recording to `target_fp`.

    Recordings without events get an empty (header-only) selection table. The
    upload goes through `gsutil`; a failure propagates as
    `subprocess.CalledProcessError`.

    Parameters
    ----------
    anno : pd.DataFrame
        Formatted annotations with ``Filename`` and `ST_COLUMNS` columns.
    audio_fns : list[str]
        Audio file names, one manifest row each.
    target_fp : str
        GCS path of the manifest CSV.
    """
    info = {"audio_file_name": [], "audio_path": [], "selection_table": []}
    for audio_fn in audio_fns:
        info["audio_file_name"].append(audio_fn)
        info["audio_path"].append(os.path.join("audio", audio_fn))
        st = anno[anno["Filename"] == audio_fn].sort_values("Begin Time (s)", kind="stable")
        info["selection_table"].append(st[ST_COLUMNS].to_csv(sep="\t", index=False))

    out_fn = os.path.basename(target_fp)
    pd.DataFrame(info).to_csv(out_fn, index=False)
    subprocess.run(["gsutil", "cp", out_fn, target_fp], check=True)
    os.remove(out_fn)
    print(f"Wrote {len(audio_fns)} recordings, {len(anno)} events to {target_fp}")


def format_dataset() -> None:
    """
    Copy audio and write the ``all`` and ``all_with_nonbirds`` manifests.

    Noise events are dropped from both. ``all`` additionally drops all
    non-bird events.

    Writes go through `gsutil` (user credentials) since the default `gcsfs`
    credentials may only have read access to the bucket. A failed `gsutil`
    call propagates as `subprocess.CalledProcessError`.
    """
    class_map = build_class_map()
    events, audio_fns = load_annotations(class_map)
    noise = events["type"] == "noise"
    print(f"Dropping {noise.sum()} noise events")
    events = events[~noise]

    subprocess.run(
        ["gsutil", "-m", "cp", "-I", f"gs://{OUT_DIR}/audio/"],
        input="\n".join(f"gs://{AUDIO_DIR}/{fn}" for fn in audio_fns),
        text=True,
        check=True,
    )

    write_manifest(events, audio_fns, TARGET_FPS["all_with_nonbirds"])
    write_manifest(events[events["type"] == "bird"], audio_fns, TARGET_FPS["all"])


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
        st_end = float(st["End Time (s)"].max()) if not st.empty else 0.0
        if st_end > audio_end + 1e-6:
            problems.append(
                {"idx": i, "audio_path": row.get("audio_path", ""), "issue": "events_after_audio"}
            )

        species_in_st = list(st["Species"].unique())
        for species in species_in_st:
            if species == UNKNOWN_LABEL or species in NON_GBIF_NAMES:
                continue
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
        qc_fp = f"nips4bplus_{split}_qc_report.csv"
        qc_df.to_csv(qc_fp, index=False)
        print(f"Wrote QC report with {len(qc_df)} issues to: {qc_fp}")
