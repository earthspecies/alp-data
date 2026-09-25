"""
Script to pre-process subsegmentation dataset provided by Logan James

Each .not.mat file corresponds to a segment of audio
The existence of a .not.mat file means that an annotator assessed the file.
'audio_sr' is the sampling rate.

'labels' will be a list of labels assigned to syllables within an audio clip
'onsets' is a list of syllable onset times (ms) from the beginning of the file
'offsets' is a list of syllable offset times (ms) from the beginning of the file

These are parallel lists.

labels:
'a' indicates a syllable that is the beginning of a song
(we define as at least 500 ms silence before)
'z' indicates a syllable that is the end of a song (we define as at least 500 ms silence after)
's' indicates all other syllables

'0', '-' or any label other than 'a', 's', or 'z' DOES NOT INDICATE A SYLLABLE.
These may be bits of noise outside the song that were automatically segmented,
"garbled" syllables from artifacts, or sections of the song that were deemed unlabelable

If a .not.mat file does not contain any 'a', 's', or 'z' labels,
the annotator deemed the file unlabelable.
If an audio segment does not have a corresponding .not.mat file, the audio segment was not viewed.

NOTE: There was one duplicate file in the original dataset that has since been removed.
So if re-generating the dataset, the hash will change.
"""

import os
import tempfile
from io import StringIO
from typing import Dict

import numpy as np
import pandas as pd
import requests
import scipy.io
import soundfile as sf

from alp_data.io import anypath, audio_stereo_to_mono, filesystem, read_audio

BASE_PATH = "gs://subsegmentation/xeno_canto_annotations"
AUDIO_PATH_REL_BASE = "segments_all_together"
ANNOTATIONS_PATH = (
    "gs://subsegmentation/xeno_canto_annotations/syllable_annotations_from_evsonganaly"
)

SPECIES_LABEL_FIX = {}

SINGLE_SONG_GCS_PATH = "gs://subsegmentation/single_song"
SINGLE_SONG_AUDIO_SUBPATH = "audio"


def _taxonomy_lookup(species_name: str, base_url: str | None) -> Dict[str, str]:
    """
    Return a dict with keys: genus, family, order, species_common.
    If base_url is None, returns empty values (identity mapping).

    Returns
    --------
    Dict with keys: genus, family, order
    """
    species_name = SPECIES_LABEL_FIX.get(species_name, species_name)

    if not base_url:
        # Best-effort local mapping only for Species; leave others blank
        return {
            "genus": "",
            "family": "",
            "order": "",
            "species_common": "",
            "species": species_name,
        }

    r = requests.get(f"{base_url}/taxonomy/{species_name}")
    if r.status_code != 200:
        print(f"Could not locate {species_name}")
        return {
            "genus": "",
            "family": "",
            "order": "",
            "species_common": "",
            "species": species_name,
        }
    j = r.json()
    return {
        "genus": j.get("genus", "") or "",
        "family": j.get("family", "") or "",
        "order": j.get("order", "") or "",
        "species_common": j.get("species_common", "") or "",
        "species": species_name,
    }


def all_split() -> None:
    """
    Process the dataset into one "all" info file
    """

    # Get annotations locally
    temp_dir = "subsegmentation_annots_raw_temp"
    os.makedirs(temp_dir, exist_ok=True)
    fs = filesystem("gcs")
    annotation_files = sorted(fs.glob(os.path.join(ANNOTATIONS_PATH, "*.mat")))

    info = {
        "audio_file_name": [],
        "audio_path": [],
        "selection_table": [],
        "pass_qc": [],
        "Species": [],
        "Genus": [],
        "Order": [],
        "Family": [],
    }

    print("Processing annotations")

    taxonomy_cache = {}
    for k, annotation_file in enumerate(annotation_files):
        if k % 100 == 0:
            print(f"{k} / {len(annotation_files)}")

        fn = os.path.basename(annotation_file)
        raw_fp = os.path.join(temp_dir, fn)
        fs.get(annotation_file, raw_fp)
        mat = scipy.io.loadmat(raw_fp)

        if len(mat["labels"]) == 0:
            continue
        annots = [x for x in str(mat["labels"][0])]
        offsets = [float(mat["offsets"][i][0]) for i in range(len(mat["offsets"]))]
        onsets = [float(mat["onsets"][i][0]) for i in range(len(mat["onsets"]))]

        species = " ".join(fn.split("_")[3].split("-")[:2])
        if species in taxonomy_cache:
            taxonomy_GBIF = taxonomy_cache[species]
        else:
            taxonomy_GBIF = _taxonomy_lookup(species, "http://gagan-dev:8000")
            if species == "Chloris chloris":
                # patch for taxonomy app thinking this bird is a type of grass
                taxonomy_GBIF["family"] = "Fringillidae"
                taxonomy_GBIF["order"] = "Passeriformes"

            taxonomy_cache[species] = taxonomy_GBIF

        st = {
            "Begin Time (s)": [],
            "End Time (s)": [],
            "Annotation": [],
            "Species": [],
            "Genus": [],
            "Order": [],
            "Family": [],
        }

        for onset, offset, anno in zip(onsets, offsets, annots, strict=False):
            if anno not in ["a", "s", "z"]:
                continue
            st["Begin Time (s)"].append(onset / 1000)
            st["End Time (s)"].append(offset / 1000)
            st["Annotation"].append(anno)
            st["Species"].append(taxonomy_GBIF["species"])
            st["Order"].append(taxonomy_GBIF["order"])
            st["Family"].append(taxonomy_GBIF["family"])
            st["Genus"].append(taxonomy_GBIF["genus"])

        st = pd.DataFrame(st)
        pass_qc = len(st) > 0

        audio_fn = fn.replace(".not.mat", "").replace(".not(1).mat", "")
        audio_path = os.path.join(AUDIO_PATH_REL_BASE, audio_fn)

        info["audio_file_name"].append(audio_fn)
        info["audio_path"].append(audio_path)
        info["selection_table"].append(st.to_csv(sep="\t", index=False))
        info["pass_qc"].append(pass_qc)
        info["Species"].append(taxonomy_GBIF["species"])
        info["Order"].append(taxonomy_GBIF["order"])
        info["Family"].append(taxonomy_GBIF["family"])
        info["Genus"].append(taxonomy_GBIF["genus"])

    info = pd.DataFrame(info)
    info.to_csv("all.csv", index=False)


def train_val_test_split() -> None:
    """
    Splits data from all.csv into train, val, test splits
    """

    all_data = pd.read_csv("all.csv")

    # form test data
    rng = np.random.default_rng(333)

    species_nonpasserine = sorted(
        all_data[all_data["Order"] != "Passeriformes"]["Species"].unique()
    )  # there are 99
    passerine_species = sorted(
        all_data[all_data["Order"] == "Passeriformes"]["Species"].unique()
    )  # there are 735
    passerine_species = rng.permutation(passerine_species)

    n_passerine_species_test = 50
    n_passerine_species_val = 50

    passerine_species_test = list(passerine_species[:n_passerine_species_test])
    passerine_species_val = list(
        passerine_species[
            n_passerine_species_test : n_passerine_species_test + n_passerine_species_val
        ]
    )
    passerine_species_train = list(
        passerine_species[n_passerine_species_test + n_passerine_species_val :]
    )

    species_test = species_nonpasserine + passerine_species_test
    species_val = passerine_species_val
    species_train = passerine_species_train

    data_train = all_data[all_data["Species"].isin(species_train)].copy().reset_index(drop=True)
    data_val = all_data[all_data["Species"].isin(species_val)].copy().reset_index(drop=True)
    data_test = all_data[all_data["Species"].isin(species_test)].copy().reset_index(drop=True)

    data_train.to_csv("train.csv", index=False)
    data_val.to_csv("val.csv", index=False)
    data_test.to_csv("test.csv", index=False)

    os.system("gsutil -m cp -r train.csv gs://subsegmentation/xeno_canto_annotations")
    os.system("gsutil -m cp -r val.csv gs://subsegmentation/xeno_canto_annotations")
    os.system("gsutil -m cp -r test.csv gs://subsegmentation/xeno_canto_annotations")
    os.system("gsutil -m cp -r all.csv gs://subsegmentation/xeno_canto_annotations")


def _extract_songs(st: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Split a selection table into individual songs.

    A song is a contiguous sequence of syllables starting with 'a' and ending
    with 'z'. Middle syllables ('s') are included. Incomplete songs (missing
    either endpoint) are discarded. Assumes rows are sorted by Begin Time (s).

    Returns
    ---------
    list of selection tables (pd dataframes), each is an individual song
    """
    songs = []
    current: list = []
    in_song = False

    for _, row in st.iterrows():
        anno = row["Annotation"]
        if anno == "a":
            current = [row]
            in_song = True
        elif anno in ("s", "z") and in_song:
            current.append(row)
            if anno == "z":
                songs.append(pd.DataFrame(current).reset_index(drop=True))
                current = []
                in_song = False

    return songs


def single_song() -> None:
    """
    For each split (all, train, val, test), create a single-song variant where
    each data item is one song extracted from a multi-song recording.

    Song boundaries are defined by 'a' (song start) and 'z' (song end)
    annotations. Output audio is trimmed from the Begin Time of the 'a'
    syllable to the End Time of the 'z' syllable. Selection table times are
    re-zeroed to be relative to the song start.

    Audio files are generated once from the 'all' split and uploaded in one
    parallel gsutil call. The train/val/test CSVs are derived by filtering
    all.csv using source_audio_file_name, so no audio file is written twice.

    Results are saved under gs://subsegmentation/single_song/.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_dir = os.path.join(temp_dir, SINGLE_SONG_AUDIO_SUBPATH)
        os.makedirs(audio_dir, exist_ok=True)

        # ── Step 1: process 'all' split, write each audio file exactly once ──
        print("\nProcessing all split")
        df_all = pd.read_csv(f"{BASE_PATH}/all.csv")

        new_info: dict[str, list] = {
            "audio_file_name": [],
            "audio_path": [],
            "selection_table": [],
            "pass_qc": [],
            "Species": [],
            "Genus": [],
            "Order": [],
            "Family": [],
            "source_audio_file_name": [],
        }

        for i, row in df_all.iterrows():
            if i % 100 == 0:
                print(f"  {i} / {len(df_all)}")

            if not row["pass_qc"]:
                continue

            try:
                st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
            except Exception:
                continue

            if st.empty:
                continue

            st = st.sort_values("Begin Time (s)").reset_index(drop=True)
            songs = _extract_songs(st)
            if not songs:
                continue

            audio_path = anypath(BASE_PATH) / row["audio_path"]
            try:
                audio, sr = read_audio(audio_path)
                audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
            except Exception as e:
                print(f"  Skipping {row['audio_path']}: {e}")
                continue

            orig_stem = os.path.splitext(row["audio_file_name"])[0]

            for song_idx, song_st in enumerate(songs):
                song_start = float(song_st["Begin Time (s)"].min())
                song_end = float(song_st["End Time (s)"].max())

                start_sample = max(0, int(song_start * sr))
                end_sample = min(len(audio), int(song_end * sr))
                song_audio = audio[start_sample:end_sample]

                if len(song_audio) < 10:
                    continue

                # Re-zero selection table times relative to song start
                song_st = song_st.copy()
                song_st["Begin Time (s)"] = song_st["Begin Time (s)"] - song_start
                song_st["End Time (s)"] = song_st["End Time (s)"] - song_start

                new_audio_fn = f"{orig_stem}_song{song_idx}.wav"
                local_wav = os.path.join(audio_dir, new_audio_fn)
                sf.write(local_wav, song_audio, sr)

                new_info["audio_file_name"].append(new_audio_fn)
                new_info["audio_path"].append(os.path.join(SINGLE_SONG_AUDIO_SUBPATH, new_audio_fn))
                new_info["selection_table"].append(song_st.to_csv(sep="\t", index=False))
                new_info["pass_qc"].append(True)
                new_info["Species"].append(row["Species"])
                new_info["Genus"].append(row["Genus"])
                new_info["Order"].append(row["Order"])
                new_info["Family"].append(row["Family"])
                new_info["source_audio_file_name"].append(row["audio_file_name"])

        new_df_all = pd.DataFrame(new_info)
        new_df_all.to_csv(os.path.join(temp_dir, "all.csv"), index=False)
        print(f"  all: {len(new_df_all)} songs")

        # ── Step 2: derive sub-split CSVs by filtering (no audio re-write) ──
        for split_name in ["train", "val", "test"]:
            orig_fns = set(pd.read_csv(f"{BASE_PATH}/{split_name}.csv")["audio_file_name"])
            split_df = new_df_all[new_df_all["source_audio_file_name"].isin(orig_fns)].reset_index(
                drop=True
            )
            split_df.to_csv(os.path.join(temp_dir, f"{split_name}.csv"), index=False)
            print(f"  {split_name}: {len(split_df)} songs")

        # ── Step 3: batch upload ──────────────────────────────────────────
        os.system(f"gsutil -m cp -r {audio_dir} {SINGLE_SONG_GCS_PATH}/")
        for split_name in ["all", "train", "val", "test"]:
            os.system(
                f"gsutil cp {os.path.join(temp_dir, split_name + '.csv')}"
                f" {SINGLE_SONG_GCS_PATH}/{split_name}.csv"
            )


def iterate_qc(
    df: pd.DataFrame,
    data_root: str | None,
) -> pd.DataFrame:
    """
    Iterate dataset and run basic QC. Returns a DataFrame of issues.

    Returns
    ---------
    DataFrame of issues
    """
    problems = []
    for i, row in df.iterrows():
        if i % 100 == 0:
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

        audio_end = len(audio) / float(sr)
        st_end = float(st["Begin Time (s)"].max()) if not st.empty else 0.0
        if st_end > audio_end + 1e-6:
            problems.append(
                {"idx": i, "audio_path": row.get("audio_path", ""), "issue": "events_after_audio"}
            )
        durs = st["End Time (s)"] - st["Begin Time (s)"]
        if durs.min() <= 0:
            problems.append(
                {"idx": i, "audio_path": row.get("audio_path", ""), "issue": "events of dur <= 0"}
            )

    return pd.DataFrame(problems)


def quality_check() -> None:
    """Run QC for original and single-song splits"""

    # Original splits
    info = pd.read_csv(f"{BASE_PATH}/all.csv")
    assert len(info) == len(pd.read_csv(f"{BASE_PATH}/train.csv")) + len(
        pd.read_csv(f"{BASE_PATH}/val.csv")
    ) + len(pd.read_csv(f"{BASE_PATH}/test.csv"))
    problems = iterate_qc(info, BASE_PATH)
    problems.to_csv("subsegmentation_qc_report.csv")

    # Single-song splits
    ss_all = pd.read_csv(f"{SINGLE_SONG_GCS_PATH}/all.csv")
    ss_train = pd.read_csv(f"{SINGLE_SONG_GCS_PATH}/train.csv")
    ss_val = pd.read_csv(f"{SINGLE_SONG_GCS_PATH}/val.csv")
    ss_test = pd.read_csv(f"{SINGLE_SONG_GCS_PATH}/test.csv")
    assert len(ss_all) == len(ss_train) + len(ss_val) + len(ss_test)
    ss_problems = iterate_qc(ss_all, SINGLE_SONG_GCS_PATH)
    ss_problems.to_csv("subsegmentation_single_song_qc_report.csv")


if __name__ == "__main__":
    all_split()
    train_val_test_split()
    single_song()
    quality_check()
