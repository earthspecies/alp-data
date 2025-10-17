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
"""

import os
from io import StringIO
from typing import Dict

import numpy as np
import pandas as pd
import requests
import scipy.io

from esp_data.io import anypath, audio_stereo_to_mono, filesystem, read_audio

BASE_PATH = "gs://subsegmentation/xeno_canto_annotations"
AUDIO_PATH_REL_BASE = "segments_all_together"
ANNOTATIONS_PATH = (
    "gs://subsegmentation/xeno_canto_annotations/syllable_annotations_from_evsonganaly"
)

SPECIES_LABEL_FIX = {}


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
        onsets = [float(mat["offsets"][i][0]) for i in range(len(mat["onsets"]))]

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

    return pd.DataFrame(problems)


def quality_check() -> None:
    """Run QC for dataset"""

    info = pd.read_csv("all.csv")
    assert len(info) == len(pd.read_csv("train.csv")) + len(pd.read_csv("val.csv")) + len(
        pd.read_csv("test.csv")
    )
    problems = iterate_qc(info, BASE_PATH)
    problems.to_csv("subsegmentation_qc_report.csv")


if __name__ == "__main__":
    all_split()
    train_val_test_split()
    quality_check()
