"""
Preprocess XCJ23:
- Map labels into GBIF taxonomy
- Iterate all items for a dataset-level quality check (QC) and emit a CSV report.

Usage:
    python format_xenocanto_annotated_jeantet_2023.py
"""

from __future__ import annotations

import argparse
import os
from io import StringIO
from typing import Dict

import numpy as np
import pandas as pd
from taxonomy.gbif_converter import GBIFConverter

from alp_data.io import anypath, audio_stereo_to_mono, read_audio

converter = GBIFConverter()

SPECIES_LABEL_FIX = {
    "Paridae_Saxicola_gutturalis": "Saxicola gutturalis",
    "Paridae_Hypocnemis_hypoxantha": "Hypocnemis hypoxantha",
    "Turdidae_Catharus_aurantiirostris_Mexico": "Catharus aurantiirostris",
    "Paridae_Saxicola_torquatus": "Saxicola torquatus",
    "Turdidae_Catharus_guttatus": "Catharus guttatus",
    "Turdidae_Catharus_aurantiirostris ": "Catharus aurantiirostris",
    "Paridae_Hypocnemis_striata": "Hypocnemis striata",
    "Turdidae_Catharus_ustulatus": "Catharus ustulatus",
    "aridae_Saxicola_rubetra": "Saxicola rubetra",
    "idae_Saxicola_rubetra": "Saxicola rubetra",
    "Turdidae_Catharus_fuscescens": "Catharus fuscescens",
    "Paridae_Hypocnemis_cantator": "Hypocnemis cantator",
    "ae_Saxicola_rubicola": "Saxicola rubicola",
    "Turdidae_Catharus_aurantiirostris": "Catharus aurantiirostris",
    " Paridae_Saxicola_rubicola": "Saxicola rubicola",
    "Paridae_Hypocnemis_peruviana": "Hypocnemis peruviana",
    "Turdidae_Catharus_fuscater_Ecuador_2002-03-Turdidae_Catharus_fuscater": "Catharus fuscater",
    "Paridae_Saxicola_rubicola": "Saxicola rubicola",
    "Fringillidae_Serinus_icollis": "Serinus canicollis",
    "Troglodytidae_Troglodytes_pacificus": "Troglodytes pacificus",
    "Turdidae_Catharus_fuscater": "Catharus fuscater",
    "Troglodytidae_Troglodytes_troglodytes": "Troglodytes troglodytes",
    "Troglodytidae_Troglodytes_hiemalis": "Troglodytes hiemalis",
    "Turdidae_Catharus_minimus": "Catharus minimus",
    "Turdidae_Catharus_bicknelli": "Catharus bicknelli",
    "Paridae_Saxicola_tectes": "Saxicola tectes",
    "Troglodytidae_Troglodytes_troglodytes. 8": "Troglodytes troglodytes",
    "Fringillidae_Serinus_canicollis": "Serinus canicollis",
    # "nan": "nan",  # gets removed later
    "Fringillidae_Serinus_serinus_France": "Serinus serinus",
    "oglodytidae_Troglodytes_hiemalis": "Troglodytes hiemalis",
    "Paridae_Saxicola_rubetra": "Saxicola rubetra",
    "roglodytidae_Troglodytes_troglodytes": "Troglodytes troglodytes",
    "Troglodytidae_Troglodytes_aedon": "Troglodytes aedon",
    "Fringillidae_Serinus_serinus": "Serinus serinus",
}
SPECIES_LABEL_FIX["Troglodytidae_Troglodytes_troglodytes_Spain_2020-10-12_XC595581_song"] = (
    "Troglodytes troglodytes"
)
DEFAULT_ANNO = "Species"
ANNOTATION_COLUMNS = ["Species"]  # default_anno needs to be last


def _taxonomy_lookup(species_name: str) -> Dict[str, str]:
    """
    Return a dict with keys: species

    Returns
    --------
    Dict with keys: species
    """
    species_name = SPECIES_LABEL_FIX.get(species_name, species_name)

    if species_name == "nan":
        return {
            "species": species_name,
        }

    species_info, matched = converter(species_name)

    if not matched:
        print(species_name)
        return {"species": species_name}

    return {"species": species_info["canonicalName"]}


def build_label_mapping(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, str]]:
    """
    Build a mapping from default Species → {Species}.

    Returns
    -----------
    Dict of Dicts
        label_mapping[taxon_level][species_name] = species_name_mapped_to_that_taxon_level
    """
    # Collect all species labels in the dataset
    species: set[str] = set()
    for _, row in df.iterrows():
        st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
        species.update(st[DEFAULT_ANNO].astype(str).tolist())

    label_mappings: Dict[str, Dict[str, str]] = {k: {} for k in ANNOTATION_COLUMNS}

    for i, sp in enumerate(sorted(species)):
        if i % 100 == 0:
            print(f"{i} / {len(species)}")

        info = _taxonomy_lookup(sp)
        # Species (possibly corrected)
        label_mappings["Species"][sp] = info["species"]

    return label_mappings


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
        if i % 10 == 0:
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
            if species == "Unknown":
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
            if not speciesinfo["canonicalName"] == species:
                problems.append(
                    {
                        "idx": i,
                        "audio_path": row.get("audio_path", ""),
                        "issue": f"non-gbif species {species}",
                    }
                )

    return pd.DataFrame(problems)


def update_st(row: pd.Series, label_mapping: Dict) -> str:
    """
    Read in selection table (as a string)
    Convert to pandas dataframe
    Remove nan's
    Map label_mapping over columns
    Convert back to string

    Returns
    -----------
    str
    """

    st = row["selection_table"]

    # some rows are nan's, but we can fill in the correct name
    # by looking at the filename
    inferred_label = "_".join(row["audio_file_name"].split("_")[:3])

    st = pd.read_csv(StringIO(st), sep="\t")
    st["Species"] = inferred_label

    for anno_col in ANNOTATION_COLUMNS:
        st[anno_col] = label_mapping[anno_col][inferred_label]

    # fix: event after audio in one file
    if row["audio_file_name"] == "Paridae_Saxicola_rubetra_Sweden_2007-05-06_XC121169_song.wav":
        st = st[st["Begin Time (s)"] < 60]

    st = st.to_csv(sep="\t")
    return st


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--split-csv",
        default="gs://esp-ml-datasets/xeno_canto_annotated_jeantet_2023/all.csv",
        help="Path to split CSV (e.g., all_info.csv)",
    )
    p.add_argument(
        "--out-fp",
        default="gs://esp-ml-datasets/xeno_canto_annotated_jeantet_2023/all_gbif.csv",
        help="Directory to write outputs",
    )
    args = p.parse_args()
    data_root = anypath(args.split_csv).parent

    df = pd.read_csv(args.split_csv, keep_default_na=False, na_values=[""])

    # 1) Build label mappings and update selection tables
    print("Building label mappings")
    label_mappings = build_label_mapping(df)

    df["xeno_canto_id"] = df["audio_file_name"].map(lambda x: x.split("_")[-2])

    for _, row in df.iterrows():
        assert row["xeno_canto_id"][:2] == "XC"

    df["selection_table"] = df.apply(lambda x: update_st(x, label_mappings), axis=1)

    out_dir, out_fn = os.path.split(args.out_fp)
    df.to_csv(out_fn, index=False)
    os.system(f"gsutil cp {out_fn} {out_dir}")
    os.remove(out_fn)

    # 2) Iterate QC
    print("QC")
    df = pd.read_csv(args.out_fp)
    qc_df = iterate_qc(df, data_root)
    qc_fp = "xeno_canto_jeantet_23_qc_report.csv"
    qc_df.to_csv(qc_fp, index=False)
    print(f"Wrote QC report with {len(qc_df)} issues to: {qc_fp}")


if __name__ == "__main__":
    main()
