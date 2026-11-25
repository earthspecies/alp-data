"""
Preprocess Powdermill:
- Map labels into GBIF taxonomy
- Iterate all items for a dataset-level quality check (QC) and emit a CSV report.

Usage:
    python powdermill_taxonomy_and_verification
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

ANNOTATION_INFO_FP = "gs://esp-ml-datasets/powdermill/powdermill_species.csv"
anno_info = pd.read_csv(ANNOTATION_INFO_FP)
SPECIES_LABEL_FIX = {}
for _, row in anno_info.iterrows():
    SPECIES_LABEL_FIX[row["Alpha Code"]] = row["Latin Name"]
SPECIES_LABEL_FIX["DOWO"] = "Dryobates pubescens"

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


def update_st(st: str, label_mapping: Dict) -> str:
    """
    Read in selection table (as a string)
    Convert to pandas dataframe
    Map label_mapping over columns
    Convert back to string

    Returns
    -----------
    str
    """

    st = pd.read_csv(StringIO(st), sep="\t")
    for anno_col in ANNOTATION_COLUMNS:

        def lm(x: str, anno_col: str = anno_col) -> str:
            return label_mapping[anno_col][x]

        st[anno_col] = st[DEFAULT_ANNO].map(lm)
    st = st.to_csv(sep="\t")
    return st


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--split-csv",
        default="gs://esp-ml-datasets/powdermill/all.csv",
        help="Path to split CSV (e.g., all_info.csv)",
    )
    p.add_argument(
        "--out-fp",
        default="gs://esp-ml-datasets/powdermill/all_gbif.csv",
        help="Directory to write outputs",
    )
    args = p.parse_args()
    data_root = anypath(args.split_csv).parent

    df = pd.read_csv(args.split_csv, keep_default_na=False, na_values=[""])

    # 1) Build label mappings and update selection tables
    print("Building label mappings")
    label_mappings = build_label_mapping(df)

    df["selection_table"] = df["selection_table"].map(lambda x: update_st(x, label_mappings))

    out_dir, out_fn = os.path.split(args.out_fp)
    df.to_csv(out_fn, index=False)
    os.system(f"gsutil cp {out_fn} {out_dir}")
    os.remove(out_fn)

    # 2) Iterate QC
    print("QC")
    df = pd.read_csv(args.out_fp)
    qc_df = iterate_qc(df, data_root)
    qc_fp = "powdermill_qc_report.csv"
    qc_df.to_csv(qc_fp, index=False)
    print(f"Wrote QC report with {len(qc_df)} issues to: {qc_fp}")

    # 3) Print some basic label stats
    for col in ["Species"]:
        unique_labels = sorted(set(label_mappings[col].values()))
        print(f"{col}: {len(unique_labels)} unique labels")
        print(unique_labels)


if __name__ == "__main__":
    main()
