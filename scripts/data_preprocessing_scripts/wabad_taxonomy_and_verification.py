"""
Preprocess WABAD:
- Build label_mapping.json from the "Species" column -> {Species, Genus, Family, Order, Common}
- Iterate all items for a dataset-level quality check (QC) and emit a CSV report.

Usage:
    python preprocess_wabad.py
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import requests

from alp_data.io import anypath, audio_stereo_to_mono, read_audio

SPECIES_LABEL_FIX = {
    "Campethera nivosa": "Pardipicus nivosus",
    "Eopsaltria flaviventris": "Cryptomicroeca flaviventris",
    "Gliciphila undulata": "Glycifohia undulata",
    "Lanius corvinus": "Corvinella corvina",
    "Neocossyphus fraseri": "Stizorhina fraseri",
    "Oreolais rufogularis": "Oreolais pulcher",
    "Phylloscartes ophthalmicus": "Pogonotriccus ophthalmicus",
    "Rubigula cyaniventris": "Ixodia cyaniventris",
    "Rubigula erythropthalmos": "Ixodia erythropthalmos",
    "Streptopelia chinensis": "Spilopelia chinensis",
    "Streptopelia senegalensis": "Spilopelia senegalensis",
    "Telophorus multicolor": "Chlorophoneus multicolor",
}

ANNOTATION_COLUMNS = ["Genus", "Family", "Order", "Common", "Species"]
DEFAULT_ANNO = "Species"


def _taxonomy_lookup(species_name: str, base_url: str | None) -> Dict[str, str]:
    """
    Return a dict with keys: genus, family, order, species_common.
    If base_url is None, returns empty values (identity mapping).

    Returns
    --------
    Dict with keys: genus, family, order, species_common.
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
        breakpoint()
        return {
            "genus": "",
            "family": "",
            "order": "",
            "species_common": "",
            "species": species_name,
        }
    j = r.json()
    inferred_g = j["genus"]
    input_g = species_name.split(" ")[0]
    if inferred_g != input_g:
        breakpoint()
    return {
        "genus": j.get("genus", "") or "",
        "family": j.get("family", "") or "",
        "order": j.get("order", "") or "",
        "species_common": j.get("species_common", "") or "",
        "species": species_name,
    }


def build_label_mapping(
    df: pd.DataFrame,
    taxonomy_api_url: str | None = None,
) -> Dict[str, Dict[str, str]]:
    """
    Build a mapping from default Species → {Species, Genus, Family, Order, Common}.

    Returns
    -----------
    Dict of Dicts
        label_mapping[taxon_level][species_name] = species_name_mapped_to_that_taxon_level
    """
    # Collect all species labels in the dataset
    species: set[str] = set()
    for _, row in df.iterrows():
        st = pd.read_csv(StringIO(row["selection_table_str"]), sep="\t")
        species.update(st[DEFAULT_ANNO].astype(str).tolist())

    label_mappings: Dict[str, Dict[str, str]] = {k: {} for k in ANNOTATION_COLUMNS}

    for i, sp in enumerate(sorted(species)):
        if i % 100 == 0:
            print(f"{i} / {len(species)}")

        info = _taxonomy_lookup(sp, taxonomy_api_url)
        # Species (possibly corrected)
        label_mappings["Species"][sp] = info["species"]
        # Hierarchy fields
        label_mappings["Genus"][sp] = info["genus"]
        label_mappings["Family"][sp] = info["family"]
        label_mappings["Order"][sp] = info["order"]
        label_mappings["Common"][sp] = (
            "" if info["species_common"] is None else info["species_common"]
        )

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
        if i % 100 == 0:
            print(f"{i} / {len(df)}")
        audio_path = anypath(data_root) / row["audio_fp"] if data_root else anypath(row["audio_fp"])

        try:
            audio, sr = read_audio(audio_path)
        except Exception as e:
            problems.append(
                {"idx": i, "audio_fp": row.get("audio_fp", ""), "issue": f"read_audio_failed: {e}"}
            )
            continue

        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        # selection table
        try:
            st = pd.read_csv(StringIO(row["selection_table_str"]), sep="\t")
        except Exception as e:
            problems.append(
                {"idx": i, "audio_fp": row.get("audio_fp", ""), "issue": f"st_parse_failed: {e}"}
            )
            continue

        # QC checks
        if audio.size < 10:
            problems.append({"idx": i, "audio_fp": row.get("audio_fp", ""), "issue": "too_short"})

        if np.any(np.isnan(audio)):
            problems.append(
                {"idx": i, "audio_fp": row.get("audio_fp", ""), "issue": "nan_in_audio"}
            )

        if np.all(audio == 0):
            problems.append({"idx": i, "audio_fp": row.get("audio_fp", ""), "issue": "all_zeros"})

        audio_end = len(audio) / float(sr)
        st_end = float(st["Begin Time (s)"].max()) if not st.empty else 0.0
        if st_end > audio_end + 1e-6:
            problems.append(
                {"idx": i, "audio_fp": row.get("audio_fp", ""), "issue": "events_after_audio"}
            )

    return pd.DataFrame(problems)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--split-csv",
        default="gs://esp-ml-datasets/wabad/v0.1.0/raw/all_info.csv",
        help="Path to split CSV (e.g., all_info.csv)",
    )
    p.add_argument("--out-dir", default="", help="Directory to write outputs")
    p.add_argument("--taxonomy-api-url", default="http://gagan-dev:8000")
    args = p.parse_args()
    data_root = anypath(args.split_csv).parent

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.split_csv, keep_default_na=False, na_values=[""])

    # 1) Build label mappings
    print("Building label mappings")
    label_mappings = build_label_mapping(df, args.taxonomy_api_url)

    year = date.today().year
    month = date.today().month
    day = date.today().day
    lm_fp = out_dir / f"wabad_label_mapping_{day}_{month}_{year}.json"
    with open(lm_fp, "w") as f:
        json.dump(label_mappings, f, ensure_ascii=False, indent=2)
    print(f"Wrote label mapping to: {lm_fp}")

    # 2) Iterate QC
    print("QC")
    qc_df = iterate_qc(df, data_root)
    qc_fp = out_dir / "qc_report.csv"
    qc_df.to_csv(qc_fp, index=False)
    print(f"Wrote QC report with {len(qc_df)} issues to: {qc_fp}")

    # 3) Print some basic label stats
    for col in ["Species", "Genus", "Family", "Order", "Common"]:
        unique_labels = sorted(set(label_mappings[col].values()))
        print(f"{col}: {len(unique_labels)} unique labels")


if __name__ == "__main__":
    main()
