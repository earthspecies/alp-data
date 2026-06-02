import os
from io import StringIO

import numpy as np
import pandas as pd
from taxonomy.gbif_converter import GBIFConverter

from alp_data.io import anypath, audio_stereo_to_mono, filesystem, read_audio

converter = GBIFConverter()

DATA_DIR = "gs://esp-ml-datasets/birdeep"

fs = filesystem("gcs")  # "gs" also works as an alias


def iterate_qc(
    df: pd.DataFrame,
    data_root: str | None = None,
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


# get anno's

unknowns = ["Alaudidae", "Bird", "Curruca", "Fringilla", "Lanius", "Passer", "Sturnus", "Sylvia"]
anno_to_drop = ["No audio", "No bird"]
species_label_correction = {
    "Curruca undata": "Sylvia undata",
    "Galerida Cristata": "Galerida cristata",
    "Linaria Cannabina": "Linaria cannabina",
}

for unk in unknowns:
    species_label_correction[unk] = "Unknown"
gslocs = {
    "train": "gs://esp-ml-datasets/birdeep/train_file_corrected.csv",
    "val": "gs://esp-ml-datasets/birdeep/validation_file.csv",
    "test": "gs://esp-ml-datasets/birdeep/test_file.csv",
}
for split in ["train", "val", "test", "all"]:
    if split == "all":
        anno = pd.concat([pd.read_csv(gslocs[x]) for x in ["train", "val", "test"]])
    else:
        gsloc = gslocs[split]
        anno = pd.read_csv(gsloc)

    anno = anno[~anno["specie"].isin(anno_to_drop)].reset_index(drop=True)

    # create species_label_correction

    anno["specie"] = anno["specie"].map(lambda x: species_label_correction.get(x, x))
    labels_present = sorted(anno["specie"].unique())

    species_label_correction_part2 = {}

    for species_name in labels_present:
        if species_name == "Unknown":
            species_label_correction_part2[species_name] = species_name
            continue

        # check if species name is in gbif
        species_info, matched = converter(species_name)

        if not matched:
            print(species_name)
            species_label_correction_part2[species_name] = species_name

        else:
            species_label_correction_part2[species_name] = species_info["canonicalName"]

    anno["specie"].map(species_label_correction_part2)
    print(anno["specie"].unique())

    info = {"audio_file_name": [], "audio_path": [], "selection_table": []}

    audio_fps = sorted(anno["path"].unique())

    for audio_fp in audio_fps:
        audio_fn = os.path.basename(audio_fp)
        st = anno[anno["path"] == audio_fp].copy().reset_index()
        audio_fp = os.path.join("Audios", audio_fp)

        st["Begin Time (s)"] = st["start_time"]
        st["End Time (s)"] = st["end_time"]
        st["High Freq (Hz)"] = st["high_frequency"]
        st["Low Freq (Hz)"] = st["low_frequency"]
        st["Species"] = st["specie"]

        st = st[["Begin Time (s)", "End Time (s)", "High Freq (Hz)", "Low Freq (Hz)", "Species"]]

        if audio_fp == "Audios/AM8/2023_05_30/AM8_20230530_065000.WAV":
            st = st[st["Begin Time (s)"] < 60]

        info["audio_file_name"].append(audio_fn)
        info["audio_path"].append(audio_fp)
        info["selection_table"].append(st.to_csv(sep="\t", index=False))

    target_path = f"gs://esp-ml-datasets/birdeep/{split}_formatted.csv"
    out_dir, out_fn = os.path.split(target_path)
    info = pd.DataFrame(info)
    info.to_csv(out_fn, index=False)
    os.system(f"gsutil cp {out_fn} {out_dir}")
    os.remove(out_fn)

    # 2) Iterate QC
    print("QC")
    df = pd.read_csv(target_path)
    data_root = anypath(target_path).parent
    qc_df = iterate_qc(df, data_root)
    qc_fp = f"birdeep_{split}_qc_report.csv"
    qc_df.to_csv(qc_fp, index=False)
    print(f"Wrote QC report with {len(qc_df)} issues to: {qc_fp}")
