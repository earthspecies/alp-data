import os
from io import StringIO

import numpy as np
import pandas as pd
from taxonomy.gbif_converter import GBIFConverter

from esp_data.io import anypath, audio_stereo_to_mono, filesystem, read_audio

converter = GBIFConverter()

TARGET_FP = "gs://esp-ml-datasets/hawaiian_birds/all.csv"

fs = filesystem("gcs")  # "gs" also works as an alias

# get anno's
raw_anno = pd.read_csv("gs://esp-ml-datasets/hawaiian_birds/annotations.csv")

# make species label correction
species_info = pd.read_csv("gs://esp-ml-datasets/hawaiian_birds/species.csv")
species_label_correction = {}
for _, row in species_info.iterrows():
    species_name = row["Scientific Name"]

    if species_name == "Hydrobates castro":
        species_name = "Oceanodroma castro"
    if species_name == "Drepanis coccinea":
        species_name = "Vestiaria coccinea"

    # # check if species name is in gbif
    # r = requests.get(f"{BASE_URL}/taxonomy/{species_name}")
    # if r.status_code != 200:
    #     print(species_name)
    #     breakpoint()
    # j = r.json()
    # inferred_g = j["genus"]
    # input_g = species_name.split(" ")[0]
    # if inferred_g != input_g:
    #     print("wrong genus")
    #     breakpoint()

    species_label_correction[row["Species eBird Code"]] = species_name


info = {"audio_file_name": [], "audio_path": [], "selection_table": []}

# st_fps = sorted(fs.glob("fewshot/evaluation/raw/Anuraset/strong_labels/**/*.txt"))
audio_fns = fs.glob("esp-ml-datasets/hawaiian_birds/audio/*.wav")

for audio_fn in audio_fns:
    audio_fn = os.path.basename(audio_fn)
    info["audio_file_name"].append(audio_fn)
    info["audio_path"].append(os.path.join("audio", audio_fn))
    st = (
        raw_anno[raw_anno["Filename"] == audio_fn.replace(".wav", ".flac")]
        .copy()
        .reset_index(drop=True)
    )
    st["Begin Time (s)"] = st["Start Time (s)"]
    st["Species"] = st["Species eBird Code"].map(lambda x: species_label_correction[x])

    labels_present = sorted(st["Species"].unique())
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

    st["Species"].map(species_label_correction_part2)

    st = st[["Begin Time (s)", "End Time (s)", "Species"]]

    info["selection_table"].append(st.to_csv(sep="\t", index=False))


out_dir, out_fn = os.path.split(TARGET_FP)
info = pd.DataFrame(info)
info.to_csv(out_fn, index=False)
os.system(f"gsutil cp {out_fn} {out_dir}")
os.remove(out_fn)


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


# 2) Iterate QC
print("QC")
df = pd.read_csv(TARGET_FP)
data_root = anypath(TARGET_FP).parent
qc_df = iterate_qc(df, data_root)
qc_fp = "hawaii_strong_qc_report.csv"
qc_df.to_csv(qc_fp, index=False)
print(f"Wrote QC report with {len(qc_df)} issues to: {qc_fp}")
