import os
from io import StringIO

import numpy as np
import pandas as pd
from taxonomy.gbif_converter import GBIFConverter

from esp_data.io import anypath, audio_stereo_to_mono, filesystem, read_audio

converter = GBIFConverter()

DATA_DIR = "gs://esp-ml-datasets/arctic_bird_sounds"

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
        durs = st["End Time (s)"] - st["Begin Time (s)"]
        if durs.min() <= 0:
            problems.append(
                {"idx": i, "audio_path": row.get("audio_path", ""), "issue": "events of dur <= 0"}
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

unknowns = [
    "Duck",
    "Goose",
    "Gull",
    "Biophony",
    "Loon",
    "Passerine",
    "Plover",
    "Sandpiper",
    "Shorebird",
    "UNKN",
    "Unknown sound",
]
anno_to_drop = ["Human", "Insect", "Motor", "Wings"]

species_label_correction = {
    "AMGP": "Pluvialis dominica",
    "ARTE": "Sterna paradisaea",
    "BASA": "Calidris bairdii",
    "BBPL": "Pluvialis squatarola",
    "BRAN": "Branta bernicla",
    "BTGO": "Limosa lapponica",
    "CANG": "Branta canadensis",
    "CORA": "Corvus corax",
    "DUNL": "Calidris alpina",
    "GWFG": "Anser albifrons",  # codespell:ignore
    "HERG": "Larus argentatus",
    "KIEI": "Somateria spectabilis",
    "LALO": "Calcarius lapponicus",
    "LBDO": "Limnodromus scolopaceus",
    "LTDU": "Clangula hyemalis",
    "LTJA": "Stercorarius longicaudus",
    "PALO": "Gavia pacifica",
    "PESA": "Calidris melanotos",
    "POJA": "Stercorarius pomarinus",
    "PUSA": "Calidris maritima",
    "REPH": "Phalaropus fulicarius",
    "RTLO": "Gavia stellata",
    "RUTU": "Arenaria interpres",
    "SAGU": "Xema sabini",
    "SAND": "Calidris alba",
    "SEPL": "Charadrius semipalmatus",
    "SESA": "Calidris pusilla",
    "SNBU": "Plectrophenax nivalis",
    "SNGO": "Anser caerulescens",  # codespell:ignore
    "SPEI": "Somateria fischeri",
    "TUSW": "Cygnus columbianus",
    "WRSA": "Calidris fuscicollis",
}

for unk in unknowns:
    species_label_correction[unk] = "Unknown"

for split in ["all"]:
    audio_fps = sorted(fs.glob("esp-ml-datasets/arctic_bird_sounds/DataS1/audio_raw/*.wav"))

    info = {"audio_file_name": [], "audio_path": [], "selection_table": []}
    for i, audio_fp in enumerate(audio_fps):
        if i % 10 == 0:
            print(f"{i} / {len(audio_fps)}")
        st_fp = audio_fp.replace("audio_raw", "annots").replace(".wav", "-tags.csv")
        if not fs.exists(st_fp):
            print(f"{st_fp} not found, skipping")
            continue
        st = pd.read_csv("gs://" + st_fp)
        st["Species"] = st["tag"]
        st["Begin Time (s)"] = st["start"]
        st["End Time (s)"] = st["end"]
        st["High Frequency (Hz)"] = st["frequency_max"]
        st["Low Frequency (Hz)"] = st["frequency_min"]

        st = st[
            [
                "Begin Time (s)",
                "End Time (s)",
                "Species",
                "Low Frequency (Hz)",
                "High Frequency (Hz)",
            ]
        ]

        st = st[~st["Species"].isin(anno_to_drop)].reset_index(drop=True)

        st["Species"] = st["Species"].map(lambda x: species_label_correction.get(x, x))
        st["Species"] = st["Species"].map(lambda x: " ".join(x.split(" ")[:2]))  # remove subspecies

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

        info["audio_file_name"].append(os.path.basename(audio_fp))
        info["audio_path"].append(audio_fp.split("arctic_bird_sounds/")[-1])
        info["selection_table"].append(st.to_csv(sep="\t", index=False))

    info = pd.DataFrame(info)
    labels_present = set()
    for _, row in info.iterrows():
        st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
        labels_present.update(set(st["Species"].unique()))

    print(labels_present)
    # errors = []
    # for species_name in labels_present:
    #     if species_name == "Unknown":
    #         continue
    #     # check if species name is in gbif
    #     r = requests.get(f"{BASE_URL}/taxonomy/{species_name}")
    #     if r.status_code != 200:
    #         print(species_name)
    #         errors.append(species_name)
    #         # breakpoint()
    #     else:
    #         j = r.json()
    #         inferred_g = j["genus"]
    #         input_g = species_name.split(" ")[0]
    #         if inferred_g != input_g:
    #             print("wrong genus")
    #             breakpoint()
    # print(errors)

    info = pd.DataFrame(info)
    info.to_csv(f"{split}.csv")
    os.system(f"gsutil -m cp -r {split}.csv {DATA_DIR}")

    # 2) Iterate QC
for split in ["all"]:
    print("QC")
    target_path = os.path.join(DATA_DIR, split + ".csv")
    df = pd.read_csv(target_path)
    data_root = anypath(target_path).parent
    qc_df = iterate_qc(df, data_root)
    qc_fp = f"arctic_bird_sounds_{split}_qc_report.csv"
    qc_df.to_csv(qc_fp, index=False)
    print(f"Wrote QC report with {len(qc_df)} issues to: {qc_fp}")
