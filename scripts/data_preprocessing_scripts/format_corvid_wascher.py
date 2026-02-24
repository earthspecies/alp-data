import os
from io import StringIO

import numpy as np
import pandas as pd

from esp_data.io import anypath, audio_stereo_to_mono, exists, filesystem, read_audio

TARGET_FP = "gs://esp-ml-datasets/wascher_corvid_comparison/all.csv"

fs = filesystem("gcs")  # "gs" also works as an alias

###############
# get anno's
metadata = pd.read_excel("gs://esp-ml-datasets/wascher_corvid_comparison/corvidae_metadata.xlsx")

info = {
    "audio_file_name": [],
    "audio_path": [],
    "selection_table": [],
    "species": [],
    "xeno_canto_id": [],
    "stage": [],
    "lat": [],
    "lng": [],
    "type": [],
    "cnt": [],
    "sex": [],
    "method": [],
}
confirmed_data = 0
missing_audio = []
missing_st = []


def fix_filename(x: str) -> str:
    """
    fix filename differences between metadata and what's actually there

    Returns
    -----
    str fixed filename
    """
    if "35/a19.mp3" in x:
        x = x.replace("35/a19.mp3", "35_a19.mp3")
    if "2083/b08.mp3" in x:
        x = x.replace("2083/b08.mp3", "2083_b08.mp3")
    if ("[" in x) and ("]" in x):
        x = x.replace("[", "")
        x = x.replace("]", "")
    return x


for i, row in metadata.iterrows():
    if i % 100 == 0:
        print(f"processing {i}/{len(metadata)}")
    genus = row["gen"]
    species = row["sp"]
    audio_fn = row["file-name"]
    xc_id = row["id"]
    if int(xc_id) == 504887:
        print(f"skipping {audio_fn} because file corrupt")
        continue

    audio_absolute_path = (
        f"gs://esp-ml-datasets/wascher_corvid_comparison/audio/{genus}_{species}/{audio_fn}"
    )
    st_absolute_path = f"gs://esp-ml-datasets/wascher_corvid_comparison/Corvidae_annotations/{genus}_{species}/{xc_id}.txt"

    t = type(audio_absolute_path)
    audio_absolute_path = t(fix_filename(str(audio_absolute_path)))

    if not exists(st_absolute_path):
        continue

    if audio_absolute_path[-4] != ".":
        print(f"no extension for {audio_absolute_path}, skipping")
        continue

    if not exists(audio_absolute_path):
        # print(f"Found no audio {audio_absolute_path} matching {st_absolute_path}")
        foundmatch = False
        for nparts in range(5):
            fn_first_part = "-".join(audio_fn.split("-")[: nparts + 1])
            audio_glob = f"gs://esp-ml-datasets/wascher_corvid_comparison/audio/{genus}_{species}/{fn_first_part}*"
            audio_fps = fs.glob(audio_glob)
            if len(audio_fps) == 1:
                audio_absolute_path = "gs://" + audio_fps[0]
                foundmatch = True
                break
        if not foundmatch:
            missing_audio.append(audio_absolute_path)
            missing_st.append(st_absolute_path)
            continue

    confirmed_data += 1
    if confirmed_data % 10 == 0:
        print(f"Found {confirmed_data} pairs so far")

    info["audio_file_name"].append(str(xc_id))
    info["audio_path"].append(audio_absolute_path.split("wascher_corvid_comparison/")[1])
    st = pd.read_csv(st_absolute_path, sep="\t")
    if len(st["View"].unique()) > 1:
        chosen_view = sorted(st["View"].unique())[
            -1
        ]  # opt for Waveform over Spectrogram by reverse sort
        st = st[st["View"] == chosen_view]
    st["Species"] = f"{genus} {species}"
    # Fix events_after_audio error
    if int(xc_id) in [168657, 161255, 186480, 335157, 340750, 340553]:
        audio, sr = read_audio(audio_absolute_path)
        audio_end = len(audio) / float(sr)
        st = st[st["Begin Time (s)"] < audio_end]

    info["selection_table"].append(st.to_csv(sep="\t", index=False))
    info["species"].append(f"{genus} {species}")
    info["xeno_canto_id"].append(int(xc_id))
    for k in ["stage", "lat", "lng", "type", "cnt", "sex", "method"]:
        info[k].append(row[k])

print(f"found {confirmed_data} audio, selection table pairs")

print("-------MISSING-------")

for x in missing_audio:
    print("-", x, ",")

print("------MISSING (the associated sts)------")

for x in missing_st:
    print("-", x, ",")

print(
    "Note that 'https://xeno-canto.org/490447' "
    "changed its species id on xeno-canto from Corvus "
    "mellori to Corvus orru, so we exclude it"
)

out_dir, out_fn = os.path.split(TARGET_FP)
info = pd.DataFrame(info)
info.to_csv(out_fn, index=False)
os.system(f"gsutil cp {out_fn} {out_dir}")
os.remove(out_fn)
##########


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

    return pd.DataFrame(problems)


# 2) Iterate QC
print("QC")
df = pd.read_csv(TARGET_FP)
data_root = anypath(TARGET_FP).parent
qc_df = iterate_qc(df, data_root)
qc_fp = "corvid_wascher_strong_qc_report.csv"
qc_df.to_csv(qc_fp, index=False)
print(f"Wrote QC report with {len(qc_df)} issues to: {qc_fp}")
