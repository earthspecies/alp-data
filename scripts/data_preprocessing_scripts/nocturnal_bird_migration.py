import os
from io import StringIO

import numpy as np
import pandas as pd
from taxonomy.gbif_converter import GBIFConverter

from alp_data.io import anypath, audio_stereo_to_mono, filesystem, read_audio

DATA_DIR = "gs://esp-ml-datasets/nocturnal_bird_migration"
# BASE_URL = "http://gagan-dev:8000"
converter = GBIFConverter()

fs = filesystem("gcs")  # "gs" also works as an alias


def merge_rows_skip_backslash_tsv(path: str) -> pd.DataFrame:
    """
    Merge every two logical rows in a tab-delimited annotation file (local or cloud),
    producing a single dataframe with timing, species, and frequency bounds.

    Expected patterns in the source file:
    1) Standard pair:
        <begin>\t<end>\t<species>
        \t<low_hz>\t<high_hz>

       Example:
        4.041116\t4.727713\tActitis hypoleucos
        \t3233.910156\t5813.999023

       -> Low/High Frequency (Hz) taken from second line.

    2) Noise/background pair without backslash on second line:
        96.764613\t96.814477\tbruit de fond
        104.204346\t104.943217\tbruit de fond

       -> We IGNORE the second line's times/label and instead
          set Low/High Frequency (Hz) = 0 / 8000.

    3) Lone final line (no partner line after it):
        12.34\t56.78\tsomething

       -> Also gets Low/High Frequency (Hz) = 0 / 8000.

    Parameters
    ----------
    path : str
        Path to the input TSV file. Can be a local path or a cloud URI
        (e.g. "gs://bucket/file.tsv", "s3://bucket/file.tsv"), as long as
        the appropriate fsspec filesystem is installed.

    Returns
    -------
    pd.DataFrame
        Columns:
        ["Begin Time (s)",
         "End Time (s)",
         "Species",
         "Low Frequency (Hz)",
         "High Frequency (Hz)"]
    """

    # Use pandas' IO machinery so cloud URIs work (requires gcsfs/s3fs/etc).
    with pd.io.common.get_handle(path, mode="r", encoding="utf-8") as handle:
        # strip() to drop trailing newline/whitespace,
        # and skip empty lines entirely
        lines = [line.strip() for line in handle.handle if line.strip()]

    rows_out = []
    i = 0
    n = len(lines)

    while i < n:
        first = lines[i]
        # Parse the first line: begin, end, species
        first_fields = first.split("\t", maxsplit=2)
        if len(first_fields) != 3:
            # If it's malformed, skip it rather than crash
            i += 1
            continue

        begin_time_str, end_time_str, species = first_fields

        # Defaults in case we don't get a usable second line
        low_hz = 0.0
        high_hz = 8000.0

        # Check if we have a partner line
        if i + 1 < n:
            second = lines[i + 1]

            if second.startswith("\\"):
                # Case 1: second line begins with backslash, so it encodes freqs.
                # Remove leading backslash, tabs, spaces
                second_clean = second.lstrip("\\\t ").strip()
                hz_fields = second_clean.split("\t")
                if len(hz_fields) >= 2:
                    try:
                        low_hz = float(hz_fields[0])
                        high_hz = float(hz_fields[1])
                    except ValueError:
                        # fall back to defaults if parse fails
                        low_hz = 0.0
                        high_hz = 8000.0
                # consume two lines
                i += 2
            else:
                # Case 2: second line does NOT start with backslash.
                # Example:
                #   bruit de fond
                #   bruit de fond
                # We ignore its contents for timing/frequency and just
                # assign default band 0-8000 Hz.
                i += 2
        else:
            # Case 3: no partner line, keep defaults 0 / 8000
            i += 1

        # Build the output row (cast numeric columns)
        try:
            row_dict = {
                "Begin Time (s)": float(begin_time_str),
                "End Time (s)": float(end_time_str),
                "Species": species,
                "Low Frequency (Hz)": float(low_hz),
                "High Frequency (Hz)": float(high_hz),
            }
            rows_out.append(row_dict)
        except ValueError:
            # If begin/end fail to parse as float, silently drop row
            # (or you could append NaN; adjust if you prefer strictness)
            continue

    try:
        df = pd.DataFrame(
            rows_out,
            columns=[
                "Begin Time (s)",
                "End Time (s)",
                "Species",
                "Low Frequency (Hz)",
                "High Frequency (Hz)",
            ],
        )
    except Exception:
        breakpoint()

    return df


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
    "Oiseau sp Botaurus stellaris?? requete XC604392",
    "Parus sp.",
    "0: Unknown",
    "1: Oiseau sp",
    "Oiseau sp.",
    "Oiseau sp",
    "oiseau sp",
    "Inconnu",
    "0: Bird sp",
    "Larus sp",
]
anno_to_drop = [
    "Bruit de fond",
    "Autre biophonie micromam?",
    "Autre biophonie Grillon",
    "background",
    "Autre anthropophonie pompe",
    "Backgroud",  # codespell:ignore
    "Autre biophonie grenouilles",
    "Autre biophonie chat cri",
    "Autre bizarrophonie ça roule sur le toit",
    "Autre biophonie Grenouille",
    "Autre anthropophonie moteur aspiration insert ?",  # codespell:ignore
    "Autre anthropophonie klaxon",
    "Autre biophonie aile",
    "0: Background",
    "Autre biophonie grenouilles",
    "Autre anthropophonie pompe?",
    "Autre biophonie Grillons",
    "Autre bizarrophonie + grillon",
    "bruit parasite",
    "Autre anthropophonie",
    "Autre biophonie chien",
    "Autre biophonie grillons",
    "Autre geophonie",
    "Autre anthropophonie voiture",
    "1: Autre biophonie",
    "Autre biophonie chat cri",
    "Autre biophonie Grillon",
    "Autre anthropophonie pompe",
    "Back ground",
    "parasite",
    "Autre biophonie grillon",
    "chien",
    "background",
    "Background",
    "Chevalier sylvain",
    "1: Autre geophonie",
    "Autre geophonie chene qui grince",
    "Autre biophonie Chiroptere",
    "Autre biophonie grillon diffu",
    "bruit de fond",
    "Autre biophonie (chat)",
    "Other",
    "backgroound",  # codespell:ignore
    "Autre bizarrophonie ça roule sur le toit",
    "possible battement d'aile",
    "Back groung",
    "Autre biophonie Coq domestique",
    "Autre biophonie Vulpes vulpes",
    "Autre biophonie aile",
    "Autre biophonie Grenouille",
    "Autre biophonie grenouille",
    "Autre biophonie Pigeon Tourterelle envol",
    "vulpes vulpes",
    "Vulpes vulpes",
    "Autre biophonie chat cri",
    "Autre anthropophonie voiture",
    "Autre antropophonie",
    "Autre biophonie grillon diffu",
    "Autre biophonie grillon",
    "Autre biophonie chien",
    "Autre bizzarophonie",
    "0: Background",
    "Back groung",
    "Autre biophonie Grenouille",
    " Background",
    "Autre anthropophonie pompe?",
    "autre geophonie",
    "Backgroud",  # codespell:ignore
    "Autrez biophonie",
    "Autre biophonie Vulpes vulpes",
    "Autre biophonie chien hurlement",
    "Autre anthropophonie klaxon",
    "autre biophonie grenouille",
    "Autre biophonie renard ou brocard ?",
    "Autre biophopnie grillon",
    "Autre biophonie insectes",
    "Autre biophonie (chat)",
    "Autre biophonie amphibien",
    "Other",
    "0: Other antropophonia",
    "Autre biophonie bruit aile",
    "bruit parasite",
    "1: Autre biophonie",
    "Autre biophonie requete XC594305",
    "Bruant des roseaux",
    "possible battement d'aile",
    "1: Autre antropophonie",
    "0: Other biophonia",
    "Autre biophonie",
    "Autre biophonie Coq domestique",
    "Autre biophonie chat miaulement",
    "Autre biophonie Chiroptere",
    "Autre biophonie Grillons",
    "Autre geophonie",
    "Background",
    "Hannetons par milliers",
    "orthoptère",
    "Back ground",
    "0: Other geophonia",
    "Bruit de fond",
    "Coq",
    "Autre anthropophonie pompe",
    "Autre biophonie grenouille",
    "Autre biophonie Renard ou brocard?",
    "Chevalier sylvain",
    "0: Bruit parasite",
    "Autre geophonie chene qui grince",
    "Autre bizarrophonie ça roule sur le toit",
    "background",
    "Autre biophonie Pigeon Tourterelle envol",
    "Autre anthropophonie",
    "parasite",
    "Autre biophonie Grillon",
    "Autre bizarrophonie je sais pas d'où vient ce"  # codespell:ignore
    + " fond en forme de vague ds les hautes freq...",  # codespell:ignore
    "Autre biophonie chat grognement",
    "bruit de fond",
    "1: Autre geophonie",
    "Autre biophonie micromam?",
    "Autre biophonie Sus scrofa",
    "Vent geophonie",
    "chien",
    "Autre bizarrophonie + grillon",
    "Autre anthropophonie moteur aspiration insert ?",  # codespell:ignore
    "Autre biophonie (chien)",
    "Autre biophonie Chiropere",
    "autre biophonie",
    "Autre biophonie aile",
    "Autre biophonie chiroptere",
    "Autre bizarrophonie requete XC588141",
    "Autre biophonie hurlement chien",
    "Autre biophonie grillons",
    "Autre biophonie Cervus elaphus",
    "Autre biophonie grenouilles",
    "Autre bizarrophonie",
    "backgroound",  # codespell:ignore
    "Cervus elaphus brame",
    "Capreolus capreolus",
    "Alytes obstetricans",
    "Oecanthus pellucens",
    "Pelophylax sp.",
    "Burhinus burhinus",
    "Bernicla bernicla",
]
species_label_correction = {
    "Anas platyrhyncos": "Anas platyrhynchos",
    "Emberiza ortulana": "Emberiza hortulana",
    "Otusscops": "Otus scops",
    "Stix aluco": "Strix aluco",
    "Ardea nycticorax": "Nycticorax nycticorax",
    "Coloelus monedula": "Coloeus monedula",
    "caprimulgus europaeus": "Caprimulgus europaeus",
    "tachybaptus ruficollis": "Tachybaptus ruficollis",
    "Philloscopus collybita": "Phylloscopus collybita",
    "Numenius arquata XC570503": "Numenius arquata",
    "Anas platyrhycos": "Anas platyrhynchos",
    "Melanitta nigrab": "Melanitta nigra",
    "Motacilla alba*": "Motacilla alba",
    "Turdus philomelus": "Turdus philomelos",
    "Luscinia megarynchos": "Luscinia megarhynchos",
    "chant Luscinia megarhynchos": "Luscinia megarhynchos",
}

for unk in unknowns:
    species_label_correction[unk] = "Unknown"

for split in ["train", "train_nonxc", "train_xc", "test"]:
    if split == "test":
        audio_fps = sorted(
            fs.glob("esp-ml-datasets/nocturnal_bird_migration/zenodo_nbm_db/test/*.wav")
        )
    if split == "train_nonxc":
        audio_fps = sorted(
            fs.glob("esp-ml-datasets/nocturnal_bird_migration/zenodo_nbm_db/train_nbm_orig/*.wav")
        )
    if split == "train_xc":
        audio_fps = sorted(
            fs.glob("esp-ml-datasets/nocturnal_bird_migration/zenodo_nbm_db/train_nbm_xc/*.wav")
        )
    if split == "train":
        audio_fps = sorted(
            fs.glob("esp-ml-datasets/nocturnal_bird_migration/zenodo_nbm_db/train_nbm_orig/*.wav")
        )
        audio_fps += sorted(
            fs.glob("esp-ml-datasets/nocturnal_bird_migration/zenodo_nbm_db/train_nbm_xc/*.wav")
        )

    info = {"audio_file_name": [], "audio_path": [], "selection_table": [], "xeno_canto_id": []}
    for i, audio_fp in enumerate(audio_fps):
        if i % 10 == 0:
            print(f"{i} / {len(audio_fps)}")
        st_fp = audio_fp.replace(".wav", ".txt")
        if not fs.exists(st_fp):
            print(f"{st_fp} not found, skipping")
            continue
        st = merge_rows_skip_backslash_tsv("gs://" + st_fp)

        st = st[~st["Species"].isin(anno_to_drop)].reset_index(drop=True)

        st["Species"] = st["Species"].map(lambda x: species_label_correction.get(x, x))
        st["Species"] = st["Species"].map(lambda x: " ".join(x.split(" ")[:2]))  # remove subspecies
        st["Species"] = st["Species"].map(lambda x: species_label_correction.get(x, x))

        labels_present = st["Species"].unique()
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
        info["audio_path"].append(audio_fp.split("nocturnal_bird_migration/")[-1])
        info["selection_table"].append(st.to_csv(sep="\t", index=False))

        if ("/test/" in audio_fp) or ("/train_nbm_xc/" in audio_fp):
            xc_id = audio_fp.split("#")[-1].split(".")[0]
        else:
            xc_id = ""

        info["xeno_canto_id"].append(xc_id)

    info = pd.DataFrame(info)
    labels_present = set()
    for _, row in info.iterrows():
        st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
        labels_present.update(set(st["Species"].unique()))

    print(labels_present)
    errors = []
    for species_name in labels_present:
        if species_name == "Unknown":
            continue
        # check if species name is in gbif
        species_info, matched = converter(species_name)

        if not matched:
            print(species_name)
        else:
            species_name = species_info["canonicalName"]
    print(errors)

    info = pd.DataFrame(info)
    info.to_csv(f"{split}.csv")
    os.system(f"gsutil -m cp -r {split}.csv {DATA_DIR}")

    # 2) Iterate QC
for split in ["train", "train_nonxc", "train_xc", "test"]:
    print("QC")
    target_path = os.path.join(DATA_DIR, split + ".csv")
    df = pd.read_csv(target_path)
    data_root = anypath(target_path).parent
    qc_df = iterate_qc(df, data_root)
    qc_fp = f"nbm_{split}_qc_report.csv"
    qc_df.to_csv(qc_fp, index=False)
    print(f"Wrote QC report with {len(qc_df)} issues to: {qc_fp}")
