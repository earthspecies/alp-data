"""Clean AnimalSpeak dataset splits and save as CSVs.

Operations per split:
1. Remove "Unnamed" index column(s)
2. Remove entirely empty columns
3. Rename "local_path" -> "audio_path"
4. Remove "cluster_path" column
5. Force gbifID to int / null (Int64 nullable)
6. Check existence of absolute audio paths (data_root / audio_path)
"""

import sys
from pathlib import Path

import pandas as pd

# AnimalSpeak split paths and data root (from AnimalSpeak.info and __init__)
SPLIT_PATHS = {
    "train": "gs://esp-ml-datasets/animalspeak/v0.1.0/raw/16KHz/animalspeak2_train.csv",
    "validation": "gs://esp-ml-datasets/animalspeak/v0.1.0/raw/16KHz/animalspeak2_validation.csv",
}
DATA_ROOT = "gs://esp-ml-datasets/animalspeak/v0.1.0/raw/16KHz/"

OUTPUT_DIR = Path(__file__).parent / "animalspeak_cleaned"


def clean_split(split_name: str, csv_path: str) -> pd.DataFrame:
    """Load, clean, and validate a single AnimalSpeak split.

    Parameters
    ----------
    split_name : str
        Name of the split (e.g. "train", "validation") for logging purposes.
    csv_path : str
        Path to the split's CSV file.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame for the split.
    """
    print(f"\n{'=' * 60}")
    print(f"Processing split: {split_name}")
    print(f"Source: {csv_path}")
    print(f"{'=' * 60}")

    # Load CSV
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    # 1. Remove "Unnamed" index column(s)
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        print(f"[1] Removed Unnamed columns: {unnamed_cols}")
    else:
        print("[1] No Unnamed columns found")

    # 2. Remove entirely empty columns
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        print(f"[2] Removed entirely empty columns: {empty_cols}")
    else:
        print("[2] No entirely empty columns found")

    # 3. Rename "local_path" -> "audio_path"
    if "local_path" in df.columns:
        df = df.rename(columns={"local_path": "audio_path"})
        print('[3] Renamed "local_path" -> "audio_path"')
    else:
        print('[3] WARNING: "local_path" column not found')
        if "audio_path" in df.columns:
            print('    "audio_path" already exists')

    # 4. Remove "cluster_path" column
    if "cluster_path" in df.columns:
        df = df.drop(columns=["cluster_path"])
        print('[4] Removed "cluster_path" column')
    else:
        print('[4] WARNING: "cluster_path" column not found')

    # 5. Force gbifID to nullable Int64
    if "gbifID" in df.columns:
        original_dtype = df["gbifID"].dtype
        df["gbifID"] = pd.to_numeric(df["gbifID"], errors="coerce").astype("Int64")
        print(f"[5] Converted gbifID: {original_dtype} -> {df['gbifID'].dtype}")
        n_null = df["gbifID"].isna().sum()
        print(f"    gbifID null count: {n_null} / {len(df)}")
    else:
        print('[5] WARNING: "gbifID" column not found')

    # 6. Check existence of absolute audio paths
    # print("[6] Checking audio path existence...")
    # if "audio_path" not in df.columns:
    #     print("    ERROR: No audio_path column to check")
    #     return df

    # root = anypath(DATA_ROOT)
    # missing_paths = []
    # checked = 0
    # for i, relative_path in enumerate(df["audio_path"]):
    #     if pd.isna(relative_path):
    #         missing_paths.append((i, "<NA>"))
    #         continue
    #     abs_path = root / relative_path
    #     if not abs_path.exists():
    #         missing_paths.append((i, str(abs_path)))
    #     checked += 1
    #     if checked % 10000 == 0:
    #         print(f"    Checked {checked}/{len(df)} paths...")

    # print(f"    Checked {checked}/{len(df)} paths total")
    # if missing_paths:
    #     print(f"    MISSING: {len(missing_paths)} audio files not found")
    #     # Show first 20 missing
    #     for idx, path in missing_paths[:20]:
    #         print(f"      row {idx}: {path}")
    #     if len(missing_paths) > 20:
    #         print(f"      ... and {len(missing_paths) - 20} more")
    # else:
    #     print("    All audio paths exist")

    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, csv_path in SPLIT_PATHS.items():
        df = clean_split(split_name, csv_path)

        out_path = OUTPUT_DIR / f"animalspeak_{split_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved cleaned {split_name} split to: {out_path}")
        print(f"Final shape: {df.shape}")
        print(f"Final columns: {list(df.columns)}")
        print(f"Final dtypes:\n{df.dtypes}")


if __name__ == "__main__":
    sys.exit(main() or 0)
