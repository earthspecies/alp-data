"""
https://figshare.com/articles/dataset/Vocal_repertoires_from_adult_and_chick_male_and_female_zebra_finches_Taeniopygia_guttata_/11905533/1
"""

import argparse
import os
import random
import shutil
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

VERSION = "1.0"

SAMPLE_RATE = 16000
MIN_DURATION_MIX = 2  # seconds
NBIRDS = 4
MAX_CALLS_PER_BIRD = 6

CALL_TYPES_1 = {
    "Ag": "distress",
    "DC": "contact",
    "Di": "distress",
    "LT": "contact",
    "Te": "contact",
    "Th": "alarm",
    "Tu": "alarm",
    "Wh": "nest",
}

CALL_TYPES_2 = {
    "Be": "begging",
    "Ne": "nest",
    "So": "song",
}
SEED = 42
randomg = random.Random(SEED)
randomnp = np.random.RandomState(SEED)

call_types_1 = list(set(list(CALL_TYPES_1.values())))
call_types_2 = list(set(list(CALL_TYPES_2.values())))

parser = argparse.ArgumentParser(
    "convert_zebra_finch_elie", description="Generate the zebra finch chick calls dataset"
)
parser.add_argument(
    "--data_dir", type=str, required=True, help="directory containing the zip files"
)
parser.add_argument("--unzip", action="store_true", help="unzip the dataset")


def unzip_dataset(data_dir: str, out_dir: str) -> None:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    print(
        "You need to download the dataset from https://figshare.com/articles/dataset/Vocal_repertoires_from_adult_and_chick_male_and_female_zebra_finches_Taeniopygia_guttata_/11905533/1"
    )
    assert os.path.exists(os.path.join(data_dir, "AdultVocalizations.zip"))
    assert os.path.exists(os.path.join(data_dir, "ChickVocalizations.zip"))
    os.makedirs(os.path.join(out_dir, "zebra_finch_elie"), exist_ok=True)
    os.system(
        "unzip -o {}/AdultVocalizations.zip -d {}".format(
            data_dir, os.path.join(out_dir, "zebra_finch_elie", "AdultVocalizations")
        )
    )
    os.system(
        "unzip -o {}/ChickVocalizations.zip -d {}".format(
            data_dir, os.path.join(out_dir, "zebra_finch_elie", "ChickVocalizations")
        )
    )


def copy_audio_files(audio_dir: str, data_dir: str) -> None:
    """
    Copy all audio files from AdultVocalizations and ChickVocalizations folders
    to the audio subfolder in the data directory.
    """
    audio_output_dir = os.path.join(data_dir, "audio")
    os.makedirs(audio_output_dir, exist_ok=True)

    for subfolder in ["AdultVocalizations", "ChickVocalizations"]:
        source_dir = os.path.join(audio_dir, subfolder)
        target_dir = os.path.join(audio_output_dir, subfolder)

        if os.path.exists(source_dir):
            # Create target subfolder
            os.makedirs(target_dir, exist_ok=True)

            # Copy all .wav files
            for file in os.listdir(source_dir):
                if file.endswith(".wav"):
                    source_file = os.path.join(source_dir, file)
                    target_file = os.path.join(target_dir, file)
                    shutil.copy2(source_file, target_file)
                    print(f"Copied: {file} to {target_file}")
        else:
            print(f"Warning: Source directory {source_dir} does not exist")


def stratified_train_val_test_split(
    df: pd.DataFrame,
    stratify_columns: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified train/validation/test split ensuring all classes are represented
    in each split.

    Args:
        df: DataFrame to split
        stratify_columns: List of column names to stratify on
        train_ratio: Ratio for training set (default: 0.8)
        val_ratio: Ratio for validation set (default: 0.1)
        test_ratio: Ratio for test set (default: 0.1)
        random_state: Random seed for reproducibility

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Create a combined stratification key
    df_copy = df.copy()
    df_copy["stratify_key"] = df_copy[stratify_columns].apply(
        lambda x: "_".join(x.astype(str)), axis=1
    )

    # Get unique stratification keys
    unique_keys = df_copy["stratify_key"].unique()
    print(f"Found {len(unique_keys)} unique stratification combinations")

    # Initialize empty DataFrames for each split
    train_dfs = []
    val_dfs = []
    test_dfs = []

    for key in unique_keys:
        key_data = df_copy[df_copy["stratify_key"] == key].copy()
        n_samples = len(key_data)

        if n_samples < 3:
            # If we have less than 3 samples, put all in train
            print(f"Warning: Only {n_samples} samples for key '{key}', putting all in train set")
            train_dfs.append(key_data)
            continue

        # Calculate split sizes
        n_train = max(1, int(n_samples * train_ratio))
        n_val = max(1, int(n_samples * val_ratio))
        n_test = n_samples - n_train - n_val

        # Ensure we have at least 1 sample in each split
        if n_test < 1:
            n_test = 1
            n_val = max(1, n_val - 1)
            n_train = n_samples - n_val - n_test

        # Perform the split
        train_temp, temp_data = train_test_split(
            key_data,
            train_size=n_train,
            random_state=random_state,
            stratify=key_data["stratify_key"],
        )

        if len(temp_data) > 1:
            val_temp, test_temp = train_test_split(
                temp_data,
                train_size=n_val,
                random_state=random_state,
                stratify=temp_data["stratify_key"],
            )
        else:
            val_temp = temp_data
            test_temp = pd.DataFrame(columns=temp_data.columns)

        train_dfs.append(train_temp)
        val_dfs.append(val_temp)
        test_dfs.append(test_temp)

    # Combine all splits
    train_df = (
        pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame(columns=df.columns)
    )
    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame(columns=df.columns)
    test_df = (
        pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame(columns=df.columns)
    )

    # Remove the temporary stratify_key column
    train_df = train_df.drop("stratify_key", axis=1)
    val_df = val_df.drop("stratify_key", axis=1)
    test_df = test_df.drop("stratify_key", axis=1)

    print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Verify all classes are present in each split
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n{split_name} set class distribution:")
        for col in stratify_columns:
            unique_vals = split_df[col].unique()
            print(f"  {col}: {len(unique_vals)} unique values - {sorted(unique_vals)}")

    return train_df, val_df, test_df


def create_dataset(audio_dir: str, data_dir: str) -> None:
    # First copy all audio files
    copy_audio_files(audio_dir, data_dir)

    df = pd.DataFrame(
        columns=["id", "call_type_id", "call_type_1", "call_type_2", "age", "local_path"]
    )

    for subfolder in ["AdultVocalizations", "ChickVocalizations"]:
        for file in os.listdir(os.path.join(data_dir, "audio", subfolder)):
            if file.endswith(".wav"):
                id_bird = file.split("_")[0]
                call_type_code = file.split("-")[1][:2]
                call_type_1 = (
                    CALL_TYPES_1[call_type_code] if call_type_code in CALL_TYPES_1 else "unknown"
                )
                call_type_2 = (
                    CALL_TYPES_2[call_type_code] if call_type_code in CALL_TYPES_2 else "unknown"
                )
                age = "adult" if subfolder == "AdultVocalizations" else "chick"

                file_path = os.path.join("audio", subfolder, file)

                df.loc[len(df)] = {
                    "id": id_bird,
                    "call_type_id": call_type_code,
                    "call_type_1": call_type_1,
                    "call_type_2": call_type_2,
                    "age": age,
                    "local_path": file_path,
                }

    os.makedirs(os.path.join(data_dir, "csv_data"), exist_ok=True)

    # Perform stratified train/validation/test split
    stratify_columns = ["id", "call_type_id", "call_type_1", "call_type_2", "age"]
    train_df, val_df, test_df = stratified_train_val_test_split(
        df, stratify_columns, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=SEED
    )

    # Save the splits
    train_df.to_csv(os.path.join(data_dir, "csv_data", "train.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "csv_data", "val.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "csv_data", "test.csv"), index=False)

    # Save the full dataset as well
    df.to_csv(os.path.join(data_dir, "csv_data", "full_dataset.csv"), index=False)

    print("Dataset created successfully!")
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Files saved to: {os.path.join(data_dir, 'csv_data')}")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.unzip:
        unzip_dataset(args.data_dir, args.data_dir)
    create_dataset(args.data_dir, "tmp/zebra_finch_elie/v0.1.0/raw")
