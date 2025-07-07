import tarfile

import pandas as pd
from sklearn.model_selection import train_test_split

from esp_data.io import anypath

"""
This is a public domain dataset containing macaque vocalizations.
For more information, please refer to the following paper:
@article{fukushima2015distributed,
  title={Distributed acoustic cues for caller identity in macaque vocalization},
  author={Fukushima, Makoto and Doyle, Alex M and Mullarkey, Matthew P and
  Mishkin, Mortimer and Averbeck, Bruno B},
  journal={Royal Society open science},
  volume={2},
  number={12},
  pages={150432},
  year={2015},
  publisher={The Royal Society Publishing}
}
"""
DATA_ROOT = "gs://biodenoising-datasets/orchive/"
OUT_PATH = "/home/marius/data/orchive"


def convert_orchive(data_dir: str, out_path: str, untar: bool = False) -> None:
    if untar:
        tar_file = anypath(DATA_ROOT) / "orchive-dataset.tar.gz"
        with tarfile.open(tar_file, "r") as tar_ref:
            tar_ref.extractall(anypath(out_path))

    # Debug: Check if the directory exists
    out_path_obj = anypath(out_path)
    call_catalog_path = out_path_obj / "orchive-dataset/call-catalog/wav"
    print(f"Checking path: {call_catalog_path}")
    print(f"Path exists: {call_catalog_path.exists()}")

    # List all files in the directory to see what's there
    if call_catalog_path.exists():
        print("Files in call_catalog/wav:")
        for item in call_catalog_path.iterdir():
            print(f"  {item}")

    df = pd.DataFrame(columns=["id", "local_path", "call_type"])
    # exclude files starting with '._'
    wav_files = [
        file for file in call_catalog_path.glob("**/*.wav") if not file.name.startswith("._")
    ]
    print(f"Found {len(wav_files)} WAV files")

    for file in wav_files:
        id = file.stem.split("-")[0]
        call_type = file.stem.split("-")[1]
        local_path = file.relative_to(out_path_obj)
        df = pd.concat(
            [df, pd.DataFrame([{"id": id, "local_path": local_path, "call_type": call_type}])],
            ignore_index=True,
        )
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df = df.sort_values(by="id")

    # Create simplified call type labels (top 6 + other)
    call_type_counts = df["call_type"].value_counts()
    top_6_call_types = call_type_counts.head(6).index.tolist()
    print(f"\nTop 6 call types: {top_6_call_types}")

    # Create simplified call type column
    df["call_type_simple"] = df["call_type"].apply(
        lambda x: x if x in top_6_call_types else "other"
    )

    # Debug: Analyze data distribution
    print("\nDataset statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Unique IDs: {df['id'].nunique()}")
    print(f"Unique call types (original): {df['call_type'].nunique()}")
    print(f"Unique call types (simplified): {df['call_type_simple'].nunique()}")
    print("\nSimplified call type distribution:")
    print(df["call_type_simple"].value_counts())
    print("\nID distribution:")
    print(df["id"].value_counts().head(10))

    # Create stratification groups using simplified call types
    df["strat_group"] = df["id"] + "_" + df["call_type_simple"]
    strat_counts = df["strat_group"].value_counts()
    print("\nStratification groups with counts (simplified):")
    print(strat_counts.head(20))
    print(f"\nGroups with only 1 sample: {len(strat_counts[strat_counts == 1])}")
    print(f"Groups with 2+ samples: {len(strat_counts[strat_counts >= 2])}")

    # create train, val, test split with simplified stratification
    # Strategy: Use simplified call types for stratification
    # This should significantly reduce the number of classes

    print("\nSplitting strategy (simplified):")
    print(f"Total stratification groups: {len(strat_counts)}")
    print(f"Groups with sufficient samples: {len(strat_counts[strat_counts >= 2])}")
    print(f"Groups with insufficient samples: {len(strat_counts[strat_counts < 2])}")

    # Check if we can do stratified split with simplified labels
    if len(strat_counts[strat_counts >= 2]) > 0:
        # Use simplified stratification
        df_train, df_test = train_test_split(
            df, test_size=0.2, stratify=df[["call_type_simple"]], random_state=42
        )
        df_train, df_val = train_test_split(
            df_train, test_size=0.2, stratify=df_train[["call_type_simple"]], random_state=42
        )
        print("\nSuccessfully used simplified stratification!")
    else:
        # Fallback to random split if still too many classes
        print("\nFalling back to random split due to insufficient samples per group")
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)

    # Remove the strat_group and call_type_simple columns before saving
    df_train = df_train.drop(["strat_group", "call_type_simple"], axis=1)
    df_val = df_val.drop(["strat_group", "call_type_simple"], axis=1)
    df_test = df_test.drop(["strat_group", "call_type_simple"], axis=1)

    print("\nFinal split sizes:")
    print(f"Train: {len(df_train)}")
    print(f"Validation: {len(df_val)}")
    print(f"Test: {len(df_test)}")

    # Show distribution of original call types in each split
    print("\nCall type distribution in splits:")
    print("Train:")
    print(df_train["call_type"].value_counts().head(10))
    print("\nValidation:")
    print(df_val["call_type"].value_counts().head(10))
    print("\nTest:")
    print(df_test["call_type"].value_counts().head(10))

    df_train.to_csv(anypath(out_path) / "train.csv", index=False)
    df_val.to_csv(anypath(out_path) / "validation.csv", index=False)
    df_test.to_csv(anypath(out_path) / "test.csv", index=False)

    # process the unsupervised folder
    df_unsupervised = pd.DataFrame(columns=["local_path"])
    unsupervised_path = anypath(out_path) / "orchive-dataset/extract"
    for file in unsupervised_path.glob("**/*.wav"):
        local_path = file.relative_to(out_path_obj)
        df_unsupervised = pd.concat(
            [df_unsupervised, pd.DataFrame([{"local_path": local_path}])], ignore_index=True
        )
    df_unsupervised = df_unsupervised.drop_duplicates()
    df_unsupervised = df_unsupervised.reset_index(drop=True)
    df_unsupervised = df_unsupervised.sort_values(by="local_path")
    df_unsupervised.to_csv(anypath(out_path) / "unsupervised.csv", index=False)


def __main__() -> None:
    convert_orchive(DATA_ROOT, OUT_PATH)


if __name__ == "__main__":
    __main__()
