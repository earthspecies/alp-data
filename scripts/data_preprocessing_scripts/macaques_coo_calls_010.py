import shutil
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from alp_data.io import anypath

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
DATA_ROOT = "gs://biodenoising-datasets/macaques/"
OUT_PATH = "/home/marius/data/macaques_coo_calls/v0.1.0/raw/"

macaques = {
    "AL": {"sex": "male", "weight_kg": 8.45, "call_sample_size": 999},
    "BE": {"sex": "male", "weight_kg": 8.05, "call_sample_size": 478},
    "QU": {"sex": "male", "weight_kg": 4.9, "call_sample_size": 975},
    "MU": {"sex": "male", "weight_kg": 5.7, "call_sample_size": 1017},
    "IO": {"sex": "female", "weight_kg": 4.58, "call_sample_size": 1002},
    "SN": {"sex": "female", "weight_kg": 8.2, "call_sample_size": 1001},
    "TH": {"sex": "female", "weight_kg": 4.75, "call_sample_size": 1345},
    "TW": {"sex": "female", "weight_kg": 5.8, "call_sample_size": 468},
}


def convert_macaques(data_dir: str, out_path: str, num_workers: int = 1) -> None:
    csv_path = anypath(DATA_ROOT) / "annotations.csv"
    # copy csv to out_path
    shutil.copy(csv_path, anypath(out_path) / "annotations.csv")
    df = pd.read_csv(csv_path)
    # add sex and weight to df
    df["sex"] = df["class"].map(lambda x: macaques[x]["sex"])
    df["weight_kg"] = df["class"].map(lambda x: macaques[x]["weight_kg"])
    # rename class as id
    df["id"] = df["class"]
    # construct local_path by concatenating 'audio', split, and filename field
    df["local_path"] = df.apply(lambda row: f"audio/{row['split']}/{row['filename']}", axis=1)
    # create a stratified validation set of multiple labels: id and sex
    df_train = df[df.split == "train"]
    df_train, df_val = train_test_split(
        df_train, test_size=0.2, stratify=df_train[["id", "sex"]], random_state=42
    )
    df_test = df[df.split == "valid"]
    # save train and test to csv
    df_train.to_csv(anypath(out_path) / "train.csv", index=False)
    df_val.to_csv(anypath(out_path) / "validation.csv", index=False)
    df_test.to_csv(anypath(out_path) / "test.csv", index=False)

    zip_file = anypath(DATA_ROOT) / "macaques.zip"
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(anypath(out_path) / "audio")


def __main__() -> None:
    convert_macaques(DATA_ROOT, OUT_PATH)


if __name__ == "__main__":
    __main__()
