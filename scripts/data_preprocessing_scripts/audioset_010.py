from typing import List

import pandas as pd

from alp_data.io import anypath, filesystem_from_path

"""
This is the Audioset dataset.
For more information check the paper:
@inproceedings{gemmeke2017audio,
  title={Audio set: An ontology and human-labeled dataset for audio events},
  author={Gemmeke, Jort F and Ellis, Daniel PW and Freedman, Dylan and Jansen, Aren and
          Lawrence, Wade and Moore, R Channing and Plakal, Manoj and Ritter, Marvin},
  booktitle={2017 IEEE international conference on acoustics, speech and signal
             processing (ICASSP)},
  pages={776--780},
  year={2017},
  organization={IEEE}
}
"""
DATA_ROOT = "gs://audioset-2021/"


def label_to_class(row: str, all_classes: pd.DataFrame) -> List[str]:
    labels = row.replace('"', "").split(",")
    labels_text = [
        all_classes.loc[all_classes["mid"] == label].display_name.values[0] for label in labels
    ]
    return labels_text


def filter_labels(row: pd.Series, classes: pd.DataFrame) -> bool:
    # When using apply on a DataFrame, row is a Series with column names as index
    labels = row["labels"].replace('"', "").split(",")
    return any(label in classes["mid"].values for label in labels)


def convert_audioset(data_dir: str, out_path: str, num_workers: int = 1) -> None:
    csv_path = anypath(DATA_ROOT) / "csv-data"
    all_classes = pd.read_csv(csv_path / "class_labels_indices.csv")
    noise_classes = pd.read_csv(csv_path / "noise_classes.csv")
    animal_classes = pd.read_csv(
        csv_path / "ontology.animal.tsv",
        names=["mid", "display_name"],
        skiprows=[0],
        header=None,
        sep="\t",
    )
    animal_classes.loc[len(animal_classes)] = {"mid": "/m/0jbk", "display_name": "Animal"}
    df = {}
    colnames = ["youtube_id", "start", "end", "labels"]

    df["unbalanced_train_segments"] = pd.read_csv(
        csv_path / "unbalanced_train_segments.csv",
        names=colnames,
        skiprows=[0, 1, 2],
        on_bad_lines="skip",
        header=None,
        quotechar='"',
        sep=", ",
    )
    df["eval_segments"] = pd.read_csv(
        csv_path / "eval_segments.csv",
        names=colnames,
        skiprows=[0, 1, 2],
        on_bad_lines="skip",
        header=None,
        quotechar='"',
        sep=", ",
    )
    df["train-balanced"] = pd.read_csv(
        csv_path / "balanced_train_segments.csv",
        names=colnames,
        skiprows=[0, 1, 2],
        on_bad_lines="skip",
        header=None,
        quotechar='"',
        sep=", ",
    )
    # add local path to the dataframes
    df["unbalanced_train_segments"]["local_path"] = df["unbalanced_train_segments"][
        "youtube_id"
    ].apply(lambda x: "audio_files/unbalanced_train_segments/" + x + ".wav")
    df["eval_segments"]["local_path"] = df["eval_segments"]["youtube_id"].apply(
        lambda x: "audio_files/eval_segments/" + x + ".wav"
    )
    df["train-balanced"]["local_path"] = df["train-balanced"]["youtube_id"].apply(
        lambda x: "audio_files/train-balanced/" + x + ".wav"
    )

    # Filter rows that contain animal labels
    animal_mask = df["eval_segments"].apply(lambda row: filter_labels(row, animal_classes), axis=1)
    df["eval_segments_animal"] = df["eval_segments"][animal_mask].copy()
    animal_mask = df["train-balanced"].apply(lambda row: filter_labels(row, animal_classes), axis=1)
    df["train-balanced_animal"] = df["train-balanced"][animal_mask].copy()
    animal_mask = df["unbalanced_train_segments"].apply(
        lambda row: filter_labels(row, animal_classes), axis=1
    )
    df["unbalanced_train_segments_animal"] = df["unbalanced_train_segments"][animal_mask].copy()

    noise_mask = df["eval_segments"].apply(lambda row: filter_labels(row, noise_classes), axis=1)
    df["eval_segments_noise"] = df["eval_segments"][noise_mask].copy()
    noise_mask = df["train-balanced"].apply(lambda row: filter_labels(row, noise_classes), axis=1)
    df["train-balanced_noise"] = df["train-balanced"][noise_mask].copy()
    noise_mask = df["unbalanced_train_segments"].apply(
        lambda row: filter_labels(row, noise_classes), axis=1
    )
    df["unbalanced_train_segments_noise"] = df["unbalanced_train_segments"][noise_mask].copy()

    for split in df.keys():
        df[split]["labels"] = df[split]["labels"].apply(lambda x: label_to_class(x, all_classes))
        # save the dataframes to disk as csv
        out_path = anypath(DATA_ROOT) / "csv-data" / f"{split}_processed.csv"

        # Use filesystem interface to write to GCS
        fs = filesystem_from_path(out_path)
        csv_content = df[split].to_csv(index=False)
        with fs.open(str(out_path), "w") as f:
            f.write(csv_content)


def __main__() -> None:
    convert_audioset(DATA_ROOT, "gs://audioset-2021/")


if __name__ == "__main__":
    __main__()
