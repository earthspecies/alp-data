import argparse
import json
import os
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from beans_cfg import beans0_cfg

from esp_data.dataset.web_dataset.audio_ds import AudioDataset, prepare_audio_sample_for_sharding
from esp_data.paths import is_cloud_path


def prepare_audio_sample_for_beans0(row: pd.Series, remove_inaturalist: bool = True) -> dict[str, Any]:
    # if we encounter iNaturalist as the 'source_dataset', then we cant add audio to the dataset
    if remove_inaturalist and row["source_dataset"] == "iNaturalist":
        # make 0 bytes audio
        if "file_path" in row:
            del row["file_path"]
        return {"audio.wav": b"", "metadata.json": json.dumps(row.to_dict())}

    # otherwise prep data
    return prepare_audio_sample_for_sharding(row)


def checks(metadata_df: pd.DataFrame, original_paths_df: pd.DataFrame):
    assert len(metadata_df) == len(original_paths_df)
    assert metadata_df.instruction.isnull().sum() == 0
    assert metadata_df.file_name.isnull().sum() == 0
    assert metadata_df.output.isnull().sum() == 0
    assert metadata_df.source_dataset.isnull().sum() == 0


def main():
    parser = argparse.ArgumentParser(description="Create a sharded Beans0 dataset from raw audio files")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to the metadata CSV file")
    parser.add_argument(
        "--web_dataset_path",
        type=str,
        required=True,
        help="Path to the directory where the sharded dataset will be stored",
    )
    parser.add_argument(
        "--original_paths_file",
        type=str,
        required=True,
        help="Path to the file containing the original paths of the audio files",
    )
    parser.add_argument("--num_samples_per_shard", type=int, default=1000, help="Number of samples per shard")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for parallel processing")
    parser.add_argument("--version", type=str, required=True, help="Version of the dataset, e.g. v0.1.1")
    parser.add_argument(
        "-r",
        "--remove_inaturalist",
        action="store_true",
        help="Remove samples from iNaturalist dataset from the sharded dataset",
    )
    parser.add_argument("--changelog", type=str, help="Changelog for the dataset")

    args = parser.parse_args()

    # Load metadata
    metadata_df = pd.read_csv(args.metadata_path)
    metadata_df["output"] = metadata_df["output"].astype(str)  # convert to string
    metadata_df["instruction"] = metadata_df["instruction"].astype(str)  # convert to string

    # Add file paths to metadata
    original_paths_df = pd.read_csv(args.original_paths_file)
    metadata_df["file_path"] = original_paths_df["path"]  # this will be dropped after sharding

    # shuffle the rows, the ensures equal sized shards
    metadata_df = metadata_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Perform checks
    checks(metadata_df, original_paths_df)

    sample_prep_function = partial(prepare_audio_sample_for_beans0, remove_inaturalist=args.remove_inaturalist)

    output_path = os.path.join(args.web_dataset_path, args.version)
    if is_cloud_path(output_path):
        storage_options = {"project": os.getenv("GCP_DEFAULT_PROJECT")}
    else:
        storage_options = None

    # Create an AudioDataset instance
    ds = AudioDataset(
        metadata_df=metadata_df,
        web_dataset_path=output_path,
        num_samples_per_shard=args.num_samples_per_shard,
        storage_options=storage_options,
        sample_prep_function=sample_prep_function,
        num_workers=args.num_workers,
    )

    ds.create_sharded_dataset()

    # write new dataset config file
    beans0_cfg.version = args.version.replace("v", "")
    beans0_cfg.changelog = args.changelog
    with open(os.path.join(output_path, "dataset_config.json"), "w") as fp:
        json.dump(beans0_cfg.to_dict(make_serializable=True), fp)

    # test the dataset
    for audio, metadata in ds:
        print(f"Audio of type: {type(audio)} and shape {audio.shape}")
        print(f"Metadata of type: {type(metadata)} and keys {metadata.keys()}")
        assert isinstance(audio, np.ndarray)
        break

    print(f"Sharded dataset created at: {output_path}")
    audio, metadata = ds[0]
    print(audio.shape, metadata)


if __name__ == "__main__":
    main()
