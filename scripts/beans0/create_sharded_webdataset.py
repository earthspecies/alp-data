import argparse
import io
import json
import os
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from beans_cfg import beans0_cfg

from esp_data.dataset.shard_creator import create_sharded_dataset
from esp_data.io.parsers import read_audio


def validate_sample(sample: dict, remove_inaturalist: bool = False):
    # if not remove_inaturalist:
    #   assert np.std(sample["audio"]) > 0.0, "Audio is empty"
    assert np.sum(np.isnan(sample["audio"])) == 0, "Audio contains nan"
    assert len(sample["output"]) > 0, "Output is empty"
    assert sample["output"] != "nan", "Output is nan"
    assert "Audio" in sample["instruction"], "Instruction does not contain 'Audio'"
    assert isinstance(sample["metadata"], str), "Metadata is not a string"
    assert len(sample["file_name"]) > 0, "File name is empty"
    assert len(sample["license"]) > 0, "License is empty"
    assert len(sample["task"]) > 0, "Task is empty"
    return sample


def prepare_audio_sample_for_beans0(row: dict, remove_inaturalist: bool = False) -> dict[str, Any]:
    # Handle metadata differently based on its type
    if "metadata" in row:
        # Check if metadata is already a string that needs parsing
        if isinstance(row["metadata"], str):
            try:
                row["metadata"] = json.loads(row["metadata"])
            except json.JSONDecodeError:
                row["metadata"] = {}
        # If it's not a string or dict, convert to dict
        elif not isinstance(row["metadata"], dict):
            row["metadata"] = {}
    else:
        row["metadata"] = {}

    # if we encounter iNaturalist as the 'source_dataset', then we cant add audio to the dataset
    if remove_inaturalist and row["source_dataset"] == "iNaturalist":
        # Create silent audio data (array of zeros)
        # num_samples = int(0.1 * 16000)
        # audio_data = np.zeros(num_samples, dtype=np.float64)
        audio_data = [0.0]
        duration = 0
        sr = 0
    else:
        # Read audio file
        audio_data, sr = read_audio(row["file_path"])
        # compute duration
        duration = len(audio_data) / sr
        if np.std(audio_data) == 0.0:
            print(f"WARNING: Audio is empty for sample {row['id']}, filename {row['file_name']}")

    row["metadata"]["duration"] = duration
    row["metadata"]["sample_rate"] = sr

    row["metadata"] = json.dumps(row["metadata"])

    if "file_path" in row:
        del row["file_path"]

    if not isinstance(row["created_at"], str):
        row["created_at"] = row["created_at"].isoformat()

    # Store as bytes in memory
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_data, sr, format="WAV")

    row["output"] = "None" if (pd.isna(row["output"]) or row["output"] == "nan") else row["output"]

    sample = {"audio": audio_data, **row}

    try:
        validate_sample(sample)
    except AssertionError as e:
        print(f"Validation failed for sample {row['id']}: {e}")
        raise e

    return {"audio.wav": audio_buffer.getvalue(), "metadata.json": json.dumps(row)}


def checks(metadata_df: pd.DataFrame, original_paths_df: pd.DataFrame):
    assert len(metadata_df) == len(original_paths_df)
    assert metadata_df.instruction.isnull().sum() == 0
    assert metadata_df.file_name.isnull().sum() == 0
    assert metadata_df.output.isnull().sum() == 0
    assert metadata_df.source_dataset.isnull().sum() == 0


def main():
    parser = argparse.ArgumentParser(
        description="Create a sharded Beans0 dataset in webdataset tar file format from raw audio files"
    )
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to the metadata CSV file")
    parser.add_argument(
        "--dataset_path",
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
    parser.add_argument("--log_every", type=int, default=50, help="Log every n samples")

    args = parser.parse_args()

    # Load metadata
    # metadata_df = pd.read_csv(args.metadata_path)
    # metadata_df["output"] = metadata_df["output"].astype(str)  # convert to string
    # metadata_df["instruction"] = metadata_df["instruction"].astype(str)  # convert to string

    # # Add file paths to metadata
    # original_paths_df = pd.read_csv(args.original_paths_file)
    # metadata_df["file_path"] = original_paths_df["path"]  # this will be dropped after sharding

    metadata_df = pd.read_json(args.metadata_path, lines=True, orient="records")
    original_paths_df = pd.read_json(args.original_paths_file, lines=True, orient="records")
    metadata_df["file_path"] = original_paths_df["path"]  # this will be dropped after sharding

    # shuffle the rows, the ensures equal sized shards
    metadata_df = metadata_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Perform checks
    checks(metadata_df, original_paths_df)

    sample_prep_function = partial(prepare_audio_sample_for_beans0, remove_inaturalist=args.remove_inaturalist)

    output_path = os.path.join(args.dataset_path, args.version)

    # write new dataset config file
    beans0_cfg.version = args.version.replace("v", "")
    beans0_cfg.update_changelog(args.changelog)

    # Create a sharded dataset
    _ = create_sharded_dataset(
        metadata_df,
        output_path,
        sample_prep_function,
        num_samples_per_shard=args.num_samples_per_shard,
        num_workers=args.num_workers,
        shard_type="webdataset",
        dataset_config=beans0_cfg,
        save_metadata_as="metadata.jsonl",
        merge_data_and_metadata=True,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
