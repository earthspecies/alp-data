import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd
from beans_cfg import beans0_cfg
from tqdm import tqdm

from esp_data.dataset.shard_creator import write_arrow_shard
from esp_data.dataset.web_dataset.utils import load_checkpoint, save_checkpoint
from esp_data.file_io.parsers import read_audio_bytes_from_path
from esp_data.paths import AnyPath, is_cloud_path
from esp_data.utils import make_simple_logger

logger = make_simple_logger("sharded_arrow_dataset_creator")


def prepare_audio_sample_for_beans0(row: dict, remove_inaturalist: bool = True) -> dict[str, Any]:
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
        audio_data, sr = read_audio_bytes_from_path(row["file_path"])
        # compute duration
        duration = len(audio_data) / sr

    row["metadata"]["duration"] = duration
    row["metadata"]["sample_rate"] = sr

    if "file_path" in row:
        del row["file_path"]

    return {"audio": audio_data, **row}


def checks(metadata_df: pd.DataFrame, original_paths_df: pd.DataFrame):
    assert len(metadata_df) == len(original_paths_df)
    assert metadata_df.instruction.isnull().sum() == 0
    assert metadata_df.file_name.isnull().sum() == 0
    assert metadata_df.output.isnull().sum() == 0
    assert metadata_df.source_dataset.isnull().sum() == 0


def create_sharded_dataset(
    metadata_df: pd.DataFrame | list[dict],
    output_path: str | AnyPath,
    sample_prep_function: Callable,
    num_samples_per_shard: int = 1000,
    num_workers: int = 4,
    storage_options: dict = None,
    shard_type: str = "arrow",
):
    """Create sharded dataset from information in metadata dataframe and a sample prep function,
    in parallel with checkpointing.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing metadata for samples, importantly the 'id' column.
            the "file_name" column is also required if the sample_prep_function requires it.
        data_root (str | AnyPath): Path to the root directory containing raw data (e.g. audio files)
        output_path (str | AnyPath): Path to save the sharded dataset
        sample_prep_function (Callable): Function to prepare a sample for sharding
        num_samples_per_shard (int, optional): Number of samples per shard. Defaults to 1000.
        num_workers (int, optional): Number of workers for parallel processing. Defaults to 4.
        storage_options (dict, optional): Storage options for reading and writing files. Defaults to None.
        shard_type (str, optional): Type of sharded dataset to create. Defaults to "arrow".

    """
    t0 = time.time()
    output_path = AnyPath(output_path)

    if metadata_df.empty:
        logger.error("Metadata is empty. Nothing to process.")
        return

    # Load checkpoint if exists
    checkpoint_data = load_checkpoint(output_path, metadata_df)
    completed_chunks = checkpoint_data["completed_chunks"] if checkpoint_data else {}

    # Calculate number of shards needed
    num_samples = len(metadata_df)
    num_shards = int(np.ceil(num_samples / num_samples_per_shard))

    # Split metadata into chunks for parallel processing
    chunks = np.array_split(metadata_df, num_shards)

    # Filter out completed chunks
    chunks_to_process = [(idx, chunk) for idx, chunk in enumerate(chunks) if str(idx) not in completed_chunks]

    if not chunks_to_process:
        logger.info("All chunks already processed. Nothing to do.")
        return

    # Create partial function with fixed arguments
    process_chunk_partial = partial(
        write_arrow_shard, output_path=output_path, arrow_prep_function=sample_prep_function, format=shard_type
    )

    # Process chunks in parallel with progress bar
    processed_samples = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_chunk_partial, batch=chunk, shard_id=chunk_id)
            for chunk_id, chunk in chunks_to_process
        ]

        # Track progress with tqdm
        with tqdm(total=len(chunks_to_process), desc="Processing shards") as pbar:
            for future in futures:
                result = future.result()
                chunk_id = result["shard_id"]

                # Collect processed samples info
                processed_samples.extend(result["processed_samples"])

                # Update checkpoint
                completed_chunks[str(chunk_id)] = result
                save_checkpoint(output_path, completed_chunks, metadata_df)

                pbar.update(1)

    # Create DataFrame with shard information
    shard_info_df = pd.DataFrame(processed_samples)

    # Merge shard information with original metadata
    metadata_df = metadata_df.merge(shard_info_df[["id", "shard_path", "shard_id"]], on="id", how="left")

    # drop any "file_path" column
    if "file_path" in metadata_df.columns:
        metadata_df.drop(columns=["file_path"], inplace=True)

    # Save updated metadata as parquet
    metadata_df.to_parquet(str(output_path / "metadata.parquet"), storage_options=storage_options)

    # Report final statistics
    total_successful = len(processed_samples)
    total_failed = sum(len(chunk["failed_ids"]) for chunk in completed_chunks.values())

    tend = time.time()
    logger.info(f"""
    Processing completed:
    - Total files processed successfully: {total_successful}
    - Total files failed: {total_failed}
    - Success rate: {(total_successful / (total_successful + total_failed)) * 100:.2f}%
    - Total time taken: {tend - t0:.2f} seconds
    """)

    return metadata_df


def main():
    parser = argparse.ArgumentParser(description="Create a sharded Beans0 dataset from raw audio files")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to the metadata CSV file")
    parser.add_argument(
        "--arrow_dataset_path",
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
    parser.add_argument("--shard_type", type=str, default="arrow", help="Type of sharded dataset to create")
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

    output_path = os.path.join(args.arrow_dataset_path, args.version)
    if is_cloud_path(output_path):
        storage_options = {"project": os.getenv("GCP_DEFAULT_PROJECT")}
    else:
        storage_options = None

    # Create an AudioDataset instance
    create_sharded_dataset(
        metadata_df,
        output_path,
        sample_prep_function,
        num_samples_per_shard=args.num_samples_per_shard,
        num_workers=args.num_workers,
        storage_options=storage_options,
        shard_type=args.shard_type,
    )

    # write new dataset config file
    beans0_cfg.version = args.version.replace("v", "")
    beans0_cfg.changelog = args.changelog
    with AnyPath(os.path.join(output_path, "dataset_config.json")).open("w") as fp:
        json.dump(beans0_cfg.to_dict(make_serializable=True), fp)

    # Write a README using the confg description and changelog
    with AnyPath(os.path.join(output_path, "README.md")).open("w") as fp:
        fp.write(f"# Beans0\n\n## Version {args.version}\n\n")
        fp.write(f"{beans0_cfg.description}\n\n")
        fp.write(f"### Changelog\n\n{args.changelog}\n")

    # test the dataset
    from datasets import load_dataset

    ds = load_dataset(
        args.shard_type, data_files=str(AnyPath(output_path) / f"*{args.shard_type}"), split="train", streaming=True
    )
    print(ds.column_names)
    print(f"Number of samples = {len(ds)}")

    audio = ds[0]["audio"]
    print(f"Source dataset = {ds[0]['source_dataset']}")
    print(f"Audio shape = {audio.shape}")
    print(f"Audio mean = {audio.mean(): .3f}, and std = {audio.std(): .3f}")


if __name__ == "__main__":
    main()
