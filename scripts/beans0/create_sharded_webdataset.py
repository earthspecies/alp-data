import argparse
import io
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Callable

import colorama
import numpy as np
import pandas as pd
import soundfile as sf
from beans_cfg import beans0_cfg
from colorama import Fore, Style
from tqdm import tqdm

from esp_data.dataset.shard_creator import load_checkpoint, save_checkpoint, write_webdataset_shard
from esp_data.file_io.parsers import read_audio_bytes_from_path
from esp_data.paths import AnyPath, is_cloud_path
from esp_data.utils import make_simple_logger

# Initialize colorama for cross-platform colored terminal output
colorama.init()

# Define color schemes for different progress levels
BATCH_COLOR = Fore.BLUE
SHARD_COLOR = Fore.GREEN
SAMPLE_COLOR = Fore.CYAN
SUCCESS_COLOR = Fore.GREEN
ERROR_COLOR = Fore.RED
RESET_COLOR = Style.RESET_ALL

logger = make_simple_logger("sharded_web_dataset_creator")


def validate_sample(sample: dict, remove_inaturalist: bool = False):
    if not remove_inaturalist:
        assert np.std(sample["audio"]) > 0.0
    assert len(sample["output"]) > 0
    assert sample["output"] != "nan"
    assert "Audio" in sample["instruction"]
    assert isinstance(sample["metadata"], str)
    assert len(sample["file_name"]) > 0
    assert len(sample["license"]) > 0
    assert len(sample["task"]) > 0
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
        audio_data, sr = read_audio_bytes_from_path(row["file_path"])
        # compute duration
        duration = len(audio_data) / sr

    row["metadata"]["duration"] = duration
    row["metadata"]["sample_rate"] = sr

    row["metadata"] = json.dumps(row["metadata"])

    if "file_path" in row:
        del row["file_path"]

    # Store as bytes in memory
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_data, sr, format="WAV")

    row["output"] = "None" if (pd.isna(row["output"]) or row["output"] == "nan") else row["output"]

    sample = {"audio": audio_data, **row}

    try:
        validate_sample(sample)
    except AssertionError as e:
        logger.error(f"Validation failed for sample {row['id']}: {e}")
        raise e

    return {"audio.wav": audio_buffer.getvalue(), "metadata.json": json.dumps(row)}


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

    print(f"\n{BATCH_COLOR}=== Dataset Sharding Process ==={RESET_COLOR}")
    print(
        f"{BATCH_COLOR}Total samples: {num_samples}, Shards: {num_shards}, Samples per shard: {num_samples_per_shard}{RESET_COLOR}\n"
    )

    # Filter out completed chunks
    chunks_to_process = [(idx, chunk) for idx, chunk in enumerate(chunks) if str(idx) not in completed_chunks]

    if not chunks_to_process:
        logger.info("All chunks already processed. Nothing to do.")
        return

    # Create partial function with fixed arguments
    process_chunk_partial = partial(
        write_webdataset_shard,
        output_path=output_path,
        sample_prep_function=sample_prep_function,
    )

    # Process chunks in parallel with progress bar
    processed_samples = []
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_chunk_partial, batch=chunk, shard_id=chunk_id)
                for chunk_id, chunk in chunks_to_process
            ]

            # Track progress with tqdm
            with tqdm(
                total=len(chunks_to_process),
                desc=f"{SHARD_COLOR}Processing shards{RESET_COLOR}",
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (SHARD_COLOR, RESET_COLOR),
                ncols=100,
            ) as pbar:
                for future in futures:
                    result = future.result()
                    chunk_id = result["shard_id"]
                    processed_samples.extend(result["processed_samples"])

                    # Update checkpoint
                    completed_chunks[str(chunk_id)] = result
                    save_checkpoint(output_path, completed_chunks, completed_chunks)

                    # Update progress bar with shard info
                    pbar.set_postfix(
                        shard=f"{chunk_id:05d}",
                        success=len(result["processed_samples"]),
                        failed=len(result["failed_ids"]),
                    )
                    pbar.update(1)
    else:
        with tqdm(
            total=len(chunks_to_process),
            desc=f"{SHARD_COLOR}Processing shards{RESET_COLOR}",
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (SHARD_COLOR, RESET_COLOR),
            position=0,
            leave=True,
            ncols=100,
        ) as pbar_shards:
            for chunk_id, chunk in chunks_to_process:
                result = process_chunk_partial(batch=chunk, shard_id=chunk_id)
                processed_samples.extend(result["processed_samples"])

                # Update checkpoint
                completed_chunks[str(chunk_id)] = result
                save_checkpoint(output_path, completed_chunks, completed_chunks)

                # Update progress bar with shard info
                pbar_shards.set_postfix(
                    shard=f"{chunk_id:05d}", success=len(result["processed_samples"]), failed=len(result["failed_ids"])
                )
                pbar_shards.update(1)

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
    success_rate = (
        (total_successful / (total_successful + total_failed)) * 100 if (total_successful + total_failed) > 0 else 0
    )

    tend = time.time()
    logger.info(f"""
    Processing completed:
    - Total files processed successfully: {total_successful}
    - Total files failed: {total_failed}
    - Success rate: {(total_successful / (total_successful + total_failed)) * 100:.2f}%
    - Total time taken: {tend - t0:.2f} seconds
    """)
    print(f"\n{BATCH_COLOR}=== Processing Summary ==={RESET_COLOR}")
    print(f"{SUCCESS_COLOR}- Total files processed successfully: {total_successful}{RESET_COLOR}")
    print(f"{ERROR_COLOR}- Total files failed: {total_failed}{RESET_COLOR}")
    print(f"{BATCH_COLOR}- Success rate: {success_rate:.2f}%{RESET_COLOR}")
    print(f"{BATCH_COLOR}- Total time taken: {tend - t0:.2f} seconds{RESET_COLOR}")

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
    )

    # write new dataset config file
    beans0_cfg.version = args.version.replace("v", "")
    beans0_cfg.update_changelog(args.changelog)
    with AnyPath(os.path.join(output_path, "dataset_config.json")).open("w") as fp:
        json.dump(beans0_cfg.to_dict(make_serializable=True), fp)

    beans0_cfg.generate_readme(AnyPath(output_path) / "README.md")

    # test the dataset


if __name__ == "__main__":
    main()
