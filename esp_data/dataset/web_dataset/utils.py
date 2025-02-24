import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import webdataset as wds
from tqdm import tqdm

import esp_data.file_io.functional as F
from esp_data.paths import AnyPath, is_cloud_path, is_local_path
from esp_data.utils import make_simple_logger

logger = make_simple_logger(name="web_dataset_utils")


def _make_file_opener(file_path: str | AnyPath) -> callable:
    """Make a file opener function for WebDataset"""
    file_path = AnyPath(file_path)

    if is_local_path(file_path):
        file_path.touch(exist_ok=True)
        return open(file_path, "wb")

    if is_cloud_path(file_path):
        return partial(F.open_file, mode="wb", use_fs=True)


def write_shard(metadata: pd.DataFrame, output_path: AnyPath, shard_id: int, sample_prep_function: Callable) -> dict:
    """Write a shard of samples to a WebDataset shard."""
    results = {"chunk_id": shard_id, "processed_samples": [], "failed": []}

    shard_path = (
        output_path / f"shard_{shard_id:05d}%s.tar"
    )  # an additional zero will be added to the shard_id by webdataset ShardWriter

    # here we're setting maxcount and maxsize to very large values because we want all files in this
    # batch in ONE shard.
    # The actual number of shards for the dataset is set in the self.shard_size variable
    sink = wds.ShardWriter(
        str(shard_path),
        maxcount=1e12,
        maxsize=1e12,
        opener=_make_file_opener(shard_path),
    )

    for _, row in metadata.iterrows():
        shard_data = sample_prep_function(row)
        sample_id = str(row["id"])
        try:
            sample = {
                "__key__": sample_id,
                **shard_data,
            }
            sink.write(sample)

            # Track successful sample with shard info
            results["processed_samples"].append(
                {"id": sample_id, "shard_path": f"shard_{shard_id:05d}0.tar", "shard_id": shard_id}
            )

        except Exception as e:
            logger.error(f"Error processing sample {sample_id}: {str(e)}")
            results["failed_ids"].append(sample_id)

    sink.close()

    return results


def _save_checkpoint(output_path: AnyPath, completed_chunks: dict, metadata_df: pd.DataFrame):
    """Save checkpoint information"""
    checkpoint_data = {
        "completed_chunks": completed_chunks,
        "metadata_hash": int(pd.util.hash_pandas_object(metadata_df).sum()),
    }

    # can write to a cloud path
    with (output_path / "checkpoint.json").open("w") as f:
        json.dump(checkpoint_data, f)


def _load_checkpoint(output_path: AnyPath, metadata_df: pd.DataFrame) -> Optional[dict]:
    """Load checkpoint if it exists and is valid"""
    checkpoint_path = output_path / "checkpoint.json"

    if not checkpoint_path.exists():
        return None

    try:
        with checkpoint_path.open("r") as f:
            checkpoint_data = json.load(f)

        # Verify metadata hasn't changed
        current_hash = int(pd.util.hash_pandas_object(metadata_df).sum())
        if checkpoint_data["metadata_hash"] != current_hash:
            logger.warning("Metadata has changed since last checkpoint. Starting fresh.")
            return None

        return checkpoint_data

    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return None


def create_sharded_dataset(
    metadata_df: pd.DataFrame,
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
    output_path = AnyPath(output_path)
    # Create output directory if it doesn't exist
    # output_path.mkdir(exist_ok=True, parents=True)

    if metadata_df.empty:
        logger.error("Metadata is empty. Nothing to process.")
        return

    # Load checkpoint if exists
    checkpoint_data = _load_checkpoint(output_path, metadata_df)
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

    process_chunk_partial = partial(write_shard, output_path=output_path, sample_prep_function=sample_prep_function)

    # Process chunks in parallel with progress bar
    processed_samples = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_chunk_partial, metadata=chunk, shard_id=chunk_id)
            for chunk_id, chunk in chunks_to_process
        ]

        # Track progress with tqdm
        with tqdm(total=len(chunks_to_process), desc="Processing shards") as pbar:
            for future in futures:
                result = future.result()
                chunk_id = result["chunk_id"]

                # Collect processed samples info
                processed_samples.extend(result["processed_samples"])

                # Update checkpoint
                completed_chunks[str(chunk_id)] = result
                _save_checkpoint(output_path, completed_chunks, metadata_df)

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
    total_failed = sum(len(chunk["failed"]) for chunk in completed_chunks.values())

    logger.info(f"""
    Processing completed:
    - Total files processed successfully: {total_successful}
    - Total files failed: {total_failed}
    - Success rate: {(total_successful / (total_successful + total_failed)) * 100:.2f}%
    """)

    return metadata_df


def get_item_from_dataset(
    idx: int,
    dataset_path: str | AnyPath,
    data_processor: Callable,
    metadata_df: pd.DataFrame = None,
) -> dict[str, dict[str, Any]]:
    """Get a single item from the dataset using metadata lookup.

    Args:
        idx: Integer index to get from metadata DataFrame

    Returns:
        Dictionary containing the sample data (usually raw data and metadata)
    """
    if metadata_df is None:
        metadata_df = pd.read_parquet(AnyPath(dataset_path) / "metadata.parquet")

    # Get row from metadata
    row = metadata_df.iloc[idx]
    shard_path = row["shard_path"]
    sample_id = str(row["id"])

    # Load the specific shard
    ds = wds.WebDataset(str(AnyPath(dataset_path) / shard_path))

    # Find the specific sample by id
    for sample in ds:
        if sample["__key__"] == sample_id:
            return data_processor(sample)

    raise ValueError(f"Sample {sample_id} not found in shard {shard_path}")


def get_batch(
    indices: list[int],
    dataset_path: str | AnyPath,
    data_processor: Callable,
    metadata_df: pd.DataFrame = None,
) -> list[dict[str, dict[str, Any]]]:
    """Get a batch of items using metadata lookup.

    Args:
        indices: List of indices to get from metadata DataFrame

    Returns:
        list[dict[str, dict[str, Any]]]: List of dictionaries containing:
    """
    batch = []
    for idx in indices:
        item = get_item_from_dataset(idx, dataset_path, data_processor, metadata_df)
        batch.append(item)

    return batch
