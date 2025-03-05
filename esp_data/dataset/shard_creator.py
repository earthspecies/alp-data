"""
Modular functions to create sharded datasets in both WebDataset (tar) and Arrow formats
"""

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Callable, Iterable, Optional

import colorama
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import webdataset as wds
from colorama import Fore, Style
from datasets import Dataset
from tqdm.auto import tqdm

import esp_data.file_io.functional as F
from esp_data.config import DatasetConfig
from esp_data.paths import AnyPath, make_storage_options
from esp_data.utils import make_simple_logger

from .utils import _make_file_opener

logger = make_simple_logger(name="shard_creator_module")

# Initialize colorama for cross-platform colored terminal output
colorama.init()

# Define color schemes for different progress levels
BATCH_COLOR = Fore.BLUE
SHARD_COLOR = Fore.GREEN
SAMPLE_COLOR = Fore.CYAN
SUCCESS_COLOR = Fore.GREEN
ERROR_COLOR = Fore.RED
RESET_COLOR = Style.RESET_ALL

PBAR_EVERY = 10  # Update progress bar every N samples


def write_webdataset_shard(
    batch: Iterable[dict] | pd.DataFrame | pd.Series,
    shard_id: int,
    output_path: str | AnyPath,
    sample_prep_function: Optional[Callable] = None,
) -> dict:
    """
    Write a batch of samples to a WebDataset shard.

    Args:
        batch: list of dictionaries or dataframe or series containing sample data
        shard_id: ID for this shard
        output_path: Path to save the shard
        sample_prep_function: Function to prepare a sample for the WebDataset format

    Returns:
        Dictionary with processing results
    """
    results = {"shard_id": shard_id, "processed_samples": [], "failed_ids": []}

    # Create shard path
    output_path = AnyPath(output_path)
    shard_path = output_path / f"shard_%s{shard_id:05d}.tar"

    # Initialize shard writer
    sink = wds.ShardWriter(
        str(shard_path),
        maxcount=100_000_000_000,  # Set very high to ensure all samples go in one shard
        maxsize=100_000_000_000,
        opener=_make_file_opener(shard_path),
    )

    # Process each sample
    iterator = batch.iterrows() if isinstance(batch, pd.DataFrame) else enumerate(batch)
    with tqdm(total=len(batch), desc=f"Shard {shard_id:05d}", position=None, leave=False, ncols=90) as pbar_samples:
        for i, item in iterator:
            # get item and sample_id
            if isinstance(item, pd.Series):
                item = item.to_dict()

            sample_id = str(item.get("id", i))

            if i % PBAR_EVERY == 0:  # Only update description occasionally to reduce output
                pbar_samples.set_description(f"Shard {shard_id:05d} - Sample {sample_id}")

            try:
                # Prepare the sample
                if sample_prep_function is None:
                    shard_data = item
                else:
                    shard_data = sample_prep_function(item)

                # Write to shard
                sample = {
                    "__key__": sample_id,
                    **shard_data,
                }
                sink.write(sample)

                # Track successful sample
                results["processed_samples"].append(
                    {"id": sample_id, "shard_path": f"shard_0{shard_id:05d}.tar", "shard_id": shard_id}
                )

            except Exception as e:
                logger.error(f"Error processing sample {sample_id}: {e}")
                results["failed_ids"].append(sample_id)
                # if Exception is KeyboardInterrupt then raise and exit
                if isinstance(e, KeyboardInterrupt):
                    raise e

            pbar_samples.update(1)

    # Close the shard
    sink.close()

    return results


def determine_pa_field_type(value, default_float_type: str = "float32") -> pa.DataType:
    float_type = pa.float32() if default_float_type == "float32" else pa.float64()

    def build_struct(value):
        return pa.struct(
            {
                k: pa.string()
                if isinstance(v, str)
                else float_type
                if isinstance(v, float)
                else pa.int32()
                if isinstance(v, int)
                else build_struct(v)
                if isinstance(v, dict)
                else pa.string()
                for k, v in value.items()
            }
        )

    # Determine PyArrow data type based on value type
    if isinstance(value, (list, np.ndarray)):
        # For arrays/lists
        if isinstance(value, np.ndarray):
            if value.dtype == np.float32:
                field_type = pa.list_(pa.float32())
            elif value.dtype == np.float64:
                field_type = pa.list_(pa.float64())
            elif value.dtype == np.int16:
                field_type = pa.list_(pa.int16())
            elif value.dtype == np.int32:
                field_type = pa.list_(pa.int32())
            else:
                field_type = pa.list_(float_type)  # Default
        else:
            # Python lists
            if all(isinstance(x, float) for x in value):
                field_type = pa.list_(float_type)
            elif all(isinstance(x, int) for x in value):
                field_type = pa.list_(pa.int32())
            else:
                field_type = pa.list_(pa.string())
    elif isinstance(value, dict):
        # For metadata dictionaries
        field_type = build_struct(value)
    elif isinstance(value, str):
        field_type = pa.string()
    elif isinstance(value, float):
        field_type = float_type
    elif isinstance(value, int):
        field_type = pa.int32()
    elif isinstance(value, bool):
        field_type = pa.bool_()
    else:
        # Default to string for unknown types
        field_type = pa.string()

    return field_type


def write_arrow_shard(
    batch: Iterable[dict] | pd.DataFrame | pd.Series,
    shard_id: int,
    output_path: str | AnyPath,
    sample_prep_function: Optional[Callable] = None,
    format: str = "parquet",
) -> dict:
    """
    Write a batch of samples to an Arrow / Parquet shard.

    Args:
        batch: list of dictionaries or dataframe or series containing sample data
        shard_id: ID for this shard
        output_path: Path to save the shard
        sample_prep_function: Function to prepare a sample for Arrow format
        format: Output format for the sshard (parquet or arrow)

    Returns:
        Dictionary with processing results
    """
    results = {"shard_id": shard_id, "processed_samples": [], "failed_ids": []}

    # Create shard path
    output_path = AnyPath(output_path)
    shard_path = output_path / (f"shard_{shard_id:06d}." + format)

    # Process batch data
    prepared_data = []
    iterator = batch.iterrows() if isinstance(batch, pd.DataFrame) else enumerate(batch)

    with tqdm(total=len(batch), desc=f"Shard {shard_id:05d}", position=None, leave=False, ncols=90) as pbar_samples:
        for i, item in iterator:
            # get item and sample_id
            if isinstance(item, pd.Series):
                item = item.to_dict()

            sample_id = str(item["id"] if "id" in item else i)
            if i % PBAR_EVERY == 0:
                pbar_samples.set_description(f"Shard {shard_id:05d} - Sample {sample_id}")

            try:
                # Prepare the sample
                if sample_prep_function is None:
                    shard_data = item
                else:
                    shard_data = sample_prep_function(item)

                prepared_data.append(shard_data)

                # Track successful sample
                results["processed_samples"].append(
                    {"id": sample_id, "shard_path": f"shard_{shard_id:06d}.arrow", "shard_id": shard_id}
                )

            except Exception as e:
                logger.error(f"Error processing sample {sample_id} for Arrow: {e}")
                results["failed_ids"].append(sample_id)
                # if Exception is KeyboardInterrupt then raise and exit
                if isinstance(e, KeyboardInterrupt):
                    raise e

            pbar_samples.update(1)

    if prepared_data:
        # Create Arrow table
        # We assume arrow_prep_function returns a dict with field names
        # matching the Arrow schema
        first_sample = prepared_data[0]

        # Construct schema based on the first sample
        fields = []
        data_arrays = {}

        for field_name, value in first_sample.items():
            field_type = determine_pa_field_type(value)
            fields.append(pa.field(field_name, field_type))
            data_arrays[field_name] = [sample.get(field_name) for sample in prepared_data]

        # Create schema and table
        schema = pa.schema(fields)
        arrays = [pa.array(data_arrays[field.name]) for field in schema]
        table = pa.Table.from_arrays(arrays, schema=schema)

        # Write to file
        opener = _make_file_opener(shard_path)
        with opener(str(shard_path)) as f:
            if format == "parquet":
                # Write as Parquet
                pq.write_table(table, f)
            else:
                with pa.ipc.new_file(f, schema) as writer:
                    writer.write_table(table)

    return results


def write_huggingface_shard(
    batch: Iterable[dict] | pd.DataFrame | pd.Series,
    shard_id: int,
    output_path: str | AnyPath,
    sample_prep_function: Optional[Callable] = None,
    storage_options: Optional[dict] = None,
    num_proc: int = 1,
):
    """
    Write a batch of samples to an Arrow shard in the Hugging Face format.

    Args:
        batch: list of dictionaries containing sample data
        shard_id: ID for this shard
        output_path: Path to save the shard
        sample_prep_function: Function to prepare a sample for dataset Arrow format
        storage_options: Optional storage options for saving the dataset
        num_proc: Number of processes to use for saving the dataset
        num_shards: Number of shards to split the dataset into

    """
    results = {"shard_id": shard_id, "processed_samples": [], "failed_ids": []}

    # Create shard path
    output_path = AnyPath(output_path)
    shard_path = output_path / (f"shard_{shard_id:06d}")
    storage_options = make_storage_options(output_path) if storage_options is None else storage_options

    # Process batch data
    prepared_data = []
    iterator = batch.iterrows() if isinstance(batch, pd.DataFrame) else enumerate(batch)

    with tqdm(total=len(batch), desc=f"Shard {shard_id:05d}", position=None, leave=False, ncols=90) as pbar_samples:
        for i, item in iterator:
            # get item and sample_id
            if isinstance(item, pd.Series):
                item = item.to_dict()

            sample_id = str(item["id"] if "id" in item else i)
            if i % PBAR_EVERY == 0:
                pbar_samples.set_description(f"Shard {shard_id:05d} - Sample {sample_id}")

            try:
                # Prepare the sample
                if sample_prep_function is None:
                    shard_data = item
                else:
                    shard_data = sample_prep_function(item)

                prepared_data.append(shard_data)

                # Track successful sample
                results["processed_samples"].append(
                    {"id": sample_id, "shard_path": f"shard_{shard_id:06d}.arrow", "shard_id": shard_id}
                )

            except Exception as e:
                logger.error(f"Error processing sample {sample_id} for HF: {e}")
                results["failed_ids"].append(sample_id)
                # if Exception is KeyboardInterrupt then raise and exit
                if isinstance(e, KeyboardInterrupt):
                    raise e

            pbar_samples.update(1)

    if prepared_data:
        ds = Dataset.from_list(prepared_data)
        ds.save_to_disk(shard_path, storage_options=storage_options, num_proc=num_proc, num_shards=1)
        # this saves a few files in a folder, like state.json, data*.arrow, dataset_info.json.
        # we just want the .arrow file
        arrow_file = F.list_files(shard_path, pattern="*.arrow")[0]
        F.move_file(arrow_file, output_path / (f"shard_{shard_id:06d}.arrow"))
        F.delete_dir(shard_path)

    return results


def write_shard(
    batch: list[dict],
    shard_id: int,
    output_path: str | AnyPath,
    sample_prep_function: Callable,
    output_format: str = "webdataset",
    **kwargs,
) -> dict:
    """
    Write a batch of samples to a shard in the specified format.

    Args:
        batch: list of dictionaries containing sample data
        shard_id: ID for this shard
        output_path: Path to save the shard
        sample_prep_function: Function to prepare a sample for the specified format
        output_format: Output format for the shard (webdataset or arrow)
    """
    if output_format == "webdataset":
        return write_webdataset_shard(batch, shard_id, output_path, sample_prep_function, **kwargs)
    elif output_format in ["arrow", "parquet"]:
        return write_arrow_shard(batch, shard_id, output_path, sample_prep_function, format=output_format, **kwargs)
    elif output_format == "hf":
        return write_huggingface_shard(batch, shard_id, output_path, sample_prep_function, **kwargs)
    else:
        raise ValueError(
            f"Unsupported output format: {output_format}, must be 'webdataset', 'arrow', 'parquet' or 'hf'"
        )


def compute_metadata_hash(metadata: pd.DataFrame | list[dict] | dict | Any) -> int:
    """
    Compute a hash for metadata of various types.

    Args:
        metadata: Can be a pandas DataFrame, list of dictionaries, dictionary, or other serializable data

    Returns:
        int: A hash value representing the metadata
    """
    if isinstance(metadata, pd.DataFrame):
        # Use pandas' built-in hashing for DataFrames
        return int(pd.util.hash_pandas_object(metadata).sum())

    # For other types, convert to JSON and hash that
    try:
        # Convert to JSON string in a deterministic way (sorted keys)
        metadata_str = json.dumps(metadata, sort_keys=True)
        # Use hash to get a consistent int value
        hash_object = hashlib.md5(metadata_str.encode())
        # Convert first 8 bytes of MD5 to int (to keep similar scale to pandas hash)
        return int.from_bytes(hash_object.digest()[:8], byteorder="big")
    except (TypeError, ValueError):
        # If the metadata can't be converted to JSON, use its string representation
        logger.warning("Could not JSON-serialize metadata, using string representation for hashing")
        metadata_str = str(metadata)
        hash_object = hashlib.md5(metadata_str.encode())
        return int.from_bytes(hash_object.digest()[:8], byteorder="big")


def save_checkpoint(
    output_path: AnyPath, completed_chunks: dict, metadata: Any, checkpoint_name: str = "checkpoint.json"
) -> None:
    """
    Save checkpoint information for any type of metadata.

    Args:
        output_path: Path to save the checkpoint
        completed_chunks: Dictionary of completed chunks
        metadata: Metadata of any serializable type
        checkpoint_name: Name of the checkpoint file

    """
    checkpoint_data = {
        "completed_chunks": completed_chunks,
        "metadata_hash": compute_metadata_hash(metadata),
    }

    # can write to a cloud path
    with (output_path / checkpoint_name).open("w") as f:
        json.dump(checkpoint_data, f)


def load_checkpoint(output_path: AnyPath, metadata: Any, checkpoint_name: str = "checkpoint.json") -> Optional[dict]:
    """
    Load checkpoint if it exists and is valid for any type of metadata.

    Args:
        output_path: Path to the checkpoint
        metadata: Current metadata to compare against saved checkpoint
        checkpoint_name: Name of the checkpoint file

    Returns:
        Optional[dict]: Checkpoint data if valid, None otherwise
    """
    checkpoint_path = output_path / checkpoint_name

    if not checkpoint_path.exists():
        return None

    try:
        with checkpoint_path.open("r") as f:
            checkpoint_data = json.load(f)

        # Verify metadata hasn't changed
        current_hash = compute_metadata_hash(metadata)
        if checkpoint_data["metadata_hash"] != current_hash:
            logger.warning("Metadata has changed since last checkpoint. Starting fresh.")
            return None

        return checkpoint_data

    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return None


def create_sharded_dataset(
    data: pd.DataFrame | Iterable[dict],
    output_path: str | AnyPath,
    sample_prep_function: Callable,
    num_samples_per_shard: int = 1000,
    num_workers: int = 4,
    storage_options: dict = None,
    shard_type: str = "arrow",
    dataset_config: DatasetConfig = None,
    save_metadata_as: str | None = None,
    merge_data_and_metadata: bool = False,
    **sharding_kwargs,
):
    """Create sharded dataset from information in a dataframe (or a Iterable[dict]) and a sample prep function,
    in parallel with checkpointing.

    Args:
        data (pd.DataFrame | Iterable[dict]): DataFrame containing data for samples. Please try and include an 'id' column.
        data_root (str | AnyPath): Path to the root directory containing raw data (e.g. audio files)
        output_path (str | AnyPath): Path to save the sharded dataset
        sample_prep_function (Callable): Function to prepare a sample for sharding
        num_samples_per_shard (int, optional): Number of samples per shard. Defaults to 1000.
        num_workers (int, optional): Number of workers for parallel processing. Defaults to 4.
        storage_options (dict, optional): Storage options for reading and writing files. Defaults to None.
        shard_type (str, optional): Type of sharded dataset to create. Defaults to "arrow".
        dataset_config (DatasetConfig, optional): Dataset configuration object to save with the data. Defaults to None.
        save_metadata_as (str | None, optional): File name to save updated metadata as. Defaults to None.
            Best practice is to save as 'metadata.parquet' or 'metadata.json'.
        merge_data_and_metadata (bool, optional): Merge data and metadata into a single DataFrame. Defaults to False.
        **sharding_kwargs: Additional keyword arguments for sharding function.

    """
    t0 = time.time()
    output_path = AnyPath(output_path)

    if len(data) == 0:
        logger.error("Metadata is empty. Nothing to process.")
        return

    if shard_type not in ["webdataset", "arrow", "parquet", "hf"]:
        raise ValueError(f"Unsupported shard type: {shard_type}. Must be 'webdataset', 'arrow', 'parquet' or 'hf'")

    # Load checkpoint if exists
    checkpoint_data = load_checkpoint(output_path, data)
    completed_chunks = checkpoint_data["completed_chunks"] if checkpoint_data else {}
    logger.info(f"Loaded checkpoint with {len(completed_chunks)} completed chunks.")

    # Calculate number of shards needed
    num_samples = len(data)
    num_shards = int(np.ceil(num_samples / num_samples_per_shard))

    # Split metadata into chunks for parallel processing
    chunks = np.array_split(data, num_shards)

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
        write_shard,
        output_path=output_path,
        sample_prep_function=sample_prep_function,
        output_format=shard_type,
        **sharding_kwargs,
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
        # Track progress with tqdm
        # Sequential processing with nested progress bars
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

    if merge_data_and_metadata:
        # Merge shard information with original metadata
        metadata_df = pd.DataFrame(data) if isinstance(data, Iterable) else data
        metadata_df = metadata_df.merge(shard_info_df[["id", "shard_path", "shard_id"]], on="id", how="left")
    else:
        metadata_df = shard_info_df

    # Save updated metadata as parquet
    if save_metadata_as:
        if "parquet" in save_metadata_as:
            metadata_df.to_parquet(str(output_path / save_metadata_as), index=False, storage_options=storage_options)
        elif "csv" in save_metadata_as:
            metadata_df.to_csv(str(output_path / save_metadata_as), index=False)
        elif "json" in save_metadata_as:
            metadata_df.to_json(str(output_path / save_metadata_as), orient="records", lines=True)
        elif "tsv" in save_metadata_as:
            metadata_df.to_csv(str(output_path / save_metadata_as), sep="\t", index=False)
        else:
            logger.warning(f"Unsupported metadata format: {save_metadata_as}. Skipping metadata save.")

    # save config
    if not dataset_config:
        logger.warning("No dataset config provided. Creating skeleton config..")
        dataset_config = DatasetConfig.from_skeleton()

    dataset_config.write_json(output_path / "dataset_config.json")
    dataset_config.generate_readme(output_path / "README.md")

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
    logger.info(f"\n{BATCH_COLOR}=== Processing Summary ==={RESET_COLOR}")
    logger.info(f"{SUCCESS_COLOR}- Total files processed successfully: {total_successful}{RESET_COLOR}")
    logger.info(f"{ERROR_COLOR}- Total files failed: {total_failed}{RESET_COLOR}")
    logger.info(f"{BATCH_COLOR}- Success rate: {success_rate:.2f}%{RESET_COLOR}")
    logger.info(f"{BATCH_COLOR}- Total time taken: {tend - t0:.2f} seconds{RESET_COLOR}")

    return metadata_df
