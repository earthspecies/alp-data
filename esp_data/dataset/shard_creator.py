"""
Modular functions to create sharded datasets in both WebDataset (tar) and Arrow formats
"""

import hashlib
import json
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import webdataset as wds
from datasets import Dataset
from pydantic import BaseModel

from esp_data.paths import AnyPath
from esp_data.utils import make_simple_logger

from .utils import _make_file_opener

logger = make_simple_logger(name="shard_creator_module")


def validate_batch(batch: list[dict], validation_model: Optional[type[BaseModel]] = None) -> list[dict]:
    """
    Validate a batch of data using a Pydantic model.

    Args:
        batch: list of dictionaries containing sample data
        validation_model: Optional pydantic model for validation

    Returns:
        list of validated dictionaries (filters out invalid items)
    """
    if validation_model is None:
        return batch

    valid_items = []
    for item in batch:
        try:
            # Validate with pydantic model
            validated = validation_model(**item)
            valid_items.append(validated.dict())
        except Exception as e:
            # Skip invalid items
            logger.error(f"Validation failed for item: {e}")
            continue

    return valid_items


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
        maxcount=1e12,  # Set very high to ensure all samples go in one shard
        maxsize=1e12,  # Set very high
        opener=_make_file_opener(shard_path),
    )

    # Process each sample
    iterator = batch.iterrows() if isinstance(batch, pd.DataFrame) else enumerate(batch)
    for i, item in enumerate(iterator):
        # get item and sample_id
        if isinstance(item, pd.Series):
            item = item.to_dict()

        sample_id = str(item.get("id", i))

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

    # Close the shard
    sink.close()

    return results


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

    for i, item in iterator:
        # get item and sample_id
        if isinstance(item, pd.Series):
            item = item.to_dict()

        sample_id = str(item["id"] if "id" in item else i)

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

    if prepared_data:
        # Create Arrow table
        # We assume arrow_prep_function returns a dict with field names
        # matching the Arrow schema
        first_sample = prepared_data[0]

        # Construct schema based on the first sample
        fields = []
        data_arrays = {}

        for field_name, value in first_sample.items():
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
                        field_type = pa.list_(pa.float32())  # Default
                else:
                    # Python lists
                    if all(isinstance(x, float) for x in value):
                        field_type = pa.list_(pa.float32())
                    elif all(isinstance(x, int) for x in value):
                        field_type = pa.list_(pa.int32())
                    else:
                        field_type = pa.list_(pa.string())
            elif isinstance(value, dict):
                # For metadata dictionaries
                field_type = pa.struct(
                    {
                        k: pa.string()
                        if isinstance(v, str)
                        else pa.float32()
                        if isinstance(v, float)
                        else pa.int32()
                        if isinstance(v, int)
                        else pa.string()
                        for k, v in value.items()
                    }
                )
            elif isinstance(value, str):
                field_type = pa.string()
            elif isinstance(value, float):
                field_type = pa.float32()
            elif isinstance(value, int):
                field_type = pa.int32()
            elif isinstance(value, bool):
                field_type = pa.bool_()
            else:
                # Default to string for unknown types
                field_type = pa.string()

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

    # Process batch data
    prepared_data = []
    iterator = batch.iterrows() if isinstance(batch, pd.DataFrame) else enumerate(batch)

    for i, item in iterator:
        # get item and sample_id
        if isinstance(item, pd.Series):
            item = item.to_dict()

        sample_id = str(item["id"] if "id" in item else i)
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

    if prepared_data:
        ds = Dataset.from_list(prepared_data)
        ds.save_to_disk(shard_path, storage_options=storage_options, num_proc=num_proc, num_shards=1)

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


def save_checkpoint(output_path: AnyPath, completed_chunks: dict, metadata: Any) -> None:
    """
    Save checkpoint information for any type of metadata.

    Args:
        output_path: Path to save the checkpoint
        completed_chunks: Dictionary of completed chunks
        metadata: Metadata of any serializable type
    """
    checkpoint_data = {
        "completed_chunks": completed_chunks,
        "metadata_hash": compute_metadata_hash(metadata),
    }

    # can write to a cloud path
    with (output_path / "checkpoint.json").open("w") as f:
        json.dump(checkpoint_data, f)


def load_checkpoint(output_path: AnyPath, metadata: Any) -> Optional[dict]:
    """
    Load checkpoint if it exists and is valid for any type of metadata.

    Args:
        output_path: Path to the checkpoint
        metadata: Current metadata to compare against saved checkpoint

    Returns:
        Optional[dict]: Checkpoint data if valid, None otherwise
    """
    checkpoint_path = output_path / "checkpoint.json"

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
