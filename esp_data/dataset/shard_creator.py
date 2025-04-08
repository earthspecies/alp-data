"""
Modular functions to create sharded datasets in both WebDataset (tar) and Arrow formats
"""

import hashlib
import json
import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Callable, Iterable, List, Optional, Union

import colorama
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import webdataset as wds
from colorama import Fore, Style
from datasets import Dataset
from tqdm.auto import tqdm

import esp_data.io.functional as F
from esp_data.config import DatasetConfig
from esp_data.config.project_config import default_shard_creator_cfg
from esp_data.io import AnyPath
from esp_data.utils import make_id

logger = logging.getLogger("esp_data")

# Initialize colorama for cross-platform colored terminal output
colorama.init()

# Define color schemes for different progress levels
BATCH_COLOR = Fore.BLUE
SHARD_COLOR = Fore.GREEN
SAMPLE_COLOR = Fore.CYAN
SUCCESS_COLOR = Fore.GREEN
ERROR_COLOR = Fore.RED
RESET_COLOR = Style.RESET_ALL

SHARD_TYPES = ["webdataset", "arrow", "parquet", "hf"]


def _error_handler(
    e: Exception, sample_id: str, error_handling: str = default_shard_creator_cfg.error_handling
) -> None:
    """Handle errors during sample processing."""
    if error_handling == "warn":
        logger.error(f"Error processing sample {sample_id}: {e}")
        logger.error(f"Exception traceback:\n{traceback.format_exc()}")
    elif error_handling == "raise":
        raise e
    elif error_handling == "ignore":
        pass

    if isinstance(e, KeyboardInterrupt):
        raise e


def write_webdataset_shard(
    batch: Union[Iterable[dict], pd.DataFrame, pd.Series],
    shard_id: int,
    output_path: Union[str, AnyPath],
    sample_prep_function: Optional[Callable] = None,
    log_every: int = default_shard_creator_cfg.log_every,
    error_handling: str = default_shard_creator_cfg.error_handling,
    shard_name: str = default_shard_creator_cfg.shard_name,
) -> dict:
    """
    Write a batch of samples to a WebDataset shard.

    Arguments
    ---------
    batch: Union[Iterable[dict], pd.DataFrame, pd.Series]
        Iterable of dictionaries or dataframe or series containing sample data
    shard_id: int
        ID for this shard
    output_path: Union[str, AnyPath],
        Path to save the shard
    sample_prep_function: Optional[Callable]
        Function to prepare a sample for the WebDataset format
    log_every: int
        Log progress every N samples
    error_handling: str
        How to handle errors ("warn", "raise", "ignore")

    Returns
    -------
        dict: Dictionary with processing results. Contains "processed_ids" and "failed_ids".
            "processed_ids" is a list of dicts, each containing the sample ID, shard ID, and shard path.
            "failed_ids" is a list of sample IDs that failed processing.

    Example
    -------
        >>> items = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]
        >>> results = write_webdataset_shard(items, 0, "/tmp/webds_output")
        # writing /tmp/webds_output/shard_000000.tar 0 0.0 GB 0
        >>> print(len(results["processed_ids"]))
        2
        >>> from pathlib import Path
        >>> Path("/tmp/webds_output/shard_000000.tar").exists()
        True
    """
    results = {"shard_id": shard_id, "processed_ids": [], "failed_ids": []}

    # Create shard path
    output_path = AnyPath(output_path)
    shard_path = output_path / f"{shard_name}_%s{shard_id:05d}.tar"  # one 0 added by the ShardWriter class

    # Initialize shard writer
    sink = wds.ShardWriter(
        str(shard_path),
        maxcount=100_000_000_000,  # Set very high to ensure all samples go in one shard
        maxsize=100_000_000_000,
        opener=lambda: AnyPath(shard_path).open("wb"),
    )

    # Process each sample
    iterator = batch.iterrows() if isinstance(batch, pd.DataFrame) else enumerate(batch)
    total_samples = len(batch)
    j = 0
    for _, item in iterator:
        # get item and sample_id
        if isinstance(item, pd.Series):
            item = item.to_dict()

        sample_id = str(item.get("id", make_id()))
        item["id"] = sample_id

        if (j + 1) % log_every == 0:
            logger.info(f"Shard {shard_id:05d} - Processing sample {j}/{total_samples} (id: {sample_id})")

        try:
            shard_data = sample_prep_function(item) if sample_prep_function else item

            # Write to shard
            sample = {
                "__key__": sample_id,
                **shard_data,
            }
            sink.write(sample)

            # Track successful sample
            results["processed_ids"].append({"id": sample_id, "shard_id": shard_id, "shard_path": str(shard_path)})

        except Exception as e:
            results["failed_ids"].append(sample_id)
            _error_handler(e, sample_id, error_handling)

        j += 1

    sink.close()
    logger.info(
        f"Finished shard {shard_id:06d} - Processed: {len(results['processed_ids'])}, Failed: {len(results['failed_ids'])}"
    )

    return results


def determine_pa_field_type(
    value: Any, default_float_type: str = default_shard_creator_cfg.pyarrow_default_float_type
) -> pa.DataType:
    """
    Determine the appropriate PyArrow data type for a given value.

    Arguments
    ---------
    value:
        The value to determine the type for
    default_float_type: str
        The default float type to use ("float32" or "float64")

    Returns
    -------
        pa.DataType: The PyArrow data type that best matches the input value

    Example
    -------
        >>> determine_pa_field_type(3.14, "float32")
        DataType(float)
        >>> determine_pa_field_type("hello")
        DataType(string)
        >>> determine_pa_field_type([1, 2, 3])
        ListType(list<item: int32>)
    """
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


def infer_schema_from_sample(
    sample: dict[str, Any], default_float_type: str = default_shard_creator_cfg.pyarrow_default_float_type
) -> pa.Schema:
    """Infer a PyArrow schema from a sample dictionary.

    Arguments
    ---------
    sample: dict[str, Any]
        A dictionary to infer the schema from
    default_float_type: str
        The default float type to use, either "float32" or "float64"

    Returns
    -------
        pa.Schema: The inferred schema

    Example
    -------
        >>> sample = {"id": "1", "value": 3.14, "count": 42}
        >>> schema = infer_schema_from_sample(sample)
        >>> print(schema)
        id: string
        value: float
        count: int32
    """
    fields = []
    for field_name, value in sample.items():
        field_type = determine_pa_field_type(value, default_float_type)
        fields.append(pa.field(field_name, field_type))
    return pa.schema(fields)


def create_iterative_writer(
    path: Union[str, AnyPath],
    sample: dict,
    format: str = default_shard_creator_cfg.pyarrow_shard_type,
    default_float_type: str = default_shard_creator_cfg.pyarrow_default_float_type,
) -> Union[pq.ParquetWriter, pa.ipc.RecordBatchFileWriter]:
    """Create an iterative writer for Parquet or Arrow formats based on a sample.

    Arguments
    ---------
    path: Union[str, AnyPath]
        Path to write the file to
    sample: dict
        A sample dictionary to infer the schema from
    format: str
        Output format ("parquet" or "arrow")
    default_float_type:
        Default float type to use

    Returns
    -------
        Writer object for the specified format

    Example
    -------
        >>> sample = [{"id": "1", "value": 3.14}, {"id": "2", "value": 2.71}]
        >>> writer = create_iterative_writer("/tmp/test.parquet", sample[0])
        >>> writer.write_table(pa.Table.from_pylist(sample, schema=writer.schema))
        >>> writer.close()
    """
    # Infer schema from the sample
    schema = infer_schema_from_sample(sample, default_float_type)

    file_obj = AnyPath(path).open("wb")

    # Create appropriate writer
    if format == "parquet":
        return pq.ParquetWriter(file_obj, schema)

    writer = pa.ipc.new_file(file_obj, schema)
    writer.schema = schema

    return writer


def write_batch_to_writer(
    batch_data: List[dict], writer: Union[pq.ParquetWriter, pa.ipc.RecordBatchFileWriter]
) -> None:
    """Write a batch of dictionaries to an open writer.

    Arguments
    ---------
    batch_data: List[dict]
        List of dictionaries to write
    writer: Union[pq.ParquetWriter, pa.ipc.RecordBatchFileWriter]
        An open writer object

    Example
    -------
        >>> batch = [{"id": "1", "value": 3.14}, {"id": "2", "value": 2.71}]
        >>> schema = pa.schema([pa.field("id", pa.string()), pa.field("value", pa.float64())])
        >>> with pq.ParquetWriter("/tmp/test.parquet", schema) as writer:
        ...     write_batch_to_writer(batch, writer)
        >>> table = pq.read_table("/tmp/test.parquet")
        >>> print(table)
        pyarrow.Table
        id: string
        value: double
        ----
        id: [["1","2"]]
        value: [[3.14,2.71]]
    """
    if not batch_data:
        return
    table = pa.Table.from_pylist(batch_data, schema=writer.schema)
    writer.write_table(table)


def write_arrow_shard(
    batch: Union[Iterable[dict], pd.DataFrame, pd.Series],
    shard_id: int,
    output_path: Union[str, AnyPath],
    sample_prep_function: Optional[Callable] = None,
    format: str = default_shard_creator_cfg.pyarrow_shard_type,
    log_every: int = default_shard_creator_cfg.log_every,
    writer_batch_size: int = default_shard_creator_cfg.pyarrow_writer_batch_size,
    error_handling: str = default_shard_creator_cfg.error_handling,
    shard_name: str = default_shard_creator_cfg.shard_name,
) -> dict:
    """Write a batch of samples to an Arrow/Parquet shard iteratively, processing samples as they come in.

    Arguments
    ---------
    batch: Union[Iterable[dict], pd.DataFrame, pd.Series]
        Iterable of dictionaries or dataframe or series containing sample data
    shard_id: int
        ID for this shard
    output_path: Union[str, AnyPath]
        Path to save the shard
    sample_prep_function: Optional[Callable]
        Function to prepare a sample for the Arrow format
    format: str
        Output format for the shard ("parquet" or "arrow")
    log_every: int
        Log progress every N samples
    writer_batch_size: int
        Number of samples to process before writing to the shard
    error_handling: str
        How to handle errors ("warn", "raise", "ignore")

    Returns
    -------
        dict: Dictionary with processing results. Contains "processed_ids" and "failed_ids".
            "processed_ids" is a list of dicts, each containing the sample ID, shard ID, and shard path.
            "failed_ids" is a list of sample IDs that failed processing.

    Example
    -------
        >>> data = [
        ...     {"id": 1, "name": "Alice", "age": 25},
        ...     {"id": 2, "name": "Bob", "age": 30},
        ...     {"id": 3, "name": "Charlie", "age": 35},
        ... ]
        >>> result = write_arrow_shard(data, 0, "/tmp/data.parquet", format="parquet")
        >>> print(len(result["processed_ids"]))
        3
    """
    results = {"shard_id": shard_id, "processed_ids": [], "failed_ids": []}

    # Create shard path
    output_path = AnyPath(output_path)
    shard_path = output_path / (f"{shard_name}_{shard_id:06d}.{format}")

    # Process batch data
    iterator = batch.iterrows() if isinstance(batch, pd.DataFrame) else enumerate(batch)
    total_samples = len(batch)

    # Initialize with first valid sample to get schema
    writer = None
    current_batch = []

    j = 0
    for _, item in iterator:
        # Get item and sample_id
        if isinstance(item, pd.Series):
            item = item.to_dict()

        sample_id = str(item.get("id", make_id()))
        item["id"] = sample_id

        if (j + 1) % log_every == 0:
            logger.info(f"Shard {shard_id:06d} - Processing sample {j}/{total_samples} (id: {sample_id})")

        try:
            # Prepare the sample
            shard_data = sample_prep_function(item) if sample_prep_function else item

            # Initialize writer with first sample if not already done
            if writer is None:
                writer = create_iterative_writer(shard_path, shard_data, format=format)

            current_batch.append(shard_data)
            results["processed_ids"].append({"id": sample_id, "shard_id": shard_id, "shard_path": str(shard_path)})

            # Write batch if we've reached batch size
            if len(current_batch) == writer_batch_size:
                write_batch_to_writer(current_batch, writer)
                current_batch = []

        except Exception as e:
            results["failed_ids"].append(sample_id)
            _error_handler(e, sample_id, error_handling)

        j += 1

    # Write any remaining samples
    if current_batch and writer is not None:
        write_batch_to_writer(current_batch, writer)

    # Close the writer
    if writer is not None:
        writer.close()

    logger.info(
        f"Finished shard {shard_id:06d} - Processed: {len(results['processed_ids'])}, Failed: {len(results['failed_ids'])}"
    )

    return results


def write_huggingface_shard(
    batch: Union[Iterable[dict], pd.DataFrame, pd.Series],
    shard_id: int,
    output_path: Union[str, AnyPath],
    sample_prep_function: Optional[Callable] = None,
    storage_options: Optional[dict] = None,
    log_every: int = default_shard_creator_cfg.log_every,
    error_handling: str = default_shard_creator_cfg.error_handling,
    shard_name: str = default_shard_creator_cfg.shard_name,
) -> dict:
    """
    Write a batch of samples to an Arrow shard in the Hugging Face format using a generator.

    Arguments
    ---------
    batch: Union[Iterable[dict], pd.DataFrame, pd.Series]
        Iterable of dictionaries containing sample data
    shard_id: int
        ID for this shard
    output_path: Union[str, AnyPath]
        Path to save the shard
    sample_prep_function: Optional[Callable]
        Function to prepare a sample for dataset Arrow format
    storage_options: Optional[dict]
        Optional storage options for saving the dataset
    log_every: int
        Log progress every N samples
    error_handling: str
        How to handle errors ("warn", "raise", "ignore")

    Returns
    -------
        dict: Dictionary with processing results. Contains "processed_ids" and "failed_ids".
            "processed_ids" is a list of dicts, each containing the sample ID, shard ID, and shard path.
            "failed_ids" is a list of sample IDs that failed processing.

    Example
    -------
        >>> data = [
        ...     {"id": "1", "text": "Hello world", "label": 1},
        ...     {"id": "2", "text": "Goodbye world", "label": 0},
        ...     {"id": "3", "text": "Hello again", "label": 1},
        ... ]
        >>> result = write_huggingface_shard(data, 0, "/tmp/hf_data")
        >>> print(len(result["processed_ids"]))
        3
        >>> print(result["failed_ids"])
        []
        >>> from pathlib import Path
        >>> Path("/tmp/hf_data/shard_000000.arrow").exists()
        True
    """
    results = {"shard_id": shard_id, "processed_ids": [], "failed_ids": []}

    # Create shard path
    output_path = AnyPath(output_path)
    shard_path = output_path / (f"{shard_name}_{shard_id:06d}")
    storage_options = output_path.storage_options if storage_options is None else storage_options

    # Process the data using generator for memory efficiency
    total_samples = len(batch)
    logger.info(f"Creating HuggingFace dataset for shard {shard_id:06d} with {total_samples} samples")

    iterator = batch.iterrows() if isinstance(batch, pd.DataFrame) else enumerate(batch)
    total_samples = len(batch)

    prepared_samples = []
    j = 0
    for _, item in iterator:
        # Get item
        if isinstance(item, pd.Series):
            item = item.to_dict()

        sample_id = str(item.get("id", make_id()))
        item["id"] = sample_id

        if (j + 1) % log_every == 0:
            logger.info(f"Shard {shard_id:06d} - Processing sample {j}/{total_samples} (id: {sample_id})")

        try:
            # Prepare the sample
            prepared_sample = sample_prep_function(item) if sample_prep_function else item
            prepared_samples.append(prepared_sample)
            results["processed_ids"].append({"id": sample_id, "shard_id": shard_id, "shard_path": str(shard_path)})

        except Exception as e:
            results["failed_ids"].append(sample_id)
            _error_handler(e, sample_id, error_handling)

        j += 1

    ds = Dataset.from_list(prepared_samples)

    # Save dataset
    ds.save_to_disk(shard_path, storage_options=storage_options, num_proc=1, num_shards=1)

    # Move the arrow file to the final location
    arrow_file = next(F.yield_files(shard_path, pattern="*.arrow"))
    F.cp_to_cloud(arrow_file, output_path / (f"{shard_name}_{shard_id:06d}.arrow"))
    F.delete_dir(shard_path)

    logger.info(
        f"Finished shard {shard_id:05d} - Processed: {len(results['processed_ids'])}, Failed: {len(results['failed_ids'])}"
    )

    return results


def write_shard(
    batch: Union[List[dict], pd.DataFrame, pd.Series],
    shard_id: int,
    output_path: Union[str, AnyPath],
    output_format: str,
    sample_prep_function: Optional[Callable] = None,
    log_every: int = default_shard_creator_cfg.log_every,
    error_handling: str = default_shard_creator_cfg.error_handling,
    **kwargs,
) -> dict:
    """
    Write a batch of samples to a shard in the specified format.

    Arguments
    ---------
    batch: Union[List[dict], pd.DataFrame, pd.Series]
        List of dictionaries or dataframe or series containing sample data
    shard_id: int
        ID for this shard
    output_path: Union[str, AnyPath]
        Path to save the shard
    sample_prep_function: Optional[Callable]
        Function to prepare a sample for the shard format
    output_format: str
        Output format for the shard ("webdataset", "arrow", "parquet", "hf")
    log_every: int
        Log progress every N samples
    error_handling: str
        How to handle errors ("warn", "raise", "ignore")
    kwargs:
        Additional keyword arguments for the specific shard writer

    Returns
    -------
        dict: Results of the sharding process

    Example
    -------
        >>> data = [{"id": 1, "text": "example"}]
        >>> result = write_shard(data, 0, "/tmp/output", output_format="parquet")
        >>> print(result["shard_id"])
        0
    """
    if output_format == "webdataset":
        return write_webdataset_shard(
            batch,
            shard_id,
            output_path,
            sample_prep_function,
            log_every=log_every,
            error_handling=error_handling,
            **kwargs,
        )
    elif output_format in ["arrow", "parquet"]:
        return write_arrow_shard(
            batch,
            shard_id,
            output_path,
            sample_prep_function,
            format=output_format,
            log_every=log_every,
            error_handling=error_handling,
            **kwargs,
        )
    elif output_format == "hf":
        return write_huggingface_shard(
            batch,
            shard_id,
            output_path,
            sample_prep_function,
            log_every=log_every,
            error_handling=error_handling,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported output format: {output_format}, must be {', '.join(SHARD_TYPES)}.")


def compute_metadata_hash(metadata: Union[pd.DataFrame, List[dict], dict, Any]) -> int:
    """
    Compute a hash for metadata of various types.

    Arguments
    ---------
    metadata: Union[pd.DataFrame, List[dict], dict, Any]
        Can be a pandas DataFrame, list of dictionaries, dictionary, or other serializable data

    Returns
    -------
        int: A hash value representing the metadata

    Example
    -------
        >>> value = compute_metadata_hash({"id": 1, "value": 3.14})
        >>> assert isinstance(value, int)
    """
    if isinstance(metadata, pd.DataFrame):
        metadata = metadata.to_dict()

    try:
        metadata_str = json.dumps(metadata, sort_keys=True)
        hash_object = hashlib.md5(metadata_str.encode())
        return int.from_bytes(hash_object.digest(), byteorder="big")

    except (TypeError, ValueError):
        metadata_str = str(metadata)
        hash_object = hashlib.md5(metadata_str.encode())

        return int.from_bytes(hash_object.digest(), byteorder="big")


def save_checkpoint(
    output_path: Union[str, AnyPath], completed_chunks: dict, metadata: Any, checkpoint_name: str = "checkpoint.json"
) -> None:
    """
    Save checkpoint information for any type of metadata.

    Arguments
    ---------
    output_path: Union[str, AnyPath]
        Path to save the checkpoint
    completed_chunks: dict
        Dictionary of completed chunks
    metadata: Any
        Metadata of any serializable type
    checkpoint_name: str
        Name of the checkpoint file

    Example
    -------
        >>> completed_chunks = {"0": {"processed_ids": ["1", "2"], "failed_ids": []}}
        >>> save_checkpoint("/tmp", completed_chunks, {"some": "metadata"})
        >>> from pathlib import Path
        >>> Path("/tmp/checkpoint.json").exists()
        True
        >>> checkpoint = load_checkpoint("/tmp", {"some": "metadata"})
        >>> "completed_chunks" in checkpoint
        True
    """
    output_path = AnyPath(output_path)
    checkpoint_data = {
        "completed_chunks": completed_chunks,
        "metadata_hash": compute_metadata_hash(metadata),
    }

    with (output_path / checkpoint_name).open("w") as f:
        json.dump(checkpoint_data, f)


def save_metadata(
    metadata_df: pd.DataFrame, output_path: Union[str, AnyPath], storage_options: Optional[dict] = None
) -> None:
    """Save metadata DataFrame to a file in the appropriate format based on the file extension.

    Arguments
    ---------
     metadata_df: pd.DataFrame
        DataFrame containing metadata
    output_path: Union[str, AnyPath]
        Path to save the metadata
    storage_options: Optional[dict]
        Storage options for cloud storage

    Example
    -------
        >>> df1 = pd.DataFrame([{"id": 1, "name": "test"}, {"id": 2, "name": "example"}])
        >>> save_metadata(df1, "/tmp/metadata.parquet")
        >>> df2 = pd.read_parquet("/tmp/metadata.parquet")
        >>> df1.equals(df2)
        True
    """
    output_path = AnyPath(output_path)
    ext = output_path.suffix.lstrip(".")

    try:
        if ext == "parquet":
            metadata_df.to_parquet(str(output_path), index=False, storage_options=storage_options)
        elif ext == "csv":
            metadata_df.to_csv(str(output_path), index=False, storage_options=storage_options)
        elif ext == "json" or ext == "jsonl":
            metadata_df.to_json(str(output_path), orient="records", lines=True, storage_options=storage_options)
        elif ext == "tsv":
            metadata_df.to_csv(str(output_path), sep="\t", index=False, storage_options=storage_options)
        else:
            logger.warning(f"Unsupported metadata format: {ext}. Saving as JSON.")
            metadata_df.to_json(str(output_path), orient="records", lines=True, storage_options=storage_options)

        logger.info(f"Saved metadata to {output_path} ({len(metadata_df)} records)")
    except Exception as e:
        logger.error(f"Error saving metadata to {output_path}: {e}")


def load_checkpoint(
    output_path: Union[str, AnyPath], metadata: Any, checkpoint_name: str = "checkpoint.json"
) -> Optional[dict]:
    """
    Load checkpoint if it exists and is valid for any type of metadata.

    Arguments
    ---------
    output_path: Union[str, AnyPath]
        Path to the checkpoint
    metadata: Any
        Current metadata to compare against saved checkpoint
    checkpoint_name: str
        Name of the checkpoint file

    Returns
    -------
        Optional[dict]: Checkpoint data if valid, None otherwise

    Example:
        >>> checkpoint = load_checkpoint("/tmp/random/random/random", {"some": "metadata"})
        >>> checkpoint is None
        True
    """
    output_path = AnyPath(output_path)
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
    data: Union[pd.DataFrame, Iterable[dict]],
    output_path: Union[str, AnyPath],
    sample_prep_function: Optional[Callable] = None,
    dataset_config: DatasetConfig = None,
    save_metadata_as: Optional[str] = None,
    merge_data_and_metadata: bool = False,
    num_samples_per_shard: int = default_shard_creator_cfg.num_samples_per_shard,
    num_workers: int = default_shard_creator_cfg.num_workers,
    shard_type: str = default_shard_creator_cfg.shard_type,
    log_every: int = default_shard_creator_cfg.log_every,
    error_handling: str = default_shard_creator_cfg.error_handling,
    **sharding_kwargs,
) -> pd.DataFrame:
    """
    Create sharded dataset from information in a dataframe (or a Iterable[dict]) and a sample prep function,
    in parallel, with checkpointing.

    Arguments
    ---------
    data: Union[pd.DataFrame, Iterable[dict]]
        DataFrame or list of dictionaries containing sample data
    output_path: Union[str, AnyPath]
        Path to save the shards
    sample_prep_function: Callable
        Function to prepare a sample for the shard format. Must return a dictionary.
        For e.g., this function could be tokenizer, a feature preprocessor, a pydantic validator etc.
    num_samples_per_shard: int
        Number of samples to include in each shard
    num_workers: int
        Number of workers to use for parallel processing
    shard_type: str
        Output format for the shards ("webdataset", "arrow", "parquet", "hf")
    dataset_config: DatasetConfig
        Dataset configuration object, will create skeleton config if not provided.
    save_metadata_as: Optional[str]
        File format to save metadata as (parquet, csv, json, jsonl, tsv). This acts as an index for the shards.
    merge_data_and_metadata: bool
        Merge the shard information with the original data. If your data is a small DataFrame, you can combine the
        shard information with the original data to create a single DataFrame with shard information.
    log_every: int
        Log progress every N samples
    error_handling: str
        How to handle errors ("warn", "raise", "ignore")
    sharding_kwargs:
        Additional keyword arguments for the specific shard writer

    Returns
    -------
        pd.DataFrame: DataFrame containing metadata about the created shards

    Example
    -------
        >>> data = pd.DataFrame([
        ...     {"id": 1, "text": "Example 1"},
        ...     {"id": 2, "text": "Example 2"},
        ... ])
        >>> def prep_sample(item):
        ...     return {"length": len(item["text"]), **item}
        >>> metadata_df = create_sharded_dataset(
        ...     data,
        ...     "/tmp/output",
        ...     prep_sample,
        ...     num_samples_per_shard=2
        ... )
    """
    t0 = time.time()
    output_path = AnyPath(output_path)

    if len(data) == 0:
        logger.error("Metadata is empty. Nothing to process.")
        return

    if shard_type not in ["webdataset", "arrow", "parquet", "hf"]:
        raise ValueError(f"Unsupported shard type: {shard_type}. Must be one of {', '.join(SHARD_TYPES)}.")

    # Load checkpoint if exists
    checkpoint_data = load_checkpoint(output_path, data)
    completed_chunks = checkpoint_data["completed_chunks"] if checkpoint_data else {}
    logger.info(f"Loaded checkpoint with {len(completed_chunks)} completed chunks.")

    # Calculate number of shards needed
    num_samples = len(data)
    num_shards = int(np.ceil(num_samples / num_samples_per_shard))

    # Split metadata into chunks for parallel processing
    chunks = [data[i * num_samples_per_shard : (i + 1) * num_samples_per_shard] for i in range(num_shards)]

    logger.info(f"\n{BATCH_COLOR}=== Dataset Sharding Process ==={RESET_COLOR}")
    logger.info(
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
        log_every=log_every,
        error_handling=error_handling,
        **sharding_kwargs,
    )

    # Process chunks in parallel or sequentially based on num_workers
    processed_samples = []

    if num_workers > 1:
        # Parallel processing using ProcessPoolExecutor
        logger.info(f"Processing {len(chunks_to_process)} shards using {num_workers} workers")

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
                    processed_samples.extend(result["processed_ids"])

                    # Update checkpoint after each chunk is completed
                    completed_chunks[str(chunk_id)] = result
                    save_checkpoint(output_path, completed_chunks, data)

                    # Update progress bar with shard info
                    pbar.set_postfix(
                        shard=f"{chunk_id:06d}",
                        success=len(result["processed_ids"]),
                        failed=len(result["failed_ids"]),
                    )
                    pbar.update(1)
    else:
        # Sequential processing
        logger.info(f"Processing {len(chunks_to_process)} shards sequentially")

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
                processed_samples.extend(result["processed_ids"])

                # Update checkpoint after each chunk is completed
                completed_chunks[str(chunk_id)] = result
                save_checkpoint(output_path, completed_chunks, data)

                # Update progress bar with shard info
                pbar_shards.set_postfix(
                    shard=f"{chunk_id:05d}", success=len(result["processed_ids"]), failed=len(result["failed_ids"])
                )
                pbar_shards.update(1)

    # Save dataset configuration
    if not dataset_config:
        logger.warning("No dataset config provided. Creating skeleton config..")
        dataset_config = DatasetConfig.from_skeleton()

    dataset_config.write_json(output_path / "dataset_config.json")
    dataset_config.generate_readme(output_path / "README.md")

    # Create DataFrame with shard information
    shard_info_df = pd.DataFrame(processed_samples, columns=["id", "shard_path", "shard_id"])

    if merge_data_and_metadata and len(shard_info_df) > 0:
        # Merge shard information with original metadata
        metadata_df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        metadata_df = metadata_df.merge(shard_info_df, on="id", how="left")
    else:
        metadata_df = shard_info_df

    # Save updated metadata if requested
    if save_metadata_as and len(metadata_df) > 0:
        storage_options = AnyPath(output_path).storage_options
        save_metadata(metadata_df, output_path / save_metadata_as, storage_options)

    # Report final statistics
    all_successful = sum(len(chunk.get("processed_ids", [])) for chunk in completed_chunks.values())
    all_failed = sum(len(chunk.get("failed_ids", [])) for chunk in completed_chunks.values())
    success_rate = (all_successful / (all_successful + all_failed)) * 100 if (all_successful + all_failed) > 0 else 0

    tend = time.time()
    logger.info(f"\n{BATCH_COLOR}=== Processing Summary ==={RESET_COLOR}")
    logger.info(f"{SUCCESS_COLOR}- Total files processed successfully: {all_successful}{RESET_COLOR}")
    logger.info(f"{ERROR_COLOR}- Total files failed: {all_failed}{RESET_COLOR}")
    logger.info(f"{BATCH_COLOR}- Success rate: {success_rate:.2f}%{RESET_COLOR}")
    logger.info(f"{BATCH_COLOR}- Total time taken: {tend - t0:.2f} seconds{RESET_COLOR}")

    return metadata_df


if __name__ == "__main__":
    import doctest

    doctest.testmod()
