"""
Modular functions to create sharded datasets in both WebDataset (tar) and Arrow formats
"""

import json
import time
from functools import partial
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import webdataset as wds
from pydantic import BaseModel
from tqdm import tqdm

import esp_data.file_io.functional as F
from esp_data.paths import AnyPath, is_cloud_path, is_local_path


def _make_file_opener(file_path: str | AnyPath, mode: str = "wb") -> callable:
    """Make a file opener function for WebDataset or Arrow"""
    file_path = AnyPath(file_path)

    if is_local_path(file_path):
        # Create parent directories if they don't exist
        parent_dir = file_path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        # Return a callable function that opens the file
        return partial(open, mode=mode)

    if is_cloud_path(file_path):
        return partial(F.open_file, mode=mode, use_fs=True)


def validate_batch(batch: List[Dict], validation_model: Optional[type[BaseModel]] = None) -> List[Dict]:
    """
    Validate a batch of data using a Pydantic model.

    Args:
        batch: List of dictionaries containing sample data
        validation_model: Optional pydantic model for validation

    Returns:
        List of validated dictionaries (filters out invalid items)
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
            print(f"Validation failed for item: {e}")
            continue

    return valid_items


def write_webdataset_shard(
    batch: Iterable[dict] | pd.DataFrame | pd.Series,
    shard_id: int,
    output_path: str | AnyPath,
    sample_prep_function: Callable,
) -> Dict:
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
            print(f"Error processing sample {sample_id}: {str(e)}")
            results["failed_ids"].append(sample_id)

    # Close the shard
    sink.close()

    return results


def write_arrow_shard(
    batch: Iterable[dict] | pd.DataFrame | pd.Series,
    shard_id: int,
    output_path: str | AnyPath,
    arrow_prep_function: Callable,
    format: str = "parquet",
) -> Dict:
    """
    Write a batch of samples to an Arrow / Parquet shard.

    Args:
        batch: list of dictionaries or dataframe or series containing sample data
        shard_id: ID for this shard
        output_path: Path to save the shard
        arrow_prep_function: Function to prepare a sample for Arrow format
        format: Output format for the Arrow shard (parquet or arrow)

    Returns:
        Dictionary with processing results
    """
    results = {"shard_id": shard_id, "processed_samples": [], "failed_ids": []}

    # Create shard path
    output_path = AnyPath(output_path)
    shard_path = output_path / f"shard_{shard_id:06d}." + format

    # Process batch data
    prepared_data = []
    iterator = batch.iterrows() if isinstance(batch, pd.DataFrame) else enumerate(batch)

    for i, item in iterator:
        # get item and sample_id
        if isinstance(item, pd.Series):
            item = item.to_dict()

        sample_id = str(item["id"] if "id" in item else i)

        try:
            prepared_sample = arrow_prep_function(item)
            prepared_data.append(prepared_sample)

            # Track successful sample
            results["processed_samples"].append(
                {"id": sample_id, "shard_path": f"shard_{shard_id:06d}.arrow", "shard_id": shard_id}
            )

        except Exception as e:
            print(f"Error processing sample {sample_id} for Arrow: {e}")
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


def write_shard(
    batch: List[Dict],
    shard_id: int,
    output_path: str | AnyPath,
    sample_prep_function: Callable,
    output_format: str = "webdataset",
) -> Dict:
    """
    Write a batch of samples to a shard in the specified format.

    Args:
        batch: List of dictionaries containing sample data
        shard_id: ID for this shard
        output_path: Path to save the shard
        sample_prep_function: Function to prepare a sample for the specified format
        output_format: Output format for the shard (webdataset or arrow)
    """
    if output_format == "webdataset":
        return write_webdataset_shard(batch, shard_id, output_path, sample_prep_function)
    elif output_format == "arrow":
        return write_arrow_shard(batch, shard_id, output_path, sample_prep_function)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def create_dual_sharded_dataset(
    data_generator: Iterable[List[Dict]],
    webdataset_output_path: str | AnyPath,
    arrow_output_path: str | AnyPath,
    webdataset_sample_prep_function: Callable,
    arrow_sample_prep_function: Callable,
    validation_model: Optional[type[BaseModel]] = None,
    num_workers: int = 1,
) -> Dict:
    """
    Create sharded datasets in both WebDataset and Arrow formats.

    Args:
        data_generator: Generator yielding lists of dictionaries (each dictionary is a sample)
        webdataset_output_path: Path to save WebDataset shards
        arrow_output_path: Path to save Arrow shards
        webdataset_sample_prep_function: Function to prepare samples for WebDataset
        arrow_sample_prep_function: Function to prepare samples for Arrow
        validation_model: Optional pydantic model for validation
        num_workers: Number of workers for parallel processing

    Returns:
        Dictionary with processing results
    """
    webdataset_output_path = AnyPath(webdataset_output_path)
    arrow_output_path = AnyPath(arrow_output_path)

    # Create output directories
    for path in [webdataset_output_path, arrow_output_path]:
        if is_local_path(path):
            path.mkdir(parents=True, exist_ok=True)

    # Initialize counters and results
    shard_id = 0
    total_processed = 0
    total_failed = 0
    all_results = {"webdataset": [], "arrow": [], "metadata": []}
    start_time = time.time()

    # Process batches
    with tqdm(data_generator, desc="Processing batches") as pbar:
        for batch in pbar:
            # Validate batch
            valid_batch = validate_batch(batch, validation_model)

            if not valid_batch:
                print("Skipping empty batch after validation")
                continue

            # Create WebDataset shard
            webdataset_result = write_webdataset_shard(
                valid_batch, shard_id, webdataset_output_path, webdataset_sample_prep_function
            )

            # Create Arrow shard
            arrow_result = write_arrow_shard(valid_batch, shard_id, arrow_output_path, arrow_sample_prep_function)

            # Update results
            all_results["webdataset"].append(webdataset_result)
            all_results["arrow"].append(arrow_result)

            # Merge sample info for metadata
            for sample in webdataset_result["processed_samples"]:
                # Find corresponding arrow info
                arrow_info = next((a for a in arrow_result["processed_samples"] if a["id"] == sample["id"]), None)

                if arrow_info:
                    all_results["metadata"].append(
                        {
                            "id": sample["id"],
                            "webdataset_shard": sample["shard_path"],
                            "arrow_shard": arrow_info["arrow_path"],
                            "shard_id": shard_id,
                        }
                    )

            # Update counters
            total_processed += len(webdataset_result["processed_samples"])
            total_failed += len(webdataset_result["failed_ids"])

            # Update progress bar
            pbar.set_postfix({"shard": shard_id, "processed": total_processed, "failed": total_failed})

            # Increment shard ID for next batch
            shard_id += 1

    # Write metadata
    metadata_dict = {"samples": all_results["metadata"]}
    metadata_json_path = webdataset_output_path / "metadata.json"
    with _make_file_opener(metadata_json_path, mode="w")(str(metadata_json_path)) as f:
        json.dump(metadata_dict, f)

    arrow_metadata_json_path = arrow_output_path / "metadata.json"
    with _make_file_opener(arrow_metadata_json_path, mode="w")(str(arrow_metadata_json_path)) as f:
        json.dump(metadata_dict, f)

    # Report final statistics
    processing_time = time.time() - start_time
    summary = {
        "total_processed": total_processed,
        "total_failed": total_failed,
        "success_rate": (total_processed / (total_processed + total_failed) * 100)
        if (total_processed + total_failed) > 0
        else 0,
        "total_shards": shard_id,
        "processing_time_seconds": processing_time,
    }

    print(f"""
    Processing completed:
    - Total files processed successfully: {summary["total_processed"]}
    - Total files failed: {summary["total_failed"]}
    - Success rate: {summary["success_rate"]:.2f}%
    - Total shards created: {summary["total_shards"]}
    - Total time taken: {summary["processing_time_seconds"]:.2f} seconds
    """)

    return {"results": all_results, "summary": summary}


# Example usage as CLI
def main():
    import argparse

    from pydantic import BaseModel, Field

    # Example Pydantic validation model
    class AudioSample(BaseModel):
        id: str
        file_path: str
        metadata: dict = Field(default_factory=dict)

    parser = argparse.ArgumentParser(description="Create dual-format sharded datasets")
    parser.add_argument("--webdataset_output", required=True, help="Path for WebDataset output")
    parser.add_argument("--arrow_output", required=True, help="Path for Arrow output")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")

    args = parser.parse_args()

    # This is where you would define your custom sample prep functions
    # These are just placeholders - replace with your actual implementations
    def example_webdataset_prep(item):
        # Example function - replace with your actual implementation
        return {"metadata.json": json.dumps(item)}

    def example_arrow_prep(item):
        # Example function - replace with your actual implementation
        return item

    # Example generator function that yields batches of dictionaries
    def example_data_generator():
        # In a real implementation, this would read from a source
        for i in range(5):  # 5 batches
            batch = [
                {"id": f"{i}_{j}", "file_path": f"/path/to/file_{i}_{j}", "metadata": {"tag": f"batch_{i}"}}
                for j in range(10)  # 10 items per batch
            ]
            yield batch

    # Process the dataset
    create_dual_sharded_dataset(
        data_generator=example_data_generator(),
        webdataset_output_path=args.webdataset_output,
        arrow_output_path=args.arrow_output,
        webdataset_sample_prep_function=example_webdataset_prep,
        arrow_sample_prep_function=example_arrow_prep,
        validation_model=AudioSample,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
