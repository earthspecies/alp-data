"""Module for exporting datasets multiple formats"""

import concurrent.futures
import gc
import logging
import multiprocessing as mp
import traceback
import uuid
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import pandas as pd
import webdataset as wds
from tqdm import tqdm

from esp_data import Dataset
from esp_data.io import AnyPathT, anypath, filesystem_from_path

from .webdataset_utils import audio_encoder

logger = logging.getLogger("esp_data")


def _make_id() -> str:
    """Generate a unique identifier for a sample.
    This helps to find unique samples in a tar ball.

    Returns
    -------
    str: A unique identifier string.
    """
    return str(uuid.uuid4())


def _error_handler(
    e: Exception,
    sample_id: str,
    error_handling: str = "warn",
) -> None:
    """Handle errors during sample processing.

    Parameters
    ----------
    e: Exception
        The exception that was raised
    sample_id: str
        The ID of the sample that caused the error
    error_handling: str
        How to handle errors ("warn", "raise", "ignore")
    """
    if error_handling == "warn":
        logger.error(f"Error processing sample {sample_id}: {e}")
        logger.error(f"Exception traceback:\n{traceback.format_exc()}")
    elif error_handling == "raise":
        raise e
    elif error_handling == "ignore":
        pass
    if isinstance(e, KeyboardInterrupt):
        raise e


def _make_file_opener_for_wds(
    file_path: str | AnyPathT,
    mode: str = "wb",
    block_size: int = 1024 * 1024 * 100,
) -> Callable:
    """Make a file opener function for WebDataset.
    If local path, create parent dirs if needed.

    Arguments
    ---------
    file_path: str | AnyPathT
        The file path to open
    mode: str
        The mode in which to open the file (default: "wb")
    block_size: int
        Block size for WebDataset (default: 100 MB)

    Returns
    -------
        Callable: A function that opens the file in the specified mode
        or a file object if the path is local.
    """
    file_path = anypath(file_path)

    if isinstance(file_path, Path):
        parent_dir = file_path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        return open(file_path, mode=mode)

    fs = filesystem_from_path(str(file_path))
    return fs.open(str(file_path), mode=mode, block_size=block_size)


def _write_webdataset_shard(
    batch: Iterable[dict],
    output_path: str | AnyPathT,
    shard_id: int,
    sample_prep_function: Callable | None = None,
    log_every: int = 100,
    error_handling: str = "warn",
    shard_name: str = "shard",
    compression: Literal["gz", "bz2", "xz"] | None = None,
) -> dict:
    """
    Write a batch of samples to a WebDataset shard.

    Arguments
    ---------
    batch: Union[Iterable[dict], pd.DataFrame, pd.Series]
        Iterable of dictionaries or dataframe or series containing sample data
    output_path: Union[str, AnyPathT],
        Path to the directory or bucket 'folder' to save the shard
    shard_id: int
        ID for this shard
    sample_prep_function: Optional[Callable]
        Function to prepare a sample for the WebDataset format
    log_every: int
        Log progress every N samples
    error_handling: str
        How to handle errors ("warn", "raise", "ignore")
    shard_name: str
        Name (prefix) of the shard files (default: "shard")
    compression: Literal["gz", "bz2", "xz"] | None

    Returns
    -------
    dict: Dictionary with processing results. Contains "processed_ids"
        and "failed_ids". "processed_ids" is a list of dicts,
        each containing the sample ID, shard ID,
        and shard path. "failed_ids" is a list of sample IDs that failed processing.

    Raises
    ------
    ValueError: If compression format is not supported.

    Example
    -------
        >>> items = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]
        >>> results = _write_webdataset_shard(items, "/tmp/webds_output", 0)
        # writing /tmp/webds_output/shard_000000.tar 0 0.0 GB 0
        >>> print(len(results["processed_ids"]))
        2
        >>> from pathlib import Path; Path("/tmp/webds_output/shard_000000.tar").exists()
        True
    """
    results = {"shard_id": shard_id, "processed_ids": [], "failed_ids": []}

    # Create shard path
    output_path = anypath(output_path)
    shard_path = str(output_path / f"{shard_name}_%s{shard_id:05d}.tar")

    if compression:
        if compression not in ["gz", "bz2", "xz"]:
            raise ValueError(
                f"Unsupported compression format: {compression}. "
                "Supported formats are 'gz', 'bz2', 'xz'."
            )
        # append compression extension if specified
        shard_path = f"{str(shard_path)}.{compression}"

    # Initialize shard writer
    sink = wds.ShardWriter(
        shard_path,
        maxcount=100_000_000_000,  # Set very high to ensure all samples go in one shard
        maxsize=100_000_000_000,
        opener=partial(_make_file_opener_for_wds),
        compress=compression,
    )

    # Process each sample
    total_samples = len(batch)
    for j, item in enumerate(batch):
        sample_id = str(item.get("id", _make_id()))
        item["id"] = sample_id

        if (j + 1) % log_every == 0:
            logger.info(
                f"Shard {shard_id:05d} - Processing sample {j}/{total_samples} (id: {sample_id})"
            )

        try:
            shard_data = sample_prep_function(item) if sample_prep_function else item

            # Write to shard
            # __key__ was just taken from the examples in the webdataset docs
            sample = {
                "__key__": sample_id,
                **shard_data,
            }
            sink.write(sample)

            # Track successful sample
            results["processed_ids"].append(
                {
                    "id": sample_id,
                    "shard_id": shard_id,
                    "shard_path": str(output_path / f"{shard_name}_{shard_id:06d}.tar"),
                }
            )

        except Exception as e:
            results["failed_ids"].append(sample_id)
            _error_handler(e, sample_id, error_handling)
        finally:
            gc.collect()

    sink.close()
    logger.info(
        f"Finished shard {shard_id:06d} - Processed: {len(results['processed_ids'])}, "
        f"Failed: {len(results['failed_ids'])}"
    )

    return results


def _chunk_dataset_indices(
    ds: Dataset, chunk_size: int, shuffle: bool = False, seed: int = 42
) -> list[tuple[int, int]]:
    """Split dataset into index chunks for parallel processing.
    Shuffles indices if specified.

    Parameters
    ----------
    ds: Dataset
        The dataset to chunk (only uses len())
    chunk_size: int
        Size of each chunk
    shuffle: bool
        Whether to shuffle dataset indices before chunking
    seed: int
        Random seed for shuffling (default: 42)

    Returns
    -------
    list[tuple[int, int]]: List of (start_idx, end_idx) tuples for each chunk

    Raises
    ------
    ValueError: If chunk_size is not a positive integer or dataset is empty
    """
    dataset_len = len(ds)
    if dataset_len == 0:
        raise ValueError("Dataset is empty, cannot chunk indices")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    if shuffle:
        # Generate shuffled indices
        indices = list(range(dataset_len))
        rng = mp.Random(seed)
        rng.shuffle(indices)
    else:
        # Use sequential indices
        indices = list(range(dataset_len))

    chunks = []
    for start in range(0, dataset_len, chunk_size):
        end = min(start + chunk_size, dataset_len)
        chunks.append((indices[start], indices[end - 1] + 1))  # end is exclusive

    return chunks


def _write_shard_wrapper(args: tuple) -> dict:
    """Wrapper function for multiprocessing shard writing.

    Parameters
    ----------
    args: tuple
        Tuple containing (ds, start_idx, end_idx, shard_id, output_path,
                         sample_prep_function, log_every, error_handling,
                         shard_name, compression)

    Returns
    -------
    dict: Results from _write_webdataset_shard
    """
    (
        ds,
        start_idx,
        end_idx,
        shard_id,
        output_path,
        sample_prep_function,
        log_every,
        error_handling,
        shard_name,
        compression,
    ) = args

    # Create batch by reading the specific slice of the dataset
    batch = []
    for i in range(start_idx, end_idx):
        batch.append(ds[i])

    # TODO: extend this function to be a factory for different formats
    return _write_webdataset_shard(
        batch=batch,
        shard_id=shard_id,
        output_path=output_path,
        sample_prep_function=sample_prep_function,
        log_every=log_every,
        error_handling=error_handling,
        shard_name=shard_name,
        compression=compression,
    )


def export_as_tar(
    ds: Dataset,
    output_path: str | AnyPathT,
    *,
    num_samples_per_shard: int = 1000,
    sample_prep_function: Callable | None = audio_encoder,
    log_every: int = 100,
    error_handling: str = "warn",
    shard_name: str = "shard",
    shard_start_id: int = 0,
    shuffle: bool = False,
    seed: int = 42,
    compression: Literal["gz", "bz2", "xz"] | None = None,
    max_workers: int | None = None,
    use_threading: bool = False,
    sample_prep_kwargs: dict | None = None,
) -> dict[str, Any]:
    """
    Export a dataset as WebDataset tar files with parallel processing
    and checkpointing support.

    Parameters
    ----------
    ds: Dataset
        The dataset to export
    output_path: str | AnyPathT
        Path to save the exported tar files
    num_samples_per_shard: int
        Number of samples per shard (default: 1000)
    sample_prep_function: Callable | None
        Function to prepare a sample for the WebDataset format.
        If None, uses the default audio encoder.
    log_every: int
        Log progress every N samples (default: 100)
    error_handling: str
        How to handle errors ("warn", "raise", "ignore") (default: "warn")
    shard_name: str
        Name (prefix) of the shard files (default: "shard")
    shard_start_id: int
        Starting ID for the shards (default: 0)
    shuffle: bool
        Whether to shuffle dataset indices before chunking (default: False)
    seed: int
        Random seed for shuffling (default: 42)
    compression: Literal["gz", "bz2", "xz"] | None
        Compression format for the tar file (default: None, no compression)
    max_workers: int | None
        Maximum number of worker processes/threads. If None, uses CPU count.
    use_threading: bool
        If True, uses ThreadPoolExecutor instead of ProcessPoolExecutor.
        Useful for I/O bound tasks or when sample_prep_function is not pickleable.

    Returns
    -------
    Dict[str, Any]: Summary of export results including:
        - total_shards: Number of shards created
        - total_processed: Total samples processed successfully
        - total_failed: Total samples that failed processing
        - shard_results: List of results from each shard
    """
    logger.info(f"Starting export of dataset to {output_path}")

    # Create output directory
    output_path = anypath(output_path)
    if isinstance(output_path, Path):
        output_path.mkdir(parents=True, exist_ok=True)

    # We will create a checkpoint to track progress and resume if needed
    checkpoint_path = output_path / "checkpoint.jsonl"
    if checkpoint_path.exists():
        checkpoint_df = pd.read_json(checkpoint_path, lines=True, orient="records")
    else:
        checkpoint_df = pd.DataFrame()

    # Split dataset into index chunks
    logger.info(f"Splitting dataset with {num_samples_per_shard} samples per shard")
    dataset_len = len(ds)
    chunks = _chunk_dataset_indices(ds, num_samples_per_shard, shuffle, seed)
    total_shards = len(chunks)

    logger.info(f"Dataset has {dataset_len} samples, creating {total_shards} shards for processing")

    # Create a partial sample_prep_function if kwargs are provided
    if sample_prep_function is not None and sample_prep_kwargs is not None:
        sample_prep_function = partial(sample_prep_function, **sample_prep_kwargs)

    # Prepare arguments for parallel processing
    shard_args = []
    for i, (start_idx, end_idx) in enumerate(chunks):
        shard_id = shard_start_id + i
        if not checkpoint_df.empty:
            # Check if this shard has already been processed
            if shard_id in checkpoint_df["shard_id"].values:
                logger.info(f"Skipping already processed shard {shard_id:05d}")
                continue

        args = (
            ds,
            start_idx,
            end_idx,
            shard_id,
            output_path,
            sample_prep_function,
            log_every,
            error_handling,
            shard_name,
            compression,
        )
        shard_args.append(args)

    # Determine number of workers
    if max_workers is None:
        max_workers = min(mp.cpu_count(), total_shards)

    logger.info(f"Using {max_workers} workers for parallel processing")

    # Process shards in parallel
    executor_class = (
        concurrent.futures.ThreadPoolExecutor
        if use_threading
        else concurrent.futures.ProcessPoolExecutor
    )

    total_processed = 0
    total_failed = 0

    with executor_class(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_shard = {
            executor.submit(_write_shard_wrapper, args): i for i, args in enumerate(shard_args)
        }

        # Process completed tasks
        with tqdm(
            total=total_shards,
            desc="Processing shards",
            ncols=100,
        ) as pbar:
            for future in concurrent.futures.as_completed(future_to_shard):
                shard_index = future_to_shard[future]
                try:
                    result = future.result()
                    total_processed += len(result["processed_ids"])
                    total_failed += len(result["failed_ids"])

                    logger.info(
                        f"Completed shard {result['shard_id']:05d} "
                        f"({shard_index + 1}/{total_shards}): "
                        f"processed={len(result['processed_ids'])}, "
                        f"failed={len(result['failed_ids'])}"
                    )

                    # Update checkpoint
                    checkpoint_entry = {
                        "shard_id": result["shard_id"],
                        "processed_ids": result["processed_ids"],
                        "failed_ids": result["failed_ids"],
                    }
                    checkpoint_df = pd.concat(
                        [checkpoint_df, pd.DataFrame([checkpoint_entry])],
                        ignore_index=True,
                    )
                    # Update checkpoint file
                    checkpoint_df.to_json(checkpoint_path, orient="records", lines=True)

                    # Update progress bar with shard info
                    pbar.set_postfix(
                        shard=f"{shard_index:06d}",
                        success=len(result["processed_ids"]),
                        failed=len(result["failed_ids"]),
                    )
                    pbar.update(1)

                except Exception as e:
                    logger.error(f"Shard {shard_index} failed with error: {e}")
                    if error_handling == "raise":
                        raise e

    # Create summary
    summary = {
        "total_shards": total_shards,
        "total_processed": total_processed,
        "total_failed": total_failed,
        "output_path": str(output_path),
        "compression": compression,
    }

    logger.info(
        f"Export completed: {total_shards} shards, "
        f"{total_processed} samples processed, "
        f"{total_failed} samples failed"
    )

    return summary
