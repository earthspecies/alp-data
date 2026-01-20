"""Resample all BirdSet files and save them to a target directory

This script provides two processing methods:

1. ThreadPoolExecutor (default):
   - Uses threading for parallel processing
   - Compatible with GCS fork-safety limitations
   - Recommended for faster processing with multiple workers
   - Use with --workers flag to control thread count

2. PyTorch DataLoader (--use-dataloader):
   - Uses PyTorch's DataLoader infrastructure
   - Limited to single-process mode due to GCS fork-safety
   - Useful for integration with PyTorch workflows
   - Access to batch processing capabilities

Note: Google Cloud Storage (gcsfs) is not fork-safe, so multiprocessing
cannot be used when reading from GCS. ThreadPoolExecutor uses threads
which don't have this limitation.
"""

import argparse
import functools
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, TypeVar

try:
    import numpy as np  # noqa: F401
except ImportError:
    print("Error: numpy is required but not installed. Install with: pip install numpy")
    exit(1)

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required but not installed. Install with: pip install pandas")
    exit(1)

try:
    import soundfile as sf
except ImportError:
    print("Error: soundfile is required but not installed. Install with: pip install soundfile")
    exit(1)

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print("Error: torch is required but not installed. Install with: pip install torch")
    exit(1)

try:
    from esp_data.datasets import BirdSet
    from esp_data.io import anypath
except ImportError as e:
    print(f"Error: Could not import esp_data modules: {e}")
    print("Make sure you're in the correct environment and esp_data is installed.")
    exit(1)

# Thread-safe counter for progress reporting
progress_lock = threading.Lock()
first_file_written = threading.Lock()
first_file_path_printed = [False]

# Thread-local storage for BirdSet instances
thread_local_data = threading.local()


@dataclass
class BenchmarkResult:
    """Container for benchmark timing results."""

    method: str
    split: str
    total_files: int
    total_time: float
    avg_time_per_file: float
    successful: int
    errors: int
    skipped: int
    workers: int
    batch_size: int
    files_per_second: float
    setup_time: float
    processing_time: float
    cleanup_time: float


F = TypeVar("F", bound=Callable[..., Any])


def benchmark_timing(func: F) -> F:
    """Decorator to measure function execution time.

    Returns
    -------
    F
        The decorated function
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Add timing info to result if it's a dict
        if isinstance(result, dict):
            result["execution_time"] = end_time - start_time
            result["start_time"] = start_time
            result["end_time"] = end_time

        return result

    return wrapper  # type: ignore[return-value]


def get_thread_local_birdset(split_name: str, target_sr: int) -> BirdSet:
    """Get or create a thread-local BirdSet instance for better performance.

    Returns
    -------
    BirdSet
        Thread-local BirdSet instance
    """
    if (
        not hasattr(thread_local_data, "birdset")
        or thread_local_data.split_name != split_name
        or thread_local_data.target_sr != target_sr
    ):
        thread_local_data.birdset = BirdSet(
            split=split_name,
            sample_rate=target_sr,
            data_root="gs://foundation-model-data/",
        )
        thread_local_data.split_name = split_name
        thread_local_data.target_sr = target_sr

    return thread_local_data.birdset


def should_skip_resample(target_path: Path) -> bool:
    """Check if the target file already exists and should be skipped.

    Returns
    -------
    bool
        True if the file exists and should be skipped
    """
    return target_path.exists()


def process_file_batch(
    indices: List[int],
    split_name: str,
    target_dir: Path,
    target_sr: int,
    total_files: int,
    progress_counter: list,
) -> List[Dict[str, Any]]:
    """Process a batch of files from the BirdSet dataset for better efficiency.

    Parameters
    ----------
    indices : List[int]
        List of indices to process in this batch
    split_name : str
        Name of the split being processed
    target_dir : Path
        Target directory for saving files
    target_sr : int
        Target sample rate
    total_files : int
        Total number of files for progress reporting
    progress_counter : list
        Shared counter for progress tracking (list to make it mutable)

    Returns
    -------
    List[Dict[str, Any]]
        Processing results for this batch

    Raises
    ------
    KeyError
        If neither 'path' nor 'local_path' found in sample
    """
    batch_start_time = time.time()
    results = []

    try:
        # Get thread-local BirdSet instance for efficiency
        birdset = get_thread_local_birdset(split_name, target_sr)

        for idx in indices:
            file_start_time = time.time()

            try:
                # Get sample with resampled audio
                sample = birdset[idx]

                audio_data = sample["audio"]

                # Try to get the path - check both possible key names
                if "path" in sample:
                    file_path = sample["path"]
                elif "local_path" in sample:
                    file_path = sample["local_path"]
                else:
                    raise KeyError(
                        f"Neither 'path' nor 'local_path' found in sample keys: "
                        f"{list(sample.keys())}"
                    )

                # Create target file path - preserve the original directory structure
                relative_path = Path(file_path)
                if relative_path.is_absolute():
                    relative_path = Path(*relative_path.parts[1:])

                target_path = target_dir / relative_path

                # Create parent directories if they don't exist
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Print the first file path regardless of whether it's written or skipped
                with first_file_written:
                    if not first_file_path_printed[0]:
                        print(f"📁 First file target path: {target_path}")
                        print(f"   Original path: {file_path}")
                        print(f"   Target directory: {target_dir}")
                        print(f"   File exists: {target_path.exists()}")
                        first_file_path_printed[0] = True

                # Check if file already exists
                if should_skip_resample(target_path):
                    result = {
                        "idx": idx,
                        "original_path": file_path,
                        "target_path": str(target_path),
                        "status": "skipped",
                        "error": None,
                        "processing_time": time.time() - file_start_time,
                    }
                else:
                    # Save the resampled audio
                    try:
                        save_start_time = time.time()
                        sf.write(target_path, audio_data, target_sr, format="FLAC")
                        save_time = time.time() - save_start_time

                        result = {
                            "idx": idx,
                            "original_path": file_path,
                            "target_path": str(target_path),
                            "status": "success",
                            "error": None,
                            "processing_time": time.time() - file_start_time,
                            "save_time": save_time,
                        }
                    except Exception as save_error:
                        result = {
                            "idx": idx,
                            "original_path": file_path,
                            "target_path": str(target_path),
                            "status": "error",
                            "error": f"Save error: {save_error}",
                            "processing_time": time.time() - file_start_time,
                        }

            except Exception as load_error:
                result = {
                    "idx": idx,
                    "original_path": "unknown",
                    "target_path": "unknown",
                    "status": "error",
                    "error": f"Load error: {load_error}",
                    "processing_time": time.time() - file_start_time,
                }

            results.append(result)

    except Exception as batch_error:
        # If there's an error with the entire batch, create error results for all indices
        for idx in indices:
            results.append(
                {
                    "idx": idx,
                    "original_path": "unknown",
                    "target_path": "unknown",
                    "status": "error",
                    "error": f"Batch error: {batch_error}",
                    "processing_time": time.time() - batch_start_time,
                }
            )

    # Update progress counter in a thread-safe manner
    with progress_lock:
        progress_counter[0] += len(results)
        if progress_counter[0] % 100 == 0:
            print(f"  Processed {progress_counter[0]}/{total_files} files in {split_name}")

    return results


def process_single_file(
    idx: int,
    split_name: str,
    target_dir: Path,
    target_sr: int,
    total_files: int,
    progress_counter: list,
) -> Dict[str, Any]:
    """Process a single file from the BirdSet dataset.

    This is kept for backward compatibility, but internally uses batch processing.

    Returns
    -------
    Dict[str, Any]
        Processing result for the file
    """
    return process_file_batch(
        [idx], split_name, target_dir, target_sr, total_files, progress_counter
    )[0]


class BirdSetResampleDataset(Dataset):
    """PyTorch Dataset wrapper for BirdSet with resampling and saving functionality."""

    def __init__(
        self,
        split_name: str,
        target_dir: Path,
        target_sr: int = 16000,
        save_files: bool = True,
        skip_existing: bool = True,
        test_mode: bool = False,
        test_samples: int = 10,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        split_name : str
            Name of the BirdSet split to process
        target_dir : Path
            Target directory for saving files
        target_sr : int
            Target sample rate for resampling
        save_files : bool
            Whether to save files to disk (default: True)
        skip_existing : bool
            Whether to skip existing files (default: True)
        test_mode : bool
            Whether to run in test mode with limited samples (default: False)
        test_samples : int
            Number of samples to process in test mode (default: 10)
        """
        setup_start = time.time()

        self.split_name = split_name
        self.target_dir = Path(target_dir)
        self.target_sr = target_sr
        self.save_files = save_files
        self.skip_existing = skip_existing
        self.test_mode = test_mode
        self.test_samples = test_samples

        # Initialize BirdSet dataset
        self.birdset = BirdSet(
            split=split_name,
            sample_rate=target_sr,
            data_root="gs://foundation-model-data/",
        )

        # Create target directory
        self.target_dir.mkdir(parents=True, exist_ok=True)

        total_files = len(self.birdset)

        if self.test_mode:
            self.actual_length = min(self.test_samples, total_files)
            print(
                f"🧪 TEST MODE: Processing {self.actual_length} files out of "
                f"{total_files} in {split_name}..."
            )
        else:
            self.actual_length = total_files
            print(f"Processing {total_files} files in {split_name}...")

        # Debug: print available keys for the first few samples to catch None issues
        if total_files > 0:
            print("Checking first few samples for debugging...")
            for i in range(min(3, total_files)):
                try:
                    sample = self.birdset[i]
                    if sample is None:
                        print(f"⚠️  Sample {i} is None!")
                        continue

                    print(f"Sample {i} - Available keys: {list(sample.keys())}")

                    # Check for None values in critical keys
                    critical_keys = ["audio", "path", "local_path"]
                    for key in critical_keys:
                        if key in sample:
                            if sample[key] is None:
                                print(f"⚠️  Sample {i} has None value for key '{key}'")
                            else:
                                if key == "audio":
                                    print(
                                        f"  {key}: <array "
                                        f"shape={getattr(sample[key], 'shape', 'unknown')}> "
                                        f"dtype={getattr(sample[key], 'dtype', 'unknown')}"
                                    )
                                else:
                                    print(f"  {key}: {sample[key]}")

                    # Only print a subset of the sample to avoid too much output
                    sample_subset = {k: v for k, v in sample.items() if k != "audio"}
                    if "audio" in sample:
                        sample_subset["audio"] = (
                            f"<array shape={getattr(sample['audio'], 'shape', 'unknown')}>"
                        )
                    print(f"  Full sample (subset): {sample_subset}")

                except Exception as e:
                    print(f"⚠️  Error accessing sample {i}: {e}")
                    continue

        self.setup_time = time.time() - setup_start
        print(f"Dataset setup completed in {self.setup_time:.2f}s")

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns
        -------
        int
            Number of samples in the dataset
        """
        return self.actual_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get and process a single item from the dataset.

        This method performs the same operations as process_single_file:
        - Gets the sample from BirdSet
        - Extracts audio data and file path
        - Creates target path structure
        - Optionally saves the resampled audio

        Parameters
        ----------
        idx : int
            Index of the item to retrieve

        Returns
        -------
        Dict[str, Any]
            Dictionary containing audio data, paths, and processing status

        Raises
        ------
        IndexError
            If index is out of range
        KeyError
            If neither 'path' nor 'local_path' found in sample
        ValueError
            If sample is None or invalid
        """
        start_time = time.time()

        # Validate index is within our actual length (important for test mode)
        if idx >= self.actual_length:
            raise IndexError(f"Index {idx} is out of range for dataset length {self.actual_length}")

        try:
            # Get sample first to avoid multiple dataset accesses
            load_start = time.time()
            sample = self.birdset[idx]
            load_time = time.time() - load_start

            # Check if sample is None or empty
            if sample is None:
                raise ValueError(f"BirdSet returned None for index {idx}")

            if not isinstance(sample, dict):
                raise ValueError(
                    f"BirdSet returned non-dict sample for index {idx}: {type(sample)}"
                )

            # Try to get the path - check both possible key names
            if "local_path" in sample:
                file_path = sample["local_path"]
            elif "path" in sample:
                file_path = sample["path"]
            else:
                raise KeyError(
                    f"Neither 'path' nor 'local_path' found in sample keys: {list(sample.keys())}"
                )

            # Validate path is not None
            if file_path is None:
                raise ValueError(f"File path is None for index {idx}")

            # Create target file path - preserve the original directory structure
            relative_path = Path(file_path)
            if relative_path.is_absolute():
                # If it's absolute, make it relative by removing the root
                relative_path = Path(*relative_path.parts[1:])

            target_path = self.target_dir / relative_path

            if self.skip_existing and target_path.exists():
                return {
                    "idx": idx,
                    "original_path": file_path,
                    "target_path": str(target_path),
                    "status": "skipped",
                    "error": None,
                    "processing_time": time.time() - start_time,
                    "load_time": load_time,
                    "save_time": 0.0,
                }

            # Create parent directories if they don't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract audio data from the already-fetched sample
            audio_data = sample.get("audio")
            if audio_data is None:
                raise ValueError(f"Audio data is None for index {idx}")

            # Initialize result dictionary
            result = {
                "idx": idx,
                "original_path": file_path,
                "target_path": str(target_path),
                "sample_rate": self.target_sr,
                "status": "success",
                "error": None,
                "metadata": {
                    k: v for k, v in sample.items() if k not in ["audio", "path", "local_path"]
                },
                "load_time": load_time,
                "save_time": 0.0,
            }

            # Optionally save the file
            if self.save_files:
                # Check if file already exists and should be skipped
                if self.skip_existing and target_path.exists():
                    result["status"] = "skipped"
                else:
                    try:
                        save_start = time.time()
                        sf.write(target_path, audio_data, self.target_sr, format="FLAC")
                        save_time = time.time() - save_start
                        result["status"] = "saved"
                        result["save_time"] = save_time

                        if idx == 0:
                            print(f"📁 First file target path: {target_path}")
                            print(f"   Original path: {file_path}")
                            print(f"   Target directory: {self.target_dir}")
                            print(f"   File exists: {target_path.exists()}")

                    except Exception as save_error:
                        result["status"] = "save_error"
                        result["error"] = f"Save error: {save_error}"

            result["processing_time"] = time.time() - start_time

            # Final safety check - ensure we never return None or invalid data
            if result is None:
                print(f"Warning: Result is None for idx {idx}, creating fallback")
                result = self._create_fallback_result(idx, "Result was None")

            # Ensure all required keys exist
            required_keys = [
                "idx",
                "status",
                "error",
                "original_path",
                "target_path",
                "sample_rate",
            ]
            for req_key in required_keys:
                if req_key not in result:
                    if req_key == "idx":
                        result[req_key] = idx
                    elif req_key == "status":
                        result[req_key] = "error"
                    elif req_key == "error":
                        result[req_key] = f"Missing key: {req_key}"
                    elif req_key in ["original_path", "target_path"]:
                        result[req_key] = "unknown"
                    elif req_key == "sample_rate":
                        result[req_key] = self.target_sr

            return result

        except Exception as load_error:
            print(f"Exception in __getitem__ for idx {idx}: {load_error}")
            return self._create_fallback_result(idx, f"Load error: {load_error}")

    def _create_fallback_result(self, idx: int, error_msg: str) -> Dict[str, Any]:
        """Create a fallback result when everything else fails.

        Returns
        -------
        Dict[str, Any]
            Fallback result dictionary with error information
        """
        return {
            "idx": idx,
            "original_path": "unknown",
            "target_path": "unknown",
            "sample_rate": self.target_sr,
            "status": "error",
            "error": error_msg,
            "metadata": {},
            "processing_time": 0.0,
            "load_time": 0.0,
            "save_time": 0.0,
        }


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching dataset results.

    This function is designed to handle any None values or malformed data
    that might come from the dataset, ensuring the DataLoader never fails.

    Parameters
    ----------
    batch : List[Dict[str, Any]]
        List of samples from the dataset

    Returns
    -------
    Dict[str, Any]
        Batched data
    """
    # Light debug output - only show critical issues
    none_count = sum(1 for item in batch if item is None)
    if none_count > 0:
        print(f"Warning: Found {none_count} None items in batch of {len(batch)}")

    non_dict_count = sum(1 for item in batch if item is not None and not isinstance(item, dict))
    if non_dict_count > 0:
        print(f"Warning: Found {non_dict_count} non-dict items in batch")

    # Filter out None values and non-dict items from batch
    filtered_batch = []
    for _i, item in enumerate(batch):
        if item is None:
            continue
        if not isinstance(item, dict):
            continue
        filtered_batch.append(item)

    if not filtered_batch:
        print("Error: All items in batch are None or invalid - creating fallback batch")
        # Return a valid but empty-like result to prevent crashes
        return {
            "idx": torch.tensor([-1]),
            "status": ["error"],
            "error": ["All batch items were None or invalid"],
            "original_path": ["unknown"],
            "target_path": ["unknown"],
            "sample_rate": torch.tensor([16000]),
        }

    # Initialize the result dictionary
    batched = {}

    # Get all keys from the first valid sample
    if not filtered_batch[0]:
        print("Error: First filtered item is empty - creating fallback batch")
        return {
            "idx": torch.tensor([-1]),
            "status": ["error"],
            "error": ["First item is empty"],
            "original_path": ["unknown"],
            "target_path": ["unknown"],
            "sample_rate": torch.tensor([16000]),
        }

    first_sample_keys = list(filtered_batch[0].keys())

    for key in first_sample_keys:
        try:
            if key in ["idx", "sample_rate"]:
                # Convert to tensor for numeric fields, handling None values
                values = []
                for sample in filtered_batch:
                    if sample is not None and key in sample and sample[key] is not None:
                        # Ensure the value is numeric
                        try:
                            val = int(sample[key]) if key == "idx" else float(sample[key])
                            values.append(val)
                        except (ValueError, TypeError):
                            print(f"Warning: Non-numeric value for {key}: {sample[key]}")
                            values.append(-1 if key == "idx" else 16000)
                    else:
                        # Use a default value for missing/None entries
                        values.append(-1 if key == "idx" else 16000)

                batched[key] = torch.tensor(values)
            else:
                # Keep as lists for string/mixed types, handling None values
                values = []
                for sample in filtered_batch:
                    if sample is not None and key in sample:
                        values.append(sample[key])
                    else:
                        # Use appropriate default for missing entries
                        if key == "status":
                            values.append("error")
                        elif key == "error":
                            values.append("Missing sample data")
                        elif key in ["original_path", "target_path"]:
                            values.append("unknown")
                        else:
                            values.append(None)

                batched[key] = values

        except Exception as e:
            # If there's an error with a specific key, create a safe default
            print(f"Error processing key '{key}' in batch: {e}")
            if key in ["idx", "sample_rate"]:
                batched[key] = torch.tensor([-1 if key == "idx" else 16000] * len(filtered_batch))
            else:
                default_val = "error" if key == "status" else f"Error processing {key}"
                batched[key] = [default_val] * len(filtered_batch)

    return batched


def create_dataloader(
    split_name: str,
    target_dir: Path,
    target_sr: int = 16000,
    batch_size: int = 1,
    num_workers: int = 4,
    save_files: bool = True,
    skip_existing: bool = True,
    test_mode: bool = False,
    test_samples: int = 10,
) -> DataLoader:
    """Create a PyTorch DataLoader for BirdSet resampling.

    Parameters
    ----------
    split_name : str
        Name of the BirdSet split to process
    target_dir : Path
        Target directory for saving files
    target_sr : int
        Target sample rate for resampling
    batch_size : int
        Batch size for the DataLoader
    num_workers : int
        Number of worker processes for data loading
    save_files : bool
        Whether to save files to disk
    skip_existing : bool
        Whether to skip existing files
    test_mode : bool
        Whether to run in test mode with limited samples
    test_samples : int
        Number of samples to process in test mode

    Returns
    -------
    DataLoader
        PyTorch DataLoader instance
    """
    dataset = BirdSetResampleDataset(
        split_name=split_name,
        target_dir=target_dir,
        target_sr=target_sr,
        save_files=save_files,
        skip_existing=skip_existing,
        test_mode=test_mode,
        test_samples=test_samples,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep original order for resampling
        num_workers=num_workers,
        collate_fn=custom_collate_fn,  # Always use custom collate function
    )


def process_split_with_dataloader(
    split_name: str,
    target_base_dir: Path,
    target_sr: int = 16000,
    batch_size: int = 1,
    num_workers: int = 4,
    save_files: bool = True,
    test_mode: bool = False,
    test_samples: int = 10,
) -> Dict[str, Any]:
    """Process a single BirdSet split using PyTorch DataLoader.

    Parameters
    ----------
    split_name : str
        Name of the split to process
    target_base_dir : Path
        Base target directory
    target_sr : int
        Target sample rate
    batch_size : int
        Batch size for processing
    num_workers : int
        Number of worker processes (will be set to 0 for GCS compatibility)
    save_files : bool
        Whether to save files to disk
    test_mode : bool
        Whether to run in test mode with limited samples
    test_samples : int
        Number of samples to process in test mode

    Returns
    -------
    Dict[str, Any]
        Processing statistics
    """
    total_start_time = time.time()
    setup_start_time = time.time()

    # Force num_workers=0 due to GCS fork-safety issues
    if num_workers > 0:
        print("⚠️  Warning: Setting num_workers=0 due to GCS fork-safety limitations.")
        print("   The gcsfs library used for Google Cloud Storage is not fork-safe.")
        print("   DataLoader will use single-process mode for compatibility.")
        num_workers = 0

    print(
        f"Processing split: {split_name} with DataLoader "
        f"(batch_size={batch_size}, num_workers={num_workers})"
    )

    # Create target directory for this split
    target_dir = target_base_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    # Create DataLoader
    dataloader = create_dataloader(
        split_name=split_name,
        target_dir=target_dir,
        target_sr=target_sr,
        batch_size=batch_size,
        num_workers=num_workers,
        save_files=save_files,
        skip_existing=True,
        test_mode=test_mode,
        test_samples=test_samples,
    )

    setup_time = time.time() - setup_start_time

    total_files = len(dataloader.dataset)
    print(f"Processing {total_files} files in {split_name}...")
    print(f"Setup completed in {setup_time:.2f}s")

    # Process batches
    processing_start_time = time.time()
    results = []
    # Initialize status_counters with all possible status values
    status_counters = {
        "success": 0,
        "saved": 0,
        "skipped": 0,
        "error": 0,
        "save_error": 0,
        "load_error": 0,  # Additional status that might occur
        "unknown": 0,  # Fallback for unexpected statuses
    }

    for batch_idx, batch in enumerate(dataloader):
        try:
            # Validate batch is not None
            if batch is None:
                print(f"Warning: Batch {batch_idx} is None, skipping...")
                continue

            # Debug: Print batch structure for first few batches
            if batch_idx < 3:
                print(f"Debug: Batch {batch_idx} structure:")
                for key, value in batch.items():
                    print(
                        f"  {key}: type={type(value)}, "
                        f"len={len(value) if hasattr(value, '__len__') else 'N/A'}"
                    )
                    if isinstance(value, (list, torch.Tensor)) and len(value) > 0:
                        first_elem = value[0]
                        # More detailed type info
                        if isinstance(first_elem, torch.Tensor):
                            elem_info = (
                                f"{first_elem} (tensor, shape={first_elem.shape}, "
                                f"dtype={first_elem.dtype})"
                            )
                        else:
                            elem_info = f"{first_elem} (type: {type(first_elem)})"
                        print(f"    First element: {elem_info}")

            # Handle both single items and batches - always extract individual items
            batch_results = []
            if "idx" not in batch:
                print(f"Warning: Batch {batch_idx} missing 'idx' key, skipping...")
                continue

            # Determine actual batch size from the idx tensor/list
            if isinstance(batch["idx"], torch.Tensor):
                batch_size_actual = len(batch["idx"])
            elif isinstance(batch["idx"], list):
                batch_size_actual = len(batch["idx"])
            else:
                print(f"Warning: Unexpected type for idx: {type(batch['idx'])}")
                batch_size_actual = 1

            # Extract individual items from the batch
            for i in range(batch_size_actual):
                item = {}
                for key, value in batch.items():
                    try:
                        if isinstance(value, torch.Tensor):
                            # Extract the i-th element from tensor
                            if len(value) == 1 and batch_size_actual == 1:
                                # Single element tensor for single item batch
                                extracted = value.item()
                                if batch_idx < 3:  # Debug for first few batches
                                    print(
                                        f"    Extracted {key} (single): {extracted} "
                                        f"(type: {type(extracted)})"
                                    )
                            else:
                                # Multi-element tensor or multi-item batch
                                extracted = value[i]
                                if batch_idx < 3:  # Debug for first few batches
                                    print(
                                        f"    Pre-conversion {key}[{i}]: {extracted} "
                                        f"(type: {type(extracted)})"
                                    )
                                # If the extracted value is still a tensor, convert to scalar
                                if isinstance(extracted, torch.Tensor):
                                    extracted = extracted.item()
                                    if batch_idx < 3:  # Debug for first few batches
                                        print(
                                            f"    Post-conversion {key}: {extracted} "
                                            f"(type: {type(extracted)})"
                                        )
                            item[key] = extracted
                        elif isinstance(value, list):
                            # Extract the i-th element from list
                            if len(value) == 1 and batch_size_actual == 1:
                                # Single element list for single item batch
                                item[key] = value[0]
                            else:
                                # Multi-element list or multi-item batch
                                item[key] = value[i]
                        else:
                            # Scalar value - same for all items in batch
                            item[key] = value
                    except (IndexError, AttributeError) as e:
                        print(f"Warning: Error extracting key '{key}' from batch item {i}: {e}")
                        print(f"  Value type: {type(value)}, Value: {value}")
                        # Set appropriate default based on key type
                        if key == "idx":
                            item[key] = -1
                        elif key == "status":
                            item[key] = "error"
                        elif key == "error":
                            item[key] = f"Batch extraction error: {e}"
                        elif key in ["original_path", "target_path"]:
                            item[key] = "unknown"
                        elif key == "sample_rate":
                            item[key] = 16000
                        else:
                            item[key] = None

                # Validate extracted item has correct types
                if batch_idx < 3:  # Debug for first few batches
                    print(f"    Final item {i}: {item}")

                # Ensure critical fields are correct types
                if "status" in item and isinstance(item["status"], (list, torch.Tensor)):
                    print(f"Warning: Status is still {type(item['status'])}: {item['status']}")
                    if isinstance(item["status"], list):
                        item["status"] = item["status"][0] if item["status"] else "error"
                    elif isinstance(item["status"], torch.Tensor):
                        item["status"] = (
                            item["status"].item()
                            if item["status"].numel() == 1
                            else str(item["status"])
                        )

                if "idx" in item and isinstance(item["idx"], (list, torch.Tensor)):
                    print(f"Warning: idx is still {type(item['idx'])}: {item['idx']}")
                    if isinstance(item["idx"], list):
                        item["idx"] = item["idx"][0] if item["idx"] else -1
                    elif isinstance(item["idx"], torch.Tensor):
                        item["idx"] = item["idx"].item() if item["idx"].numel() == 1 else -1

                batch_results.append(item)

            # Process each item in the batch
            for item in batch_results:
                try:
                    # Convert any remaining tensor values back to python types
                    result = {}

                    if not isinstance(item, dict):
                        print(f"Warning: Item is not a dict: {type(item)}")
                        # Create error result for non-dict item
                        error_result = {
                            "idx": -1,
                            "status": "error",
                            "error": f"Item is not a dict: {type(item)}",
                            "original_path": "unknown",
                            "target_path": "unknown",
                            "sample_rate": 16000,
                        }
                        results.append(error_result)
                        status_counters["error"] += 1
                        continue

                    for key, value in item.items():
                        try:
                            if isinstance(value, torch.Tensor):
                                try:
                                    if value.numel() == 1:
                                        # Convert single-element tensor to scalar
                                        result[key] = value.item()
                                    else:
                                        # Convert multi-element tensor to list
                                        # (more compatible than numpy array)
                                        result[key] = value.tolist()
                                except Exception as tensor_error:
                                    print(
                                        f"Warning: Error converting tensor for key "
                                        f"'{key}': {tensor_error}"
                                    )
                                    # Fallback: try to convert to string
                                    result[key] = str(value)
                            elif isinstance(value, list):
                                # Handle lists directly - don't try to make them hashable
                                result[key] = value
                            elif hasattr(value, "item"):
                                # Handle numpy scalars and similar
                                try:
                                    result[key] = value.item()
                                except Exception:
                                    result[key] = str(value)
                            else:
                                # Handle regular python types
                                result[key] = value

                        except Exception as key_error:
                            print(
                                f"Warning: Error processing key '{key}' with value type "
                                f"{type(value)}: {key_error}"
                            )
                            # Set a safe default based on key name
                            if key == "idx":
                                result[key] = -1
                            elif key == "status":
                                result[key] = "error"
                            elif key == "error":
                                result[key] = f"Key processing error: {key_error}"
                            elif key in ["original_path", "target_path"]:
                                result[key] = "unknown"
                            elif key == "sample_rate":
                                result[key] = 16000
                            else:
                                result[key] = None

                    # Ensure result has required keys and correct types
                    required_keys = [
                        "idx",
                        "status",
                        "error",
                        "original_path",
                        "target_path",
                        "sample_rate",
                    ]
                    for req_key in required_keys:
                        if req_key not in result:
                            if req_key == "idx":
                                result[req_key] = -1
                            elif req_key == "status":
                                result[req_key] = "error"
                            elif req_key == "error":
                                result[req_key] = "Missing required key"
                            elif req_key in ["original_path", "target_path"]:
                                result[req_key] = "unknown"
                            elif req_key == "sample_rate":
                                result[req_key] = 16000

                    # Validate critical types to avoid unhashable errors
                    if not isinstance(result["status"], str):
                        print(
                            f"Warning: Status is not a string: "
                            f"{type(result['status'])}, value: {result['status']}"
                        )
                        if isinstance(result["status"], list):
                            result["status"] = result["status"][0] if result["status"] else "error"
                        else:
                            result["status"] = (
                                str(result["status"]) if result["status"] is not None else "error"
                            )

                    if not isinstance(result["idx"], (int, float)):
                        print(
                            f"Warning: idx is not numeric: "
                            f"{type(result['idx'])}, value: {result['idx']}"
                        )
                        try:
                            if isinstance(result["idx"], list):
                                result["idx"] = result["idx"][0] if result["idx"] else -1
                            else:
                                result["idx"] = (
                                    int(result["idx"]) if result["idx"] is not None else -1
                                )
                        except (ValueError, TypeError):
                            result["idx"] = -1

                    results.append(result)

                    # Safely use status as dictionary key
                    safe_status = result["status"]
                    if safe_status not in status_counters:
                        print(f"Warning: Unknown status '{safe_status}', treating as 'unknown'")
                        safe_status = "unknown"

                    try:
                        status_counters[safe_status] += 1
                    except Exception as counter_error:
                        print(
                            f"Error incrementing counter for status "
                            f"'{safe_status}': {counter_error}"
                        )
                        print(
                            f"Status type: {type(safe_status)}, "
                            f"Counter keys: {list(status_counters.keys())}"
                        )
                        # Fallback to unknown
                        status_counters["unknown"] += 1

                except Exception as item_error:
                    print(f"Warning: Error processing item in batch {batch_idx}: {item_error}")
                    import traceback

                    print(f"Traceback: {traceback.format_exc()}")

                    # Try to extract idx if possible
                    try:
                        idx = item.get("idx", -1) if isinstance(item, dict) else -1
                    except Exception:
                        idx = -1

                    error_result = {
                        "idx": idx,
                        "status": "error",
                        "error": f"Item processing error: {item_error}",
                        "original_path": "unknown",
                        "target_path": "unknown",
                        "sample_rate": 16000,
                    }
                    results.append(error_result)
                    status_counters["error"] += 1

            # Print progress every 100 files
            if (batch_idx + 1) * batch_size % 100 == 0:
                total_processed = len(results)
                print(
                    f"  Progress {total_processed}/{total_files}: "
                    f"✅ {status_counters['saved']} saved, "
                    f"⏭️  {status_counters['skipped']} skipped, "
                    f"❌ {status_counters['error'] + status_counters['save_error']} errors"
                )

        except Exception as batch_error:
            print(f"Error processing batch {batch_idx}: {batch_error}")
            # Create error results for this batch
            error_result = {
                "idx": batch_idx,
                "status": "error",
                "error": f"Batch processing error: {batch_error}",
                "original_path": "unknown",
                "target_path": "unknown",
            }
            results.append(error_result)
            status_counters["error"] += 1

    # Calculate final statistics
    processing_time = time.time() - processing_start_time
    total_time = time.time() - total_start_time
    cleanup_time = 0.0  # No significant cleanup for DataLoader

    successful = status_counters["success"] + status_counters["saved"]
    errors = status_counters["error"] + status_counters["save_error"]
    skipped = status_counters["skipped"]

    # Calculate timing statistics
    avg_time_per_file = total_time / max(total_files, 1)
    files_per_second = total_files / max(total_time, 0.001)

    # Calculate average processing times from individual results
    individual_times = [r.get("processing_time", 0) for r in results if "processing_time" in r]
    avg_individual_time = sum(individual_times) / max(len(individual_times), 1)

    print(f"Split {split_name} completed:")
    print(f"  Success: {successful}")
    print(f"  Errors: {errors}")
    print(f"  Skipped: {skipped}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Files per second: {files_per_second:.2f}")
    print(f"  Avg time per file: {avg_time_per_file:.3f}s")

    return {
        "split": split_name,
        "status": "completed",
        "processed": len(results),
        "successful": successful,
        "errors": errors,
        "skipped": skipped,
        "results": results,
        # Benchmark data
        "benchmark": BenchmarkResult(
            method="DataLoader",
            split=split_name,
            total_files=total_files,
            total_time=total_time,
            avg_time_per_file=avg_time_per_file,
            successful=successful,
            errors=errors,
            skipped=skipped,
            workers=num_workers,
            batch_size=batch_size,
            files_per_second=files_per_second,
            setup_time=setup_time,
            processing_time=processing_time,
            cleanup_time=cleanup_time,
        ),
        "timing": {
            "total_time": total_time,
            "setup_time": setup_time,
            "processing_time": processing_time,
            "cleanup_time": cleanup_time,
            "avg_time_per_file": avg_time_per_file,
            "avg_individual_time": avg_individual_time,
            "files_per_second": files_per_second,
        },
    }


def process_split(
    split_name: str,
    target_base_dir: Path,
    target_sr: int = 16000,
    max_workers: int = 4,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Process a single BirdSet split using parallel processing with ThreadPoolExecutor.

    Parameters
    ----------
    split_name : str
        Name of the split to process
    target_base_dir : Path
        Base target directory
    target_sr : int
        Target sample rate
    max_workers : int
        Maximum number of worker threads
    batch_size : int
        Number of files to process per batch (for batch processing optimization)

    Returns
    -------
    Dict[str, Any]
        Processing statistics
    """
    import random

    total_start_time = time.time()
    setup_start_time = time.time()

    print(f"Processing split: {split_name} with {max_workers} workers (batch_size={batch_size})")

    # Reset first file flag for this split
    with first_file_written:
        first_file_path_printed[0] = False

    # Create target directory for this split
    target_dir = target_base_dir  # Do not append split_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Initialize BirdSet dataset to get the total number of files
    birdset = BirdSet(
        split=split_name,
        sample_rate=target_sr,
        data_root="gs://foundation-model-data/",
    )

    total_files = len(birdset)
    print(f"Processing {total_files} files in {split_name}...")

    # Debug: print available keys for the first sample
    if total_files > 0:
        sample = birdset[0]
        print(f"Available keys in sample: {list(sample.keys())}")
        # Only print a subset of the sample to avoid too much output
        sample_subset = {k: v for k, v in sample.items() if k != "audio"}
        sample_subset["audio"] = f"<array shape={sample['audio'].shape}>"
        print(f"Sample: {sample_subset}")

    setup_time = time.time() - setup_start_time
    print(f"Setup completed in {setup_time:.2f}s")

    # Initialize progress counter
    progress_counter = [0]
    status_counters = {"success": 0, "skipped": 0, "error": 0}
    status_lock = threading.Lock()

    # Process files in parallel
    processing_start_time = time.time()
    results = []

    # Create batches for processing
    if batch_size > 1:
        # Batch processing for better efficiency
        # Randomize file order for better load balancing
        all_indices = list(range(total_files))
        random.shuffle(all_indices)

        batches = []
        for i in range(0, total_files, batch_size):
            batch_indices = all_indices[i : i + batch_size]
            batches.append(batch_indices)

        print(f"Processing {len(batches)} batches of size {batch_size}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batch tasks
            future_to_batch = {
                executor.submit(
                    process_file_batch,
                    batch_indices,
                    split_name,
                    target_dir,
                    target_sr,
                    total_files,
                    progress_counter,
                ): batch_indices
                for batch_indices in batches
            }

            # Collect results as they complete
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)

                    # Update status counters
                    with status_lock:
                        for result in batch_results:
                            status_counters[result["status"]] += 1

                        # Print progress with status breakdown every 100 files
                        total_completed = sum(status_counters.values())
                        if total_completed % 100 == 0:
                            print(
                                f"  Progress {total_completed}/{total_files}: "
                                f"✅ {status_counters['success']} written, "
                                f"⏭️  {status_counters['skipped']} skipped, "
                                f"❌ {status_counters['error']} errors"
                            )

                except Exception as exc:
                    batch_indices = future_to_batch[future]
                    print(f"Batch {batch_indices} generated an exception: {exc}")
                    for idx in batch_indices:
                        results.append(
                            {
                                "idx": idx,
                                "original_path": "unknown",
                                "target_path": "unknown",
                                "status": "error",
                                "error": f"Batch exception: {exc}",
                                "processing_time": 0.0,
                            }
                        )

                    # Update error counter
                    with status_lock:
                        status_counters["error"] += len(batch_indices)
    else:
        # Single file processing (original method)
        # Randomize file order for better load balancing
        all_indices = list(range(total_files))
        random.shuffle(all_indices)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    process_single_file,
                    idx,
                    split_name,
                    target_dir,
                    target_sr,
                    total_files,
                    progress_counter,
                ): idx
                for idx in all_indices
            }

            # Collect results as they complete
            for future in as_completed(future_to_idx):
                try:
                    result = future.result()
                    results.append(result)

                    # Update status counters
                    with status_lock:
                        status_counters[result["status"]] += 1

                        # Print progress with status breakdown every 100 files
                        total_completed = sum(status_counters.values())
                        if total_completed % 100 == 0:
                            print(
                                f"  Progress {total_completed}/{total_files}: "
                                f"✅ {status_counters['success']} written, "
                                f"⏭️  {status_counters['skipped']} skipped, "
                                f"❌ {status_counters['error']} errors"
                            )

                except Exception as exc:
                    idx = future_to_idx[future]
                    print(f"File {idx} generated an exception: {exc}")
                    results.append(
                        {
                            "idx": idx,
                            "original_path": "unknown",
                            "target_path": "unknown",
                            "status": "error",
                            "error": f"Worker exception: {exc}",
                            "processing_time": 0.0,
                        }
                    )

                    # Update error counter
                    with status_lock:
                        status_counters["error"] += 1

    # Calculate final statistics
    processing_time = time.time() - processing_start_time
    total_time = time.time() - total_start_time
    cleanup_time = 0.0  # No significant cleanup for ThreadPoolExecutor

    successful = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")
    skipped = sum(1 for r in results if r["status"] == "skipped")

    # Calculate timing statistics
    avg_time_per_file = total_time / max(total_files, 1)
    files_per_second = total_files / max(total_time, 0.001)

    # Calculate average processing times from individual results
    individual_times = [r.get("processing_time", 0) for r in results if "processing_time" in r]
    avg_individual_time = sum(individual_times) / max(len(individual_times), 1)

    # Calculate save times if available
    save_times = [r.get("save_time", 0) for r in results if "save_time" in r and r["save_time"] > 0]
    avg_save_time = sum(save_times) / max(len(save_times), 1) if save_times else 0.0

    print(f"Split {split_name} completed:")
    print(f"  Success: {successful}")
    print(f"  Errors: {errors}")
    print(f"  Skipped: {skipped}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Files per second: {files_per_second:.2f}")
    print(f"  Avg time per file: {avg_time_per_file:.3f}s")
    if save_times:
        print(f"  Avg save time: {avg_save_time:.3f}s")

    return {
        "split": split_name,
        "status": "completed",
        "processed": len(results),
        "successful": successful,
        "errors": errors,
        "skipped": skipped,
        "results": results,
        # Benchmark data
        "benchmark": BenchmarkResult(
            method=f"ThreadPoolExecutor (batch_size={batch_size})",
            split=split_name,
            total_files=total_files,
            total_time=total_time,
            avg_time_per_file=avg_time_per_file,
            successful=successful,
            errors=errors,
            skipped=skipped,
            workers=max_workers,
            batch_size=batch_size,
            files_per_second=files_per_second,
            setup_time=setup_time,
            processing_time=processing_time,
            cleanup_time=cleanup_time,
        ),
        "timing": {
            "total_time": total_time,
            "setup_time": setup_time,
            "processing_time": processing_time,
            "cleanup_time": cleanup_time,
            "avg_time_per_file": avg_time_per_file,
            "avg_individual_time": avg_individual_time,
            "avg_save_time": avg_save_time,
            "files_per_second": files_per_second,
        },
    }


def run_benchmark_comparison(
    splits_to_test: List[str],
    target_dir: Path,
    target_sr: int = 16000,
    max_files_per_split: int | None = None,
    worker_counts: List[int] | None = None,
    batch_sizes: List[int] | None = None,
) -> Dict[str, Any]:
    """Run a comprehensive benchmark comparison between different processing methods.

    Parameters
    ----------
    splits_to_test : List[str]
        List of splits to test
    target_dir : Path
        Target directory for testing
    target_sr : int
        Target sample rate
    max_files_per_split : Optional[int]
        Maximum number of files to process per split for testing (None = all files)
    worker_counts : List[int] | None
        List of worker counts to test (default: [1, 2, 4, 8])
    batch_sizes : List[int] | None
        List of batch sizes to test for ThreadPoolExecutor (default: [1, 5, 10])

    Returns
    -------
    Dict[str, Any]
        Comprehensive benchmark results
    """
    if worker_counts is None:
        worker_counts = [1, 2, 4, 8]
    if batch_sizes is None:
        batch_sizes = [1, 5, 10]
    print("=" * 60)
    print("COMPREHENSIVE BENCHMARK COMPARISON")
    print("=" * 60)

    benchmark_results = []
    test_dir = target_dir / "benchmark_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    for split in splits_to_test:
        print(f"\n🧪 Testing split: {split}")

        # Test DataLoader method (single configuration due to GCS limitations)
        print("\n--- Testing DataLoader Method ---")
        try:
            result_dataloader = process_split_with_dataloader(
                split_name=split,
                target_base_dir=test_dir / f"dataloader_{split}",
                target_sr=target_sr,
                batch_size=1,  # Fixed due to GCS limitations
                num_workers=0,  # Fixed due to GCS limitations
                save_files=True,
            )
            benchmark_results.append(result_dataloader["benchmark"])
            print(
                f"✅ DataLoader completed: "
                f"{result_dataloader['timing']['files_per_second']:.2f} files/sec"
            )
        except Exception as e:
            print(f"❌ DataLoader failed: {e}")

        # Test ThreadPoolExecutor method with different configurations
        print("\n--- Testing ThreadPoolExecutor Method ---")
        for workers in worker_counts:
            for batch_size in batch_sizes:
                config_name = f"workers_{workers}_batch_{batch_size}"
                print(f"Testing: {workers} workers, batch size {batch_size}")

                try:
                    result_threads = process_split(
                        split_name=split,
                        target_base_dir=test_dir / f"threads_{split}_{config_name}",
                        target_sr=target_sr,
                        max_workers=workers,
                        batch_size=batch_size,
                    )
                    benchmark_results.append(result_threads["benchmark"])
                    print(
                        f"✅ ThreadPool {config_name}: "
                        f"{result_threads['timing']['files_per_second']:.2f} files/sec"
                    )
                except Exception as e:
                    print(f"❌ ThreadPool {config_name} failed: {e}")

    # Analyze results
    print("\n" + "=" * 60)
    print("BENCHMARK ANALYSIS")
    print("=" * 60)

    if benchmark_results:
        # Convert to DataFrame for analysis
        df = pd.DataFrame([asdict(br) for br in benchmark_results])

        # Group by method for comparison
        method_groups = df.groupby("method")

        print("\n📊 Performance Summary (files per second):")
        for method, group in method_groups:
            avg_fps = group["files_per_second"].mean()
            max_fps = group["files_per_second"].max()
            min_fps = group["files_per_second"].min()
            print(f"{method:30s}: avg={avg_fps:6.2f}, max={max_fps:6.2f}, min={min_fps:6.2f}")

        # Find best configuration
        best_result = df.loc[df["files_per_second"].idxmax()]
        print("\n🏆 Best Configuration:")
        print(f"   Method: {best_result['method']}")
        print(f"   Workers: {best_result['workers']}")
        print(f"   Batch Size: {best_result['batch_size']}")
        print(f"   Performance: {best_result['files_per_second']:.2f} files/sec")
        print(f"   Avg Time per File: {best_result['avg_time_per_file']:.3f}s")

        # Save detailed results
        benchmark_file = target_dir / "benchmark_results.csv"
        df.to_csv(benchmark_file, index=False)
        print(f"\n📄 Detailed results saved to: {benchmark_file}")

        return {
            "benchmark_results": benchmark_results,
            "summary_df": df,
            "best_config": best_result.to_dict(),
            "benchmark_file": str(benchmark_file),
        }
    else:
        print("❌ No benchmark results collected")
        return {"benchmark_results": [], "error": "No results collected"}


def main() -> None:
    """Resample all BirdSet files and save them to a target directory."""

    parser = argparse.ArgumentParser(
        description="Resample all BirdSet files and save them to a target directory."
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Target directory to copy the resampled files to.",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=16000,
        help="Sample rate to which the audio files should be resampled.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers to use for processing (default: 4).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        help=(
            "Specific splits to process. Use 'all' to process all available splits. "
            "If not provided, processes all splits."
        ),
    )
    parser.add_argument(
        "--use-dataloader",
        action="store_true",
        help="Use PyTorch DataLoader for processing instead of ThreadPoolExecutor. "
        "Note: Uses single-process mode due to GCS fork-safety limitations.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Batch size for processing. For DataLoader: batch size for DataLoader. "
            "For ThreadPoolExecutor: files per batch."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Run comprehensive benchmark comparison between methods with different configurations."
        ),
    )
    parser.add_argument(
        "--benchmark-splits",
        type=str,
        nargs="+",
        default=["val"],
        help=(
            "Splits to use for benchmarking (default: val). Use smaller splits for "
            "faster benchmarking."
        ),
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help=(
            "Run in test mode with only a few samples for debugging. Processes only first 10 files."
        ),
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=10,
        help="Number of samples to process in test mode (default: 10).",
    )
    parser.add_argument(
        "--test-dataloader",
        action="store_true",
        help="Run DataLoader functionality test and exit. Useful for debugging DataLoader issues.",
    )

    args = parser.parse_args()

    # Handle test-dataloader mode
    if args.test_dataloader:
        print("🧪 Running DataLoader functionality test...")
        test_splits = args.benchmark_splits if hasattr(args, "benchmark_splits") else ["val"]
        for split in test_splits:
            print(f"\nTesting split: {split}")
            success = test_dataloader_functionality(split, args.test_samples)
            if not success:
                print(f"❌ Test failed for split {split}")
                return
            print(f"✅ Test passed for split {split}")
        print("\n🎉 All DataLoader tests passed!")
        return

    target_dir = anypath(args.target_dir)

    # Validate number of workers
    if args.workers <= 0:
        print("Error: Number of workers must be greater than 0")
        return

    # Get all available splits
    all_splits = BirdSet.info.split_paths.keys()

    if args.splits:
        # Check if 'all' is specified
        if "all" in args.splits:
            splits_to_process = list(all_splits)
            print(f"Processing all available splits: {splits_to_process}")
        else:
            # Validate provided splits
            invalid_splits = [s for s in args.splits if s not in all_splits]
            if invalid_splits:
                print(f"Invalid splits: {invalid_splits}")
                print(f"Available splits: {list(all_splits)}")
                return
            splits_to_process = args.splits
    else:
        splits_to_process = list(all_splits)

    # Handle benchmark mode
    if args.benchmark:
        print("🧪 Running benchmark mode...")

        # Validate benchmark splits
        invalid_benchmark_splits = [s for s in args.benchmark_splits if s not in all_splits]
        if invalid_benchmark_splits:
            print(f"Invalid benchmark splits: {invalid_benchmark_splits}")
            print(f"Available splits: {list(all_splits)}")
            return

        run_benchmark_comparison(
            splits_to_test=args.benchmark_splits,
            target_dir=target_dir,
            target_sr=args.target_sr,
            max_files_per_split=None,
            worker_counts=[1, 2, 4, 8] if args.workers >= 8 else [1, 2, args.workers],
            batch_sizes=[1, 5, 10, 20] if args.batch_size <= 10 else [1, args.batch_size],
        )

        return  # Exit after benchmarking

    processing_method = "PyTorch DataLoader" if args.use_dataloader else "ThreadPoolExecutor"
    print(f"Processing {len(splits_to_process)} splits: {splits_to_process}")
    print(f"Processing method: {processing_method}")

    if args.test_mode:
        print(f"🧪 TEST MODE ENABLED: Processing only {args.test_samples} samples per split")

    if args.use_dataloader:
        print(
            "ℹ️  Note: DataLoader mode uses single-process execution due to GCS "
            "fork-safety limitations."
        )
        print("   For faster parallel processing, use ThreadPoolExecutor mode (default).")

    print(f"Target directory: {target_dir}")
    print(f"Target sample rate: {args.target_sr}")
    print(f"Batch size: {args.batch_size}")

    if args.use_dataloader:
        print(f"DataLoader batch size: {args.batch_size}")
        print(f"Number of workers: {args.workers}")
    else:
        print(f"Number of worker threads: {args.workers}")
        print(f"Files per batch: {args.batch_size}")

    print("Note: This script requires access to Google Cloud Storage (GCS).")
    print("Make sure you have proper GCS authentication set up (gcloud auth or service account).")

    # Process each split
    all_results = []
    all_errors = []
    all_benchmarks = []

    for split_name in splits_to_process:
        try:
            if args.use_dataloader:
                result = process_split_with_dataloader(
                    split_name=split_name,
                    target_base_dir=target_dir,
                    target_sr=args.target_sr,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    save_files=True,
                    test_mode=args.test_mode,
                    test_samples=args.test_samples,
                )
            else:
                result = process_split(
                    split_name=split_name,
                    target_base_dir=target_dir,
                    target_sr=args.target_sr,
                    max_workers=args.workers,
                    batch_size=args.batch_size,
                )

            all_results.append(result)

            # Collect benchmark data if available
            if "benchmark" in result:
                all_benchmarks.append(result["benchmark"])

            # Collect errors from this split
            if result["status"] == "completed":
                split_errors = [r for r in result["results"] if r["status"] == "error"]
                all_errors.extend(split_errors)

        except Exception as e:
            print(f"Error processing split {split_name}: {e}")
            all_results.append(
                {
                    "split": split_name,
                    "status": "error",
                    "error": str(e),
                    "processed": 0,
                    "errors": 0,
                    "skipped": 0,
                }
            )

    # Save results to CSV files
    if all_errors:
        error_df = pd.DataFrame(all_errors)
        error_file = target_dir / "resample_errors.csv"
        error_df.to_csv(error_file, index=False)
        print(f"Errors saved to {error_file}")

    # Create summary
    summary_data = []
    for result in all_results:
        base_data = {
            "split": result["split"],
            "status": result["status"],
            "processed": result.get("processed", 0),
            "successful": result.get("successful", 0),
            "errors": result.get("errors", 0),
            "skipped": result.get("skipped", 0),
            "error": result.get("error", ""),
            "processing_method": processing_method,
        }

        # Add timing information if available
        if "timing" in result:
            timing = result["timing"]
            base_data.update(
                {
                    "total_time": timing.get("total_time", 0),
                    "setup_time": timing.get("setup_time", 0),
                    "processing_time": timing.get("processing_time", 0),
                    "files_per_second": timing.get("files_per_second", 0),
                    "avg_time_per_file": timing.get("avg_time_per_file", 0),
                    "avg_save_time": timing.get("avg_save_time", 0),
                }
            )

        summary_data.append(base_data)

    summary_df = pd.DataFrame(summary_data)
    summary_file = target_dir / "resample_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")

    # Save benchmark results if available
    if all_benchmarks:
        benchmark_df = pd.DataFrame([asdict(br) for br in all_benchmarks])
        benchmark_file = target_dir / "performance_benchmarks.csv"
        benchmark_df.to_csv(benchmark_file, index=False)
        print(f"Benchmark results saved to {benchmark_file}")

    # Print final summary
    total_processed = sum(r.get("processed", 0) for r in all_results)
    total_successful = sum(r.get("successful", 0) for r in all_results)
    total_errors = sum(r.get("errors", 0) for r in all_results)
    total_skipped = sum(r.get("skipped", 0) for r in all_results)

    # Calculate total timing
    total_time = sum(r.get("timing", {}).get("total_time", 0) for r in all_results)
    overall_fps = total_processed / max(total_time, 0.001) if total_time > 0 else 0

    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(f"Processing method: {processing_method}")
    print(f"Total files processed: {total_processed}")
    print(f"Total successful: {total_successful}")
    print(f"Total errors: {total_errors}")
    print(f"Total skipped: {total_skipped}")
    print(f"Success rate: {total_successful / (total_processed or 1) * 100:.1f}%")

    if total_time > 0:
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Overall throughput: {overall_fps:.2f} files/second")

    # Print performance summary if benchmarks available
    if all_benchmarks:
        print("\n📊 PERFORMANCE BREAKDOWN:")
        for benchmark in all_benchmarks:
            print(f"  {benchmark.split} ({benchmark.method}):")
            print(f"    Files/sec: {benchmark.files_per_second:.2f}")
            print(f"    Avg time: {benchmark.avg_time_per_file:.3f}s per file")
            print(f"    Setup: {benchmark.setup_time:.2f}s")
            print(f"    Workers: {benchmark.workers}, Batch: {benchmark.batch_size}")

    print("\n📄 Results saved to:")
    print(f"  Summary: {summary_file}")
    if all_benchmarks:
        print(f"  Benchmarks: {benchmark_file}")
    if all_errors:
        print(f"  Errors: {error_file}")


def test_dataloader_functionality(split_name: str = "val", num_samples: int = 5) -> bool:
    """Test that the DataLoader works correctly with a small number of samples.

    Returns
    -------
    bool
        True if test passed, False otherwise
    """
    print("=" * 60)
    print("TESTING DATALOADER FUNCTIONALITY")
    print("=" * 60)

    try:
        # Test dataset creation
        print(f"1. Creating dataset for split '{split_name}' with {num_samples} samples...")
        dataset = BirdSetResampleDataset(
            split_name=split_name,
            target_dir=Path("./test_dataloader"),
            target_sr=16000,
            save_files=False,  # Don't save files during testing
            skip_existing=True,
            test_mode=True,
            test_samples=num_samples,
        )
        print(f"   ✅ Dataset created successfully with {len(dataset)} samples")

        # Test individual dataset access
        print("\n2. Testing individual dataset access...")
        for i in range(min(3, len(dataset))):
            try:
                sample = dataset[i]
                print(f"   Sample {i}: status={sample['status']}, keys={list(sample.keys())}")
                if sample is None:
                    print(f"   ⚠️  Sample {i} is None!")
                elif not isinstance(sample, dict):
                    print(f"   ⚠️  Sample {i} is not a dict: {type(sample)}")
            except Exception as e:
                print(f"   ❌ Error accessing sample {i}: {e}")

        # Test DataLoader creation
        print("\n3. Creating DataLoader...")
        dataloader = create_dataloader(
            split_name=split_name,
            target_dir=Path("./test_dataloader"),
            target_sr=16000,
            batch_size=2,
            num_workers=0,  # Use 0 for testing
            save_files=False,
            skip_existing=True,
            test_mode=True,
            test_samples=num_samples,
        )
        print("   ✅ DataLoader created successfully")

        # Test DataLoader iteration
        print("\n4. Testing DataLoader iteration...")
        for batch_idx, batch in enumerate(dataloader):
            print(f"   Batch {batch_idx}:")
            if batch is None:
                print("     ❌ Batch is None!")
            else:
                print(f"     Keys: {list(batch.keys())}")
                print(f"     Batch size: {len(batch.get('idx', []))}")
                print(f"     Status values: {batch.get('status', [])}")

            if batch_idx >= 2:  # Only test first few batches
                break

        print("\n✅ DataLoader test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ DataLoader test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def example_usage_pytorch_components() -> None:
    """Example of how to use the PyTorch dataset and dataloader components independently."""

    # First run the test
    if not test_dataloader_functionality():
        print("DataLoader test failed, skipping examples")
        return

    # Example 1: Using the dataset directly
    print("\n=== Example 1: Using BirdSetResampleDataset directly ===")

    dataset = BirdSetResampleDataset(
        split_name="val",
        target_dir=Path("./resampled_data"),
        target_sr=16000,
        save_files=False,  # Don't save files, just get audio data
        skip_existing=True,
        test_mode=True,
        test_samples=5,
    )

    # Get a single sample
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Sample rate: {sample['sample_rate']}")
    print(f"Status: {sample['status']}")

    # Example 2: Using the dataloader
    print("\n=== Example 2: Using DataLoader ===")

    dataloader = create_dataloader(
        split_name="val",
        target_dir=Path("./resampled_data"),
        target_sr=16000,
        batch_size=2,
        num_workers=0,
        save_files=False,  # Don't save files in this example
        skip_existing=True,
        test_mode=True,
        test_samples=5,
    )

    # Process a few batches
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Batch size: {len(batch['idx'])}")
        print(f"  Status values: {batch['status']}")

        if batch_idx >= 1:  # Just show first 2 batches
            break

    print("\n=== Integration Tips ===")
    print("1. Set save_files=False if you only want audio data for training")
    print("2. Use batch_size > 1 for efficient batch processing")
    print("3. Adjust num_workers based on your system's capabilities")
    print("4. The dataset preserves all metadata from the original BirdSet")
    print("5. Audio data is already resampled to target_sr")
    print("6. Use test_mode=True for debugging with small samples")


if __name__ == "__main__":
    # Uncomment the line below to run the example instead of main processing
    # example_usage_pytorch_components()
    main()
