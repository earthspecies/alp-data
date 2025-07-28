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
import os
import warnings
import time
import functools
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass, asdict

try:
    import numpy as np
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
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("Error: torch is required but not installed. Install with: pip install torch")
    exit(1)

try:
    from esp_data.datasets import BirdSet
    from esp_data.io import anypath, audio_stereo_to_mono, read_audio
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


def benchmark_timing(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Add timing info to result if it's a dict
        if isinstance(result, dict):
            result['execution_time'] = end_time - start_time
            result['start_time'] = start_time
            result['end_time'] = end_time
        
        return result
    return wrapper


def get_thread_local_birdset(split_name: str, target_sr: int) -> BirdSet:
    """Get or create a thread-local BirdSet instance for better performance."""
    if not hasattr(thread_local_data, 'birdset') or \
       thread_local_data.split_name != split_name or \
       thread_local_data.target_sr != target_sr:
        
        thread_local_data.birdset = BirdSet(
            split=split_name,
            sample_rate=target_sr,
            data_root="gs://foundation-model-data/",
        )
        thread_local_data.split_name = split_name
        thread_local_data.target_sr = target_sr
    
    return thread_local_data.birdset


def should_skip_resample(target_path: Path) -> bool:
    """Check if the target file already exists and should be skipped."""
    return target_path.exists()


def process_file_batch(
    indices: List[int], 
    split_name: str, 
    target_dir: Path, 
    target_sr: int,
    total_files: int,
    progress_counter: list
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
                    raise KeyError(f"Neither 'path' nor 'local_path' found in sample keys: {list(sample.keys())}")
                
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
                        "processing_time": time.time() - file_start_time
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
                            "save_time": save_time
                        }
                    except Exception as save_error:
                        result = {
                            "idx": idx,
                            "original_path": file_path,
                            "target_path": str(target_path),
                            "status": "error",
                            "error": f"Save error: {save_error}",
                            "processing_time": time.time() - file_start_time
                        }
                        
            except Exception as load_error:
                result = {
                    "idx": idx,
                    "original_path": "unknown",
                    "target_path": "unknown",
                    "status": "error",
                    "error": f"Load error: {load_error}",
                    "processing_time": time.time() - file_start_time
                }
            
            results.append(result)
    
    except Exception as batch_error:
        # If there's an error with the entire batch, create error results for all indices
        for idx in indices:
            results.append({
                "idx": idx,
                "original_path": "unknown",
                "target_path": "unknown",
                "status": "error",
                "error": f"Batch error: {batch_error}",
                "processing_time": time.time() - batch_start_time
            })
    
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
    progress_counter: list
) -> Dict[str, Any]:
    """Process a single file from the BirdSet dataset.
    
    This is kept for backward compatibility, but internally uses batch processing.
    """
    return process_file_batch([idx], split_name, target_dir, target_sr, total_files, progress_counter)[0]


class BirdSetResampleDataset(Dataset):
    """PyTorch Dataset wrapper for BirdSet with resampling and saving functionality."""
    
    def __init__(
        self,
        split_name: str,
        target_dir: Path,
        target_sr: int = 16000,
        save_files: bool = True,
        skip_existing: bool = True
    ):
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
        """
        setup_start = time.time()
        
        self.split_name = split_name
        self.target_dir = Path(target_dir)
        self.target_sr = target_sr
        self.save_files = save_files
        self.skip_existing = skip_existing
        
        # Initialize BirdSet dataset
        self.birdset = BirdSet(
            split=split_name,
            sample_rate=target_sr,
            data_root="gs://foundation-model-data/",
        )
        
        # Create target directory
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        total_files = len(self.birdset)
        print(f"Processing {total_files} files in {split_name}...")
        
        # Debug: print available keys for the first sample
        if total_files > 0:
            sample = self.birdset[0]
            print(f"Available keys in sample: {list(sample.keys())}")
            # Only print a subset of the sample to avoid too much output
            sample_subset = {k: v for k, v in sample.items() if k != "audio"}
            sample_subset["audio"] = f"<array shape={sample['audio'].shape}>"
            print(f"Sample: {sample_subset}")
        
        self.setup_time = time.time() - setup_start
        print(f"Dataset setup completed in {self.setup_time:.2f}s")
            
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.birdset)
    
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
        """
        start_time = time.time()
        
        try:
            # Get sample first to avoid multiple dataset accesses
            load_start = time.time()
            sample = self.birdset[idx]
            load_time = time.time() - load_start
            
            # Try to get the path - check both possible key names
            if "local_path" in sample:
                file_path = sample["local_path"]
            elif "path" in sample:
                file_path = sample["path"]
            else:
                raise KeyError(f"Neither 'path' nor 'local_path' found in sample keys: {list(sample.keys())}")
            
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
                    "save_time": 0.0
                }
            
            # Create parent directories if they don't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Extract audio data from the already-fetched sample
            audio_data = sample["audio"]

            # Initialize result dictionary
            result = {
                "idx": idx,
                "original_path": file_path,
                "target_path": str(target_path),
                "sample_rate": self.target_sr,
                "status": "success",
                "error": None,
                "metadata": {k: v for k, v in sample.items() if k not in ["audio", "path", "local_path"]},
                "load_time": load_time,
                "save_time": 0.0
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
            return result
            
        except Exception as load_error:
            return {
                "idx": idx,
                "original_path": "unknown",
                "target_path": "unknown",
                "sample_rate": self.target_sr,
                "status": "error",
                "error": f"Load error: {load_error}",
                "metadata": {},
                "processing_time": time.time() - start_time,
                "load_time": 0.0,
                "save_time": 0.0
            }


def custom_collate_fn(batch):
    """Custom collate function for batching dataset results.
    
    Parameters
    ----------
    batch : List[Dict[str, Any]]
        List of samples from the dataset
        
    Returns
    -------
    Dict[str, Any]
        Batched data
    """
    # Initialize the result dictionary
    batched = {}
    
    # Get all keys from the first sample
    if batch:
        for key in batch[0].keys():
            if key in ["idx", "sample_rate"]:
                # Convert to tensor for numeric fields
                batched[key] = torch.tensor([sample[key] for sample in batch])
            else:
                # Keep as lists for string/mixed types
                batched[key] = [sample[key] for sample in batch]
    
    return batched


def create_dataloader(
    split_name: str,
    target_dir: Path,
    target_sr: int = 16000,
    batch_size: int = 1,
    num_workers: int = 4,
    save_files: bool = True,
    skip_existing: bool = True
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
        skip_existing=skip_existing
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep original order for resampling
        num_workers=num_workers,
        collate_fn=custom_collate_fn if batch_size > 1 else None
    )


def process_split_with_dataloader(
    split_name: str,
    target_base_dir: Path,
    target_sr: int = 16000,
    batch_size: int = 1,
    num_workers: int = 4,
    save_files: bool = True
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
        
    Returns
    -------
    Dict[str, Any]
        Processing statistics
    """
    total_start_time = time.time()
    setup_start_time = time.time()
    
    # Force num_workers=0 due to GCS fork-safety issues
    if num_workers > 0:
        print(f"⚠️  Warning: Setting num_workers=0 due to GCS fork-safety limitations.")
        print(f"   The gcsfs library used for Google Cloud Storage is not fork-safe.")
        print(f"   DataLoader will use single-process mode for compatibility.")
        num_workers = 0
    
    print(f"Processing split: {split_name} with DataLoader (batch_size={batch_size}, num_workers={num_workers})")
    
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
        skip_existing=True
    )
    
    setup_time = time.time() - setup_start_time
    
    total_files = len(dataloader.dataset)
    print(f"Processing {total_files} files in {split_name}...")
    print(f"Setup completed in {setup_time:.2f}s")
    
    # Process batches
    processing_start_time = time.time()
    results = []
    status_counters = {"success": 0, "saved": 0, "skipped": 0, "error": 0, "save_error": 0}
    
    for batch_idx, batch in enumerate(dataloader):
        # Handle both single items and batches
        if batch_size == 1:
            # Single item - convert to list for uniform processing
            batch_results = [batch]
        else:
            # Multiple items in batch - extract individual items
            batch_results = []
            batch_size_actual = len(batch["idx"])
            for i in range(batch_size_actual):
                item = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        item[key] = value[i].item()
                    elif isinstance(value, list):
                        item[key] = value[i]
                    else:
                        item[key] = value
                batch_results.append(item)
        
        # Process each item in the batch
        for item in batch_results:
            # Convert any remaining tensor values back to python types
            result = {}
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        result[key] = value.item()
                    else:
                        result[key] = value.numpy()
                else:
                    result[key] = value
            
            results.append(result)
            status_counters[result["status"]] += 1
        
        # Print progress every 100 files
        if (batch_idx + 1) * batch_size % 100 == 0:
            total_processed = len(results)
            print(f"  Progress {total_processed}/{total_files}: "
                  f"✅ {status_counters['saved']} saved, "
                  f"⏭️  {status_counters['skipped']} skipped, "
                  f"❌ {status_counters['error'] + status_counters['save_error']} errors")
    
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
            cleanup_time=cleanup_time
        ),
        "timing": {
            "total_time": total_time,
            "setup_time": setup_time,
            "processing_time": processing_time,
            "cleanup_time": cleanup_time,
            "avg_time_per_file": avg_time_per_file,
            "avg_individual_time": avg_individual_time,
            "files_per_second": files_per_second
        }
    }


def process_split(
    split_name: str, 
    target_base_dir: Path, 
    target_sr: int = 16000,
    max_workers: int = 4,
    batch_size: int = 1
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
        batches = []
        for i in range(0, total_files, batch_size):
            batch_indices = list(range(i, min(i + batch_size, total_files)))
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
                    progress_counter
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
                            print(f"  Progress {total_completed}/{total_files}: "
                                  f"✅ {status_counters['success']} written, "
                                  f"⏭️  {status_counters['skipped']} skipped, "
                                  f"❌ {status_counters['error']} errors")
                            
                except Exception as exc:
                    batch_indices = future_to_batch[future]
                    print(f'Batch {batch_indices} generated an exception: {exc}')
                    for idx in batch_indices:
                        results.append({
                            "idx": idx,
                            "original_path": "unknown",
                            "target_path": "unknown",
                            "status": "error",
                            "error": f"Batch exception: {exc}",
                            "processing_time": 0.0
                        })
                    
                    # Update error counter
                    with status_lock:
                        status_counters["error"] += len(batch_indices)
    else:
        # Single file processing (original method)
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
                    progress_counter
                ): idx 
                for idx in range(total_files)
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
                            print(f"  Progress {total_completed}/{total_files}: "
                                  f"✅ {status_counters['success']} written, "
                                  f"⏭️  {status_counters['skipped']} skipped, "
                                  f"❌ {status_counters['error']} errors")
                            
                except Exception as exc:
                    idx = future_to_idx[future]
                    print(f'File {idx} generated an exception: {exc}')
                    results.append({
                        "idx": idx,
                        "original_path": "unknown",
                        "target_path": "unknown",
                        "status": "error",
                        "error": f"Worker exception: {exc}",
                        "processing_time": 0.0
                    })
                    
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
            cleanup_time=cleanup_time
        ),
        "timing": {
            "total_time": total_time,
            "setup_time": setup_time,
            "processing_time": processing_time,
            "cleanup_time": cleanup_time,
            "avg_time_per_file": avg_time_per_file,
            "avg_individual_time": avg_individual_time,
            "avg_save_time": avg_save_time,
            "files_per_second": files_per_second
        }
    }


def run_benchmark_comparison(
    splits_to_test: List[str],
    target_dir: Path,
    target_sr: int = 16000,
    max_files_per_split: Optional[int] = None,
    worker_counts: List[int] = [1, 2, 4, 8],
    batch_sizes: List[int] = [1, 5, 10]
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
    worker_counts : List[int]
        List of worker counts to test
    batch_sizes : List[int]
        List of batch sizes to test for ThreadPoolExecutor
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive benchmark results
    """
    print("="*60)
    print("COMPREHENSIVE BENCHMARK COMPARISON")
    print("="*60)
    
    benchmark_results = []
    test_dir = target_dir / "benchmark_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    for split in splits_to_test:
        print(f"\n🧪 Testing split: {split}")
        
        # Test DataLoader method (single configuration due to GCS limitations)
        print(f"\n--- Testing DataLoader Method ---")
        try:
            result_dataloader = process_split_with_dataloader(
                split_name=split,
                target_base_dir=test_dir / f"dataloader_{split}",
                target_sr=target_sr,
                batch_size=1,  # Fixed due to GCS limitations
                num_workers=0,  # Fixed due to GCS limitations
                save_files=True
            )
            benchmark_results.append(result_dataloader["benchmark"])
            print(f"✅ DataLoader completed: {result_dataloader['timing']['files_per_second']:.2f} files/sec")
        except Exception as e:
            print(f"❌ DataLoader failed: {e}")
        
        # Test ThreadPoolExecutor method with different configurations
        print(f"\n--- Testing ThreadPoolExecutor Method ---")
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
                        batch_size=batch_size
                    )
                    benchmark_results.append(result_threads["benchmark"])
                    print(f"✅ ThreadPool {config_name}: {result_threads['timing']['files_per_second']:.2f} files/sec")
                except Exception as e:
                    print(f"❌ ThreadPool {config_name} failed: {e}")
    
    # Analyze results
    print("\n" + "="*60)
    print("BENCHMARK ANALYSIS")
    print("="*60)
    
    if benchmark_results:
        # Convert to DataFrame for analysis
        df = pd.DataFrame([asdict(br) for br in benchmark_results])
        
        # Group by method for comparison
        method_groups = df.groupby('method')
        
        print("\n📊 Performance Summary (files per second):")
        for method, group in method_groups:
            avg_fps = group['files_per_second'].mean()
            max_fps = group['files_per_second'].max()
            min_fps = group['files_per_second'].min()
            print(f"{method:30s}: avg={avg_fps:6.2f}, max={max_fps:6.2f}, min={min_fps:6.2f}")
        
        # Find best configuration
        best_result = df.loc[df['files_per_second'].idxmax()]
        print(f"\n🏆 Best Configuration:")
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
            "benchmark_file": str(benchmark_file)
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
        help="Specific splits to process. Use 'all' to process all available splits. If not provided, processes all splits.",
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
        help="Batch size for processing. For DataLoader: batch size for DataLoader. For ThreadPoolExecutor: files per batch.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run comprehensive benchmark comparison between methods with different configurations.",
    )
    parser.add_argument(
        "--benchmark-splits",
        type=str,
        nargs="+",
        default=["val"],
        help="Splits to use for benchmarking (default: val). Use smaller splits for faster benchmarking.",
    )
    
    args = parser.parse_args()
    target_dir = anypath(args.target_dir)
    
    # Validate number of workers
    if args.workers <= 0:
        print("Error: Number of workers must be greater than 0")
        return
    
    # Get all available splits
    all_splits = BirdSet.info.split_paths.keys()
    
    if args.splits:
        # Check if 'all' is specified
        if 'all' in args.splits:
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
        
        benchmark_results = run_benchmark_comparison(
            splits_to_test=args.benchmark_splits,
            target_dir=target_dir,
            target_sr=args.target_sr,
            max_files_per_split=None,
            worker_counts=[1, 2, 4, 8] if args.workers >= 8 else [1, 2, args.workers],
            batch_sizes=[1, 5, 10, 20] if args.batch_size <= 10 else [1, args.batch_size]
        )
        
        return  # Exit after benchmarking
    
    processing_method = "PyTorch DataLoader" if args.use_dataloader else "ThreadPoolExecutor"
    print(f"Processing {len(splits_to_process)} splits: {splits_to_process}")
    print(f"Processing method: {processing_method}")
    
    if args.use_dataloader:
        print("ℹ️  Note: DataLoader mode uses single-process execution due to GCS fork-safety limitations.")
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
                    save_files=True
                )
            else:
                result = process_split(
                    split_name=split_name,
                    target_base_dir=target_dir,
                    target_sr=args.target_sr,
                    max_workers=args.workers,
                    batch_size=args.batch_size
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
            all_results.append({
                "split": split_name,
                "status": "error",
                "error": str(e),
                "processed": 0,
                "errors": 0,
                "skipped": 0
            })
    
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
            "processing_method": processing_method
        }
        
        # Add timing information if available
        if "timing" in result:
            timing = result["timing"]
            base_data.update({
                "total_time": timing.get("total_time", 0),
                "setup_time": timing.get("setup_time", 0),
                "processing_time": timing.get("processing_time", 0),
                "files_per_second": timing.get("files_per_second", 0),
                "avg_time_per_file": timing.get("avg_time_per_file", 0),
                "avg_save_time": timing.get("avg_save_time", 0)
            })
        
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
    
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Processing method: {processing_method}")
    print(f"Total files processed: {total_processed}")
    print(f"Total successful: {total_successful}")
    print(f"Total errors: {total_errors}")
    print(f"Total skipped: {total_skipped}")
    print(f"Success rate: {total_successful/(total_processed or 1)*100:.1f}%")
    
    if total_time > 0:
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Overall throughput: {overall_fps:.2f} files/second")
        
    # Print performance summary if benchmarks available
    if all_benchmarks:
        print(f"\n📊 PERFORMANCE BREAKDOWN:")
        for benchmark in all_benchmarks:
            print(f"  {benchmark.split} ({benchmark.method}):")
            print(f"    Files/sec: {benchmark.files_per_second:.2f}")
            print(f"    Avg time: {benchmark.avg_time_per_file:.3f}s per file")
            print(f"    Setup: {benchmark.setup_time:.2f}s")
            print(f"    Workers: {benchmark.workers}, Batch: {benchmark.batch_size}")
    
    print(f"\n📄 Results saved to:")
    print(f"  Summary: {summary_file}")
    if all_benchmarks:
        print(f"  Benchmarks: {benchmark_file}")
    if all_errors:
        print(f"  Errors: {error_file}")


def example_usage_pytorch_components():
    """Example of how to use the PyTorch dataset and dataloader components independently."""
    
    # Example 1: Using the dataset directly
    print("=== Example 1: Using BirdSetResampleDataset directly ===")
    
    dataset = BirdSetResampleDataset(
        split_name="train", 
        target_dir=Path("./resampled_data"),
        target_sr=16000,
        save_files=False,  # Don't save files, just get audio data
        skip_existing=True
    )
    
    # Get a single sample
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Audio shape: {sample['audio'].shape}")
    print(f"Sample rate: {sample['sample_rate']}")
    print(f"Status: {sample['status']}")
    
    # Example 2: Using the dataloader
    print("\n=== Example 2: Using DataLoader ===")
    
    dataloader = create_dataloader(
        split_name="train",
        target_dir=Path("./resampled_data"),
        target_sr=16000,
        batch_size=4,
        num_workers=2,
        save_files=False,  # Don't save files in this example
        skip_existing=True
    )
    
    # Process a few batches
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Batch size: {len(batch['idx'])}")
        print(f"  Audio shapes: {[audio.shape for audio in batch['audio']]}")
        
        if batch_idx >= 2:  # Just show first 3 batches
            break
    
    print("\n=== Integration Tips ===")
    print("1. Set save_files=False if you only want audio data for training")
    print("2. Use batch_size > 1 for efficient batch processing")
    print("3. Adjust num_workers based on your system's capabilities")
    print("4. The dataset preserves all metadata from the original BirdSet")
    print("5. Audio data is already resampled to target_sr")


if __name__ == "__main__":
    # Uncomment the line below to run the example instead of main processing
    # example_usage_pytorch_components()
    main()
