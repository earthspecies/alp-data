"""Resample all BirdSet files and save them to a target directory"""

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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


def should_skip_resample(target_path: Path) -> bool:
    """Check if the target file already exists and should be skipped."""
    return target_path.exists()


def process_single_file(
    idx: int, 
    split_name: str, 
    target_dir: Path, 
    target_sr: int,
    total_files: int,
    progress_counter: list
) -> Dict[str, Any]:
    """Process a single file from the BirdSet dataset.
    
    Parameters
    ----------
    idx : int
        Index of the file to process
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
    Dict[str, Any]
        Processing result for this file
    """
    try:
        # Create a new BirdSet instance for this worker (thread-safe)
        birdset = BirdSet(
            split=split_name,
            sample_rate=target_sr,
            data_root="gs://foundation-model-data/",
        )
        
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
        # Remove any leading path separators and convert to Path
        relative_path = Path(file_path)
        if relative_path.is_absolute():
            # If it's absolute, make it relative by removing the root
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
                "error": None
            }
        else:
            # Save the resampled audio
            try:
                sf.write(target_path, audio_data, target_sr, format="FLAC")
                
                result = {
                    "idx": idx,
                    "original_path": file_path,
                    "target_path": str(target_path),
                    "status": "success",
                    "error": None
                }
            except Exception as save_error:
                result = {
                    "idx": idx,
                    "original_path": file_path,
                    "target_path": str(target_path),
                    "status": "error",
                    "error": f"Save error: {save_error}"
                }
                
    except Exception as load_error:
        result = {
            "idx": idx,
            "original_path": "unknown",
            "target_path": "unknown",
            "status": "error",
            "error": f"Load error: {load_error}"
        }
    
    # Update progress counter in a thread-safe manner
    with progress_lock:
        progress_counter[0] += 1
        if progress_counter[0] % 100 == 0:
            print(f"  Processed {progress_counter[0]}/{total_files} files in {split_name}")
    
    return result


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
        file_path = self.birdset[idx]["local_path"] if "local_path" in self.birdset[idx] else self.birdset[idx]["path"]
        
        # Create target file path - preserve the original directory structure
        relative_path = Path(file_path)
        if relative_path.is_absolute():
            # If it's absolute, make it relative by removing the root
            relative_path = Path(*relative_path.parts[1:])
        
        target_path = self.target_dir / relative_path
        
        if self.skip_existing and target_path.exists():
            return {
                "idx": idx,
                "audio": None,
                "original_path": file_path,
                "target_path": str(target_path),
                "status": "skipped",
                "error": None
            }
        
        # Create parent directories if they don't exist
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get sample with resampled audio
            sample = self.birdset[idx]
            audio_data = sample["audio"]

            # Initialize result dictionary
            result = {
                "idx": idx,
                "audio": audio_data,
                "original_path": file_path,
                "target_path": str(target_path),
                "sample_rate": self.target_sr,
                "status": "success",
                "error": None,
                "metadata": {k: v for k, v in sample.items() if k not in ["audio", "path", "local_path"]}
            }
            
            # Optionally save the file
            if self.save_files:
                # Check if file already exists and should be skipped
                if self.skip_existing and target_path.exists():
                    result["status"] = "skipped"
                else:
                    try:
                        sf.write(target_path, audio_data, self.target_sr, format="FLAC")
                        result["status"] = "saved"
                    except Exception as save_error:
                        result["status"] = "save_error"
                        result["error"] = f"Save error: {save_error}"
            
            return result
            
        except Exception as load_error:
            return {
                "idx": idx,
                "audio": None,
                "original_path": "unknown",
                "target_path": "unknown",
                "sample_rate": self.target_sr,
                "status": "error",
                "error": f"Load error: {load_error}",
                "metadata": {}
            }


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
        collate_fn=None  # Use default collate function
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
        Number of worker processes
    save_files : bool
        Whether to save files to disk
        
    Returns
    -------
    Dict[str, Any]
        Processing statistics
    """
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
    
    total_files = len(dataloader.dataset)
    print(f"Processing {total_files} files in {split_name}...")
    
    # Process batches
    results = []
    status_counters = {"success": 0, "saved": 0, "skipped": 0, "error": 0, "save_error": 0}
    
    for batch_idx, batch in enumerate(dataloader):
        # Handle both single items and batches
        if batch_size == 1:
            # Single item - convert to list for uniform processing
            batch_results = [batch]
        else:
            # Multiple items in batch
            batch_results = []
            for i in range(len(batch["idx"])):
                item = {key: value[i] if isinstance(value, (list, torch.Tensor)) else value 
                       for key, value in batch.items()}
                batch_results.append(item)
        
        # Process each item in the batch
        for item in batch_results:
            # Convert tensor values back to python types
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
    successful = status_counters["success"] + status_counters["saved"]
    errors = status_counters["error"] + status_counters["save_error"]
    skipped = status_counters["skipped"]
    
    print(f"Split {split_name} completed:")
    print(f"  Success: {successful}")
    print(f"  Errors: {errors}")
    print(f"  Skipped: {skipped}")
    
    return {
        "split": split_name,
        "status": "completed",
        "processed": len(results),
        "successful": successful,
        "errors": errors,
        "skipped": skipped,
        "results": results
    }


def process_split(
    split_name: str, 
    target_base_dir: Path, 
    target_sr: int = 16000,
    max_workers: int = 4
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
    
    Returns
    -------
    Dict[str, Any]
        Processing statistics
    """
    print(f"Processing split: {split_name} with {max_workers} workers")
    
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
        print(f"Sample: {sample}")
    
    # Initialize progress counter
    progress_counter = [0]
    status_counters = {"success": 0, "skipped": 0, "error": 0}
    status_lock = threading.Lock()
    
    # Process files in parallel
    results = []
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
                    "error": f"Worker exception: {exc}"
                })
                
                # Update error counter
                with status_lock:
                    status_counters["error"] += 1
    
    # Calculate statistics
    successful = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    
    print(f"Split {split_name} completed:")
    print(f"  Success: {successful}")
    print(f"  Errors: {errors}")
    print(f"  Skipped: {skipped}")
    
    return {
        "split": split_name,
        "status": "completed",
        "processed": len(results),
        "successful": successful,
        "errors": errors,
        "skipped": skipped,
        "results": results
    }


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
        help="Use PyTorch DataLoader for processing instead of ThreadPoolExecutor.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for DataLoader processing (only used with --use-dataloader).",
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
    
    processing_method = "PyTorch DataLoader" if args.use_dataloader else "ThreadPoolExecutor"
    print(f"Processing {len(splits_to_process)} splits: {splits_to_process}")
    print(f"Processing method: {processing_method}")
    print(f"Target directory: {target_dir}")
    print(f"Target sample rate: {args.target_sr}")
    
    if args.use_dataloader:
        print(f"Batch size: {args.batch_size}")
        print(f"Number of workers: {args.workers}")
    else:
        print(f"Number of worker threads: {args.workers}")
    
    print("Note: This script requires access to Google Cloud Storage (GCS).")
    print("Make sure you have proper GCS authentication set up (gcloud auth or service account).")
    
    # Process each split
    all_results = []
    all_errors = []
    
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
                    max_workers=args.workers
                )
            
            all_results.append(result)
            
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
        summary_data.append({
            "split": result["split"],
            "status": result["status"],
            "processed": result.get("processed", 0),
            "successful": result.get("successful", 0),
            "errors": result.get("errors", 0),
            "skipped": result.get("skipped", 0),
            "error": result.get("error", ""),
            "processing_method": processing_method
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = target_dir / "resample_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")
    
    # Print final summary
    total_processed = sum(r.get("processed", 0) for r in all_results)
    total_successful = sum(r.get("successful", 0) for r in all_results)
    total_errors = sum(r.get("errors", 0) for r in all_results)
    total_skipped = sum(r.get("skipped", 0) for r in all_results)
    
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Processing method: {processing_method}")
    print(f"Total files processed: {total_processed}")
    print(f"Total successful: {total_successful}")
    print(f"Total errors: {total_errors}")
    print(f"Total skipped: {total_skipped}")
    print(f"Success rate: {total_successful/(total_processed or 1)*100:.1f}%")


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
