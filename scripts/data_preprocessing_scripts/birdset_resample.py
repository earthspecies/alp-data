"""Resample all BirdSet files using PyTorch DataLoader for parallel processing"""

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, Any

# Check for required dependencies
try:
    import librosa
except ImportError:
    print("Error: librosa is required but not installed. Install with: pip install librosa")
    exit(1)

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
    from torch.utils.data import DataLoader, Dataset
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

# Suppress warnings from librosa about resampling
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


class BirdSetRewriteDataset(Dataset):
    """Dataset wrapper for resampling BirdSet files"""
    
    def __init__(self, birdset_dataset: BirdSet, target_dir: Path, target_sr: int = 16000):
        """Initialize the resampling dataset.
        
        Parameters
        ----------
        birdset_dataset : BirdSet
            The BirdSet dataset instance
        target_dir : Path
            Target directory for resampled files
        target_sr : int
            Target sample rate
        """
        self.birdset_dataset = birdset_dataset
        self.target_dir = target_dir
        self.target_sr = target_sr
        
        # Create target directory if it doesn't exist
        self.target_dir.mkdir(parents=True, exist_ok=True)
    
    def __len__(self) -> int:
        return len(self.birdset_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample and save the already-resampled audio from BirdSet.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing original path, target path, and error info
        """
        try:
            # Get the original row data to extract the path
            original_row = self.birdset_dataset._data.iloc[idx]
            original_path = original_row["path"]
            
            # Construct source path
            if self.birdset_dataset.data_root:
                source_path = anypath(self.birdset_dataset.data_root) / original_path
            else:
                source_path = anypath(original_path)
            
            # Create target path with same structure but in target directory
            relative_path = Path(original_path)
            target_path = self.target_dir / relative_path.with_suffix('.flac')
            
            # Create parent directories if they don't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file already exists
            if target_path.exists():
                return {
                    "original_path": str(source_path),
                    "target_path": str(target_path),
                    "status": "skipped",
                    "error": None
                }
            
            # Get the already-resampled audio from the BirdSet dataset
            try:
                sample = self.birdset_dataset[idx]
                audio = sample["audio"]
                
                # Write the resampled audio as flac file
                with target_path.open("wb") as f:
                    sf.write(f, audio, self.target_sr, format="FLAC")
                
                return {
                    "original_path": str(source_path),
                    "target_path": str(target_path),
                    "status": "success",
                    "error": None
                }
                
            except Exception as e:
                return {
                    "original_path": str(source_path),
                    "target_path": str(target_path),
                    "status": "error",
                    "error": str(e)
                }
                
        except Exception as e:
            return {
                "original_path": f"index_{idx}",
                "target_path": "",
                "status": "error",
                "error": str(e)
            }


def process_split(
    split_name: str, 
    target_base_dir: Path, 
    target_sr: int = 16000,
    num_workers: int = 4,
    batch_size: int = 1
) -> Dict[str, Any]:
    """Process a single BirdSet split.
    
    Parameters
    ----------
    split_name : str
        Name of the split to process
    target_base_dir : Path
        Base target directory
    target_sr : int
        Target sample rate
    num_workers : int
        Number of worker processes
    batch_size : int
        Batch size for DataLoader
    
    Returns
    -------
    Dict[str, Any]
        Processing statistics
    """
    print(f"Processing split: {split_name}")
    
    # Create target directory for this split
    target_dir = target_base_dir / split_name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize BirdSet dataset with resampling enabled
    try:
        birdset = BirdSet(
            split=split_name,
            sample_rate=target_sr,  # Let BirdSet handle the resampling
            data_root="gs://foundation-model-data/"  # Explicitly set GCS data root
        )
    except Exception as e:
        error_msg = str(e)
        if "google.auth" in error_msg or "GCS" in error_msg or "gs://" in error_msg:
            print(f"Error loading split {split_name}: GCS authentication issue. Make sure you have proper GCS credentials set up.")
            print(f"Original error: {error_msg}")
        else:
            print(f"Error loading split {split_name}: {error_msg}")
        return {
            "split": split_name,
            "status": "error",
            "error": error_msg,
            "processed": 0,
            "errors": 0,
            "skipped": 0
        }
    
    # Create resampling dataset
    resample_dataset = BirdSetRewriteDataset(birdset, target_dir, target_sr)
    
    # Create DataLoader
    dataloader = DataLoader(
        resample_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=lambda x: x  # Don't batch, return list of results
    )
    
    # Process all files
    results = []
    total_files = len(resample_dataset)
    
    print(f"Processing {total_files} files in {split_name}...")
    
    for batch_idx, batch_results in enumerate(dataloader):
        # Since batch_size=1, batch_results is a list with one item
        for result in batch_results:
            results.append(result)
            
            # Print progress every 100 files
            if len(results) % 100 == 0:
                print(f"  Processed {len(results)}/{total_files} files in {split_name}")
    
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
    """Resample all BirdSet files using parallel processing."""
    
    parser = argparse.ArgumentParser(
        description="Resample all BirdSet files using PyTorch DataLoader for parallel processing."
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
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel processing.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for DataLoader (usually 1 for file processing).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        help="Specific splits to process. Use 'all' to process all available splits. If not provided, processes all splits.",
    )
    
    args = parser.parse_args()
    target_dir = anypath(args.target_dir)
    
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
    
    print(f"Processing {len(splits_to_process)} splits: {splits_to_process}")
    print(f"Target directory: {target_dir}")
    print(f"Target sample rate: {args.target_sr}")
    print(f"Number of workers: {args.num_workers}")
    print("Note: This script requires access to Google Cloud Storage (GCS).")
    print("Make sure you have proper GCS authentication set up (gcloud auth or service account).")
    
    # Process each split
    all_results = []
    all_errors = []
    
    for split_name in splits_to_process:
        try:
            result = process_split(
                split_name=split_name,
                target_base_dir=target_dir,
                target_sr=args.target_sr,
                num_workers=args.num_workers,
                batch_size=args.batch_size
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
            "error": result.get("error", "")
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
    print(f"Total files processed: {total_processed}")
    print(f"Total successful: {total_successful}")
    print(f"Total errors: {total_errors}")
    print(f"Total skipped: {total_skipped}")
    print(f"Success rate: {total_successful/(total_processed or 1)*100:.1f}%")


if __name__ == "__main__":
    main()
