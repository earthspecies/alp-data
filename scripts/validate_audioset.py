#!/usr/bin/env python3
"""
Validate AudioSet dataset for missing or corrupt audio files.

This script checks for:
1. Missing audio files
2. Audio files with non-finite values (NaN or Inf)
3. Empty audio files

Usage:
    python validate_audioset.py --split train --version 0.2.0 --sample-rate 32000 --num-workers 8
    python validate_audioset.py --split train --version 0.2.0 --check-originals --num-workers 8
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# Add the parent directory to the path so we can import esp_data
sys.path.insert(0, str(Path(__file__).parent.parent))

from esp_data.io import anypath, read_audio, audio_stereo_to_mono

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Version registry (matching audioset.py)
VERSIONS = {
    "0.1.0": {
        "split_paths": {
            "train": "gs://esp-ml-datasets/audioset/v0.1.0/raw/csv-data/unbalanced_train_segments_processed.csv",
            "train-balanced": "gs://esp-ml-datasets/audioset/v0.1.0/raw/csv-data/balanced_train_segments_processed.csv",
            "validation": "gs://esp-ml-datasets/audioset/v0.1.0/raw/csv-data/eval_segments_processed.csv",
        },
        "data_root": "gs://esp-ml-datasets/audioset/v0.1.0/raw/",
    },
    "0.2.0": {
        "split_paths": {
            "train": "gs://esp-ml-datasets/audioset/v0.2.0/raw/csv-data/unbalanced_train_segments_processed.csv",
            "validation": "gs://esp-ml-datasets/audioset/v0.2.0/raw/csv-data/eval_segments_processed.csv",
            "train-environmental": "gs://esp-ml-datasets/audioset/v0.2.0/raw/csv-data/unbalanced_train_environmental_sounds.csv",
        },
        "data_root": "gs://esp-ml-datasets/audioset/v0.2.0/raw/",
    },
}

# Sample rate to path column mapping (for v0.2.0)
SAMPLE_RATE_PATHS = {
    32000: "32khz_path",
}

ORIGINALS_PATH_COLUMN = "local_path"


@dataclass
class ValidationResult:
    """Result of validating a single audio file."""
    idx: int
    path: str
    status: str  # "ok", "missing", "non_finite", "empty", "short", "error"
    error_message: str | None = None
    nan_count: int = 0
    inf_count: int = 0
    audio_length: int = 0
    sample_rate: int = 0
    expected_length: int = 0


def validate_single_audio(
    idx: int,
    row: dict[str, Any],
    data_root: str,
    path_column: str,
) -> ValidationResult:
    """Validate a single audio file.
    
    Parameters
    ----------
    idx : int
        Index of the sample in the dataset
    row : dict[str, Any]
        Row data containing the path column
    data_root : str
        Root directory for the dataset
    path_column : str
        Column name containing the audio file path
    
    Returns
    -------
    ValidationResult
        Result of the validation
    """
    try:
        # Check if path column exists and has valid value
        if path_column not in row:
            return ValidationResult(
                idx=idx,
                path="",
                status="error",
                error_message=f"Path column '{path_column}' not found in row",
            )
        
        path_value = row[path_column]
        if path_value is None or path_value == "":
            return ValidationResult(
                idx=idx,
                path="",
                status="missing",
                error_message="Path column is empty or None",
            )
        
        if isinstance(path_value, float) and np.isnan(path_value):
            return ValidationResult(
                idx=idx,
                path="",
                status="missing",
                error_message="Path column is NaN",
            )
        
        # Construct full path
        audio_path = anypath(data_root) / str(path_value)
        
        # Read audio - this will fail if file doesn't exist
        try:
            audio, sr = read_audio(audio_path)
        except FileNotFoundError:
            return ValidationResult(
                idx=idx,
                path=str(audio_path),
                status="missing",
                error_message="File does not exist",
            )
        except Exception as read_err:
            # Check if it's a "not found" type error for cloud paths
            err_str = str(read_err).lower()
            if "not found" in err_str or "no such file" in err_str or "404" in err_str:
                return ValidationResult(
                    idx=idx,
                    path=str(audio_path),
                    status="missing",
                    error_message=f"File not found: {read_err}",
                )
            raise
        
        audio = audio.astype(np.float32)
        
        # Check for empty audio before mono conversion
        if len(audio) == 0:
            return ValidationResult(
                idx=idx,
                path=str(audio_path),
                status="empty",
                error_message="Audio file is empty",
                audio_length=0,
                sample_rate=sr,
            )
        
        # Check for short audio (less than 90% of expected 10 seconds)
        expected_length = sr * 10  # 10 seconds is the standard AudioSet duration
        min_expected = int(expected_length * 0.5)  # Allow some tolerance (5 seconds minimum)
        
        raw_length = len(audio) if audio.ndim == 1 else audio.shape[0]
        if raw_length < min_expected:
            return ValidationResult(
                idx=idx,
                path=str(audio_path),
                status="short",
                error_message=f"Audio too short: {raw_length} samples (expected ~{expected_length} at {sr}Hz)",
                audio_length=raw_length,
                sample_rate=sr,
                expected_length=expected_length,
            )
        
        audio = audio_stereo_to_mono(audio, mono_method="average")
        
        # Check for non-finite values
        nan_mask = np.isnan(audio)
        inf_mask = np.isinf(audio)
        nan_count = int(np.sum(nan_mask))
        inf_count = int(np.sum(inf_mask))
        
        if nan_count > 0 or inf_count > 0:
            return ValidationResult(
                idx=idx,
                path=str(audio_path),
                status="non_finite",
                error_message=f"Audio contains {nan_count} NaN and {inf_count} Inf values",
                nan_count=nan_count,
                inf_count=inf_count,
                audio_length=len(audio),
                sample_rate=sr,
            )
        
        return ValidationResult(
            idx=idx,
            path=str(audio_path),
            status="ok",
            audio_length=len(audio),
            sample_rate=sr,
        )
        
    except Exception as e:
        return ValidationResult(
            idx=idx,
            path=str(row.get(path_column, "")),
            status="error",
            error_message=str(e),
        )


def validate_batch(
    batch_data: list[tuple[int, dict[str, Any]]],
    data_root: str,
    path_column: str,
) -> list[ValidationResult]:
    """Validate a batch of audio files.
    
    Parameters
    ----------
    batch_data : list[tuple[int, dict]]
        List of (index, row) tuples
    data_root : str
        Root directory for the dataset
    path_column : str
        Column name containing the audio file path
    
    Returns
    -------
    list[ValidationResult]
        List of validation results
    """
    results = []
    for idx, row in batch_data:
        result = validate_single_audio(idx, row, data_root, path_column)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate AudioSet dataset for missing or corrupt audio files"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to validate (default: train)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="0.2.0",
        help="Dataset version (default: 0.2.0)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Sample rate to check (e.g., 32000 for pre-resampled audio in v0.2.0). "
             "If not specified, checks original files.",
    )
    parser.add_argument(
        "--check-originals",
        action="store_true",
        help="Check original audio files (local_path column) instead of pre-resampled",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for parallel processing (default: 100)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index for validation (default: 0)",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index for validation (default: None = all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (default: audioset_validation_{version}_{split}_{path_type}.json)",
    )
    parser.add_argument(
        "--check-single-idx",
        type=int,
        default=None,
        help="Check a single index and print detailed info",
    )
    
    args = parser.parse_args()
    
    # Validate version
    if args.version not in VERSIONS:
        logger.error(f"Version '{args.version}' not found. Available: {list(VERSIONS.keys())}")
        sys.exit(1)
    
    version_config = VERSIONS[args.version]
    
    # Validate split
    if args.split not in version_config["split_paths"]:
        logger.error(
            f"Split '{args.split}' not found for version {args.version}. "
            f"Available: {list(version_config['split_paths'].keys())}"
        )
        sys.exit(1)
    
    data_root = version_config["data_root"]
    csv_path = version_config["split_paths"][args.split]
    
    # Determine which path column to use
    if args.check_originals:
        path_column = ORIGINALS_PATH_COLUMN
        path_type = "originals"
    elif args.sample_rate is not None:
        if args.sample_rate not in SAMPLE_RATE_PATHS:
            logger.error(
                f"Sample rate {args.sample_rate} not available. "
                f"Available pre-resampled rates: {list(SAMPLE_RATE_PATHS.keys())}"
            )
            sys.exit(1)
        path_column = SAMPLE_RATE_PATHS[args.sample_rate]
        path_type = f"{args.sample_rate}hz"
    else:
        path_column = ORIGINALS_PATH_COLUMN
        path_type = "originals"
    
    logger.info(f"Loading dataset metadata from: {csv_path}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Path column: {path_column}")
    
    # Load the CSV
    df = pl.read_csv(csv_path)
    total_samples = len(df)
    logger.info(f"Total samples in dataset: {total_samples}")
    logger.info(f"Columns: {df.columns}")
    
    # Check if path column exists
    if path_column not in df.columns:
        logger.error(f"Path column '{path_column}' not found in dataset columns: {df.columns}")
        sys.exit(1)
    
    # Handle single index check
    if args.check_single_idx is not None:
        idx = args.check_single_idx
        if idx >= total_samples:
            logger.error(f"Index {idx} out of bounds (total samples: {total_samples})")
            sys.exit(1)
        
        row = df.row(idx, named=True)
        logger.info(f"\n=== Checking single index {idx} ===")
        logger.info(f"Row data: {row}")
        
        result = validate_single_audio(idx, row, data_root, path_column)
        logger.info(f"\nValidation result:")
        logger.info(f"  Status: {result.status}")
        logger.info(f"  Path: {result.path}")
        if result.error_message:
            logger.info(f"  Error: {result.error_message}")
        if result.status == "non_finite":
            logger.info(f"  NaN count: {result.nan_count}")
            logger.info(f"  Inf count: {result.inf_count}")
        logger.info(f"  Audio length: {result.audio_length}")
        return
    
    # Determine range to validate
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else total_samples
    end_idx = min(end_idx, total_samples)
    
    logger.info(f"Validating indices {start_idx} to {end_idx - 1}")
    
    # Prepare batches
    rows_to_validate = [
        (i, df.row(i, named=True))
        for i in range(start_idx, end_idx)
    ]
    
    batches = [
        rows_to_validate[i:i + args.batch_size]
        for i in range(0, len(rows_to_validate), args.batch_size)
    ]
    
    logger.info(f"Processing {len(batches)} batches with {args.num_workers} workers")
    
    # Collect results
    all_results: list[ValidationResult] = []
    issues: list[ValidationResult] = []
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(validate_batch, batch, data_root, path_column): batch_idx
            for batch_idx, batch in enumerate(batches)
        }
        
        completed = 0
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                
                # Track issues
                for result in batch_results:
                    if result.status != "ok":
                        issues.append(result)
                        logger.warning(
                            f"Issue at idx {result.idx}: {result.status} - {result.error_message}"
                        )
                
            except Exception as e:
                logger.error(f"Batch {batch_idx} failed: {e}")
            
            completed += 1
            if completed % 10 == 0 or completed == len(batches):
                progress = completed / len(batches) * 100
                logger.info(
                    f"Progress: {completed}/{len(batches)} batches ({progress:.1f}%), "
                    f"Issues found: {len(issues)}"
                )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples validated: {len(all_results)}")
    logger.info(f"OK: {sum(1 for r in all_results if r.status == 'ok')}")
    logger.info(f"Missing: {sum(1 for r in all_results if r.status == 'missing')}")
    logger.info(f"Short/truncated: {sum(1 for r in all_results if r.status == 'short')}")
    logger.info(f"Non-finite (NaN/Inf): {sum(1 for r in all_results if r.status == 'non_finite')}")
    logger.info(f"Empty: {sum(1 for r in all_results if r.status == 'empty')}")
    logger.info(f"Errors: {sum(1 for r in all_results if r.status == 'error')}")
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = f"audioset_validation_{args.version}_{args.split}_{path_type}.json"
    
    # Prepare output data
    output_data = {
        "version": args.version,
        "split": args.split,
        "path_column": path_column,
        "path_type": path_type,
        "data_root": data_root,
        "total_validated": len(all_results),
        "summary": {
            "ok": sum(1 for r in all_results if r.status == "ok"),
            "missing": sum(1 for r in all_results if r.status == "missing"),
            "short": sum(1 for r in all_results if r.status == "short"),
            "non_finite": sum(1 for r in all_results if r.status == "non_finite"),
            "empty": sum(1 for r in all_results if r.status == "empty"),
            "error": sum(1 for r in all_results if r.status == "error"),
        },
        "issues": [
            {
                "idx": r.idx,
                "path": r.path,
                "status": r.status,
                "error_message": r.error_message,
                "nan_count": r.nan_count,
                "inf_count": r.inf_count,
                "audio_length": r.audio_length,
                "sample_rate": r.sample_rate,
                "expected_length": r.expected_length,
            }
            for r in issues
        ],
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    # Print first few issues for quick reference
    if issues:
        logger.info("\n" + "=" * 60)
        logger.info("FIRST 10 ISSUES:")
        logger.info("=" * 60)
        for issue in issues[:10]:
            logger.info(f"  idx={issue.idx}, status={issue.status}, path={issue.path}")
            if issue.error_message:
                logger.info(f"    -> {issue.error_message}")


if __name__ == "__main__":
    main()

