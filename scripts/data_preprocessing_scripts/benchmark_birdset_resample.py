#!/usr/bin/env python3
"""
Quick benchmark script for BirdSet resampling performance comparison.

This script runs a comprehensive benchmark of different processing methods
and configurations to help you choose the optimal settings for your system.

Usage:
    python benchmark_birdset_resample.py --target_dir /path/to/test/output
    
Example:
    python benchmark_birdset_resample.py --target_dir ./benchmark_test --split val --max_files 100
"""

import argparse
import sys
from pathlib import Path

# Add the current directory to path so we can import our script
sys.path.insert(0, str(Path(__file__).parent))

from birdset_resample import run_benchmark_comparison, anypath, BirdSet


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BirdSet resampling performance comparison"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Target directory for benchmark test outputs"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Split to use for benchmarking (default: val). Use smaller splits for faster testing."
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=16000,
        help="Target sample rate for resampling (default: 16000)"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process for testing (default: all files in split)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick benchmark with fewer configurations"
    )
    
    args = parser.parse_args()
    
    target_dir = anypath(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate split
    all_splits = list(BirdSet.info.split_paths.keys())
    if args.split not in all_splits:
        print(f"Error: Invalid split '{args.split}'")
        print(f"Available splits: {all_splits}")
        return
    
    print("🧪 BirdSet Resampling Performance Benchmark")
    print("=" * 50)
    print(f"Split: {args.split}")
    print(f"Target directory: {target_dir}")
    print(f"Target sample rate: {args.target_sr}")
    if args.max_files:
        print(f"Max files: {args.max_files}")
    
    # Configure benchmark parameters
    if args.quick:
        worker_counts = [1, 4]
        batch_sizes = [1, 10]
        print("Running quick benchmark...")
    else:
        worker_counts = [1, 2, 4, 8, 16]
        batch_sizes = [1, 5, 10, 20, 50]
        print("Running comprehensive benchmark...")
    
    print(f"Worker counts to test: {worker_counts}")
    print(f"Batch sizes to test: {batch_sizes}")
    
    # Run the benchmark
    results = run_benchmark_comparison(
        splits_to_test=[args.split],
        target_dir=target_dir,
        target_sr=args.target_sr,
        max_files_per_split=args.max_files,
        worker_counts=worker_counts,
        batch_sizes=batch_sizes
    )
    
    if "best_config" in results:
        print("\n🎯 RECOMMENDATIONS:")
        best = results["best_config"]
        print(f"✅ Fastest configuration: {best['method']}")
        print(f"   Workers: {best['workers']}")
        print(f"   Batch size: {best['batch_size']}")
        print(f"   Performance: {best['files_per_second']:.2f} files/second")
        
        print(f"\n📋 To use this configuration in production:")
        if "DataLoader" in best['method']:
            print(f"   python birdset_resample.py --use-dataloader \\")
            print(f"     --workers {best['workers']} --batch-size {best['batch_size']} \\")
            print(f"     --target_dir /your/target/dir --target_sr {args.target_sr}")
        else:
            print(f"   python birdset_resample.py \\")
            print(f"     --workers {best['workers']} --batch-size {best['batch_size']} \\")
            print(f"     --target_dir /your/target/dir --target_sr {args.target_sr}")
    
    print(f"\n📊 Detailed results saved to: {results.get('benchmark_file', 'benchmark_results.csv')}")


if __name__ == "__main__":
    main() 