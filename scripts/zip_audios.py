import os
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

import esp_data.io.functional as F
from esp_data.io import AnyPath


def create_zip_batch(files: list[str], batch_number: int, output_bucket_path: str, temp_dir: str):
    """Create a zip file containing a batch of files from GCS bucket"""
    start_time = time.time()

    # Create temporary zip file
    zip_path = os.path.join(temp_dir, f"batch_{batch_number:04d}.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            # Download blob to memory
            print(f"Fecthing {file}...")
            try:
                content = AnyPath(file).read_bytes()
            except Exception as e:
                raise e
            # Add to zip with just the filename (not the full path)
            filename = os.path.basename(file)
            zipf.writestr(filename, content)

    # Upload the zip file to the output bucket
    output_zip_path = AnyPath(output_bucket_path) / f"batch_{batch_number:04d}.zip"
    F.cp_to_cloud(zip_path, output_zip_path)

    # Remove temporary file
    os.remove(zip_path)

    elapsed = time.time() - start_time
    return f"Batch {batch_number} completed in {elapsed:.2f} seconds"


def batch_and_zip_files(
    source_bucket: str,
    destination_bucket: str,
    source_files: list[str],
    exclude_files: set[str] = None,
    batch_size=10000,
    max_workers=8,
):
    """
    List files in a GCS bucket and create zip archives with batches of files

    Args:
        source_bucket: Source bucket name
        destination_bucket: Destination bucket for zip files
        source_files: List of files to include
        exclude_files: List of files to exclude
        batch_size: Number of files per zip archive
        max_workers: Number of parallel workers
    """
    # Create client and get bucket
    bucket_path = AnyPath(source_bucket)

    # List all blobs
    exclude_files = set() if exclude_files is None else exclude_files
    print(f"Listing blobs in {bucket_path}...")

    if not source_files:
        iterator = F.yield_files(bucket_path)
    else:
        iterator = source_files

    file_names = []
    for file in iterator:
        if file in exclude_files:
            continue
        file_names.append(file)

    total_files = len(file_names)
    print(f"Found {total_files} files. Creating batches of {batch_size}...")

    # Create batches
    batches = [file_names[i : i + batch_size] for i in range(0, total_files, batch_size)]

    # Create temporary directory for the zip files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, batch in enumerate(batches):
                future = executor.submit(create_zip_batch, batch, i, destination_bucket, temp_dir)
                futures.append(future)

            # Show progress
            for future in tqdm(futures, total=len(batches), desc="Creating zip archives"):
                print(future.result())

    print(f"All done! {len(batches)} zip archives created in {destination_bucket}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch and zip files from a GCS bucket")
    parser.add_argument("source_bucket", help="Source bucket name (without gs://)")
    parser.add_argument("destination_bucket", help="Destination bucket for zip files (without gs://)")
    parser.add_argument("--source-files-csv", default=None, help="CSV file with files to include")
    parser.add_argument("--exclude-files-csv", default=None, help="CSV file with files to exclude")
    parser.add_argument("--batch-size", type=int, default=10000, help="Files per zip archive (default: 10000)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8)")

    args = parser.parse_args()

    exclude_files = None
    if args.exclude_files_csv:
        exclude_files = set(pd.read_csv(args.exclude_files_csv)["path"].tolist())

    source_files = None
    if args.source_files_csv:
        source_files = pd.read_csv(args.source_files_csv)["path"].tolist()

    batch_and_zip_files(
        args.source_bucket, args.destination_bucket, source_files, exclude_files, args.batch_size, args.workers
    )
