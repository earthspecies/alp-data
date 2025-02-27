import os
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from google.cloud import storage
from tqdm import tqdm


def create_zip_batch(files, batch_number, bucket_name, output_bucket_name, temp_dir):
    """Create a zip file containing a batch of files from GCS bucket"""
    start_time = time.time()

    # Create temporary zip file
    zip_path = os.path.join(temp_dir, f"batch_{batch_number:04d}.zip")

    # Download and zip files
    client = storage.Client()
    source_bucket = client.bucket(bucket_name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for i, blob_name in enumerate(files):
            # Download blob to memory
            blob = source_bucket.blob(blob_name)
            content = blob.download_as_bytes()

            # Add to zip with just the filename (not the full path)
            filename = os.path.basename(blob_name)
            zipf.writestr(filename, content)

    # Upload the zip file to the output bucket
    output_bucket = client.bucket(output_bucket_name)
    zip_blob = output_bucket.blob(f"batches/batch_{batch_number:04d}.zip")
    zip_blob.upload_from_filename(zip_path)

    # Remove temporary file
    os.remove(zip_path)

    elapsed = time.time() - start_time
    return f"Batch {batch_number} completed in {elapsed:.2f} seconds"


def batch_and_zip_files(
    bucket_name, output_bucket_name, prefix="", exclude_files: set[str] = None, batch_size=10000, max_workers=8
):
    """
    List files in a GCS bucket and create zip archives with batches of files

    Args:
        bucket_name: Source bucket name (without gs://)
        output_bucket_name: Destination bucket for zip files (without gs://)
        prefix: Optional prefix to filter files in the source bucket
        batch_size: Number of files per zip archive
        max_workers: Number of parallel workers
    """
    # Create client and get bucket
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List all blobs
    exclude_files = set() if exclude_files is None else exclude_files
    print(f"Listing blobs in gs://{bucket_name}/{prefix}...")
    blobs = list(bucket.list_blobs(prefix=prefix))
    blob_names = [blob.name for blob in blobs if blob.name not in exclude_files]

    total_files = len(blob_names)
    print(f"Found {total_files} files. Creating batches of {batch_size}...")

    # Create batches
    batches = [blob_names[i : i + batch_size] for i in range(0, total_files, batch_size)]

    # Create temporary directory for the zip files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, batch in enumerate(batches):
                future = executor.submit(create_zip_batch, batch, i, bucket_name, output_bucket_name, temp_dir)
                futures.append(future)

            # Show progress
            for future in tqdm(futures, total=len(batches), desc="Creating zip archives"):
                print(future.result())

    print(f"All done! {len(batches)} zip archives created in gs://{output_bucket_name}/batches/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch and zip files from a GCS bucket")
    parser.add_argument("source_bucket", help="Source bucket name (without gs://)")
    parser.add_argument("output_bucket", help="Destination bucket for zip files (without gs://)")
    parser.add_argument("--prefix", default="", help="Optional prefix to filter files")
    parser.add_argument("--exclude-files-csv", default=None, help="CSV file with files to exclude")
    parser.add_argument("--batch-size", type=int, default=10000, help="Files per zip archive (default: 10000)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8)")

    args = parser.parse_args()

    exclude_files = None
    if args.exclude_files_csv:
        exclude_files = set(pd.read_csv(args.exclude_files_csv)["filename"].tolist())

    batch_and_zip_files(
        args.source_bucket, args.output_bucket, args.prefix, exclude_files, args.batch_size, args.workers
    )
