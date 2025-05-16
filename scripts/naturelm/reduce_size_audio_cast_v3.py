"""Using Dask"""

import io

import dask
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, Dataset
from gcsfs import GCSFileSystem

from esp_data import AnyPath


def encode_bytes(audio_data):
    """Convert numpy audio data to FLAC bytes."""
    if audio_data is None or len(audio_data) == 0:
        return None

    try:
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, 16000, format="flac")
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        print(f"Error encoding audio: {e}")
        return None


def process_single_file(file_path, output_path):
    """Process a single parquet file."""
    fs = GCSFileSystem(access="read_write")
    try:
        # Read the file
        with fs.open(file_path, "rb") as f:
            table = pq.read_table(f)
            df = table.to_pandas()

        # Process the audio column
        df["audio"] = df["audio"].map(encode_bytes)

        # Convert to dataset and write
        ds = Dataset.from_pandas(df).cast_column("audio", Audio(sampling_rate=16_000))

        with fs.open(output_path, "wb") as f:
            ds.to_parquet(f)

        return {"status": "success", "file": file_path}
    except Exception as e:
        return {"status": "error", "file": file_path, "error": str(e)}


def main():
    # GCS configuration
    bucket_name = "esp-ml-datasets"
    input_prefix = "naturelm/processed/v0.1.0/parquet/train"
    output_prefix = "naturelm/processed/v0.1.1/parquet/train"

    # Initialize GCS filesystem
    fs = GCSFileSystem()

    # Get all parquet files
    all_files = fs.glob(f"gs://{bucket_name}/{input_prefix}/*.parquet")
    print(f"Found {len(all_files)} parquet files to process")
    # skip files shards 000000 to 000313
    # all_files = all_files[314:]

    # Create output directory if it doesn't exist
    # fs.makedirs(f"gs://{bucket_name}/{output_prefix}", exist_ok=True)

    # Prepare tasks
    tasks = []
    for input_file in all_files:
        # Create corresponding output path
        rel_path = AnyPath(input_file).name
        output_file = f"gs://{bucket_name}/{output_prefix}/{rel_path}"
        # Check if it exists, skip if it does
        if AnyPath(output_file).exists():
            # print(f"Skipping {output_file} as it already exists")
            continue
        # Create a delayed task
        task = dask.delayed(process_single_file)(input_file, output_file)
        tasks.append(task)

    # Execute with progress tracking
    print(f"Processing {len(tasks)} files in parallel...")

    # Set number of workers - adjust based on your machine/cluster capabilities
    num_workers = 30  # min(32, os.cpu_count() * 2)

    # Set computation options
    dask.config.set(
        {
            "distributed.worker.memory.target": 0.9,  # Target 80% memory usage
            "distributed.worker.memory.spill": 0.9,  # Spill to disk at 90%
            "distributed.worker.memory.pause": 0.95,  # Pause worker at 95%
            "distributed.worker.memory.terminate": 0.98,  # Terminate at 98%
        }
    )

    # Process in batches to avoid overwhelming resources
    batch_size = 100
    all_results = []

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(tasks) + batch_size - 1) // batch_size}")
        results = dask.compute(*batch, num_workers=num_workers)
        all_results.extend(results)

        # Summarize batch results
        successes = sum(1 for r in results if r["status"] == "success")
        errors = len(results) - successes
        print(f"Batch complete: {successes} successes, {errors} errors")

    # Final summary
    successes = sum(1 for r in all_results if r["status"] == "success")
    errors = len(all_results) - successes
    print(f"Processing complete: {successes}/{len(all_results)} files processed successfully")

    # Log errors for investigation
    if errors > 0:
        print("Files with errors:")
        for result in all_results:
            if result["status"] == "error":
                print(f"- {result['file']}: {result['error']}")


# from dask.distributed import Client, progress

# def main_distributed():
#     # Set up distributed client - adjust memory limits based on your machine
#     client = Client(n_workers=8, threads_per_worker=4, memory_limit='4GB')
#     print(f"Dask dashboard available at: {client.dashboard_link}")

#     # Same setup code as before...

#     # Submit all tasks to the distributed scheduler
#     futures = [client.submit(process_single_file, input_file, output_file)
#                for input_file, output_file in zip(input_files, output_files)]

#     # Show progress bar
#     progress(futures)

#     # Gather results
#     results = client.gather(futures)

#     # Close the client when done
#     client.close()


if __name__ == "__main__":
    main()
