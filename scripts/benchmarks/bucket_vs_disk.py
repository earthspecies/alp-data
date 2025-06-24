import time

import pandas as pd

from esp_data.datasets import AnimalSpeak


def benchmark_bucket_vs_disk() -> None:
    """Iterate through datasets and compare
    performance between local disk and cloud bucket."""
    disk_path = (
        "/home/milad_earthspecies_org/data-migration/"
        "marius-highmem/mnt/foundation-model-data/audio_16k"
    )
    disk_ds = AnimalSpeak(split="validation", data_root=disk_path)

    # First bucket is the original gs://animalspeak2 bucket (multi-region)
    bucket_ds1 = AnimalSpeak(split="validation", data_root="gs://")

    # Second bucket is the new
    # gs://esp-ml-datasets/animalspeak/v0.1.0/raw/16KHz bucket (regional)
    bucket_ds2 = AnimalSpeak(
        split="validation",
        data_root="gs://esp-ml-datasets/animalspeak/v0.1.0/raw/16KHz",
    )

    N = len(disk_ds)  # Assuming all datasets have the same length
    datasets = [disk_ds, bucket_ds1, bucket_ds2]
    dataset_aliases = [
        "Local Disk",
        "Original Multi-region bucket",
        "New regional bucket with Hierarchical Namespace",
    ]
    time_taken = []

    print("Benchmarking dataset iteration performance...")
    for ds, alias in zip(datasets, dataset_aliases, strict=False):
        start_time = time.time()
        for _sample in ds:
            pass  # Just iterate through the dataset
        elapsed_time = time.time() - start_time
        print(f"Time taken for {alias}: {elapsed_time:.2f} seconds")
        time_taken.append(elapsed_time)

    # Compute average time per sample
    avg_time_per_sample = [t / N for t in time_taken]
    print("\nAverage time per sample:")
    for alias, avg_time in zip(dataset_aliases, avg_time_per_sample, strict=False):
        print(f"{alias}: {avg_time:.6f} seconds/sample")

    # Save data as CSV
    df = pd.DataFrame(
        {
            "Dataset": dataset_aliases,
            "Time Taken (seconds)": time_taken,
            "Average Time per Sample (seconds)": avg_time_per_sample,
        }
    )
    df.to_csv("bucket_vs_disk_benchmark_results.csv", index=False)


if __name__ == "__main__":
    benchmark_bucket_vs_disk()
    # This will print the time taken to iterate through each dataset
    # on local disk and in both cloud buckets.
