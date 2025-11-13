"""
Generate a txt file listing available datasets for benchmarking.
This used when running loading_time benchmarks over all datasets (see job 'lt.sh').
"""

from esp_data import list_registered_datasets


def main() -> None:
    """Print available datasets for benchmarking and write configs."""
    registry = list_registered_datasets()
    print("Available datasets for benchmarking:")
    print(registry)

    # Save registry in a dataset_list.txt file
    with open("scripts/benchmarks/dataset_list.txt", "w", encoding="utf-8") as f:
        for dataset in registry:
            f.write(f"{dataset}\n")


if __name__ == "__main__":
    main()
