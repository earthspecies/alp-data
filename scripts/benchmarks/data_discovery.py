"""
Generate a txt file listing available datasets for benchmarking.
This used when running loading_time benchmarks over all datasets (see job 'lt.sh').
"""

import pandas as pd

from esp_data import dataset_class_from_name, list_registered_datasets


def generate_dataset_table_with_attributes() -> None:
    """Generate a table of datasets and attributes from DataInfo."""

    registry = list_registered_datasets()
    # print(registry)
    attributes_list = []
    for dataset in registry:
        dataset_class = dataset_class_from_name(dataset)
        # attributes = dataset.get_attributes()  # Placeholder for actual attribute retrieval
        print(f"{dataset_class=}")
        print(f"{dataset_class.info=}")
        attributes = dataset_class.info.model_dump()
        attributes_list.append(attributes)
    table = pd.DataFrame(attributes_list, index=registry)
    table.to_csv("scripts/benchmarks/dataset_attributes_table_test.csv")


if __name__ == "__main__":
    generate_dataset_table_with_attributes()
