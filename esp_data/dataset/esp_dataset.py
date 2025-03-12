import pandas as pd

from esp_data.paths import AnyPath, make_storage_options

from .hf import HF_DATASET_TYPES, load_hf_dataset
from .webds import WebDataset

DATASET_TYPES = ["pandas", "webdataset"] + HF_DATASET_TYPES


def load_esp_dataset(dataset_type: str, path: str | AnyPath, **kwargs):
    f"""A function to load different types of datasets.

    Args:
        dataset_type (str): The type of dataset to load.
            one of {DATASET_TYPES}
        path (str | AnyPath): The path to the dataset.
        **kwargs: Additional keyword arguments to pass to the loading function.

    Returns:
        pd.DataFrame | WebDataset | HFDataset: The loaded dataset.
    """
    if dataset_type not in DATASET_TYPES:
        raise ValueError(f"Unsupported dataset type: {dataset_type}, supported types are: {DATASET_TYPES}")

    storage_options = make_storage_options(path)

    if dataset_type == "webdataset":
        return WebDataset.from_path(path, storage_options=storage_options, **kwargs)

    elif dataset_type in HF_DATASET_TYPES:
        return load_hf_dataset(dataset_type, path, storage_options=storage_options, **kwargs)

    elif dataset_type == "pandas":
        # get extension
        ext = AnyPath(path).suffix
        if ext == ".csv":
            return pd.read_csv(path, storage_options=storage_options, **kwargs)
        elif ext == ".tsv":
            return pd.read_csv(path, sep="\t", storage_options=storage_options, **kwargs)
        elif ext == ".json":
            return pd.read_json(path, storage_options=storage_options, **kwargs)
        elif ext == ".parquet":
            return pd.read_parquet(path, storage_options=storage_options, **kwargs)

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
