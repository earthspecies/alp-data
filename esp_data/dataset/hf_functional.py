from enum import Enum
from typing import Any, Iterable

from datasets import Dataset
from datasets import load_dataset as hf_load_dataset
from datasets import load_from_disk as hf_load_from_disk

from esp_data.config import DataSample
from esp_data.paths import AnyPath


class DatasetType(Enum):
    JSON = "json"
    JSONL = "jsonl"
    ARROW = "arrow"
    CSV = "csv"
    AUDIO = "audiofolder"
    PARQUET = "parquet"
    FEATHER = "feather"


def load_dataset(
    dataset_id: str | None = None,
    data_dir: str | AnyPath = None,
    data_files: str | AnyPath | dict[str, str] = None,
    storage_options: dict = None,
    split: str = None,
    streaming: bool = False,
    **kwargs,
) -> Dataset:
    """Load a dataset from a given directory

    Args:
        data_dir (str | AnyPath): Directory containing the dataset
        data_files (str | list[str | AnyPath]): File(s) to load
        dataset_id (str): Type of dataset to load, or, path on huggingface hub, e.g. 'common_voice/en'
            see https://huggingface.co/docs/datasets/loading#local-and-remote-files
        split (str, optional): Split to load. Defaults to None.
        storage_options (dict, optional): Storage options for the dataset. Defaults to None.
        streaming (bool, optional): Whether to stream the dataset. Defaults to False.

    Returns:
        Dataset: Loaded dataset

    Raises:
        ValueError: If the dataset type is not supported
    """
    assert dataset_id is not None, "Dataset id is required"
    # first check if load_from_disk is required
    if dataset_id is None:
        ds = hf_load_from_disk(data_dir, storage_options=storage_options, **kwargs)
        if split is not None:
            ds = ds[split]

        if streaming:
            ds = ds.to_iterable_dataset()

    elif dataset_id is not None:
        ds = hf_load_dataset(
            dataset_id,
            data_files=data_files,
            data_dir=data_dir,
            streaming=streaming,
            storage_options=storage_options,
            **kwargs,
        )

    raise ValueError(f"Unsupported dataset type: {dataset_id}")


def make_dataset(samples: Iterable[DataSample | dict], data_dict: dict[str, Any]) -> Dataset:
    """Create a dataset from a list of data samples

    Args:
        samples (Iterable[DataSample]): List of data samples
        data_dict (dict[str, Any]): Dictionary containing the data

    Returns:
        Dataset: Dataset containing the data
    """
    if samples and isinstance(samples[0], DataSample):
        samples = [s.to_dict() for s in samples]

        return Dataset.from_list(samples)

    return Dataset.from_dict(data_dict)
