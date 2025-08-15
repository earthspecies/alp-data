"""Unit tests for the dataset module."""

from pathlib import Path
import yaml
from typing import Any, Dict, Optional, Literal
import pandas as pd
from pydantic import BaseModel

from esp_data.io import anypath, AnyPathT
from esp_data import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    dataset_from_config,
    register_dataset,
    list_registered_datasets,
    print_registered_datasets
)
from esp_data.transforms import register_transform, transform_from_config


def test_register_dataset():
    """Test registering a dataset."""
    class DummyDataset(Dataset):
        info = DatasetInfo(
            name="dummy_dataset",
            owner="test_owner",
            split_paths={"train": "dummy_train_path", "validation": "dummy_validation_path"},
            version="0.1.0",
            description="A dummy dataset for testing purposes.",
            sources=["test_source"],
            license="CC BY"
        )

    register_dataset(DummyDataset)
    assert "dummy_dataset" in list_registered_datasets()


def test_list_registered_datasets():
    """Test listing registered datasets."""
    datasets = list_registered_datasets()
    assert isinstance(datasets, list)
    assert len(datasets) > 0  # Assuming at least one dataset is registered
    assert "animalspeak" in datasets  # Assuming animalspeak is registered by default


def test_print_registered_datasets(capsys):
    """Test printing registered datasets."""
    # Capture the output of print_registered_datasets
    print_registered_datasets()
    captured = capsys.readouterr()
    assert "animalspeak" in captured.out  # Assuming animalspeak is registered by default
    assert "birdset" in captured.out  # Assuming DummyDataset was registered in this test


def test_dataset_from_config():
    """Test creating a dataset from configuration."""
    dataset_config = DatasetConfig(dataset_name="animalspeak", split="validation")
    dataset, _ = dataset_from_config(dataset_config)
    assert isinstance(dataset, Dataset)
    assert dataset.info.name == "animalspeak"
    assert dataset.split == "validation"
    assert dataset.info.split_paths["validation"] is not None  # Assuming the path exists



@register_dataset
class MyCustomDataset(Dataset):
    """My custom dataset description.

    Parameters
    ----------
    split : str
        The split to load. One of info.split_paths keys.
    output_take_and_give : dict[str, str], optional
        A dictionary mapping the original column names to the new column names.
    data_root : str | AnyPathT, optional
        Custom data root directory.
    """

    # Define dataset metadata
    info = DatasetInfo(
        name="my_custom_dataset",
        owner="your_name",
        split_paths={
            "train": "path/to/train.csv",
            "validation": "path/to/validation.csv",
        },
        version="0.1.0",
        description="Description of your dataset",
        sources=["Source 1", "Source 2"],
        license="Your License",
    )

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: Optional[dict[str, str]] = None,
        data_root: Optional[str | AnyPathT] = None,
    ) -> None:
        """Initialize the dataset."""
        super().__init__(output_take_and_give)
        self.split = split
        self._data = None
        self._load()
        self.data_root = data_root

    def _load(self) -> None:
        """Load the dataset data."""
        # Generate the data
        self._data = pd.DataFrame(
            {
                "path": [f"data/sample_{i}.csv" for i in range(100)],
                "label": [str(i % 2) for i in range(100)],
                "text": [f"Sample text {i}" for i in range(100)],
            },
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self._data is None:
            raise RuntimeError("No split has been loaded yet.")
        return len(self._data)

    @property
    def available_splits(self) -> list[str]:
        return ["train"]

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a specific sample from the dataset."""
        if idx < 0 or idx >= len(self._data):
            raise IndexError(f"Index {idx} out of bounds.")

        # Implement your sample loading logic here
        row = self._data.iloc[idx].to_dict()

        # Apply output_take_and_give if specified
        if self.output_take_and_give:
            item = {}
            for key, value in self.output_take_and_give.items():
                item[value] = row[key]
        else:
            item = row

        return item

    def __iter__(self) -> Any:
        for sample in self:
            yield sample

    def __str__(self) -> str:
        return ""

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> "MyCustomDataset":
        """Create a Dataset instance from a configuration."""
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})

        split = cfg.get("split", None)
        if not split or split not in cls.info.split_paths:
            raise LookupError(
                f"Invalid split '{split}'. "
                f"Available splits: {', '.join(cls.info.split_paths.keys())}"
            )

        ds = cls(
            split=split,
            output_take_and_give=cfg.get("output_take_and_give"),
            data_root=cfg.get("data_root"),
        )

        if dataset_config.transformations:
            transform_metadata = ds.apply_transformations(dataset_config.transformations)
            return ds, transform_metadata

        return ds, {}



class RenameConfig(BaseModel):
    type: Literal["rename_transform"]
    feature_map: dict[str, str] | None = None


class RenameTransform:
    def __init__(
        self,
        feature_map: dict[str, str] | None = None,
    ) -> None:
        """Initialize the RenameTransform."""
        self.feature_map = feature_map

    @classmethod
    def from_config(cls, cfg: RenameConfig) -> "RenameTransform":
        return cls(**cfg.model_dump(exclude=("type",)))

    def __call__(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        # Rename
        transformed_data = data.rename(columns=self.feature_map)
        return transformed_data, self.feature_map


register_transform(RenameConfig, RenameTransform)


def test_my_custom_dataset():
    """Test instantiating the MyCustomDataset class."""
    dataset_config = DatasetConfig(dataset_name="my_custom_dataset", split="train")
    dataset, _ = MyCustomDataset.from_config(dataset_config)

    assert isinstance(dataset, MyCustomDataset)
    assert dataset.split == "train"
    assert len(dataset) == 100  # Assuming we generated 100 samples
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "path" in sample
    assert "label" in sample
    assert "text" in sample
    assert sample["label"] in ["0", "1"]  # Assuming binary labels


def test_custom_transform():
    """Test the RenameTransform with a custom configuration."""
    transform_config = RenameConfig(
    type="rename_transform",
    input_features=["fn"],
    output_features=["fn"],
    )

    transform = RenameTransform.from_config(transform_config)
    transform2 = transform_from_config(transform_config)

    assert isinstance(transform, RenameTransform)
    assert isinstance(transform2, RenameTransform)
    assert transform.feature_map == transform2.feature_map


def test_my_custom_dataset_from_yaml():
    """Test loading MyCustomDataset from a YAML configuration file.
    Includes RenameTransform in the configuration.
    """

    def _run_asserts(dataset: Dataset):
        assert isinstance(dataset, MyCustomDataset)
        assert dataset.split == "train"
        assert len(dataset) == 100  # Assuming we generated 100 samples
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "path" not in sample
        assert "label" in sample
        assert "pure_text" in sample
        assert sample["label"] in [0, 1]  # Assuming binary labels

    sample_cfg = Path("tests/samples/my_custom_dataset_cfg.yml")
    with open(sample_cfg, "r") as f:
        cfg = yaml.safe_load(f)
    dataset_config = DatasetConfig(**cfg)
    dataset, _ = dataset_from_config(dataset_config)
    _run_asserts(dataset)

    dataset, _ = dataset_from_config(sample_cfg)
    _run_asserts(dataset)


    dataset, _ = dataset_from_config(str(sample_cfg))
    _run_asserts(dataset)
