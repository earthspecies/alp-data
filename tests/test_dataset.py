"""Unit tests for the dataset module."""

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, Literal

import pandas as pd
import pytest
import yaml
from pydantic import BaseModel

from esp_data import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    GenericDataset,
    dataset_from_config,
    list_registered_datasets,
    print_registered_datasets,
    register_config,
    register_dataset,
)
from esp_data.backends import PandasBackend
from esp_data.io import AnyPathT
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


@register_config
class MyCustomConfig(DatasetConfig):
    dataset_name: str = "my_custom_dataset"
    split: str = "train"
    output_take_and_give: dict[str, str] | None = None
    data_root: str | AnyPathT | None = None


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
        output_take_and_give: dict[str, str] | None = None,
        data_root: str | AnyPathT | None = None,
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
        df = pd.DataFrame(
            {
                "path": [f"data/sample_{i}.csv" for i in range(100)],
                "label": [str(i % 2) for i in range(100)],
                "text": [f"Sample text {i}" for i in range(100)],
            },
        )
        self._data = PandasBackend(df, streaming=False)

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
        row = self._data[idx]

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
    def from_config(cls, dataset_config: MyCustomConfig) -> "MyCustomDataset":
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

    def __call__(self, data: PandasBackend) -> tuple[pd.DataFrame, dict]:
        # Rename
        transformed_data = data.rename_columns(self.feature_map)
        return transformed_data, self.feature_map


register_transform(RenameConfig, RenameTransform)


def test_my_custom_dataset():
    """Test instantiating the MyCustomDataset class."""
    dataset_config = MyCustomConfig(dataset_name="my_custom_dataset", split="train")
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
    dataset_config = MyCustomConfig(**cfg["dataset"])
    dataset, _ = dataset_from_config(dataset_config)
    _run_asserts(dataset)

    dataset, _ = dataset_from_config(sample_cfg)
    _run_asserts(dataset)

    dataset, _ = dataset_from_config(str(sample_cfg))
    _run_asserts(dataset)


def test_wrong_collection_from_config():
    """Test that an error is raised when trying to create a dataset from an invalid collection config."""
    with pytest.raises(ValueError):
        dataset_from_config("tests/samples/test_wrong_config.yml")

    with pytest.raises(KeyError):
        dataset_from_config("tests/samples/test_wrong_config.yml", key="non_existent_key")

    with pytest.raises(ValueError, match="Invalid configuration format."):
        dataset_from_config("tests/samples/test_wrong_config.yml", key="nested_collection1")

    with pytest.raises(ValueError, match="Invalid dataset configurations found. Please provide a specific key to select one."):
        dataset_from_config("tests/samples/test_wrong_config.yml", key="some_collection2")

    with pytest.raises(ValueError, match="Invalid configuration format."):
        dataset_from_config("tests/samples/test_wrong_config.yml", key="config3")


# ---------------------------------------------------------------------------
# GenericDataset / Dataset.from_path / Dataset.save_to
# ---------------------------------------------------------------------------

_SAMPLE_ROWS = [{"id": i, "name": f"item_{i}"} for i in range(5)]


def _make_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "data.parquet"
    pd.DataFrame(_SAMPLE_ROWS).to_parquet(str(path), index=False)
    return path


def _make_csv(tmp_path: Path) -> Path:
    path = tmp_path / "data.csv"
    pd.DataFrame(_SAMPLE_ROWS).to_csv(str(path), index=False)
    return path


def _make_json(tmp_path: Path) -> Path:
    path = tmp_path / "data.jsonl"
    with open(path, "w") as f:
        for row in _SAMPLE_ROWS:
            f.write(json.dumps(row) + "\n")
    return path


def _make_webdataset_dir(tmp_path: Path) -> Path:
    wds_dir = tmp_path / "wds"
    wds_dir.mkdir()
    tar_path = wds_dir / "shard_0000.tar"
    with tarfile.open(tar_path, "w") as tar:
        for row in _SAMPLE_ROWS:
            key = f"sample_{row['id']:04d}"
            data = json.dumps(row, indent=2).encode("utf-8")
            info = tarfile.TarInfo(name=f"{key}.sample.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return wds_dir

def _make_info_yaml(tmp_path: Path, backend: str, streaming: bool) -> Path:
    info = {
        "name": "test_dataset",
        "owner": "test_owner",
        "split_paths": {"train": "data.parquet"},
        "version": "1.0.0",
        "description": "Test dataset",
        "sources": ["test"],
        "license": "CC0",
        "backend": backend,
        "streaming": streaming,
    }
    path = tmp_path / "info.yaml"
    with open(path, "w") as f:
        yaml.dump(info, f)
    return path


def test_from_path_parquet(tmp_path):
    """Dataset.from_path loads parquet and returns GenericDataset."""
    _make_parquet(tmp_path)
    _make_info_yaml(tmp_path, backend="polars", streaming=False)
    ds = Dataset.from_path(tmp_path)
    assert isinstance(ds, GenericDataset)
    assert len(ds) == 5
    rows = list(ds)
    assert {r["id"] for r in rows} == {0, 1, 2, 3, 4}


def test_from_path_csv(tmp_path):
    """Dataset.from_path loads CSV and returns GenericDataset."""
    _make_csv(tmp_path)
    _make_info_yaml(tmp_path, backend="polars", streaming=False)

    ds = Dataset.from_path(tmp_path)
    assert isinstance(ds, GenericDataset)
    assert len(ds) == 5


def test_from_path_jsonl(tmp_path):
    """Dataset.from_path loads JSON lines and returns GenericDataset."""
    _make_json(tmp_path)
    _make_info_yaml(tmp_path, backend="polars", streaming=False)
    ds = Dataset.from_path(tmp_path)
    assert isinstance(ds, GenericDataset)
    assert len(ds) == 5


def test_from_path_webdataset(tmp_path):
    """Dataset.from_path loads webdataset directory and returns GenericDataset."""
    from esp_data.backends.webdataset_utils import json_decoder

    wds_dir = _make_webdataset_dir(tmp_path)
    _make_info_yaml(wds_dir, backend="webdataset", streaming=True)
    ds = Dataset.from_path(str(wds_dir), data_processor=json_decoder)
    assert isinstance(ds, GenericDataset)
    rows = [row for row in ds]
    assert len(rows) == 5
    assert {r["id"] for r in rows} == {0, 1, 2, 3, 4}


def test_from_path_webdataset_no_len(tmp_path):
    """GenericDataset wrapping webdataset raises on __len__ and __getitem__."""
    from esp_data.backends.webdataset_utils import json_decoder

    wds_dir = _make_webdataset_dir(tmp_path)
    _make_info_yaml(wds_dir, backend="webdataset", streaming=True)
    ds = Dataset.from_path(str(wds_dir), data_processor=json_decoder)
    with pytest.raises(NotImplementedError):
        len(ds)
    with pytest.raises(RuntimeError):
        ds[0]


def test_from_path_reads_config_yaml(tmp_path):
    """Dataset.from_path populates info from config.yaml if present."""
    _make_parquet(tmp_path)
    info = {
        "name": "my_exported_dataset",
        "owner": "test_owner",
        "split_paths": {"train": "data.parquet"},
        "version": "1.0.0",
        "description": "Exported dataset",
        "sources": ["test"],
        "license": "CC0",
        "backend": "polars",
        "streaming": False,
    }
    with open(tmp_path / "info.yaml", "w") as f:
        yaml.dump(info, f)
    ds = Dataset.from_path(str(tmp_path))
    assert ds.info.name == "my_exported_dataset"
    assert ds.info.version == "1.0.0"


def test_dataset_save_to_no_data_raises(tmp_path):
    """Dataset.save_to raises RuntimeError when _data is None."""
    _make_parquet(tmp_path)
    _make_info_yaml(tmp_path, backend="polars", streaming=False)
    ds = Dataset.from_path(str(tmp_path))
    ds._data = None
    with pytest.raises(RuntimeError, match="No data loaded"):
        ds.save_to(str(tmp_path / "out"))
