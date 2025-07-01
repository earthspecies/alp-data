"""Unit tests for the dataset module."""

from esp_data import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    dataset_from_config,
    register_dataset,
    list_registered_datasets,
    print_registered_datasets
)

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
