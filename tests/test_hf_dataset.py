import tempfile

import pytest

from esp_data.config import DataSample, DatasetConfig
from esp_data.dataset import HFDataset
from esp_data.io import anypath

# constants for all tests
cfg = DatasetConfig(
    name="test",
    creator="test",
    version="0.0.0",
    description="test",
    sources="test",
)

source_dataset = "test"
metadata = {}


class TestSample(DataSample):
    col1: int
    col2: str


samples = [
    TestSample(source_dataset=source_dataset, metadata=metadata, col1=1, col2="a"),
    TestSample(source_dataset=source_dataset, metadata=metadata, col1=2, col2="b"),
]


def test_hf_empty_init():
    with pytest.raises(TypeError):
        HFDataset()

    ds = HFDataset(cfg)
    assert ds.ds is None
    assert ds.config == cfg


def test_hf_from_dict():
    cfg = DatasetConfig(
        name="test",
        creator="test",
        version="0.0.0",
        description="test",
        sources="test",
    )

    # incorrect lengths of columns
    wrong_data = {"col1": [1, 2, 3], "col2": ["a", "b"]}

    with pytest.raises(ValueError):
        HFDataset.from_dict(wrong_data, cfg)

    right_data = {
        "id": ["1", "2", "3"],
        "source_dataset": ["test", "test", "test"],
        "metadata": [{}, {}, {}],
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"],
    }
    ds = HFDataset.from_dict(right_data, cfg)
    assert len(ds) == 3
    assert ds.columns == list(right_data.keys())
    assert ds[0] == {
        "id": "1",
        "source_dataset": "test",
        "metadata": {},
        "col1": 1,
        "col2": "a",
    }


def test_hf_from_samples():
    ds = HFDataset.from_samples(samples, cfg)
    assert ds.ds is not None
    assert len(ds) == 2
    print(ds.columns)
    assert ds.columns == [
        "source_dataset",
        "metadata",
        "id",
        "created_at",
        "derived_from",
        "license",
        "version",
        "col1",
        "col2",
    ]
    assert "id" in ds[0]
    assert "created_at" in ds[0]
    assert isinstance(ds[0]["created_at"], str)
    assert "derived_from" in ds[0]
    assert ds[0]["derived_from"] is None


def test_hf_methods():
    ds = HFDataset.from_samples(samples, cfg)

    # subset
    subset = ds.subset([0])
    assert isinstance(subset, HFDataset)
    assert len(subset) == 1
    assert subset[0]["col1"] == 1

    # filter
    filtered_ds = ds.filter(lambda x: x["col1"] > 1)
    assert isinstance(filtered_ds, HFDataset)
    assert len(filtered_ds) == 1
    assert filtered_ds[0]["col1"] == 2

    # sample
    sampled_ds = ds.sample(1, with_replacement=False)
    assert isinstance(sampled_ds, HFDataset)
    assert len(sampled_ds) == 1

    # add column
    with pytest.raises(ValueError):
        new_ds = ds.add_column("col3", [4, 5, 6])
    new_ds = ds.add_column("col3", [4, 5])
    assert isinstance(new_ds, HFDataset)
    assert len(new_ds) == 2
    assert "col3" in new_ds.columns

    # map
    def fn(x):
        x["col1"] += 1
        return x

    mapped_ds = ds.map(fn)
    assert isinstance(mapped_ds, HFDataset)
    assert len(mapped_ds) == 2
    assert mapped_ds[0]["col1"] == 2


def test_concatenate():
    ds1 = HFDataset.from_samples(samples, cfg)
    ds2 = HFDataset.from_samples(samples, cfg)
    ds3 = ds1.concatenate(ds2, version_update_mode="minor")
    assert isinstance(ds3, HFDataset)
    assert len(ds3) == 4
    assert ds3.columns == ds1.columns
    assert ds3[0] == ds1[0]
    assert ds3[2] == ds2[0]


def test_saving_methods():
    ds = HFDataset.from_samples(samples, cfg)

    # test save config as json
    with tempfile.TemporaryDirectory() as tmpdir:
        ds.save_config(tmpdir)
        f = anypath(tmpdir) / "dataset_config.json"
        assert f.exists()
        # load the config back
        cfg2 = DatasetConfig.from_json(f)
        assert cfg2.name == "test"
        assert cfg2.creator == "test"

    # test save config to cloud
    ds.save_config("gs://esp-ci-cd-tests/esp-data-tests/hf_test_dataset")
    f = anypath("gs://esp-ci-cd-tests/esp-data-tests/hf_test_dataset/dataset_config.json")
    assert f.exists()
    f.unlink()
    assert not f.exists()

    # test save dataset locally
    with tempfile.TemporaryDirectory() as tmpdir:
        ds.save_to_path(tmpdir)
        b = anypath(tmpdir)
        assert b.exists()
        assert (anypath(tmpdir) / "dataset_config.json").exists()

    # test save dataset to cloud
    ds.save_to_path(
        "gs://esp-ci-cd-tests/esp-data-tests/hf_test_dataset",
    )

    assert anypath("gs://esp-ci-cd-tests/esp-data-tests/hf_test_dataset").exists()
    assert anypath("gs://esp-ci-cd-tests/esp-data-tests/hf_test_dataset/dataset_config.json").exists()
    # Deleting a dir doesn't work with gcsfs or AnyPath because cloud folders
    # are not folders, so you have to delete each file inside.
    # F.delete_dir("gs://esp-ci-cd-tests/esp-data-tests/hf_test_dataset")


def test_load_from_path():
    ds = HFDataset.from_samples(samples, cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        ds.save_to_path(tmpdir)
        ds2 = HFDataset.from_path(tmpdir, hf_dataset_type="local_hf")
        assert len(ds2) == 2
        assert ds2.columns == ds.columns
        assert ds2[0] == ds[0]
