import json

import pandas as pd

from esp_data.dataset.datasets import load_dataset as lds
from esp_data.dataset.hf import HFDataset
from esp_data.dataset.shard_creator import write_webdataset_shard
from esp_data.dataset.webds import WebDataset, apply_fn


def make_dummy_dataset(N: int = 1000):
    # dummy dataset
    cols = ["col1_str", "col2_int", "col3_jsonstr", "col4_float", "id"]
    data = {}
    data[cols[0]] = [f"str_{i}" for i in range(N)]
    data[cols[1]] = [i for i in range(N)]
    data[cols[2]] = [f'{{"key_{i}": "value_{i}"}}' for i in range(N)]
    data[cols[3]] = [i * 0.1 for i in range(N)]
    data[cols[4]] = [f"id_{i}" for i in range(N)]

    return data


def test_load_pandas_remote():
    df = lds(dataset_type="pandas", path="gs://esp-ci-cd-tests/esp-data-tests/benchmark_dataset/csv_dataset/test.csv")
    assert df.shape == (1000, 5)
    assert "col1_str" in df.columns
    assert "col2_int" in df.columns


def test_load_arrow_remote(tmp_path):
    ds = lds(
        dataset_type="bucket_hf",
        path="gs://esp-ci-cd-tests/esp-data-tests/benchmark_dataset/arrow_dataset/tempfolder",
        streaming=False,
        file_pattern="shard*arrow",
        split="train",
    )

    assert len(ds) > 0

    for batch in ds:
        assert "col1_str" in batch
        assert "col2_int" in batch
        break

    # save to local tmp_path
    ds.save_to_path(tmp_path)
    # check that dataset_config.json is saved
    assert (tmp_path / "dataset_config.json").exists()
    assert (tmp_path / "README.md").exists()

    # load back
    dslocal = lds(dataset_type="local_hf", path=tmp_path, streaming=False, split="train")
    assert "col1_str" in dslocal.columns


def test_load_from_hfhub():
    ds = lds(dataset_type="hf_hub", path="stanfordnlp/imdb", trust_remote_code=True, split="train")
    assert isinstance(ds, HFDataset)
    assert len(ds) == 25000
    assert "text" in ds.columns


def test_load_webdataset_streaming(tmp_path):
    dummy_data = pd.DataFrame(make_dummy_dataset())
    write_webdataset_shard(
        batch=dummy_data, shard_id=0, output_path=tmp_path, sample_prep_function=lambda x: {"data": json.dumps(x)}
    )
    dummy_data.to_csv(tmp_path / "metadata.csv", index=False)
    ds = WebDataset.from_path(tmp_path, load_metadata=False, data_processor=lambda x: json.loads(x["data"]))
    assert ds.config.name == "unknown"

    for sample in apply_fn(ds, lambda x: x["col1_str"].upper(), version_update_mode="patch", changelog="random"):
        assert sample.isupper()
        break
