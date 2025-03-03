import pandas as pd

from esp_data.dataset.shard_creator import load_checkpoint, save_checkpoint, write_shard


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


def test_write_parquet_shard(tmp_path):
    data = make_dummy_dataset()
    data = pd.DataFrame(data)
    prep_function = None
    write_shard(
        batch=data, shard_id=0, output_path=tmp_path, output_format="parquet", sample_prep_function=prep_function
    )
    shard_files = list(tmp_path.glob("*.parquet"))
    assert len(shard_files) == 1


def test_write_webdataset_shard(tmp_path):
    data = make_dummy_dataset(N=1000)
    data = pd.DataFrame(data)
    write_shard(batch=data, shard_id=0, output_path=tmp_path, output_format="webdataset", sample_prep_function=None)
    shard_files = list(tmp_path.glob("*.tar"))
    assert len(shard_files) == 1


def write_multiple_hf_shards(tmp_path):
    data = make_dummy_dataset(N=1000)
    data = pd.DataFrame(data)
    num_samples_per_shard = 100
    batches = [data.iloc[i : i + num_samples_per_shard] for i in range(0, len(data), num_samples_per_shard)]
    for i, batch in enumerate(batches):
        write_shard(batch=batch, shard_id=i, output_path=tmp_path, output_format="hf", sample_prep_function=None)
    shard_files = list(tmp_path.glob("*.arrow"))
    assert len(shard_files) == 10


def test_load_and_save_checkpoint(tmp_path):
    data = make_dummy_dataset(N=1000)
    data = pd.DataFrame(data)
    num_samples_per_shard = 100
    batches = [data.iloc[i : i + num_samples_per_shard] for i in range(0, len(data), num_samples_per_shard)]
    for i, batch in enumerate(batches):
        result = write_shard(
            batch=batch, shard_id=i, output_path=tmp_path, output_format="hf", sample_prep_function=None
        )
        save_checkpoint(tmp_path, result, batch, "checkpoint.json")

    shard_files = list(tmp_path.glob("*.arrow"))
    assert len(shard_files) == 10
    checkpoint = load_checkpoint(tmp_path, batch, "checkpoint.json")
    assert checkpoint is not None
