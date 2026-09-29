"""Tests for `PackedDataset` and `PackedDatasetConfig`."""

import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from alp_data.dataset import ChainedDatasetConfig, dataset_from_config
from alp_data.packing import PackedDataset, PackedDatasetConfig, pack
from alp_data.packing.columns import BOOKKEEPING_COLS
from alp_data.transforms import FilterConfig
from tests.packing.pack_test_dataset import PackTestConfig, make_source


@pytest.fixture
def packed(tmp_path: Path) -> tuple[PackTestConfig, Path]:
    source = make_source(tmp_path / "src", n=5)
    out = pack(source, tmp_path / "pack", samples_per_shard=2)
    return source, out


def _assert_same_item(got: dict[str, Any], expected: dict[str, Any]) -> None:
    assert set(got) == set(expected)
    for key, value in expected.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(got[key], value)
        elif isinstance(value, np.ndarray):
            assert got[key].dtype == value.dtype
            np.testing.assert_array_equal(got[key], value)
        else:
            assert got[key] == value, key


def test_len_info_and_split_come_from_the_pack(packed: tuple[PackTestConfig, Path]) -> None:
    _, out = packed
    ds = PackedDataset(out)
    assert len(ds) == 5
    assert ds.info.name == "pack_test"
    assert ds.info.version == "0.1.0"
    assert ds.split == "train"
    assert ds.available_splits == ["train"]
    assert "label" in ds.columns
    assert "packed" in str(ds).lower()


def test_every_item_matches_the_source(packed: tuple[PackTestConfig, Path]) -> None:
    source, out = packed
    src, _ = dataset_from_config(source)
    ds = PackedDataset(out)
    for i in range(len(ds)):
        _assert_same_item(ds[i], src[i])


def test_items_do_not_expose_bookkeeping_columns(packed: tuple[PackTestConfig, Path]) -> None:
    _, out = packed
    item = PackedDataset(out)[0]
    assert not set(BOOKKEEPING_COLS) & set(item)


def test_audio_is_float32_and_sample_rate_is_int(packed: tuple[PackTestConfig, Path]) -> None:
    _, out = packed
    item = PackedDataset(out)[1]
    assert item["audio"].dtype == np.float32
    assert isinstance(item["sample_rate"], int)


def test_iteration_yields_every_item_in_order(packed: tuple[PackTestConfig, Path]) -> None:
    source, out = packed
    src, _ = dataset_from_config(source)
    items = list(PackedDataset(out))
    assert len(items) == 5
    for i, item in enumerate(items):
        _assert_same_item(item, src[i])


def test_index_out_of_range_raises(packed: tuple[PackTestConfig, Path]) -> None:
    _, out = packed
    with pytest.raises(IndexError):
        PackedDataset(out)[5]


def test_output_take_and_give_applies_on_top_of_the_frozen_one(
    packed: tuple[PackTestConfig, Path],
) -> None:
    _, out = packed
    ds = PackedDataset(out, output_take_and_give={"audio": "x", "label": "y"})
    item = ds[2]
    assert set(item) == {"x", "y"}
    assert item["y"] == "species_0"


def test_transformations_filter_rows_and_audio_still_resolves(
    packed: tuple[PackTestConfig, Path],
) -> None:
    source, out = packed
    src, _ = dataset_from_config(source)
    ds = PackedDataset(out)
    ds.apply_transformations(
        [FilterConfig(type="filter", mode="include", property="label", values=["species_1"])]
    )
    assert len(ds) == 2
    _assert_same_item(ds[0], src[1])
    _assert_same_item(ds[1], src[3])


def test_pandas_backend_gives_the_same_items(packed: tuple[PackTestConfig, Path]) -> None:
    source, out = packed
    src, _ = dataset_from_config(source)
    ds = PackedDataset(out, backend="pandas")
    _assert_same_item(ds[4], src[4])


def test_config_builds_through_the_registry(packed: tuple[PackTestConfig, Path]) -> None:
    _, out = packed
    cfg = PackedDatasetConfig(path=str(out))
    assert cfg.dataset_name == "packed_dataset"
    ds, meta = dataset_from_config(cfg)
    assert isinstance(ds, PackedDataset)
    assert len(ds) == 5
    assert meta == {}


def test_yaml_chain_mixes_a_pack_with_a_live_dataset(
    packed: tuple[PackTestConfig, Path], tmp_path: Path
) -> None:
    source, out = packed
    chain = ChainedDatasetConfig(
        datasets=[PackedDatasetConfig(path=str(out)), source],
    )
    ds, _ = dataset_from_config(chain)
    assert len(ds) == 10

    # Written the way a user would write it: pydantic serialises chain children
    # as the base DatasetConfig, so model_dump would drop custom fields.
    yaml_path = tmp_path / "chain.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "chain": {
                    "datasets": [
                        {"dataset_name": "packed_dataset", "path": str(out)},
                        {
                            "dataset_name": "pack_test",
                            "csv_path": source.csv_path,
                            "backend": "pandas",
                        },
                    ]
                }
            }
        )
    )
    from_yaml, _ = dataset_from_config(yaml_path)
    assert len(from_yaml) == 10
    _assert_same_item(from_yaml[7], from_yaml[2])


def _read_items(args: tuple[bytes, list[int]]) -> list[dict[str, Any]]:
    ds = pickle.loads(args[0])
    return [ds[i] for i in args[1]]


def test_spawned_workers_read_the_same_items(packed: tuple[PackTestConfig, Path]) -> None:
    source, out = packed
    src, _ = dataset_from_config(source)
    ds = PackedDataset(out)
    ds[0]  # open a handle in the parent to prove it is not inherited
    blob = pickle.dumps(ds)
    with mp.get_context("spawn").Pool(2) as pool:
        chunks = pool.map(_read_items, [(blob, [0, 1, 2]), (blob, [3, 4])])
    items = chunks[0] + chunks[1]
    for i, item in enumerate(items):
        _assert_same_item(item, src[i])


def test_missing_pack_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PackedDataset(tmp_path / "nope")
