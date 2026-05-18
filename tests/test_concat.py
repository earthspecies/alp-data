"""Tests for dataset concatenation functionality."""

import math
import pytest
import pandas as pd
from typing import Dict, Any, Iterator, Optional

from esp_data.dataset import Dataset, DatasetInfo, dataset_from_config
from esp_data.concat import MergeException, ConcatenatedDataset
from esp_data.backends import PandasBackend
from esp_data import AnimalSpeak, Beans
from esp_data.transforms import DeduplicateConfig, FilterConfig


# Mock Dataset classes for testing
class MockDataset(Dataset):
    """Mock dataset for testing using backend abstraction."""

    def __init__(
        self,
        backend,
        info: DatasetInfo,
        sample_rate: Optional[int] = None,
        output_take_and_give: Optional[dict] = None,
        backend_type: str = "pandas",
    ):
        super().__init__(output_take_and_give, backend=backend_type)
        self._data = backend
        self.info = info
        self.sample_rate = sample_rate
        self.split = "test"
        self.data_root = None

    @property
    def columns(self) -> list[str]:
        return self._data.columns

    @property
    def available_splits(self) -> list[str]:
        return ["test"]

    def _load(self):
        pass

    @classmethod
    def from_config(cls, dataset_config):
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self._data):
            raise IndexError(f"Index {idx} out of bounds")

        row = self._data[idx]

        if self.output_take_and_give:
            item = {}
            for key, value in self.output_take_and_give.items():
                if key in row:
                    item[value] = row[key]
            return item
        return row

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for idx in range(len(self)):
            yield self[idx]

    def __str__(self) -> str:
        return f"MockDataset({self.info.name})"


@pytest.fixture
def dataset_info1():
    """First dataset info."""
    return DatasetInfo(
        name="dataset1",
        owner="owner1",
        split_paths={"train": "virtual://path1.csv"},
        version="1.0.0",
        description="First dataset",
        sources=["source1"],
        license="MIT",
        changelog="Initial version",
    )


@pytest.fixture
def dataset_info2():
    """Second dataset info."""
    return DatasetInfo(
        name="dataset2",
        owner="owner2",
        split_paths={"train": "virtual://path2.csv"},
        version="1.1.0",
        description="Second dataset",
        sources=["source2"],
        license="MIT",
        changelog="Updated version",
    )


@pytest.fixture
def dataset1_identical_columns(dataset_info1):
    """Dataset with columns A, B, C."""
    data = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"], "C": [1.1, 2.2, 3.3]})
    backend = PandasBackend(data)
    return MockDataset(backend, dataset_info1, sample_rate=16000)


@pytest.fixture
def dataset2_identical_columns(dataset_info2):
    """Dataset with columns A, B, C."""
    data = pd.DataFrame({"A": [4, 5], "B": ["a", "b"], "C": [4.4, 5.5]})
    backend = PandasBackend(data)
    return MockDataset(backend, dataset_info2, sample_rate=16000)


@pytest.fixture
def dataset1_different_columns(dataset_info1):
    """Dataset with columns A, B, C."""
    data = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"], "C": [1.1, 2.2, 3.3]})
    backend = PandasBackend(data)
    return MockDataset(backend, dataset_info1, sample_rate=16000)


@pytest.fixture
def dataset2_different_columns(dataset_info2):
    """Dataset with columns A, D, E."""
    data = pd.DataFrame({"A": [4, 5], "D": ["a", "b"], "E": [4.4, 5.5]})
    backend = PandasBackend(data)
    return MockDataset(backend, dataset_info2, sample_rate=16000)


class TestConcatenateDatasets:
    """Test suite for concatenate_datasets function."""

    def test_hard_merge_identical_columns(
        self, dataset1_identical_columns, dataset2_identical_columns
    ):
        """Test hard merge with identical columns."""
        result = ConcatenatedDataset(
            [dataset1_identical_columns, dataset2_identical_columns], "hard"
        )

        assert len(result) == 5
        assert set(result.columns) == {"A", "B", "C"}
        # Verify data by iterating
        values_A = [result[i]["A"] for i in range(len(result))]
        values_B = [result[i]["B"] for i in range(len(result))]
        assert values_A == [1, 2, 3, 4, 5]
        assert values_B == ["x", "y", "z", "a", "b"]
        assert result.sample_rate == 16000

    def test_hard_merge_different_columns_fails(
        self, dataset1_different_columns, dataset2_different_columns
    ):
        """Test hard merge fails with different columns."""
        with pytest.raises(
            MergeException, match="Hard merge requires identical columns"
        ):
            ConcatenatedDataset(
                [dataset1_different_columns, dataset2_different_columns], "hard"
            )

    def test_overlap_merge_with_common_columns(
        self, dataset1_different_columns, dataset2_different_columns
    ):
        """Test overlap merge keeps only common columns."""
        result = ConcatenatedDataset(
            [dataset1_different_columns, dataset2_different_columns], "overlap"
        )

        assert len(result) == 5
        assert set(result.columns) == {"A"}  # Only common column
        values_A = [result[i]["A"] for i in range(len(result))]
        assert values_A == [1, 2, 3, 4, 5]

    def test_overlap_merge_no_common_columns_fails(self, dataset_info1, dataset_info2):
        """Test overlap merge fails with no common columns."""
        data1 = pd.DataFrame({"A": [1, 2]})
        data2 = pd.DataFrame({"B": [3, 4]})
        ds1 = MockDataset(PandasBackend(data1), dataset_info1)
        ds2 = MockDataset(PandasBackend(data2), dataset_info2)

        with pytest.raises(
            MergeException, match="No common columns found for overlap merge"
        ):
            ConcatenatedDataset([ds1, ds2], "overlap")

    def test_soft_merge_all_columns(
        self, dataset1_different_columns, dataset2_different_columns
    ):
        """Test soft merge keeps all columns."""
        result = ConcatenatedDataset(
            [dataset1_different_columns, dataset2_different_columns], "soft"
        )

        assert len(result) == 5
        assert set(result.columns) == {"A", "B", "C", "D", "E"}
        values_A = [result[i]["A"] for i in range(len(result))]
        assert values_A == [1, 2, 3, 4, 5]
        # Check None/NaN values where columns don't overlap
        assert result[3]["B"] is None or (
            isinstance(result[3]["B"], float) and math.isnan(result[3]["B"])
        )  # dataset2 doesn't have B
        assert result[0]["D"] is None or (
            isinstance(result[0]["D"], float) and math.isnan(result[0]["D"])
        )  # dataset1 doesn't have D

    def test_invalid_merge_level(
        self, dataset1_identical_columns, dataset2_identical_columns
    ):
        """Test invalid merge level raises exception."""
        with pytest.raises(MergeException, match="Invalid merge_level"):
            ConcatenatedDataset(
                [dataset1_identical_columns, dataset2_identical_columns], "invalid"
            )

    def test_different_sample_rates(self, dataset_info1, dataset_info2):
        """Test different sample rates raise exception."""
        data1 = pd.DataFrame({"A": [1, 2]})
        data2 = pd.DataFrame({"A": [3, 4]})
        ds1 = MockDataset(PandasBackend(data1), dataset_info1, sample_rate=16000)
        ds2 = MockDataset(PandasBackend(data2), dataset_info2, sample_rate=22050)

        ds = ConcatenatedDataset([ds1, ds2])
        assert ds.sample_rate is None

    def test_none_sample_rates(self, dataset_info1, dataset_info2):
        """Test handling of None sample rates."""
        data1 = pd.DataFrame({"A": [1, 2]})
        data2 = pd.DataFrame({"A": [3, 4]})
        ds1 = MockDataset(PandasBackend(data1), dataset_info1, sample_rate=None)
        ds2 = MockDataset(PandasBackend(data2), dataset_info2, sample_rate=16000)

        result = ConcatenatedDataset([ds1, ds2])
        assert result.sample_rate is None

    def test_output_take_and_give_merge(self, dataset_info1, dataset_info2):
        """Test merging of output_take_and_give dictionaries."""
        data1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        data2 = pd.DataFrame({"A": [5, 6], "C": [7, 8]})

        otag1 = {"A": "feature1", "B": "feature2"}
        otag2 = {"A": "feature1", "C": "feature3"}  # Same value for A

        ds1 = MockDataset(
            PandasBackend(data1), dataset_info1, output_take_and_give=otag1
        )
        ds2 = MockDataset(
            PandasBackend(data2), dataset_info2, output_take_and_give=otag2
        )

        result = ConcatenatedDataset([ds1, ds2])
        expected_otag = {"A": "feature1", "B": "feature2", "C": "feature3"}
        assert result.output_take_and_give == expected_otag

    def test_concatenate_multiple_datasets(self, dataset_info1, dataset_info2):
        """Test concatenating more than two datasets."""
        # Create a third dataset
        dataset_info3 = DatasetInfo(
            name="dataset3",
            owner="owner3",
            split_paths={"train": "virtual://test3"},
            version="1.2.0",
            description="Third dataset",
            sources=["source3"],
            license="Apache",
        )

        data1 = pd.DataFrame({"A": [1, 2], "B": [10, 20]})
        data2 = pd.DataFrame({"A": [3, 4], "B": [30, 40]})
        data3 = pd.DataFrame({"A": [5, 6], "B": [50, 60]})

        ds1 = MockDataset(PandasBackend(data1), dataset_info1, sample_rate=16000)
        ds2 = MockDataset(PandasBackend(data2), dataset_info2, sample_rate=16000)
        ds3 = MockDataset(PandasBackend(data3), dataset_info3, sample_rate=16000)

        result = ConcatenatedDataset([ds1, ds2, ds3], "soft")

        assert len(result) == 6
        assert result.info.name == "dataset1+dataset2+dataset3"
        assert "owner1; owner2; owner3" in result.info.owner
        assert result.info.version == "1.2.0"  # Highest version
        assert result.info.license == "MIT; Apache"  # Combined licenses

        # Test that we can access items from all three datasets
        items = []
        for i in range(len(result)):
            items.append(result[i])

        assert len(items) == 6
        assert items[0]["A"] == 1  # From dataset1
        assert items[2]["A"] == 3  # From dataset2
        assert items[4]["A"] == 5  # From dataset3

    def test_conflicting_output_take_and_give_fails(self, dataset_info1, dataset_info2):
        """Test conflicting output_take_and_give values raise exception."""
        data1 = pd.DataFrame({"A": [1, 2]})
        data2 = pd.DataFrame({"A": [3, 4]})

        otag1 = {"A": "feature1"}
        otag2 = {"A": "different_feature"}  # Conflicting value

        ds1 = MockDataset(
            PandasBackend(data1), dataset_info1, output_take_and_give=otag1
        )
        ds2 = MockDataset(
            PandasBackend(data2), dataset_info2, output_take_and_give=otag2
        )

        with pytest.raises(MergeException, match="Conflicting values for key 'A'"):
            ConcatenatedDataset([ds1, ds2])

    def test_dataset_info_merging(
        self, dataset1_identical_columns, dataset2_identical_columns
    ):
        """Test DatasetInfo merging."""
        result = ConcatenatedDataset(
            [dataset1_identical_columns, dataset2_identical_columns]
        )

        assert result.info.name == "dataset1+dataset2"
        assert result.info.owner == "owner1; owner2"
        assert result.info.version == "1.1.0"  # Higher version
        assert "Concatenated dataset from:" in result.info.description
        assert result.info.sources == ["source1", "source2"]
        assert result.info.license == "MIT"  # Same license

    def test_dataset_info_different_licenses(self, dataset_info1, dataset_info2):
        """Test DatasetInfo merging with different licenses."""
        dataset_info2.license = "Apache"

        data1 = pd.DataFrame({"A": [1, 2]})
        data2 = pd.DataFrame({"A": [3, 4]})
        ds1 = MockDataset(PandasBackend(data1), dataset_info1)
        ds2 = MockDataset(PandasBackend(data2), dataset_info2)

        result = ConcatenatedDataset([ds1, ds2])
        assert result.info.license == "MIT; Apache"

    def test_no_data_loaded_fails(self, dataset_info1, dataset_info2):
        """Test exception when datasets have no data loaded."""
        data1 = pd.DataFrame({"A": [1, 2]})
        ds1 = MockDataset(PandasBackend(data1), dataset_info1)
        ds2 = MockDataset(PandasBackend(pd.DataFrame()), dataset_info2)
        ds2._data = None  # Simulate no data loaded

        with pytest.raises(
            MergeException, match="Dataset at index 1 has no data loaded"
        ):
            ConcatenatedDataset([ds1, ds2])

    def test_non_dataset_objects_fail(self):
        """Test exception with non-Dataset objects."""
        with pytest.raises(
            MergeException, match="All objects must be Dataset instances"
        ):
            ConcatenatedDataset(["not_a_dataset", "also_not_a_dataset"])

    def test_concatenateddataset_methods(
        self, dataset1_identical_columns, dataset2_identical_columns
    ):
        """Test that concatenated dataset methods work correctly."""
        result = ConcatenatedDataset(
            [dataset1_identical_columns, dataset2_identical_columns]
        )

        # Test __len__
        assert len(result) == 5

        # Test __getitem__
        first_item = result[0]
        assert first_item["A"] == 1
        assert first_item["B"] == "x"

        # Test __iter__
        items = list(result)
        assert len(items) == 5
        assert items[0]["A"] == 1

        # Test out of bounds access
        with pytest.raises(IndexError):
            result[10]

        # Test __str__
        str_repr = str(result)
        assert "dataset1+dataset2" in str_repr
        assert "Length: 5" in str_repr

    def test_output_take_and_give_in_concatenateddataset(
        self, dataset_info1, dataset_info2
    ):
        """Test output_take_and_give works in concatenated dataset."""
        data1 = pd.DataFrame({"original_name": [1, 2], "B": [3, 4]})
        data2 = pd.DataFrame({"original_name": [5, 6], "B": [7, 8]})

        otag = {"original_name": "new_name"}

        ds1 = MockDataset(
            PandasBackend(data1), dataset_info1, output_take_and_give=otag
        )
        ds2 = MockDataset(
            PandasBackend(data2), dataset_info2, output_take_and_give=otag
        )

        result = ConcatenatedDataset([ds1, ds2])

        # Test that output mapping works
        first_item = result[0]
        assert "new_name" in first_item
        assert first_item["new_name"] == 1
        assert "original_name" not in first_item


class TestIntegrationRealDatasets:
    """Integration tests with real datasets."""

    def test_concatenate_animalspeak_and_beans_validation(self):
        """Integration test with AnimalSpeak validation and Beans datasets."""
        # Load small validation splits (should be smaller than train splits)
        animalspeak = AnimalSpeak(
            split="validation", sample_rate=16000, backend="polars"
        )
        beans = Beans(
            split="cbi_validation", sample_rate=16000, backend="polars"
        )
        # Test soft merge (should work despite different columns)
        result = ConcatenatedDataset([animalspeak, beans], merge_level="soft")

        # Verify basic properties
        assert len(result) == len(animalspeak) + len(beans)
        assert result.sample_rate == 16000

        # Verify merged info
        assert result.info.name == "animalspeak+beans"
        assert "david; marius; masato; gagan" in result.info.owner
        assert "Concatenated dataset from:" in result.info.description

        # Test that we can iterate over the result
        first_few_items = []
        for i, item in enumerate(result):
            first_few_items.append(item)
            if i >= 2:  # Just test first few items to avoid long test times
                break

        assert len(first_few_items) == 3

        # Test that columns include both datasets
        animalspeak_cols = set(animalspeak.columns)
        beans_cols = set(beans.columns)
        result_cols = set(result.columns)

        # Soft merge should include all columns
        assert animalspeak_cols.issubset(result_cols)
        assert beans_cols.issubset(result_cols)

    def test_concatenate_same_dataset_different_output_mappings(self):
        """Test concatenating same dataset with different output_take_and_give mappings."""
        # Create two AnimalSpeak datasets with compatible output mappings
        otag1 = {"species_common": "species", "caption": "text"}
        otag2 = {
            "species_common": "species",
            "audio_path": "path",
        }  # Different but compatible

        animalspeak1 = AnimalSpeak(
            split="validation",
            sample_rate=16000,
            output_take_and_give=otag1,
            backend="polars",
        )
        animalspeak2 = AnimalSpeak(
            split="validation",
            sample_rate=16000,
            output_take_and_give=otag2,
            backend="polars",
        )

        result = ConcatenatedDataset([animalspeak1, animalspeak2])

        # Should merge the mappings
        expected_otag = {
            "species_common": "species",
            "caption": "text",
            "audio_path": "path",
        }
        assert result.output_take_and_give == expected_otag

        # get first and last items as check
        first_item = result[0]
        last_item = result[-1]
        assert "species" in first_item
        assert "text" in first_item
        assert "path" in last_item
        assert "species" in last_item

        # Test with conflicting mappings
        otag3 = {"species_common": "different_species"}  # Conflicts with otag1
        animalspeak3 = AnimalSpeak(
            split="validation",
            sample_rate=16000,
            output_take_and_give=otag3,
            backend="polars",
        )

        with pytest.raises(MergeException, match="Conflicting values"):
            ConcatenatedDataset([animalspeak1, animalspeak3])

    def test_overlap_merge_real_datasets(self):
        """Test overlap merge with real datasets that have some common columns."""
        animalspeak = AnimalSpeak(
            split="validation", sample_rate=16000, backend="pandas"
        )
        beans = Beans(
            split="cbi_validation", sample_rate=16000, backend="pandas"
        )

        # Find common columns
        common_cols = set(animalspeak.columns) & set(beans.columns)

        if common_cols:
            result = ConcatenatedDataset(
                [animalspeak, beans], merge_level="overlap"
            )

            # Should only have common columns
            assert set(result.columns) == common_cols
            assert len(result) == len(animalspeak) + len(beans)
        else:
            # If no common columns, should fail
            with pytest.raises(MergeException, match="No common columns found"):
                ConcatenatedDataset(
                    [animalspeak, beans], merge_level="overlap"
                )


# Test to reproduce issue #98
# https://github.com/earthspecies/esp-data/issues/98
def test_pretransformed_before_concat():
    """Test applying transformations before and after concatenation."""
    dedup_cfg = DeduplicateConfig(type="deduplicate", subset=["audio_path"])

    # Use AnimalSpeak - already updated to use backends
    aspeak = AnimalSpeak(split="validation", sample_rate=16000, backend="pandas")
    original_len = len(aspeak)
    # Duplicate the whole dataset to test deduplication
    backend_class = type(aspeak._data)
    aspeak._data = backend_class.concat([aspeak._data, aspeak._data], ignore_index=True)
    # Shuffle it
    aspeak._data = aspeak._data.sample_rows(n=len(aspeak._data), seed=42, replace=False)
    len_before_dedup = len(aspeak)
    assert len_before_dedup == original_len * 2  # Verify we duplicated
    _ = aspeak.apply_transformations([dedup_cfg])
    len_after_dedup = len(aspeak)
    # Should have removed duplicates (approximately back to original length)
    assert len_after_dedup <= original_len + 10  # Allow small variance

    # Use Beans - already updated to use backends
    beans = Beans(split="cbi_validation", sample_rate=16000, backend="pandas")
    filter_cfg = FilterConfig(
        type="filter",
        mode="exclude",
        property="label",
        values=["moublu"],
    )  # Filter out a species
    len_before_filter = len(beans)
    _ = beans.apply_transformations([filter_cfg])
    len_after_filter = len(beans)
    # Should have filtered some rows
    assert len_after_filter <= len_before_filter

    # Concatenate datasets after transformations
    ds = ConcatenatedDataset([aspeak, beans], merge_level="soft")

    assert len(ds) == len(aspeak) + len(beans)

    # Try indexing to ensure it works
    sample1 = ds[0]
    sample2 = ds[len(ds) // 2]
    sample3 = ds[len(ds) - 1]

    assert isinstance(sample1, dict)
    assert isinstance(sample2, dict)
    assert isinstance(sample3, dict)


def test_concat_from_config() -> None:
    """Test concatenating datasets from config files."""
    ds, _ = dataset_from_config("tests/samples/test_concat_config.yml")
    assert isinstance(ds, ConcatenatedDataset)
    assert len(ds._source_datasets) == 2
    assert ds.split == "concatenated"

    for sample in ds:
        assert isinstance(sample, dict)
        assert "audio" in sample
        assert len(sample["audio"]) > 0
        assert "label" in sample
        break


@pytest.mark.skip(reason="Skipping dtype test for now as Polars dtypes may vary.")
@pytest.mark.parametrize("backend_type", ["pandas", "polars"])
def test_dtypes_after_concat(backend_type: str) -> None:
    # Load small validation splits (should be smaller than train splits)
    animalspeak = AnimalSpeak(
        split="validation", sample_rate=16000, backend=backend_type
    )
    beans = Beans(
        split="cbi_validation", sample_rate=16000, backend=backend_type
    )
    # Test soft merge (should work despite different columns)
    result = ConcatenatedDataset([animalspeak, beans], merge_level="soft")

    input_dtypes1 = animalspeak._data.unwrap.dtypes
    input_cols1 = animalspeak.columns
    input_dtype_data1 = {
        col: dtype
        for col, dtype in zip(input_cols1, input_dtypes1)
    }
    input_dtypes2 = beans._data.unwrap.dtypes
    input_cols2 = beans.columns
    input_dtype_data2 = {
        col: dtype
        for col, dtype in zip(input_cols2, input_dtypes2)
    }

    result_dtypes = result._data.unwrap.dtypes
    result_cols = result.columns
    result_dtype_data = {
        col: dtype
        for col, dtype in zip(result_cols, result_dtypes)
    }

    # Check that columns from animalspeak have same dtypes in result
    for col in input_cols1:
        assert col in result_cols
        assert input_dtype_data1[col] == result_dtype_data[col]

    # Check that columns from beans have same dtypes in result
    for col in input_cols2:
        assert col in result_cols
        assert input_dtype_data2[col] == result_dtype_data[col]
