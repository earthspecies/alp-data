import pytest

from esp_data import ConcatenatedDataset, ChainedDataset
from esp_data.datasets import HawaiianBirds, NocturnalBirdMigration
from esp_data.concat import MergeException


def test_concat_of_selection_table_datasets() -> None:
    hb = HawaiianBirds(split="all", backend="pandas", streaming=False)
    nbm = NocturnalBirdMigration(split="test", backend="pandas", streaming=False)

    result = ConcatenatedDataset(
        [hb, nbm], merge_level="soft"
    )
    assert len(result) == len(hb) + len(nbm)
    assert "selection_table" in result.columns

    # Check that we can index into the concatenated dataset
    sample1 = next(iter(result))
    # Issue #179: concatenated dataset will lead to overwriting the source
    # datasets with the concatenated backend
    assert isinstance(sample1["selection_table"], str)

    # Now try with chained
    result = ChainedDataset([hb, nbm])
    assert len(result) == len(hb) + len(nbm)

    # Check that we can index into the concatenated dataset
    sample1 = next(iter(result))
    # Issue #179: With chained dataset, no overwrite
    assert isinstance(sample1["selection_table"], pd.DataFrame)


def test_chained_dataset_streaming() -> None:
    nbm = NocturnalBirdMigration(split="test", backend="polars", streaming=True)
    hb = HawaiianBirds(split="all", backend="polars", streaming=True)

    # ConcatenatedDataset doesn't allow streaming
    with pytest.raises(MergeException, match="Concatenation is only allowed with streaming=False"):
        ConcatenatedDataset([nbm, hb], merge_level="soft")

    ds = ChainedDataset([nbm, hb])

    with pytest.raises(RuntimeError, match="Indexing is not supported in streaming mode"):
        ds[-1]

    with pytest.raises(RuntimeError, match="Length is not supported in streaming mode"):
        len(ds)

    sample = next(iter(ds))
    assert "audio" in sample
