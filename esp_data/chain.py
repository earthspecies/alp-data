from typing import Any, Iterator

from esp_data.dataset import (
    ChainedDatasetConfig,
    Dataset,
    DatasetInfo,
    dataset_from_config,
    register_dataset,
)


class ChainException(Exception):
    """Exception raised when dataset chaining fails."""

    pass


@register_dataset
class ChainedDataset(Dataset):
    """Helper class to chain multiple datasets for iteration and indexing.

    This class allows iterating over multiple datasets as if they were a single dataset.

    Parameters
    ----------
    datasets : list[Dataset]
        List of datasets to concatenate for iteration

    Examples
    --------
    >>> from esp_data.datasets import InsectSet459, BirdSet
    >>> from esp_data.concat import ChainedDataset
    >>> dataset1 = InsectSet459(split="validation")
    >>> dataset2 = BirdSet(split="HSN-test")
    >>> concat_iter = ChainedDataset([dataset1, dataset2])
    >>> total_length = len(dataset1) + len(dataset2)
    >>> item = next(iter(concat_iter))
    >>> assert len(concat_iter) == total_length, \
        "Concatenated iterator length should match sum of source datasets lengths"
    """

    info = DatasetInfo(
        name="chained_dataset",
        owner="ESP Data Team",
        split_paths={"chained": "virtual://chained_dataset"},
        version="0.1.0",
        description="A dataset created by chaining multiple datasets for iteration.",
        sources=["Multiple datasets"],
        license="CC0-1.0",
    )

    def __init__(self, datasets: list[Dataset]) -> None:
        if not datasets:
            raise ChainException("At least one dataset must be provided")

        if not all(isinstance(ds, Dataset) for ds in datasets):
            raise ChainException("All objects must be Dataset instances")

        # determine streaming mode based on source datasets
        # all datasets must have the same streaming mode
        streaming_modes = {ds.streaming for ds in datasets}
        if len(streaming_modes) > 1:
            raise ChainException(
                "All datasets must have the same streaming mode "
                "to be concatenated into a ConcatenatedDataset."
            )
        _streaming = streaming_modes.pop()

        # _backend_class doesn'gt matter here since we override all data access methods
        super().__init__(streaming=_streaming)

        self._source_datasets = datasets
        try:
            self._lengths = [len(ds) for ds in datasets]
            self._total_length = sum(self._lengths)
        except RuntimeError:
            self._lengths = []
            self._total_length = -1

        self._all_columns = set()
        for ds in datasets:
            self._all_columns.update(ds.columns)
        self._all_columns = sorted(list(self._all_columns))

    @property
    def columns(self) -> list[str]:
        return self._all_columns

    @property
    def available_splits(self) -> list[str]:
        return ["chained"]

    def _load(self) -> None:
        pass  # Data is already loaded

    def __len__(self) -> int:
        if self._streaming:
            raise RuntimeError("Length is not supported in streaming mode")
        return self._total_length

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for dataset in self._source_datasets:
            for item in dataset:
                yield item

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get item by global index across chained datasets.

        Parameters
        ----------
        idx : int
            Global index across all chained datasets.

        Returns
        -------
        dict[str, Any]
            The item at the specified global index.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        RuntimeError
            If indexing is attempted in streaming mode.
        """
        if self._streaming:
            raise RuntimeError("Indexing is not supported in streaming mode")

        if idx < 0:
            raise IndexError("Negative indexing is not supported")

        if idx >= self._total_length:
            raise IndexError(
                f"Index {idx} out of bounds for concatenated dataset of length {self._total_length}"
            )

        # Determine which dataset the index falls into
        cumulative_length = 0
        for dataset, length in zip(self._source_datasets, self._lengths, strict=True):
            if idx < cumulative_length + length:
                return dataset[idx - cumulative_length]
            cumulative_length += length

    @classmethod
    def from_config(
        cls, chain_config: ChainedDatasetConfig
    ) -> tuple["ChainedDataset", dict[str, Any]]:
        """Create a ConcatenatedDataset from a ConcatConfig object.

        Parameters
        ----------
        chain_config : ChainedDatasetConfig
            Configuration object specifying the datasets to concatenate
            and how to merge them.

        Returns
        -------
        tuple[ChainedDataset, dict]
            A tuple containing the ConcatenatedDataset instance
            and metadata about transformations applied.
        """
        datasets = []
        metadata = {}
        for cfg in chain_config.datasets:
            ds, meta = dataset_from_config(cfg)
            datasets.append(ds)
            metadata.update({f"{cfg.dataset_name}_metadata": meta})
        ds = cls(datasets)

        return ds, metadata

    def __str__(self) -> str:
        return (
            f"{self.info.name} (v{self.info.version})\n"
            f"Description: {self.info.description}\n"
            f"Length: {len(self)}\n"
            f"Columns: {', '.join(self.columns)}\n"
            f"Source datasets: {len(self._datasets)}"
        )
