"""Dataset concatenation utilities."""

import logging
from typing import Any, Dict, Iterator, Literal, Optional

import pandas as pd
import semver

from esp_data.dataset import Dataset, DatasetConfig, DatasetInfo

logger = logging.getLogger("esp_data")


class MergeException(Exception):
    """Exception raised when dataset concatenation fails."""

    pass


def _merge_dataframes(
    dataframes: list[pd.DataFrame], merge_level: Literal["hard", "overlap", "soft"]
) -> pd.DataFrame:
    """Merge multiple DataFrames based on the specified merge level.

    Parameters
    ----------
    dataframes : list[pd.DataFrame]
        DataFrames to merge
    merge_level : {"hard", "overlap", "soft"}
        Merge strategy

    Returns
    -------
    pd.DataFrame
        Merged DataFrame

    Raises
    ------
    MergeException
        If merge cannot be performed according to merge_level
    """
    if not dataframes:
        raise MergeException("No dataframes to merge")

    if len(dataframes) == 1:
        return dataframes[0].copy()

    if merge_level == "hard":
        # All dataframes must have identical columns
        first_columns = set(dataframes[0].columns)
        for i, df in enumerate(dataframes[1:], 1):
            if set(df.columns) != first_columns:
                raise MergeException(
                    f"Hard merge requires identical columns. "
                    f"Dataset 0 columns: {first_columns}, "
                    f"Dataset {i} columns: {set(df.columns)}"
                )
        return pd.concat(dataframes, ignore_index=True)

    elif merge_level == "overlap":
        # Find common columns across all dataframes
        common_columns = set(dataframes[0].columns)
        for df in dataframes[1:]:
            common_columns &= set(df.columns)

        # remove _source_dataset and _source_index
        # because they are not part of the original data
        common_columns.discard("_source_dataset")
        common_columns.discard("_source_index")

        if not common_columns:
            raise MergeException("No common columns found for overlap merge")

        # Preserve column order from first dataframe
        ordered_common_columns = [col for col in dataframes[0].columns if col in common_columns]
        selected_dfs = [df[ordered_common_columns] for df in dataframes]
        return pd.concat(selected_dfs, ignore_index=True)

    elif merge_level == "soft":
        return pd.concat(dataframes, ignore_index=True, sort=False)

    else:
        raise MergeException(
            f"Invalid merge_level: {merge_level}. Must be 'hard', 'overlap', or 'soft'"
        )


def _merge_sample_rates(sample_rates: list[Optional[int]]) -> Optional[int]:
    """Merge sample rates from multiple datasets.

    Parameters
    ----------
    sample_rates : list[int or None]
        Sample rates from the datasets

    Returns
    -------
    int or None
        Combined sample rate

    Raises
    ------
    MergeException
        If sample rates are different
    """
    # Filter out None values
    non_none_rates = [sr for sr in sample_rates if sr is not None]

    if not non_none_rates:
        return None

    # Check that all non-None rates are the same
    first_rate = non_none_rates[0]
    for i, rate in enumerate(non_none_rates[1:], 1):
        if rate != first_rate:
            raise MergeException(
                f"Sample rates must match: dataset 0={first_rate}, dataset {i}={rate}"
            )

    return first_rate


def _merge_output_take_and_give(otags: list[Optional[dict]]) -> Optional[dict]:
    """Merge output_take_and_give dictionaries from multiple datasets.

    Parameters
    ----------
    otags : list[dict or None]
        output_take_and_give dictionaries from the datasets

    Returns
    -------
    dict or None
        Combined output_take_and_give dictionary

    Raises
    ------
    MergeException
        If there are conflicting values for the same key
    """
    combined_otag = {}

    for i, otag in enumerate(otags):
        otag = otag or {}

        # Check for conflicting values
        for key, value in otag.items():
            if key in combined_otag and combined_otag[key] != value:
                raise MergeException(
                    f"Conflicting values for key '{key}' in output_take_and_give: "
                    f"'{combined_otag[key]}' vs '{value}' (from dataset {i})"
                )
            combined_otag[key] = value

    return combined_otag if combined_otag else None


def _merge_dataset_info(dataset_infos: list[DatasetInfo]) -> DatasetInfo:
    """Merge multiple DatasetInfo objects.

    Parameters
    ----------
    dataset_infos : list[DatasetInfo]
        DatasetInfo objects to merge

    Returns
    -------
    DatasetInfo
        Merged DatasetInfo object

    Raises
    ------
    MergeException
        If no dataset infos are provided or if they cannot be merged
    """
    if not dataset_infos:
        raise MergeException("No dataset infos to merge")

    if len(dataset_infos) == 1:
        # Still need to create a copy with virtual split path
        info = dataset_infos[0]
        return DatasetInfo(
            name=info.name,
            owner=info.owner,
            split_paths={"concatenated": "virtual://concatenated_dataset"},
            version=info.version,
            description=info.description,
            sources=info.sources,
            license=info.license,
            changelog=info.changelog,
        )

    # Merge names
    merged_name = "+".join(info.name for info in dataset_infos)

    # Merge owners - combine and deduplicate
    all_owners = []
    for info in dataset_infos:
        owner_list = info.owner.split(";") if isinstance(info.owner, str) else [info.owner]
        all_owners.extend([o.strip() for o in owner_list])
    merged_owners = list(dict.fromkeys(all_owners))  # Preserve order, remove duplicates
    merged_owner = "; ".join(merged_owners)

    # Use virtual split path for concatenated dataset
    merged_split_paths = {"concatenated": "virtual://concatenated_dataset"}

    # Merge versions - use the highest version
    try:
        versions = []
        for info in dataset_infos:
            try:
                versions.append((semver.VersionInfo.parse(info.version), info.version))
            except ValueError:
                # If parsing fails, treat as 0.0.0 for comparison
                versions.append((semver.VersionInfo.parse("0.0.0"), info.version))
        merged_version = max(versions, key=lambda x: x[0])[1]
    except ImportError:
        # Fallback to lexicographic comparison
        merged_version = max(info.version for info in dataset_infos)

    # Merge descriptions
    desc_lines = [f"{i + 1}. {info.description}" for i, info in enumerate(dataset_infos)]
    merged_description = "Concatenated dataset from:\n" + "\n".join(desc_lines)

    # Merge sources
    all_sources = []
    for info in dataset_infos:
        sources = info.sources if isinstance(info.sources, list) else [info.sources]
        all_sources.extend(sources)
    merged_sources = list(dict.fromkeys(all_sources))  # Remove duplicates

    # Merge licenses
    unique_licenses = list(dict.fromkeys([info.license for info in dataset_infos]))
    merged_license = "; ".join(unique_licenses)

    # Merge changelogs
    changelog_lines = [f"{i + 1}. {info.changelog}" for i, info in enumerate(dataset_infos)]
    merged_changelog = "Concatenated from:\n" + "\n".join(changelog_lines)

    return DatasetInfo(
        name=merged_name,
        owner=merged_owner,
        split_paths=merged_split_paths,
        version=merged_version,
        description=merged_description,
        sources=merged_sources,
        license=merged_license,
        changelog=merged_changelog,
    )


def concatenate_datasets(
    datasets: list[Dataset], merge_level: Literal["hard", "overlap", "soft"] = "soft"
) -> Dataset:
    """Concatenate multiple Dataset objects into a single Dataset.

    Parameters
    ----------
    datasets : list[Dataset]
        List of datasets to concatenate
    merge_level : {"hard", "overlap", "soft"}, default="soft"
        Strategy for handling different columns:
        - "hard": All columns must match exactly across all datasets
        - "overlap": Keep only common columns across all datasets
        - "soft": Keep all columns from all datasets (fill missing with NaN)

    Returns
    -------
    Dataset
        New Dataset object with concatenated data

    Raises
    ------
    MergeException
        If datasets cannot be merged according to the specified merge_level,
        or if there are conflicting values for identical keys in output_take_and_give
    """
    # Validate inputs
    if not datasets:
        raise MergeException("At least one dataset must be provided")

    if not all(isinstance(ds, Dataset) for ds in datasets):
        raise MergeException("All objects must be Dataset instances")

    if len(datasets) == 1:
        # If only one dataset, return it directly
        return datasets[0]

    for i, dataset in enumerate(datasets):
        if dataset._data is None:
            raise MergeException(f"Dataset at index {i} has no data loaded")

    # Store original output_take_and_give mappings and clear them from source datasets
    original_otags = []
    for dataset in datasets:
        original_otags.append(getattr(dataset, "output_take_and_give", None))
        # Clear the output_take_and_give from source datasets
        dataset.output_take_and_give = None

    try:
        # Create enhanced dataframes with source tracking
        enhanced_dfs = []
        for i, dataset in enumerate(datasets):
            df_enhanced = dataset._data.copy()
            df_enhanced["_source_dataset"] = i  # Track which dataset each row came from
            df_enhanced["_source_index"] = df_enhanced.index  # Original index in source dataset
            enhanced_dfs.append(df_enhanced)

        # Merge DataFrames
        concatenated_df = _merge_dataframes(enhanced_dfs, merge_level)

        # Merge sample rates
        sample_rates = [getattr(ds, "sample_rate", None) for ds in datasets]
        combined_sample_rate = _merge_sample_rates(sample_rates)

        # Merge output_take_and_give from original mappings
        combined_otag = _merge_output_take_and_give(original_otags)

        # Merge DatasetInfo
        dataset_infos = [ds.info for ds in datasets]
        merged_info = _merge_dataset_info(dataset_infos)

        return ConcatenatedDataset(
            data=concatenated_df,
            dataset_info=merged_info,
            source_datasets=datasets,
            sample_rate=combined_sample_rate,
            output_take_and_give=combined_otag,
        )

    finally:
        # Restore original output_take_and_give mappings to source datasets
        for dataset, original_otag in zip(datasets, original_otags, strict=False):
            dataset.output_take_and_give = original_otag


class ConcatenatedDataset(Dataset):
    """A dataset created by concatenating multiple datasets.

    This dataset maintains references to the original datasets to enable
    proper audio loading and other dataset-specific functionality.
    """

    # No class-level info attribute to avoid overwriting issues

    def __init__(
        self,
        data: pd.DataFrame,
        dataset_info: DatasetInfo,
        source_datasets: list[Dataset],
        sample_rate: Optional[int] = None,
        output_take_and_give: Optional[dict] = None,
    ) -> None:
        super().__init__(output_take_and_give)
        self._data = data
        self.info = dataset_info  # Instance attribute, not class attribute
        self._source_datasets = source_datasets  # Private attribute
        self.sample_rate = sample_rate
        self.split = "concatenated"

    @property
    def columns(self) -> list[str]:
        # Filter out internal tracking columns
        return [col for col in self._data.columns if not col.startswith("_source_")]

    @property
    def available_splits(self) -> list[str]:
        return ["concatenated"]

    def _load(self) -> None:
        pass  # Data is already loaded

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> None:
        raise NotImplementedError(
            "ConcatenatedDataset cannot be instantiated from config. "
            "Use concatenate_datasets function instead."
        )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self._data):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self._data)}")

        row = self._data.iloc[idx].to_dict()

        # Determine which source dataset this row came from
        source_dataset_idx = int(row["_source_dataset"])
        source_row_idx = int(row["_source_index"])
        source_dataset = self._source_datasets[source_dataset_idx]

        # Get the original item from the source dataset
        try:
            # Temporarily restore the original output_take_and_give to get raw data
            original_otag = source_dataset.output_take_and_give
            source_dataset.output_take_and_give = None

            source_item = source_dataset[source_row_idx]

            # Restore the output_take_and_give
            source_dataset.output_take_and_give = original_otag

        except Exception as e:
            raise RuntimeError(
                f"Failed to load item {source_row_idx} "
                f"from source dataset {source_dataset_idx}: {e}"
            ) from e

        # Apply the concatenated dataset's output_take_and_give mapping
        if self.output_take_and_give:
            item = {}
            for key, value in self.output_take_and_give.items():
                if key in source_item:
                    item[value] = source_item[key]
            return item
        return source_item

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for idx in range(len(self)):
            yield self[idx]

    def __str__(self) -> str:
        return (
            f"{self.info.name} (v{self.info.version})\n"
            f"Description: {self.info.description}\n"
            f"Length: {len(self._data)}\n"
            f"Columns: {', '.join(self.columns)}\n"
            f"Source datasets: {len(self._source_datasets)}"
        )
