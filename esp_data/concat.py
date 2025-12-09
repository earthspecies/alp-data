"""Dataset concatenation utilities."""

import logging
from typing import Any, Dict, Iterator, Literal

import semver

from esp_data.backends.protocol import DataBackend
from esp_data.dataset import (
    ConcatConfig,
    Dataset,
    DatasetInfo,
    dataset_from_config,
    register_dataset,
)

logger = logging.getLogger("esp_data")


class MergeException(Exception):
    """Exception raised when dataset concatenation fails."""

    pass


def _merge_backends(
    backends: list["DataBackend"], merge_level: Literal["hard", "overlap", "soft"]
) -> "DataBackend":
    """Merge multiple backend instances based on the specified merge level.

    Parameters
    ----------
    backends : list[DataBackend]
        Backend instances to merge
    merge_level : {"hard", "overlap", "soft"}
        Merge strategy

    Returns
    -------
    DataBackend
        Merged backend instance

    Raises
    ------
    MergeException
        If merge cannot be performed according to merge_level
    """
    if not backends:
        raise MergeException("No backends to merge")

    if len(backends) == 1:
        return backends[0].copy()

    # Get the backend class from the first backend
    backend_class = type(backends[0])

    if merge_level == "hard":
        # All backends must have identical columns
        first_columns = set(backends[0].columns)
        for i, backend in enumerate(backends[1:], 1):
            if set(backend.columns) != first_columns:
                raise MergeException(
                    f"Hard merge requires identical columns. "
                    f"Dataset 0 columns: {first_columns}, "
                    f"Dataset {i} columns: {set(backend.columns)}"
                )
        return backend_class.concat(backends, ignore_index=True)

    elif merge_level == "overlap":
        # Find common columns across all backends
        common_columns = set(backends[0].columns)
        for backend in backends[1:]:
            common_columns &= set(backend.columns)

        # remove _source_dataset and _source_index
        # because they are not part of the original data
        common_columns.discard("_source_dataset")
        common_columns.discard("_source_index")

        if not common_columns:
            raise MergeException("No common columns found for overlap merge")

        # Preserve column order from first backend
        ordered_common_columns = [col for col in backends[0].columns if col in common_columns]
        ordered_common_columns += ["_source_dataset", "_source_index"]
        selected_backends = [backend.select_columns(ordered_common_columns) for backend in backends]
        return backend_class.concat(selected_backends, ignore_index=True)

    elif merge_level == "soft":
        return backend_class.concat(backends, ignore_index=True, sort=False)

    else:
        raise MergeException(
            f"Invalid merge_level: {merge_level}. Must be 'hard', 'overlap', or 'soft'"
        )


def _merge_sample_rates(sample_rates: list[int | None]) -> int | None:
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


def _merge_output_take_and_give(otags: list[dict | None]) -> dict | None:
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
) -> tuple["DataBackend", DatasetInfo, list[Dataset], int, dict]:
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
    tuple[DataBackend, DatasetInfo, list[Dataset], Optional[int], Optional[dict]]
        Tuple containing:
        - concatenated_backend: Merged backend instance
        - merged_info: Merged dataset info
        - datasets: List of source datasets
        - combined_sample_rate: Merged sample rate
        - combined_otag: Merged output_take_and_give dict

    Raises
    ------
    MergeException
        If datasets cannot be concatenated according to merge_level

    Examples
    --------
    >>> from esp_data.datasets import InsectSet459, BirdSet
    >>> from esp_data.concat import concatenate_datasets
    >>> dataset1 = InsectSet459(split="validation")
    >>> dataset2 = BirdSet(split="HSN-test")
    >>> backend, info, datasets, sr, otag = concatenate_datasets(
    ...    [dataset1, dataset2], merge_level="soft")
    >>> assert len(backend) == len(dataset1) + len(dataset2)
    """
    if len(datasets) == 1:
        # If only one dataset, return a copy of its data
        ds = datasets[0]
        return (
            ds._data.copy(),
            ds.info,
            datasets,
            ds.sample_rate,
            ds.output_take_and_give,
        )

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
        # Create enhanced backends with source tracking
        enhanced_backends = []
        for i, dataset in enumerate(datasets):
            backend_copy = dataset._data.copy()
            # Add source dataset tracking
            backend_with_tracking = backend_copy.add_column("_source_dataset", i)
            # Add source index tracking (original row indices)
            indices = list(range(len(backend_copy)))
            backend_enhanced = backend_with_tracking.add_column("_source_index", indices)
            enhanced_backends.append(backend_enhanced)

        # Merge backends
        concatenated_backend = _merge_backends(enhanced_backends, merge_level)

        # Merge sample rates
        sample_rates = [getattr(ds, "sample_rate", None) for ds in datasets]
        combined_sample_rate = _merge_sample_rates(sample_rates)

        # Merge output_take_and_give from original mappings
        combined_otag = _merge_output_take_and_give(original_otags)

        # Merge DatasetInfo
        dataset_infos = [ds.info for ds in datasets]
        merged_info = _merge_dataset_info(dataset_infos)

        return (
            concatenated_backend,
            merged_info,
            datasets,
            combined_sample_rate,
            combined_otag,
        )

    finally:
        # Restore original output_take_and_give mappings to source datasets
        for dataset, original_otag in zip(datasets, original_otags, strict=False):
            dataset.output_take_and_give = original_otag


@register_dataset
class ConcatenatedDataset(Dataset):
    """A dataset created by concatenating multiple datasets.

    This dataset maintains references to the original datasets to enable
    proper audio loading and other dataset-specific functionality.

    Parameters
    ----------
    datasets : list[Dataset]
        List of datasets to concatenate
    merge_level : {"hard", "overlap", "soft"}, default="soft"
        Strategy for handling different columns
        - "hard": All columns must match exactly across all datasets
        - "overlap": Keep only common columns across all datasets
        - "soft": Keep all columns from all datasets (fill missing with NaN)
    collision_policy : {"raise", "suffix", "source-only", "concat-first"}, default="concat-first"
        Policy for handling column name collisions:
        - "raise": Raise an error on collision of any column names
        - "suffix": Append '_concat' to colliding column names in the concatenated Backend
        - "source-only": Keep only columns from source datasets, this discards any transformations
        - "concat-first": In case of collision, keep the columns from the concatenated Backend

    Examples
    --------
    >>> from esp_data.datasets import InsectSet459, BirdSet
    >>> from esp_data.concat import concatenate_datasets
    >>> dataset1 = InsectSet459(split="validation")
    >>> dataset2 = BirdSet(split="HSN-test")
    >>> ds = ConcatenatedDataset([dataset1, dataset2], merge_level="soft")
    >>> assert len(ds) > 0, "Concatenated dataset should not be empty"
    >>> assert len(ds) == len(dataset1) + len(dataset2), \
        "Concatenated dataset length should match sum of source datasets lengths"
    """

    info = DatasetInfo(
        name="concatenated_dataset",
        owner="ESP Data Team",
        split_paths={"concatenated": "virtual://concatenated_dataset"},
        version="0.1.0",
        description="A dataset created by concatenating multiple datasets.",
        sources=["Multiple datasets"],
        license="CC0-1.0",
    )

    def __init__(
        self,
        datasets: list[Dataset] | None = None,
        merge_level: Literal["hard", "overlap", "soft"] = "soft",
        collision_policy: Literal[
            "raise", "suffix", "source-only", "concat-first"
        ] = "concat-first",
    ) -> None:
        # Validate inputs
        if not datasets:
            raise MergeException("At least one dataset must be provided")

        if not all(isinstance(ds, Dataset) for ds in datasets):
            raise MergeException("All objects must be Dataset instances")

        backend_type = getattr(datasets[0], "_backend_class", None) if datasets else None
        # Make sure all backend types are the same
        if not backend_type or not all(
            getattr(ds, "_backend_class", None) == backend_type for ds in datasets
        ):
            raise ValueError(
                "All datasets must have the same backend type "
                "to be concatenated into a ConcatenatedDataset."
            )

        super().__init__(backend=backend_type.__name__.replace("Backend", "").lower())

        (
            self._data,
            self.info,
            self._source_datasets,
            self.sample_rate,
            output_take_and_give,
        ) = concatenate_datasets(datasets, merge_level=merge_level)

        self.output_take_and_give = output_take_and_give
        self.split = "concatenated"
        self.collision_policy = collision_policy
        if len(self._data) == 0:
            raise ValueError("Concatenated dataset is empty. Check input datasets or merge level.")

    @property
    def columns(self) -> list[str]:
        # Filter out internal tracking columns
        all_columns = self._data.columns
        return [col for col in all_columns if not col.startswith("_source_")]

    @property
    def available_splits(self) -> list[str]:
        return ["concatenated"]

    def _load(self) -> None:
        pass  # Data is already loaded

    @classmethod
    def from_config(
        cls, concat_config: ConcatConfig
    ) -> tuple["ConcatenatedDataset", Dict[str, Any]]:
        """Create a ConcatenatedDataset from a ConcatConfig object.

        Parameters
        ----------
        concat_config : ConcatConfig
            Configuration object specifying the datasets to concatenate
            and how to merge them.

        Returns
        -------
        tuple[ConcatenatedDataset, dict]
            A tuple containing the ConcatenatedDataset instance
            and metadata about transformations applied.
        """
        datasets = [dataset_from_config(cfg)[0] for cfg in concat_config.datasets]
        ds = cls(
            datasets,
            merge_level=concat_config.merge_level,
            collision_policy=concat_config.collision_policy,
        )

        if concat_config.transformations:
            transform_metadata = ds.apply_transformations(concat_config.transformations)
            return ds, transform_metadata

        return ds, {}

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self._data):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self._data)}")

        # Get row as dict from backend
        # This dict has transforms applied at concat level
        row = self._data[idx]

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

        # Use collision policy to determine final item
        if self.collision_policy == "raise":
            # Check for collisions
            for key in row.keys():
                if key in source_item and not key.startswith("_source_"):
                    raise MergeException(
                        f"Column name collision for key '{key}' "
                        f"in concatenated dataset at index {idx}"
                    )
            item = {
                **source_item,
                **{k: v for k, v in row.items() if not k.startswith("_source_")},
            }

        elif self.collision_policy == "suffix":
            item = source_item.copy()
            for key, value in row.items():
                if key.startswith("_source_"):
                    continue
                if key in item:
                    item[f"{key}_concat"] = value
                else:
                    item[key] = value

        elif self.collision_policy == "source-only":
            item = source_item

        elif self.collision_policy == "concat-first":
            item = {}
            for key, value in row.items():
                if key.startswith("_source_"):
                    continue
                item[key] = value
            for key, value in source_item.items():
                if key not in item:
                    item[key] = value

        else:
            raise MergeException(
                f"Invalid collision_policy:{self.collision_policy}. "
                f"Must be 'raise', 'suffix', 'source-only', or 'concat-first'."
            )

        # Apply the concatenated dataset's output_take_and_give mapping
        if self.output_take_and_give:
            mapped_item = {}
            for key, value in self.output_take_and_give.items():
                if key in item:
                    mapped_item[value] = item[key]
            return mapped_item

        return item

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for idx in range(len(self)):
            yield self[idx]

    def __str__(self) -> str:
        return (
            f"{self.info.name} (v{self.info.version})\n"
            f"Description: {self.info.description}\n"
            f"Length: {len(self)}\n"
            f"Columns: {', '.join(self.columns)}\n"
            f"Source datasets: {len(self._source_datasets)}"
        )
