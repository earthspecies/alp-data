from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Dict, Iterator, Optional

import semver
from pydantic import BaseModel, ConfigDict, Field, field_validator

from esp_data.io import anypath
from esp_data.transforms import RegisteredTransformConfigs, transform_from_config


class DatasetConfig(BaseModel):
    """A Pydantic base model for the configuration of a dataset.

    Parameters
    ---------
    dataset_name : str
        Name of the dataset, must match a registered dataset class
    transformations : list[RegisteredTransformConfigs] | None
        List of transformations to apply to the dataset.
        If None, no transformations are applied.
    sample_rate : int | None
        Target sample rate for the audio data. If None, the default is 16000.
    output_take_and_give : dict[str, str] | None
        A dictionary mapping output fields to their corresponding input fields.
        If None, no output mapping is applied. For example, if the dataset has a field
        "species_scientific" and you want to map it to "species", you can set
        output_take_and_give={"species_scientific": "species"}.
    split : str
        The split of the dataset to load. Defaults to "train".
    data_root : Optional[str]
        The root directory for the dataset. This is optionally appended to the
        path item of a sample in the dataset.
        If None, the default is the parent directory of the split path.

    Example
    -------
    >>> dataset_config = DatasetConfig(
    ...    dataset_name="barkley_canyon",
    ...    transformations=[
    ...        {
    ...            "type": "label_from_feature",
    ...            "feature": "species_scientific",
    ...            "output_feature": "label",
    ...        }
    ...    ])

    """

    dataset_name: str
    transformations: list[RegisteredTransformConfigs] | None = None
    sample_rate: int | None = None
    output_take_and_give: dict[str, str] | None = None
    split: str = "train"
    data_root: Optional[str] = None

    @field_validator("transformations", mode="before")
    @classmethod
    def convert_none(cls, v: Any) -> Any:  # noqa: ANN401
        if v in ("None", "none"):
            return None
        return v


class DatasetInfo(BaseModel):
    """A Pydantic base model for the info (cfg) of a dataset.

    Parameters
    ---------
    name : str
        Name of the dataset
    owner : str | list[str]
        ESP team owner(s) of the dataset
    split_paths : dict[str, str]
        Paths to the dataset splits. The keys are the split names
        and the values are the paths to the splits. The paths can be
    version : str
        Version of the dataset, root dataset is 0.0
    description : str
        Description of the dataset, could act as a README, preferably in markdown format
    sources : list[str] | str
        Source(s) of the dataset e.g. 'Xeno-canto' or a url to website(s),
        or multiple sources in a comma-separated list
    license : Optional[str]
        License for the dataset, if applicable
    changelog : Optional[str]
        Changelog from previous version
    **kwargs : Any (optional)
        Not validated, but can be used to pass additional information

    Examples
    --------
    >>> info = DatasetInfo(
    ...     name="animalspeak",
    ...     owner="marius; masato",
    ...     split_paths={
    ...         "train": "gs://animalspeak2/splits/v1/animalspeak_train_v1.3.csv",
    ...         "validation": "gs://animalspeak2/splits/v1/animalspeak_eval_v1.3.csv",
    ...     },
    ...     version="0.1.0",
    ...     description="AnimalSpeak dataset",
    ...     sources=["Xeno-canto", "iNaturalist", "Watkins"],
    ...     license="unknown",
    ...     changelog="Initial version",
    ... )
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        extra="allow",
    )

    # required params
    name: str = Field(min_length=1, description="Name of the dataset")

    owner: str = Field(min_length=1, description="ESP team owner(s) of the dataset")

    split_paths: dict = Field(
        description="""Paths to the dataset splits. The keys are the split names
        and the values are the paths to the splits""",
    )

    version: str = Field(min_length=5, description="Version of the dataset")

    description: str = Field(
        min_length=1,
        description="""Description of the dataset, could act as a README,
        preferably in markdown format, and include changelog to previous version""",
    )

    sources: list[str] | str = Field(
        min_length=1,
        description="""Source(s) of the dataset e.g. 'Xeno-canto' or a url to
        website(s) or multiple sources in a comma-separated list""",
    )

    license: str = Field(
        default_factory=lambda: "unknown",
        description="License for the dataset, if applicable",
    )

    changelog: str = Field(
        default_factory=lambda: "", description="Changelog from previous version"
    )

    @field_validator("split_paths", mode="after")
    @classmethod
    def validate_split_exists(cls, v: dict) -> str:
        """Validate that the split path exists in cloud storage or locally

        Parameters
        ---------
        v : dict[str, str]
            The locations to validate

        Returns
        -------
        dict[str, str]
            The validated locations

        Raises
        ------
        ValueError
            If the location does not exist in cloud storage or locally
        ValueError
            If the location is a directory and is empty
        """
        if not v:
            raise ValueError("Split paths cannot be empty.")
        for _, value in v.items():
            path = anypath(value)
            if not path.exists():
                raise ValueError(f"Local path {value} does not exist.")

            # if location is directory, check that it is not empty
            if path.is_dir() and not any(path.iterdir()):
                raise ValueError(f"Directory {value} is empty.")

        return v

    @field_validator("version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        """Validates that the version follows semantic versioning (MAJOR.MINOR.PATCH)
        using the semver package.

        Parameters
        ---------
        v : str
            The version string to validate

        Returns
        -------
        str
            The validated version string

        Raises
        ------
        ValueError
            If the version does not follow semantic versioning
        """
        try:
            semver.VersionInfo.parse(v)
        except ValueError as e:
            raise ValueError(f"""Version '{v}' does not follow semantic versioning
                            (MAJOR.MINOR.PATCH).
                    Error: {str(e)}. See https://semver.org/ for details.""") from e
        return v


class Dataset(ABC):
    """Abstract base class defining the interface for ESP datasets.
    Any new dataset should inherit from this class to be added to the registry
    of available ESP datasets.

    Attributes
    ----------
    info : DatasetInfo
        Required attribute containing metadata about the dataset.
        Must be defined by all implementing classes.

    Methods
    -------
    _load() -> pd.DataFrame
        Required method to load a specific split of the dataset.
    __len__() -> int
        Required method to return the number of samples in the dataset.
    __iter__() -> Iterator[Dict[str, Any]]
        Required method to iterate over the samples in the dataset.
    __getitem__(idx: int) -> Dict[str, Any]
        Required method to get a specific sample from the dataset.
    """

    info: DatasetInfo

    def __init__(self, output_take_and_give: dict[str, str] = None) -> None:
        """A DatasetConfig can be passed to the constructor to, for instance,
        apply transformations to the dataset during instantiation or modify its
        fields of output.

        Parameters
        ----------
        output_take_and_give : dict[str, str], optional
            A dictionary mapping output fields to their corresponding input fields.

        """
        self.output_take_and_give = output_take_and_give

    @property
    def available_splits(self) -> Sequence[str]:
        """Get the available splits of the dataset.

        Returns
        -------
        Sequence[str]
            A sequence of split names available in the dataset.
        """
        raise NotImplementedError

    @property
    def columns(self) -> Sequence[str]:
        """Get the columns of the dataset.

        Returns
        -------
        Sequence[str]
            A sequence of column names in the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def _load(self) -> Optional[Sequence[Any]]:
        """Load one split of the dataset.s

        Returns
        -------
        Sequence[Any]
            The requested split of the dataset.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        dataset_config: DatasetConfig,
    ) -> tuple["Dataset", dict[str, Any]]:
        """Create a dataset instance from a configuration.

        Parameters
        ----------
        dataset_config : DatasetInfo
            The configuration for the dataset.

        Returns
        -------
        Dataset
            The dataset instance.
        dict[str, Any]
            Metadata about transformations applied, if any. Can be empty.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset
        """
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Get the iterator over the dataset.

        Returns
        -------
        Iterator[Dict[str, Any]]
            Iterator over samples in the dataset
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a specific sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to get

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the sample data

        Raises
        ------
        IndexError
            If the index is out of bounds
        """
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the dataset.

        This method should provide a human-readable description of the dataset,
        typically including its name, version, and basic statistics.

        Returns
        -------
        str
            A string representation of the dataset
        """
        raise NotImplementedError

    def apply_transformations(self, transformations: list[RegisteredTransformConfigs]) -> list[Any]:
        """Apply the given list of transformations to the dataset.

        This method applies each transformation in sequence to the dataset's data.
        The transformations are applied in-place, modifying the dataset's data.

        Parameters
        ----------
        transformations : list[RegisteredTransformConfigs]
            List of transformation configurations to apply to the dataset.

        Returns
        -------
        dict[str, Any]
            A dictionary containing metadata for each transformation applied.
            The keys are the transformation types, and the values are the metadata
            returned by each transformation.

        Raises
        -------
        RuntimeError
            If the dataset's data is not loaded yet.
        """
        if self._data is None:
            raise RuntimeError("No data loaded. Call load() first.")

        transform_metadata = {}
        for cfg in transformations:
            transform = transform_from_config(cfg)
            self._data, metadata = transform(self._data)
            transform_metadata[cfg.type] = metadata

            # TODO (milad): what about metadata?
        return transform_metadata


# Global registry instance
_dataset_registry: dict[str, type[Dataset]] = {}


def register_dataset(cls: type[Dataset]) -> type[Dataset]:
    """A decorator to register a dataset class.

    Parameters
    ----------
    cls : Type[Dataset]
        The dataset class to register

    Returns
    -------
    Type[Dataset]
        The registered dataset class
    """
    name = cls.info.name
    _dataset_registry[name] = cls
    return cls


def list_registered_datasets() -> list[str]:
    """List all registered datasets.

    Returns
    -------
    list[str]
        List of dataset names
    """
    return list(_dataset_registry.keys())


def print_registered_datasets() -> None:
    """Print all registered datasets."""
    for dataset_class in _dataset_registry.values():
        print(dataset_class.info.model_dump_json(indent=2))


def dataset_from_config(
    dataset_config: DatasetConfig,
) -> tuple[Dataset, dict[str, Any]]:
    """Load a dataset from a configuration.

    Parameters
    ----------
    dataset_config : DatasetConfig
        The configuration for the dataset.
    transform_metadata : dict[str, Any]
        Metadata about transformations applied, if any. Can be empty.

    Returns
    -------
    Dataset
        The requested dataset instance

    Raises
    ------
    KeyError
        If the dataset is not registered
    """
    _dataset_class = _dataset_registry.get(dataset_config.dataset_name, None)
    if _dataset_class is None:
        raise KeyError(f"Dataset '{dataset_config.dataset_name}' is not registered.")
    return _dataset_class.from_config(dataset_config)
