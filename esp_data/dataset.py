from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, Iterator, Literal

import semver
import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from esp_data.backends import BackendType, get_backend
from esp_data.io import AnyPathT, anypath
from esp_data.transforms import transform_from_config
from esp_data.transforms.registry import RegisteredTransformConfigs


class DatasetConfig(BaseModel):
    """A Pydantic base model for the configuration of a dataset.

    Attributes
    ----------
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

    Examples
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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        extra="allow",  # TODO: better 'forbid' but custom configs for datasets needed
    )

    dataset_name: str

    # The logical type of RegisteredTransformConfigs is list[RegisteredTransformConfigs]
    # but we don't want to use RegisteredTransformConfigs directly here as its value
    # will be bound at definition time. Instead we rely on a field validator for late
    # evaluation of RegisteredTransformConfigs at instantiation time. This is useful for
    # user-registered transformations that can happen at any time.
    transformations: list | None = None

    sample_rate: int | None = None
    output_take_and_give: dict[str, str] | None = None
    split: str = "train"
    data_root: str | None = None
    streaming: bool = False
    backend: BackendType = "polars"

    @field_validator("transformations", mode="before")
    @classmethod
    def convert_none(cls, v: Any) -> Any:  # noqa: ANN401
        if v in ("None", "none"):
            return None
        return v

    @field_validator("transformations", mode="after")
    @classmethod
    def delay_importing_reg(cls, v: Any) -> Any:  # noqa: ANN401
        if v:
            # Import the registry here, i.e. as late as possible to make sure it
            # includes all the user-registered transforms as well.
            from esp_data.transforms.registry import RegisteredTransformConfigs

            # RegisteredTransformConfigs is of type Annotated[...] and we can't use it
            # as a Pydantic type/model directly. We first have to adapt it for a
            # Pydantic:
            adapter = TypeAdapter(RegisteredTransformConfigs)

            validated = []
            for t in v:
                validated.append(adapter.validate_python(t))
            return validated
        else:
            return None


class ConcatConfig(BaseModel):
    """A Pydantic base model for the configuration of a ConcatenatedDataset

    Attributes
    ----------
    datasets : list[DatasetConfig]
        List of DatasetConfig objects to concatenate.

    merge_level : {"hard", "overlap", "soft"}, default="soft"
        Strategy for handling different columns:
        - "hard": All columns must match exactly across all datasets
        - "overlap": Keep only common columns across all datasets
        - "soft": Keep all columns from all datasets (fill missing with NaN)

    collision_policy : {"raise", "suffix", "source-only", "concat-first"},
        default="concat-first"
        Policy for handling column name collisions:
        - "raise": Raise an error on collision of any column names
        - "suffix": Append '_concat' to colliding column names in the concatenated Backend
        - "source-only": Keep only columns from source datasets, this discards any transformations
        - "concat-first": In case of collision, keep the columns from the concatenated Backend

    transformations : list | None, optional
        List of transforms to apply to the concatenated dataset.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        extra="forbid",
    )

    dataset_name: str = "concatenated_dataset"
    datasets: list[DatasetConfig]
    merge_level: Literal["hard", "overlap", "soft"] = "soft"
    collision_policy: Literal["raise", "suffix", "source-only", "concat-first"] = "concat-first"
    transformations: list | None = None

    @field_validator("transformations", mode="before")
    @classmethod
    def convert_none(cls, v: Any) -> Any:  # noqa: ANN401
        if v in ("None", "none"):
            return None
        return v

    @field_validator("transformations", mode="after")
    @classmethod
    def delay_importing_reg(cls, v: Any) -> Any:  # noqa: ANN401
        if v:
            from esp_data.transforms.registry import RegisteredTransformConfigs

            adapter = TypeAdapter(RegisteredTransformConfigs)

            validated = []
            for t in v:
                validated.append(adapter.validate_python(t))
            return validated
        else:
            return None


class DatasetInfo(BaseModel):
    """A Pydantic base model for the info (cfg) of a dataset.

    Attributes
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

    def __init__(
        self,
        output_take_and_give: dict[str, str] = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """A DatasetConfig can be passed to the constructor to, for instance,
        apply transformations to the dataset during instantiation or modify its
        fields of output.

        Parameters
        ----------
        output_take_and_give : dict[str, str], optional
            A dictionary mapping output fields to their corresponding input fields.
        backend : BackendType, optional
            The backend to use for DataFrame operations ("pandas" or "polars"), by default "polars"
        streaming : bool, optional
            Whether to use streaming mode for dataset processing, by default False

        """
        self.output_take_and_give = output_take_and_give
        self._streaming = streaming
        self._backend_class = get_backend(backend)

    @property
    @abstractmethod
    def available_splits(self) -> Sequence[str]:
        """Get the available splits of the dataset.

        Returns
        -------
        Sequence[str]
            A sequence of split names available in the dataset.
        """
        pass

    @property
    @abstractmethod
    def columns(self) -> Sequence[str]:
        """Get the columns of the dataset.

        Returns
        -------
        Sequence[str]
            A sequence of column names in the dataset.
        """
        pass

    @abstractmethod
    def _load(self) -> Sequence[Any] | None:
        """Load one split of the dataset.s

        Returns
        -------
        Sequence[Any]
            The requested split of the dataset.
        """
        pass

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
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Get the iterator over the dataset.

        Returns
        -------
        Iterator[Dict[str, Any]]
            Iterator over samples in the dataset
        """
        pass

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
        pass

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
        pass

    def apply_transformations(
        self, transformations: list[RegisteredTransformConfigs]
    ) -> dict[str, Any]:
        """Apply the given list of transformations to the dataset.

        This method applies each transformation in sequence to the dataset's data.
        The transformations are applied in-place, modifying the dataset's data.

        Parameters
        ----------
        transformations : list[RegisteredTransformConfigs]
            List of transformation configurations to apply to the dataset.

        Returns
        -------
        transform_metadata: dict[str, Any]
            A dictionary containing metadata for each transformation applied.
            The keys are the transformation types, and the values are the metadata
            returned by each transformation.

        Raises
        -------
        RuntimeError
            If the dataset's data is not loaded yet.
            If using pandas backend in streaming mode (not supported).
        """
        if self._data is None:
            raise RuntimeError("No data loaded. Call load() first.")

        transform_metadata = {}
        for cfg in transformations:
            transform = transform_from_config(cfg)
            # Transform operates on the backend directly
            self._data, metadata = transform(self._data)
            transform_metadata[cfg.type] = metadata

        return transform_metadata


# Global registry instance
_dataset_registry: dict[str, type[Dataset]] = {}

# We may also have custom configs for specific datasets
_custom_config_registry: dict[str, type[DatasetConfig]] = {}


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

    Raises
    -------
    ValueError
        If the dataset name is already registered
    """
    name = cls.info.name
    if name in _dataset_registry:
        raise ValueError(f"Dataset '{name}' is already registered.")
    _dataset_registry[name] = cls
    return cls


def register_config(cls: type[DatasetConfig]) -> type[DatasetConfig]:
    """A decorator to register a custom dataset configuration class.

    Parameters
    ----------
    cls : Type[DatasetConfig]
        The dataset configuration class to register

    Returns
    -------
    Type[DatasetConfig]
        The registered dataset configuration class

    Raises
    -------
    ValueError
        If the dataset name is already registered
    """
    # TODO: should we have ClassVar instead ?
    name = cls().dataset_name
    if name in _custom_config_registry:
        raise ValueError(f"Config '{name}' is already registered.")
    _custom_config_registry[name] = cls
    return cls


def list_registered_datasets() -> list[str]:
    """List all registered datasets.

    Returns
    -------
    list[str]
        List of dataset names
    """
    return list(_dataset_registry.keys())


def dataset_class_from_name(dataset_name: str) -> type[Dataset]:
    """Get the dataset class from its name.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset

    Returns
    -------
    Type[Dataset]
        The dataset class

    Raises
    -------
    KeyError
        If the dataset name is not registered
    """
    dataset_class = _dataset_registry.get(dataset_name, None)
    if dataset_class is None:
        raise KeyError(f"Dataset '{dataset_name}' is not registered.")
    return dataset_class


def print_registered_datasets() -> None:
    """Print all registered datasets."""
    for dataset_class in _dataset_registry.values():
        print(dataset_class.info.model_dump_json(indent=2))


def _make_dataset_from_config(dataset_config: DatasetConfig | ConcatConfig) -> Dataset:
    """Create a dataset instance from the given configuration.

    Parameters
    ----------
    dataset_config : DatasetConfig | ConcatConfig
        The configuration for the dataset to create. This can be either a
        DatasetConfig or a ConcatConfig.

    Returns
    -------
    Dataset
        The dataset instance created from the configuration.

    Raises
    ------
    KeyError
        If the dataset is not registered
    """
    _dataset_class = _dataset_registry.get(dataset_config.dataset_name, None)
    if _dataset_class is None:
        raise KeyError(f"Dataset '{dataset_config.dataset_name}' is not registered.")

    return _dataset_class.from_config(dataset_config)


def dataset_from_config(
    dataset_config: DatasetConfig | ConcatConfig | AnyPathT | Path | str,
    key: str | None = None,
) -> tuple[Dataset, dict[str, Any]]:
    """Load a single dataset or a dataset collection from a configuration.

    Parameters
    ----------
    dataset_config : DatasetConfig | ConcatConfig | AnyPathT | Path | str
        The configuration for the dataset. This can be either a DatasetConfig object,
        a ConcatConfig object or instead a path to a YAML file containing the
        configuration.

    Returns
    -------
    A single tuple:
        Dataset
            The requested dataset instance
        transform_metadata : dict[str, Any]
            Metadata about transformations applied, if any. Can be empty.

    Raises
    ------
    ValueError
        If multiple / invalid dataset configurations are found in the provided data
        and no specific key is provided to select one.

    KeyError
        If the specified key does not match any dataset configuration in the data.
    """
    if isinstance(dataset_config, (DatasetConfig, ConcatConfig)):
        # If a DatasetConfig is passed, we can directly create the dataset
        return _make_dataset_from_config(dataset_config)

    if isinstance(dataset_config, (Path, str, AnyPathT)):
        with anypath(dataset_config).open("r") as fp:
            data = yaml.safe_load(fp)

    if key is not None:
        if key not in data:
            raise KeyError(f"Required key '{key}' not found in the provided configuration data.")
        data = data[key]

    if isinstance(data, dict):
        if len(data) == 1 and ("dataset" in data or "concat" in data):
            if "concat" in data and "dataset" in data:
                raise ValueError("Configuration cannot contain both 'concat' and 'dataset' keys.")

            if "dataset" in data:
                cfg = data["dataset"]
                cfg_class = _custom_config_registry.get(cfg["dataset_name"], DatasetConfig)
                return _make_dataset_from_config(cfg_class.model_validate(cfg))

            elif "concat" in data:
                cfg = data["concat"]
                return _make_dataset_from_config(ConcatConfig.model_validate(cfg))

            else:
                raise ValueError("Configuration must contain either 'dataset' or 'concat' key.")
        else:
            raise ValueError(
                "Multiple / Invalid dataset configurations found. "
                "Please provide a specific key to select one."
            )

    raise ValueError("""Invalid configuration format.
    Your configuration must either be:
    1. A DatasetConfig represented as the value of a dict with a single 'dataset' key
    2. A ConcatConfig represented as the value of a dict with a single 'concat' key
    """)
