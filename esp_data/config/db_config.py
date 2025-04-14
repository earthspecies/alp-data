"""Pydantic models for ESP data configuration"""

import json
import os
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from esp_data.io import anypath

from ..utils import (
    increment_version,
    make_id,
    utc_now_str,
    validate_datetime,
    validate_id,
    validate_version,
)


class LicenseEnum(str, Enum):
    """Enum for common open source licenses"""

    CC_BY_4_0 = "CC-BY-4.0"
    CC_BY_SA_4_0 = "CC-BY-SA-4.0"
    CC_BY_NC_4_0 = "CC-BY-NC-4.0"
    CC_BY_NC_SA_4_0 = "CC-BY-NC-SA-4.0"
    CC_BY_NC_ND_4_0 = "CC-BY-NC-ND-4.0"
    UNKNOWN = "unknown"
    MIT = "MIT"
    APACHE_2_0 = "Apache-2.0"


class DataSample(BaseModel):
    """A Pydantic model for a data sample configuration. A data sample is a single
    row / record in a dataset.

    Arguments
    ---------
    source_dataset : str
        Name of the source dataset. e.g. 'Xeno-canto' or 'esc-50' or 'esc-50v0.1.0'
    metadata: Optional[dict]
        Metadata for the data sample, could be an empty dictionary. Must be JSON serializable.
    id : Optional[str]
        Unique identifier, will be auto-generated if None
    created_at : Optional[str]
        Datetime of creation in UTC timezone, will be auto-generated if None
    derived_from : Optional[str | list[str]]
        ID of the parent sample if this is derived, maybe a list of IDs if multiple parents
    license : Optional[str]
        License for the data sample, if applicable. For e.g. Xeno-canto can
        have per recording licenses. Default is 'unknown'
    version : Optional[str]
        Version number following semantic versioning, can be left empty but then the dataset version must be provided

    Example
    -------
    >>> data = {
    ...     "source_dataset": "test",
    ...     "creator": "test",
    ...     "metadata": {"something": "else"},
    ...     "version": "0.0.0",
    ... }
    >>> sample = DataSample(**data)
    >>> print(sample.source_dataset)

    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    # required params
    # dataset_name is included in sample because some datasets may be aggregated from multiple sources
    # e.g. AnimalSpeak or NatureLM dataset have component datasets like Xeno-canto, esc-50, etc.
    source_dataset: str = Field(
        min_length=1,
        description="Name of the source dataset. e.g. 'Xeno-canto' or 'esc-50' or 'esc-50v0.1.0",
    )

    # optional or auto-generated params
    # DISCUSS: should metadata be optional? Should it be a json string or a dictionary?
    # metadata: Annotated[str, BeforeValidator(validate_json_str)] = Field(description="JSON metadata string")
    metadata: dict = Field(default_factory=lambda: {}, description="Metadata for the data sample")

    id: str = Field(
        default_factory=make_id,
        description="Unique identifier, will be auto-generated if None",
    )

    created_at: str = Field(
        default_factory=utc_now_str,
        description="Datetime of creation in UTC timezone, will be auto-generated if None",
    )

    derived_from: Optional[str | list[str]] = Field(
        default=None,
        description="ID of the parent sample if this is derived, maybe a list of IDs if multiple parents",
    )

    license: Optional[str] = Field(
        default_factory=lambda: LicenseEnum.UNKNOWN,
        description="License for the data sample, if applicable. For e.g. Xeno-canto can have per recording licenses",
    )

    version: Optional[str] = Field(
        default=None,
        description="Version number following versioning, can be left empty but the dataset version must be provided",
    )

    # Additional validators
    @field_validator("id", mode="after")
    @classmethod
    def validate_id(cls, v: str) -> str:
        return validate_id(v)

    @field_validator("metadata", mode="after")
    @classmethod
    def validate_metadata(cls, v: dict) -> dict:
        if not isinstance(v, dict):
            raise ValueError("metadata must be a dictionary")
        # test if metadata is JSON serializable
        try:
            json.dumps(v)
        except TypeError:
            raise ValueError("metadata must be JSON serializable")

        return v

    @field_validator("derived_from", mode="after")
    @classmethod
    def validate_derived_from(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            if isinstance(v, str):
                validate_id(v)
            elif isinstance(v, list):
                for item in v:
                    validate_id(item)
            else:
                raise ValueError("derived_from must be a string or a list of strings")
        return v

    # @field_validator("license", mode="before")
    # @classmethod
    # def validate_license(cls, v: Optional[str]) -> Optional[str]:
    #     if v is not None:
    #         if v not in LicenseEnum.__members__:
    #             raise ValueError(f"License must be one of: {', '.join(LicenseEnum.__members__)}")
    #     return v

    @field_validator("version", mode="after")
    @classmethod
    def validate_version(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return validate_version(v)
        return v

    @field_validator("created_at", mode="after")
    @classmethod
    def validate_created_at(cls, v: str) -> str:
        return validate_datetime(v)

    # Helper methods
    def created_at_timestamp(self) -> int:
        """Returns created_at as a Unix timestamp"""
        # get datetime object
        return datetime.fromisoformat(self.created_at).timestamp()

    def get_metadata_dict(self) -> dict:
        """Returns metadata as a Python dictionary

        Returns
        ------
        dict
            Metadata as a Python dictionary.
        """
        return self.metadata

    def update_metadata(self, new_metadata: dict) -> None:
        """Update metadata with new dictionary"""
        self.metadata = {**self.metadata, **new_metadata}

    def increment_version(self, mode: str = "patch") -> None:
        """Increment the version number following semantic versioning"""
        self.version = increment_version(self.version, mode)

    def copy(self) -> "DataSample":
        """Returns a copy of the data sample

        Returns
        ------
        DataSample
            A copy of the data sample.
        """
        return self.model_copy(deep=True)

    def to_dict(self) -> dict:
        """Convert the data sample to a dictionary.

        Returns
        ------
        dict
            A dictionary representation of the data sample.
        """
        return self.model_dump()

    def to_json(self) -> str:
        """Convert the data sample to a JSON string

        Returns
        ------
        str
            A JSON representation of the data sample.
        """
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, file_path: str | os.PathLike) -> "DataSample":
        """Load data sample from a JSON file

        Returns
        ------
        DataSample
            A data sample loaded from the JSON file.
        """
        with anypath(file_path).open("r") as f:
            data = json.load(f)
        return cls(**data)

    def write_json(self, file_path: str | os.PathLike, indent: int = 2) -> None:
        """Write the data sample to a JSON file"""
        with anypath(file_path).open("w") as f:
            d = self.to_dict()
            json.dump(d, f, indent=indent)

    def __str__(self):
        return f"""# {self.source_dataset}
        ## ID\n\n{self.id}\n\n
        ## Created At\n\n{self.created_at}\n\n
        ## License\n\n{self.license}\n\n
        ## Version\n\n{self.version}\n\n
        ## Metadata\n\n{self.metadata}\n\n"""

    def __repr__(self):
        return f"DataSample(source_dataset={self.source_dataset}, id={self.id}, created_at={self.created_at})"


class TextDataSample(DataSample):
    """A Pydantic model for a text data sample configuration. A text data sample is a single
    row / record in a dataset that contains text data.
    """

    # required params
    text: str = Field(min_length=1, description="Text data represented by this sample")


# Dataset Config
class DatasetConfig(BaseModel):
    """A Pydantic model for a dataset configuration. A dataset configuration defines the
    schema of the dataset and the path to the dataset.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    # required params
    name: str = Field(min_length=1, description="Name of the dataset")

    creator: str = Field(min_length=1, description="Creator of the dataset")

    version: str = Field(min_length=5, description="Version of the dataset, root dataset is 0.0")

    description: str = Field(
        min_length=1,
        description="Description of the dataset like a README, preferably in markdown format, and include changelog.",
    )

    sources: list[str] | str = Field(
        min_length=1,
        description=(
            "Source(s) of the dataset e.g. 'Xeno-canto' or a url to website(s), "
            "or multiple sources in a comma-separated list."
        ),
    )

    # optional or auto-generated params
    created_at: str = Field(
        default_factory=utc_now_str,
        description="Datetime of creation in UTC timezone, will be auto-generated if None",
    )

    license: Optional[str] = Field(
        default_factory=lambda: LicenseEnum.UNKNOWN,
        description="License for the dataset, if applicable",
    )

    changelog: Optional[str] = Field(default_factory=lambda: "", description="Changelog from previous version")

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """

        Returns
        ------
        str
            Version number following semantic versioning.
        """
        # raises ValueError if format not like "0.0.0"
        return validate_version(v)

    def increment_version(self, mode: str = "patch") -> None:
        """Increment the version number following semantic versioning"""
        self.version = increment_version(self.version, mode)

    @classmethod
    def from_json(cls, file_path: str | os.PathLike) -> None:
        """Load data sample from a JSON file

        Returns
        ------
        DataSample
            A data sample loaded from the JSON file.
        """
        with anypath(file_path).open("r") as f:
            data = json.load(f)
        return cls(**data)

    def copy(self) -> "DataSample":
        """Returns a copy of the data sample"""
        return self.model_copy(deep=True)

    def to_dict(self) -> dict:
        """Convert the dataset to a dictionary"""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert the dataset to a JSON string"""
        data = self.to_dict()
        return json.dumps(data, indent=2)

    def write_json(self, file_path: str | os.PathLike) -> None:
        """Write the dataset to a JSON file"""
        with anypath(file_path).open("w") as f:
            d = self.to_dict()
            json.dump(d, f, indent=2)

    def update_changelog(self, new_log: str) -> None:
        """Update changelog with new string"""
        # get current datetime as str
        dt_str = utc_now_str()
        # append new changelog with datetime
        if not self.changelog:
            self.changelog = f"{dt_str}:\n{new_log}"
        else:
            self.changelog += f"\n\n{dt_str}:\n{new_log}"

    def __str__(self) -> str:
        """
        Returns
        ------
        str
            A string representation of the dataset.
        """
        return f"""# {self.name}
        ## Version\n\n{self.version}\n\n
        ## Created At\n\n{self.created_at}\n\n
        ## Creator\n\n{self.creator}\n\n
        ## License\n\n{self.license}\n\n
        ## Description\n\n{self.description}\n\n
        ## Sources\n\n{self.sources}\n\n
        ## Changelog\n\n{self.changelog}\n\n"""

    def generate_readme(self, file_path: str | os.PathLike) -> None:
        """Generate a README file for the dataset"""
        text = str(self)
        with anypath(file_path).open("w") as f:
            f.write(text)

    @classmethod
    def from_skeleton(cls) -> "DatasetConfig":
        """Returns a skeleton dataset configuratio

        Returns
        ------
        DatasetConfig
            A skeleton dataset configuration.
        """
        return cls(
            name="unknown",
            creator="unknown",
            version="0.0.0",
            description="none",
            sources=["none"],
        )
