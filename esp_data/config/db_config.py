"""Pydantic models for ESP data configuration"""

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
)
from typing_extensions import Annotated

from ..utils import increment_version, utc_now, validate_json_str, validate_path_exists, validate_version


class LicenseEnum(str, Enum):
    """Enum for common open source licenses"""

    CC_BY_4_0 = "CC-BY-4.0"
    CC_BY_SA_4_0 = "CC-BY-SA-4.0"
    CC_BY_NC_4_0 = "CC-BY-NC-4.0"
    CC_BY_NC_SA_4_0 = "CC-BY-NC-SA-4.0"
    CC_BY_NC_ND_4_0 = "CC-BY-NC-ND-4.0"


class DataSample(BaseModel):
    """A Pydantic model for a data sample configuration. A data sample is a single
    row / record in a dataset.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, str_strip_whitespace=True)

    # required params
    # dataset_name is included in sample because some datasets may be aggregated from multiple sources
    # e.g. AnimalSpeak or NatureLM dataset.
    dataset_name: str = Field(min_length=1, description="Name of the dataset. e.g. 'Xeno-canto'")

    creator: str = Field(min_length=1, description="Creator of the dataset")

    metadata: Annotated[str, BeforeValidator(validate_json_str)] = Field(description="JSON metadata string")

    version: str = Field(description="Version number following semantic versioning")

    # optional or auto-generated params
    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique identifier, will be auto-generated if None"
    )

    created_at: datetime = Field(
        default_factory=utc_now,
        description="Datetime of creation in UTC timezone, will be auto-generated if None",
    )

    derived_from: Optional[str | list[str]] = Field(
        default=None, description="ID of the parent sample if this is derived, maybe a list of IDs if multiple parents"
    )

    license: Optional[str] = Field(default=None, description="License for the data sample, if applicable")

    # Additional validators
    @field_validator("id", mode="after")
    @classmethod
    def validate_id(cls, v: str) -> str:
        try:
            UUID(v)
            return v
        except ValueError:
            raise ValueError("Invalid UUID format")

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, v: str) -> str:
        # raises ValueError if format not like "0.0.0"
        return validate_version(v)

    @field_validator("derived_from", mode="after")
    @classmethod
    def validate_derived_from(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                if isinstance(v, list):
                    for item in v:
                        UUID(item)
                else:
                    UUID(v)

            except ValueError:
                raise ValueError("derived_from must be a valid UUID if provided")
        return v

    @field_validator("license", mode="before")
    @classmethod
    def validate_license(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            if v not in LicenseEnum.__members__:
                raise ValueError(f"License must be one of: {', '.join(LicenseEnum.__members__)}")
        return v

    # Helper methods
    def created_at_datetime(self) -> datetime:
        """Return created_at timestamp as a datetime object"""
        return datetime.fromtimestamp(self.created_at, tz=timezone.utc)

    def created_at_isoformat(self) -> str:
        return self.created_at_datetime().isoformat()

    def get_metadata_dict(self) -> dict:
        """Return metadata as a Python dictionary"""
        return json.loads(self.metadata)

    def update_metadata(self, new_metadata: dict) -> None:
        """Update metadata with new dictionary"""
        self.metadata = json.dumps(new_metadata)

    def increment_version(self, mode: str = "patch") -> None:
        """Increment the version number following semantic versioning"""
        self.version = increment_version(self.version, mode)

    def to_dict(self) -> dict:
        """Convert the data sample to a dictionary"""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert the data sample to a JSON string"""
        return self.model_dump_json()


class TextDataSample(DataSample):
    """A Pydantic model for a text data sample configuration. A text data sample is a single
    row / record in a dataset that contains text data.
    """

    # required params
    text: str = Field(min_length=1, description="Text data represented by this sample")


class AudioDataSample(DataSample):
    """A Pydantic model for an audio data sample configuration. An audio data sample is a single
    row / record in a dataset that contains audio data.
    """

    # required params
    audio_path: Annotated[str, BeforeValidator(validate_path_exists)] = Field(
        description="Path to audio file represented by this sample, maybe a local path or a cloud path"
    )

    label: str = Field(min_length=1, description="Label for the audio data")

    # an audio data sample could optionally include a list of floats representing the audio data
    audio: Optional[list[float]] = Field(
        default=None, description="List of floats representing the audio data, if available"
    )


class ImageDataSample(DataSample):
    """A Pydantic model for an image data sample configuration. An image data sample is a single
    row / record in a dataset that contains image data.
    """

    # required params
    image_path: Annotated[str, BeforeValidator(validate_path_exists)] = Field(
        description="Path to image file represented by this sample, maybe a local path or a cloud path"
    )

    label: str = Field(min_length=1, description="Label for the image data")


class NatureLMDataSample(DataSample):
    """A Pydantic model for an audio-text data sample configuration. An audio-text data sample is a single
    row / record in a dataset that contains both audio and text data.
    """

    # required params
    audio_path: Annotated[str, BeforeValidator(validate_path_exists)] = Field(
        description="Path to audio file represented by this sample, maybe a local path or a cloud path"
    )

    prompt: str = Field(min_length=1, description="Text data represented by this sample")


# Dataset Config
class DatasetConfig(BaseModel):
    """A Pydantic model for a dataset configuration. A dataset configuration defines the
    schema of the dataset and the path to the dataset.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, str_strip_whitespace=True)

    # required params
    dataset_name: str = Field(min_length=1, description="Name of the dataset")

    creator: str = Field(min_length=1, description="Creator of the dataset")

    version: str = Field(min_length=3, description="Version of the dataset, root dataset is 0.0")

    description: str = Field(
        min_length=1,
        description="Description of the dataset, could act as a README, can be markdown format, and include changelog to previous version",
    )

    source: str = Field(
        min_length=1,
        description="Source(s) of the dataset e.g. 'Xeno-canto' or a url to website(s), or multiple sources in a comma-separated list",
    )

    created_at: int = Field(
        default_factory=utc_now,
        description="Datetime of creation in UTC timezone, will be auto-generated if None",
    )
