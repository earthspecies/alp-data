from typing import Any

from pydantic import BaseModel, field_validator

from .transforms import RegisteredTransformConfigs


class DatasetConfig(BaseModel):
    dataset_name: str
    transformations: list[RegisteredTransformConfigs] | None = None
    multi_label: bool | None = None
    sample_rate: int | None = None  # Sample rate for audio data
    metrics: list[str] | None = None
    audio_path_col: str | None = None
    output_take_and_give: dict[str, str] | None = None
    split: str = "train"

    @field_validator("transformations", mode="before")
    @classmethod
    def convert_none(cls, v: Any) -> Any:  # noqa: ANN401
        if v in ("None", "none"):
            return None
        return v
