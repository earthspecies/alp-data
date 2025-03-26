from typing import Callable, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

REQUIRED_DATASAMPLE_FIELDS = [
    "id",
    "source_dataset",
    "metadata",
]


REQUIRED_DATASET_FIELDS = [
    "name",
    "creator",
    "version",
    "description",
    "sources",
]


### SHARD  CREATOR PARAMS ###
class ShardCreatorConfig(BaseModel):
    num_samples_per_shard: int = 1000
    num_workers: int = 1
    pyarrow_shard_type: Literal["arrow", "parquet"] = "parquet"
    shard_name: str = "shard"
    log_every: int = 100
    pyarrow_writer_batch_size: int = 100
    pyarrow_default_float_type: str = "float32"  # TODO: DISCUSS THIS!
    error_handling: Literal["raise", "warn", "ignore"] = "warn"
    shard_type: Literal["webdataset", "arrow", "parquet", "hf"] = "webdataset"


default_shard_creator_cfg = ShardCreatorConfig()


### WEB DATASET LOADER PARAMS ###
class WebDsLoaderConfig(BaseModel):
    """Configuration for loading a WebDataset"""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, str_strip_whitespace=True)

    load_metadata: bool = False
    metadata_df: Optional[pd.DataFrame] = None
    file_pattern: str = "shard_*.tar"
    storage_options: Optional[dict] = None
    metadata_path: Optional[str] = None
    data_processor: Optional[Callable] = None
    shuffle_size: Optional[int] = None
    seed: Union[bool, int, None] = 0  # True means random seed, None means no shuffling
    shard_shuffle: bool = False
    shard_shuffle_size: int = 1000
    batch_size: Optional[int] = None
    batch_collate_fn: Optional[Callable] = None
    split_by_worker: bool = False

    @model_validator(mode="after")
    def check_args(self) -> Self:
        if self.batch_size and self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        if self.batch_size and not self.batch_collate_fn:
            raise ValueError("batch_collate_fn must be provided if batch_size is set")

        if self.metadata_df is not None:
            assert not self.metadata_df.empty, "metadata_df cannot be empty"

        if self.shuffle_size and self.shuffle_size <= 0:
            raise ValueError("shuffle_size must be greater than 0")

        if self.seed and self.seed <= 0:
            raise ValueError("seed must be greater than 0")

        if "tar" not in self.file_pattern:
            raise ValueError("file_pattern must contain 'tar'")

        return self


default_webds_loader_cfg = WebDsLoaderConfig()
