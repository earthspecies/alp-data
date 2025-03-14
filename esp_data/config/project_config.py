from pydantic import BaseModel

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
LOG_EVERY = 10
WRITER_BATCH_SIZE = 100
DEFAULT_FLOAT_TYPE = "float32"  # TODO: DISCUSS THIS!


### WEB DATASET LOADER PARAMS ###
class WebDsLoaderConfig(BaseModel):
    load_metadata: bool = True
    metadata_df: str = None
    file_pattern: str = "shard_*.tar"
    storage_options: str = None
    metadata_path: str = None
    data_processor: str = None
    shuffle_size: int = None
    seed: int = 0
    shard_shuffle: bool = False
    shard_shuffle_size: int = 1000
    batch_size: int = None
    batch_collate_fn: str = None
    split_by_worker: bool = False


webds_loader_config = WebDsLoaderConfig()
