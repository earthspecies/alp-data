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
WEBDS_DEFAULT_CFG = dict(
    load_metadata=True,
    metadata_df=None,
    file_pattern="shard_*.tar",
    storage_options=None,
    metadata_path=None,
    data_processor=None,
    shuffle_size=1000,
    seed=0,
    shard_shuffle=True,
    shard_shuffle_size=1000,
    batch_size=None,
    batch_collate_fn=None,
    split_by_worker=False,
)
