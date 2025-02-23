import io
import json
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import webdataset as wds

from esp_data.file_io.parsers import read_audio_bytes
from esp_data.paths import AnyPath
from esp_data.utils import make_simple_logger

from .utils import create_sharded_dataset, get_batch, get_item_from_dataset

logger = make_simple_logger("audio_dataset")


def prepare_audio_sample_for_sharding(row: pd.Series) -> dict[str, Any]:
    # Read audio file
    audio_data, sr = read_audio_bytes(row["file_path"])

    # compute duration
    duration = len(audio_data) / sr

    # Store as bytes in memory
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_data, sr, format="WAV")

    # Write to shard with metadata
    md = row.to_dict()
    md["duration"] = duration
    md["sample_rate"] = sr

    return {"audio.wav": audio_buffer.getvalue(), "metadata.json": json.dumps(md)}


class AudioDataset:
    """Class for building a sharded audio dataset from raw audio files
    and for loading and accessing the dataset.

    Args:
        web_dataset_path (str): Path to the directory where the sharded dataset will be stored or is already stored.
        metadata_df (pd.DataFrame, optional): Optional metadata DataFrame, if not provided will be read from disk. Defaults to None.
        shard_size (int, optional): Number of samples per shard. Defaults to 1000.
        num_workers (int, optional): Number of workers for parallel processing. Defaults to 4.
        batch_size (int, optional): Batch size for processing audio files. Defaults to 100.
        data_root (str): Path to the root directory containing raw audio files. Defaults to ".".
    """

    def __init__(
        self,
        web_dataset_path: str,
        metadata_df: pd.DataFrame = None,
        num_samples_per_shard: int = 1000,
        num_workers: int = 4,
        storage_options: dict = None,
        data_root: str = ".",
    ):
        self.data_root = AnyPath(data_root)
        self.web_dataset_path = AnyPath(web_dataset_path)
        self.num_samples_per_shard = num_samples_per_shard
        self.num_workers = num_workers
        self.metadata_df = metadata_df
        self.ds = None  # Placeholder for dataset
        self.storage_options = storage_options

        # Read metadata if missing
        if self.metadata_df is None:
            try:
                if AnyPath(data_root / "metadata.parquet").exists():
                    self.metadata_df = pd.read_parquet(data_root / "metadata.parquet")
                elif AnyPath(data_root / "metadata.csv").exists():
                    self.metadata_df = pd.read_csv(data_root / "metadata.csv")
                elif AnyPath(data_root / "metadata.json").exists():
                    self.metadata_df = pd.read_json(data_root / "metadata.json")
                else:
                    logger.error("No metadata found. Nothing to process.")
                    return

            except Exception as e:
                logger.error(f"Error reading metadata: {str(e)}")
                return

        self.sample_prep_function = prepare_audio_sample_for_sharding

    def create_sharded_dataset(self):
        """Create the sharded dataset from the metadata and audio files"""
        self.metadata_df = create_sharded_dataset(
            self.metadata_df,
            output_path=self.web_dataset_path,
            num_samples_per_shard=self.num_samples_per_shard,
            sample_prep_function=self.sample_prep_function,
            num_workers=self.num_workers,
            storage_options=self.storage_options,
        )

    def load_dataset(self, shuffle_size: int = 1000):
        """Create a pipeline for loading the dataset

        Args:
            path (str): Path to the sharded dataset
            shuffle_size (int, optional): Size of the shuffle buffer. Defaults to 1000.
        """
        self.ds = (
            wds.WebDataset(str(self.web_dataset_path / "shard_{000000..999999}.tar"))
            .shuffle(shuffle_size)
            .decode()
            .to_tuple("audio.wav", "metadata.json")
        )

    def query_metadata(self, shard_path: str, query: str):
        """Query the metadata using Parquet"""
        metadata = pq.read_table(AnyPath(shard_path) / "metadata.parquet")
        return metadata.filter(query)

    def get_sample_shard(self, sample_id: str) -> str:
        metadata = self.query_metadata(f"id == {sample_id}")
        return metadata["shard_path"][0]

    # Example query to get all samples in a specific shard
    def get_shard_samples(self, shard_id: int) -> pd.DataFrame:
        metadata = self.query_metadata("shard_id == shard_id")
        return metadata

    def _data_processor(self, data: dict) -> tuple[np.ndarray, dict]:
        audio = data["audio.wav"]
        metadata = json.loads(data["metadata.json"])
        audio_data, _ = sf.read(io.BytesIO(audio))
        return audio_data, metadata

    def __getitem__(self, idx: int) -> tuple[np.ndarray, dict]:
        if isinstance(idx, slice):
            return get_batch(
                indices=list(range(idx.start, idx.stop, idx.step)),
                dataset_path=self.web_dataset_path,
                data_processor=self._data_processor,
                metadata_df=self.metadata_df,
            )

        return get_item_from_dataset(
            idx=idx,
            dataset_path=self.web_dataset_path,
            data_processor=self._data_processor,
            metadata_df=self.metadata_df,
        )

    def __len__(self):
        return len(self.metadata_df)
