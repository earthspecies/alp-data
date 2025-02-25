import io
import json
from typing import Any, Callable

import numpy as np
import pandas as pd
import soundfile as sf
import webdataset as wds

import esp_data.file_io.functional as F
from esp_data.file_io.parsers import read_audio_bytes, read_audio_bytes_from_path
from esp_data.paths import AnyPath
from esp_data.utils import make_simple_logger

from .utils import create_sharded_dataset, get_batch, get_item_from_dataset

logger = make_simple_logger("audio_dataset")


def prepare_audio_sample_for_sharding(row: pd.Series) -> dict[str, Any]:
    # Read audio file
    audio_data, sr = read_audio_bytes_from_path(row["file_path"])

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
        metadata_path (str): Path to the metadata file, if different from web_dataset_path. Defaults to None.
        sample_prep_function (Callable, optional): Function to prepare a sample for sharding. Defaults to None.
        shuffle_size (int, optional): Size of the shuffle buffer. Defaults to 1000.
        storage_options (dict, optional): Storage options for reading and writing files from buckets. Defaults to None.

    """

    def __init__(
        self,
        web_dataset_path: str | AnyPath,
        metadata_df: pd.DataFrame = None,
        num_samples_per_shard: int = 1000,
        num_workers: int = 4,
        storage_options: dict = None,
        metadata_path: str | None = None,
        sample_prep_function: Callable | None = None,
        shuffle_size: int = 1000,
    ):
        self.metadata_path = AnyPath(metadata_path if metadata_path is not None else web_dataset_path)
        self.web_dataset_path = AnyPath(web_dataset_path)
        self.num_samples_per_shard = num_samples_per_shard
        self.num_workers = num_workers
        self.metadata_df = metadata_df
        self.storage_options = storage_options
        self.shuffle_size = shuffle_size

        # Read metadata if missing
        if self.metadata_df is None:
            if AnyPath(metadata_path / "metadata.parquet").exists():
                self.metadata_df = pd.read_parquet(metadata_path / "metadata.parquet")
            elif AnyPath(metadata_path / "metadata.csv").exists():
                self.metadata_df = pd.read_csv(metadata_path / "metadata.csv")
            elif AnyPath(metadata_path / "metadata.json").exists():
                self.metadata_df = pd.read_json(metadata_path / "metadata.json")
            else:
                logger.warning(
                    "No metadata found. Won't be able to create a sharded dataset or index directly into the data"
                )

        self.sample_prep_function = sample_prep_function

        # load dataset if available
        shard_files = F.list_files(self.web_dataset_path, pattern="shard_*.tar")
        if len(shard_files) > 0:
            self._load_dataset(shuffle_size=self.shuffle_size)

    def create_sharded_dataset(self):
        """Create the sharded dataset from the metadata and audio files"""
        if self.sample_prep_function is None:
            logger.error("No sample prep function provided. Cannot create sharded dataset.")
            return

        self.metadata_df = create_sharded_dataset(
            self.metadata_df,
            output_path=self.web_dataset_path,
            num_samples_per_shard=self.num_samples_per_shard,
            sample_prep_function=self.sample_prep_function,
            num_workers=self.num_workers,
            storage_options=self.storage_options,
        )

    def _load_dataset(self, shuffle_size: int = 1000, **webdataset_kwargs):
        """Create a pipeline for loading the dataset

        Args:
            path (str): Path to the sharded dataset
            shuffle_size (int, optional): Size of the shuffle buffer. Defaults to 1000.
            **webdataset_kwargs: Additional arguments for WebDataset. See here:

        """
        self.ds = (
            wds.WebDataset(str(self.web_dataset_path / "shard_{000000..999999}.tar"), **webdataset_kwargs)
            .shuffle(shuffle_size)
            .map(self._data_processor)
        )

    @classmethod
    def from_path(cls, path: str | AnyPath, **kwargs):
        return cls(web_dataset_path=path, **kwargs)

    def get_sample_shard_path(self, idx: int) -> str:
        return self.metadata_df["shard_path"].iloc[idx]

    def _data_processor(self, data: dict) -> tuple[np.ndarray, dict]:
        audio = data["audio.wav"]
        metadata = json.loads(data["metadata.json"])
        audio_data, _ = read_audio_bytes(audio, "wav")
        return audio_data, metadata

    def __getitem__(self, idx: int) -> tuple[np.ndarray, dict]:
        if self.metadata_df is None:
            raise ValueError("No metadata found. Cannot access individual samples.")

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
        return len(self.metadata_df) or None

    def __iter__(self):
        if self.ds is None:
            self._load_dataset(shuffle_size=self.shuffle_size)

        return iter(self.ds)
