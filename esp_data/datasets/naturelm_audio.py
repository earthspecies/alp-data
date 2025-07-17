"""NatureLM-audio v1 training data"""

from typing import Any, Callable, Iterator

import librosa

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.dataset_utils import audio_decoder, load_webdataset
from esp_data.io import AnyPathT


@register_dataset
class NatureLMAudio(Dataset):
    """NatureLM-audio dataset
    (Tar file version)

    Description
    -----------
    NatureLM-audio-training is a large and diverse audio-language dataset designed
    for training bioacoustic models that can generate a natural language answer to
    a natural language query on a reference bioacoustic audio recording. For example,
    for an in-the-wild audio recording of a bird species, a relevant query might be
    "What is the common name for the focal species in the audio?" to which an
    audio-language model trained on this dataset may respond with "Common yellowthroat".

    It consists of over 26 million audio-text pairs derived from diverse sources
    including animal vocalizations, insects, human speech, music and environmental
    sounds.

    References
    ----------
    HuggingFace dataset: https://huggingface.co/datasets/EarthSpeciesProject/NatureLM-audio-training
    Paper: https://arxiv.org/pdf/2411.07186
    """

    info = DatasetInfo(
        name="naturelm_audio",
        owner="david; marius; masato; gagan; milad",
        split_paths={
            "train": "gs://esp-ml-datasets/naturelm/processed/v0.1.0/tar/train",
        },
        version="0.1.0",
        description="NatureLM-audio dataset",
        sources=[
            "Xeno-canto",
            "iNaturalist",
            "Watkins",
            "WavCaps",
            "AudioCaps",
            "AnimalSoundArchive",
            "LibriSpeechTTS",
            "NSynth",
            "Clotho",
            "SapsuckerWoods",
            "BarkleyCanyon",
            "UrbanSound",
        ],
        license="CC BY",
    )

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT = None,
        within_shard_shuffle: bool = False,
        within_shard_shuffle_size: int = 1000,
        across_shard_shuffle: bool = False,
        across_shard_shuffle_size: int = 1000,
        seed: int | None = 42,
        split_by_worker: bool = False,
        batch_collate_fn: Callable = None,
        batch_size: int | None = None,
    ) -> None:
        """Initialize the NatureLMAudio dataset.

        Parameters
        ----------
        split: str
            The dataset split to load (default: "train")
        output_take_and_give : dict[str, str]
            A dictionary mapping the original column names to the new column names.
            It acts as a filter as well.
        sample_rate : int
            The sample rate to which audio files should be resampled.
        data_root : str | AnyPathT, optional
            The root directory for the dataset. This is optionally appended to the
            path item of a sample in the dataset.
            If None, the default is the parent directory of the split path.
        within_shard_shuffle : bool
            Whether to shuffle the samples within each shard.
        within_shard_shuffle_size : int
            The size of the shuffle buffer for within-shard shuffling.
        across_shard_shuffle : bool
            Whether to shuffle the shards.
        across_shard_shuffle_size : int
            The size of the shuffle buffer for across-shard shuffling.
        seed : int | None
            Seed for shuffling. If None, no shuffling is applied.
            Defaults to 42.
        split_by_worker : bool
            Whether to split the dataset by worker.
            This is useful for distributed training.
        batch_collate_fn : Callable, optional
            Function to collate the batch into fixed size tensor.
            If None, batching will not work.
        batch_size : int | None
            The batch size for processing audio files. If None, no batching is applied.
        """
        super().__init__(
            output_take_and_give=output_take_and_give,
        )
        self.split = split
        self.sample_rate = sample_rate
        self.data_root = data_root
        if self.data_root is None:
            self.data_root = self.info.split_paths[split]

        self._columns = None

        self._data = load_webdataset(
            self.data_root,
            data_processor=audio_decoder,
            shuffle_size=within_shard_shuffle_size if within_shard_shuffle else None,
            shard_shuffle=across_shard_shuffle,
            shard_shuffle_size=across_shard_shuffle_size,
            split_by_worker=split_by_worker,
            batch_collate_fn=batch_collate_fn,
            batch_size=batch_size,
            seed=seed,
        )

    @property
    def columns(self) -> list[str]:
        """Return the columns of the dataset."""
        # iter once over the webdataset to get the columns
        if not self._columns:
            self._columns = list(self.webdataset[0].keys())

    @property
    def available_splits(self) -> list[str]:
        """Return the available splits of the dataset."""
        return list(self.info.split_paths.keys())

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["NatureLMAudio", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary containing dataset parametesf

        Returns
        -------
        tuple[Dataset, dict[str, Any]]
            A tuple containing the dataset instance and metadata.
            If the dataset_config contains transformations, they will be applied
            and the metadata will be returned as dict, otherwise an empty dict.

        Raises
        -------
        LookupError
            If the specified split is not available in the dataset info.
        """
        cfg = dataset_config.model_dump(exclude=("dataset_name", "transformations"))

        split = cfg.get("split", None)
        if not split or split not in cls.info.split_paths:
            raise LookupError(
                f"Invalid split '{split}'."
                f"Available splits: {', '.join(cls.info.split_paths.keys())}"
            )

        ds = cls(
            split=split,
            output_take_and_give=cfg.get("output_take_and_give", None),
            data_root=cfg.get("data_root"),
            sample_rate=cfg.get("sample_rate", None),
            within_shard_shuffle=cfg.get("within_shard_shuffle", False),
            within_shard_shuffle_size=cfg.get("within_shard_shuffle_size", 1000),
            across_shard_shuffle=cfg.get("across_shard_shuffle", False),
            across_shard_shuffle_size=cfg.get("across_shard_shuffle_size", 1000),
            seed=cfg.get("seed", 42),
            split_by_worker=cfg.get("split_by_worker", False),
            batch_collate_fn=cfg.get("batch_collate_fn", None),
            batch_size=cfg.get("batch_size", None),
        )

        if dataset_config.transformations:
            raise NotImplementedError(
                "Transformations are not supported for NatureLMAudio (tar) dataset."
            )
        return ds, {}

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError(
            "Length is not defined for NatureLMAudio tar dataset. because it is an iterable dataset"
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample from the dataset by index."""
        raise NotImplementedError(
            "Indexing is not supported for NatureLMAudio tar dataset. "
            "because it is an iterable dataset"
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over the dataset.

        Yields
        -------
        dict[str, Any]: A dictionary containing the audio sample and its metadata.
        """
        for row in self._data:
            # Resample audio if needed
            sr = row["sample_rate"]
            if self.sample_rate is not None and sr != self.sample_rate:
                audio = librosa.resample(
                    y=row["audio"],
                    orig_sr=sr,
                    target_sr=self.sample_rate,
                    scale=True,
                    res_type="kaiser_best",
                )
                row["audio"] = audio

            # If output_take_and_give is defined, filter the keys
            if self.output_take_and_give:
                item = {}
                for key, value in self.output_take_and_give.items():
                    item[value] = row[key]
            else:
                item = row

            yield item

    def __str__(self) -> str:
        """Return a string representation of the dataset.

        Returns
        -------
        str
            A string representation of the dataset including its name, version,
            and basic statistics if data is loaded.
        """
        base_info = f"{self.info.name} (v{self.info.version})"

        return (
            f"{base_info}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
