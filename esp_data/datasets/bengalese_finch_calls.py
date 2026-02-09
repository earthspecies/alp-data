"""Bengalese Finch Calls dataset

This dataset contains individual calls from Bengalese finches. Each row in the
metadata CSV corresponds to a *single call* extracted as an audio snippet.
The dataset is organized by individual birds, with each bird having its own
split containing its vocal repertoire. Note that the repertoires
are per-bird, so labels should not be compared across splits.

The data is hosted in the `esp-ml-datasets` GCS bucket in folder bengalese_finch with:
- Individual CSV files per bird, plus bird-level splits (Bird0.csv, Bird1.csv, etc.)
- Extracted call audio snippets in `wav/BirdX/` subdirectories

The CSVs have the following columns:
* ``local_path``     - relative path to the extracted call audio snippet
* ``call_type``      - call-type ID (string)
* ``individual_id``  - identifier of the individual bird
* ``local_path``     - relative path to the extracted call audio snippet
* ``call_type``      - call-type ID (string)
* ``individual_id``  - identifier of the individual bird

**Available Split Types:**
- ``{BirdX}_train``: Full training set (~70% of data)
- ``{BirdX}_train_small``: Limited training set (max 80 samples per call type)
- ``{BirdX}_valid``: Validation set (~15% of data)
- ``{BirdX}_test``: Test set (~15% of data)

Examples
--------
# Individual bird
>>> from esp_data.datasets import BengaleseFinchCalls
>>> ds = BengaleseFinchCalls(split="Bird0", sample_rate=16000)
>>> first = ds[0]
>>> first.keys()
dict_keys(['local_path', 'call_type', 'individual_id', 'audio', 'sample_rate'])

# Bird2 training split
>>> train_ds = BengaleseFinchCalls(split="Bird2_train", sample_rate=16000)
>>> print(f"Training samples: {len(train_ds)}")
Training samples: 18303

# Learning with limited data
>>> small_train_ds = BengaleseFinchCalls(split="Bird2_train_small", sample_rate=16000)
>>> print(f"Small training samples: {len(small_train_ds)}")
Small training samples: 1360

# Any bird's splits are available
>>> bird1_train = BengaleseFinchCalls(split="Bird1_train", sample_rate=16000)
>>> bird8_valid = BengaleseFinchCalls(split="Bird8_valid", sample_rate=16000)
>>> print(f"Bird1 training: {len(bird1_train)} samples")
Bird1 training: 25024 samples
>>> print(f"Bird8 validation: {len(bird8_valid)} samples")
Bird8 validation: 746 samples
"""

from typing import Any, Dict, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio
from esp_data.schema import ColumnSchema, DatasetSchema


@register_dataset
class BengaleseFinchCalls(Dataset):
    """Bengalese Finch call-type dataset with individual bird splits."""

    info = DatasetInfo(
        name="Bengalese Finch Calls",
        owner="david",
        split_paths={
            # Original bird datasets (complete individual repertoires)
            "Bird0": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird0.csv",
            "Bird1": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird1.csv",
            "Bird2": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird2.csv",
            "Bird3": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird3.csv",
            "Bird4": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird4.csv",
            "Bird5": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird5.csv",
            "Bird6": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird6.csv",
            "Bird7": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird7.csv",
            "Bird8": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird8.csv",
            "Bird9": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird9.csv",
            "Bird10": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird10.csv",
            # Bird0 splits (9 call types, 7,652 samples)
            "Bird0_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird0_train.csv",
            "Bird0_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird0_train_small.csv",
            "Bird0_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird0_valid.csv",
            "Bird0_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird0_test.csv",
            # Bird1 splits (12 call types, 35,728 samples)
            "Bird1_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird1_train.csv",
            "Bird1_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird1_train_small.csv",
            "Bird1_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird1_valid.csv",
            "Bird1_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird1_test.csv",
            # Bird2 splits (17 call types, 26,127 samples) - highest diversity
            "Bird2_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird2_train.csv",
            "Bird2_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird2_train_small.csv",
            "Bird2_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird2_valid.csv",
            "Bird2_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird2_test.csv",
            # Bird3 splits (9 call types, 29,470 samples)
            "Bird3_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird3_train.csv",
            "Bird3_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird3_train_small.csv",
            "Bird3_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird3_valid.csv",
            "Bird3_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird3_test.csv",
            # Bird4 splits (5 call types, 26,891 samples)
            "Bird4_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird4_train.csv",
            "Bird4_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird4_train_small.csv",
            "Bird4_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird4_valid.csv",
            "Bird4_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird4_test.csv",
            # Bird5 splits (7 call types, 20,525 samples)
            "Bird5_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird5_train.csv",
            "Bird5_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird5_train_small.csv",
            "Bird5_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird5_valid.csv",
            "Bird5_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird5_test.csv",
            # Bird6 splits (5 call types, 17,653 samples)
            "Bird6_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird6_train.csv",
            "Bird6_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird6_train_small.csv",
            "Bird6_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird6_valid.csv",
            "Bird6_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird6_test.csv",
            # Bird7 splits (7 call types, 20,722 samples)
            "Bird7_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird7_train.csv",
            "Bird7_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird7_train_small.csv",
            "Bird7_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird7_valid.csv",
            "Bird7_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird7_test.csv",
            # Bird8 splits (4 call types, 4,985 samples)
            "Bird8_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird8_train.csv",
            "Bird8_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird8_train_small.csv",
            "Bird8_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird8_valid.csv",
            "Bird8_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird8_test.csv",
            # Bird9 splits (6 call types, 19,541 samples)
            "Bird9_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird9_train.csv",
            "Bird9_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird9_train_small.csv",
            "Bird9_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird9_valid.csv",
            "Bird9_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird9_test.csv",
            # Bird10 splits (12 call types, 5,743 samples)
            "Bird10_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird10_train.csv",
            "Bird10_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird10_train_small.csv",
            "Bird10_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird10_valid.csv",
            "Bird10_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/Bird10_test.csv",
        },
        version="0.1.0",
        description=(
            "Bengalese Finch calls annotated with call-type and individual IDs, "
            "organized by individual birds."
        ),
        sources=["BanglaJp_whistle"],
        license="CC-BY-4.0, CC0",
    )

    schema = DatasetSchema(
        columns=[
            ColumnSchema(name="local_path", dtype="str", required=True),
            ColumnSchema(name="call_type", dtype="int", required=True),
            ColumnSchema(name="individual_id", dtype="str", required=True),
        ]
    )

    def __init__(
        self,
        split: str = "Bird2_train",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Create a :class:`BengaleseFinchCalls` instance.

        Parameters
        ----------
        split: str
            Which bird/split to load. Options include:
            - Individual birds: "Bird0", "Bird1", ..., "Bird10" (complete repertoires)
            - ML splits: "{BirdX}_train", "{BirdX}_train_small", "{BirdX}_valid", "{BirdX}_test"
        output_take_and_give: dict[str, str], optional
            Mapping from original column names to desired output names.  When
            provided, the dataset __getitem__ will return only the mapped
            columns and use the *values* of this dict as keys.
        sample_rate: int, optional
            Target sample-rate.  If provided and differs from the original, the
            audio is resampled with ``librosa.resample``.
        data_root: str | AnyPathT, optional
            Custom root directory for audio files.  When *None* (default), we
            automatically use the parent directory of the metadata CSV.
        backend: BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars"
        streaming: bool, optional
            Whether to use streaming mode, by default False

        Raises
        ------
        LookupError
            If the specified split is not available in the dataset.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate

        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split '{self.split}'. Available: {list(self.info.split_paths)}"
            )

        if data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent
        else:
            self.data_root = data_root

        self._data = None
        self._load()

    @property
    def columns(self) -> list[str]:
        """Return the DataFrame column names."""
        return list(self._data.columns)

    @property
    def available_splits(self) -> list[str]:
        """Return the names of available splits."""
        return list(self.info.split_paths)

    def _load(self) -> None:
        """Load the CSV for the chosen split into :pyattr:`_data`."""
        csv_path = self.info.split_paths[self.split]

        # Read as DataFrame (avoid NA coercion so that strings stay strings)
        self._data = self._backend_class.from_csv(
            csv_path,
            streaming=self._streaming,  # keep_default_na=False, na_values=[""]
        )

    def __len__(self) -> int:
        if self._data is None:
            raise RuntimeError("Dataset not loaded - call _load() first.")
        if self._streaming:
            raise NotImplementedError("Length not available in streaming mode.")
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        # Construct full audio path ("local_path" is relative)
        audio_path = anypath(self.data_root) / row["local_path"]

        # Load the audio file
        audio, sample_rate = read_audio(audio_path)
        audio = audio.astype(np.float32)
        audio = audio_stereo_to_mono(audio, mono_method="average")

        # Resample if the user requested a specific sample-rate
        if self.sample_rate is not None and sample_rate != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sample_rate,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
            sample_rate = self.sample_rate

        row["audio"] = audio
        row["sample_rate"] = sample_rate

        # Apply output mapping if requested
        if self.output_take_and_give:
            mapped: dict[str, Any] = {}
            for src, dst in self.output_take_and_give.items():
                mapped[dst] = row[src]

            # Always include audio unless explicitly mapped
            if "audio" not in self.output_take_and_give:
                mapped["audio"] = row["audio"]
            return mapped

        return row

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a specific sample from the dataset.
        Parameters
        ----------
        idx : int
            Index of the sample to get.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the data.
        """
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples in the dataset.
        Yields
        -------
        Dict[str, Any]
            Each sample in the dataset.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["BengaleseFinchCalls", dict[str, Any]]:
        """Instantiate from a :class:`DatasetConfig`.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary containing dataset parameters.

        Returns
        -------
        tuple[BengaleseFinchCalls, dict[str, Any]]
            A tuple containing the dataset instance and metadata from transformations.
            If no transformations are applied, metadata will be an empty dict.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})

        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            data_root=cfg["data_root"],
            sample_rate=cfg["sample_rate"],
            backend=cfg["backend"],
            streaming=cfg["streaming"],
        )

        if dataset_config.transformations:
            transform_metadata = ds.apply_transformations(dataset_config.transformations)
            return ds, transform_metadata
        return ds, {}

    def __str__(self) -> str:  # noqa: D401 – keep style consistent
        base = f"{self.info.name} (v{self.info.version}), split='{self.split}'"
        return (
            f"{base}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths)}"
        )
