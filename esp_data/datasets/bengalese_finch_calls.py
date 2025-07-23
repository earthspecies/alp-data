"""Bengalese Finch Calls dataset

This dataset contains individual calls from Bengalese finches. Each row in the
metadata CSV corresponds to a *single call* extracted as an audio snippet.
The dataset is organized by individual birds, with each bird having its own
split containing its vocal repertoire. Note that the repertoires
are per-bird, so labels should not be compared across splits.

**Available Split Types:**
- ``{BirdX}_train``: Full training set (~70% of data)
- ``{BirdX}_train_small``: Limited training set (max 80 samples per call type)
- ``{BirdX}_valid``: Validation set (~15% of data)
- ``{BirdX}_test``: Test set (~15% of data)

Examples
--------
>>> from esp_data.datasets import BengaleseFinchCalls
>>> # Individual bird (legacy usage)
>>> ds = BengaleseFinchCalls(split="Bird0", sample_rate=16000)
>>> first = ds[0]
>>> first.keys()
dict_keys(['local_path', 'call_type', 'individual_id', 'audio'])

>>> # Default: Bird2 training split (highest diversity - recommended)
>>> train_ds = BengaleseFinchCalls(split="Bird2_train", sample_rate=16000)
>>> print(f"Training samples: {len(train_ds)}")
Training samples: 18303

>>> # Learning with limited data
>>> small_train_ds = BengaleseFinchCalls(split="Bird2_train_small", sample_rate=16000)
>>> print(f"Small training samples: {len(small_train_ds)}")
Small training samples: 1360

>>> # Any bird's splits are available
>>> bird1_train = BengaleseFinchCalls(split="Bird1_train", sample_rate=16000)
>>> bird8_valid = BengaleseFinchCalls(split="Bird8_valid", sample_rate=16000)
>>> print(f"Bird1 training: {len(bird1_train)} samples")
Bird1 training: 25024
>>> print(f"Bird8 validation: {len(bird8_valid)} samples")
Bird8 validation: 746
"""

from typing import Any, Dict, Iterator, Optional

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


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
            "Bird0_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird0_train.csv",
            "Bird0_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird0_train_small.csv",
            "Bird0_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird0_valid.csv",
            "Bird0_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird0_test.csv",
            # Bird1 splits (12 call types, 35,728 samples)
            "Bird1_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird1_train.csv",
            "Bird1_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird1_train_small.csv",
            "Bird1_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird1_valid.csv",
            "Bird1_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird1_test.csv",
            # Bird2 splits (17 call types, 26,127 samples) - highest diversity
            "Bird2_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird2_train.csv",
            "Bird2_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird2_train_small.csv",
            "Bird2_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird2_valid.csv",
            "Bird2_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird2_test.csv",
            # Bird3 splits (9 call types, 29,470 samples)
            "Bird3_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird3_train.csv",
            "Bird3_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird3_train_small.csv",
            "Bird3_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird3_valid.csv",
            "Bird3_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird3_test.csv",
            # Bird4 splits (5 call types, 26,891 samples)
            "Bird4_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird4_train.csv",
            "Bird4_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird4_train_small.csv",
            "Bird4_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird4_valid.csv",
            "Bird4_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird4_test.csv",
            # Bird5 splits (7 call types, 20,525 samples)
            "Bird5_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird5_train.csv",
            "Bird5_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird5_train_small.csv",
            "Bird5_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird5_valid.csv",
            "Bird5_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird5_test.csv",
            # Bird6 splits (5 call types, 17,653 samples)
            "Bird6_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird6_train.csv",
            "Bird6_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird6_train_small.csv",
            "Bird6_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird6_valid.csv",
            "Bird6_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird6_test.csv",
            # Bird7 splits (7 call types, 20,722 samples)
            "Bird7_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird7_train.csv",
            "Bird7_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird7_train_small.csv",
            "Bird7_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird7_valid.csv",
            "Bird7_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird7_test.csv",
            # Bird8 splits (4 call types, 4,985 samples)
            "Bird8_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird8_train.csv",
            "Bird8_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird8_train_small.csv",
            "Bird8_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird8_valid.csv",
            "Bird8_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird8_test.csv",
            # Bird9 splits (6 call types, 19,541 samples)
            "Bird9_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird9_train.csv",
            "Bird9_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird9_train_small.csv",
            "Bird9_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird9_valid.csv",
            "Bird9_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird9_test.csv",
            # Bird10 splits (12 call types, 5,743 samples)
            "Bird10_train": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird10_train.csv",
            "Bird10_train_small": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird10_train_small.csv",
            "Bird10_valid": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird10_valid.csv",
            "Bird10_test": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/splits/Bird10_test.csv",
        },
        version="0.1.0",
        description=(
            "Bengalese Finch calls annotated with call-type and individual IDs, "
            "organized by individual birds. Includes comprehensive train/valid/test/"
            "train_small splits for all 11 birds (Bird0-Bird10), enabling systematic "
            "ML research across varying call type diversities (4-17 types per bird). "
            "Train_small splits limit samples to max 80 per call type for few-shot learning."
        ),
        sources=["BanglaJp_whistle"],
        license="CC-BY-4.0, CC0",
    )

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------

    def __init__(
        self,
        split: str = "Bird2_train",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: Optional[int] = None,
        data_root: Optional[str | AnyPathT] = None,
    ) -> None:
        """Create a :class:`BengaleseFinchCalls` instance.

        Parameters
        ----------
        split
            Which bird/split to load. Options include:
            - Individual birds: "Bird0", "Bird1", ..., "Bird10" (complete repertoires)
            - ML splits: "{BirdX}_train", "{BirdX}_train_small", "{BirdX}_valid", "{BirdX}_test"
            where BirdX can be any of Bird0-Bird10. Defaults to "Bird2_train" (highest diversity).
        output_take_and_give
            Mapping from original column names to desired output names.  When
            provided, the dataset __getitem__ will return only the mapped
            columns and use the *values* of this dict as keys.
        sample_rate
            Target sample-rate.  If provided and differs from the original, the
            audio is resampled with ``librosa.resample``.
        data_root
            Custom root directory for audio files.  When *None* (default), we
            automatically use the parent directory of the metadata CSV.

        Raises
        ------
        LookupError
            If the specified split is not available in the dataset.
        """
        super().__init__(output_take_and_give)
        self.split = split
        self.sample_rate = sample_rate
        self.data_root = data_root

        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split '{self.split}'. Available: {list(self.info.split_paths)}"
            )

        # Default data root – parent directory of the split CSV.
        if self.data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent

        self._data: pd.DataFrame | None = None
        self._load()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def columns(self) -> list[str]:
        """Return the DataFrame column names."""
        return list(self._data.columns)

    @property
    def available_splits(self) -> list[str]:
        """Return the names of available splits."""
        return list(self.info.split_paths)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the CSV for the chosen split into :pyattr:`_data`."""
        csv_path = self.info.split_paths[self.split]

        # Read as DataFrame (avoid NA coercion so that strings stay strings)
        self._data = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:  # noqa: D401 – consistent with Dataset interface
        if self._data is None:
            raise RuntimeError("Dataset not loaded – call _load() first.")
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range (dataset length: {len(self)})")

        row = self._data.iloc[idx].to_dict()

        # Construct full audio path ("local_path" is relative)
        if self.data_root is not None:
            audio_path = anypath(self.data_root) / row["local_path"]
        else:
            audio_path = anypath(row["local_path"])

        # Load the audio file
        audio, sr = read_audio(audio_path)
        audio = audio.astype(np.float32)
        audio = audio_stereo_to_mono(audio, mono_method="average")

        # Resample if the user requested a specific sample-rate
        if self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )

        row["audio"] = audio

        # Apply output mapping if requested
        if self.output_take_and_give:
            mapped: Dict[str, Any] = {}
            for src, dst in self.output_take_and_give.items():
                mapped[dst] = row[src]
            # Always include audio unless explicitly mapped
            if "audio" not in self.output_take_and_give:
                mapped["audio"] = row["audio"]
            return mapped

        return row

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

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

        Raises
        ------
        LookupError
            If the specified split is not available in the dataset.
        """
        cfg = dataset_config.model_dump(exclude=("dataset_name", "transformations"))

        split = cfg.get("split", "Bird2_train")
        if split not in cls.info.split_paths:
            raise LookupError(
                f"Invalid split '{split}'. Available splits: {', '.join(cls.info.split_paths)}"
            )

        ds = cls(
            split=split,
            output_take_and_give=cfg.get("output_take_and_give"),
            data_root=cfg.get("data_root"),
            sample_rate=cfg.get("sample_rate"),
        )

        if dataset_config.transformations:
            transform_metadata = ds.apply_transformations(dataset_config.transformations)
            return ds, transform_metadata
        return ds, {}

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    def __str__(self) -> str:  # noqa: D401 – keep style consistent
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths)}"
        )
