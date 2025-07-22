"""Wabad dataset"""

from io import StringIO
from typing import Any, Dict, Iterator, Optional

import librosa
import numpy as np
import pandas as pd
from numpy.random import default_rng

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class Wabad(Dataset):
    """WABAD dataset as individual events.

    Description
    -----------
    A global PAM dataset with temporally strong labels for bird vocalizations.
    Includes 90,662 boxed vocalisations from 1,147 bird species recorded at 70
    recording sites in 27 countries and distributed across 13 biomes.

    For this dataset variant, each sample corresponds to an individual boxed vocalization.
    The audio window around each event can be randomly sampled, potentially including
    other species' vocalizations in the labels when they fall within the window.

    Available Splits
    ----------------
    - ``all``: Complete dataset (4,297 files, all events)
    - ``train``: Training split (3,437 files, 80% random sample)
    - ``validation``: Validation split (860 files, 20% random sample)
    - ``train_sites``: Training split (3,435 files, geographic split)
    - ``validation_sites``: Validation split (862 files, held-out sites)
    - ``*_16khz``: 16kHz resampled versions of above splits

    Use random splits (``train``/``validation``) for standard ML workflows.
    Use site-based splits (``train_sites``/``validation_sites``) for testing
    geographic generalization on completely unseen recording locations.

    References
    ----------
    https://www.researchsquare.com/article/rs-5729784/v1
    https://zenodo.org/records/15629388

    Licensing
    ---------
    Creative Commons Attribution Non Commercial 4.0 International copyright.

    Examples
    -------
    >>> from esp_data.datasets import Wabad
    >>> # Use training split for model training
    >>> dataset = Wabad(
    ...     split="train",
    ...     window_duration=5.0,
    ...     random_window=True,
    ...     seed=42
    ... )
    >>> print(dataset.info.name)
    wabad
    >>> print(f"Number of events: {len(dataset)}")
    Number of events: 73441
    >>>
    >>> # Use validation split for evaluation
    >>> val_dataset = Wabad(
    ...     split="validation_16khz",
    ...     window_duration=3.0,
    ...     sample_rate=16000,
    ...     seed=42
    ... )
    >>> sample = val_dataset[0]
    >>> print(f"Target species: {sample['target_species']}")
    >>> print(f"Audio shape: {sample['audio'].shape}")
    >>> print(f"Sample rate: 16000 Hz")
    >>>
    >>> # Use site-based splits for geographic generalization
    >>> site_val = Wabad(split="validation_sites", seed=42)
    >>> print(f"Site-based validation events: {len(site_val)}")
    """

    info = DatasetInfo(
        name="wabad",
        owner="esp-team",
        split_paths={
            "all": "gs://esp-ml-datasets/wabad/v0.1.0/raw/all_info.csv",
            "all_16khz": "gs://esp-ml-datasets/wabad/v0.1.0/raw_16khz/all_info.csv",
            "train": "gs://esp-ml-datasets/wabad/v0.1.0/raw/train_info.csv",
            "train_16khz": "gs://esp-ml-datasets/wabad/v0.1.0/raw_16khz/train_info.csv",
            "validation": "gs://esp-ml-datasets/wabad/v0.1.0/raw/validation_info.csv",
            "validation_16khz": "gs://esp-ml-datasets/wabad/v0.1.0/raw_16khz/validation_info.csv",
            "train_sites": "gs://esp-ml-datasets/wabad/v0.1.0/raw/train_sites_info.csv",
            "train_sites_16khz": "gs://esp-ml-datasets/wabad/v0.1.0/raw_16khz/train_sites_info.csv",
            "validation_sites": "gs://esp-ml-datasets/wabad/v0.1.0/raw/validation_sites_info.csv",
            "validation_sites_16khz": "gs://esp-ml-datasets/wabad/v0.1.0/raw_16khz/validation_sites_info.csv",
        },
        version="0.1.0",
        description=(
            "Wabad events dataset with temporally strong labels for individual "
            "bird vocalization events"
        ),
        sources=["Global PAM recordings"],
        license="CC BY-NC 4.0",
    )

    def __init__(
        self,
        split: str = "master",
        output_take_and_give: dict[str, str] = None,
        sample_rate: int = 16000,
        data_root: Optional[str | AnyPathT] = None,
        mono_method: str = "average",
        window_duration: float = 5.0,
        random_window: bool = True,
        min_padding: float = 0.0,
        max_padding: float = 2.0,
        unknown_label: str = "Unknown",
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the Wabad dataset.

        Parameters
        ----------
        split : str
            The split to load.
        output_take_and_give : dict[str, str]
            A dictionary mapping original column names to new column names.
        sample_rate : int
            The sample rate to which audio files should be resampled.
        data_root : str | AnyPathT, optional
            The root directory for the dataset.
        mono_method : str
            The method to convert stereo audio to mono.
        window_duration : float
            Duration of the audio window around each event in seconds.
        random_window : bool
            If True, randomly position the event within the window.
            If False, center the event in the window.
        min_padding : float
            Minimum padding around the event in seconds.
        max_padding : float
            Maximum padding around the event in seconds.
        unknown_label : str
            Label for unknown/unlabeled regions.
        seed : int, optional
            Random seed for reproducible window sampling.
        """
        super().__init__(output_take_and_give)
        self.split = split
        self._data: pd.DataFrame = None
        self._load()
        self.sample_rate = sample_rate
        self.data_root = data_root
        self.mono_method = mono_method
        self.window_duration = window_duration
        self.random_window = random_window
        self.min_padding = min_padding
        self.max_padding = max_padding
        self.unknown_label = unknown_label
        self.seed = seed
        self.rng = default_rng(seed)

        if self.data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent

        self._create_events_metadata()

    @property
    def columns(self) -> list[str]:
        """Return the columns of the dataset.

        Returns
        -------
        list[str]
            List of column names in the master dataframe.
        """
        return list(self._data.columns)

    @property
    def available_splits(self) -> list[str]:
        """Return the available splits of the dataset.

        Returns
        -------
        list[str]
            List of available split names.
        """
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        """Load the dataset.

        Raises
        ------
        LookupError
            If the specified split is not available in the dataset.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )

        location = self.info.split_paths[self.split]
        csv_text = anypath(location).read_text(encoding="utf-8")
        self._data = pd.read_csv(StringIO(csv_text))

    def _create_events_metadata(self) -> None:
        """Create metadata for individual events."""
        events = []
        selection_tables = {}

        for _idx, row in self._data.iterrows():
            fn = row["fn"]
            audio_fp = row["audio_fp"]

            # Parse selection table
            selection_table = pd.read_csv(StringIO(row["selection_table_str"]), sep="\t")
            selection_tables[fn] = selection_table

            # Create event for each annotation
            for _, annotation in selection_table.iterrows():
                event = {
                    "fn": fn,
                    "audio_fp": audio_fp,
                    "start_time": annotation["Begin Time (s)"],
                    "end_time": annotation["End Time (s)"],
                    "species": annotation["Species"],
                    "low_freq": annotation["Low Freq (Hz)"],
                    "high_freq": annotation["High Freq (Hz)"],
                    "audio_duration": row.get("audio_duration", None),
                }
                events.append(event)

        self._events = events
        self._selection_tables = selection_tables

    def _get_window_bounds(self, event: dict) -> tuple[float, float]:
        """Get the audio window bounds for an event.

        Parameters
        ----------
        event : dict
            Event metadata containing start_time and end_time.

        Returns
        -------
        tuple[float, float]
            (window_start, window_end) in seconds.
        """
        event_start = event["start_time"]
        event_end = event["end_time"]

        if self.random_window:
            # Random padding on each side
            padding_before = self.rng.uniform(self.min_padding, self.max_padding)
            padding_after = self.rng.uniform(self.min_padding, self.max_padding)

            window_start = event_start - padding_before
            window_end = event_end + padding_after

            # Adjust if window is too short
            current_duration = window_end - window_start
            if current_duration < self.window_duration:
                extra_needed = self.window_duration - current_duration
                window_start -= extra_needed / 2
                window_end += extra_needed / 2
        else:
            # Center the event in the window
            event_center = (event_start + event_end) / 2
            window_start = event_center - self.window_duration / 2
            window_end = event_center + self.window_duration / 2

        # Ensure window doesn't go before start of audio
        if window_start < 0:
            window_end -= window_start
            window_start = 0

        # Ensure window doesn't exceed audio duration if known
        if event["audio_duration"] is not None:
            if window_end > event["audio_duration"]:
                excess = window_end - event["audio_duration"]
                window_start = max(0, window_start - excess)
                window_end = event["audio_duration"]

        return window_start, window_end

    def _get_labels_in_window(self, fn: str, window_start: float, window_end: float) -> list[dict]:
        """Get all species labels within the specified window.

        Parameters
        ----------
        fn : str
            Filename identifier.
        window_start : float
            Window start time in seconds.
        window_end : float
            Window end time in seconds.

        Returns
        -------
        list[dict]
            List of label dictionaries with relative timing within the window.
        """
        selection_table = self._selection_tables[fn]
        labels = []

        for _, annotation in selection_table.iterrows():
            ann_start = annotation["Begin Time (s)"]
            ann_end = annotation["End Time (s)"]

            # Check if annotation overlaps with window
            if ann_end > window_start and ann_start < window_end:
                # Calculate relative times within the window
                rel_start = max(0, ann_start - window_start)
                rel_end = min(window_end - window_start, ann_end - window_start)

                label = {
                    "species": annotation["Species"],
                    "start_time": rel_start,
                    "end_time": rel_end,
                    "low_freq": annotation["Low Freq (Hz)"],
                    "high_freq": annotation["High Freq (Hz)"],
                }
                labels.append(label)

        return labels

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["Wabad", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary containing dataset parameters.

        Returns
        -------
        tuple[Dataset, dict[str, Any]]
            A tuple containing the dataset instance and metadata.
            If the dataset_config contains transformations, they will be applied
            and the metadata will be returned as dict, otherwise an empty dict.

        Raises
        ------
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
            sample_rate=cfg.get("sample_rate", 16000),
            mono_method=cfg.get("mono_method", "average"),
            window_duration=cfg.get("window_duration", 5.0),
            random_window=cfg.get("random_window", True),
            min_padding=cfg.get("min_padding", 0.0),
            max_padding=cfg.get("max_padding", 2.0),
            unknown_label=cfg.get("unknown_label", "Unknown"),
            seed=cfg.get("seed", None),
        )

        if dataset_config.transformations:
            transform_metadata = ds.apply_transformations(dataset_config.transformations)
            return ds, transform_metadata

        return ds, {}

    def __len__(self) -> int:
        """Return the number of events in the dataset.

        Returns
        -------
        int
            Number of individual vocalization events in the dataset.
        """
        return len(self._events)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a specific event from the dataset.

        Parameters
        ----------
        idx : int
            Index of the event to retrieve.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the audio data and metadata for the event.
            Keys include: 'audio', 'fn', 'target_species', 'target_start_time',
            'target_end_time', 'target_low_freq', 'target_high_freq',
            'window_start', 'window_end', 'window_duration', 'all_labels'.

        Raises
        ------
        IndexError
            If the index is out of bounds for the dataset.
        """
        if idx >= len(self._events):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {len(self._events)}."
            )

        event = self._events[idx]

        # Get window bounds
        window_start, window_end = self._get_window_bounds(event)

        # Load audio for the window
        if self.data_root:
            audio_path = anypath(self.data_root) / event["audio_fp"]
        else:
            audio_path = anypath(event["audio_fp"])

        audio, sr = read_audio(audio_path, start_time=window_start, end_time=window_end)
        audio = audio.astype(np.float32)

        if self.mono_method:
            audio = audio_stereo_to_mono(audio, mono_method=self.mono_method)

        if self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )

        # Get all labels within the window
        labels = self._get_labels_in_window(event["fn"], window_start, window_end)

        # Create the output item
        item = {
            "audio": audio,
            "fn": event["fn"],
            "target_species": event["species"],  # The main target species for this event
            "target_start_time": max(0, event["start_time"] - window_start),
            "target_end_time": min(window_end - window_start, event["end_time"] - window_start),
            "target_low_freq": event["low_freq"],
            "target_high_freq": event["high_freq"],
            "window_start": window_start,
            "window_end": window_end,
            "window_duration": window_end - window_start,
            "all_labels": labels,  # All species labels within the window
        }

        if self.output_take_and_give:
            filtered_item = {}
            for key, value in self.output_take_and_give.items():
                if key in item:
                    filtered_item[value] = item[key]
            return filtered_item
        else:
            return item

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over events in the dataset.

        Yields
        ------
        Dict[str, Any]
            Each event in the dataset as returned by __getitem__.
        """
        for idx in range(len(self)):
            yield self[idx]

    def __str__(self) -> str:
        """Return a string representation of the dataset.

        Returns
        -------
        str
            A formatted string containing dataset metadata including name, version,
            description, sources, license, available splits, and configuration.
        """
        base_info = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base_info}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}\n"
            f"Number of events: {len(self._events)}\n"
            f"Window duration: {self.window_duration}s\n"
            f"Random window: {self.random_window}\n"
            f"Seed: {self.seed}"
        )
