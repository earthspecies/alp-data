"""Voxaboxen-data dataset"""

import math
from io import StringIO
from typing import Any, Dict, Iterator, Optional

import librosa
import numpy as np
import pandas as pd
from intervaltree import IntervalTree
from numpy.random import default_rng

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.io import (
    AnyPathT,
    anypath,
    audio_stereo_to_mono,
    read_audio,
)

LABEL_SETS = {
    "Anuraset_train": [
        "SPHSUR_M",
        "PHYALB_L",
        "Unknown",
        "LEPPOD_L",
        "DENMIN_M",
        "BOABIS_M",
        "BOABIS_L",
        "PITAZU_L",
        "LEPLAT_L",
        "BOAALB_L",
        "DENMIN_L",
    ],
    "Anuraset_val": [
        "SPHSUR_M",
        "PHYALB_L",
        "Unknown",
        "LEPPOD_L",
        "DENMIN_M",
        "BOABIS_M",
        "BOABIS_L",
        "PITAZU_L",
        "LEPLAT_L",
        "BOAALB_L",
        "DENMIN_L",
    ],
    "Anuraset_test": [
        "SPHSUR_M",
        "PHYALB_L",
        "Unknown",
        "LEPPOD_L",
        "DENMIN_M",
        "BOABIS_M",
        "BOABIS_L",
        "PITAZU_L",
        "LEPLAT_L",
        "BOAALB_L",
        "DENMIN_L",
    ],
    "BV_train": ["voc", "Unknown"],
    "BV_val": ["voc", "Unknown"],
    "BV_test": ["voc", "Unknown"],
    "MT_train": ["voc", "Unknown"],
    "MT_val": ["voc", "Unknown"],
    "MT_test": ["voc", "Unknown"],
    "OZF_train": ["voc", "Unknown"],
    "OZF_val": ["voc", "Unknown"],
    "OZF_test": ["voc", "Unknown"],
    "hawaii_train": [
        "ercfra",
        "reblei",
        "Unknown",
        "warwhe1",
        "skylar",
        "hawama",
        "houfin",
        "apapan",
        "omao",
        "iiwi",
    ],
    "hawaii_val": [
        "ercfra",
        "reblei",
        "Unknown",
        "warwhe1",
        "skylar",
        "hawama",
        "houfin",
        "apapan",
        "omao",
        "iiwi",
    ],
    "hawaii_test": [
        "ercfra",
        "reblei",
        "Unknown",
        "warwhe1",
        "hawama",
        "skylar",
        "houfin",
        "apapan",
        "omao",
        "iiwi",
    ],
    "humpback_train": ["Mn", "Unknown"],
    "humpback_val": ["Mn", "Unknown"],
    "humpback_test": ["Mn", "Unknown"],
    "katydids_train": ["Unknown", "katydid"],
    "katydids_val": ["Unknown", "katydid"],
    "katydids_test": ["Unknown", "katydid"],
    "powdermill_train": ["Unknown", "NOCA", "BCCH", "EATO", "BTNW", "REVI", "TUTI"],
    "powdermill_val": ["Unknown", "NOCA", "BCCH", "EATO", "BTNW", "REVI", "TUTI"],
    "powdermill_test": ["Unknown", "NOCA", "BCCH", "EATO", "BTNW", "REVI", "TUTI"],
    "OZF_synthetic_overlap_0_train": ["Unknown", "POS"],
    "OZF_synthetic_overlap_0_val": ["Unknown", "POS"],
    "OZF_synthetic_overlap_0_test": ["Unknown", "POS"],
    "OZF_synthetic_overlap_0.2_train": ["Unknown", "POS"],
    "OZF_synthetic_overlap_0.2_val": ["Unknown", "POS"],
    "OZF_synthetic_overlap_0.2_test": ["Unknown", "POS"],
    "OZF_synthetic_overlap_0.4_train": ["Unknown", "POS"],
    "OZF_synthetic_overlap_0.4_val": ["Unknown", "POS"],
    "OZF_synthetic_overlap_0.4_test": ["Unknown", "POS"],
    "OZF_synthetic_overlap_0.6_train": ["Unknown", "POS"],
    "OZF_synthetic_overlap_0.6_val": ["Unknown", "POS"],
    "OZF_synthetic_overlap_0.6_test": ["Unknown", "POS"],
    "OZF_synthetic_overlap_1_train": ["Unknown", "POS"],
    "OZF_synthetic_overlap_1_val": ["Unknown", "POS"],
    "OZF_synthetic_overlap_1_test": ["Unknown", "POS"],
}


@register_dataset
class Voxaboxen(Dataset):
    """Voxaboxen dataset.

    Description
    -----------
    Voxaboxen is the dataset used in the Voxaboxen project. It consists of
    several datasets with annotated call start and end times via selection tables.
    Excerpt from paper:
    "...a method for accurately detecting bioacoustic sound events that is robust
    to overlapping events... We also release a new dataset designed to measure
    performance on detecting overlapping vocalizations. This consists of recordings of
    zebra finches annotated with temporally-strong labels and showing frequent overlaps.
    We test Voxaboxen on seven existing data sets and on our new data set."

    References
    ----------
    Robust detection of overlapping bioacoustic sound events
    Louis Mahon, Benjamin Hoffman, Logan S James, Maddie Cusimano,
    Masato Hagiwara, Sarah C Woolley, Olivier Pietquin
    https://arxiv.org/abs/2503.02389

    Examples
    -------
    >>> from esp_data.datasets import Voxaboxen
    >>> dataset = Voxaboxen(
    ...     split="BV_val",
    ...     output_take_and_give={"selection_table": "st"}
    ... )
    >>> print(dataset.info.name)
    voxaboxen
    """

    info = DatasetInfo(
        name="voxaboxen",
        owner="benjamin; gagan",
        split_paths={
            "Anuraset_train": "gs://esp-ml-datasets/voxaboxen/files/Anuraset/formatted/train_info.csv",
            "Anuraset_val": "gs://esp-ml-datasets/voxaboxen/files/Anuraset/formatted/val_info.csv",
            "Anuraset_test": "gs://esp-ml-datasets/voxaboxen/files/Anuraset/formatted/test_info.csv",
            "BV_train": "gs://esp-ml-datasets/voxaboxen/files/BV/formatted/train_info.csv",
            "BV_val": "gs://esp-ml-datasets/voxaboxen/files/BV/formatted/val_info.csv",
            "BV_test": "gs://esp-ml-datasets/voxaboxen/files/BV/formatted/test_info.csv",
            "MT_train": "gs://esp-ml-datasets/voxaboxen/files/MT/formatted/train_info.csv",
            "MT_val": "gs://esp-ml-datasets/voxaboxen/files/MT/formatted/val_info.csv",
            "MT_test": "gs://esp-ml-datasets/voxaboxen/files/MT/formatted/test_info.csv",
            "OZF_train": "gs://esp-ml-datasets/voxaboxen/files/OZF/formatted/train_info.csv",
            "OZF_val": "gs://esp-ml-datasets/voxaboxen/files/OZF/formatted/val_info.csv",
            "OZF_test": "gs://esp-ml-datasets/voxaboxen/files/OZF/formatted/test_info.csv",
            "hawaii_train": "gs://esp-ml-datasets/voxaboxen/files/hawaii/formatted/train_info.csv",
            "hawaii_val": "gs://esp-ml-datasets/voxaboxen/files/hawaii/formatted/val_info.csv",
            "hawaii_test": "gs://esp-ml-datasets/voxaboxen/files/hawaii/formatted/test_info.csv",
            "humpback_train": "gs://esp-ml-datasets/voxaboxen/files/humpback/formatted/train_info.csv",
            "humpback_val": "gs://esp-ml-datasets/voxaboxen/files/humpback/formatted/val_info.csv",
            "humpback_test": "gs://esp-ml-datasets/voxaboxen/files/humpback/formatted/test_info.csv",
            "katydids_train": "gs://esp-ml-datasets/voxaboxen/files/katydids/formatted/train_info.csv",
            "katydids_val": "gs://esp-ml-datasets/voxaboxen/files/katydids/formatted/val_info.csv",
            "katydids_test": "gs://esp-ml-datasets/voxaboxen/files/katydids/formatted/test_info.csv",
            "powdermill_train": "gs://esp-ml-datasets/voxaboxen/files/powdermill/formatted/train_info.csv",
            "powdermill_val": "gs://esp-ml-datasets/voxaboxen/files/powdermill/formatted/val_info.csv",
            "powdermill_test": "gs://esp-ml-datasets/voxaboxen/files/powdermill/formatted/test_info.csv",
            "OZF_synthetic_overlap_0_train": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0/train_info.csv",
            "OZF_synthetic_overlap_0_val": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0/val_info.csv",
            "OZF_synthetic_overlap_0_test": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0/test_info.csv",
            "OZF_synthetic_overlap_0.2_train": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0.2/train_info.csv",
            "OZF_synthetic_overlap_0.2_val": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0.2/val_info.csv",
            "OZF_synthetic_overlap_0.2_test": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0.2/test_info.csv",
            "OZF_synthetic_overlap_0.4_train": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0.4/train_info.csv",
            "OZF_synthetic_overlap_0.4_val": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0.4/val_info.csv",
            "OZF_synthetic_overlap_0.4_test": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0.4/test_info.csv",
            "OZF_synthetic_overlap_0.6_train": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0.6/train_info.csv",
            "OZF_synthetic_overlap_0.6_val": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0.6/val_info.csv",
            "OZF_synthetic_overlap_0.6_test": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0.6/test_info.csv",
            "OZF_synthetic_overlap_1_train": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_1/train_info.csv",
            "OZF_synthetic_overlap_1_val": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_1/val_info.csv",
            "OZF_synthetic_overlap_1_test": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_1/test_info.csv",
        },
        version="0.1.0",
        description="Voxaboxen dataset for acoustic sound event detection",
        sources=[
            "Anuraset",
            "BV",
            "MT",
            "OZF",
            "Hawaii",
            "Humpback",
            "Katydids",
            "Powdermill",
        ],
        license="CC BY",
    )

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] = None,
        sample_rate: Optional[int] = None,
        data_root: Optional[str | AnyPathT] = None,
        mono_method: Optional[str] = "average",
    ) -> None:
        """Initialize the Voxaboxen dataset.

        Parameters
        ----------
        split : str
            The split to load. One of info.split_paths keys.
        output_take_and_give : dict[str, str]
            A dictionary mapping the original column names to the new column names.
            It acts as a filter as well.
        sample_rate : int
            The sample rate to which audio files should be resampled.
        data_root : str | AnyPathT, optional
            The root directory for the dataset. This is optionally appended to the
            path item of a sample in the dataset.
            If None, the default is the parent directory of the split path.
        """
        super().__init__(output_take_and_give)  # Initialize the parent Dataset class
        self.split = split
        self._data: pd.DataFrame = None
        self._load()  # Load the dataset (fills self._data)
        self.sample_rate = sample_rate
        self.data_root = data_root
        self.mono_method = mono_method
        if self.data_root is None:
            # we assume that parent dir of the split path is the data root
            self.data_root = anypath(self.info.split_paths[self.split]).parent

    @property
    def columns(self) -> list[str]:
        """Return the columns of the dataset."""
        return list(self._data.columns)

    @property
    def available_splits(self) -> list[str]:
        """Return the available splits of the dataset."""
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        """Load the dataset.

        Raises
        ------
        LookupError
            If the split is not valid.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}."
                "Expected one of {list(self.info.split_paths.keys())}"
            )

        location = self.info.split_paths[self.split]
        # Read CSV content
        csv_text = anypath(location).read_text(encoding="utf-8")
        self._data = pd.read_csv(StringIO(csv_text))

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["Voxaboxen", dict[str, Any]]:
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
            sample_rate=cfg["sample_rate"],
            mono_method=cfg.get("mono_method", "average"),
        )

        if dataset_config.transformations:
            transform_metadata = ds.apply_transformations(dataset_config.transformations)
            return ds, transform_metadata

        return ds, {}

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the current split.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a specific sample from the dataset.
        Parameters
        ----------
        idx : int
            Index of the sample to get.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the audio data, text label, label, and path.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if idx >= len(self._data):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self._data)}.")

        row = self._data.iloc[idx].to_dict()

        # Ensure audio path is valid
        if self.data_root:
            audio_path = anypath(self.data_root) / row["audio_fp"]
        else:
            audio_path = anypath(row["audio_fp"])

        audio, sr = read_audio(audio_path)
        audio = audio.astype(np.float32)

        if self.mono_method:
            audio = audio_stereo_to_mono(audio, mono_method="average")

        if self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )

        row["audio"] = audio

        # read selection table
        row["selection_table"] = pd.read_csv(StringIO(row["selection_table_str"]), sep="\t")

        if self.output_take_and_give:
            item = {}
            for key, value in self.output_take_and_give.items():
                item[value] = row[key]
        else:
            item = row

        return item

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        -------
        Dict[str, Any]
            Each sample in the dataset.
        """
        for idx in range(len(self)):
            yield self[idx]

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


@register_dataset
class VoxaboxenEvents(Dataset):
    """Voxaboxen dataset as events

    Description
    -----------
    Same as Voxaboxen, but the audio is split according to the information
    in the selection table.

    References
    ----------
    Robust detection of overlapping bioacoustic sound events
    Louis Mahon, Benjamin Hoffman, Logan S James, Maddie Cusimano,
    Masato Hagiwara, Sarah C Woolley, Olivier Pietquin
    https://arxiv.org/abs/2503.02389

    Examples
    -------
    >>> from esp_data.datasets import VoxaboxenEvents
    >>> dataset = VoxaboxenEvents(
    ...     split="BV_val",
    ...     output_take_and_give={"selection_table": "st"}
    ... )
    >>> print(dataset.info.name)
    voxaboxen_events
    """

    info = DatasetInfo(
        name="voxaboxen_events",
        owner="benjamin; gagan",
        split_paths={
            "Anuraset_train": "gs://esp-ml-datasets/voxaboxen/files/Anuraset/formatted/train_info.csv",
            "Anuraset_val": "gs://esp-ml-datasets/voxaboxen/files/Anuraset/formatted/val_info.csv",
            "Anuraset_test": "gs://esp-ml-datasets/voxaboxen/files/Anuraset/formatted/test_info.csv",
            "BV_train": "gs://esp-ml-datasets/voxaboxen/files/BV/formatted/train_info.csv",
            "BV_val": "gs://esp-ml-datasets/voxaboxen/files/BV/formatted/val_info.csv",
            "BV_test": "gs://esp-ml-datasets/voxaboxen/files/BV/formatted/test_info.csv",
            "MT_train": "gs://esp-ml-datasets/voxaboxen/files/MT/formatted/train_info.csv",
            "MT_val": "gs://esp-ml-datasets/voxaboxen/files/MT/formatted/val_info.csv",
            "MT_test": "gs://esp-ml-datasets/voxaboxen/files/MT/formatted/test_info.csv",
            "OZF_train": "gs://esp-ml-datasets/voxaboxen/files/OZF/formatted/train_info.csv",
            "OZF_val": "gs://esp-ml-datasets/voxaboxen/files/OZF/formatted/val_info.csv",
            "OZF_test": "gs://esp-ml-datasets/voxaboxen/files/OZF/formatted/test_info.csv",
            "hawaii_train": "gs://esp-ml-datasets/voxaboxen/files/hawaii/formatted/train_info.csv",
            "hawaii_val": "gs://esp-ml-datasets/voxaboxen/files/hawaii/formatted/val_info.csv",
            "hawaii_test": "gs://esp-ml-datasets/voxaboxen/files/hawaii/formatted/test_info.csv",
            "humpback_train": "gs://esp-ml-datasets/voxaboxen/files/humpback/formatted/train_info.csv",
            "humpback_val": "gs://esp-ml-datasets/voxaboxen/files/humpback/formatted/val_info.csv",
            "humpback_test": "gs://esp-ml-datasets/voxaboxen/files/humpback/formatted/test_info.csv",
            "katydids_train": "gs://esp-ml-datasets/voxaboxen/files/katydids/formatted/train_info.csv",
            "katydids_val": "gs://esp-ml-datasets/voxaboxen/files/katydids/formatted/val_info.csv",
            "katydids_test": "gs://esp-ml-datasets/voxaboxen/files/katydids/formatted/test_info.csv",
            "powdermill_train": "gs://esp-ml-datasets/voxaboxen/files/powdermill/formatted/train_info.csv",
            "powdermill_val": "gs://esp-ml-datasets/voxaboxen/files/powdermill/formatted/val_info.csv",
            "powdermill_test": "gs://esp-ml-datasets/voxaboxen/files/powdermill/formatted/test_info.csv",
            "OZF_synthetic_overlap_0_train": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0/train_info.csv",
            "OZF_synthetic_overlap_0_val": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0/val_info.csv",
            "OZF_synthetic_overlap_0_test": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0/test_info.csv",
            "OZF_synthetic_overlap_0.2_train": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0.2/train_info.csv",
            "OZF_synthetic_overlap_0.2_val": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0.2/val_info.csv",
            "OZF_synthetic_overlap_0.2_test": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0.2/test_info.csv",
            "OZF_synthetic_overlap_0.4_train": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0.4/train_info.csv",
            "OZF_synthetic_overlap_0.4_val": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0.4/val_info.csv",
            "OZF_synthetic_overlap_0.4_test": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0.4/test_info.csv",
            "OZF_synthetic_overlap_0.6_train": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0.6/train_info.csv",
            "OZF_synthetic_overlap_0.6_val": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0.6/val_info.csv",
            "OZF_synthetic_overlap_0.6_test": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_0.6/test_info.csv",
            "OZF_synthetic_overlap_1_train": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_1/train_info.csv",
            "OZF_synthetic_overlap_1_val": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_1/val_info.csv",
            "OZF_synthetic_overlap_1_test": "gs://esp-ml-datasets/voxaboxen/files/OZF_synthetic/overlap_1/test_info.csv",
        },
        version="0.1.0",
        description="Voxaboxen events dataset for acoustic sound event detection",
        sources=[
            "Anuraset",
            "BV",
            "MT",
            "OZF",
            "Hawaii",
            "Humpback",
            "Katydids",
            "Powdermill",
        ],
        license="CC BY",
    )

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] = None,
        sample_rate: int = 16000,
        data_root: Optional[str | AnyPathT] = None,
        stereo_or_mono: str = "stereo",
        mono_method: str = "average",
        clip_duration: float = 10.0,
        clip_hop: float = 5.0,
        clip_start_offset: float = 0.0,
        omit_empty_clip_prob: float = 0.0,
        scale_factor: int = 1,
        segmentation_based: bool = True,
        unknown_label: str = "Unknown",
    ) -> None:
        """Initialize the VoxaboxenEvents dataset.

        Parameters
        ----------
        split : str
            The split to load. One of info.split_paths keys.
        output_take_and_give : dict[str, str]
            A dictionary mapping the original column names to the new column names.
            It acts as a filter as well.
        sample_rate : int
            The sample rate to which audio files should be resampled.
        data_root : str | AnyPathT, optional
            The root directory for the dataset. This is optionally appended to the
            path item of a sample in the dataset.
            If None, the default is the parent directory of the split path.
        mono_method : str, optional
            The method to convert stereo audio to mono. Defaults to "average".
            Other options are "average" and "keep_first"
        clip_duration : float, optional
            Duration of each audio clip in seconds.
        clip_hop : float, optional
            Hop size between consecutive audio clips in seconds. If None, the full
            audio is used without overlapping clips.
        clip_start_offset : float, optional
            Offset in seconds to start the first clip. Defaults to 0.0 seconds.
            This is useful for skipping a portion of the audio before the first clip.
        scale_factor : float, optional
            Scale factor for downsampling the audio. This is needed when representations
            have been downsampled by encoders like AVES.
        omit_empty_clip_prob : float, optional
            Probability of omitting empty clips (no annotations).
            Defaults to 0.0, meaning no empty clips are omitted.
        segmentation_based : bool, optional
            If True, the dataset is segmented based on the selection table.
            If False, the entire audio file is treated as a single segment.
            Defaults to True.
        unknown_label : str, optional
            The label used for unknown annotations. Defaults to "unknown".
        """
        super().__init__(output_take_and_give)  # Initialize the parent Dataset class
        self.split = split
        self._data: pd.DataFrame = None
        self._load()  # Load the dataset (fills self._data)
        self.sample_rate = sample_rate
        self.data_root = data_root
        self.stereo_or_mono = stereo_or_mono
        self.mono_method = mono_method
        self.clip_duration = clip_duration
        self.clip_hop = clip_hop
        self.clip_start_offset = clip_start_offset
        self.scale_factor = scale_factor
        self.segmentation_based = segmentation_based
        self.unknown_label = unknown_label

        self.omit_empty_clip_prob = omit_empty_clip_prob
        self.rng = default_rng()
        if self.data_root is None:
            # we assume that parent dir of the split path is the data root
            self.data_root = anypath(self.info.split_paths[self.split]).parent

        self.label_mapping: dict = None
        if split in LABEL_SETS:
            self.label_set: list = LABEL_SETS[split]
        else:
            self.label_set = []

        self.n_classes = 0
        self._create_label_map()

        self._make_metadata()

    @property
    def columns(self) -> list[str]:
        """Return the columns of the dataset."""
        return list(self._data.columns)

    @property
    def available_splits(self) -> list[str]:
        """Return the available splits of the dataset."""
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        """Load the dataset.

        Raises
        ------
        LookupError
            If the split is not valid.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}."
                "Expected one of {list(self.info.split_paths.keys())}"
            )

        location = self.info.split_paths[self.split]
        # Read CSV content
        csv_text = anypath(location).read_text(encoding="utf-8")
        self._data = pd.read_csv(StringIO(csv_text))

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["VoxaboxenEvents", dict[str, Any]]:
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
            sample_rate=cfg.get("sample_rate", 16000),
            mono_method=cfg.get("mono_method", "average"),
            clip_duration=cfg.get("clip_duration", 10.0),
            clip_hop=cfg.get("clip_hop", 5.0),
            clip_start_offset=cfg.get("clip_start_offset", 0.0),
            omit_empty_clip_prob=cfg.get("omit_empty_clip_prob", 0.0),
            scale_factor=cfg.get("scale_factor", 1),
            segmentation_based=cfg.get("segmentation_based", True),
            unknown_label=cfg.get("unknown_label", "Unknown"),
        )

        if dataset_config.transformations:
            transform_metadata = ds.apply_transformations(dataset_config.transformations)
            return ds, transform_metadata

        return ds, {}

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the current split.
        """
        return len(self._metadata)

    def _create_label_map(self) -> None:
        """Create a mapping from label names to indices.

        Raises
        ------
        ValueError
            If no labels are found in the dataset.
        """
        # load all selection tables and create a label set
        if not self.label_set:
            self.label_set = set()
            for row in self._data.itertuples():
                selection_table = pd.read_csv(StringIO(row.selection_table_str), sep="\t")

                labels = selection_table["Annotation"].unique().tolist()
                if not hasattr(self, "label_set"):
                    self.label_set = set(labels)
                else:
                    self.label_set.update(labels)

            self.label_set.add(self.unknown_label)
            self.label_set = list(self.label_set)

        if not self.label_set:
            raise ValueError("No labels found in the dataset.")

        self.label_mapping = {label: label for label in self.label_set}
        self.n_classes = len(self.label_set)

    def _process_selection_table(self, selection_table_str: str) -> IntervalTree:
        """
        Process annotation file into interval tree format.

        Parameters
        ----------
        selection_table_str : str
            String representation of the selection table in TSV format.

        Returns
        -------
        IntervalTree
            Tree containing labeled time intervals
        """
        selection_table = pd.read_csv(StringIO(selection_table_str), sep="\t")
        tree = IntervalTree()

        for _, row in selection_table.iterrows():
            start = row["Begin Time (s)"]
            end = row["End Time (s)"]
            label = row["Annotation"]

            if end <= start:
                continue

            if label in self.label_mapping:
                label = self.label_mapping[label]
            else:
                continue

            if label == self.unknown_label:
                label_idx = -1
            else:
                label_idx = self.label_set.index(label)
            tree.addi(start, end, label_idx)

        return tree

    def _make_metadata(self) -> None:
        """Generate dataset metadata including clip boundaries."""

        selection_table_dict = dict()
        metadata = []

        for _ii, row in self._data.iterrows():
            fn = row["fn"]
            audio_fp = anypath(self.data_root) / row["audio_fp"]

            selection_table = self._process_selection_table(row["selection_table_str"])
            selection_table_dict[fn] = selection_table

            # Determine number of clips based on audio duration
            duration = row["audio_duration"]
            num_clips = max(
                0,
                int(
                    np.floor(
                        (duration - self.clip_duration - self.clip_start_offset) // self.clip_hop
                    )
                ),
            )

            for tt in range(num_clips):
                start = tt * self.clip_hop + self.clip_start_offset
                end = start + self.clip_duration

                ivs: IntervalTree = selection_table[start:end]
                # if no annotated intervals, skip with specified probability
                if not ivs:
                    if self.omit_empty_clip_prob > self.rng.uniform():
                        continue

                metadata.append([fn, str(audio_fp), start, end])

        self._selection_table_dict = selection_table_dict
        self._metadata = metadata

    def _get_pos_intervals(
        self, fn: str, start: float, end: float
    ) -> list[tuple[float, float, str]]:
        """
        Get annotated intervals within specified time range.

        Parameters
        ----------
        fn : str
            Filename identifier
        start : float
            Start time in seconds
        end : float
            End time in seconds

        Returns
        -------
        list[tuple[float, float, str]]
            List of tuples containing (start, end, label) for each interval
        """

        tree = self._selection_table_dict[fn]

        intervals = tree[start:end]
        intervals = [
            (max(iv.begin, start) - start, min(iv.end, end) - start, iv.data) for iv in intervals
        ]

        return intervals

    def _get_class_proportions(self) -> np.ndarray:
        """
        Calculate class distribution in dataset.

        Returns
        -------
        numpy.ndarray
            Array of class proportions
        """

        counts = np.zeros((self.n_classes,))

        for k in self.selection_table_dict:
            st = self.selection_table_dict[k]
            for interval in st:
                annot = interval.data
                if annot == -1:
                    continue
                else:
                    counts[annot] += 1

        total_count = np.sum(counts)
        proportions = counts / total_count

        return proportions

    def _get_annotation(
        self, pos_intervals: list[tuple[float, float, int]], audio: np.ndarray
    ) -> tuple[
        np.ndarray,  # anchor_annos
        np.ndarray,  # regression_annos
        np.ndarray,  # class_annos
        np.ndarray,  # rev_anchor_annos
        np.ndarray,  # rev_regression_annos
        np.ndarray,  # rev_class_annos
    ]:
        """
        Generate target annotations from positive intervals.

        Parameters
        ----------
        pos_intervals : list
            List of (start, end, label_idx) tuples
        audio : np.ndarray
            Input audio tensor

        Returns
        -------
        tuple
            Tuple containing:
            - anchor_annos: Anchor point annotations
            - regression_annos: Duration annotations
            - class_annos: Class probability annotations
            - rev_anchor_annos: Reverse anchor points
            - rev_regression_annos: Reverse duration
            - rev_class_annos: Reverse class probabilities
        """

        raw_seq_len = audio.shape[-1]
        seq_len = int(math.ceil(raw_seq_len / self.scale_factor))

        regression_annos = np.zeros((seq_len,))
        class_annos = np.zeros((seq_len, self.n_classes))
        anchor_annos = [
            np.zeros(
                seq_len,
            )
        ]
        rev_regression_annos = np.zeros((seq_len,))
        rev_class_annos = np.zeros((seq_len, self.n_classes))
        rev_anchor_annos = [
            np.zeros(
                seq_len,
            )
        ]

        for iv in pos_intervals:
            start, end, class_idx = iv
            dur = end - start
            dur_samples = np.ceil(dur * self.sample_rate)

            start_idx = int(math.floor(start * self.sample_rate))
            start_idx = max(min(start_idx, seq_len - 1), 0)

            end_idx = int(math.ceil(end * self.sample_rate))
            end_idx = max(min(end_idx, seq_len - 1), 0)
            dur_samples = int(np.ceil(dur * self.sample_rate))

            anchor_anno = _get_anchor_anno(start_idx, dur_samples, seq_len)
            anchor_annos.append(anchor_anno)
            regression_annos[start_idx] = dur

            rev_anchor_anno = _get_anchor_anno(end_idx, dur_samples, seq_len)
            rev_anchor_annos.append(rev_anchor_anno)
            rev_regression_annos[end_idx] = dur

            if self.segmentation_based:
                if class_idx == -1:
                    pass
                else:
                    class_annos[start_idx : start_idx + dur_samples, class_idx] = 1.0

            else:
                if class_idx != -1:
                    class_annos[start_idx, class_idx] = 1.0
                    rev_class_annos[end_idx, class_idx] = 1.0
                else:
                    class_annos[start_idx, :] = (
                        1.0 / self.n_classes
                    )  # if unknown, enforce uncertainty
                    rev_class_annos[end_idx, :] = (
                        1.0 / self.n_classes
                    )  # if unknown, enforce uncertainty

        anchor_annos = np.stack(anchor_annos)
        anchor_annos = np.amax(anchor_annos, axis=0)
        rev_anchor_annos = np.stack(rev_anchor_annos)
        rev_anchor_annos = np.amax(rev_anchor_annos, axis=0)
        # shapes [time_steps, ], [time_steps, ], [time_steps, n_classes] (times two)
        return (
            anchor_annos,
            regression_annos,
            class_annos,
            rev_anchor_annos,
            rev_regression_annos,
            rev_class_annos,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a specific sample from the dataset.
        Parameters
        ----------
        idx : int
            Index of the sample to get.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the audio data, text label, label, and path.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if idx >= len(self._data):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self._data)}.")

        fn, audio_fp, start, end = self._metadata[idx]

        # Read audio clip
        audio, sr = read_audio(audio_fp, start_time=start, end_time=end)
        audio = audio.astype(np.float32)

        if self.stereo_or_mono == "mono":
            audio = audio_stereo_to_mono(audio, mono_method=self.mono_method)
        else:
            channel_dim = np.argmin(audio.shape)
            if channel_dim != 0:
                audio = audio.T

        if self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )

        pos_intervals = self._get_pos_intervals(fn, start, end)
        (
            anchor_anno,
            regression_anno,
            class_anno,
            rev_anchor_anno,
            rev_regression_anno,
            rev_class_anno,
        ) = self._get_annotation(pos_intervals, audio)

        row = {
            "audio": audio,
            "fn": fn,
            "audio_fp": audio_fp,
            "start": start,
            "end": end,
            "anchor_anno": anchor_anno,
            "regression_anno": regression_anno,
            "class_anno": class_anno,
            "rev_regression_anno": rev_regression_anno,
            "rev_anchor_anno": rev_anchor_anno,
            "rev_class_anno": rev_class_anno,
        }

        if self.output_take_and_give:
            item = {}
            for key, value in self.output_take_and_give.items():
                item[value] = row[key]
        else:
            item = row

        return item

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        -------
        Dict[str, Any]
            Each sample in the dataset.
        """
        for idx in range(len(self)):
            yield self[idx]

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


def _get_anchor_anno(start_idx: int, dur_samples: int, seq_len: int) -> np.ndarray:
    """
    Represent start idx as a Gaussian blurred onehot encoding.

        Parameters
    ----------
    start_idx : int
        Start index of annotation
    dur_samples : int
        Duration in samples
    seq_len : int
        Total sequence length

    Returns
    -------
    numpy.ndarray
        Anchor point annotations

    Notes
    -----
    This setting of `std` follows CornerNet, where adaptive standard deviation
    is set to 1/3 image radius.

    """

    std = dur_samples / 6
    x = (np.arange(seq_len) - start_idx) ** 2
    x = x / (2 * std**2)
    x = np.exp(-x)
    return x
