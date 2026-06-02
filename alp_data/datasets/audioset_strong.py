"""AudioSet Strong dataset"""

from __future__ import annotations

from io import StringIO
from typing import Any, Dict, Iterator, List

import librosa
import numpy as np
import polars as pl

from alp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from alp_data.backends import BackendType
from alp_data.io import DATA_HOME, AnyPathT, anypath, audio_stereo_to_mono, read_audio

_CSV_ROOT = f"{DATA_HOME}/audioset/v0.2.0/raw/csv-data"


@register_dataset
class AudioSetStrong(Dataset):
    """AudioSet Strong Dataset

    Description
    -----------
    AudioSet Strong is a strongly-labeled subset of AudioSet with temporal annotations
    (start and end times) for sound events. This dataset provides precise timing
    information for when each sound event occurs within the 10-second audio clips.

    AudioSet is a large-scale dataset of manually-annotated audio events that endeavors
    to bridge the gap in data availability between image and audio research, using a
    carefully structured hierarchical ontology of 632 audio classes in 10-second
    segments of YouTube videos.

    This class makes the AudioSet Strong subset available in the alp-data strongly-labeled
    format, where each entry consists of:
    - An audio recording (10 seconds, pre-resampled to 32kHz)
    - A selection table with temporal annotations (begin time, end time, label)

    The strong labels provide temporal boundaries for sound events, making this dataset
    suitable for sound event detection and temporal localization tasks.

    AudioSet recordings include those available in this huggingface dataset:
    https://huggingface.co/datasets/agkphysics/AudioSet

    Available Splits
    ----------------
    - ``train``: AudioSet Strong training set with 32kHz pre-resampled audio (8115 rows).
    - ``train-environmental``: Filtered to rows where ALL labels are environmental sounds
      (from AudioSet's environmental subset). 1109 rows.

    References
    ----------
    AUDIO SET: AN ONTOLOGY AND HUMAN-LABELED DATASET FOR AUDIO EVENTS
    Gemmeke et al. 2017
    https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45857.pdf

    AudioSet Homepage:
    https://research.google.com/audioset/

    Examples
    --------
    >>> from alp_data.datasets import AudioSetStrong
    >>> dataset = AudioSetStrong(split="train", sample_rate=32000)
    >>> print(len(dataset))
    8115
    >>> item = dataset[0]
    >>> keys = sorted([k for k in item.keys() if k != '32khz_path'])
    >>> len(keys)
    7
    >>> 'sample_rate' in keys and 'audio' in keys
    True
    >>> print(list(item['selection_table'].columns))
    ['Selection', 'Begin Time (s)', 'End Time (s)', 'Label']

    >>> env_dataset = AudioSetStrong(split="train-environmental", sample_rate=32000)
    >>> print(len(env_dataset))
    1109
    """

    info = DatasetInfo(
        name="audioset_strong",
        owner="david; marius; masato",
        split_paths={
            "train": f"{_CSV_ROOT}/audioset_train_strong_32khz_only.csv",
            "train-environmental": f"{_CSV_ROOT}/audioset_train_strong_32khz_environmental.csv",
        },
        version="0.1.0",
        description="AudioSet Strong: Strongly-labeled subset with temporal annotations",
        sources=["YouTube"],
        license="CC BY 4.0",
    )

    _sample_rate_paths = {
        32000: "32khz_path",
    }
    _originals_path_column = "audio_path"

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: Dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        split : str
            Split to load. Available splits:
            - "train": Full set with 32kHz pre-resampled audio (8115 rows)
            - "train-environmental": Environmental sounds only (1109 rows)
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys (filters columns as well).
        sample_rate : int | None
            Target sample rate for audio. If sample_rate=32000, pre-resampled audio
            is loaded directly. Other sample rates resample on-the-fly.
        data_root : str | AnyPathT | None
            Optional root directory to prepend to each row['audio_path'].
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars"
        streaming : bool, optional
            Whether to use streaming mode, by default False
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self.annotation_columns = ["Label"]

        self.sample_rate = sample_rate
        self.data_root = anypath(data_root) if data_root is not None else None

        # Load split CSV
        self._load()

        # If no explicit data_root, set to the raw directory (go up two levels from csv file)
        if self.data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent.parent

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    @property
    def available_sample_rates(self) -> list[int]:
        """Return the available pre-resampled sample rates.

        Returns
        -------
        list[int]
            List of sample rates (in Hz) for which pre-resampled audio is available.
            Audio at these sample rates can be loaded directly without on-the-fly
            resampling. This checks which path columns actually exist in the loaded data.
        """
        available = []
        for sr, path_column in self._sample_rate_paths.items():
            if path_column in self._data.columns:
                available.append(sr)
        return available

    def _load(self) -> None:
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        location = self.info.split_paths[self.split]
        self._data = self._backend_class.from_csv(
            location, streaming=self._streaming, keep_default_na=False, na_values=[""]
        )

    def __len__(self) -> int:
        return len(self._data)

    @staticmethod
    def _empty_selection_table() -> pl.DataFrame:
        # Default Raven-style selection table columns we expect for strong labels.
        return pl.DataFrame(
            schema={
                "Selection": pl.Int64,
                "Begin Time (s)": pl.Float64,
                "End Time (s)": pl.Float64,
                "Label": pl.Utf8,
            }
        )

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        audio = None
        sr = None

        # Use pre-resampled audio if available for the requested sample rate
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            path_column = self._sample_rate_paths[self.sample_rate]
            if (
                path_column in row
                and row[path_column] not in (None, "")
                and not (isinstance(row[path_column], float) and np.isnan(row[path_column]))
            ):
                presampled_path = self.data_root / str(row[path_column])
                try:
                    audio, sr = read_audio(presampled_path)
                    sample_rate = sr
                    audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
                    # Validate audio length (corrupt files may be very short)
                    if len(audio) < self.sample_rate:
                        audio = None
                except Exception:
                    audio = None

        # Fall back to original audio with on-the-fly resampling if needed
        if audio is None:
            audio_path = (
                (self.data_root / row[self._originals_path_column])
                if self.data_root
                else anypath(row[self._originals_path_column])
            )
            audio, sample_rate = read_audio(audio_path)
            audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

            # Resample if necessary
            if self.sample_rate is not None and sample_rate != self.sample_rate:
                audio = librosa.resample(
                    y=audio,
                    orig_sr=sample_rate,
                    target_sr=self.sample_rate,
                    scale=True,
                    res_type="kaiser_best",
                )
                sample_rate = self.sample_rate

        # Selection table (using polars for ~5x faster parsing)
        selection_table_blob = row.get("selection_table", "")
        if selection_table_blob is None or selection_table_blob == "":
            st = self._empty_selection_table()
        else:
            st = pl.read_csv(StringIO(selection_table_blob), separator="\t")

        # Clip events outside audio (keep only events that begin before audio end)
        audio_dur = len(audio) / float(sample_rate)
        if "Begin Time (s)" in st.columns:
            st = st.filter(pl.col("Begin Time (s)") < audio_dur)

        # Build output
        row["audio"] = audio
        row["sample_rate"] = sample_rate
        row["selection_table"] = (
            st.to_pandas()
        )  # to adhere to the rest of the selection_table datasets

        if self.output_take_and_give:
            item: dict[str, Any] = {}
            for old_key, new_key in self.output_take_and_give.items():
                item[new_key] = row[old_key]
            return item

        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self._data):
            raise IndexError(f"Index {idx} out of bounds for dataset length {len(self._data)}")

        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["AudioSetStrong", dict[str, Any]]:
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
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def get_available_labels(self) -> List[str]:
        """
        Return all possible labels found in the dataset.

        Returns
        -------
        List[str]
            A sorted list of all unique labels in the dataset.
        """
        labels: set[str] = set()
        for row in self._data:
            selection_table_blob = row.get("selection_table", "")
            if selection_table_blob is None or selection_table_blob == "":
                continue
            st = pl.read_csv(StringIO(selection_table_blob), separator="\t")
            if "Label" in st.columns:
                labels.update(st["Label"].cast(pl.Utf8).to_list())

        return sorted(labels)

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
