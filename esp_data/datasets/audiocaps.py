"""AudioCaps dataset — human-written captions for AudioSet 10-s clips."""

from __future__ import annotations

from typing import Any, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://esp-data-ingestion/audiocaps/v0.1.0"

# AudioCaps clip audio lives under the existing AudioSet v0.2.0 bucket;
# the manifest's audio_path / 16khz_path / 32khz_path columns are relative
# to ``_AUDIOSET_DATA_ROOT``.
_AUDIOSET_DATA_ROOT = "gs://esp-ml-datasets/audioset/v0.2.0/raw/"


@register_dataset
class AudioCaps(Dataset):
    """AudioCaps Dataset.

    Description
    -----------
    AudioCaps (Kim et al., NAACL-HLT 2019) pairs 10-second YouTube/AudioSet
    clips with **human-written captions describing the auditory scene**. Each
    train clip has one caption; each validation/test clip has five
    independently-written captions.

    The audio for every clip is sourced from our existing AudioSet v0.2.0
    GCS dump (``gs://esp-ml-datasets/audioset/v0.2.0/raw/audio_32khz/...``).
    No additional audio download is required: at ingest time, every AudioCaps
    ``youtube_id`` was looked up against the AudioSet inventory and rows
    pointing at YouTube videos that no longer exist were dropped. Net
    retention was ~91% (52,183 captions across 46,831 unique clips).

    Splits
    ------
        - ``all``   : every kept row across all three official AudioCaps splits.
        - ``train`` : 45,493 captions / 45,493 unique clips (one caption per clip).
        - ``val``   : 2,245 captions / 449 unique clips (5 captions per clip).
        - ``test``  : 4,445 captions / 889 unique clips (5 captions per clip).

    Pre-resampled Audio
    -------------------
    Originals from AudioSet are at the YouTube video sample rate (variable).
    Pre-resampled 32 kHz audio is shipped under
    ``audio_32khz/{eval_segments,unbalanced_train_segments}/<yid>.wav`` and
    is the default load path for ``sample_rate=32000``. Pre-resampled 16 kHz
    versions are available under ``audio_16khz/...`` when present.

    References
    ----------
    Kim et al. (2019) "AudioCaps: Generating Captions for Audios in The Wild",
    NAACL-HLT 2019. https://github.com/cdjkim/audiocaps
    DOI: 10.18653/v1/N19-1011
    """

    info = DatasetInfo(
        name="audiocaps",
        owner="david",
        split_paths={
            "all": f"{_GCS_ROOT}/all.csv",
            "train": f"{_GCS_ROOT}/train.csv",
            "val": f"{_GCS_ROOT}/val.csv",
            "test": f"{_GCS_ROOT}/test.csv",
        },
        version="0.1.0",
        description=(
            "AudioCaps — human-written captions for 10-s AudioSet/YouTube "
            "clips (Kim et al., NAACL 2019). 46,831 unique clips / 52,183 "
            "caption rows (train: 1 caption/clip; val+test: 5 captions/clip). "
            "Audio resolved against AudioSet v0.2.0 on GCS; no additional "
            "audio download required."
        ),
        sources=["https://github.com/cdjkim/audiocaps"],
        license="academic-use-only",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _originals_path_column = "audio_path"

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "pandas",
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        split : str
            Split to load (key in info.split_paths): ``all`` / ``train`` /
            ``val`` / ``test``.
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys (filters columns).
        sample_rate : int | None
            If set, audio is loaded at this rate (defaults to 32 kHz which is
            the native rate of the AudioSet v0.2.0 pre-resampled audio).
        data_root : str | AnyPathT | None
            Root directory prepended to row paths. Defaults to the AudioSet
            v0.2.0 GCS root, since AudioCaps' audio columns are relative to
            that location.
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "pandas".
        streaming : bool, optional
            Whether to use streaming mode, by default False.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self.sample_rate = sample_rate

        self._load()

        if data_root is None:
            # Default: AudioSet bucket root, since manifest audio paths are
            # relative to that location (audio_32khz/<subdir>/<yid>.wav).
            self.data_root = anypath(_AUDIOSET_DATA_ROOT)
        else:
            self.data_root = anypath(data_root)

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    @property
    def available_sample_rates(self) -> list[int]:
        """Return pre-resampled sample rates whose path columns exist.

        Returns
        -------
        list[int]
            Sample rates (Hz) for which pre-resampled audio is available.
        """
        return [sr for sr, col in self._sample_rate_paths.items() if col in self._data.columns]

    def _load(self) -> None:
        """Load the manifest CSV for ``self.split``.

        Raises
        ------
        LookupError
            If ``self.split`` is not a key in ``info.split_paths``.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        location = self.info.split_paths[self.split]
        self._data = self._backend_class.from_csv(
            location,
            streaming=self._streaming,
            keep_default_na=False,
            na_values=[""],
        )

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
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. Iterate over the dataset instead."
            )
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row: load audio, return caption + metadata.

        Returns
        -------
        dict[str, Any]
            Row dict with ``audio`` (numpy float32 array, mono) and
            ``sample_rate`` (int) populated.
        """
        use_presampled = False
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            path_column = self._sample_rate_paths[self.sample_rate]
            if path_column in row and row[path_column] is not None and row[path_column] != "":
                audio_path = anypath(self.data_root) / row[path_column]
                use_presampled = True

        if not use_presampled:
            audio_path = anypath(self.data_root) / row[self._originals_path_column]

        audio, sr = read_audio(audio_path)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        if not use_presampled and self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
            sr = self.sample_rate

        row["audio"] = audio
        row["sample_rate"] = sr

        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}
        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a specific sample from the dataset.

        Returns
        -------
        dict[str, Any]
            Dict with audio, sample_rate, caption, youtube_id, etc.
        """
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        ------
        dict[str, Any]
            Each sample in the dataset.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["AudioCaps", dict[str, Any]]:
        """Create an AudioCaps instance from a DatasetConfig.

        Returns
        -------
        tuple[AudioCaps, dict[str, Any]]
            The instantiated dataset + apply_transformations metadata (empty
            when no transforms are configured).
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
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
