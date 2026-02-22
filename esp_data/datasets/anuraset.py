"""Anuraset dataset"""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator, List

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

# Pre-resampled path columns
_SAMPLE_RATE_PATHS: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}


@register_dataset
class AnuraSetStrong(Dataset):
    """AnuraSetStrong Dataset

    Description
    -----------
    This is the strongly labeled portion of AnuraSet, i.e. the portion with
    start- and stop-times annotated.

    Description from "AnuraSet: A dataset for benchmarking Neotropical anuran
    calls identification in passive acoustic monitoring" by Canas et al. (2023)

    "We introduce a large-scale multi-species dataset of anuran amphibians
    calls recorded by PAM, that comprises 27 hours of expert annotations
    for 42 different species from two Brazilian biomes.

    To provide precise annotations, we identified bouts of advertisement
    calls within each audio file and generated strong labels for them (step 1).
    Using Audacity 3.2 software, we conducted a detailed visual and aural
    inspection of the spectrogram to identify temporal limits (beginning and end)
    of audio segments containing species-specific calls with an inter-call interval
    of less than 1 second. These annotations ensured fine-scale specificity (Figure 3).
    For longer intervals, we split the calls into different time boxes and labeled
    them independently. Detailed labels assigned to time boxes were composed of (i)
    the species ID, tagged with a unique 6-letter code built from the scientific
    name of each identified species (Table 2), and (ii) the perceived quality of the
    recorded signal, included as a single letter indicating a Low ('L'), Medium ('M'),
    or High ('H') quality (Figure 4). To ensure consistency among the perceptual quality
    labels, we set up the following criteria: A high-quality call has a high signal-to-noise
    ratio, no overlap with other sounds, has a well-identifiable structure on the spectrogram,
    and can be easily visualized on the oscillogram. A medium-quality call can be
    visually identified on the spectrogram but may overlap with other sounds that can be
    difficult to identify in the oscillogram. A low-quality call shows a low signal-to-noise
    ratio, is partially masked by other sounds, appears with low intensity on the spectrogram,
    and cannot be easily identified on the oscillogram. This information was used to increase
    the usability of the data and improve the error analysis of the learning model."

    Note that we omitted the quality assessments.

    Each entry consists of:
    - an audio recording
    - a selection table (Raven format), with Species labels

    Pre-resampled Audio
    -------------------
    Pre-resampled audio is available at 16 kHz and 32 kHz. When
    ``sample_rate`` matches one of these rates, the pre-resampled files are
    loaded directly (no on-the-fly resampling). For any other target rate,
    audio is resampled on-the-fly using librosa's ``kaiser_best`` method.

    References
    ----------
    https://arxiv.org/pdf/2307.06860

    """

    info = DatasetInfo(
        name="anuraset_strong",
        owner="benjamin",
        split_paths={
            "all": "gs://esp-ml-datasets/anuraset/anuraset_all_gbif.csv",
        },
        version="0.1.0",
        description="[MISSING]",
        sources="Zenodo",
        license="CC BY 1.0",
    )

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        split : str
            Split to load (key in info.split_paths).
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys (filters columns as well).
        sample_rate : int | None
            If set, audio is resampled to this rate.
        data_root : str | AnyPathT | None
            Optional root directory to prepend to each row['audio_path'].
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars"
        streaming : bool, optional
            Whether to use streaming mode, by default False
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.annotation_columns = ["Species"]

        self.sample_rate = sample_rate
        self.data_root = anypath(data_root) if data_root is not None else None
        self._data = None

        # Load split CSV
        self._load()

        # If no explicit data_root, assume parent dir of the split path
        if self.data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent

        self._data_root_16k = self.data_root / "audio_16k"
        self._data_root_32k = self.data_root / "audio_32k"

    @property
    def columns(self) -> list[str]:
        return self._data.columns

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    def _resolve_audio_path(self, row: dict[str, Any]) -> tuple[AnyPathT, bool]:
        """Return (full_audio_path, is_presampled). Prefers pre-resampled when available."""
        if self.sample_rate is not None and self.sample_rate in _SAMPLE_RATE_PATHS:
            col = _SAMPLE_RATE_PATHS[self.sample_rate]
            if col in row and row[col] is not None and str(row[col]).strip():
                root = self._data_root_16k if self.sample_rate == 16000 else self._data_root_32k
                return root / row[col], True
        return self.data_root / row["audio_path"], False

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
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. Iterate over the dataset instead."
            )
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row of the dataset.

        Parameters
        ----------
        row : dict[str, Any]
            A dictionary representing a single row of the dataset.

        Returns
        -------
        dict[str, Any]
            The processed row.
        """
        audio_path, is_presampled = self._resolve_audio_path(row)

        # Read audio
        audio, sr = read_audio(audio_path)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        # Resample on-the-fly only when not using pre-resampled
        target_sr = self.sample_rate
        if not is_presampled and target_sr is not None and sr != target_sr:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=target_sr,
                scale=True,
                res_type="kaiser_best",
            )
            sr = target_sr

        # Selection table
        st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")

        # Clip events outside audio (keep only events that begin before audio end)
        audio_dur = len(audio) / float(sr)
        st = st[st["Begin Time (s)"] < audio_dur].copy()

        # Build output
        row["audio"] = audio
        row["selection_table"] = st

        if self.output_take_and_give:
            item = {}
            for old_key, new_key in self.output_take_and_give.items():
                item[new_key] = row[old_key]
            return item

        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a specific sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to get.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the processed data.
        """
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        -------
        dict[str, Any]
            Each sample in the dataset.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["AnuraSetStrong", dict[str, Any]]:
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

    def get_available_labels(self, anno_column: str = "Species") -> List[str]:
        """
        Return all possible labels for a given annotation column

        Returns
        ---------
        A list of all the available labels for anno_column
        """
        available_labels = set()
        for _, row in self._data.iterrows():
            st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
            available_labels.update(st[anno_column].astype(str).tolist())

        return sorted(available_labels)

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
