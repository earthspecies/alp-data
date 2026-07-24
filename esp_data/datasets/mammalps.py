"""MammAlps — multi-view audiovisual wild-mammal behavior dataset (CVPR 2025).

MammAlps (Gabeff et al., 2025, "MammAlps: A Multi-view Video Behavior
Monitoring Dataset of Wild Mammals in the Swiss Alps", CVPR; repo
``eceo-epfl/MammAlps``) is a multimodal wildlife-behavior dataset recorded by
nine camera traps (with built-in microphones) in the Swiss National Park.

This class exposes **Benchmark I** (multimodal species + behavior recognition):
6,135 single-animal clips, each a video file *with an aligned audio track*,
labelled with a species (5), an activity (11), and one or more actions (19).
Because every clip carries both modalities, it supports a SSW60-style
vision-only / audio-only / vision+audio evaluation from a single file.

Each row is one clip with ``modality="video"``; ``_load_video`` decodes the
frames *and* the aligned audio track (resampled to ``sample_rate``), exactly
as SSW60 does for its video rows.

The build script (``scripts/data_preprocessing_scripts/mammalps/
build_mammalps.py``) parses the Benchmark I split CSVs + ``labels_mapping_b1``
and uploads the clips to ``gs://esp-data-ingestion/mammalps/v0.1.0/``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_video

_GCS_ROOT = "gs://esp-data-ingestion/mammalps/v0.1.0"


@register_dataset
class MammAlps(Dataset):
    """MammAlps — Benchmark I audiovisual wild-mammal behavior dataset.

    Description
    -----------
    Single-animal clips (video with an aligned audio track) of five wild
    mammal species in the Swiss Alps, each labelled with an ``activity``
    (behavior; 11 classes), a ``species`` (5), and one or more ``actions``
    (19). Every clip carries both video and audio, enabling a vision-only /
    audio-only / vision+audio evaluation from the same file.

    Columns
    -------
    asset_id : str
        Benchmark I clip / sample id (``sample_id`` in the source CSVs).
    modality : str
        Always ``"video"`` (each clip is a video with an aligned audio track).
    activity : str
        Activity (behavior) label — the primary classification target.
    activity_label : int
        0-based activity class index (per ``labels_mapping_b1``).
    species : str
        Species label.
    actions : str
        The clip's action set as a ``", "``-joined string (multi-label).
    split : str
        Source split (``train`` / ``val`` / ``test``).
    video_path : str
        Absolute ``gs://`` path to the mp4 clip (contains the audio track).

    Splits
    ------
    - ``all`` : every clip.
    - ``train`` / ``val`` / ``test`` : the official day-level splits.

    Loader behaviour
    ----------------
    - ``video`` → decodes frames into ``video_frames`` (T,H,W,C uint8) *and*
      the aligned audio track into ``audio`` + ``sample_rate`` (mono,
      resampled to ``sample_rate``), honouring ``max_frames`` / ``target_fps``.

    References
    ----------
    - Gabeff et al. (2025) "MammAlps: A Multi-view Video Behavior Monitoring
      Dataset of Wild Mammals in the Swiss Alps", CVPR. arXiv:2503.18223;
      data: Zenodo 10.5281/zenodo.15040901; repo ``eceo-epfl/MammAlps``.

    License: MIT (per the MammAlps release); staged on internal GCS for ESP
    research.
    """

    info = DatasetInfo(
        name="mammalps",
        owner="david",
        split_paths={
            "all": f"{_GCS_ROOT}/mammalps_all.csv",
            "train": f"{_GCS_ROOT}/mammalps_train.csv",
            "val": f"{_GCS_ROOT}/mammalps_val.csv",
            "test": f"{_GCS_ROOT}/mammalps_test.csv",
        },
        version="0.1.0",
        description=(
            "MammAlps Benchmark I: 6,135 single-animal audiovisual clips of "
            "five wild mammal species, labelled with activity (11), species "
            "(5) and actions (19). Video clips carry an aligned audio track, "
            "supporting SSW60-style vision / audio / vision+audio evaluation."
        ),
        sources=("Gabeff et al. (2025) CVPR, arXiv:2503.18223; Zenodo 10.5281/zenodo.15040901"),
        license="MIT (MammAlps); internal ESP research staging",
    )

    _mixup_group = "animal"

    def __init__(
        self,
        split: str = "test",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
        max_frames: int | None = 16,
        target_fps: float | None = None,
        with_video_audio: bool = True,
    ) -> None:
        """Initialise the MammAlps dataset.

        Parameters
        ----------
        split : str
            Split to load (key in :attr:`info.split_paths`).
        output_take_and_give : dict[str, str] | None
            Optional column rename / selection mapping.
        sample_rate : int | None
            Target audio sample rate for the aligned track. ``None`` keeps the
            native rate.
        data_root : str | AnyPathT | None
            Unused — path columns hold absolute ``gs://`` URIs. Accepted for
            API parity with other datasets.
        backend : BackendType
            ``"polars"`` or ``"pandas"``.
        streaming : bool
            Whether to use streaming mode.
        max_frames : int | None
            Maximum number of frames to decode per clip. ``None`` decodes
            every frame. Defaults to 16.
        target_fps : float | None
            If set, frames are subsampled to approximately this frame rate.
            Defaults to None (keep native fps).
        with_video_audio : bool
            Whether to decode the aligned audio track. Defaults to True.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.with_video_audio = with_video_audio
        self._data = None
        self.data_root = anypath(data_root) if data_root else None
        self._load()

    def _load(self) -> None:
        """Load the split CSV.

        Raises
        ------
        LookupError
            If the split is not valid.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        self._data = self._backend_class.from_csv(
            self.info.split_paths[self.split],
            streaming=self._streaming,
            keep_default_na=False,
            na_values=[""],
        )

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    def _load_video(self, row: dict[str, Any]) -> dict[str, Any]:
        """Populate ``video_frames`` plus the aligned ``audio`` for a clip.

        Returns
        -------
        dict[str, Any]
            The row with ``video_frames`` / ``fps`` and (when available)
            ``audio`` / ``sample_rate`` populated.
        """
        decoded = read_video(
            str(row["video_path"]),
            max_frames=self.max_frames,
            target_fps=self.target_fps,
            with_audio=self.with_video_audio,
        )
        row["video_frames"] = decoded["frames"]
        row["fps"] = decoded["fps"]
        audio = decoded["audio"]
        sr = decoded["sample_rate"]
        if audio is not None:
            audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
            if self.sample_rate is not None and sr is not None and sr != self.sample_rate:
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
        return row

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Load + return one clip.

        Returns
        -------
        dict[str, Any]
            The row with ``video_frames`` and aligned ``audio`` populated.
        """
        row = self._load_video(row)
        row["mixup_group"] = self._mixup_group
        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}
        return row

    def __len__(self) -> int:
        """Return the number of clips in the split.

        Returns
        -------
        int
            Number of clips in the current split.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        NotImplementedError
            If the dataset is in streaming mode.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError("Length is not available in streaming mode.")
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single processed clip.

        Returns
        -------
        dict[str, Any]
            The processed row (frames + aligned audio + metadata).
        """
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over processed clips.

        Yields
        ------
        dict[str, Any]
            Each processed clip.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple[MammAlps, dict[str, Any]]:
        """Create a MammAlps instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Dataset configuration.

        Returns
        -------
        tuple[MammAlps, dict[str, Any]]
            The dataset instance and any transformation metadata.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            sample_rate=cfg["sample_rate"],
            data_root=cfg["data_root"],
            backend=cfg["backend"],
            streaming=cfg["streaming"],
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        n = len(self) if self._data is not None and not self._streaming else "?"
        return (
            f"{base}\n"
            f"Assets: {n}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
