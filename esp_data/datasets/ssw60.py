"""SSW60 — Sapsucker Woods 60 audiovisual bird dataset (ECCV 2022).

SSW60 (Van Horn et al., 2022, arXiv:2207.10664; repo ``visipedia/ssw60``)
is a fine-grained *audiovisual* bird dataset covering the same 60 species
across three modalities:

- **audio** — 3,861 Macaulay-Library focal recordings (22.05 kHz mono);
- **video** — 5,400 expert-curated clips, each intrinsically audio+visual
  (the only instance-level audio↔visual pairing in the dataset is the
  audio track *inside* each video);
- **image** — 21,600 iNat2021 images + 10,221 NABirds images.

This is esp-data's first multimodal dataset. Each row carries a
``modality`` column (``audio`` / ``video`` / ``image``) and `_process`
dispatches accordingly: audio rows load audio, image rows load an image
array, and video rows return both decoded frames *and* the aligned audio
track from the same file. Cross-modal alignment across the four source
collections is at the species-label level (the shared 60-class ``label``).

Phase 2 (deferred): extracting the audio of ``reliable_audio`` videos into
standalone audio training clips. For now only the 3,861 stand-alone
Macaulay clips are exposed as ``modality=audio``; video rows still expose
their own audio track at load time.

The build script (``scripts/data_preprocessing_scripts/ssw60/
build_ssw60.py``) downloads the single public S3 tarball, GBIF-links the
60 taxa, resamples the audio to 16 kHz / 32 kHz mirrors, and uploads the
full multimodal set to ``gs://esp-data-ingestion/ssw60/v0.1.0/``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio, read_image, read_video

_GCS_ROOT = "gs://esp-data-ingestion/ssw60/v0.1.0"

MODALITIES = ["audio", "video", "image"]


@register_dataset
class SSW60(Dataset):
    """SSW60 — Sapsucker Woods 60 audiovisual bird dataset.

    Description
    -----------
    A fine-grained audiovisual dataset of 60 bird species from Sapsucker
    Woods (Ithaca, NY), spanning three modalities — audio (Macaulay
    Library focal recordings), video (expert-curated clips with an aligned
    audio track), and images (iNat2021 + NABirds) — all labelled to the
    same 60-class taxonomy and GBIF-linked.

    Each row is a single asset; the ``modality`` column selects how the
    loader decodes it. Audio has pre-resampled 16 kHz / 32 kHz mirrors;
    images and videos are stored in their native encodings and decoded
    lazily at load time.

    Columns
    -------
    asset_id : str
        Source asset identifier (Macaulay / iNat / NABirds id).
    modality : str
        One of ``audio`` / ``video`` / ``image`` (see :data:`MODALITIES`).
    label : int
        0-based SSW60 class index (0–59).
    species_code : str
        eBird / Clements species code.
    canonical_name : str
        GBIF canonical scientific name.
    species_common : str
        English common name.
    family, order : str
        Source taxonomy fields from ``taxa.csv``.
    kingdom, phylum, class, genus, gbifID, taxonKey : str
        GBIF taxonomy fields populated by the builder.
    split : str
        Source split label (``train`` / ``test`` / ``validation``).
    audio_path, 16khz_path, 32khz_path : str
        Absolute ``gs://`` paths to the source 22.05 kHz WAV and the
        pre-resampled mirrors (populated for ``modality=audio``).
    image_path : str
        Absolute ``gs://`` path to the image (populated for
        ``modality=image``).
    video_path : str
        Absolute ``gs://`` path to the video (populated for
        ``modality=video``).
    fps, frame_count, duration_seconds, frame_height, frame_width : float
        Video metadata (populated for ``modality=video``).
    reliable_audio : bool
        Whether the video's audio track is considered reliable by the
        SSW60 curators (populated for ``modality=video``).

    Splits
    ------
    - ``all`` : every asset across all modalities.
    - ``audio_all`` / ``audio_train`` / ``audio_test``
    - ``video_all`` / ``video_train`` / ``video_test``
    - ``image_all`` / ``image_train`` / ``image_test`` / ``image_val``

    Loader behaviour
    ----------------
    - ``audio`` → reads the WAV (16 kHz / 32 kHz mirror if available,
      otherwise resamples 22.05 kHz on the fly), mono, into ``audio`` +
      ``sample_rate``.
    - ``image`` → reads the image into ``image`` (HWC uint8), optionally
      resized to ``image_size``.
    - ``video`` → decodes frames into ``video_frames`` (T,H,W,C uint8)
      *and* the aligned audio track into ``audio`` + ``sample_rate``,
      honouring ``max_frames`` / ``target_fps``.

    References
    ----------
    - Van Horn et al. (2022) "Exploring Fine-Grained Audiovisual
      Categorization with the SSW60 Dataset", ECCV.
      arXiv:2207.10664; repo ``visipedia/ssw60``.

    License: research-only (the SSW60 terms forbid redistribution / non-
    research use); staged on internal GCS for ESP research only.
    """

    info = DatasetInfo(
        name="ssw60",
        owner="david",
        split_paths={
            "all": f"{_GCS_ROOT}/ssw60_all.csv",
            "audio_all": f"{_GCS_ROOT}/ssw60_audio_all.csv",
            "audio_train": f"{_GCS_ROOT}/ssw60_audio_train.csv",
            "audio_test": f"{_GCS_ROOT}/ssw60_audio_test.csv",
            "video_all": f"{_GCS_ROOT}/ssw60_video_all.csv",
            "video_train": f"{_GCS_ROOT}/ssw60_video_train.csv",
            "video_test": f"{_GCS_ROOT}/ssw60_video_test.csv",
            "image_all": f"{_GCS_ROOT}/ssw60_images_all.csv",
            "image_train": f"{_GCS_ROOT}/ssw60_images_train.csv",
            "image_test": f"{_GCS_ROOT}/ssw60_images_test.csv",
            "image_val": f"{_GCS_ROOT}/ssw60_images_val.csv",
        },
        version="0.1.0",
        description=(
            "SSW60 (Sapsucker Woods 60) audiovisual bird dataset: 60 "
            "species across 3,861 Macaulay audio clips, 5,400 expert "
            "videos (audio+visual aligned), and 31,821 iNat2021 + NABirds "
            "images, all GBIF-linked. esp-data's first multimodal dataset."
        ),
        sources=("Van Horn et al. (2022) ECCV, arXiv:2207.10664; repo visipedia/ssw60"),
        license="research-only (no redistribution; internal ESP research use)",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _originals_path_column = "audio_path"
    _mixup_group = "bird"

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
        max_frames: int | None = 16,
        target_fps: float | None = None,
        image_size: int | None = None,
        with_video_audio: bool = True,
    ) -> None:
        """Initialise the SSW60 dataset.

        Parameters
        ----------
        split : str
            Split to load (key in :attr:`info.split_paths`).
        output_take_and_give : dict[str, str] | None
            Optional column rename / selection mapping.
        sample_rate : int | None
            Target audio sample rate. Pre-resampled 16 kHz / 32 kHz files
            used when ``{sr}khz_path`` is populated, otherwise resampled on
            the fly. ``None`` returns the file's native rate. Applies to
            ``audio`` rows and the aligned audio of ``video`` rows.
        data_root : str | AnyPathT | None
            Unused — path columns hold absolute ``gs://`` URIs. Accepted
            for API parity with other datasets.
        backend : BackendType
            ``"polars"`` or ``"pandas"``.
        streaming : bool
            Whether to use streaming mode.
        max_frames : int | None
            Maximum number of frames to decode for ``video`` rows. ``None``
            decodes every frame. Defaults to 16.
        target_fps : float | None
            If set, ``video`` frames are subsampled to approximately this
            frame rate. Defaults to None (keep native fps).
        image_size : int | None
            If set, ``image`` rows (and not video frames) are resized to
            ``(image_size, image_size)``. Defaults to None (native size).
        with_video_audio : bool
            Whether to decode the aligned audio track of ``video`` rows.
            Defaults to True.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.image_size = image_size
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

    def _resolve_audio_path(self, row: dict[str, Any]) -> tuple[AnyPathT, bool]:
        """Return ``(absolute_path, is_presampled)`` for an audio row.

        Returns
        -------
        tuple[AnyPathT, bool]
            The audio path and whether it is already at ``self.sample_rate``.
        """
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            col = self._sample_rate_paths[self.sample_rate]
            val = row.get(col)
            if val is not None:
                s = str(val).strip()
                if s and s.lower() != "nan":
                    return anypath(s), True
        return anypath(str(row[self._originals_path_column])), False

    def _load_audio(self, row: dict[str, Any]) -> dict[str, Any]:
        """Populate ``audio`` / ``sample_rate`` for an ``audio`` row.

        Returns
        -------
        dict[str, Any]
            The row with audio fields populated.
        """
        audio_fp, is_presampled = self._resolve_audio_path(row)
        audio, sr = read_audio(audio_fp)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
        if not is_presampled and self.sample_rate is not None and sr != self.sample_rate:
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

    def _load_image(self, row: dict[str, Any]) -> dict[str, Any]:
        """Populate ``image`` (HWC uint8) for an ``image`` row.

        Returns
        -------
        dict[str, Any]
            The row with the ``image`` field populated.
        """
        image = read_image(str(row["image_path"]))
        if self.image_size is not None:
            from PIL import Image

            pil = Image.fromarray(image).resize((self.image_size, self.image_size), Image.BILINEAR)
            image = np.asarray(pil)
        row["image"] = image
        return row

    def _load_video(self, row: dict[str, Any]) -> dict[str, Any]:
        """Populate ``video_frames`` plus the aligned ``audio`` for a video row.

        Returns
        -------
        dict[str, Any]
            The row with ``video_frames`` and (when available) ``audio`` /
            ``sample_rate`` fields populated.
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
        """Load + return one asset, dispatching on ``modality``.

        Returns
        -------
        dict[str, Any]
            The row with the modality-appropriate media fields populated.

        Raises
        ------
        ValueError
            If the row's modality is not one of the supported values
            (audio, video, image).
        """
        modality = str(row.get("modality", "")).strip().lower()
        if modality == "audio":
            row = self._load_audio(row)
        elif modality == "image":
            row = self._load_image(row)
        elif modality == "video":
            row = self._load_video(row)
        else:
            raise ValueError(f"Unknown modality {modality!r}; expected one of {MODALITIES}.")

        row["mixup_group"] = self._mixup_group

        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}
        return row

    def __len__(self) -> int:
        """Return the number of assets in the split.

        Returns
        -------
        int
            Number of assets in the current split.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError("Length is not available in streaming mode.")
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single processed asset.

        Returns
        -------
        dict[str, Any]
            The processed row (media + metadata).
        """
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over processed assets.

        Yields
        ------
        dict[str, Any]
            Each processed asset.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple[SSW60, dict[str, Any]]:
        """Create an SSW60 instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Dataset configuration.

        Returns
        -------
        tuple[SSW60, dict[str, Any]]
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

    def get_available_labels(self, annotation_column: str = "modality") -> list[str]:
        """Return all possible values for a column with a predefined set.

        Parameters
        ----------
        annotation_column : str
            Currently only ``"modality"`` has a predefined value set.

        Returns
        -------
        list[str]
            The available values.

        Raises
        ------
        ValueError
            If ``annotation_column`` has no predefined value set.
        """
        if annotation_column == "modality":
            return MODALITIES
        raise ValueError(
            f"No predefined value set for '{annotation_column}'. "
            f"Columns with predefined values: modality"
        )

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
