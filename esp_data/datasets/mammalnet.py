"""MammalNet — large-scale mammal behavior video benchmark (CVPR 2023).

MammalNet (Chen et al., 2023, "MammalNet: A Large-scale Video Benchmark for
Mammal Recognition and Behavior Understanding", CVPR) is a video benchmark of
~18,346 clips (≈539 h) covering 12 high-level animal behaviors across
hundreds of mammal species. It is one of the direct-video behavior-recognition
benchmarks reviewed by Fazzari et al. (ESWA 2025).

esp-data exposes it as a **video-only** behavior-classification dataset: each
row is one trimmed clip labelled with a single high-level behavior. Frames are
decoded lazily via `esp_data.io.read_video` (no audio track is used).

The build script (``scripts/data_preprocessing_scripts/mammalnet/
build_mammalnet.py``) maps the MammalNet clips + annotations to per-split
manifest CSVs and uploads the clips to
``gs://esp-data-ingestion/mammalnet/v0.1.0/``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, read_video

_GCS_ROOT = "gs://esp-data-ingestion/mammalnet/v0.1.0"


@register_dataset
class MammalNet(Dataset):
    """MammalNet — mammal behavior video benchmark (video-only).

    Description
    -----------
    Trimmed video clips labelled with one of 12 high-level mammal behaviors
    (e.g. hunting, grooming, resting, feeding). Video-only; frames decoded
    lazily. This is the single-label behavior-classification view of
    MammalNet (its composition/species tasks are not exposed here).

    Columns
    -------
    asset_id : str
        Source clip identifier.
    modality : str
        Always ``"video"`` (parity with other esp-data vision datasets).
    label : int
        0-based behavior class index (0–11).
    behavior : str
        Behavior name (the classification target).
    split : str
        Source split label (``train`` / ``val`` / ``test``).
    video_path : str
        Absolute ``gs://`` path to the mp4 clip.

    Splits
    ------
    - ``all`` : every clip.
    - ``train`` / ``val`` / ``test`` : the official MammalNet splits.

    Loader behaviour
    ----------------
    - ``video`` → decodes frames into ``video_frames`` (T,H,W,C uint8),
      honouring ``max_frames`` / ``target_fps`` (audio is not decoded).

    References
    ----------
    - Chen et al. (2023) "MammalNet: A Large-scale Video Benchmark for Mammal
      Recognition and Behavior Understanding", CVPR.

    License: research-only (see the MammalNet release terms); staged on
    internal GCS for ESP research.
    """

    info = DatasetInfo(
        name="mammalnet",
        owner="david",
        split_paths={
            "all": f"{_GCS_ROOT}/mammalnet_all.csv",
            "train": f"{_GCS_ROOT}/mammalnet_train.csv",
            "val": f"{_GCS_ROOT}/mammalnet_val.csv",
            "test": f"{_GCS_ROOT}/mammalnet_test.csv",
        },
        version="0.1.0",
        description=(
            "MammalNet: large-scale mammal behavior video benchmark — trimmed "
            "clips labelled with 12 high-level behaviors across hundreds of "
            "mammal species. Video-only behavior-classification view."
        ),
        sources=("Chen et al. (2023) CVPR, MammalNet"),
        license="research-only (MammalNet terms); internal ESP research staging",
    )

    _mixup_group = "animal"

    def __init__(
        self,
        split: str = "test",
        output_take_and_give: dict[str, str] | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
        max_frames: int | None = 16,
        target_fps: float | None = None,
    ) -> None:
        """Initialise the MammalNet dataset.

        Parameters
        ----------
        split : str
            Split to load (key in :attr:`info.split_paths`).
        output_take_and_give : dict[str, str] | None
            Optional column rename / selection mapping.
        data_root : str | AnyPathT | None
            Unused — path columns hold absolute ``gs://`` URIs. Accepted
            for API parity with other datasets.
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
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.max_frames = max_frames
        self.target_fps = target_fps
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
        """Populate ``video_frames`` (T,H,W,C uint8) for a video row.

        Returns
        -------
        dict[str, Any]
            The row with ``video_frames`` and ``fps`` populated.
        """
        decoded = read_video(
            str(row["video_path"]),
            max_frames=self.max_frames,
            target_fps=self.target_fps,
            with_audio=False,
        )
        row["video_frames"] = decoded["frames"]
        row["fps"] = decoded["fps"]
        return row

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Load + return one clip.

        Returns
        -------
        dict[str, Any]
            The row with ``video_frames`` populated.
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
            The processed row (frames + metadata).
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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple[MammalNet, dict[str, Any]]:
        """Create a MammalNet instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Dataset configuration.

        Returns
        -------
        tuple[MammalNet, dict[str, Any]]
            The dataset instance and any transformation metadata.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
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
