"""Animal Kingdom — animal action-recognition video benchmark (CVPR 2022).

Animal Kingdom (Ng et al., 2022, "Animal Kingdom: A Large and Diverse Dataset
for Animal Behavior Understanding", CVPR) provides, among other tasks, an
action-recognition (AR) benchmark of video clips labelled with a
**multi-label** set drawn from 140 actions spanning 850 species. It is the
flagship animal action-recognition benchmark reviewed by Fazzari et al.
(ESWA 2025) and used by MSQNet / Mamba-MSQNet.

esp-data exposes the AR subset as a **video-only, multi-label** dataset: each
row is one clip carrying the set of actions present (as a ``", "``-joined
string). Frames are decoded lazily via `esp_data.io.read_video`.

The build script (``scripts/data_preprocessing_scripts/animal_kingdom/
build_animal_kingdom.py``) trims the annotated action segments to clips, builds
per-split manifest CSVs, and uploads the clips to
``gs://esp-data-ingestion/animal_kingdom/v0.1.0/``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, read_video

_GCS_ROOT = "gs://esp-data-ingestion/animal_kingdom/v0.1.0"


@register_dataset
class AnimalKingdom(Dataset):
    """Animal Kingdom — animal action-recognition video benchmark (multi-label).

    Description
    -----------
    Video clips from the Animal Kingdom action-recognition (AR) benchmark,
    each labelled with the **set** of actions present (multi-label over 140
    actions, 850 species). Video-only; frames decoded lazily.

    Columns
    -------
    asset_id : str
        Source clip identifier.
    modality : str
        Always ``"video"`` (parity with other esp-data vision datasets).
    labels : str
        The clip's action set as a ``", "``-joined string of action names
        (multi-label; may contain one or more actions).
    split : str
        Source split label (``train`` / ``test``).
    video_path : str
        Absolute ``gs://`` path to the mp4 clip.

    Splits
    ------
    - ``all`` : every clip.
    - ``train`` / ``test`` : the official Animal Kingdom AR splits.

    Loader behaviour
    ----------------
    - ``video`` → decodes frames into ``video_frames`` (T,H,W,C uint8),
      honouring ``max_frames`` / ``target_fps`` (audio is not decoded).

    References
    ----------
    - Ng et al. (2022) "Animal Kingdom: A Large and Diverse Dataset for Animal
      Behavior Understanding", CVPR. arXiv:2204.08129.

    License: research-only (Animal Kingdom terms require a usage agreement);
    staged on internal GCS for ESP research.
    """

    info = DatasetInfo(
        name="animal_kingdom",
        owner="david",
        split_paths={
            "all": f"{_GCS_ROOT}/animal_kingdom_all.csv",
            "train": f"{_GCS_ROOT}/animal_kingdom_train.csv",
            "test": f"{_GCS_ROOT}/animal_kingdom_test.csv",
        },
        version="0.1.0",
        description=(
            "Animal Kingdom action-recognition (AR) benchmark: multi-label "
            "video clips over 140 actions and 850 species. Video-only view."
        ),
        sources=("Ng et al. (2022) CVPR, arXiv:2204.08129; Animal Kingdom"),
        license="research-only (Animal Kingdom usage agreement); internal ESP research staging",
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
        """Initialise the Animal Kingdom (AR) dataset.

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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple[AnimalKingdom, dict[str, Any]]:
        """Create an Animal Kingdom instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Dataset configuration.

        Returns
        -------
        tuple[AnimalKingdom, dict[str, Any]]
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
