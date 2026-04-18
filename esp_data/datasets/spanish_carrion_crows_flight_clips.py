"""Spanish carrion crow flight clips dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_DATASET_DIRNAME = "spanish_carrion_crows_flight_test_clips_5s_100"
_MANIFEST_NAME = "manifest.csv"


def _candidate_dataset_roots() -> list[Path]:
    """Return candidate local roots that may contain the extracted clip bundle.

    Returns
    -------
    list[Path]
        Candidate dataset root directories, ordered from most to least likely.
    """
    roots: list[Path] = []
    seen: set[Path] = set()
    search_bases = [Path.cwd().resolve(), Path(__file__).resolve().parent]

    for base in search_bases:
        for parent in [base, *base.parents]:
            candidate = parent / "data" / _DATASET_DIRNAME
            if candidate not in seen:
                seen.add(candidate)
                roots.append(candidate)

    return roots


def _discover_dataset_root() -> Path:
    """Locate the extracted local clip dataset root.

    Returns
    -------
    Path
        Dataset root containing `manifest.csv`.

    Raises
    ------
    FileNotFoundError
        If no matching local dataset root can be found.
    """
    candidates = _candidate_dataset_roots()
    for candidate in candidates:
        if (candidate / _MANIFEST_NAME).exists():
            return candidate

    searched = "\n".join(str(path / _MANIFEST_NAME) for path in candidates)
    raise FileNotFoundError(
        "Could not locate the Spanish carrion crow flight clip manifest. "
        "Set `data_root` explicitly or place the extracted dataset under one of:\n"
        f"{searched}"
    )


@register_dataset
class SpanishCarrionCrowsFlightClips(Dataset):
    """Balanced 5-second crow flight state clips.

    Description
    -----------
    Pre-cut 5-second audio clips derived from the held-out test split of
    `gs://spanish-carrion-crows/flying_with_annotations/`.

    The default manifest contains 100 clips total:
    - 50 labeled ``flying``
    - 50 labeled ``not_flying``

    Each manifest row points to a pre-cut WAV clip and stores the binary target
    label plus provenance columns such as the original source file ID and the
    original GCS paths used to generate the clip.
    """

    info = DatasetInfo(
        name="spanish_carrion_crows_flight_clips",
        owner="david",
        split_paths={
            "test": _MANIFEST_NAME,
            "all": _MANIFEST_NAME,
        },
        version="0.1.0",
        description=(
            "Balanced 5-second pre-cut Spanish carrion crow clips labeled "
            "flying or not_flying from the held-out biologger flight test set."
        ),
        sources=[
            "gs://spanish-carrion-crows/flying_with_annotations/",
            "data/spanish_carrion_crows_flight_test_clips_5s_100/manifest.csv",
        ],
        license=(
            "For ESP internal, non-commercial use only. Follow the Spanish "
            "carrion crows dataset usage restrictions."
        ),
    )

    def __init__(
        self,
        split: str = "test",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        split : str
            Split to load. Supported values are ``"test"`` and ``"all"``.
        output_take_and_give : dict[str, str] | None
            Optional mapping of original to output keys.
        sample_rate : int | None
            If set, clips are resampled to this rate.
        data_root : str | AnyPathT | None
            Optional root directory containing the extracted clip dataset.
            Defaults to the repository-local generated clip directory.
        backend : BackendType
            Backend to use for tabular loading.
        streaming : bool
            Whether to use streaming mode.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self._data = None
        root = anypath(data_root) if data_root is not None else anypath(_discover_dataset_root())
        self.data_root = root
        self._load()

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        manifest_path = self.data_root / self.info.split_paths[self.split]
        self._data = self._backend_class.from_csv(
            manifest_path,
            streaming=self._streaming,
            keep_default_na=False,
            na_values=[""],
        )

    def __len__(self) -> int:
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. Iterate over the dataset instead."
            )
        return len(self._data)

    def _resolve_audio_path(self, row: dict[str, Any]) -> AnyPathT:
        """Resolve the pre-cut audio path for a row.

        Parameters
        ----------
        row : dict[str, Any]
            Dataset row containing either ``audio_relpath`` or ``audio_path``.

        Returns
        -------
        AnyPathT
            Resolved clip path.

        Raises
        ------
        ValueError
            If neither path field is available.
        """
        audio_relpath = row.get("audio_relpath")
        if audio_relpath:
            return self.data_root / str(audio_relpath)

        audio_path = row.get("audio_path")
        if audio_path:
            return anypath(str(audio_path))

        raise ValueError("Expected 'audio_relpath' or 'audio_path' in dataset row.")

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process one manifest row into an in-memory audio example.

        Parameters
        ----------
        row : dict[str, Any]
            One dataset row from the manifest.

        Returns
        -------
        dict[str, Any]
            Row augmented with ``audio``, ``sample_rate``, and normalized
            ``audio_path``.
        """
        audio_path = self._resolve_audio_path(row)
        audio, sample_rate = read_audio(audio_path)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        if self.sample_rate is not None and sample_rate != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sample_rate,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
            sample_rate = self.sample_rate

        row["audio"] = audio
        row["audio_path"] = str(audio_path)
        row["sample_rate"] = sample_rate

        if self.output_take_and_give:
            return {new_key: row[old_key] for old_key, new_key in self.output_take_and_give.items()}

        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get one dataset item by index.

        Parameters
        ----------
        idx : int
            Dataset index.

        Returns
        -------
        dict[str, Any]
            The processed dataset row.
        """
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over dataset items.

        Yields
        ------
        dict[str, Any]
            Processed dataset rows.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["SpanishCarrionCrowsFlightClips", dict[str, Any]]:
        """Create the dataset from a generic dataset config.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Parsed dataset configuration.

        Returns
        -------
        tuple[SpanishCarrionCrowsFlightClips, dict[str, Any]]
            Dataset instance and optional transformation metadata.
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
        """Return a human-readable dataset summary.

        Returns
        -------
        str
            Summary string for the dataset instance.
        """
        base = f"{self.info.name} (v{self.info.version}), split: {self.split}"
        return (
            f"{base}\n"
            f"Description: {self.info.description}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
