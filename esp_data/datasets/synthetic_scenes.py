"""Synthetic Scenes dataset for sound event detection training."""

from __future__ import annotations

import logging
from typing import Any, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

logger = logging.getLogger(__name__)

_GCS_BASE = "gs://esp-ml-datasets/synthetic_detection/synthetic_scenes_{version}"


def _max_end_time(selection_table_blob: str) -> float:
    """Extract the maximum end time from an inline selection table TSV.

    Parses tab-separated lines to find the largest ``End Time (s)`` value
    without constructing a full DataFrame.

    Parameters
    ----------
    selection_table_blob : str
        Tab-separated selection table string with header row.

    Returns
    -------
    float
        Maximum end time in seconds, or 0.0 if the table is empty.
    """
    if not selection_table_blob or not selection_table_blob.strip():
        return 0.0
    max_end = 0.0
    for line in selection_table_blob.strip().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                max_end = max(max_end, float(parts[2]))
            except (ValueError, IndexError):
                pass
    return max_end


def _strip_source_file_column(blob: str) -> str:
    """Remove the ``Source File`` column from an inline selection table TSV.

    Parameters
    ----------
    blob : str
        Tab-separated selection table string.

    Returns
    -------
    str
        The TSV with the ``Source File`` column removed.
    """
    if not blob or not blob.strip():
        return blob
    lines = blob.strip().split("\n")
    header_parts = lines[0].split("\t")
    try:
        idx = header_parts.index("Source File")
    except ValueError:
        return blob
    out = []
    for line in lines:
        parts = line.split("\t")
        out.append("\t".join(parts[:idx] + parts[idx + 1 :]))
    return "\n".join(out)


@register_dataset
class SyntheticScenes(Dataset):
    """Synthetic Scenes dataset for sound event detection.

    Description
    -----------
    Synthetically generated soundscapes with precise temporal annotations for
    multiple overlapping species.  Each scene is a short audio clip (up to 10 s)
    with a Raven-style selection table listing every vocalization event and its
    species label.

    The manifest CSV has two columns: ``audio_path`` (relative within the
    version directory) and ``selection_table`` (inline TSV with columns
    ``Selection``, ``Begin Time (s)``, ``End Time (s)``, ``Species``,
    ``Source File``).

    At load time the dataset resolves ``audio_path`` to a full GCS path
    (stored as ``audio_fp``), derives ``audio_duration`` from the maximum
    annotation end time, and strips the ``Source File`` column.

    Available Splits
    ----------------
    - ``all``: Full dataset (~100 k scenes).

    Examples
    --------
    >>> from esp_data.datasets import SyntheticScenes
    >>> ds = SyntheticScenes(split="all", version="0.0.3", sample_rate=32000)
    >>> row = ds[0]
    >>> sorted(k for k in row if k != "audio")
    ['audio_duration', 'audio_fp', 'sample_rate', 'selection_table']
    """

    info = DatasetInfo(
        name="synthetic_scenes",
        owner="ESP Data Team",
        split_paths={
            "all": "{base}/{version}_all.csv",
        },
        version="0.0.3",
        description="Synthetically generated soundscapes with multi-species temporal annotations.",
        sources=["AnimalSpeak pseudovox"],
        license="CC BY 4.0",
    )

    def __init__(
        self,
        split: str = "all",
        version: str = "0.0.3",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        split : str
            Split to load (default ``"all"``).
        version : str
            Dataset version, e.g. ``"0.0.3"`` or ``"0.0.4"``.
        output_take_and_give : dict[str, str] | None
            Optional column rename mapping applied on output.
        sample_rate : int | None
            Target sample rate.  Audio is resampled on the fly if needed.
        data_root : str | AnyPathT | None
            Override for the base GCS directory.
        backend : BackendType
            Backend implementation (``"polars"`` or ``"pandas"``).
        streaming : bool
            Whether to stream rows lazily.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.version = version
        self.sample_rate = sample_rate
        self._data = None

        if data_root is not None:
            self.data_root = anypath(data_root)
        else:
            self.data_root = anypath(_GCS_BASE.format(version=version))

        self._load()

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return ["all"]

    def _load(self) -> None:
        csv_path = f"{self.data_root}/{self.version}_all.csv"
        logger.info("SyntheticScenes: loading manifest from %s", csv_path)

        self._data = self._backend_class.from_csv(
            csv_path, streaming=self._streaming, keep_default_na=False, na_values=[""]
        )

        if not self._streaming:
            n_rows = len(self._data)
            logger.info(
                "SyntheticScenes: loaded %d scenes, deriving audio_fp and audio_duration",
                n_rows,
            )

            audio_fps: list[str] = []
            durations: list[float] = []
            cleaned_tables: list[str] = []

            for row in self._data:
                rel_path = row["audio_path"]
                audio_fps.append(f"{self.data_root}/{rel_path}")

                st_blob = row.get("selection_table", "")
                durations.append(_max_end_time(st_blob))
                cleaned_tables.append(_strip_source_file_column(st_blob))

            self._data = self._data.add_column("audio_fp", audio_fps)
            self._data = self._data.add_column("audio_duration", durations)

            # Replace selection_table with cleaned version (no Source File column)
            remaining = [c for c in self._data.columns if c != "selection_table"]
            self._data = self._data.select_columns(remaining)
            self._data = self._data.add_column("selection_table", cleaned_tables)

            # Drop the original audio_path (audio_fp is the resolved version)
            remaining = [c for c in self._data.columns if c != "audio_path"]
            self._data = self._data.select_columns(remaining)

            logger.info("SyntheticScenes: ready (%d scenes)", n_rows)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        audio_path = row["audio_fp"]
        audio, sr = read_audio(anypath(audio_path))
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        if self.sample_rate is not None and sr != self.sample_rate:
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
            item: dict[str, Any] = {}
            for old_key, new_key in self.output_take_and_give.items():
                item[new_key] = row[old_key]
            return item

        return row

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self._data):
            raise IndexError(
                f"Index {idx} out of bounds for dataset length {len(self._data)}"
            )
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["SyntheticScenes", dict[str, Any]]:
        """Create a SyntheticScenes instance from a configuration.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration object.  Accepts an extra ``version`` field
            (default ``"0.0.3"``).

        Returns
        -------
        tuple[SyntheticScenes, dict[str, Any]]
            The dataset and any transform metadata.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg.get("split", "all"),
            version=cfg.get("version", "0.0.3"),
            output_take_and_give=cfg.get("output_take_and_give"),
            data_root=cfg.get("data_root"),
            sample_rate=cfg.get("sample_rate"),
            backend=cfg.get("backend", "polars"),
            streaming=cfg.get("streaming", False),
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def __str__(self) -> str:
        base = f"{self.info.name} v{self.version}"
        n = len(self._data) if self._data is not None else 0
        return (
            f"{base} ({n} scenes)\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}"
        )
