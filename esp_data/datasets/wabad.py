"""WABAD dataset"""

from __future__ import annotations

import logging
from io import StringIO
from typing import Any, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, get_audio_info, read_audio

logger = logging.getLogger("esp_data")

SPECIES_INFO_PATH = "gs://esp-ml-datasets/wabad/v0.1.0/raw/gbif_labels.csv"


@register_dataset
class WABAD(Dataset):
    """WABAD Dataset

    Description
    -----------
    This class makes WABAD dataset available. Each entry is an audio recording,
    plus a selection table. Each row of the selection table has annotations at
    different taxonomic granularities (stored in annotation_columns attribute).
    Taxonomy has been coerced into GBIF.

    This class was included in esp-data (initially) for use as a zero-shot
    detection evaluation dataset.

    Description from publication:
    https://www.researchgate.net/publication/387711208_WABAD_A_World_Annotated_Bird_Acoustic_Dataset_for_Passive_Acoustic_Monitoring

    Under the current global biodiversity crisis, there is a need for automated
    and non-invasive monitoring techniques that can gather large amounts of data
    cost-effectively at various ecological scales, from local to large spatial
    scales. This data can then be analyzed to inform stakeholders and decision
    makers. One such technique is passive acoustic monitoring, which is commonly
    coupled with automatic identification of animal species based on their sound.
    Automated sound analyses usually require the training of sound detection and
    identification algorithms. These algorithms are based on annotated acoustic
    datasets which mark the occurrence of sounds of species inside sound
    recordings. However, compiling large annotated acoustic datasets is time-
    consuming and requires experts, and therefore they normally cover reduced
    spatial, temporal and taxonomic scales. This data paper presents WABAD, the
    World Annotated Bird Acoustic Dataset for passive acoustic monitoring. WABAD
    is designed to provide the public, the research community, and conservation
    managers with a novel and globally representative annotated acoustic dataset.
    This database includes 5,047 minutes of audio files annotated to species-level
    by local experts with the start and end time, and the upper and lower
    frequencies of each identified bird vocalisation in the recordings. The
    database has a wide taxonomic and spatial coverage, including information on
    91,931 vocalisations from 1,192 bird species recorded at 72 recording sites in
    29 recording locations (mainly countries) and distributed across 13 biomes.
    WABAD can be used, for example, for developing and/or validating automatic
    species detection algorithms, answering ecological questions, such as assessing
    geographical variations on bird vocalisations, or comparing acoustic diversity
    indices with species-based diversity indices. The dataset is published under a
    Creative Commons Attribution Non Commercial 4.0 International copyright.

    References
    ----------
    https://zenodo.org/records/15629388
    https://www.researchgate.net/publication/387711208_WABAD_A_World_Annotated_Bird_Acoustic_Dataset_for_Passive_Acoustic_Monitoring

    """

    info = DatasetInfo(
        name="wabad",
        owner="benjamin",
        split_paths={"all": "gs://esp-ml-datasets/wabad/v0.1.0/raw/all_info_gbif.csv"},
        version="0.1.0",
        description="[MISSING]",
        sources="zenodo.org",
        license="CC-BY-4.0",
    )

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "pandas",
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        split : str
            Split to load (key in info.split_paths).
        output_take_and_give : dict[str, str] | None
            Optional mapping of original -> new output keys (filters columns as well).
        sample_rate : int | None
            If set, audio is resampled to this rate.
        data_root : str | AnyPathT | None
            Optional root directory to prepend to each row['audio_fp'].
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars"
        streaming : bool, optional
            Whether to use streaming mode, by default False
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self.annotation_columns = ["Species"]
        self.unknown_label = "Unknown"
        self.sample_rate = sample_rate
        self.data_root = anypath(data_root) if data_root is not None else None

        self.available_labels = pd.read_csv(SPECIES_INFO_PATH)["Species"].to_list()

        # Load split CSV
        self._load()

        # If no explicit data_root, assume parent dir of the split path
        if self.data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent

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
        location = self.info.split_paths[self.split]
        self._data = self._backend_class.from_csv(
            location,
            streaming=self._streaming,
            keep_default_na=False,
            na_values=[""],
        )

    def _prepare_for_transforms(self) -> None:
        """Enrich the backend with columns needed by transforms.

        Resolves ``audio_fp`` to full paths using ``data_root`` and adds an
        ``audio_duration`` column by reading audio file headers via
        `get_audio_info` (no audio decoding).
        """
        if self._data is None:
            raise RuntimeError("No data loaded. Call _load() first.")

        df = self._data.unwrap.copy()

        if "audio_duration" not in df.columns:
            durations = []
            for audio_fp in df["audio_fp"]:
                full_path = self.data_root / audio_fp
                try:
                    info = get_audio_info(full_path)
                    durations.append(info["duration"])
                except Exception:
                    logger.warning(f"Could not read audio info for {full_path}, using 0.0")
                    durations.append(0.0)
            df["audio_duration"] = durations

        resolved_paths = [str(self.data_root / fp) for fp in df["audio_fp"]]
        df["audio_fp"] = resolved_paths

        self._data = self._backend_class(df, streaming=False)

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
                "Length is not available in streaming mode.Iterate over the dataset instead."
            )
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row, loading audio (with optional partial read).

        When ``window_start_sec`` and ``window_end_sec`` are present (set by
        the ``window_annotations`` transform), only the windowed segment is
        loaded via `read_audio(start_time=..., end_time=...)`.

        Parameters
        ----------
        row : dict[str, Any]
            A raw or windowed row from the backend.

        Returns
        -------
        dict[str, Any]
            Row enriched with ``audio`` and ``sample_rate``.
        """
        audio_fp = row["audio_fp"]
        if not str(audio_fp).startswith(("gs://", "s3://", "/")):
            audio_fp = self.data_root / audio_fp

        window_start = row.get("window_start_sec")
        window_end = row.get("window_end_sec")

        if window_start is not None and window_end is not None:
            audio, sample_rate = read_audio(
                audio_fp,
                start_time=float(window_start),
                end_time=float(window_end),
            )
        else:
            audio, sample_rate = read_audio(audio_fp)

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

        # Parse selection table if still a string
        st = row["selection_table"]
        if isinstance(st, str):
            st = pd.read_csv(StringIO(st), sep="\t")

        if window_start is None:
            audio_dur = len(audio) / float(sample_rate)
            st = st[st["Begin Time (s)"] < audio_dur].copy()

        row["audio"] = audio
        row["sample_rate"] = sample_rate
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
            A dictionary containing the audio data, text label, label, and path.
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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["WABAD", dict[str, Any]]:
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
            ds._prepare_for_transforms()
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta

        return ds, {}

    def get_available_labels(self) -> list[str]:
        """Return all possible labels for a given annotation column.

        Returns
        -------
        list[str]
            A list of all the available labels for the annotation column.
        """

        return self.available_labels

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
