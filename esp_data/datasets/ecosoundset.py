"""ECOSoundSet: finely annotated Orthoptera + Cicadidae acoustic dataset.

Event-level insect annotations from Funosas Planas et al. (Zenodo record
18636037, v3). Each 4-second clip is annotated with the time+frequency box,
species, and category of every sound present — target insects (Orthoptera,
Cicadidae) plus background animals (bats, birds, frogs, …) and abiotic noise.

Shaped like :class:`esp_data.datasets.WABAD` / :class:`esp_data.datasets.CEB`:
one row per clip with all events in an inline ``selection_table`` TSV, so the
``window_annotations`` + ``annotation_features`` chain works unchanged. Two
precomputed multilabel-classification columns are also provided:
``target_species_list`` (Orthoptera + Cicadidae only) and ``all_species_list``
(all biotic species incl. background), as ", "-joined GBIF canonical binomials.

Built by ``scripts/data_preprocessing_scripts/ecosoundset/build_ecosoundset.py``
from the raw Zenodo CSVs at ``gs://esp-data-ingestion/ecosoundset/v0.1.0/raw/``.
"""

from __future__ import annotations

from collections.abc import Iterator
from io import StringIO
from typing import Any

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_ROOT = "gs://esp-data-ingestion/ecosoundset/v0.1.0"
_AUDIO_ROOT = f"{_ROOT}/audio"


@register_dataset
class EcoSoundSet(Dataset):
    """ECOSoundSet event-level insect dataset (WABAD-shaped, multilabel-ready).

    Columns
    -------
    audio_fp : str
        Clip filename relative to :attr:`data_root` (unique basename).
    subset : str
        ``train`` / ``val`` / ``test`` (the dataset's own split).
    recording_id : int
        Source recording id.
    n_events : int
        Number of events in ``selection_table``.
    selection_table : str
        TSV with ``Begin Time (s)``, ``End Time (s)`` (clip-relative, 0–4 s),
        ``Low Freq (Hz)``, ``High Freq (Hz)``, ``Species`` (GBIF canonical),
        ``subspecies`` (verbatim label), ``label_category``.
    target_species_list : str
        ", "-joined GBIF binomials of the Orthoptera + Cicadidae present.
    all_species_list : str
        ", "-joined GBIF binomials of all biotic species present (incl. background).
    n_target_species, n_all_species : int
        Cardinalities of the two lists.

    Splits
    ------
    ``train`` (25,351), ``val`` (9,653), ``test`` (6,879).

    Notes
    -----
    License CC-BY-NC-4.0. Many Orthoptera/Cicadidae (and background bats/high
    passerines) sing above 16 kHz; at a 32 kHz load rate (Nyquist 16 kHz) that
    energy is lost, so some species are not identifiable from the audio.

    References
    ----------
    Zenodo 10.5281/zenodo.18636037 (v3).
    """

    info = DatasetInfo(
        name="ecosoundset",
        owner="david",
        split_paths={
            "train": f"{_ROOT}/ecosoundset_train_with_selection_table.csv",
            "val": f"{_ROOT}/ecosoundset_val_with_selection_table.csv",
            "test": f"{_ROOT}/ecosoundset_test_with_selection_table.csv",
        },
        version="0.1.0",
        description=(
            "ECOSoundSet: finely annotated Orthoptera + Cicadidae acoustic dataset "
            "(41,883 annotated 4s clips, 147 target species; background animals + "
            "abiotic noise also annotated). WABAD-shaped with inline selection_table "
            "plus target/all species lists for multilabel classification."
        ),
        sources=(
            "Zenodo 18636037 v3 (Funosas Planas et al.); staged at "
            "gs://esp-data-ingestion/ecosoundset/v0.1.0/raw/"
        ),
        license="CC-BY-NC-4.0",
    )

    _sample_rate_paths: dict[int, str] = {}
    _originals_path_column = "audio_fp"

    def __init__(
        self,
        split: str = "test",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "pandas",
        streaming: bool = False,
    ) -> None:
        """Initialise the ECOSoundSet dataset.

        Parameters
        ----------
        split : str
            Split key in :attr:`info.split_paths`.
        output_take_and_give : dict[str, str] | None
            Optional column rename / selection mapping.
        sample_rate : int | None
            Target sample rate; audio is resampled on load. Defaults to 32 kHz.
        data_root : str | AnyPathT | None
            Root prepended to ``audio_fp``. Defaults to the GCS audio root.
        backend : BackendType
            ``"polars"`` or ``"pandas"``.
        streaming : bool
            Whether to use streaming mode.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self._data = None
        self.annotation_columns = ["Species"]
        self._load()
        self.data_root = anypath(data_root) if data_root is not None else anypath(_AUDIO_ROOT)

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    @property
    def available_sample_rates(self) -> list[int]:
        """No pre-resampled mirrors; loading resamples on the fly."""
        return []

    def _load(self) -> None:
        """Load the split manifest CSV.

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
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError("Length is not available in streaming mode.")
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Read audio (optionally windowed) and parse the selection table.

        Parameters
        ----------
        row : dict[str, Any]
            A single manifest row.

        Returns
        -------
        dict[str, Any]
            The row with ``audio``, ``sample_rate`` and a parsed
            ``selection_table`` DataFrame.
        """
        audio_path = anypath(self.data_root) / str(row[self._originals_path_column])

        window_start = row.get("window_start_sec")
        window_end = row.get("window_end_sec")
        if window_start is not None and window_end is not None:
            audio, sr = read_audio(
                audio_path, start_time=float(window_start), end_time=float(window_end)
            )
        else:
            audio, sr = read_audio(audio_path)

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

        raw_st = row.get("selection_table")
        if raw_st is not None:
            if isinstance(raw_st, str):
                st = pd.read_csv(StringIO(raw_st), sep="\t", keep_default_na=False)
            elif isinstance(raw_st, pd.DataFrame):
                st = raw_st
            else:
                st = pd.DataFrame()
            row["selection_table"] = st

        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}
        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single processed row.

        Returns
        -------
        dict[str, Any]
            The processed row (audio + selection_table + metadata).
        """
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over processed rows.

        Yields
        ------
        dict[str, Any]
            Each processed row.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["EcoSoundSet", dict[str, Any]]:
        """Create an EcoSoundSet instance from a configuration.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Dataset configuration.

        Returns
        -------
        tuple[EcoSoundSet, dict[str, Any]]
            The dataset and any transformation metadata.
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

    def get_available_labels(self, annotation_column: str = "Species") -> list[str]:
        """Return all per-event labels found in the selection tables.

        Parameters
        ----------
        annotation_column : str
            Column inside each row's selection_table TSV. Defaults to ``"Species"``.

        Returns
        -------
        list[str]
            Sorted unique label values across the dataset.

        Raises
        ------
        ValueError
            If ``annotation_column`` is not present in the selection_table.
        """
        labels: set[str] = set()
        for row in self._data:
            raw = row.get("selection_table")
            if not raw:
                continue
            st = pd.read_csv(StringIO(raw), sep="\t", keep_default_na=False)
            if annotation_column not in st.columns:
                raise ValueError(
                    f"Column '{annotation_column}' not in selection_table; "
                    f"available: {list(st.columns)}"
                )
            labels.update(st[annotation_column].astype(str).tolist())
        return sorted(labels)

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        n = len(self) if self._data is not None and not self._streaming else "?"
        return (
            f"{base}\n"
            f"Clips: {n} (split={self.split})\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
