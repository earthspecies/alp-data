"""MosquitoSound — wingbeat audio for six mosquito species.

The MONSTER-Monash repackaging of Potamitis et al.'s Wingbeats corpus
(Kaggle / UCR archive). Six Culicidae species recorded with an infrared
sensor detecting wing-vibration; each clip is mono, 6 kHz, 0.625 s.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://esp-data-ingestion/monster-monash-mosquito-sound/v0.1.0"

# Species order matches the Wingbeats Kaggle folder ordering (alphabetical).
SPECIES = [
    "Aedes aegypti",
    "Aedes albopictus",
    "Anopheles arabiensis",
    "Anopheles gambiae",
    "Culex pipiens",
    "Culex quinquefasciatus",
]


@register_dataset
class MosquitoSound(Dataset):
    """MosquitoSound — six-species mosquito wingbeat audio.

    Description
    -----------
    279,566 single-channel 6 kHz wingbeat recordings of six mosquito
    species (*Aedes aegypti*, *Ae. albopictus*, *Anopheles arabiensis*,
    *An. gambiae*, *Culex pipiens*, *C. quinquefasciatus*) captured by
    an infrared sensor detecting wing vibration. Each clip is exactly
    3,750 samples (~0.625 s) at 6 kHz mono. Sourced via MONSTER-Monash
    (Fanioudakis, Geismar & Potamitis 2018; Potamitis 2018 *Wingbeats*).

    Columns
    -------
    clip_id : str
        Stable per-clip identifier (``mosquito_sound_NNNNNN``).
    audio_path : str
        Relative path to the native 6 kHz FLAC.
    16khz_path, 32khz_path : str
        Pre-resampled FLAC mirrors. The native bandwidth is ~3 kHz
        (Nyquist of 6 kHz), so the resampled files just upsample.
    audio_duration : float
        Clip duration in seconds (0.625 for all rows).
    native_sample_rate : int
        Always 6000 Hz.
    class_id : int
        0-5; ordered as :data:`SPECIES`.
    species, canonical_name, scientific_name_unified : str
        GBIF canonical (e.g. ``"Aedes aegypti"``).
    species_common : str
        Common name.
    gbifID, kingdom, phylum, class, order, family, genus : str
        GBIF taxonomy fields (all rows are Insecta / Diptera / Culicidae).
    fold_{0..4}_test : int
        1 if this clip is in the test split of MONSTER fold ``k``.
    license : str
        ``"Public Domain"`` (per Potamitis Kaggle release).

    Splits
    ------
    - ``all``  : every clip (279,566 rows)
    - ``train``: MONSTER fold_0 train (~223,653 rows)
    - ``val``  : MONSTER fold_0 test (~55,913 rows)

    References
    ----------
    Fanioudakis, Geismar & Potamitis (2018), "Mosquito wingbeat analysis
    and classification using deep learning", EUSIPCO; Potamitis (2018)
    *Wingbeats* (Kaggle). MONSTER paper: arXiv:2502.15122.
    """

    info = DatasetInfo(
        name="mosquito_sound",
        owner="david",
        split_paths={
            "all": f"{_GCS_ROOT}/mosquito_sound_all.csv",
            "train": f"{_GCS_ROOT}/mosquito_sound_train.csv",
            "val": f"{_GCS_ROOT}/mosquito_sound_val.csv",
        },
        version="0.1.0",
        description=(
            "MosquitoSound — 279,566 single-channel 0.625 s wingbeat "
            "clips of six Culicidae species (Aedes aegypti / albopictus, "
            "Anopheles arabiensis / gambiae, Culex pipiens / quinque"
            "fasciatus) sampled at 6 kHz via infrared wing-vibration "
            "sensors. Sourced via MONSTER-Monash (Potamitis 2018 Wingbeats)."
        ),
        sources=(
            "Fanioudakis, Geismar, Potamitis (2018) EUSIPCO; "
            "Potamitis (2018) Wingbeats Kaggle; "
            "MONSTER arXiv:2502.15122"
        ),
        license="Public Domain",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _originals_path_column = "audio_path"
    _mixup_group = "insect"

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialise the MosquitoSound dataset.

        Parameters
        ----------
        split : str
            Split to load (key in :attr:`info.split_paths`).
        output_take_and_give : dict[str, str] | None
            Optional column rename / selection mapping.
        sample_rate : int | None
            Target rate. Pre-resampled 16 kHz / 32 kHz files used when
            available; otherwise FLAC resampled on-the-fly from 6 kHz.
        data_root : str | AnyPathT | None
            Root prepended to relative audio paths. Defaults to the GCS
            dataset root.
        backend : BackendType
            ``"polars"`` or ``"pandas"``.
        streaming : bool
            Whether to use streaming mode.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self._data = None
        self.data_root = anypath(data_root) if data_root else anypath(_GCS_ROOT)
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

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Resolve audio path, optionally pre-resampled; load the clip.

        Returns
        -------
        dict[str, Any]
            The row with ``audio``, ``sample_rate`` and ``mixup_group`` set.
        """
        use_presampled = False
        audio_path = None
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            col = self._sample_rate_paths[self.sample_rate]
            val = row.get(col)
            if val is not None:
                s = str(val).strip()
                if s and s.lower() != "nan":
                    audio_path = self.data_root / s
                    use_presampled = True
        if audio_path is None:
            audio_path = self.data_root / str(row[self._originals_path_column])

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
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError("Length not available in streaming mode.")
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single processed clip.

        Returns
        -------
        dict[str, Any]
            The processed row (audio + labels).
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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["MosquitoSound", dict[str, Any]]:
        """Create a MosquitoSound instance from a configuration.

        Returns
        -------
        tuple[MosquitoSound, dict[str, Any]]
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

    def get_available_labels(self, annotation_column: str = "species") -> list[str]:
        """Return the species label set.

        Parameters
        ----------
        annotation_column : str
            One of ``"species"`` or ``"class_id"``.

        Returns
        -------
        list[str]
            For ``species`` returns SPECIES; for ``class_id`` returns the
            integer label set 0..5 stringified.

        Raises
        ------
        ValueError
            If ``annotation_column`` has no predefined label set.
        """
        if annotation_column == "species":
            return SPECIES
        if annotation_column == "class_id":
            return [str(i) for i in range(len(SPECIES))]
        raise ValueError(
            f"No predefined label set for '{annotation_column}'. "
            f"Columns with predefined labels: species, class_id"
        )

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        n = len(self) if self._data is not None and not self._streaming else "?"
        return (
            f"{base}\n"
            f"Clips: {n}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
