"""InsectSound — wingbeat audio for ten insect species × sex classes.

The MONSTER-Monash repackaging of the UCR InsectSound dataset (Chen et
al. 2014). Ten classes spanning Aedes / Culex mosquito species + sex,
plus *Musca domestica* (housefly) and *Drosophila simulans* (fruit fly).
Each clip is mono, 6 kHz, 0.1 s.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://esp-data-ingestion/monster-monash-insect-sound/v0.1.0"

# (species, sex) ordering matches the canonical UCR InsectSound mapping.
TAXA_SEX: list[tuple[str, str]] = [
    ("Aedes aegypti", "female"),
    ("Aedes aegypti", "male"),
    ("Culex stigmatosoma", "female"),
    ("Culex stigmatosoma", "male"),
    ("Culex tarsalis", "female"),
    ("Culex tarsalis", "male"),
    ("Culex quinquefasciatus", "female"),
    ("Culex quinquefasciatus", "male"),
    ("Musca domestica", "female"),
    ("Drosophila simulans", "female"),
]
SPECIES = sorted({sp for sp, _ in TAXA_SEX})


@register_dataset
class InsectSound(Dataset):
    """InsectSound — UCR ten-class insect wingbeat audio.

    Description
    -----------
    50,000 single-channel 6 kHz wingbeat clips spanning ten classes:
    Aedes aegypti × {♀,♂}, Culex stigmatosoma × {♀,♂}, Culex tarsalis
    × {♀,♂}, Culex quinquefasciatus × {♀,♂}, *Musca domestica* (♀),
    *Drosophila simulans* (♀). Classes are perfectly balanced (5,000
    each). Recorded with an infrared wing-vibration sensor. Each clip
    is exactly 600 samples (~0.1 s).

    Columns
    -------
    clip_id : str
        ``insect_sound_NNNNNN``.
    audio_path / 16khz_path / 32khz_path : str
        Relative paths to the FLAC native + pre-resampled mirrors.
    audio_duration : float
        ~0.1 s for every row.
    class_id : int
        0-9; ordered as :data:`TAXA_SEX`.
    species : str
        GBIF canonical name.
    sex : str
        ``"female"`` or ``"male"``; populated for every row.
    canonical_name, gbifID, kingdom..family, genus, species_common
        GBIF taxonomy.
    fold_{0..4}_test : int
        MONSTER 5-fold cross-validation test indicator.

    Splits
    ------
    - ``all``  : every clip (50,000 rows)
    - ``train``: fold_0 train (~40,000 rows)
    - ``val``  : fold_0 test (~10,000 rows)

    References
    ----------
    Chen, Why, Batista, Mafra-Neto, Keogh (2014), "Flying insect class"
    "ification with inexpensive sensors", *J. Insect Behavior* 27(5),
    657–677. MONSTER paper: arXiv:2502.15122.
    """

    info = DatasetInfo(
        name="insect_sound",
        owner="david",
        split_paths={
            "all": f"{_GCS_ROOT}/insect_sound_all.csv",
            "train": f"{_GCS_ROOT}/insect_sound_train.csv",
            "val": f"{_GCS_ROOT}/insect_sound_val.csv",
        },
        version="0.1.0",
        description=(
            "InsectSound — 50,000 single-channel 0.1 s wingbeat clips, "
            "10 balanced classes spanning four Culicidae species × sex "
            "plus housefly + Drosophila simulans females. UCR / Chen "
            "2014 dataset, repackaged by MONSTER-Monash."
        ),
        sources=(
            "Chen, Why, Batista, Mafra-Neto, Keogh (2014) J. Insect "
            "Behavior 27(5):657-677; MONSTER arXiv:2502.15122"
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
        """Initialise the InsectSound dataset.

        Parameters
        ----------
        split : str
            Split to load (key in :attr:`info.split_paths`).
        output_take_and_give : dict[str, str] | None
            Optional column rename / selection mapping.
        sample_rate : int | None
            Target rate; pre-resampled mirrors used when available.
        data_root : str | AnyPathT | None
            Root prepended to relative audio paths.
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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["InsectSound", dict[str, Any]]:
        """Create an InsectSound instance from a configuration.

        Returns
        -------
        tuple[InsectSound, dict[str, Any]]
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
        """Return labels for ``species``, ``class_id``, or ``sex``.

        Parameters
        ----------
        annotation_column : str
            Which column to enumerate.

        Returns
        -------
        list[str]
            Predefined label set.

        Raises
        ------
        ValueError
            If ``annotation_column`` has no predefined label set.
        """
        if annotation_column == "species":
            return SPECIES
        if annotation_column == "class_id":
            return [str(i) for i in range(len(TAXA_SEX))]
        if annotation_column == "sex":
            return ["female", "male"]
        raise ValueError(
            f"No predefined label set for '{annotation_column}'. "
            f"Columns with predefined labels: species, class_id, sex"
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
