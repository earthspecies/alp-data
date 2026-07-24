"""Madeira odontocete dataset — weak clip-level marine-mammal eval benchmark.

Underwater 288 kHz SoundTrap recordings of visually-confirmed, SINGLE-species
odontocete encounters off Madeira (2022-2025; Zenodo 17952229, CC-BY-4.0),
windowed into non-overlapping 10 s clips. Each clip carries clip-level (weak)
labels inherited from its encounter: species (GBIF canonical), common name,
semicolon call-type multilabel, plus the full field metadata (location, date,
group behaviour/size, boat context, hydrophone depth, etc.). Pre-resampled to
16 kHz and 32 kHz.

NOTE: clicks/buzzes are ultrasonic (>16 kHz) and are lost when the model runs at
32 kHz (16 kHz Nyquist); only whistles / low-freq tonals survive. Intended as an
eval benchmark (`eval` split = full set); single-species-per-clip.
"""

from __future__ import annotations

from typing import Any, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

# GCS version root (clips + CSVs rsynced here; flat layout matching dori.py:
# all.csv/eval.csv + audio_16k/ + audio_32k/ at the version root).
_ROOT = "gs://esp-data-ingestion/madeira_odontocetes/v0.1.0"


@register_dataset
class MadeiraOdontocetes(Dataset):
    """Madeira odontocete 10 s clips — weak clip-level species + call-type labels.

    Label columns
    -------------
        - ``canonical_name`` / ``species``: GBIF-canonical binomial.
        - ``species_common``: English common name.
        - ``call_type``: semicolon-joined signal types (Whistles; Echolocation; …).
        - ``presence``: 1 (all clips are positive single-species encounters).
      Plus full encounter metadata (date, latitude, longitude, sea_state,
      hydrophone_depth_m, group_size, calves, group_behaviour, boat_*, notes,
      encounter_id, clip_index, clip_start_sec, family/order/class/...).

    Splits
    ------
        - ``all`` / ``eval`` : every 10 s clip (eval-only benchmark).

    Pre-resampled audio
    -------------------
    16 kHz and 32 kHz WAVs are loaded directly when ``sample_rate`` matches;
    otherwise resampled on-the-fly (``kaiser_best``).
    """

    info = DatasetInfo(
        name="madeira_odontocetes",
        owner="david",
        split_paths={
            "all": f"{_ROOT}/all.csv",
            "eval": f"{_ROOT}/eval.csv",
        },
        version="0.1.0",
        description=(
            "Odontocete acoustic signals from the Madeira archipelago (Zenodo "
            "17952229, CC-BY-4.0): 288 kHz single-species-encounter recordings "
            "windowed into 10 s clips with weak clip-level species (GBIF) + "
            "call-type multilabel + full field metadata. 16 kHz / 32 kHz "
            "pre-resampled. Eval-only benchmark."
        ),
        sources=[
            "https://zenodo.org/records/17952229",
            "https://doi.org/10.1038/s41597-026-07675-5",
        ],
        license="CC-BY-4.0",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _originals_path_column = "audio_fp"
    _mixup_group = "marine_mammal"

    def __init__(
        self,
        split: str = "eval",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialise the Madeira odontocete dataset.

        Parameters
        ----------
        split : str
            Split key in ``info.split_paths`` (``all`` / ``eval``).
        output_take_and_give : dict[str, str] | None
            Optional column rename / selection mapping.
        sample_rate : int | None
            Target sample rate; 16k/32k pre-resampled used when available, else
            resampled on-the-fly. ``None`` returns the original rate.
        data_root : str | AnyPathT | None
            Root prepended to audio paths. Defaults to the dataset root.
        backend : BackendType
            ``"polars"`` or ``"pandas"``.
        streaming : bool
            Whether to use streaming mode.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self._data = None
        self.data_root = anypath(data_root) if data_root else anypath(_ROOT)
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

    @property
    def available_sample_rates(self) -> list[int]:
        """Pre-resampled sample rates whose path columns exist in the data."""
        return [sr for sr, col in self._sample_rate_paths.items() if col in self.columns]

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Load audio for a row, optionally resampling.

        Returns
        -------
        dict[str, Any]
            The row with ``audio``, ``sample_rate`` and ``mixup_group`` added.
        """
        use_presampled = False
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            col = self._sample_rate_paths[self.sample_rate]
            val = row.get(col)
            if val is not None and str(val).strip() and str(val).lower() != "nan":
                audio_path = self.data_root / str(val)
                use_presampled = True
        if not use_presampled:
            audio_path = self.data_root / str(row[self._originals_path_column])

        audio, sr = read_audio(audio_path)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
        if not use_presampled and self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio, orig_sr=sr, target_sr=self.sample_rate, scale=True, res_type="kaiser_best"
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
            Number of clips.

        Raises
        ------
        RuntimeError
            If no split has been loaded.
        """
        if self._data is None:
            raise RuntimeError("No data loaded.")
        if self._streaming:
            raise NotImplementedError("Length unavailable in streaming mode.")
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single processed clip.

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
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["MadeiraOdontocetes", dict[str, Any]]:
        """Create a MadeiraOdontocetes instance from a config.

        Returns
        -------
        tuple[MadeiraOdontocetes, dict[str, Any]]
            The dataset and transformation metadata (empty if none).
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
        n = len(self) if self._data is not None and not self._streaming else "?"
        return (
            f"{self.info.name} (v{self.info.version}), split='{self.split}'\n"
            f"  Clips: {n}\n"
            f"  License: {self.info.license}\n"
            f"  Available splits: {', '.join(self.info.split_paths.keys())}"
        )
