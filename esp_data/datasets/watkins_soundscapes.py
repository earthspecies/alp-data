"""Watkins semi-synthetic marine soundscape dataset.

Balanced, semi-synthetic marine-mammal soundscapes: Watkins MASTER-TAPE windows
(labelled by our SED/NatureLM pipeline) mixed onto realistic marine noise —
DCLDE 2026 mammal-call-free windows (retaining vessel/ocean noise), ship/engine
pools (orcasound/deepship/shipsear), and a small "clean" fraction. Balanced to
~1500 clips/species (per-source reuse-capped) across 55 species folders; ~20% of
clips overlay a second species for multi-label realism.

Leakage: every BEANS-Zero Watkins TEST tape (file_name[:5] == master-tape RN) is
excluded from the foreground, so this training data does not leak the Watkins
evaluation audio.

Each row is a 4-10 s mono 32 kHz WAV. Labels live in the comma-separated
``species_list`` column (scientific names present); ``canonical_name`` is the
primary. Build pipeline:
``scripts/data_preprocessing_scripts/watkins_mastertapes/build_watkins_soundscapes.py``.
"""

from __future__ import annotations

from typing import Any, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://foundation-model-data/synthetic/watkins_soundscapes/v0.3.0"


@register_dataset
class WatkinsSoundscapes(Dataset):
    """Balanced semi-synthetic marine soundscapes (Watkins targets + marine noise).

    Available Metadata Fields
    -------------------------
    **Audio path:** ``32khz_path`` (relative to ``data_root``).

    **Labels:**
        - ``species_list``: comma-separated scientific names present (1-2).
        - ``canonical_name``: primary (foreground) scientific name.
        - ``n_species``: number of species present.
        - ``species_folder``: source Watkins species folder (common name).

    **Mix / provenance:**
        - ``clip_id``, ``duration_s`` (4-10 s), ``tier`` (source label tier:
          strict / active_focal / presence).
        - ``event_start_s``, ``event_end_s``: where the foreground event sits.
        - ``snr_db``: target-over-noise RMS SNR (dB); ``noise_source`` in
          {dclde, ship, clean}.
        - ``fg_uri``, ``fg_start_s``, ``fg_end_s``: source master-tape window.

    Examples
    --------
    >>> from esp_data.datasets import WatkinsSoundscapes
    >>> ds = WatkinsSoundscapes(split="train")
    >>> row = ds[0]
    >>> row["audio"].shape, row["sample_rate"], row["species_list"]
    ((<n_samples>,), 32000, 'Physeter catodon')
    """

    info = DatasetInfo(
        name="watkins_soundscapes",
        owner="david",
        split_paths={
            "train": f"{_GCS_ROOT}/manifest.csv",
        },
        version="0.3.0",
        description=(
            "Balanced semi-synthetic marine soundscapes: Watkins master-tape "
            "target windows mixed onto DCLDE mammal-free / ship / clean noise at "
            "controlled SNR (4-10 s clips), ~1500/species over 55 species, "
            "BEANS-Zero Watkins test tapes excluded. v0.3.0 also drops foreground "
            "windows an avex-all AudioSet classifier flagged as human speech "
            "(speech_prob>=0.5; verified by ear as operator narration)."
        ),
        sources=[
            "https://cis.whoi.edu/science/B/whalesounds/index.cfm",
            "DCLDE 2026 (noise backgrounds)",
        ],
        license="internal",
    )

    _sample_rate_paths: dict[int, str] = {32000: "32khz_path"}
    _mixup_group = "marine_mammal"
    _originals_path_column = "32khz_path"

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialise the WatkinsSoundscapes dataset.

        Parameters
        ----------
        split : str
            Split to load (default ``"train"``).
        output_take_and_give : dict[str, str] | None
            Column renaming / selection mapping.
        sample_rate : int | None
            Target sample rate. ``32000`` loads the pre-resampled file directly.
        data_root : str | AnyPathT | None
            Root prepended to ``32khz_path``. Defaults to the manifest's bucket.
        backend : BackendType
            DataFrame backend (``"polars"`` or ``"pandas"``).
        streaming : bool
            Whether to use streaming mode.

        Raises
        ------
        LookupError
            If ``split`` is not a registered split name.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self._data = None
        self.data_root = anypath(_GCS_ROOT) if data_root is None else anypath(data_root)
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        self._load()

    def _load(self) -> None:
        """Load the manifest CSV from the configured split path."""
        location = self.info.split_paths[self.split]
        self._data = self._backend_class.from_csv(location, streaming=self._streaming)

    @property
    def columns(self) -> list[str]:
        """Return the columns of the loaded manifest."""
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        """Return the available splits."""
        return list(self.info.split_paths.keys())

    @property
    def available_sample_rates(self) -> list[int]:
        """Pre-resampled sample rates available in the loaded data."""
        available = []
        if self._data is not None:
            for sr, col in self._sample_rate_paths.items():
                if col in self._data.columns:
                    available.append(sr)
        return available

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["WatkinsSoundscapes", dict[str, Any]]:
        """Create a WatkinsSoundscapes instance from a config.

        Returns
        -------
        tuple[WatkinsSoundscapes, dict[str, Any]]
            The dataset instance and transformation metadata (empty if none).
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

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Load audio for a row and attach ``audio`` / ``sample_rate``.

        Returns
        -------
        dict[str, Any]
            The row with ``audio``, ``sample_rate`` and ``mixup_group`` added.
        """
        audio_path = self.data_root / row[self._originals_path_column]
        audio, sr = read_audio(audio_path)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
        if self.sample_rate is not None and sr != self.sample_rate:
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
        if self._data is None:
            raise RuntimeError("No data loaded.")
        if self._streaming:
            raise NotImplementedError("Length unavailable in streaming mode.")
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in self._data:
            yield self._process(row)

    def __str__(self) -> str:
        n = len(self) if self._data is not None and not self._streaming else "?"
        return (
            f"{self.info.name} (v{self.info.version}), split='{self.split}'\n"
            f"  Rows: {n}\n  Description: {self.info.description}\n"
            f"  License: {self.info.license}\n"
            f"  Available splits: {', '.join(self.info.split_paths.keys())}"
        )
