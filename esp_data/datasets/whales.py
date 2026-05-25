"""Whales synthetic dataset.

Watkins marine-mammal vocalizations mixed onto realistic empty backgrounds
sampled from DCLDE 2026 killer-whale recordings. The dataset is clip-level
(one row per audio file, mirroring `Watkins`) and keeps Watkins taxonomy
and call-type metadata on positive rows; negative rows are pure DCLDE
background and carry ``species="None"``.

Each row is a 2-10 s mono 32 kHz WAV. Build pipeline lives in
``scripts/build_whales_synthetic.py``.
"""

from __future__ import annotations

from typing import Any, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://foundation-model-data/synthetic/whales/v0.1.0"


@register_dataset
class Whales(Dataset):
    """Synthetic whales dataset (Watkins clips on DCLDE backgrounds).

    Description
    -----------
    Marine-mammal vocalizations from the Watkins Marine Mammal Sound
    Database (~50 species across cetaceans and pinnipeds) mixed onto
    annotated-as-empty windows of DCLDE 2026 hydrophone recordings.
    Negatives are pure DCLDE background drawn from the same empty-window
    pool and carry ``species="None"``.

    Available Metadata Fields
    -------------------------
    **Audio paths:**
        - ``32khz_path``: Path to the synthesised 32 kHz mono WAV
          (relative to ``data_root``).

    **Clip metadata:**
        - ``clip_id``: Stable identifier.
        - ``duration_s``: Actual duration of the synthesised clip in
          seconds (2-10 s).
        - ``is_positive``: ``True`` if the clip contains a Watkins event,
          ``False`` for a pure-background negative.

    **Watkins taxonomy (positives only; empty on negatives):**
        - ``species``, ``canonical_name``, ``species_common``
        - ``genus``, ``family``, ``order``, ``class``, ``phylum``, ``kingdom``
        - ``gbifID``

    **Call labels (positives only):**
        - ``call_type``: Fine-grained vocalisation types (semicolon-separated).
        - ``coarse_call_type``: Coarse categories (semicolon-separated).

    **Mix metadata (positives only):**
        - ``event_start_s``, ``event_end_s``: Where the Watkins event sits
          inside the 2-10 s clip.
        - ``snr_db``: SNR (Watkins RMS / DCLDE RMS, in dB) used in the mix.
        - ``watkins_source_path``: Source Watkins 32 kHz file mixed in.

    **DCLDE provenance (positives and negatives):**
        - ``dclde_source_path``: Source DCLDE 32 kHz file used as background.
        - ``dclde_window_start_s``, ``dclde_window_end_s``: Empty window
          inside the DCLDE file that became this clip's background.
        - ``dclde_provider``, ``dclde_audio_id``: DCLDE provider name and
          audio identifier.

    Examples
    --------
    >>> from esp_data.datasets import Whales
    >>> ds = Whales(split="train")
    >>> row = ds[0]
    >>> row["audio"].shape, row["sample_rate"], row["is_positive"]
    ((<n_samples>,), 32000, True)
    """

    info = DatasetInfo(
        name="whales",
        owner="david",
        split_paths={
            "train": f"{_GCS_ROOT}/whales.csv",
        },
        version="0.1.0",
        description=(
            "Watkins marine-mammal clips mixed onto empty DCLDE 2026 windows "
            "to produce a clip-level dataset with realistic detection "
            "backgrounds. ~50 species + balanced 'None' background negatives."
        ),
        sources=[
            "https://cis.whoi.edu/science/B/whalesounds/index.cfm",
            "Palmer et al. (2025) doi:10.1038/s41597-025-05281-5",
        ],
        license="internal",
    )

    _sample_rate_paths: dict[int, str] = {
        32000: "32khz_path",
    }

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
        """Initialise the Whales dataset.

        Parameters
        ----------
        split : str
            The split to load (default ``"train"``).
        output_take_and_give : dict[str, str] | None
            Column renaming / selection mapping.
        sample_rate : int | None
            Target sample rate. ``32000`` loads the pre-resampled 32 kHz
            file directly; other rates trigger on-the-fly resampling.
            ``None`` returns 32 kHz unchanged.
        data_root : str | AnyPathT | None
            Root directory prepended to ``32khz_path``. Defaults to the GCS
            bucket the manifest CSV lives in.
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

        if data_root is None:
            self.data_root = anypath(_GCS_ROOT)
        else:
            self.data_root = anypath(data_root)

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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["Whales", dict[str, Any]]:
        """Create a Whales instance from a config.

        Returns
        -------
        tuple[Whales, dict[str, Any]]
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

        Pre-resampled 32 kHz audio is loaded directly. Other target rates
        are reached via on-the-fly ``librosa.resample``.

        Returns
        -------
        dict[str, Any]
            The row with ``audio``, ``sample_rate``, and ``mixup_group``
            keys added.
        """
        audio_path = self.data_root / row[self._originals_path_column]
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
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in self._data:
            yield self._process(row)

    def __str__(self) -> str:
        n = len(self) if self._data is not None and not self._streaming else "?"
        return (
            f"{self.info.name} (v{self.info.version}), split='{self.split}'\n"
            f"  Rows: {n}\n"
            f"  Description: {self.info.description}\n"
            f"  License: {self.info.license}\n"
            f"  Available splits: {', '.join(self.info.split_paths.keys())}"
        )
