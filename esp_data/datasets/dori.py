"""DORI dataset — Southern Resident Killer Whale & marine-mammal acoustics.

Phase-1 ingestion of the DORI-SRKW collection (Nestor et al. 2026,
arXiv:2602.09295): HF-hosted sources (Ocean Networks Canada, Orcasound, Ocean
Observatories Initiative). Each row is one ~15 s clip with clip-level labels:
species (GBIF canonical), orca ecotype, SRKW call-type, and presence (negatives
are unlabeled ONC clips). Audio was cropped to the labelled window and
pre-resampled to 16 kHz and 32 kHz.
"""

from __future__ import annotations

from typing import Any, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://esp-data-ingestion/dori/v0.1.0"


@register_dataset
class DORI(Dataset):
    """DORI — orca / marine-mammal passive-acoustic clips (Phase 1).

    Description
    -----------
    Curated from 30+ years of public hydrophone archives via positive-unlabelled
    active learning (Nestor et al. 2026). Phase 1 covers the HuggingFace-hosted
    sources — Ocean Networks Canada (``onc``), Orcasound (``orcasound``) and
    Ocean Observatories Initiative (``ooi``) — as ~15 s clips. Labels are
    predominantly human-generated (with a minority of model pseudo-labels);
    ONC also contributes unlabelled negatives (``is_negative``).

    Available label columns
    -----------------------
        - ``species``: GBIF canonical name (e.g. ``Orcinus orca``); empty for
          negatives and non-species labels (sea lion / noise / multiple).
        - ``species_common``: cleaned common name from the source.
        - ``ecotype``: orca ecotype (``srkw``/``transient``/``offshore``/``nrkw``).
        - ``call_type``: SRKW call-catalogue type (``s1``, ``s44``, clicks, …).
        - ``presence``: 1 for labelled positives, 0 for unlabelled negatives.

    Splits
    ------
        - ``all``  : every cropped 15 s clip
        - ``train`` / ``test`` : the collection's train/test partition
        - ``onc`` / ``orcasound`` / ``ooi`` : per-source
        - ``onc_benchmark`` : the expert-labelled ONC presence/absence test set
          (385 *full* recordings, ``presence`` = expert ``mammal_present`` plus
          three amateur-annotator columns; not the cropped 15 s clips)

    Pre-resampled Audio
    -------------------
    Originals are the cropped windows (FLAC, source sample rate). Pre-resampled
    16 kHz and 32 kHz WAVs are loaded directly when ``sample_rate`` matches;
    otherwise audio is resampled on-the-fly with librosa ``kaiser_best``.

    References
    ----------
    https://huggingface.co/collections/DORI-SRKW/dori
    arXiv:2602.09295
    """

    info = DatasetInfo(
        name="dori",
        owner="david",
        split_paths={
            "all": f"{_GCS_ROOT}/all.csv",
            "train": f"{_GCS_ROOT}/train.csv",
            "test": f"{_GCS_ROOT}/test.csv",
            "onc": f"{_GCS_ROOT}/onc.csv",
            "orcasound": f"{_GCS_ROOT}/orcasound.csv",
            "ooi": f"{_GCS_ROOT}/ooi.csv",
            "onc_benchmark": f"{_GCS_ROOT}/onc_benchmark.csv",
        },
        version="0.1.0",
        description=(
            "DORI-SRKW orca / marine-mammal passive-acoustic clips (Phase 1: "
            "ONC + Orcasound + OOI). ~15 s clips with clip-level species (GBIF), "
            "orca ecotype, SRKW call-type and presence labels (human-generated + "
            "minority pseudo-labels), plus unlabelled ONC negatives. Originals "
            "cropped to the labelled window; 16 kHz and 32 kHz pre-resampled."
        ),
        sources=["https://huggingface.co/collections/DORI-SRKW/dori", "arXiv:2602.09295"],
        license="multiple (CC-BY-4.0 / CC-BY-NC-SA-4.0; see 'license' column)",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _originals_path_column = "audio_fp"
    _mixup_group = "marine_mammal"

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialise the DORI dataset.

        Parameters
        ----------
        split : str
            Split to load (key in info.split_paths).
        output_take_and_give : dict[str, str] | None
            Optional column rename / selection mapping.
        sample_rate : int | None
            Target sample rate; pre-resampled 16k/32k used when available,
            else resampled on-the-fly. ``None`` returns the original rate.
        data_root : str | AnyPathT | None
            Root prepended to audio paths. Defaults to the GCS dataset root.
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
            Number of clips in the current split.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("No data loaded.")
        if self._streaming:
            raise NotImplementedError("Length unavailable in streaming mode.")
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single clip.

        Returns
        -------
        dict[str, Any]
            The processed row (audio + labels).
        """
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over clips.

        Yields
        ------
        dict[str, Any]
            Each processed clip.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["DORI", dict[str, Any]]:
        """Create a DORI instance from a config.

        Returns
        -------
        tuple[DORI, dict[str, Any]]
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
