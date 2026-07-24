"""SubsegmentationSyntheticV9: three-mode synthetic subsegmentation corpus.

Loader for the ``subseg_v9_pool_25k`` corpus (a 2,750-clip Opus-labeled
pseudovox pool, 1,091 species). Three splits map to the corpus's three modes:

* ``mode_a`` — full-scene "song mode" (25k scenes). Read from a pre-joined
  manifest (``build_subseg_v9_pool_25k_mode_a.py``) that carries frequency +
  group structure in ``units_json`` / ``groups_json``.
* ``mode_b`` — group-crop "box mode" (39k). Manifest-only: each row references
  a mode-A scene wav and a ``crop_start_s``/``crop_end_s`` window that is
  sliced on load; ``units_json`` is already crop-local with frequency.
* ``mode_c`` — single-clip "box mode" (5.5k, orig + noise-augmented).
  ``units_json`` carries frequency; no cropping.

The row's ``units_json`` (and ``groups_json`` for mode A) columns are passed
through untouched for the ``subseg_json_target`` transform to render into the
JSON training target. Audio is resampled to ``sample_rate`` (default 32 kHz to
match the multimodal stack).
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_CORPUS_ROOT = "gs://foundation-model-data/synthetic/subseg_v9_pool_25k"
# Manifest column that holds the (corpus-root-relative) audio path, in priority
# order: mode A uses audio_path, mode C uses audio_rel, mode B references the
# parent scene via scene_audio_rel.
_AUDIO_COLS = ("audio_path", "audio_rel", "scene_audio_rel")


@register_dataset
class SubsegmentationSyntheticV9(Dataset):
    """Three-mode synthetic subsegmentation corpus (subseg_v9_pool_25k).

    Columns (per split)
    -------------------
    example_id : str
    audio_path / audio_rel / scene_audio_rel : str
        Corpus-root-relative wav path (mode B references a mode-A scene wav).
    crop_start_s, crop_end_s : float
        Present for mode B only; the scene wav is sliced to this window.
    units_json : str
        JSON list of ``{start_s, end_s[, low_hz, high_hz, group_id]}``.
    groups_json : str
        JSON list of ``{id, type, start_s, end_s, low_hz, high_hz}`` (mode A).
    species : str

    Splits
    ------
    ``mode_a`` / ``mode_b`` / ``mode_c``.
    """

    info = DatasetInfo(
        name="subsegmentation_v9",
        owner="david",
        split_paths={
            "mode_a": "gs://esp-data-ingestion/subseg_v9_pool_25k/mode_A_normalized.csv",
            "mode_b": f"{_CORPUS_ROOT}/mode_B/manifest.csv",
            "mode_c": f"{_CORPUS_ROOT}/mode_C/manifest.csv",
        },
        version="0.1.0",
        description=(
            "Synthetic subsegmentation corpus subseg_v9_pool_25k: mode A "
            "full-scene song mode (25k, groups+units+freq), mode B group-crop "
            "box mode (39k, crop-sliced units+freq), mode C single-clip box "
            "mode (5.5k orig+noise-aug, units+freq). 1,091 species."
        ),
        sources="gs://foundation-model-data/synthetic/subseg_v9_pool_25k",
        license="private",
    )

    _originals_path_column = "audio_path"

    def __init__(
        self,
        split: str = "mode_a",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "pandas",
        streaming: bool = False,
    ) -> None:
        """Initialise the dataset.

        Parameters
        ----------
        split : str
            One of ``mode_a`` / ``mode_b`` / ``mode_c``.
        output_take_and_give : dict[str, str] | None
            Optional column rename / selection mapping.
        sample_rate : int | None
            Target sample rate; audio is resampled on load. Defaults to 32 kHz.
        data_root : str | AnyPathT | None
            Root prepended to the relative audio path. Defaults to the corpus
            root (audio lives there even for the mode-A manifest, which is
            mirrored under the ingestion bucket).
        backend : BackendType
            ``"polars"`` or ``"pandas"``.
        streaming : bool
            Whether to use streaming mode.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self._data = None
        self._load()
        self.data_root = anypath(data_root) if data_root is not None else anypath(_CORPUS_ROOT)

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

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
        """Return the number of rows in the split.

        Raises
        ------
        RuntimeError
            If no split has been loaded.
        NotImplementedError
            In streaming mode.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError("Length is not available in streaming mode.")
        return len(self._data)

    def _audio_rel(self, row: dict[str, Any]) -> str:
        """Return the corpus-root-relative audio path for this row.

        Raises
        ------
        KeyError
            If no known audio-path column is present.
        """
        for col in _AUDIO_COLS:
            val = row.get(col)
            if val:
                return str(val)
        raise KeyError(f"No audio-path column in row (looked for {_AUDIO_COLS}).")

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Read audio (slicing mode-B crops) and pass through label columns.

        Returns
        -------
        dict[str, Any]
            The row with ``audio`` / ``sample_rate`` added.
        """
        audio_path = anypath(self.data_root) / self._audio_rel(row)

        crop_start = row.get("crop_start_s")
        crop_end = row.get("crop_end_s")
        if crop_start not in (None, "") and crop_end not in (None, ""):
            audio, sr = read_audio(
                audio_path, start_time=float(crop_start), end_time=float(crop_end)
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

        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}
        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single processed row."""
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over processed rows."""
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["SubsegmentationSyntheticV9", dict[str, Any]]:
        """Create an instance from a configuration."""
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
        base = f"{self.info.name} (v{self.info.version})"
        n = len(self) if self._data is not None and not self._streaming else "?"
        return (
            f"{base}\n"
            f"Rows: {n} (split={self.split})\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
