"""SubsegPseudolabel: real-audio subsegmentation pseudolabels on Xeno-canto SED boxes.

Amplitude-derived (model-corroborated) subsegmentation of the NatureLM+BirdCODE
SED-detected call boxes on real XC songs — the real-audio counterpart to the
synthetic ``subseg_v9`` corpus. Two splits:

* ``boxed`` — one row per SED box: the audio is cropped to ``[box_start_s, box_end_s]``
  and ``units_json`` (crop-relative ``{start_s,end_s}``) is the interior
  subsegmentation. Feed ``subseg_json_target`` / ``subseg_pixel_target`` +
  ``subseg_v9_boxed_clip*`` prompts. Carries ``tier`` (gold/silver).
* ``sed`` — one row per 10 s window for the UNIVERSAL "SED + subseg" pixel task:
  the audio is cropped to ``[window_start_sec, window_end_sec]`` and
  ``sed_subseg_json`` (window-relative events, each species-tagged with interior
  ``units``) is rendered by ``pixel_sed_subseg_target`` +
  ``species_pixel_sed_subseg`` prompt. Carries context columns
  (latitudeDecimal/…/eventDate) for the ``_with_context`` variant.

Audio is read from the recording's absolute ``gcs_path`` and resampled to
``sample_rate`` (32 kHz to match the multimodal stack).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_RAW = "gs://esp-data-ingestion/xeno-canto/v0.1.0/raw"


@register_dataset
class SubsegPseudolabel(Dataset):
    """Real-audio subsegmentation pseudolabels on XC SED boxes (``boxed`` / ``sed``)."""

    info = DatasetInfo(
        name="subseg_pseudolabel",
        owner="david",
        split_paths={
            "boxed": f"{_RAW}/train_subseg_pseudolabel_boxed.csv",
            "sed": f"{_RAW}/train_subseg_pseudolabel_sed.csv",
            # v2: expanded geo pool, subseg specialist verifier (m9_lw160_fsv corroborator)
            "boxed_v2": f"{_RAW}/train_subseg_pseudolabel_boxed_v2.csv",
            "sed_v2": f"{_RAW}/train_subseg_pseudolabel_sed_v2.csv",
            # structure: per-window events with box + species + call/song + lifestage + subseg units
            "structure": f"{_RAW}/train_structure_pseudolabel_structure.csv",
            # structure_xcstrong: same, on XC-strong GOLD 2-D boxes (real low_hz/high_hz per event)
            "structure_xcstrong": f"{_RAW}/train_structure_xcstrong_structure.csv",
            # sedsubseg_xcstrong_freq: freq-boxed SED+subseg (event=gold freq, units=model freq)
            "sedsubseg_xcstrong_freq": f"{_RAW}/train_sedsubseg_xcstrong_freq.csv",
            # sedsubseg_*_freq_gpt: GPT-5.6-sol gate+relabelled freq SED+subseg — event=gold freq,
            # unit time+freq from GPT (independent 3rd labeler; absent events gated out). Same
            # sed_subseg_json schema, drives BOTH the freq (use_freq=true) and time-only tasks.
            "sedsubseg_pteroset_freq_gpt": f"{_RAW}/train_sedsubseg_pteroset_freq_gpt.csv",
            "sedsubseg_dartmouth_freq_gpt": f"{_RAW}/train_sedsubseg_dartmouth_freq_gpt.csv",
            "sedsubseg_indian_fauna_freq_gpt": f"{_RAW}/train_sedsubseg_indian_fauna_freq_gpt.csv",
            "sedsubseg_xcstrong_freq_gpt": f"{_RAW}/train_sedsubseg_xcstrong_freq_gpt.csv",
            "sedsubseg_ecosoundset_freq_gpt": f"{_RAW}/train_sedsubseg_ecosoundset_freq_gpt.csv",
            # clipextract_*: per-event EXTRACTED CLIPS exploded from the freq_gpt windows —
            # real-audio analogue of synthetic subseg_v9 mode C. box_start/box_end crop each
            # event; units_json is clip-relative (time+freq). Feeds
            # subseg_v9(_pixel)_clip_extracted.
            "clipextract_xcstrong": f"{_RAW}/train_clipextract_xcstrong.csv",
            "clipextract_indian_fauna": f"{_RAW}/train_clipextract_indian_fauna.csv",
            "clipextract_pteroset": f"{_RAW}/train_clipextract_pteroset.csv",
            "clipextract_ecosoundset": f"{_RAW}/train_clipextract_ecosoundset.csv",
            "clipextract_dartmouth": f"{_RAW}/train_clipextract_dartmouth.csv",
            # clipextract_gptfc_*: GPT-only extracted clips carrying the gold event freq band
            # (event_low_hz/event_high_hz) for the FREQUENCY-CONDITIONED clip subseg prompt.
            "clipextract_gptfc_xcstrong": f"{_RAW}/train_clipextract_gptfc_xcstrong.csv",
            "clipextract_gptfc_indian_fauna": f"{_RAW}/train_clipextract_gptfc_indian_fauna.csv",
            "clipextract_gptfc_pteroset": f"{_RAW}/train_clipextract_gptfc_pteroset.csv",
            "clipextract_gptfc_ecosoundset": f"{_RAW}/train_clipextract_gptfc_ecosoundset.csv",
            "clipextract_gptfc_dartmouth": f"{_RAW}/train_clipextract_gptfc_dartmouth.csv",
            # callsong_*: event-level call-vs-song from BirdCODE detections + XC behavior
            # (single-label call/song recs, species-matched events). windowed variant carries
            # window_start_sec/window_end_sec (N-s window centred on the event) + window-relative
            # event_start_s/event_end_s; extracted variant carries box_start_s/box_end_s.
            "callsong_nlm_sample6s": f"{_RAW}/train_callsong_nlm_sample6s.csv",
            "callsong_events_full6s": f"{_RAW}/train_callsong_events_full6s.csv",
        },
        version="0.1.0",
        description=(
            "Amplitude+model subsegmentation pseudolabels on NatureLM SED boxes over "
            "real Xeno-canto: boxed (per-box interior units) + sed (window-level "
            "universal SED+subseg pixel target). Gold/silver tiers."
        ),
        sources="gs://.../xc_geo_pseudolabels.jsonl + subseg_pseudolabels_shard*.jsonl",
        license="private",
    )

    _originals_path_column = "gcs_path"

    def __init__(
        self,
        split: str = "boxed",
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
            One of ``boxed`` / ``sed``.
        output_take_and_give : dict[str, str] | None
            Optional column rename / selection mapping.
        sample_rate : int | None
            Target sample rate; audio is resampled on load. Defaults to 32 kHz.
        data_root : str | AnyPathT | None
            Root prepended to a relative audio path. ``None`` uses ``gcs_path``
            (already absolute) verbatim.
        backend : BackendType
            ``"polars"`` or ``"pandas"``.
        streaming : bool
            Whether to use streaming mode.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self.data_root = anypath(data_root) if data_root is not None else None
        self._data = None
        self._load()

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

        Returns
        -------
        int
            Row count of the loaded split.

        Raises
        ------
        RuntimeError
            If no split has been loaded.
        NotImplementedError
            In streaming mode.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet.")
        if self._streaming:
            raise NotImplementedError("Length is not available in streaming mode.")
        return len(self._data)

    def _crop_window(self, row: dict[str, Any]) -> tuple[float | None, float | None]:
        """Resolve the crop window for this row.

        Returns
        -------
        tuple[float | None, float | None]
            ``(start_s, end_s)``; ``(None, None)`` means use the whole file.
        """
        for lo, hi in (("box_start_s", "box_end_s"), ("window_start_sec", "window_end_sec")):
            a, b = row.get(lo), row.get(hi)
            if a not in (None, "") and b not in (None, ""):
                return float(a), float(b)
        return None, None

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Read the cropped audio and pass label columns through.

        Returns
        -------
        dict[str, Any]
            The row with ``audio`` / ``sample_rate`` added.
        """
        path = row["gcs_path"]
        audio_path = (self.data_root / path) if self.data_root is not None else anypath(path)
        start, end = self._crop_window(row)
        if start is not None:
            audio, sr = read_audio(audio_path, start_time=start, end_time=end)
        else:
            audio, sr = read_audio(audio_path)

        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
        if self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio, orig_sr=sr, target_sr=self.sample_rate, res_type="kaiser_best"
            )
            sr = self.sample_rate

        row["audio"] = audio
        row["sample_rate"] = sr
        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}
        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single processed row.

        Returns
        -------
        dict[str, Any]
            The processed row at ``idx``.
        """
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over processed rows.

        Yields
        ------
        dict[str, Any]
            Each processed row, in split order.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["SubsegPseudolabel", dict[str, Any]]:
        """Create an instance from a configuration.

        Returns
        -------
        tuple[SubsegPseudolabel, dict[str, Any]]
            The dataset and the remaining (unconsumed) config entries.
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
            f"{self.info.name} (v{self.info.version})\n"
            f"Rows: {n} (split={self.split})\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
