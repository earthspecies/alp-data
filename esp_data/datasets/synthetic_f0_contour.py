"""Synthetic F0 contour-tracing dataset (``contour_v1_20k``).

Purely-synthetic tonal vocalizations with known F0 trajectories, paired with
prompt/answer strings already in the F0-contour-tracing format (5 Hz-rounded
freqs, duration-aware rate in {5,10,20,30}, first point at voiced onset,
max_points 200). One clip per contour, 32 kHz mono WAV.

The precomputed ``answer``/``rate_hz`` are used verbatim (no re-derivation), and
sibling mean/range strings are built from the manifest so both the
``f0_contour`` and ``f0_summary`` templates apply directly — matching the
columns the ``f0_features`` transform emits for ``f0_bioacoustic``.

Source: ``gs://foundation-model-data/synthetic/contour_v1_20k/`` (train-only;
the 100-clip ``contour_v1_test`` is the held-out eval set).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_ROOT = "gs://foundation-model-data/synthetic/contour_v1_20k"


@register_dataset
class SyntheticF0Contour(Dataset):
    """Synthetic F0 contour-tracing corpus (20k clips).

    Emits, per row, the audio plus the template-ready columns:
    ``f0_contour_text`` (the precomputed answer), ``f0_rate_hz`` (sampled rate),
    ``f0_mean`` (e.g. ``"3870 Hz"``) and ``f0_range`` (e.g. ``"2375-7035 Hz"``),
    so the shared ``f0_contour`` and ``f0_summary`` chat templates work without
    the ``f0_features`` transform. Raw manifest columns (``shape``,
    ``voiced_duration_s``, ``noise_source`` …) are preserved for analysis.

    Splits
    ------
    ``train`` -- the 20,000-clip corpus.
    """

    info = DatasetInfo(
        name="synthetic_f0_contour",
        owner="david",
        split_paths={
            "train": f"{_ROOT}/manifest.csv",
        },
        version="1.0.0",
        description=(
            "Synthetic tonal vocalizations with known F0 trajectories and "
            "precomputed contour-tracing prompt/answer strings (20k clips, 32 kHz)."
        ),
        sources=f"{_ROOT} (contour_v1_20k)",
        license="synthetic (ESP)",
    )

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialise the dataset.

        Parameters
        ----------
        split : str
            Split key in :attr:`info.split_paths`.
        output_take_and_give : dict[str, str] | None
            Optional output key rename/selection mapping.
        sample_rate : int | None
            Target sample rate. Audio is native 32 kHz; resampled on the fly
            only if a different rate is requested.
        data_root : str | AnyPathT | None
            Root holding ``audio/`` (defaults to the manifest's parent).
        backend : BackendType
            ``"polars"`` or ``"pandas"``.
        streaming : bool
            Whether to stream the manifest.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self._data = None
        self._load()
        self.data_root = (
            anypath(data_root) if data_root is not None
            else anypath(self.info.split_paths[self.split]).parent
        )

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
        self._data = self._backend_class.from_csv(
            self.info.split_paths[self.split],
            streaming=self._streaming,
            keep_default_na=False,
            na_values=[""],
        )
        if not self._streaming:
            self._add_template_columns()

    def _add_template_columns(self) -> None:
        """Derive the template-ready columns at the BACKEND level.

        The ``drop_null_or_empty_string`` / ``chat`` chain transforms run on the
        backend BEFORE :meth:`_process`, so ``f0_contour_text`` / ``f0_rate_hz``
        / ``f0_mean`` / ``f0_range`` must exist as columns here (not only per-row
        in ``_process``). Mirrors the columns the ``f0_features`` transform emits
        for the ``f0_bioacoustic`` dataset, using the precomputed manifest values
        (``answer``/``rate_hz``/``mean_f0_hz``/``f0_min_hz``/``f0_max_hz``).
        """
        fct: list[str] = []
        frate: list[str] = []
        fmean: list[str] = []
        frange: list[str] = []
        for row in self._data:
            fct.append(str(row.get("answer") or ""))
            frate.append(str(row.get("rate_hz") or "").split(".")[0])
            mean = str(row.get("mean_f0_hz") or "")
            fmean.append(f"{int(float(mean))} Hz" if mean else "")
            mn = str(row.get("f0_min_hz") or "")
            mx = str(row.get("f0_max_hz") or "")
            frange.append(f"{int(float(mn))}-{int(float(mx))} Hz" if mn and mx else "")
        self._data = self._data.add_column("f0_contour_text", fct)
        self._data = self._data.add_column("f0_rate_hz", frate)
        self._data = self._data.add_column("f0_mean", fmean)
        self._data = self._data.add_column("f0_range", frange)

    def __len__(self) -> int:
        if self._data is None:
            raise RuntimeError("No split has been loaded yet.")
        if self._streaming:
            raise NotImplementedError("Length not available in streaming mode.")
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        audio, sr = read_audio(anypath(self.data_root) / str(row["audio_rel"]))
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
        if self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio, orig_sr=sr, target_sr=self.sample_rate, scale=True, res_type="kaiser_best"
            )
            sr = self.sample_rate

        row["audio"] = audio
        row["sample_rate"] = sr
        # Template-ready columns (f0_contour_text / f0_rate_hz / f0_mean /
        # f0_range) are added at the backend level in _add_template_columns so
        # the chat/drop-null chain transforms can see them; nothing to do here.

        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}
        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["SyntheticF0Contour", dict[str, Any]]:
        """Create an instance from a dataset configuration.

        Returns
        -------
        tuple[SyntheticF0Contour, dict[str, Any]]
            The dataset and any transformation metadata.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            sample_rate=cfg["sample_rate"],
            data_root=cfg.get("data_root"),
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
            f"Clips: {n} (split={self.split})\n"
            f"Sources: {self.info.sources}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
