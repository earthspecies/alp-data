"""BEANS-Zero call-variant binary manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_DATASET_DIRNAME = "beans_zero_call_variants_from_mapping"
_BEANS_ZERO_RAW_ROOT = "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/"


def _candidate_dataset_roots() -> list[Path]:
    """Return candidate local roots that may contain the call-variant manifests.

    Returns
    -------
    list[Path]
        Candidate manifest directories ordered from most to least likely.
    """

    roots: list[Path] = []
    seen: set[Path] = set()
    search_bases = [Path.cwd().resolve(), Path(__file__).resolve().parent]

    for base in search_bases:
        for parent in [base, *base.parents]:
            candidate = parent / "data" / _DATASET_DIRNAME
            if candidate not in seen:
                seen.add(candidate)
                roots.append(candidate)

    return roots


def _discover_dataset_root() -> Path:
    """Locate the generated call-variant manifest directory.

    Returns
    -------
    Path
        Dataset root containing the generated binary JSONL manifests.

    Raises
    ------
    FileNotFoundError
        If no matching local dataset root can be found.
    """

    candidates = _candidate_dataset_roots()
    for candidate in candidates:
        if (candidate / "flight_call_binary.jsonl").exists():
            return candidate

    searched = "\n".join(str(path / "flight_call_binary.jsonl") for path in candidates)
    raise FileNotFoundError(
        "Could not locate the BEANS-Zero call-variant manifests. "
        "Set `data_root` explicitly or place the generated manifests under one of:\n"
        f"{searched}"
    )


@register_dataset
class BeansZeroCallVariants(Dataset):
    """BEANS-Zero call-variant manifests derived from the call-type split.

    Description
    -----------
    Local JSONL manifests derived from Xeno-canto-linked BEANS-Zero call-type
    examples. The binary splits are balanced presence tasks for one vocalization
    subtype, and the ``all`` split exposes the full linked manifest for
    multilabel call-type evaluation.
    """

    info = DatasetInfo(
        name="beans_zero_call_variants",
        owner="david",
        split_paths={
            "flight_call": "flight_call_binary.jsonl",
            "alarm_call": "alarm_call_binary.jsonl",
            "begging_call": "begging_call_binary.jsonl",
            "alarm_call_unseen": "alarm_call_unseen_binary.jsonl",
            "begging_call_unseen": "begging_call_unseen_binary.jsonl",
            "all": "linkage_manifest.jsonl",
        },
        version="0.1.0",
        description=(
            "BEANS-Zero call-variant evaluation manifests for flight call, "
            "alarm call, begging call, unseen alarm/begging call, and full "
            "linked multilabel metadata."
        ),
        sources=[
            "data/beans_zero_call_variants_from_mapping/*.jsonl",
            "~/data-ingestion/scripts/beans_zero_xc_mapping/call-type_xc_mapping.csv",
        ],
        license="Mixed Xeno-canto source licenses; evaluation-only local manifests.",
    )

    _sample_rate_paths = {
        32000: "audio_path_32khz",
        16000: "audio_path_16khz",
    }
    _originals_path_column = "audio_path_original_sample_rate"

    def __init__(
        self,
        split: str = "flight_call",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        split : str
            Split to load. Supported values are ``flight_call``,
            ``alarm_call``, ``begging_call``, ``alarm_call_unseen``,
            ``begging_call_unseen``, and ``all``.
        output_take_and_give : dict[str, str] | None
            Optional mapping of original to output keys.
        sample_rate : int | None
            If set to 16000 or 32000, use pre-resampled BEANS-Zero audio when available.
        data_root : str | AnyPathT | None
            Optional manifest directory. Defaults to the repo-local generated
            manifests under ``data/beans_zero_call_variants_from_mapping``.
        backend : BackendType
            Backend to use for tabular loading.
        streaming : bool
            Whether to use streaming mode.
        """

        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self._data = None
        self.data_root = (
            anypath(data_root) if data_root is not None else anypath(_discover_dataset_root())
        )
        self._beans_zero_audio_root = anypath(_BEANS_ZERO_RAW_ROOT)
        self._load()

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

        manifest_path = self.data_root / self.info.split_paths[self.split]
        self._data = self._backend_class.from_json(
            manifest_path,
            lines=True,
            orient="records",
            streaming=self._streaming,
        )

    def __len__(self) -> int:
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. Iterate over the dataset instead."
            )
        return len(self._data)

    def _resolve_manifest_audio_path(self, path_value: Any) -> AnyPathT | None:
        """Resolve a BEANS-Zero manifest audio path to a concrete path."""

        if path_value is None:
            return None
        path_text = str(path_value).strip()
        if not path_text:
            return None
        candidate = anypath(path_text)
        if isinstance(candidate, Path) and not candidate.is_absolute():
            return self._beans_zero_audio_root / path_text
        return candidate

    def _resolve_audio_path(self, row: dict[str, Any]) -> AnyPathT:
        """Resolve the best available audio path for a row.

        Returns
        -------
        AnyPathT
            Pre-resampled BEANS-Zero path when available, otherwise the
            original BEANS-Zero audio path.

        Raises
        ------
        ValueError
            If neither a pre-resampled XC path nor an original path is present.
        """

        if self.sample_rate in self._sample_rate_paths:
            path_key = self._sample_rate_paths[self.sample_rate]
            resolved_path = self._resolve_manifest_audio_path(row.get(path_key))
            if resolved_path is not None:
                return resolved_path

        original_path = self._resolve_manifest_audio_path(row.get(self._originals_path_column))
        if original_path is not None:
            return original_path

        raise ValueError(
            "Expected a BEANS-Zero audio path in the manifest row."
        )

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process one manifest row into an in-memory audio example.

        Returns
        -------
        dict[str, Any]
            Row augmented with ``audio``, ``sample_rate``, and normalized
            ``audio_path``.
        """

        audio_path = self._resolve_audio_path(row)
        audio, sample_rate = read_audio(audio_path)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        if self.sample_rate is not None and sample_rate != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sample_rate,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
            sample_rate = self.sample_rate

        row["audio"] = audio
        row["audio_path"] = str(audio_path)
        row["sample_rate"] = sample_rate

        if self.output_take_and_give:
            return {new_key: row[old_key] for old_key, new_key in self.output_take_and_give.items()}

        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get one dataset item by index.

        Returns
        -------
        dict[str, Any]
            The processed dataset row.
        """

        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over dataset items.

        Yields
        ------
        dict[str, Any]
            Processed dataset rows.
        """

        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["BeansZeroCallVariants", dict[str, Any]]:
        """Create the dataset from a generic dataset config.

        Returns
        -------
        tuple[BeansZeroCallVariants, dict[str, Any]]
            Dataset instance and optional transformation metadata.
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
        """Return a human-readable dataset summary.

        Returns
        -------
        str
            Summary string for the dataset instance.
        """

        base = f"{self.info.name} (v{self.info.version}), split: {self.split}"
        return (
            f"{base}\n"
            f"Description: {self.info.description}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
