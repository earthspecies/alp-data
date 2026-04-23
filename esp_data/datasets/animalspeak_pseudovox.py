"""AnimalSpeak pseudovox dataset.

Silence-trimmed single-vocalization clips stored on GCS. Filenames encode
the species name and source/clip indices, e.g.::

    004-079-Rufous-Spinetail_source1_clip0.wav

The only structured metadata is the clip name itself, which is exposed as
``clip_name`` and used as the sample ID.

Splits:

- ``full`` — JSON manifest of clip stems on GCS (~4.6 M clips).
- ``train_unseen`` — CSV with a ``pseudovox_audio_fp`` column (see
  ``pseudovox_train_unseen.csv``); default object is
  ``gs://foundation-model-data/synthetic/pseudovox_train_unseen.csv``.

For large-scale runs, pass ``n_samples`` or ``percentage`` to subset any split.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import unquote_plus

import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import audio_stereo_to_mono, read_audio

# Pseudovox WAV clips live under gs://fewshot/data_large_clean/animalspeak_pseudovox/.
_DEFAULT_GCS_PATH = "gs://fewshot/data_large_clean/animalspeak_pseudovox"
_DEFAULT_MANIFEST_PATH = "gs://foundation-model-data/synthetic/animalspeak_pseudovox_manifest.json"
# Train-unseen split table (``pseudovox_audio_fp`` column). Upload the CSV from
# ``pseudovox_train_unseen.csv`` to this object before using the split.
_DEFAULT_TRAIN_UNSEEN_CSV_PATH = "gs://foundation-model-data/synthetic/pseudovox_train_unseen.csv"
# Object keys in ``pseudovox_audio_fp`` are relative to this GCS prefix.
_DEFAULT_AUDIO_BUCKET_ROOT = "gs://fewshot/data_large_clean"

_PSEUD_PREFIX = "animalspeak_pseudovox/"


def _decode_csv_uri_path(raw: str) -> str:
    """Normalize ``pseudovox_audio_fp`` values from CSV for real GCS object keys.

    Handles ``%xx`` escapes, ``+`` as space (common CSV exports), backslashes, and
    a few rounds of decoding when paths were double-encoded (e.g. ``%252F``).
    """
    s = str(raw).strip().replace("\\", "/")
    for _ in range(5):
        nxt = unquote_plus(s)
        if nxt == s:
            break
        s = nxt
    return s.lstrip("/")


def _clip_stem_from_pseudovox_audio_fp(pseudovox_audio_fp: str) -> str:
    """Strip ``animalspeak_pseudovox/`` and ``.wav`` to match GCS object basename.

    Applies the same CSV decoding as :func:`_decode_csv_uri_path` so ``%20``, ``+``,
    and double-encoded segments match object keys on GCS.
    """
    p = _decode_csv_uri_path(str(pseudovox_audio_fp).strip().replace("\\", "/"))
    if p.lower().startswith(_PSEUD_PREFIX):
        p = p[len(_PSEUD_PREFIX) :]
    return Path(p).stem


@register_dataset
class AnimalSpeakPseudovox(Dataset):
    """AnimalSpeak pseudovox clips — silence-trimmed single vocalizations.

    Description
    -----------
    Each clip is a short, silence-trimmed single vocalization extracted from
    the AnimalSpeak dataset. Filenames encode the species name and
    source/clip index (e.g. ``004-079-Rufous-Spinetail_source1_clip0``).
    The ``clip_name`` field is the only structured metadata; it serves as
    both the sample ID and a species hint.

    Use ``split="full"`` for the JSON manifest (~4.6 M clips), or
    ``split="train_unseen"`` for the CSV split (column ``pseudovox_audio_fp``).
    Use ``n_samples`` or ``percentage`` to subset either split.

    Each entry contains:

    - an audio clip (variable duration, 16 kHz mono by default)
    - ``clip_name``: filename stem (encodes species and clip index)
    - ``audio_path``: full GCS URI to the WAV file

    References
    ----------
    https://github.com/earthspecies/animalspeak
    """

    info = DatasetInfo(
        name="animalspeak_pseudovox",
        owner="christos",
        split_paths={
            "full": _DEFAULT_MANIFEST_PATH,
            "train_unseen": _DEFAULT_TRAIN_UNSEEN_CSV_PATH,
        },
        version="1.1.3",
        description=(
            "Silence-trimmed single-vocalization clips from AnimalSpeak "
            "(full manifest ~4.6 M; optional train_unseen CSV split)."
        ),
        sources="gs://fewshot/data_large_clean/animalspeak_pseudovox",
        license="CC-BY-NC-4.0",
    )

    def __init__(
        self,
        split: str = "full",
        gcs_path: str = _DEFAULT_GCS_PATH,
        manifest_path: str | None = None,
        split_path: str | None = None,
        n_samples: int | None = None,
        percentage: float = 1.0,
        shuffle: bool = False,
        random_seed: int = 42,
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        split : str
            ``"full"`` loads the JSON manifest (~4.6 M clip stems). ``"train_unseen"``
            loads stems from the CSV split (column ``pseudovox_audio_fp``). See
            ``info.split_paths`` for default URIs.
        gcs_path : str
            URI of the directory containing ``.wav`` files (default: under
            ``gs://animalspeak2``, same root as :class:`~esp_data.datasets.animalspeak.AnimalSpeak`).
        manifest_path : str | None
            When ``split == "full"``, optional override for the JSON manifest path
            (default: ``info.split_paths["full"]``). Ignored for other splits.
        split_path : str | None
            Override the URI/path for the active ``split`` (e.g. local CSV for
            ``train_unseen``).
        n_samples : int | None
            If given, restrict to at most this many clips (applied after
            shuffling and ``percentage``).
        percentage : float
            Fraction of the full listing to use (0.0–1.0). Default 1.0.
        shuffle : bool
            If ``True``, shuffle the clip listing with ``random_seed`` before
            subsetting. Default ``False``.
        random_seed : int
            RNG seed used when ``shuffle`` is ``True``. Default 42.
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys. Omit ``"audio"``
            to suppress audio loading.
        sample_rate : int | None
            Target audio sample rate. Default 16000.
        backend : BackendType
            Unused; present for API consistency.
        streaming : bool
            Unused; present for API consistency.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        if split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split {split!r} for AnimalSpeakPseudovox. "
                f"Expected one of {list(self.info.split_paths.keys())}"
            )
        self.split = split
        self.gcs_path = gcs_path.rstrip("/")
        self.sample_rate = sample_rate
        self._pseudovox_gcs_relpath: list[str] | None = None

        path = self.info.split_paths[split]
        if split == "full" and manifest_path is not None:
            path = manifest_path
        if split_path is not None:
            path = split_path

        if str(path).lower().endswith(".csv"):
            clip_names, self._pseudovox_gcs_relpath = self._load_train_unseen_id_and_relpaths(path)
        else:
            clip_names = self._load_json_manifest(path)

        if shuffle:
            rng = random.Random(random_seed)
            order = list(range(len(clip_names)))
            rng.shuffle(order)
            clip_names = [clip_names[i] for i in order]
            if self._pseudovox_gcs_relpath is not None:
                self._pseudovox_gcs_relpath = [self._pseudovox_gcs_relpath[i] for i in order]
        if percentage < 1.0:
            n_keep = max(1, int(len(clip_names) * percentage))
            clip_names = clip_names[:n_keep]
            if self._pseudovox_gcs_relpath is not None:
                self._pseudovox_gcs_relpath = self._pseudovox_gcs_relpath[:n_keep]
        if n_samples is not None:
            clip_names = clip_names[:n_samples]
            if self._pseudovox_gcs_relpath is not None:
                self._pseudovox_gcs_relpath = self._pseudovox_gcs_relpath[:n_samples]

        self._clip_names = clip_names

    def _load(self) -> Sequence[Any] | None:
        """Tabular backend is unused; clips are indexed from ``_clip_names``."""
        return None

    def _load_json_manifest(self, path: str) -> list[str]:
        """Load the clip listing from a JSON manifest file."""
        import fsspec

        if path.startswith("gs://"):
            with fsspec.open(path, "r") as f:
                data = json.load(f)
        else:
            data = json.loads(Path(path).read_text())

        clip_names = data.get("clip_names")
        if not isinstance(clip_names, list) or not clip_names:
            raise ValueError(f"Manifest at {path!r} has no valid 'clip_names' list.")
        return clip_names

    def _load_train_unseen_id_and_relpaths(self, path: str) -> tuple[list[str], list[str]]:
        """Load sample ids and GCS object paths (relative to ``_DEFAULT_AUDIO_BUCKET_ROOT``)."""
        import polars as pl

        df = pl.read_csv(path)
        col = "pseudovox_audio_fp"
        if col not in df.columns:
            raise ValueError(
                f"Train-unseen CSV at {path!r} must contain a {col!r} column; got {df.columns!r}"
            )
        ids: list[str] = []
        rels: list[str] = []
        for raw in df[col].to_list():
            raw_s = str(raw).strip().replace("\\", "/")
            decoded = _decode_csv_uri_path(raw_s)
            rels.append(decoded)
            ids.append(_clip_stem_from_pseudovox_audio_fp(raw_s))
        return ids, rels

    @property
    def columns(self) -> list[str]:
        return ["clip_name", "audio_path", "audio", "sample_rate"]

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    def __len__(self) -> int:
        return len(self._clip_names)

    def _process(
        self,
        clip_name: str,
        *,
        object_gcs_relpath: str | None = None,
    ) -> dict[str, Any]:
        """Build a sample dict for a single clip.

        Parameters
        ----------
        clip_name : str
            Clip name (filename stem), used as sample id.
        object_gcs_relpath : str | None
            For ``train_unseen``, full object path under ``_DEFAULT_AUDIO_BUCKET_ROOT``
            (e.g. ``animalspeak_pseudovox/foo.wav``). When ``None``, use ``gcs_path`` +
            ``{clip_name}.wav`` (``full`` manifest split).

        Returns
        -------
        dict[str, Any]
            Sample with ``clip_name``, ``audio_path``, and optionally
            ``audio`` and ``sample_rate``.
        """
        if object_gcs_relpath is not None:
            og = object_gcs_relpath.strip()
            if og.startswith("gs://"):
                audio_path = og
            else:
                audio_path = f"{_DEFAULT_AUDIO_BUCKET_ROOT.rstrip('/')}/{og}"
        else:
            rel = _decode_csv_uri_path(clip_name)
            audio_path = f"{self.gcs_path.rstrip('/')}/{rel}.wav"
        need_audio = self.output_take_and_give is None or "audio" in self.output_take_and_give

        item: dict[str, Any] = {"clip_name": clip_name, "audio_path": audio_path}

        if need_audio:
            audio, sr = read_audio(audio_path)
            audio = audio.astype(np.float32)
            audio = audio_stereo_to_mono(audio, mono_method="average")
            if self.sample_rate is not None and sr != self.sample_rate:
                import librosa
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=self.sample_rate,
                                         scale=True, res_type="kaiser_best")
                sr = self.sample_rate
            item["audio"] = audio
            item["sample_rate"] = sr

        if self.output_take_and_give is not None:
            return {new: item[old] for old, new in self.output_take_and_give.items() if old in item}
        return item

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return the sample at position ``idx``.

        Parameters
        ----------
        idx : int
            Index into the clip listing.

        Returns
        -------
        dict[str, Any]
            Sample dict.

        Raises
        ------
        IndexError
            If ``idx`` is out of range.
        """
        if idx < 0 or idx >= len(self._clip_names):
            raise IndexError(f"index {idx} out of range for AnimalSpeakPseudovox with {len(self)} clips")
        rel = self._pseudovox_gcs_relpath[idx] if self._pseudovox_gcs_relpath is not None else None
        return self._process(self._clip_names[idx], object_gcs_relpath=rel)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["AnimalSpeakPseudovox", dict[str, Any]]:
        """Create an instance from a DatasetConfig.

        Extra config keys (allowed by ``DatasetConfig``) include ``gcs_path``,
        ``manifest_path``, ``n_samples``, ``percentage``, ``shuffle``,
        ``random_seed``. ``data_root`` is accepted as an alias for ``gcs_path``
        to match :class:`~esp_data.datasets.animalspeak.AnimalSpeak`.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration object.

        Returns
        -------
        tuple[AnimalSpeakPseudovox, dict[str, Any]]
            Dataset instance and transformation metadata.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        split = cfg.get("split") or "full"
        if split == "train":
            split = "full"
        gcs_path = cfg.get("gcs_path") or cfg.get("data_root") or _DEFAULT_GCS_PATH
        manifest_path = cfg.get("manifest_path")
        split_path = cfg.get("split_path")
        ds = cls(
            split=split,
            gcs_path=gcs_path,
            manifest_path=manifest_path,
            split_path=split_path,
            n_samples=cfg.get("n_samples"),
            percentage=cfg.get("percentage", 1.0),
            shuffle=cfg.get("shuffle", False),
            random_seed=cfg.get("random_seed", 42),
            output_take_and_give=cfg.get("output_take_and_give"),
            sample_rate=cfg.get("sample_rate"),
            backend=cfg.get("backend", "polars"),
            streaming=cfg.get("streaming", False),
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def __str__(self) -> str:
        return (
            f"{self.info.name} (v{self.info.version})\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Clips loaded: {len(self):,}"
        )
