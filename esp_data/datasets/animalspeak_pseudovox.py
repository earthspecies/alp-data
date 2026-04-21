"""AnimalSpeak pseudovox dataset.

Silence-trimmed single-vocalization clips stored on GCS. Filenames encode
the species name and source/clip indices, e.g.::

    004-079-Rufous-Spinetail_source1_clip0.wav

The only structured metadata is the clip name itself, which is exposed as
``clip_name`` and used as the sample ID.

A pre-built manifest JSON (~4.6 M clips) lives on GCS and is loaded at
construction time. For large-scale runs, pass ``n_samples`` or ``percentage``
to work with a subset.
"""

from __future__ import annotations

import json
import random
from typing import Any, Iterator

import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import audio_stereo_to_mono, read_audio

_DEFAULT_GCS_PATH = "gs://fewshot/data_large_clean/animalspeak_pseudovox"
_DEFAULT_MANIFEST_PATH = "gs://foundation-model-data/synthetic/animalspeak_pseudovox_manifest.json"


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

    The full manifest (~4.6 M clips) is loaded from a pre-built JSON on GCS.
    Use ``n_samples`` or ``percentage`` to work with a subset.

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
        split_paths={},
        version="1.0.0",
        description="Silence-trimmed single-vocalization clips from AnimalSpeak (~4.6 M clips).",
        sources="gs://fewshot/data_large_clean/animalspeak_pseudovox",
        license="CC-BY-NC-4.0",
    )

    def __init__(
        self,
        gcs_path: str = _DEFAULT_GCS_PATH,
        manifest_path: str = _DEFAULT_MANIFEST_PATH,
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
        gcs_path : str
            GCS URI of the directory containing ``.wav`` files.
        manifest_path : str
            Path (local or GCS URI) to the pre-built JSON manifest.
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
        self.gcs_path = gcs_path.rstrip("/")
        self.manifest_path = manifest_path
        self.sample_rate = sample_rate

        clip_names = self._load_manifest(manifest_path)

        if shuffle:
            rng = random.Random(random_seed)
            rng.shuffle(clip_names)
        if percentage < 1.0:
            clip_names = clip_names[: max(1, int(len(clip_names) * percentage))]
        if n_samples is not None:
            clip_names = clip_names[:n_samples]

        self._clip_names = clip_names

    def _load_manifest(self, path: str) -> list[str]:
        """Load the clip listing from a JSON manifest file.

        Parameters
        ----------
        path : str
            Local path or GCS URI to the JSON manifest.

        Returns
        -------
        list[str]
            List of clip names (filenames without extension).

        Raises
        ------
        ValueError
            If the manifest contains no clip names or has an unexpected format.
        """
        import fsspec

        if path.startswith("gs://"):
            with fsspec.open(path, "r") as f:
                data = json.load(f)
        else:
            import pathlib
            data = json.loads(pathlib.Path(path).read_text())

        clip_names = data.get("clip_names")
        if not isinstance(clip_names, list) or not clip_names:
            raise ValueError(f"Manifest at {path!r} has no valid 'clip_names' list.")
        return clip_names

    @property
    def columns(self) -> list[str]:
        return ["clip_name", "audio_path", "audio", "sample_rate"]

    @property
    def available_splits(self) -> list[str]:
        return []

    def __len__(self) -> int:
        return len(self._clip_names)

    def _process(self, clip_name: str) -> dict[str, Any]:
        """Build a sample dict for a single clip.

        Parameters
        ----------
        clip_name : str
            Clip name (filename stem).

        Returns
        -------
        dict[str, Any]
            Sample with ``clip_name``, ``audio_path``, and optionally
            ``audio`` and ``sample_rate``.
        """
        audio_path = f"{self.gcs_path}/{clip_name}.wav"
        need_audio = self.output_take_and_give is None or "audio" in self.output_take_and_give

        item: dict[str, Any] = {"clip_name": clip_name, "audio_path": audio_path}

        if need_audio:
            audio, sr = read_audio(audio_path)
            audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
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
        return self._process(self._clip_names[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["AnimalSpeakPseudovox", dict[str, Any]]:
        """Create an instance from a DatasetConfig.

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
        ds = cls(
            output_take_and_give=cfg["output_take_and_give"],
            sample_rate=cfg["sample_rate"],
            backend=cfg["backend"],
            streaming=cfg["streaming"],
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
