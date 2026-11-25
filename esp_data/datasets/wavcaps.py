"""WavCaps dataset: web and AudioSet audio with ChatGPT-condensed captions."""

from __future__ import annotations

from typing import Any, Iterator

import librosa
import numpy as np
import polars as pl

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://esp-data-ingestion/wavcaps/v0.1.0"
# Read as text: ids and titles are often all digits.
_TEXT_COLUMNS = ("id", "title", "tags", "category", "author", "description")


@register_dataset
class WavCaps(Dataset):
    """WavCaps Dataset.

    Description
    -----------
    WavCaps (Mei et al., 2023) pairs general audio with **one short English
    caption per clip**. The captions are weak labels: ChatGPT condensed each
    clip's raw online description (FreeSound, BBC Sound Effects, SoundBible) or
    its strong event labels (AudioSet SL) into a sentence. No human wrote or
    checked them. The ``source`` column names the origin:

    - ``FreeSound``: user uploads, captions from the uploader's title, tags
      and description (``description``, ``tags``, ``title``, ``author`` kept).
    - ``BBC_Sound_Effects``: BBC archive, captions from the catalogue
      description (``description``, ``category`` kept).
    - ``SoundBible``: captions from the site's title and description.
    - ``AudioSet_SL``: 10 s AudioSet clips, captions from the temporally
      strong labels. ``id`` is the PANNs file stem, ``Y<youtube id>``.

    Only clips of **at most 60 s** (by the upstream duration) are ingested.
    A caption describes the whole recording, so it says little about a short
    window of a long one, and the long tail is most of the upstream bytes
    (about 7,560 h upstream in 403,050 clips). The longer clips can be added
    by rerunning the build with a larger ``MAX_SECONDS``.

    Splits
    ------
        - ``all``   : every ingested clip.
        - ``train`` : ``all`` minus ``val`` and minus the clips WavCaps lists
          as overlapping the AudioCaps or Clotho test sets, UrbanSound8K,
          ESC-50 or VGGSound (columns ``bl_audiocaps_clotho_test``,
          ``bl_ub8k_esc50_vggsound``).
        - ``val``   : a fixed 1% of the non-excluded clips (hash of source and id).

    ``bl_audiocaps_clotho_all`` marks WavCaps' wider list (overlap with any
    AudioCaps or Clotho split); those clips stay in ``train``. Filter on the
    column for zero-shot evaluation on AudioCaps or Clotho. The upstream
    blacklists cover FreeSound and AudioSet only.

    Audio
    -----
    Upstream audio is 32 kHz mono 16-bit FLAC and is stored unchanged under
    ``audio_32k/<source>/<id>.flac``; a 16 kHz mirror is under ``audio_16k/``.
    ``audio_duration`` is measured from the file.

    Licence
    -------
    The dataset card is tagged CC-BY-4.0 but states "Only academic uses are
    allowed for WavCaps dataset" and that the audio is for research purposes
    only. Each FreeSound, BBC and SoundBible clip also keeps its own source
    licence (``href`` / ``download_link`` point to the source page). Treat
    as research-only.

    References
    ----------
    Mei et al. (2023). "WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio
    Captioning Dataset for Audio-Language Multimodal Research". arXiv:2303.17395
    https://huggingface.co/datasets/cvssp/WavCaps
    """

    info = DatasetInfo(
        name="wavcaps",
        owner="david",
        split_paths={
            "all": f"{_GCS_ROOT}/wavcaps_all.csv",
            "train": f"{_GCS_ROOT}/wavcaps_train.csv",
            "val": f"{_GCS_ROOT}/wavcaps_val.csv",
        },
        version="0.1.0",
        description=(
            "WavCaps: general audio from FreeSound, BBC Sound Effects, SoundBible and "
            "AudioSet SL with one ChatGPT-condensed caption per clip (Mei et al., 2023). "
            "Clips of at most 60 s; 32 kHz mono FLAC with a 16 kHz mirror."
        ),
        sources=["https://huggingface.co/datasets/cvssp/WavCaps"],
        license="Research use only (dataset card); per-clip source licences for the audio",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _originals_path_column = "audio_fp"

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "pandas",
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        split : str
            Split to load: ``all`` / ``train`` / ``val``.
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys (filters columns).
        sample_rate : int | None
            If set, audio is loaded at this rate.
        data_root : str | AnyPathT | None
            Optional root directory to prepend to each row's audio path.
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "pandas".
        streaming : bool, optional
            Whether to use streaming mode, by default False.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self.sample_rate = sample_rate

        self._load()

        if data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent
        else:
            self.data_root = anypath(data_root)

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    @property
    def available_sample_rates(self) -> list[int]:
        """Return pre-resampled sample rates whose path columns exist.

        Returns
        -------
        list[int]
            Sample rates (Hz) for which pre-resampled audio is available.
        """
        return [sr for sr, col in self._sample_rate_paths.items() if col in self._data.columns]

    def _load(self) -> None:
        """Load the manifest CSV for ``self.split``.

        Raises
        ------
        LookupError
            If ``self.split`` is not a key in ``info.split_paths``.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        location = self.info.split_paths[self.split]
        self._data = self._backend_class.from_csv(
            location,
            streaming=self._streaming,
            schema_overrides={c: pl.Utf8 for c in _TEXT_COLUMNS},
            dtype={c: "string" for c in _TEXT_COLUMNS},
            null_values=["", "Not found"],
            keep_default_na=False,
            na_values=["", "Not found"],
        )

    def __len__(self) -> int:
        """Return the number of clips in the dataset.

        Returns
        -------
        int
            Number of audio clips in the current split.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. Iterate over the dataset instead."
            )
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Load audio and return the row with its caption.

        Returns
        -------
        dict[str, Any]
            Row dict with ``audio`` (numpy float32) and ``sample_rate`` (int)
            populated alongside the original caption / metadata columns.
        """
        use_presampled = False
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            path_column = self._sample_rate_paths[self.sample_rate]
            if path_column in row and row[path_column] is not None and row[path_column] != "":
                audio_path = anypath(self.data_root) / row[path_column]
                use_presampled = True

        if not use_presampled:
            audio_path = anypath(self.data_root) / row[self._originals_path_column]

        audio, sr = read_audio(audio_path)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        if not use_presampled and self.sample_rate is not None and sr != self.sample_rate:
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
        """Get a specific sample from the dataset.

        Returns
        -------
        dict[str, Any]
            Dict with audio, sample_rate, caption and metadata.
        """
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        ------
        dict[str, Any]
            Each sample in the dataset.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["WavCaps", dict[str, Any]]:
        """Create a WavCaps instance from a DatasetConfig.

        Returns
        -------
        tuple[WavCaps, dict[str, Any]]
            The instantiated dataset + apply_transformations metadata (empty
            when no transforms are configured).
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            data_root=cfg["data_root"],
            sample_rate=cfg["sample_rate"],
            backend=cfg["backend"],
            streaming=cfg["streaming"],
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
