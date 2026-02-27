"""BirdSet dataset"""

from typing import Any, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://esp-ml-datasets/birdset/v0.2.0/raw"


@register_dataset
class BirdSet(Dataset):
    """BirdSet avian bioacoustics benchmark dataset.

    Description
    -----------
    BirdSet is a large-scale benchmark dataset for audio classification focusing
    on avian bioacoustics.  It includes over 6,800 recording hours from nearly
    10,000 species for training and more than 400 hours across eight strongly
    labeled evaluation datasets.  This version (v0.2.0) contains the eight
    evaluation subsets with test and test_5s splits, GBIF-linked taxonomy, and
    pre-resampled 16 kHz / 32 kHz WAV audio. The training data is not included in this dataset,
    but is a subset of the Xeno-canto dataset.

    Available Metadata Fields
    -------------------------
    **Taxonomic Information:**
        - ``species``: Scientific species name (resolved from eBird code)
        - ``species_common``: Common English name
        - ``ebird_code``: eBird species code (primary label)
        - ``ebird_code_multilabel``: JSON list of all species codes in recording
        - ``species_multispecies``: JSON list of scientific names for all species
        - ``canonical_name_multispecies``: JSON list of canonical names for all species
        - ``gbifID_multispecies``: JSON list of GBIF backbone IDs for all species
        - ``genus``, ``order``: Taxonomic hierarchy
        - ``gbifID``: GBIF backbone identifier

    **Audio File Paths:**
        - ``audio_path``: Relative path to original 32 kHz OGG audio
        - ``16khz_path``: Relative path to pre-resampled 16 kHz WAV
        - ``32khz_path``: Relative path to pre-resampled 32 kHz WAV

    **Recording Metadata:**
        - ``duration``: Recording length in seconds
        - ``lat``, ``long``: GPS coordinates
        - ``source``: Data provenance (e.g. xeno-canto recording ID)
        - ``microphone``: Recording equipment
        - ``license``: License string

    **Annotation Boundaries (test splits only):**
        - ``start_time``, ``end_time``: Temporal boundaries (seconds)
        - ``low_freq``, ``high_freq``: Frequency range (Hz); present for
          ``test`` soundscape splits, empty for ``test_5s`` clips

    Available Splits
    ----------------
    Each of the eight evaluation subsets has two splits:

    - ``{SUBSET}-test``: Full-length soundscape recordings (variable duration)
    - ``{SUBSET}-test_5s``: 5-second clips extracted from test recordings

    Subsets: HSN, NBP, NES, PER, POW, SSW, SNE, UHH.

    - ``all``: Combined dataset across all subsets and splits.

    References
    ----------
    Rauch, Lukas, et al. "BirdSet: A multi-task benchmark for classification
    in avian bioacoustics." https://arxiv.org/abs/2403.10380

    https://github.com/DBD-research-group/BirdSet

    Examples
    --------
    >>> from esp_data.datasets import BirdSet
    >>> dataset = BirdSet(split="HSN-test_5s", sample_rate=16000)
    >>> print(dataset.available_sample_rates)
    [16000, 32000]

    Load with pre-resampled 16 kHz audio (no on-the-fly resampling):

    >>> dataset_16k = BirdSet(split="POW-test_5s", sample_rate=16000)

    Load original 32 kHz OGG (returned at native sample rate):

    >>> dataset_raw = BirdSet(split="POW-test_5s")
    """

    info = DatasetInfo(
        name="birdset",
        owner="marius; gagan; david",
        split_paths={
            "HSN-test": f"{_GCS_ROOT}/HSN_test.csv",
            "HSN-test_5s": f"{_GCS_ROOT}/HSN_test_5s.csv",
            "NBP-test": f"{_GCS_ROOT}/NBP_test.csv",
            "NBP-test_5s": f"{_GCS_ROOT}/NBP_test_5s.csv",
            "NES-test": f"{_GCS_ROOT}/NES_test.csv",
            "NES-test_5s": f"{_GCS_ROOT}/NES_test_5s.csv",
            "PER-test": f"{_GCS_ROOT}/PER_test.csv",
            "PER-test_5s": f"{_GCS_ROOT}/PER_test_5s.csv",
            "POW-test": f"{_GCS_ROOT}/POW_test.csv",
            "POW-test_5s": f"{_GCS_ROOT}/POW_test_5s.csv",
            "SSW-test": f"{_GCS_ROOT}/SSW_test.csv",
            "SSW-test_5s": f"{_GCS_ROOT}/SSW_test_5s.csv",
            "SNE-test": f"{_GCS_ROOT}/SNE_test.csv",
            "SNE-test_5s": f"{_GCS_ROOT}/SNE_test_5s.csv",
            "UHH-test": f"{_GCS_ROOT}/UHH_test.csv",
            "UHH-test_5s": f"{_GCS_ROOT}/UHH_test_5s.csv",
            "all": f"{_GCS_ROOT}/birdset_all.csv",
        },
        version="0.2.0",
        description=(
            "BirdSet avian bioacoustics benchmark with GBIF-linked taxonomy. "
            "Pre-resampled audio available at 16 kHz and 32 kHz (WAV). "
            "Original audio is 32 kHz OGG from the BirdSet HuggingFace repository."
        ),
        sources=["HSN", "NBP", "NES", "PER", "POW", "SSW", "SNE", "UHH"],
        license="CC-BY-4.0, CC0",
    )

    _sample_rate_paths: dict[int, str] = {
        16000: "16khz_path",
        32000: "32khz_path",
    }

    _originals_path_column = "audio_path"

    def __init__(
        self,
        split: str = "HSN-test_5s",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the BirdSet dataset.

        Parameters
        ----------
        split : str, default="HSN-test_5s"
            The split to load.  One of ``info.split_paths`` keys, e.g.
            ``"HSN-test"``, ``"SSW-test_5s"``, or ``"all"``.
        output_take_and_give : dict[str, str], optional
            Column rename / filter mapping.
        sample_rate : int, optional
            Target sample rate.  If a pre-resampled version exists (16 kHz or
            32 kHz), the corresponding WAV is loaded directly.  Otherwise the
            original 32 kHz OGG is loaded and resampled on-the-fly.
        data_root : str | AnyPathT, optional
            Root directory prepended to relative audio paths.  Defaults to the
            GCS path for this dataset version.
        backend : BackendType, optional
            Backend engine ("pandas" or "polars"), by default "polars".
        streaming : bool, optional
            Whether to use streaming mode, by default False.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self._load()
        self.sample_rate = sample_rate

        if data_root is None:
            self.data_root = anypath(f"{_GCS_ROOT}/")
        else:
            self.data_root = anypath(data_root)

    @property
    def columns(self) -> list[str]:
        """Return the columns of the dataset."""
        return list(self._data.columns)

    @property
    def available_splits(self) -> list[str]:
        """Return the available splits of the dataset."""
        return list(self.info.split_paths.keys())

    @property
    def available_sample_rates(self) -> list[int]:
        """Return pre-resampled sample rates available in the loaded data.

        Returns
        -------
        list[int]
            Sample rates (Hz) that can be loaded without on-the-fly resampling.
        """
        available = []
        for sr, col in self._sample_rate_paths.items():
            if col in self._data.columns:
                available.append(sr)
        return sorted(available)

    def _load(self) -> None:
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        location = self.info.split_paths[self.split]
        self._data = self._backend_class.from_csv(location, streaming=self._streaming)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["BirdSet", dict[str, Any]]:
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

    def __len__(self) -> int:
        if self._data is None:
            raise RuntimeError("No split has been loaded yet.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. Iterate over the dataset instead."
            )
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        use_presampled = False
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            col = self._sample_rate_paths[self.sample_rate]
            if col in row and row[col] is not None and str(row[col]).strip():
                audio_path = self.data_root / row[col]
                use_presampled = True

        if use_presampled:
            audio, sr = read_audio(audio_path)
            audio = audio.astype(np.float32)
            audio = audio_stereo_to_mono(audio, mono_method="average")
        else:
            audio_path = self.data_root / row[self._originals_path_column]
            audio, sr = read_audio(audio_path)
            audio = audio.astype(np.float32)
            audio = audio_stereo_to_mono(audio, mono_method="average")

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
            item = {}
            for key, value in self.output_take_and_give.items():
                item[value] = row[key]
            return item

        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in self._data:
            yield self._process(row)

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version}), split={self.split}"
        return (
            f"{base}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
