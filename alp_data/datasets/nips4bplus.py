"""NIPS4Bplus dataset"""

from __future__ import annotations

from io import StringIO
from typing import Any, Dict, Iterator, List

import librosa
import numpy as np
import pandas as pd

from alp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from alp_data.backends import BackendType
from alp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

# _RAW_ROOT = f"{DATA_HOME}/nips4bplus/v0.1.0/raw"

# temporary override; remove before merge
_RAW_ROOT = "gs://esp-data-ingestion/nips4bplus/formatted"


@register_dataset
class NIPS4BPlus(Dataset):
    """NIPS4BPlus Dataset

    Description
    -----------
    Short recordings of birds and their ecosystem from France, with
    temporal annotations of every vocalisation. The recordings are the training
    set of the NIPS4B 2013 bird song classification challenge, provided by
    Biotope. The NIPS4Bplus annotations were added later by Veronica Morfi,
    Dan Stowell and Hanna Pamuła.

    Description of the recordings from the challenge website:

    "This Bird NIPS4B competition asks participants to identify which of 87
    sound classes of birds and their ecosystem are present into 1 000
    continuous wild recordings (from different places in France -
    nearly 2 hours of recordings, frequency sample = 44.1 kHz, SM2 system).
    The data is provided by the BIOTOPE society (having the largest collection
    of wild recordings of birds in Europe). The training set matches the test
    set conditions. [...] For some species we discriminate the song to the
    call (and to the drum). We also include some species living within with
    these birds: 7 insects and a batracian. [...] Some training files are
    empty (background noise only called 'empty class') [...]"

    Description of the annotations from the Figshare (version 7):

    "The zip file contains 674 individual recording temporal annotations for
    the training set of the NIPS4B 2013 dataset in the birdsong
    classifications task (original size of dataset is 687 recordings). [...]
    Transcriptions were produced using Sonic Visualiser [...] by an
    experienced birdwatcher, Hanna Pamula. Number of missing annotations: 13
    (6 of these files contained sounds which could not be unambiguously
    labelled and the rest 7 of them only included insects). The original
    (weak) labels provided during the NIPS4B 2013 challenge were used for
    guidance. However, some files were judged to have a slightly different set
    of species present than was given in the original metadata. Extra Unknown
    label was added to the dataset for the vocalisations that couldn't be
    classified to a specific species. Also, extra Human label was added for a
    few recordings that had human sounds present in them."

    Only the 687 challenge training recordings are used, since the 1000 test
    recordings have no temporal annotations. The original weak (file-level)
    challenge labels are not used. Recordings are 1-5 s long, mono, 44.1 kHz.

    Recordings
    ----------
    Of the 687 training recordings, 669 are included:

    - the 13 recordings without an annotation file (see above) are omitted.
    - 5 recordings (``nips4b_birds_trainfile`` 061, 090, 215, 453 and 492) are
      omitted because their annotation files are 0 bytes, although their weak
      labels list a species. They are treated as unannotated.
    - the 100 recordings the challenge labels as background noise only
      (``Empty``) are kept, with empty selection tables.

    Labels
    ------
    Each event has two labels:

    - ``class_name``: the original label, e.g. ``Carcan_call``. The challenge's
      87 classes combine a species code with a sound type (``call``, ``song``
      or ``drum``). The annotators added ``Unknown`` and ``Human``.
    - ``Species``: the scientific name of the class, mapped to the accepted
      GBIF canonical name. ``Human`` becomes ``Homo sapiens`` and ``Unknown``
      stays ``Unknown``. Graf's hybrid frog (``Pelophylax kl. grafi``) is a
      klepton with no GBIF entry, so its name is kept as given. GBIF renames
      ``Tibicina tomentosa`` to ``Tibicina picta``. The source spellings
      ``lullula arborea`` and ``Regulus ignicapillus`` are corrected before
      lookup.

    Events have onsets and offsets but no frequency bounds. Events of the same
    class may overlap.

    Splits
    ------
    - ``all``: all 669 recordings, bird events only (77 classes, 51 species).
    - ``all_with_nonbirds``: all 669 recordings, with insect, amphibian,
      ``Human`` and ``Unknown`` events as well (89 classes). Noise is omitted.

    Each entry consists of:
    - an audio recording
    - a selection table (Raven format), with Species and class_name labels

    References
    ----------
    Annotations: [Figshare](https://doi.org/10.6084/m9.figshare.6798548.v7)
    Recordings: [NIPS4B 2013](http://sabiod.univ-tln.fr/nips4b/challenge1.html)

    """

    info = DatasetInfo(
        name="nips4bplus",
        owner="benjamin",
        split_paths={
            "all": f"{_RAW_ROOT}/all.csv",
            "all_with_nonbirds": f"{_RAW_ROOT}/all_with_nonbirds.csv",
        },
        version="0.1.0",
        description=(
            "669 short (1-5 s) recordings from France (NIPS4B 2013 training "
            "set), with temporal annotations for 77 bird classes (51 species), plus "
            "insect, amphibian, human and unknown sounds."
        ),
        sources="Figshare, NIPS4B 2013",
        license=("CC-BY-4.0"),
    )

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: Dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        split : str
            Split to load (key in info.split_paths).
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys (filters columns as well).
        sample_rate : int | None
            If set, audio is resampled to this rate. Defaults to None. The native
            rate is 44100.
        data_root : str | AnyPathT | None
            Optional root directory to prepend to each row['audio_path'].
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars"
        streaming : bool, optional
            Whether to use streaming mode, by default False
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self.annotation_columns = ["Species", "class_name"]

        self.sample_rate = sample_rate
        self.data_root = anypath(data_root) if data_root is not None else None

        # Load split CSV
        self._load()

        # If no explicit data_root, assume parent dir of the split path
        if self.data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns)

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        """Load the CSV of the current split.

        Raises
        ------
        LookupError
            If `self.split` is not one of `available_splits`.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        location = self.info.split_paths[self.split]
        self._data = self._backend_class.from_csv(
            location, streaming=self._streaming, keep_default_na=False, na_values=[""]
        )

    def __len__(self) -> int:
        """Number of recordings in the loaded split.

        Returns
        -------
        int
            Number of recordings.

        Raises
        ------
        RuntimeError
            If no split has been loaded.
        NotImplementedError
            If the dataset is in streaming mode.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. Iterate over the dataset instead."
            )
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row of the dataset.

        Parameters
        ----------
        row : dict[str, Any]
            A dictionary representing a single row of the dataset.

        Returns
        -------
        dict[str, Any]
            The processed row.
        """

        # Resolve audio path
        audio_path = (
            (self.data_root / row["audio_path"]) if self.data_root else anypath(row["audio_path"])
        )

        # Read audio
        audio, sample_rate = read_audio(audio_path)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        # Resample if necessary
        if self.sample_rate is not None and sample_rate != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sample_rate,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
            sample_rate = self.sample_rate

        # Selection table
        st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")

        # Clip events outside audio (keep only events that begin before audio end)
        audio_dur = len(audio) / float(sample_rate)
        st = st[st["Begin Time (s)"] < audio_dur].copy()

        # Build output
        row["audio"] = audio
        row["sample_rate"] = sample_rate
        row["selection_table"] = st

        if self.output_take_and_give:
            item = {}
            for old_key, new_key in self.output_take_and_give.items():
                item[new_key] = row[old_key]
            return item

        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a specific sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to get.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the processed data.
        """
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        -------
        dict[str, Any]
            Each sample in the dataset.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["NIPS4BPlus", dict[str, Any]]:
        """Create the dataset from a `DatasetConfig`.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration with the constructor arguments and optional transformations.

        Returns
        -------
        tuple[NIPS4BPlus, dict[str, Any]]
            The dataset and the metadata returned by its transformations (empty if none).
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

    def get_available_labels(self, anno_column: str = "Species") -> List[str]:
        """Return all possible labels for a given annotation column.

        Parameters
        ----------
        anno_column : str, optional
            Selection table column to collect labels from, "Species" (default) or
            "class_name".

        Returns
        -------
        list[str]
            Sorted unique labels found in `anno_column` across all selection tables.
        """
        available_labels = set()
        for row in self._data:
            st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
            available_labels.update(st[anno_column].astype(str).tolist())
        return sorted(available_labels)

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
