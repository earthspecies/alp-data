"""NBM dataset"""

from __future__ import annotations

from io import StringIO
from typing import Any, Dict, Iterator, List

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class ArcticBirdSounds(Dataset):
    """ArcticBirdSounds Dataset

    Description
    -----------

    Recordings of birds in the arctic. Bird vocalizations are boxed (start and
    stop, high and low freq) and labeled with species. Description from the
    original publication:

    "Tracking biodiversity shifts is central to understanding past, present,
    and future global changes. Recent advances in bioacoustics and the low cost
    of high-quality automatic recorders are revolutionizing studies in
    biogeography and community and behavioral ecology with a robust assessment
    of phenology, species occurrence, and individual activity. This large
    volume of acoustic recordings has recently generated a plethora of datasets
    that can now be handled automatically, mostly via big data methods such as
    deep learning. These approaches need high-quality annotations to classify
    and detect recorded sounds efficiently. However, very few strongly
    annotated datasets—that is, with detailed information on start and end time
    of each vocalization—are openly accessible to the public. Moreover, these
    datasets mostly cover temperate species and are usually limited to a single
    year of recordings. Here, we present ArcticBirdSounds, the first open-
    access, multisite, and multiyear strongly annotated dataset of arctic bird
    vocalizations. ArcticBirdSounds offers 20 h of annotated recordings over 2
    years (2018, 2019), taken from 15 distinct plots within six locations
    across the Arctic, from Alaska to Greenland. Recordings cover the arctic
    vertebrates' breeding period and are evenly spaced during the day; they
    capture most species breeding there with 12,933 temporal annotations in
    49 classes of sounds. While these data can be used for many pressing
    ecological questions, it is also a unique resource for methodological
    development to help meet the challenges of fast ecosystem transformations
    such as those happening in the Arctic. All data, including audio files,
    annotation files, and companion spreadsheets, are available in an Open
    Science Framework repository published under a CC BY 4.0 License."

    Each entry consists of:
    - an audio recording
    - a selection table (Raven format), with Species labels

    Note that some species labels are unknown, and labeled as "Unknown"

    References
    ----------
    https://esajournals.onlinelibrary.wiley.com/doi/full/10.1002/ecy.4047
    https://osf.io/b9trx/overview

    """

    info = DatasetInfo(
        name="arctic_bird_sounds",
        owner="benjamin",
        split_paths={
            "all": "gs://esp-ml-datasets/arctic_bird_sounds/all.csv",
        },
        version="0.1.0",
        description="[MISSING]",
        sources="OSF",
        license="CC-BY-4.0",
    )

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: Dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
    ) -> None:
        """
        Parameters
        ----------
        split : str
            Split to load (key in info.split_paths).
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys (filters columns as well).
        sample_rate : int | None
            If set, audio is resampled to this rate.
        data_root : str | AnyPathT | None
            Optional root directory to prepend to each row['audio_path'].
        """
        super().__init__(output_take_and_give)
        self.split = split
        self._data: pd.DataFrame | None = None
        self.annotation_columns = ["Species"]
        self.unknown_label = "Unknown"

        self.sample_rate = sample_rate
        self.data_root = anypath(data_root) if data_root is not None else None

        # Load split CSV
        self._load()

        # If no explicit data_root, assume parent dir of the split path
        if self.data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent

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
        location = self.info.split_paths[self.split]
        self._data = pd.read_csv(location, keep_default_na=False, na_values=[""])

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self._data):
            raise IndexError(f"Index {idx} out of bounds for dataset length {len(self._data)}")

        row = self._data.iloc[idx].to_dict()

        # Resolve audio path
        audio_path = (
            (self.data_root / row["audio_path"]) if self.data_root else anypath(row["audio_path"])
        )

        # Read audio
        audio, sr = read_audio(audio_path)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        # Resample if necessary
        target_sr = self.sample_rate
        if target_sr is not None and sr != target_sr:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=target_sr,
                scale=True,
                res_type="kaiser_best",
            )
        sr = target_sr

        # Selection table
        st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")

        # Clip events outside audio (keep only events that begin before audio end)
        audio_dur = len(audio) / float(sr)
        st = st[st["Begin Time (s)"] < audio_dur].copy()

        # Build output
        row["audio"] = audio
        row["selection_table"] = st

        if self.output_take_and_give:
            item = {}
            for old_key, new_key in self.output_take_and_give.items():
                item[new_key] = row[old_key]
            return item

        return row

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["ArcticBirdSounds", dict[str, Any]]:
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            data_root=cfg["data_root"],
            sample_rate=cfg["sample_rate"],
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def get_available_labels(self, anno_column: str = "Species") -> List[str]:
        """
        Return all possible labels for a given annotation column
        anno_column is included as an optional argument for consistency
        with other detection datasets.

        Returns
        ---------
        A list of all the available labels for anno_column
        """
        available_labels = [
            "Calcarius lapponicus",
            "Gavia stellata",
            "Calidris melanotos",
            "Anser caerulescens",  # codespell:ignore
            "Larus argentatus",
            "Anser albifrons",  # codespell:ignore
            "Pluvialis dominica",
            "Phalaropus fulicarius",
            "Limosa lapponica",
            "Xema sabini",
            "Calidris alba",
            "Somateria spectabilis",
            "Calidris maritima",
            "Corvus corax",
            "Cygnus columbianus",
            "Stercorarius longicaudus",
            "Charadrius semipalmatus",
            "Somateria fischeri",
            "Stercorarius pomarinus",
            "Calidris pusilla",
            "Sterna paradisaea",
            "Limnodromus scolopaceus",
            "Branta canadensis",
            "Calidris fuscicollis",
            "Pluvialis squatarola",
            "Plectrophenax nivalis",
            "Calidris alpina",
            "Gavia pacifica",
            "Calidris bairdii",
            "Arenaria interpres",
            "Branta bernicla",
            "Clangula hyemalis",
        ]
        return sorted(set(available_labels))

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
