"""Anuraset dataset"""

from __future__ import annotations

from io import StringIO
from typing import Any, Dict, Iterator, List

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_dataset
class AnuraSetStrong(Dataset):
    """AnuraSetStrong Dataset

        Description
        -----------
        This is the strongly labeled portion of AnuraSet, i.e. the portion with
        start- and stop-times annotated.

        Description from "AnuraSet: A dataset for benchmarking Neotropical anuran
    calls identification in passive acoustic monitoring" by Canas et al. (2023)

        "We introduce a large-scale multi-species dataset of anuran amphibians
        calls recorded by PAM, that comprises 27 hours of expert annotations
        for 42 different species from two Brazilian biomes.

        To provide precise annotations, we identified bouts of advertisement
        calls within each audio file and generated strong labels for them (step 1).
        Using Audacity 3.2 software, we conducted a detailed visual and aural
        inspection of the spectrogram to identify temporal limits (beginning and end)
        of audio segments containing species-specific calls with an inter-call interval
        of less than 1 second. These annotations ensured fine-scale specificity (Figure 3).
        For longer intervals, we split the calls into different time boxes and labeled
        them independently. Detailed labels assigned to time boxes were composed of (i)
        the species ID, tagged with a unique 6-letter code built from the scientific
        name of each identified species (Table 2), and (ii) the perceived quality of the
        recorded signal, included as a single letter indicating a Low (’L’), Medium (’M’),
        or High (’H’) quality (Figure 4). To ensure consistency among the perceptual quality
        labels, we set up the following criteria: A high-quality call has a high signal-to-noise
        ratio, no overlap with other sounds, has a well-identifiable structure on the spectrogram,
        and can be easily visualized on the oscillogram. A medium-quality call can be
        visually identified on the spectrogram but may overlap with other sounds that can be
        difficult to identify in the oscillogram. A low-quality call shows a low signal-to-noise
        ratio, is partially masked by other sounds, appears with low intensity on the spectrogram,
        and cannot be easily identified on the oscillogram. This information was used to increase
        the usability of the data and improve the error analysis of the learning model."

        Note that we omitted the quality assessments.

        Each entry consists of:
        - an audio recording
        - a selection table (Raven format), with Species labels

        References
        ----------
        https://arxiv.org/pdf/2307.06860

    """

    info = DatasetInfo(
        name="anuraset_strong",
        owner="benjamin",
        split_paths={
            "all": "gs://esp-ml-datasets/anuraset/anuraset_all_gbif.csv",
        },
        version="0.1.0",
        description="[MISSING]",
        sources="Zenodo",
        license="CC BY 1.0",
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
        self.annotation_columns = ["Species"]

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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["AnuraSetStrong", dict[str, Any]]:
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

        Returns
        ---------
        A list of all the available labels for anno_column
        """
        # available_labels = set()
        # for _, row in self._data.iterrows():
        #     st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
        #     available_labels.update(st[anno_column].astype(str).tolist())

        available_labels = [
            "Adenomera diptyx",
            "Adenomera marmorata",
            "Ameerega picta",
            "Boana albomarginata",
            "Boana albopunctata",
            "Boana bischoffi",
            "Boana faber",
            "Boana leptolineata",
            "Boana lundii",
            "Boana prasina",
            "Boana raniceps",
            "Dendropsophus cruzi",
            "Dendropsophus elegans",
            "Dendropsophus minutus",
            "Dendropsophus nahdereri",
            "Dendropsophus nanus",
            "Elachistocleis bicolor",
            "Elachistocleis matogrosso",
            "Leptodactylus elenae",
            "Leptodactylus flavopictus",
            "Leptodactylus fuscus",
            "Leptodactylus labyrinthicus",
            "Leptodactylus latrans",
            "Leptodactylus notoaktites",
            "Leptodactylus podicipinus",
            "Ololygon rizibilis",
            "Phyllomedusa distincta",
            "Phyllomedusa sauvagii",
            "Physalaemus albonotatus",
            "Physalaemus cuvieri",
            "Physalaemus marmoratus",
            "Physalaemus nattereri",
            "Pithecopus azureus",
            "Rhinella icterica",
            "Rhinella ornata",
            "Rhinella scitula",
            "Scinax alter",
            "Scinax fuscomarginatus",
            "Scinax fuscovarius",
            "Scinax nasicus",
            "Scinax perereca",
            "Sphaenorhynchus surdus",
        ]

        return sorted(available_labels)

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
