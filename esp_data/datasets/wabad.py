"""WABAD dataset"""

from __future__ import annotations

import json
import os
from functools import partial
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import pandas as pd
import torch
import torchaudio

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, filesystem, read_audio

LABEL_MAPPING_PATH = "gs://esp-ml-datasets/wabad/v0.1.0/raw/wabad_label_mapping_15_10_2025.json"


@register_dataset
class WABAD(Dataset):
    """WABAD Dataset

    Description
    -----------
    This class makes WABAD dataset available. Each entry is an audio recording,
    plus a selection table. Each row of the selection table has annotations at
    different taxonomic granularities (stored in annotation_columns attribute).
    Taxonomy has been coerced into GBIF.

    This class was included in esp-data (initially) for use as a zero-shot
    detection evaluation dataset.

    Description from publication:
    https://www.researchgate.net/publication/387711208_WABAD_A_World_Annotated_Bird_Acoustic_Dataset_for_Passive_Acoustic_Monitoring

    Under the current global biodiversity crisis, there is a need for automated
    and non-invasive monitoring techniques that can gather large amounts of data
    cost-effectively at various ecological scales, from local to large spatial
    scales. This data can then be analyzed to inform stakeholders and decision
    makers. One such technique is passive acoustic monitoring, which is commonly
    coupled with automatic identification of animal species based on their sound.
    Automated sound analyses usually require the training of sound detection and
    identification algorithms. These algorithms are based on annotated acoustic
    datasets which mark the occurrence of sounds of species inside sound
    recordings. However, compiling large annotated acoustic datasets is time-
    consuming and requires experts, and therefore they normally cover reduced
    spatial, temporal and taxonomic scales. This data paper presents WABAD, the
    World Annotated Bird Acoustic Dataset for passive acoustic monitoring. WABAD
    is designed to provide the public, the research community, and conservation
    managers with a novel and globally representative annotated acoustic dataset.
    This database includes 5,047 minutes of audio files annotated to species-level
    by local experts with the start and end time, and the upper and lower
    frequencies of each identified bird vocalisation in the recordings. The
    database has a wide taxonomic and spatial coverage, including information on
    91,931 vocalisations from 1,192 bird species recorded at 72 recording sites in
    29 recording locations (mainly countries) and distributed across 13 biomes.
    WABAD can be used, for example, for developing and/or validating automatic
    species detection algorithms, answering ecological questions, such as assessing
    geographical variations on bird vocalisations, or comparing acoustic diversity
    indices with species-based diversity indices. The dataset is published under a
    Creative Commons Attribution Non Commercial 4.0 International copyright.

    References
    ----------
    https://zenodo.org/records/15629388
    https://www.researchgate.net/publication/387711208_WABAD_A_World_Annotated_Bird_Acoustic_Dataset_for_Passive_Acoustic_Monitoring

    """

    info = DatasetInfo(
        name="wabad",
        owner="benjamin",
        split_paths={"all": "gs://esp-ml-datasets/wabad/v0.1.0/raw/all_info.csv"},
        version="0.1.0",
        description="[MISSING]",
        sources="zenodo.org",
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
            Optional root directory to prepend to each row['audio_fp'].
        """
        super().__init__(output_take_and_give)
        self.split = split
        self._data: pd.DataFrame | None = None
        self.default_anno_column = "Species"
        self.unknown_annos: List[str] = []
        self.annotation_columns = ["Genus", "Family", "Order", "Common", "Species"]
        assert self.default_anno_column == self.annotation_columns[-1]

        self.sample_rate = sample_rate
        self.data_root = anypath(data_root) if data_root is not None else None

        # Load split CSV
        self._load()

        # Default label mapping path

        label_mapping_fn = os.path.basename(LABEL_MAPPING_PATH)
        label_mapping_path_local = Path(__file__).parent / label_mapping_fn
        if not os.path.exists(label_mapping_path_local):
            print(f"Getting {LABEL_MAPPING_PATH} from GCP")
            fs = filesystem("gcs")
            fs.get(LABEL_MAPPING_PATH, label_mapping_path_local)

        self.label_mappings = self._load_label_mappings(anypath(label_mapping_path_local))

        # If no explicit data_root, assume parent dir of the split path
        if self.data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent

    # -----------------------------
    # Properties
    # -----------------------------
    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    # -----------------------------
    # Internals
    # -----------------------------
    def _load(self) -> None:
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        location = self.info.split_paths[self.split]
        self._data = pd.read_csv(location, keep_default_na=False, na_values=[""])

    @staticmethod
    def _load_label_mappings(fp: Path) -> Dict[str, Dict[str, str]]:
        if not fp.exists():
            raise FileNotFoundError(f"{fp} not found. ")
        with open(fp, "r") as f:
            lm = json.load(f)
        # Ensure all expected keys present
        for k in ["Genus", "Family", "Order", "Common", "Species"]:
            lm.setdefault(k, {})
        return lm

    def _label_mapping(self, x: str, anno_column: str) -> str:
        """Map a Species (default) label to any desired annotation column.

        Returns
        ----------
            str The new label
        """
        mapping = self.label_mappings.get(anno_column, {})
        return mapping.get(x, x)

    # -----------------------------
    # Dataset API
    # -----------------------------
    def __len__(self) -> int:
        if self._data is None:
            raise RuntimeError("No split loaded.")
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._data is None:
            raise RuntimeError("No split loaded.")
        if idx < 0 or idx >= len(self._data):
            raise IndexError(f"Index {idx} out of bounds for dataset length {len(self._data)}")

        row = self._data.iloc[idx].to_dict()

        # Resolve audio path
        audio_path = (
            (self.data_root / row["audio_fp"]) if self.data_root else anypath(row["audio_fp"])
        )

        # Read audio
        audio, sr = read_audio(audio_path)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        # Resample if needed
        target_sr = self.sample_rate
        if target_sr is not None and sr != target_sr:
            audio = torchaudio.functional.resample(
                torch.tensor(audio),
                sr,
                target_sr,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            ).numpy()
            sr = target_sr

        # Selection table
        st = pd.read_csv(StringIO(row["selection_table_str"]), sep="\t")

        # Clip events outside audio (keep only events that begin before audio end)
        audio_dur = len(audio) / float(sr)
        st = st[st["Begin Time (s)"] < audio_dur].copy()

        # Add remapped annotation columns derived from default column
        for anno_col in self.annotation_columns:
            f = partial(self._label_mapping, anno_column=anno_col)
            st[anno_col] = st[self.default_anno_column].map(f)

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

    # -----------------------------
    # Factory
    # -----------------------------
    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["WABAD", dict[str, Any]]:
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

    # -----------------------------
    # Convenience
    # -----------------------------
    def get_available_labels(self, anno_column: str) -> List[str]:
        """
        Return all possible labels for a given annotation column,
        via mapping from the default Species labels found in the split.

        Returns
        ---------
        A list of all the available labels for anno_column
        """
        if self._data is None:
            return []
        species = set()
        for _, row in self._data.iterrows():
            st = pd.read_csv(StringIO(row["selection_table_str"]), sep="\t")
            species.update(st[self.default_anno_column].astype(str).tolist())
        mapped = {self._label_mapping(x, anno_column) for x in species}
        return sorted(mapped)

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )


if __name__ == "__main__":
    ds = WABAD()

    sample = ds[10]

    assert "audio" in sample
    print(f"Audio shape: {sample['audio'].shape}")
    st1 = sample["selection_table"]
    print(st1.head(10))
