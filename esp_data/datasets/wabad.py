"""WABAD dataset"""

import os
from functools import partial
from pathlib import Path
from io import StringIO
from typing import Any, Iterator, List

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

import json

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.io import (
    AnyPathT,
    anypath,
    audio_stereo_to_mono,
    read_audio,
)

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
        description="[MISSING]", # Redundant with docstring
        sources="zenodo.org",
        license="Creative Commons Attribution 4.0 International",
    )

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        split : str
            The split to load. One of info.split_paths keys.
        output_take_and_give : dict[str, str]
            A dictionary mapping the original column names to the new column names.
            It acts as a filter as well.
        sample_rate : int
            The sample rate to which audio files should be resampled.
        data_root : str | AnyPathT, optional
            The root directory for the dataset. This is optionally appended to the
            path item of a sample in the dataset.
            If None, the default is the parent directory of the split path.
        annotation_columns : List
            List of column names that can be used as annotations in the selection table
        label_mappings : Dict
            Dict of Dicts; each sub-dict is a mapping from a Species name that can occur to a different name
        default_anno_column : str
            The default anno column name, out of which the label mappings are made
        unknown_annos : List
            List of labels in default_anno_column which are considered as unknown.
            Included for consistency with other zero-shot detection datasets.
        """
        super().__init__(output_take_and_give)  # Initialize the parent Dataset class
        self.split = split
        self._data: pd.DataFrame = None
        self.default_anno_column = "Species"
        self.unknown_annos = []
        self.annotation_columns = ["Genus", "Family", "Order", "Common", "Species"]
        assert (
            self.annotation_columns[-1] == self.default_anno_column
        )  # we need to modify this one last for label remapping to work properly
        self.label_mappings = {x: {} for x in self.annotation_columns}  # mapping from default to other name
        self._load()  # Load the dataset (fills self._data)
        self.sample_rate = sample_rate
        self.data_root = data_root
        if self.data_root is None:
            # we assume that parent dir of the split path is the data root
            self.data_root = anypath(self.info.split_paths[self.split]).parent

    @property
    def columns(self) -> list[str]:
        """Return the columns of the dataset."""
        return list(self._data.columns)

    @property
    def available_splits(self) -> list[str]:
        """Return the available splits of the dataset."""
        return list(self.info.split_paths.keys())

    def get_available_labels(self, anno_column: str) -> List:
        """
        anno_column (str) : name of annotation column

        Returns
        --------
        a list of labels that can appear under column anno_column
        """

        out = []
        for _, row in self._data.iterrows():
            st = pd.read_csv(StringIO(row["selection_table_str"]), sep="\t")
            out.extend(st[self.default_anno_column].tolist())
        out = set(out)
        out = sorted(set(sorted([self._label_mapping(x, anno_column) for x in out])))
        return out

    def _label_mapping(self, x: str, anno_column: str) -> str:
        """
        Transform label from the default
        to anno_column, assuming self.label_mappings has been pre-computed

        Returns
        -----
        str new label
        """
        if x in self.label_mappings[anno_column]:
            return self.label_mappings[anno_column][x]
        else:
            return x

    def _get_label_mappings(self) -> None:
        """
        Precompute label mapping
        """
        default_labels = self.get_available_labels(self.default_anno_column)

        _here = Path(__file__).parent
        info_fp = _here / "gbif_wabad_taxonomic_info.json"

        species_label_fix = {
            "Campethera nivosa": "Pardipicus nivosus",
            "Eopsaltria flaviventris": "Cryptomicroeca flaviventris",
            "Gliciphila undulata": "Glycifohia undulata",
            "Lanius corvinus": "Corvinella corvina",
            "Neocossyphus fraseri": "Stizorhina fraseri",
            "Oreolais rufogularis": "Oreolais pulcher",
            "Phylloscartes ophthalmicus": "Pogonotriccus ophthalmicus",
            "Rubigula cyaniventris": "Ixodia cyaniventris",
            "Rubigula erythropthalmos": "Ixodia erythropthalmos",
            "Streptopelia chinensis": "Spilopelia chinensis",
            "Streptopelia senegalensis": "Spilopelia senegalensis",
            "Telophorus multicolor": "Chlorophoneus multicolor",
        }

        if os.path.exists(info_fp):
            print("Reloading GBIF taxonomy from cache")
            with open(info_fp, 'r') as f:
                json.load(f)

        else:
            print("Checking against GBIF/animalspeak taxonomy")
            import requests

            base_url = "http://gagan-dev:8000"
            
            for default_label in tqdm(default_labels):
                if default_label in species_label_fix.keys():
                    species_label = species_label_fix[default_label]
                else:
                    species_label = default_label
                response = requests.get(f"{base_url}/taxonomy/{species_label}")

                if response.status_code == 200:
                    gbif_data = response.json()
                else:
                    print(f"Error: {response.status_code} - {response.text}")
                    continue

                self.label_mappings["Species"][default_label] = species_label
                self.label_mappings["Genus"][default_label] = gbif_data['genus']
                self.label_mappings["Order"][default_label] = gbif_data['order']
                self.label_mappings["Family"][default_label] = gbif_data['family']
                self.label_mappings["Common"][default_label] = "" if gbif_data['species_common'] is None else gbif_data['species_common']

            with open(info_fp, 'w') as f:
                json.dump(self.label_mappings, f)

        for anno_column in self.annotation_columns:
            available_labels = self.get_available_labels(anno_column)
            print(f"There are {len(available_labels)} available labels for {anno_column}:")
            print(available_labels)

    def _load(self) -> None:
        """Load the dataset.

        Raises
        ------
        LookupError
            If the split is not valid.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(f"Invalid split: {self.split}.Expected one of {list(self.info.split_paths.keys())}")

        location = self.info.split_paths[self.split]
        self._data = pd.read_csv(
            location, keep_default_na=False, na_values=[""]
        )  # This setting avoids setting 'None' to a pd.NA type

        self._get_label_mappings()

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["WABAD", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary containing dataset parametesf

        Returns
        -------
        tuple[Dataset, dict[str, Any]]
            A tuple containing the dataset instance and metadata.
            If the dataset_config contains transformations, they will be applied
            and the metadata will be returned as dict, otherwise an empty dict.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})

        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            data_root=cfg["data_root"],
            sample_rate=cfg["sample_rate"],
        )

        if dataset_config.transformations:
            transform_metadata = ds.apply_transformations(dataset_config.transformations)
            return ds, transform_metadata

        return ds, {}

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the current split.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call load() first.")
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a specific sample from the dataset.
        Parameters
        ----------
        idx : int
            Index of the sample to get.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the data.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if idx >= len(self._data):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self._data)}.")

        row = self._data.iloc[idx].to_dict()
        # Ensure audio path is valid
        if self.data_root:
            audio_path = anypath(self.data_root) / row["audio_fp"]
        else:
            audio_path = anypath(row["audio_fp"])

        # Read the audio clip
        audio, sr = read_audio(audio_path)
        audio = audio.astype(np.float32)

        # Stereo to mono if necessary.
        audio = audio_stereo_to_mono(audio, mono_method="average")

        if self.sample_rate is not None and sr != self.sample_rate:
            audio = torchaudio.functional.resample(
                torch.tensor(audio),
                sr,
                self.sample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            ).numpy()

        row["audio"] = audio
        row["selection_table"] = pd.read_csv(StringIO(row["selection_table_str"]), sep="\t")

        audio_dur = len(audio)[0] / self.sample_rate
        row['selection_table'] = row['selection_table'][row['selection_table']["Begin Time (s)"] < audio_dur]

        for anno_column in self.annotation_columns:
            f = partial(self._label_mapping, anno_column=anno_column)
            row["selection_table"][anno_column] = row["selection_table"][self.default_anno_column].map(f)

        if self.output_take_and_give:
            item = {}
            for key, value in self.output_take_and_give.items():
                item[value] = row[key]
        else:
            item = row

        return item

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        -------
        dict[str, Any]
            Each sample in the dataset.
        """
        for idx in range(len(self)):
            yield self[idx]

    def __str__(self) -> str:
        """Return a string representation of the dataset.

        Returns
        -------
        str
            A string representation of the dataset including its name, version,
            and basic statistics if data is loaded.
        """
        base_info = f"{self.info.name} (v{self.info.version})"

        return (
            f"{base_info}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
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

    print("Checking integrity of dataset")
    for sample in tqdm(ds):
        audio = sample['audio']
        st = sample['selection_table']
        audio_path = sample['audio_fp']
        if len(audio) < 10:
            print(f"too short {audio_path}")
        if np.any(np.isnan(audio)):
            print(f"nan present in {audio_path}")
        if np.all(audio == 0):
            print(f"clip is all zeros {audio_path}")
        st_end = st['Begin Time (s)'].max()
        audio_end = len(audio)/ds.sample_rate
        if st_end > audio_end:
            print(f"events happen after audio in {audio_path}")

    print("Done checking dataset")


