"""WABAD dataset"""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator

import librosa
import numpy as np
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

SPECIES_INFO_PATH = "gs://esp-ml-datasets/wabad/v0.1.0/raw/gbif_labels.csv"


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

    Pre-resampled Audio
    -------------------
    Pre-resampled audio is available at 16 kHz and 32 kHz. When
    ``sample_rate`` matches one of these rates, the pre-resampled files are
    loaded directly (no on-the-fly resampling). For any other target rate,
    audio is resampled on-the-fly using librosa's ``kaiser_best`` method.

    References
    ----------
    https://zenodo.org/records/15629388
    https://www.researchgate.net/publication/387711208_WABAD_A_World_Annotated_Bird_Acoustic_Dataset_for_Passive_Acoustic_Monitoring

    """

    info = DatasetInfo(
        name="wabad",
        owner="benjamin",
        split_paths={
            "all": "gs://esp-ml-datasets/wabad/v0.1.0/raw/all_info_gbif_v2.csv",
            "CAT": "gs://esp-ml-datasets/wabad/v0.1.0/raw/CAT_info_gbif_v2.csv",
            "POZO": "gs://esp-ml-datasets/wabad/v0.1.0/raw/POZO_info_gbif_v2.csv",
            "BRE": "gs://esp-ml-datasets/wabad/v0.1.0/raw/BRE_info_gbif_v2.csv",
            "EFFOR": "gs://esp-ml-datasets/wabad/v0.1.0/raw/EFFOR_info_gbif_v2.csv",
            "MONTEB": "gs://esp-ml-datasets/wabad/v0.1.0/raw/MONTEB_info_gbif_v2.csv",
            "CB": "gs://esp-ml-datasets/wabad/v0.1.0/raw/CB_info_gbif_v2.csv",
            "FEU": "gs://esp-ml-datasets/wabad/v0.1.0/raw/FEU_info_gbif_v2.csv",
            "BIAL": "gs://esp-ml-datasets/wabad/v0.1.0/raw/BIAL_info_gbif_v2.csv",
            "SPMCO": "gs://esp-ml-datasets/wabad/v0.1.0/raw/SPMCO_info_gbif_v2.csv",
            "OIO": "gs://esp-ml-datasets/wabad/v0.1.0/raw/OIO_info_gbif_v2.csv",
            "OESF": "gs://esp-ml-datasets/wabad/v0.1.0/raw/OESF_info_gbif_v2.csv",
            "QR": "gs://esp-ml-datasets/wabad/v0.1.0/raw/QR_info_gbif_v2.csv",
            "HAG": "gs://esp-ml-datasets/wabad/v0.1.0/raw/HAG_info_gbif_v2.csv",
            "VIL": "gs://esp-ml-datasets/wabad/v0.1.0/raw/VIL_info_gbif_v2.csv",
            "RFP": "gs://esp-ml-datasets/wabad/v0.1.0/raw/RFP_info_gbif_v2.csv",
            "HAK": "gs://esp-ml-datasets/wabad/v0.1.0/raw/HAK_info_gbif_v2.csv",
            "SLOB": "gs://esp-ml-datasets/wabad/v0.1.0/raw/SLOB_info_gbif_v2.csv",
            "BERB": "gs://esp-ml-datasets/wabad/v0.1.0/raw/BERB_info_gbif_v2.csv",
            "COU": "gs://esp-ml-datasets/wabad/v0.1.0/raw/COU_info_gbif_v2.csv",
            "OLIV": "gs://esp-ml-datasets/wabad/v0.1.0/raw/OLIV_info_gbif_v2.csv",
            "EVROS": "gs://esp-ml-datasets/wabad/v0.1.0/raw/EVROS_info_gbif_v2.csv",
            "FNCA": "gs://esp-ml-datasets/wabad/v0.1.0/raw/FNCA_info_gbif_v2.csv",
            "RGU": "gs://esp-ml-datasets/wabad/v0.1.0/raw/RGU_info_gbif_v2.csv",
            "CRUZ": "gs://esp-ml-datasets/wabad/v0.1.0/raw/CRUZ_info_gbif_v2.csv",
            "JUNCA": "gs://esp-ml-datasets/wabad/v0.1.0/raw/JUNCA_info_gbif_v2.csv",
            "PINA": "gs://esp-ml-datasets/wabad/v0.1.0/raw/PINA_info_gbif_v2.csv",
            "GTLU": "gs://esp-ml-datasets/wabad/v0.1.0/raw/GTLU_info_gbif_v2.csv",
            "MAPIMI": "gs://esp-ml-datasets/wabad/v0.1.0/raw/MAPIMI_info_gbif_v2.csv",
            "SAL": "gs://esp-ml-datasets/wabad/v0.1.0/raw/SAL_info_gbif_v2.csv",
            "ARD": "gs://esp-ml-datasets/wabad/v0.1.0/raw/ARD_info_gbif_v2.csv",
            "MARTI": "gs://esp-ml-datasets/wabad/v0.1.0/raw/MARTI_info_gbif_v2.csv",
            "DYOM": "gs://esp-ml-datasets/wabad/v0.1.0/raw/DYOM_info_gbif_v2.csv",
            "VER": "gs://esp-ml-datasets/wabad/v0.1.0/raw/VER_info_gbif_v2.csv",
            "SCHG": "gs://esp-ml-datasets/wabad/v0.1.0/raw/SCHG_info_gbif_v2.csv",
            "GLEN": "gs://esp-ml-datasets/wabad/v0.1.0/raw/GLEN_info_gbif_v2.csv",
            "HONDO": "gs://esp-ml-datasets/wabad/v0.1.0/raw/HONDO_info_gbif_v2.csv",
            "NL": "gs://esp-ml-datasets/wabad/v0.1.0/raw/NL_info_gbif_v2.csv",
            "BRCAS": "gs://esp-ml-datasets/wabad/v0.1.0/raw/BRCAS_info_gbif_v2.csv",
            "NAV": "gs://esp-ml-datasets/wabad/v0.1.0/raw/NAV_info_gbif_v2.csv",
            "KAR": "gs://esp-ml-datasets/wabad/v0.1.0/raw/KAR_info_gbif_v2.csv",
            "BUR": "gs://esp-ml-datasets/wabad/v0.1.0/raw/BUR_info_gbif_v2.csv",
            "KIB": "gs://esp-ml-datasets/wabad/v0.1.0/raw/KIB_info_gbif_v2.csv",
            "SCHF": "gs://esp-ml-datasets/wabad/v0.1.0/raw/SCHF_info_gbif_v2.csv",
            "TAM": "gs://esp-ml-datasets/wabad/v0.1.0/raw/TAM_info_gbif_v2.csv",
            "HUAP": "gs://esp-ml-datasets/wabad/v0.1.0/raw/HUAP_info_gbif_v2.csv",
            "DONG": "gs://esp-ml-datasets/wabad/v0.1.0/raw/DONG_info_gbif_v2.csv",
            "CLH": "gs://esp-ml-datasets/wabad/v0.1.0/raw/CLH_info_gbif_v2.csv",
            "HAR": "gs://esp-ml-datasets/wabad/v0.1.0/raw/HAR_info_gbif_v2.csv",
            "BOLIN": "gs://esp-ml-datasets/wabad/v0.1.0/raw/BOLIN_info_gbif_v2.csv",
            "SITH": "gs://esp-ml-datasets/wabad/v0.1.0/raw/SITH_info_gbif_v2.csv",
            "RBA": "gs://esp-ml-datasets/wabad/v0.1.0/raw/RBA_info_gbif_v2.csv",
            "MOPU": "gs://esp-ml-datasets/wabad/v0.1.0/raw/MOPU_info_gbif_v2.csv",
            "CRAT": "gs://esp-ml-datasets/wabad/v0.1.0/raw/CRAT_info_gbif_v2.csv",
            "PGF": "gs://esp-ml-datasets/wabad/v0.1.0/raw/PGF_info_gbif_v2.csv",
            "PUUL": "gs://esp-ml-datasets/wabad/v0.1.0/raw/PUUL_info_gbif_v2.csv",
            "MILLAN": "gs://esp-ml-datasets/wabad/v0.1.0/raw/MILLAN_info_gbif_v2.csv",
            "BMT": "gs://esp-ml-datasets/wabad/v0.1.0/raw/BMT_info_gbif_v2.csv",
            "SD": "gs://esp-ml-datasets/wabad/v0.1.0/raw/SD_info_gbif_v2.csv",
            "UNI": "gs://esp-ml-datasets/wabad/v0.1.0/raw/UNI_info_gbif_v2.csv",
            "SBN": "gs://esp-ml-datasets/wabad/v0.1.0/raw/SBN_info_gbif_v2.csv",
            "DUNAS": "gs://esp-ml-datasets/wabad/v0.1.0/raw/DUNAS_info_gbif_v2.csv",
            "PETI": "gs://esp-ml-datasets/wabad/v0.1.0/raw/PETI_info_gbif_v2.csv",
            "LIM": "gs://esp-ml-datasets/wabad/v0.1.0/raw/LIM_info_gbif_v2.csv",
            "BAM": "gs://esp-ml-datasets/wabad/v0.1.0/raw/BAM_info_gbif_v2.csv",
            "DEVA": "gs://esp-ml-datasets/wabad/v0.1.0/raw/DEVA_info_gbif_v2.csv",
            "ROTOK": "gs://esp-ml-datasets/wabad/v0.1.0/raw/ROTOK_info_gbif_v2.csv",
            "CARI": "gs://esp-ml-datasets/wabad/v0.1.0/raw/CARI_info_gbif_v2.csv",
            "PITI": "gs://esp-ml-datasets/wabad/v0.1.0/raw/PITI_info_gbif_v2.csv",
            "RME": "gs://esp-ml-datasets/wabad/v0.1.0/raw/RME_info_gbif_v2.csv",
            "MABI": "gs://esp-ml-datasets/wabad/v0.1.0/raw/MABI_info_gbif_v2.csv",
            "EMP": "gs://esp-ml-datasets/wabad/v0.1.0/raw/EMP_info_gbif_v2.csv",
            "EFFOU": "gs://esp-ml-datasets/wabad/v0.1.0/raw/EFFOU_info_gbif_v2.csv",
        },
        version="0.1.0",
        description="[MISSING]",
        sources="zenodo.org",
        license="CC-BY-4.0",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _originals_path_column = "audio_fp"

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "pandas",
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
            If set, audio is resampled to this rate.
        data_root : str | AnyPathT | None
            Optional root directory to prepend to each row['audio_fp'].
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars"
        streaming : bool, optional
            Whether to use streaming mode, by default False
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self.annotation_columns = ["Species"]
        self.unknown_label = "Unknown"
        self.sample_rate = sample_rate

        self.full_dataset_available_labels = None  # placeholder for labels if split == all

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
        """Return pre-resampled sample rates whose path columns exist in the data."""
        return [sr for sr, col in self._sample_rate_paths.items() if col in self._data.columns]

    def _load(self) -> None:
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        location = self.info.split_paths[self.split]
        self._data = self._backend_class.from_csv(
            location,
            streaming=self._streaming,
            keep_default_na=False,
            na_values=[""],
        )

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
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode.Iterate over the dataset instead."
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

        st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
        audio_dur = len(audio) / float(sr)
        st = st[st["Begin Time (s)"] < audio_dur].copy()

        row["audio"] = audio
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
            A dictionary containing the audio data, text label, label, and path.
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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["WABAD", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary containing dataset parameters.

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
            backend=cfg["backend"],
            streaming=cfg["streaming"],
        )

        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta

        return ds, {}

    def get_available_labels(self, anno_column: str | None = "Species") -> list[str]:
        """
        Return all possible species labels

        Returns
        ---------
        A list of all the available labels for anno_column
        """
        if self.split == "all":
            if self.full_dataset_available_labels is None:
                self.full_dataset_available_labels = pd.read_csv(SPECIES_INFO_PATH)[
                    anno_column
                ].to_list()
            return self.full_dataset_available_labels
        else:
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
