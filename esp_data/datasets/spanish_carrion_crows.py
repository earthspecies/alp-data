"""SpanishCarrionCrows dataset"""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator

import librosa
import pandas as pd

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, read_audio


@register_dataset
class SpanishCarrionCrows(Dataset):
    """SpanishCarrionCrows Dataset

    Description
    -----------
    This class makes the Spanish carrion crow biologger dataset available
    (2018, 2019, 2021). Each entry is an audio recording, sampling rate,
    year, territory, focal individual (individual wearing the biologger),
    plus a selection table derived from Voxaboxen inference. Each row of
    the selection table has `Annotation` = an annotation of the caller
    (focal adult non-focal adult, crow chick, or cuckoo chick), `Detection
    Prob` = detection probability, `Class Prob` = classification probability,
    start and stop time (in seconds within the file). In the paper, we
    filtered detections with `Detection Prob` >= 0.5, and assigned a known
    caller for `Class Prob` >= 0.5.

    This dataset does not contain several aspects of the full crow dataset,
    for example:
        - (Non-synchronized) UTC timestamps for each detection
          Note that audio files can skip, such that timestamps are not linearly
          related to sec in the file. Mostly should not be an issue.
        - Synchronized UTC timestamps across biologgers.
        - Annotations of file quality
          e.g., some biologgers prematurely detached from the bird and recorded
          background sounds only
        - Call types
    If any additional data are needed, please contact Maddie.

    There are currently no pre-computed sample rates.

    TODO: Add preprint link.

    """

    info = DatasetInfo(
        name="spanish-carrion-crows",
        owner="maddie",
        split_paths={
            "all": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/all__spanish-carrion-crows_info.csv",
            "2018_AW_Roja": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2018_AW_Roja__spanish-carrion-crows_info.csv",
            "2018_AW_Azul": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2018_AW_Azul__spanish-carrion-crows_info.csv",
            "2018_AY_Amarilla": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2018_AY_Amarilla__spanish-carrion-crows_info.csv",
            "2018_AY_Verde": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2018_AY_Verde__spanish-carrion-crows_info.csv",
            "2018_BC_Amarilla": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2018_BC_Amarilla__spanish-carrion-crows_info.csv",
            "2018_BC_Naranja": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2018_BC_Naranja__spanish-carrion-crows_info.csv",
            "2018_BC_Verde": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2018_BC_Verde__spanish-carrion-crows_info.csv",
            "2018_CT_Rosa": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2018_CT_Rosa__spanish-carrion-crows_info.csv",
            "2018_N1_Azul": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2018_N1_Azul__spanish-carrion-crows_info.csv",
            "2018_O_Roja": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2018_O_Roja__spanish-carrion-crows_info.csv",
            "2018_O_Verde": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2018_O_Verde__spanish-carrion-crows_info.csv",
            "2018_VAE_Amarilla": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2018_VAE_Amarilla__spanish-carrion-crows_info.csv",
            "2018_VAE_Roja": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2018_VAE_Roja__spanish-carrion-crows_info.csv",
            "2019_AL_Azul": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_AL_Azul__spanish-carrion-crows_info.csv",
            "2019_AL_Naranja": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_AL_Naranja__spanish-carrion-crows_info.csv",
            "2019_AL_Rosa": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_AL_Rosa__spanish-carrion-crows_info.csv",
            "2019_BA_Amarilla": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_BA_Amarilla__spanish-carrion-crows_info.csv",
            "2019_BA_Roja": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_BA_Roja__spanish-carrion-crows_info.csv",
            "2019_BD_Naranja": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_BD_Naranja__spanish-carrion-crows_info.csv",
            "2019_BD_Rosa": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_BD_Rosa__spanish-carrion-crows_info.csv",
            "2019_BPBO_Azul": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_BPBO_Azul__spanish-carrion-crows_info.csv",
            "2019_BUCK_Azul": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_BUCK_Azul__spanish-carrion-crows_info.csv",
            "2019_BUCK_Morado": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_BUCK_Morado__spanish-carrion-crows_info.csv",
            "2019_BUCK_Rosa": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_BUCK_Rosa__spanish-carrion-crows_info.csv",
            "2019_BV_Azul": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_BV_Azul__spanish-carrion-crows_info.csv",
            "2019_BV_Morado": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_BV_Morado__spanish-carrion-crows_info.csv",
            "2019_CA_Rosa": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_CA_Rosa__spanish-carrion-crows_info.csv",
            "2019_CV_Azul": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_CV_Azul__spanish-carrion-crows_info.csv",
            "2019_CV_Roja": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_CV_Roja__spanish-carrion-crows_info.csv",
            "2019_CW_Amarillo": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_CW_Amarillo__spanish-carrion-crows_info.csv",
            "2019_EB_Rojo": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_EB_Rojo__spanish-carrion-crows_info.csv",
            "2019_EB_Verde": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_EB_Verde__spanish-carrion-crows_info.csv",
            "2019_N7_Amarillo": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_N7_Amarillo__spanish-carrion-crows_info.csv",
            "2019_N7_Morado": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2019_N7_Morado__spanish-carrion-crows_info.csv",
            "2021_BPBO_Azul": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2021_BPBO_Azul__spanish-carrion-crows_info.csv",
            "2021_BPBO_Verde": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2021_BPBO_Verde__spanish-carrion-crows_info.csv",
            "2021_BQ_Naranja": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2021_BQ_Naranja__spanish-carrion-crows_info.csv",
            "2021_EB_Amarillo": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2021_EB_Amarillo__spanish-carrion-crows_info.csv",
            "2021_FA_Amarillo": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2021_FA_Amarillo__spanish-carrion-crows_info.csv",
            "2021_FA_Rojo": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2021_FA_Rojo__spanish-carrion-crows_info.csv",
            "2021_LL_Amarillo": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2021_LL_Amarillo__spanish-carrion-crows_info.csv",
            "2021_LL_Verde": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2021_LL_Verde__spanish-carrion-crows_info.csv",
            "2021_N4_Naranja": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2021_N4_Naranja__spanish-carrion-crows_info.csv",
            "2021_N4_Verde": "gs://esp-ml-datasets/spanish-carrion-crows/v0.1.0/detections/2021_N4_Verde__spanish-carrion-crows_info.csv",
        },
        version="0.1.0",
        description="Crow biologger audio with Voxaboxen detections",
        sources="University of Leon",
        license=(
            "For ESP internal, non-commerical use only. "
            "Should talk to collaborators about usage in work to be published."
        ),
    )

    _sample_rate_paths: dict[int, str] = {}
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
        self.annotation_columns = ["Annotation", "Detection Prob", "Class Prob", "RMS Amplitude"]
        self.unknown_label = "Unknown"
        self.sample_rate = sample_rate

        self.full_dataset_available_labels = [
            "focal",
            "not focal",
            "crowchicks",
            "cuckoo",
        ]  # for Annotation column

        self._load()

        if data_root is None:
            self.data_root = anypath("gs://")
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
        use_presampled = False
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            path_column = self._sample_rate_paths[self.sample_rate]
            if path_column in row and row[path_column] is not None and row[path_column] != "":
                audio_path = anypath(self.data_root) / row[path_column]
                use_presampled = True

        if not use_presampled:
            audio_path = anypath(self.data_root) / row[self._originals_path_column]

        audio, sr = read_audio(audio_path)
        # Should all be mono
        # audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

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
        row["sample_rate"] = sr
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
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["SpanishCarrionCrows", dict[str, Any]]:
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

    def get_available_labels(self, anno_column: str | None = "Annotation") -> list[str]:
        """
        Return all possible caller_id labels

        Returns
        ---------
        A list of all the available labels for anno_column
        """
        if self.split == "all" and anno_column == "Annotation":
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