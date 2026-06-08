"""Spanish Carrion Crows adult focal vocalizations dataset"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, Iterator

import librosa
import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_config, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio


@register_config
class SpanishCarrionCrowsVoxConfig(DatasetConfig):
    """Configuration for the SpanishCarrionCrowsVox dataset.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset. Must be "spanish-carrion-crows-vox".
    split : str | int
        The split to load: "all" or one of SpanishCarrionCrowsVox.available_splits.
    output_take_and_give : dict[str, str] | None
        A dictionary mapping the original column names to the new column names.
        It acts as a filter as well.
    sample_rate : int | None
        The sample rate to which audio files should be resampled.
    data_root : str | AnyPathT | None
        The root directory for the dataset. This is optionally appended to the
        path item of a sample in the dataset.
    backend : BackendType
        The backend to use ("pandas" or "polars"), by default "polars"
    streaming : bool
        Whether to use streaming mode, by default False
    denoised : bool
        If True, load the denoised (MixIT) audio instead of the noisy focal
        recording. Defaults to False.
    fallback_to_noisy : bool
        Only used when denoised=True. If True, fall back to the noisy audio
        when denoising failed rather than raising ValueError; the returned
        item will have denoised_success=False. Defaults to False.
    padding_sec : float
        Seconds of context to include before the vocalization start and after
        its end. Clamped to file boundaries. Defaults to 0.0.
    """

    dataset_name: str = "spanish-carrion-crows-vox"
    split: str | int = "all"
    version: str | None = None
    output_take_and_give: dict[str, str] | None = None
    sample_rate: int | None = 16000
    data_root: str | AnyPathT | None = None
    backend: BackendType = "polars"
    streaming: bool = False
    denoised: bool = False
    fallback_to_noisy: bool = False
    padding_sec: float = 0.0


@register_dataset
class SpanishCarrionCrowsVox(Dataset):
    """Spanish Carrion Crows individual vocalizations dataset.

    Description
    -----------
    Each entry is a single vocalization from a focal Spanish carrion crow,
    clipped from a longer biologger recording. Audio can be returned as-is
    (noisy) or denoised via MixIT source separation. Each entry is accompanied
    by call type, focal individual ID, and synchronized timestamps.

    Vocalization splits are contiguous segments with more than one focal
    individual, from the same year and territory, making vocalizations.

    See decode-library/runs/Carrion_Crow/matching_biologgers.ipynb for the
    original of conversational_preprocessed.csv. Thanks to Emmanuel Fernandez.

    Each item contains:
    - audio: float32 numpy array, clipped to the vocalization window
    - sample_rate: int
    - call_type: derived.superpile_nickname
    - focal_individual: individual wearing the biologger
    - timestamp_start: UTC timestamp of vocalization start
    - timestamp_end: UTC timestamp of vocalization end
    - concurrent_behavior_category: dict[str, str]
      - each key is an individual, value gives behavior of that individual at timestamp_start
      - possible values:
        - visit_{unknown, repeated, alternated}:
          - Video: whether the bird is at the nest or not
          - Audio/Accelerometer: not incorporated
          - repeated: visit before this one was the same individual
          - alternated: visit before this one was a different individual
          - unknown: we do not know who made the previous visit
        - flying_arriving_{short: < 10 s, long: > 10 s}
        - flying_leaving_{short, long}
        - flying_other_{short,long}
          - Video: whether the activity after/before this flight was a nest visit
          - Audio: bird is flying
          - Accelerometer: not incorporated
          - arriving/leaving: bird is flying to/from the nest, respectively
          - other: bird is flying, but not to or from the nest
        - flying_unknown
          - Video: unavailable, so we do not know if flight is to/from the nest
          - Audio: bird is flying
          - Accelerometer: not incorporated
          - bird is flying, but we do not know where
        - high_activity / low_activity:
          - Video: bird is not at the nest
          - Audio: bird is not flying
          - Accelerometer: high ODBA / low ODBA
          - interpretation: individual is in high / low activity state,
                            away from nest and not flying
        - not_flying:
          - Video: unavailable, so we do not know whether bird is at nest or not
          - Audio: bird is not flying
          - Accelerometer: not incorporated
          - interpretation: bird is not flying, but we do not know where they
                            are nor whether they are in high/low activity state
        - unknown: nothing is known about this bird's behavior
    - concurrent_is_visit: dict[str, bool | None]
      - each key is an individual
      - value gives whether the individual is at the nest or not, at the time of the vocalization
      - if None, the value is unknown (e.g., we didn't have annotated
        nest camera video, at that timepoint)
    - concurrent_is_flying: dict[str, bool | None]
      - each key is an individual
      - value gives whether the individual is flying or not, at the time of the vocalization
      - if None, the value is unknown (e.g., we didn't have biologger
        for that individual, at that timepoint)
    - has_matching_vox_on_logger: dict[str, bool]
      - each key is an individual
      - True if this vocalization was detected on individual's biologger
      - always True if key == focal_individual
      - may be True or False, if key != focal_individual. if True, means this vocalization
        was non-focal on individual's biologger
    - sec_until_next_E4: dict[str, (float, float) | None]
      - each key is an individual
      - first value gives time to next focal vocalization labeled E4, where caller = individual
      - second value gives number of unobserved seconds between current vocalization and next E4
    - overlap_window_id: split index
    - overlap_individuals: individuals which have biologgers defined in this split index

    When denoised=True, also contains:
    - denoised_success: whether MixIT denoising succeeded for this row
      Rows where denoising failed will raise ValueError at access time;
      filter by derived.denoised_focal.success == True before iterating.

    References
    ----------
    https://doi.org/10.64898/2026.04.02.715916

    Should talk to collaborators (Daniela Canestrari and Vittorio Baglione)
    about usage in work to be published.

    """

    info = DatasetInfo(
        name="spanish_carrion_crows_vox",
        owner="maddie",
        split_paths={
            # TODO: update path once finalized in GCS
            "all": "gs://esp-ml-datasets/spanish-carrion-crows-vox/conversational_preprocessed_with_behavior.csv",
        },
        version="0.1.0",
        description="Spanish carrion crow adult focal vocalizations with call type and timestamps",
        sources="University of Leon",
        license="private",
    )

    def __init__(
        self,
        split: str | int = "all",
        output_take_and_give: Dict[str, str] | None = None,
        sample_rate: int | None = 16000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
        denoised: bool = False,
        fallback_to_noisy: bool = False,
        padding_sec: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        split : str | int
            "all" to load every vocalization, or an overlap_window_id integer to
            load only vocalizations that co-occur in that context window.
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys (filters columns as well).
        sample_rate : int | None
            If set, audio is resampled to this rate. Defaults to 16000.
        data_root : str | AnyPathT | None
            Root directory prepended to each audio path in the CSV. Defaults to
            "gs://" (audio paths in the CSV start with the bucket name).
        backend : BackendType
            Backend to use ("pandas" or "polars"). Defaults to "polars".
        streaming : bool
            Whether to use streaming mode. Defaults to False.
        denoised : bool
            If True, load the denoised (MixIT) audio instead of the noisy focal
            recording. Defaults to False.
        fallback_to_noisy : bool
            Only used when denoised=True. If True, fall back to the noisy audio
            when denoising failed rather than raising ValueError; the returned
            item will have denoised_success=False. Defaults to False.
        padding_sec : float
            Seconds of context to include before the vocalization start and after
            its end. Clamped to file boundaries. Defaults to 0.0.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self.sample_rate = sample_rate
        self.denoised = denoised
        self.fallback_to_noisy = fallback_to_noisy
        self.padding_sec = padding_sec

        self.data_root = anypath(data_root) if data_root is not None else anypath("gs://")

        self._load()

    @property
    def columns(self) -> list[str]:
        return self._data.columns

    @property
    def available_splits(self) -> list[str]:
        return [
            "all",
            1,
            2,
            4,
            6,
            10,
            12,
            13,
            14,
            15,
            17,
            19,
            21,
            22,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            40,
            41,
            42,
            43,
            44,
            45,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            77,
            78,
            83,
            85,
            87,
            88,
            89,
            90,
            91,
            93,
            95,
            96,
            97,
            98,
            99,
            100,
            102,
            103,
            150,
            154,
            155,
            275,
            286,
            287,
            288,
        ]

    def _load(self) -> None:
        location = self.info.split_paths["all"]
        self._data = self._backend_class.from_csv(
            location,
            streaming=self._streaming,
            sep="\t",
            separator="\t",
            # TODO I don't know if we want this?
            keep_default_na=False,
            na_values=[""],
        )
        if self.split != "all":
            self._data = self._data.filter_isin("overlap_window_id", [float(self.split)])
            if not self._streaming and len(self._data) == 0:
                raise LookupError(
                    f"No rows found for overlap_window_id={self.split}. "
                    "Pass split='all' or a valid overlap_window_id value."
                )

    def __len__(self) -> int:
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. Iterate over the dataset instead."
            )
        return len(self._data)

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        start_sec = max(0.0, float(row["derived.centered_focal.long.start_sec"]) - self.padding_sec)
        end_sec = float(row["derived.centered_focal.long.end_sec"]) + self.padding_sec

        if self.denoised:
            denoised_success = row["derived.denoised_focal.success"]
            # if isinstance(denoised_success, str):
            #     # TODO check underlying dataframe
            #     denoised_success = denoised_success == "True"
            fp = row.get("derived.denoised_focal.fp") or ""
            if denoised_success and fp:
                audio_path = self.data_root / fp
            elif self.fallback_to_noisy:
                audio_path = self.data_root / row["derived.centered_focal.long.fp"]
            else:
                raise ValueError(
                    "Denoised audio not available for this row "
                    f"(derived.denoised_focal.success={row['derived.denoised_focal.success']}). "
                    "Set fallback_to_noisy=True to use noisy audio when denoising failed, "
                    "or filter rows by derived.denoised_focal.success == True before iterating."
                )
        else:
            audio_path = self.data_root / row["derived.centered_focal.long.fp"]

        audio, sample_rate = read_audio(audio_path, start_time=start_sec, end_time=end_sec)
        audio = audio_stereo_to_mono(audio, mono_method="keep_first").astype(np.float32)

        if self.sample_rate is not None and sample_rate != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sample_rate,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
            sample_rate = self.sample_rate

        out: dict[str, Any] = {
            "audio": audio,
            "sample_rate": sample_rate,
            "call_type": row["derived.superpile_nickname"],
            "focal_individual": row["focal_individual"],
            "timestamp_start": row["start_timestamp_flex"],
            "timestamp_end": row["end_timestamp_flex"],
            "concurrent_behavior_category": json.loads(row["concurrent_behavior_summary"]),
            "concurrent_is_visit": json.loads(row["concurrent_is_visit"]),
            "concurrent_is_flying": json.loads(row["concurrent_is_flying"]),
            "has_matching_vox_on_logger": json.loads(row["matching_loggers"]),
            "sec_until_next_E4": json.loads(row["time_until_next_E4"]),
            "overlap_window_id": int(row["overlap_window_id"]),
            "overlap_individuals": ast.literal_eval(row["overlap_window_individuals"]),
        }

        if self.denoised:
            out["denoised_success"] = denoised_success

        if self.output_take_and_give:
            return {new_key: out[old_key] for old_key, new_key in self.output_take_and_give.items()}

        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["SpanishCarrionCrowsVox", dict[str, Any]]:
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
