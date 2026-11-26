import logging
from typing import Any, Iterator

from pydantic import Field

from alp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from alp_data.backends import BackendType
from alp_data.dataset import register_config
from alp_data.io import DATA_HOME, AnyPathT, anypath, read_video_at_frames
from alp_data.transforms import validate_video_table

logger = logging.getLogger("alp_data")

# Paths in the split tables are relative to the bucket root
_ROOT = DATA_HOME
_V010_ROOT = f"{_ROOT}/mammalps/v0.1.0"


@register_config
class MammAlpsConfig(DatasetConfig):
    """Configuration for the MammAlps dataset.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset. Must be "mammalps".
    split : str
        The split to load. One of `MammAlps.info.split_paths` keys, by default
        "benchmark-1-train".
    output_take_and_give : dict[str, str] | None
        A dictionary mapping the original column names to the new column names.
        It acts as a filter as well.
    data_root : str | AnyPathT | None
        The root directory for the dataset. Paths in the split table are
        relative to it. If None, uses the bucket root.
    frames_per_second : float | None
        Frame rate to sample videos at. If None, videos are read at their native
        frame rate. Must be > 0.
    backend : BackendType
        The backend to use ("pandas" or "polars"), by default "polars".
    streaming : bool
        Whether to use streaming mode, by default False. Not supported.
    """

    dataset_name: str = "mammalps"
    split: str = "benchmark-1-train"
    output_take_and_give: dict[str, str] | None = None
    data_root: str | AnyPathT | None = None
    frames_per_second: float | None = Field(default=None, gt=0)
    backend: BackendType = "polars"
    streaming: bool = False


@register_dataset
class MammAlps(Dataset):
    """MammAlps: multi-view camera-trap videos of wild mammals in the Swiss Alps.

    Description
    -----------
    MammAlps is a multimodal and multi-view dataset of wildlife behavior monitoring.
    Nine camera-traps were placed at three monitoring sites in the Swiss National Park,
    from which we curated over 14 hours of video with audio, 2D segmentation maps and
    8.5 hours of individual tracks densely labeled for species and behavior.

    Available Metadata Fields
    -------------------------

    **Video File Path:**
        - ``video_path``: Path to video clip (.mp4)

    **Audio File Paths:**
        - ``audio_path``: Path to original audio (.wav)

    **Segmentation Map File Paths:**
        - ``segmap_path``: Path to background segmentation maps for each frame in the clip (.npz)

    **Recording Metadata:**
        - ``start_s``, ``end_s``, ``duration_seconds``: time range from beginning to end of clip,
        ``start_s`` = 0 and ``end_s`` = ``duration_seconds`` for all clips in benchmark 1
        - ``activity``: high-level behavior label exhibited in clip
        - ``species``: species of individual animal shown in clip
        - ``actions``: semicolon-separated list of two actions displayed during clip; second action
        is "none" if only one action was displayed
        - ``num_frames``, ``average_fps``: total number of frames, frames per second
        - ``width``, ``height``: video width and height in pixels

    **Frame Sampling:**
        - ``start_frame``, ``end_frame``: frame range to decode, by default the whole clip
        - ``native_fps``: native frame rate of the clip
        - ``frame_stride``, ``sampled_fps``: stride between frames to decode and resulting frame
        rate. By default, stride is 1 and sampled fps is ``native_fps``; these get overwritten when
        ``frames_per_second`` is set.

    **Split:**
        - ``split``: one of "train", "val", or "test"

    Available Splits
    ----------------
    - ``benchmark-1-train``: Training set (benchmark 1 split from MammAlps paper)
    - ``benchmark-1-val``: Validation set (benchmark 1 split from MammAlps paper)
    - ``benchmark-1-test``: Test set (benchmark 1 split from MammAlps paper)
    - ``benchmark-1-all``: Complete dataset (train + val + test)

    """

    info = DatasetInfo(
        name="mammalps",
        owner="kai",
        split_paths={
            "benchmark-1-all": f"{_V010_ROOT}/benchmark_1/metadata/all.csv",
            "benchmark-1-train": f"{_V010_ROOT}/benchmark_1/metadata/train.csv",
            "benchmark-1-val": f"{_V010_ROOT}/benchmark_1/metadata/val.csv",
            "benchmark-1-test": f"{_V010_ROOT}/benchmark_1/metadata/test.csv",
        },
        version="0.1.0",
        description=(
            "Multi-view, multi-modal camera-trap recordings of wild mammals in the Swiss "
            "Alps, annotated with species, activity, and action labels."
        ),
        sources=[
            "https://zenodo.org/records/15040900",
            "https://github.com/eceo-epfl/MammAlps",
        ],
        license="CC BY-NC 4.0",
    )

    def __init__(
        self,
        split: str = "benchmark-1-train",
        output_take_and_give: dict[str, str] | None = None,
        data_root: str | AnyPathT | None = None,
        frames_per_second: float | None = None,
        backend: BackendType = "polars",
    ) -> None:
        """Initialize the MammAlps dataset.

        Parameters
        ----------
        split : str, optional
            The split to load, one of `info.split_paths` keys. By default
            "benchmark-1-train".
        output_take_and_give : dict[str, str] | None, optional
            A dictionary mapping the original column names to the new column names.
            It acts as a filter as well.
        data_root : str | AnyPathT | None, optional
            Dataset root.
        frames_per_second : float | None, optional
            Frame rate to sample videos at. Each video's `frame_stride` is set to
            `max(1, round(native_fps / frames_per_second))` and its `sampled_fps` column to
            the resulting rate `native_fps / frame_stride`. If None, videos are read at their
            native frame rate.
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars".

        Raises
        ------
        ValueError
            If `frames_per_second` is not a positive number, or the split's table
            fails `validate_video_table`.
        LookupError
            If `split` is not a key of `info.split_paths`.
        """
        if frames_per_second is not None and (
            isinstance(frames_per_second, bool)
            or not isinstance(frames_per_second, (int, float))
            or frames_per_second <= 0
        ):
            raise ValueError(f"frames_per_second must be a number > 0, got {frames_per_second!r}")
        if split not in self.info.split_paths:
            raise LookupError(f"Invalid split: {split}. Expected one of {self.available_splits}")
        super().__init__(output_take_and_give, backend=backend)
        self.split = split
        self.data_root = anypath(_ROOT if data_root is None else data_root)
        self.fps = frames_per_second
        self._data = None
        self._load()

    @property
    def columns(self) -> list[str]:
        """Return the columns of the dataset.

        Returns
        -------
        list[str]
            The split table's columns, including the clip columns.
        """
        return list(self._data.columns)

    @property
    def available_splits(self) -> list[str]:
        """Return the available splits of the dataset.

        Returns
        -------
        list[str]
            The keys of `info.split_paths`.
        """
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        """Read the split's table into the backend.

        If `frames_per_second` was set, `frame_stride` and `sampled_fps` are overwritten to
        sample each clip at that rate. A warning is logged for each native frame rate
        that `frames_per_second` does not divide, i.e. where the frame rate of the decoded
        video is off from `frames_per_second` by more than 1%.

        Raises
        ------
        ValueError
            If the loaded table fails `validate_video_table`.
        """  # noqa: DOC502 - raised by `validate_video_table`, not directly
        location = self.info.split_paths[self.split]
        # Read empty strings as nulls but keep literal values such as "None"
        data = self._backend_class.from_csv(
            location, keep_default_na=False, na_values=[""], null_values=[]
        )
        validate_video_table(data)
        # The tables default to native frame rate (`frame_stride` 1, `sampled_fps` = `native_fps`)
        if self.fps is not None:
            native_fps = [float(f) for f in data.unwrap["native_fps"]]
            strides = [max(1, round(f / self.fps)) for f in native_fps]
            effective = [f / s for f, s in zip(native_fps, strides, strict=True)]
            # One warning per native rate; rounding merges rates that differ only by
            # float noise in the table (e.g. 29.999999999999996 and 30.0)
            rounded = {
                (round(f, 6), s, round(e, 6))
                for f, s, e in zip(native_fps, strides, effective, strict=True)
                if abs(e - self.fps) > 1e-2 * self.fps
            }
            for f, s, e in sorted(rounded):
                logger.warning(
                    "frames_per_second=%s does not divide native_fps=%.6g; sampling at %.4g fps "
                    "(stride %d)",
                    self.fps,
                    f,
                    e,
                    s,
                )
            data = data.add_column("frame_stride", strides)
            data = data.add_column("sampled_fps", effective)
        self._data = data

    @classmethod
    def from_config(cls, dataset_config: MammAlpsConfig) -> tuple["MammAlps", dict[str, Any]]:
        """Create a Dataset instance from a configuration.

        Parameters
        ----------
        dataset_config : MammAlpsConfig
            Configuration for the dataset. Its `split` must be one of
            `info.split_paths` keys. A plain `DatasetConfig` (e.g. inside a
            concat or chain config) also works; `frames_per_second` is then read
            from its extra fields.

        Returns
        -------
        tuple[MammAlps, dict[str, Any]]
            A tuple containing the dataset instance and metadata.
            If the dataset_config contains transformations, they will be applied
            and the metadata will be returned as dict, otherwise an empty dict.

        Raises
        ------
        ValueError
            If `dataset_config.streaming` is True; streaming is not supported.
        """
        if dataset_config.streaming:
            raise ValueError("MammAlps does not support streaming.")
        ds = cls(
            split=dataset_config.split,
            output_take_and_give=dataset_config.output_take_and_give,
            data_root=dataset_config.data_root,
            frames_per_second=getattr(dataset_config, "frames_per_second", None),
            backend=dataset_config.backend,
        )

        if dataset_config.transformations:
            transform_metadata = ds.apply_transformations(dataset_config.transformations)
            return ds, transform_metadata

        return ds, {}

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Decode a row's clip and apply `output_take_and_give`.

        Parameters
        ----------
        row : dict[str, Any]
            A row of the split table, including the clip columns.

        Returns
        -------
        dict[str, Any]
            The row with the decoded clip under `video`, a `uint8` `np.ndarray` of
            shape `(num_frames, 3, height, width)`.
        """
        row["video"] = read_video_at_frames(
            self.data_root / row["video_path"],
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
            stride=int(row["frame_stride"]),
        )

        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}
        return row

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the current split.
        """
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
            A dictionary containing the sample, with the decoded clip under `video`,
            a `uint8` `np.ndarray` of shape `(num_frames, 3, height, width)`.

        Raises
        ------
        IndexError
            If `idx` is out of range.
        """
        if not -len(self) <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        ------
        dict[str, Any]
            Each sample in the dataset.
        """
        for row in self._data:
            yield self._process(row)

    def __str__(self) -> str:
        """Return a string representation of the dataset.

        Returns
        -------
        str
            A string representation of the dataset including its name, version,
            split, size and sources.
        """
        return (
            f"{self.info.name} (v{self.info.version}), split='{self.split}'\n"
            f"  Rows: {len(self)}\n"
            f"  Description: {self.info.description}\n"
            f"  Sources: {', '.join(self.info.sources)}\n"
            f"  License: {self.info.license}\n"
            f"  Available splits: {', '.join(self.available_splits)}"
        )
