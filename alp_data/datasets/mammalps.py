from typing import Any, Iterator

from alp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from alp_data.backends import BackendType
from alp_data.io import AnyPathT, read_video_at_frames

_ROOT = "gs://esp-data-ingestion/"


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
            "benchmark-1-all": f"{_ROOT}/benchmark_1/metadata/all.csv",
            "benchmark-1-train": f"{_ROOT}/benchmark_1/metadata/train.csv",
            "benchmark-1-val": f"{_ROOT}/benchmark_1/metadata/val.csv",
            "benchmark-1-test": f"{_ROOT}/benchmark_1/metadata/test.csv",
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
            Dataset root path.
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars".

        Notes
        -----
        Loading raises `LookupError` if `split` is not a key of `info.split_paths`.
        """
        super().__init__(output_take_and_give, backend=backend)
        self.split = split
        self.data_root = None if data_root is None else str(data_root).rstrip("/")
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
        """Read the split's table into the backend and add the clip columns.

        Raises
        ------
        LookupError
            If `self.split` is not a key of `info.split_paths`.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {self.available_splits}"
            )
        location = self.info.split_paths[self.split]
        # Read empty strings as nulls but keep literal values such as "None"
        data = self._backend_class.from_csv(
            location, keep_default_na=False, na_values=[""], null_values=[]
        )
        table = data.unwrap

        # Add columns to prepare for any video sampling transforms
        data = data.add_column("start_frame", 0)
        data = data.add_column("end_frame", list(table["num_frames"]))
        data = data.add_column("frame_stride", 1)
        data = data.add_column("video_fps", list(table["average_fps"]))
        data = data.add_column("fps", list(table["average_fps"]))
        self._data = data

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["MammAlps", dict[str, Any]]:
        """Create a Dataset instance from a configuration.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration for the dataset. Its `split` must be one of
            `info.split_paths` keys.

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
            backend=dataset_config.backend,
        )

        if dataset_config.transformations:
            transform_metadata = ds.apply_transformations(dataset_config.transformations)
            return ds, transform_metadata

        return ds, {}

    def _video_path(self, row: dict[str, Any]) -> str:
        path = row["video_path"]
        if self.data_root is not None and path.startswith(_ROOT):
            path = self.data_root + path[len(_ROOT) :]
        return path

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Decode a row's clip and apply `output_take_and_give`.

        Parameters
        ----------
        row : dict[str, Any]
            A row of the split table, including the clip columns.

        Returns
        -------
        dict[str, Any]
            The row with the decoded clip under `video`.
        """
        row["video"] = read_video_at_frames(
            self._video_path(row),
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
            A dictionary containing the sample, with the decoded clip under `video`.

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
