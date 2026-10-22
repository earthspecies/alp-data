import logging
import pandas as pd

from alp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from alp_data.backends import BackendType
from alp_data.io import read_video

logger = logging.getLogger("alp_data")

_ROOT = "gs://esp-data-ingestion/mammalps/v0.1.1"


@register_dataset
class MammAlps(Dataset):
    """
    todo: some description here
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
        version="0.1.1",
        description=(
            "Multi-view, multi-modal camera-trap recordings of wild mammals in the Swiss "
            "Alps, annotated with species, activity, and action labels."
        ),
        sources="zenodo.org",
        license="CC BY-NC 4.0",
    )

    def __init__(
        self,
        split: str = "benchmark-1-train",
        output_take_and_give: dict[str, str] | None = None,
        fps: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "pandas",
    ) -> None:
        """Initialize the BEANS dataset.

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
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars"
        streaming : bool, optional
            Whether to use streaming mode, by default False
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data: pd.DataFrame = None
        self._load()  # Load the dataset (fills self._data)
        self.sample_rate = sample_rate
        self.data_root = data_root

        if data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent
        else:
            self.data_root = data_root

    @property
    def columns(self) -> list[str]:
        """Return the columns of the dataset."""
        return list(self._data.columns)

    @property
    def available_splits(self) -> list[str]:
        """Return the available splits of the dataset."""
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        """Read the split's CSV into the backend and, on video splits, add `video_fps`.

        Raises
        ------
        LookupError
            If `self.split` is not a key of `info.split_paths`.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        location = self.info.split_paths[self.split]
        # Read CSV content
        self._data = self._backend_class.from_csv(
            location, keep_default_na=False, na_values=[""], null_values=[]
        )  # This setting avoids setting 'None' to a pd.NA type
        # Previously there was this line as well:
        # self._data = data.add_column("video_fps", float(self._original_fps))
        # but let's just manually put video_fps as a column in the mammalps csvs
        # if it's not there already?

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["MammAlps", dict[str, Any]]:
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
            fps=cfg["fps"],
            backend=cfg["backend"],
        )

        if dataset_config.transformations:
            transform_metadata = ds.apply_transformations(dataset_config.transformations)
            return ds, transform_metadata

        return ds, {}

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        video_path = row["video_path"]

        # If frame indices are available, read by frame
        if "clip_start_idx" in row:
            # TODO: call read_video_at_indices
            video, fps = [], 0
            pass
        else:  # Otherwise, read by time, or read the whole video
            start_time = row.get("clip_start", None)
            end_time = row.get("clip_end", None)
            fps = row.get("fps", None)  # I think the fps here should be requested fps? confirm?
            # Read the video clip
            video, fps = read_video(video_path, start_time, end_time, fps)

        row["video"] = video

        if self.output_take_and_give:
            item = {}
            for key, value in self.output_take_and_give.items():
                item[value] = row[key]
        else:
            item = row

        return item

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
        if self._streaming:
            raise NotImplementedError("Length is not available in streaming mode.")
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
        """
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        -------
        Dict[str, Any]
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
            and basic statistics if data is loaded.
        """
        base_info = f"{self.info.name} (v{self.info.version}), split: {self.split}"

        return (
            f"{base_info}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
