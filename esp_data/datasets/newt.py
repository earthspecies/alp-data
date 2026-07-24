"""NeWT — Natural World Tasks binary visual benchmark (CVPR 2021).

NeWT (Van Horn et al., 2021, "Benchmarking Representation Learning for
Natural World Image Collections"; repo ``visipedia/newt``) is a benchmark
of 164 **binary** visual-classification tasks over natural-world imagery,
spanning clusters such as appearance (age / health / attribute / species),
behavior, context, gestalt, and counting. It is the "beyond species"
benchmark used by BioCLIP 2 (arXiv:2505.23883): rather than naming a
species, each task asks a single yes/no question about an organism (is it a
juvenile? is it sick? is it foraging?).

esp-data exposes it as an **image-only** dataset. Each row is one image
belonging to one binary task, carrying the task metadata and the 0/1 label
so an evaluator can group by ``task`` and score per-task / cluster / overall
accuracy.

The build script (``scripts/data_preprocessing_scripts/newt/build_newt.py``)
downloads the public NeWT tarball, builds manifest CSVs with absolute
``gs://`` image paths, and uploads images plus manifests to
``gs://esp-data-ingestion/newt/v0.1.0/``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, read_image

_GCS_ROOT = "gs://esp-data-ingestion/newt/v0.1.0"


@register_dataset
class NeWT(Dataset):
    """NeWT — Natural World Tasks binary visual benchmark.

    Description
    -----------
    164 binary visual-classification tasks over natural-world imagery. Each
    row is one image belonging to one task; the ``task`` / ``task_cluster``
    / ``task_subcluster`` columns identify the task and the ``label``
    column is the binary ground truth. Image-only; decoded lazily.

    Columns
    -------
    asset_id : str
        Source NeWT image id (``<id>.jpg``).
    modality : str
        Always ``"image"`` (present for parity with other esp-data
        vision datasets).
    label : int
        Binary ground-truth label for the row's task (0 / 1).
    text_label : str
        The task's positive-concept text (source ``text_label``).
    task : str
        Task identifier.
    task_cluster : str
        High-level cluster (``appearance`` / ``behavior`` / ``context`` /
        ``gestalt`` / ``counting``).
    task_subcluster : str
        Sub-cluster within the cluster (e.g. ``age`` / ``health`` /
        ``species`` for ``appearance``); may be empty.
    split : str
        Source split label (``train`` / ``test``).
    image_path : str
        Absolute ``gs://`` path to the JPEG.

    Splits
    ------
    - ``all`` : every image across all tasks.
    - ``train`` / ``test`` : the official NeWT per-task train/test split.

    Loader behaviour
    ----------------
    - ``image`` → reads the JPEG into ``image`` (HWC uint8), optionally
      resized to ``(image_size, image_size)``.

    References
    ----------
    - Van Horn et al. (2021) "Benchmarking Representation Learning for
      Natural World Image Collections", CVPR. Repo ``visipedia/newt``.

    License: MIT (per the ``visipedia/newt`` repository); staged on
    internal GCS for ESP research.
    """

    info = DatasetInfo(
        name="newt",
        owner="david",
        split_paths={
            "all": f"{_GCS_ROOT}/newt_all.csv",
            "train": f"{_GCS_ROOT}/newt_train.csv",
            "test": f"{_GCS_ROOT}/newt_test.csv",
        },
        version="0.1.0",
        description=(
            "NeWT (Natural World Tasks): 164 binary visual-classification "
            "tasks (appearance / behavior / context / gestalt / counting) "
            "over natural-world imagery. Image-only 'beyond species' "
            "benchmark used by BioCLIP 2."
        ),
        sources=("Van Horn et al. (2021) CVPR; repo visipedia/newt"),
        license="MIT (visipedia/newt); internal ESP research staging",
    )

    _mixup_group = "animal"

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
        image_size: int | None = None,
    ) -> None:
        """Initialise the NeWT dataset.

        Parameters
        ----------
        split : str
            Split to load (key in :attr:`info.split_paths`).
        output_take_and_give : dict[str, str] | None
            Optional column rename / selection mapping.
        data_root : str | AnyPathT | None
            Unused — path columns hold absolute ``gs://`` URIs. Accepted
            for API parity with other datasets.
        backend : BackendType
            ``"polars"`` or ``"pandas"``.
        streaming : bool
            Whether to use streaming mode.
        image_size : int | None
            If set, images are resized to ``(image_size, image_size)``.
            Defaults to None (native size).
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.image_size = image_size
        self._data = None
        self.data_root = anypath(data_root) if data_root else None
        self._load()

    def _load(self) -> None:
        """Load the split CSV.

        Raises
        ------
        LookupError
            If the split is not valid.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        self._data = self._backend_class.from_csv(
            self.info.split_paths[self.split],
            streaming=self._streaming,
            keep_default_na=False,
            na_values=[""],
        )

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    def _load_image(self, row: dict[str, Any]) -> dict[str, Any]:
        """Populate ``image`` (HWC uint8) for an image row.

        Returns
        -------
        dict[str, Any]
            The row with the ``image`` field populated.
        """
        image = read_image(str(row["image_path"]))
        if self.image_size is not None:
            from PIL import Image

            pil = Image.fromarray(image).resize((self.image_size, self.image_size), Image.BILINEAR)
            image = np.asarray(pil)
        row["image"] = image
        return row

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Load + return one image asset.

        Returns
        -------
        dict[str, Any]
            The row with the ``image`` field populated.
        """
        row = self._load_image(row)
        row["mixup_group"] = self._mixup_group
        if self.output_take_and_give:
            return {new: row[old] for old, new in self.output_take_and_give.items()}
        return row

    def __len__(self) -> int:
        """Return the number of images in the split.

        Returns
        -------
        int
            Number of images in the current split.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        NotImplementedError
            If the dataset is in streaming mode.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError("Length is not available in streaming mode.")
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single processed image.

        Returns
        -------
        dict[str, Any]
            The processed row (image + task metadata).
        """
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over processed images.

        Yields
        ------
        dict[str, Any]
            Each processed asset.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple[NeWT, dict[str, Any]]:
        """Create a NeWT instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Dataset configuration.

        Returns
        -------
        tuple[NeWT, dict[str, Any]]
            The dataset instance and any transformation metadata.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            data_root=cfg["data_root"],
            backend=cfg["backend"],
            streaming=cfg["streaming"],
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        n = len(self) if self._data is not None and not self._streaming else "?"
        return (
            f"{base}\n"
            f"Assets: {n}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
