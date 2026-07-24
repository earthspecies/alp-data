"""NABirds — North American Birds image classification dataset (CVPR 2015).

NABirds (Van Horn et al., 2015; repo ``visipedia/nabirds``) is a
fine-grained image dataset of ~48k photographs covering 400 species of
birds commonly observed in North America, organised into 555 visual
categories (species plus male / female / juvenile / plumage variants).
It is one of the species-classification benchmarks used by BioCLIP 2
(arXiv:2505.23883).

esp-data exposes it as an **image-only** species-classification dataset:
each row is a single photograph labelled to the species level. The 555
visual categories are rolled up to their species node (via the source
``hierarchy.txt``) and GBIF-linked to a canonical scientific name so the
label space matches the other esp-data vision datasets (e.g. SSW60).

The build script (``scripts/data_preprocessing_scripts/nabirds/
build_nabirds.py``) parses the NABirds annotation tables, rolls the
categories up to species, GBIF-links the species, and uploads the images
plus manifest CSVs to ``gs://esp-data-ingestion/nabirds/v0.1.0/``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, read_image

_GCS_ROOT = "gs://esp-data-ingestion/nabirds/v0.1.0"


@register_dataset
class NABirds(Dataset):
    """NABirds — North American Birds fine-grained image dataset.

    Description
    -----------
    ~48k photographs of 400 North American bird species (555 visual
    categories rolled up to species), each labelled to a GBIF-linked
    canonical scientific name. Image-only; decoded lazily at load time.

    Columns
    -------
    asset_id : str
        Source NABirds image id.
    modality : str
        Always ``"image"`` (present for parity with other esp-data
        vision datasets).
    label : int
        0-based species class index.
    species_code : str
        NABirds visual-category id of the species node (source id).
    canonical_name : str
        GBIF canonical scientific name.
    species_common : str
        English common name (from NABirds ``classes.txt``).
    family, order, kingdom, phylum, class, genus, gbifID, taxonKey : str
        GBIF taxonomy fields populated by the builder.
    split : str
        Source split label (``train`` / ``test``).
    image_path : str
        Absolute ``gs://`` path to the JPEG.

    Splits
    ------
    - ``all`` : every image.
    - ``train`` / ``test`` : the official NABirds train/test split.

    Loader behaviour
    ----------------
    - ``image`` → reads the JPEG into ``image`` (HWC uint8), optionally
      resized to ``(image_size, image_size)``.

    References
    ----------
    - Van Horn et al. (2015) "Building a Bird Recognition App and Large
      Scale Dataset With Citizen Scientists: The Fine Print in Fine-Grained
      Dataset Collection", CVPR. Repo ``visipedia/nabirds``.

    License: research-only (the NABirds terms require agreeing to a usage
    agreement and forbid redistribution); staged on internal GCS for ESP
    research only.
    """

    info = DatasetInfo(
        name="nabirds",
        owner="david",
        split_paths={
            "all": f"{_GCS_ROOT}/nabirds_all.csv",
            "train": f"{_GCS_ROOT}/nabirds_train.csv",
            "test": f"{_GCS_ROOT}/nabirds_test.csv",
        },
        version="0.1.0",
        description=(
            "NABirds (North American Birds): ~48k images of 400 bird "
            "species (555 visual categories rolled up to species), "
            "GBIF-linked. Image-only species-classification benchmark used "
            "by BioCLIP 2."
        ),
        sources=("Van Horn et al. (2015) CVPR; repo visipedia/nabirds"),
        license="research-only (NABirds usage agreement; internal ESP research use)",
    )

    _mixup_group = "bird"

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
        image_size: int | None = None,
    ) -> None:
        """Initialise the NABirds dataset.

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
            The processed row (image + metadata).
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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple[NABirds, dict[str, Any]]:
        """Create a NABirds instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Dataset configuration.

        Returns
        -------
        tuple[NABirds, dict[str, Any]]
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
