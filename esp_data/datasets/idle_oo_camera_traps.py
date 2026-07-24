"""IDLE-OO Camera Traps — BioCLIP-2 camera-trap species benchmark.

IDLE-OO Camera Traps (HDR Imageomics Institute;
``imageomics/IDLE-OO-Camera-Traps``) is the balanced camera-trap
species-classification benchmark introduced with BioCLIP 2
(arXiv:2505.23883). It curates image-level-labelled subsets of five LILA-BC
camera-trap datasets (Island Conservation, Desert Lion, Orinoquía, Ohio
Small Animals, ENA24), balanced per species, for ~2,590 images spanning 119
species.

esp-data exposes it as an **image-only** species-classification dataset. The
source already ships ``scientific_name`` and a full taxonomy, so no GBIF
crosswalk is needed — the canonical name space is the scientific names
directly.

The build script (``scripts/data_preprocessing_scripts/
idle_oo_camera_traps/build_idle_oo.py``) downloads the HuggingFace parquet,
writes the embedded images out, and uploads images plus manifest CSVs to
``gs://esp-data-ingestion/idle_oo_camera_traps/v0.1.0/``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from esp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, read_image

_GCS_ROOT = "gs://esp-data-ingestion/idle_oo_camera_traps/v0.1.0"


@register_dataset
class IDLEOOCameraTraps(Dataset):
    """IDLE-OO Camera Traps — BioCLIP-2 camera-trap species benchmark.

    Description
    -----------
    ~2,590 balanced camera-trap images across 119 species, curated from five
    LILA-BC datasets and labelled to the image level. Image-only; decoded
    lazily at load time. Species labels are scientific names (with common
    name and full taxonomy also carried).

    Columns
    -------
    asset_id : str
        Synthetic per-image id assigned by the builder.
    modality : str
        Always ``"image"`` (parity with other esp-data vision datasets).
    label : int
        0-based species class index (assigned by sorted scientific name).
    canonical_name : str
        Scientific (Latin) binomial name.
    species_common : str
        English common name.
    original_label : str
        The source dataset's raw label string.
    kingdom, phylum, class, order, family, genus, species : str
        Source taxonomy fields.
    split : str
        Always ``"test"`` (the benchmark ships a single test split).
    image_path : str
        Absolute ``gs://`` path to the JPEG.

    Splits
    ------
    - ``all`` / ``test`` : the full benchmark (identical; a single test split).

    Loader behaviour
    ----------------
    - ``image`` → reads the JPEG into ``image`` (HWC uint8), optionally
      resized to ``(image_size, image_size)``.

    References
    ----------
    - Gu et al. (2025) "BioCLIP 2: Emergent Properties from Scaling
      Hierarchical Contrastive Learning", arXiv:2505.23883. Dataset:
      ``imageomics/IDLE-OO-Camera-Traps``.

    License: see the HuggingFace dataset card; staged on internal GCS for
    ESP research.
    """

    info = DatasetInfo(
        name="idle_oo_camera_traps",
        owner="david",
        split_paths={
            "all": f"{_GCS_ROOT}/idle_oo_all.csv",
            "test": f"{_GCS_ROOT}/idle_oo_test.csv",
        },
        version="0.1.0",
        description=(
            "IDLE-OO Camera Traps: ~2,590 balanced camera-trap images across "
            "119 species from five LILA-BC datasets. BioCLIP 2's camera-trap "
            "species-classification benchmark. Image-only."
        ),
        sources=("Gu et al. (2025) BioCLIP 2, arXiv:2505.23883; imageomics/IDLE-OO-Camera-Traps"),
        license="see HuggingFace dataset card; internal ESP research staging",
    )

    _mixup_group = "animal"

    def __init__(
        self,
        split: str = "test",
        output_take_and_give: dict[str, str] | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
        image_size: int | None = None,
    ) -> None:
        """Initialise the IDLE-OO Camera Traps dataset.

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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple[IDLEOOCameraTraps, dict[str, Any]]:
        """Create an IDLE-OO Camera Traps instance from a configuration.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Dataset configuration.

        Returns
        -------
        tuple[IDLEOOCameraTraps, dict[str, Any]]
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
