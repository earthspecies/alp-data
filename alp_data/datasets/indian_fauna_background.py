"""India Ecoacoustics Network (IEN) Background noise dataset."""

from __future__ import annotations

import pandas as pd

from alp_data import DatasetInfo, register_dataset
from alp_data.datasets.indian_fauna_weak import IndianFaunaWeak
from alp_data.io import anypath

_GCS_ROOT = "gs://esp-data-ingestion/indian-fauna/v0.1.0/background"


@register_dataset
class IndianFaunaBackground(IndianFaunaWeak):
    """India Ecoacoustics Network — Background noise recordings.

    Description
    -----------
    Ramesh, Singh et al. (2025/2026), "A large-scale crowd-sourced annotated
    acoustic dataset of Indian fauna" (bioRxiv ``10.64898/2026.07.20.739496``,
    data CC-BY-NC-4.0). The Background record collects recordings dominated by
    non-target sound — wind, rain, vehicular noise, domestic animals (cattle,
    hens, roosters), and unknown vocalizations. Ingested with the same weak
    clip-level shape as :class:`IndianFaunaWeak`: one row per recording, with the
    noise categories in ``foreground_species`` and a full-clip ``selection_table``
    (``Presence`` column). Useful as negatives / noise augmentation.

    Splits
    ------
    Single ``all`` split.

    References
    ----------
    https://zenodo.org/records/18928201 . License: CC-BY-NC-4.0.
    """

    info = DatasetInfo(
        name="indian_fauna_background",
        owner="david",
        split_paths={
            "all": f"{_GCS_ROOT}/indian_fauna_background_all.csv",
        },
        version="0.1.0",
        description=(
            "India Ecoacoustics Network Background: crowd-sourced noise recordings "
            "(wind/rain/traffic/cattle/unknown) across Indian states; weak clip-level "
            "noise-category labels. Usable as negatives / noise augmentation."
        ),
        sources=["https://zenodo.org/records/18928201"],
        license="CC-BY-NC-4.0",
    )

    _mixup_group = "noise"

    def get_available_labels(self, anno_column: str = "Species") -> list[str]:
        """Return the noise-category vocabulary from the sidecar labels csv.

        Returns
        -------
        list[str]
            Sorted noise categories (e.g. ``Background``, ``unknown sp. 1``).
        """
        labels_path = (
            anypath(self.info.split_paths["all"]).parent / "indian_fauna_background_labels.csv"
        )
        df = pd.read_csv(labels_path, keep_default_na=False, na_values=[""])
        return sorted(str(s) for s in df["Species"].dropna().unique() if str(s))
