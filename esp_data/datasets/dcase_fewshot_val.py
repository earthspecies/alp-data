"""DCASE 2024 Task 5 few-shot bioacoustic VALIDATION set (DRASDIC's validation set).

FASD13-compatible: the DCASE Task 5 annotation format (``Starttime``/``Endtime``/``Q`` with
``POS``/``UNK``) is exactly what FASD13 is built from, so this reuses :class:`FASD13`'s
loader, selection-table shape, N-shot protocol and multi-audio support wiring verbatim --
only the manifest paths and split names differ. Six validation collections
(HB, PB, ME, RD, PB24, PW; 43 recordings / ~50 h / 7 classes / 3,504 events). Used by
Hoffman et al. 2025 (DRASDIC) for model selection.
"""

from __future__ import annotations

from esp_data import DatasetInfo, register_dataset
from esp_data.datasets.fasd13 import FASD13

_GCS_BASE = "gs://esp-data-ingestion/dcase_fewshot_val/v0.1.0"

VAL_CODES = ("HB", "PB", "ME", "RD", "PB24", "PW")


@register_dataset
class DCASEFewshotVal(FASD13):
    """DCASE 2024 Task 5 few-shot bioacoustic detection validation set (FASD13-shaped)."""

    info = DatasetInfo(
        name="dcase_fewshot_val",
        owner="david",
        split_paths={
            "all": f"{_GCS_BASE}/dcase_val_all.csv",
            **{code: f"{_GCS_BASE}/dcase_val_{code}.csv" for code in VAL_CODES},
        },
        version="0.1.0",
        description=(
            "DCASE 2024 Task 5 few-shot bioacoustic event detection VALIDATION set "
            "(HB/PB/ME/RD/PB24/PW), FASD13-shaped; DRASDIC's validation set."
        ),
        sources=["https://doi.org/10.5281/zenodo.10829604", "https://arxiv.org/abs/2503.00296"],
        license="See DCASE 2024 Task 5 terms.",
    )
