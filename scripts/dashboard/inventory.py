"""Dataset inventory for the dashboard precompute pipeline.

Lists every public ESP dataset along with its manifest URL on the new
canonical bucket (``$ETHOHUB_DATA_HOME``, default ``gs://esp-data-274503``).
Private datasets (`corvid_wascher`, `subsegmentation`, `esp_raincoast`)
are excluded.

The manifest URL is derived from the dataset class's first
``info.split_paths`` entry: we drop the trailing CSV/parquet filename to
obtain the dataset's ``data_root`` and append ``manifest.json``. For
datasets whose split paths sit under a subdirectory like ``raw/``, we
strip that suffix so the manifest URL points at the version root.

The output of `build_inventory` is a list of `DatasetInventoryEntry`
records consumed by `build_assets.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from esp_data import datasets as ds_module

# Datasets explicitly marked as not-for-redistribution. Excluded from the
# dashboard regardless of `info.license` content.
PRIVATE_DATASETS: frozenset[str] = frozenset({"corvid_wascher", "subsegmentation", "esp_raincoast"})

# Subdirectory names that may appear between a dataset's version root and
# the split CSV. We strip these when deriving the manifest URL.
_VERSION_SUBDIRS: tuple[str, ...] = ("raw", "metadata", "splits")


@dataclass(frozen=True)
class DatasetInventoryEntry:
    """One entry in the dashboard's dataset inventory.

    Parameters
    ----------
    class_name : str
        The class name in `esp_data.datasets` (e.g. ``"INaturalist"``).
    info_name : str
        The value of ``DatasetInfo.name`` for this class.
    version : str
        Dataset version string from ``DatasetInfo.version``.
    license : str
        Dataset-level license string.
    manifest_url : str
        Fully-qualified URL to ``manifest.json`` on the canonical bucket.
    data_root : str
        Version root URL (parent of `manifest_url`).
    """

    class_name: str
    info_name: str
    version: str
    license: str
    manifest_url: str
    data_root: str
    manifest_candidates: tuple[str, ...]


def _derive_data_root(split_path: str) -> str:
    """Derive a dataset's version root URL from one of its split paths.

    Parameters
    ----------
    split_path : str
        A URL like ``gs://bucket/<name>/<version>/raw/train.csv``.

    Returns
    -------
    str
        The version root URL ending in ``/``, e.g.
        ``gs://bucket/<name>/<version>/``. If the immediate parent of the
        CSV is one of `_VERSION_SUBDIRS`, that subdir is also stripped.
    """
    scheme, _, rest = split_path.partition("://")
    p = PurePosixPath(rest)
    parent = p.parent
    if parent.name in _VERSION_SUBDIRS:
        parent = parent.parent
    return f"{scheme}://{parent}/"


def _candidate_manifest_urls(split_path: str, max_levels: int = 3) -> tuple[str, ...]:
    """Generate a small set of plausible `manifest.json` URLs.

    The split path's parent directory often holds the manifest, but for
    datasets with extra layout (`/raw/`, `/raw/16KHz/`, `/organized_data/`,
    `/files/<sub>/formatted/`) the manifest may live one or more levels
    above. We yield candidates from the deepest sensible level upward.

    Parameters
    ----------
    split_path : str
        A URL pointing to one of the dataset's split CSVs/parquets.
    max_levels : int, default=3
        How many parent levels to walk up when generating candidates.

    Returns
    -------
    tuple[str, ...]
        Candidate manifest URLs in order of preference (deepest first).
    """
    scheme, _, rest = split_path.partition("://")
    p = PurePosixPath(rest).parent
    candidates: list[str] = []
    seen: set[str] = set()
    for _ in range(max_levels + 1):
        url = f"{scheme}://{p}/manifest.json"
        if url not in seen:
            candidates.append(url)
            seen.add(url)
        # Stop once we'd ascend above the bucket.
        if p.parent == p:
            break
        p = p.parent
    return tuple(candidates)


def build_inventory() -> list[DatasetInventoryEntry]:
    """Build the dashboard inventory from the live `esp_data.datasets`.

    Returns
    -------
    list[DatasetInventoryEntry]
        One entry per public, non-private dataset class registered in
        `esp_data.datasets.__all__`. Datasets without a `split_paths`
        mapping are skipped.
    """
    entries: list[DatasetInventoryEntry] = []
    for class_name in ds_module.__all__:
        cls = getattr(ds_module, class_name)
        info = getattr(cls, "info", None)
        if info is None:
            continue
        if info.name in PRIVATE_DATASETS:
            continue
        split_paths = getattr(info, "split_paths", None) or {}
        if not split_paths:
            continue
        first_split = next(iter(split_paths.values()))
        data_root = _derive_data_root(first_split)
        candidates = _candidate_manifest_urls(first_split)
        entries.append(
            DatasetInventoryEntry(
                class_name=class_name,
                info_name=info.name,
                version=getattr(info, "version", "unknown"),
                license=getattr(info, "license", "") or "",
                manifest_url=data_root + "manifest.json",
                data_root=data_root,
                manifest_candidates=candidates,
            )
        )
    return entries
