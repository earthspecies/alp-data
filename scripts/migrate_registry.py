"""Shared registry and path-resolution helpers for dataset migration scripts.

Consumed by ``migrate_datasets_simple.py`` and ``migrate_datasets_v2.py``.
Contains only:

- ``DATASET_REGISTRY`` — per-dataset audio path column configuration.
- ``resolve_audio_paths`` — expand a metadata row into absolute audio paths.
- ``get_dataset_class`` — resolve a registered dataset class by name.
- ``get_version_configs`` — return ``{version: {split_paths, data_root}}``.
- ``derive_data_root`` — GCS prefix for a dataset-version.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Static path-column registry
# ---------------------------------------------------------------------------
# Maps DatasetInfo.name → path column configuration.
#
# Keys per entry:
#   originals_column       – column for original (variable-rate) audio
#   presampled_columns     – list of column names for pre-resampled audio
#   presampled_subdirs     – dict[column → subdir] placed between data_root
#                            and the relative path for pre-resampled audio
#   originals_audio_prefix – if True, prepend "audio/" when the path does
#                            not already start with it (XenoCanto pattern)
#   derived_paths          – list of {"column", "find", "replace"} dicts
#                            for datasets that derive pre-resampled paths
#                            by in-place string substitution
#   data_root              – GCS prefix under which relative audio paths
#                            resolve. Omit when the dataset uses VERSIONS
#                            (per-version data_root is picked up from the
#                            dataset class) or when data_root varies per
#                            split (subsegmentation, voxaboxen*) — the
#                            fallback ``split.parent`` is used in that case.

DATASET_REGISTRY: dict[str, dict] = {
    "animal-sound-archive": {
        "originals_column": "originals_path",
        "presampled_columns": ["32khz_path", "16khz_path"],
        "data_root": "gs://esp-ml-datasets/tierstimmenarchiv/v0.1.0/raw/",
    },
    "animalspeak": {
        "originals_column": "audio_path",
        "data_root": "gs://esp-ml-datasets/animalspeak/v0.1.0/raw/16KHz/",
    },
    "anuraset_strong": {
        "originals_column": "audio_path",
        "presampled_columns": ["16khz_path", "32khz_path"],
        "data_root": "gs://esp-ml-datasets/anuraset/",
    },
    "arctic_bird_sounds": {
        "originals_column": "audio_path",
        "data_root": "gs://esp-ml-datasets/arctic_bird_sounds/",
    },
    "audioset": {
        "originals_column": "local_path",
        "presampled_columns": ["32khz_path"],
        # VERSIONS on class — data_root per version
    },
    "audioset_strong": {
        "originals_column": "audio_path",
        "presampled_columns": ["32khz_path"],
        "data_root": "gs://esp-ml-datasets/audioset/v0.2.0/raw/",
    },
    "beans": {
        "originals_column": "local_path",
        "data_root": "gs://esp-ml-datasets/beans/v0.1.0/raw/",
    },
    "beans_zero": {
        "originals_column": "audio_path_original_sample_rate",
        "presampled_columns": ["audio_path_32KHz", "audio_path_16KHz"],
        "data_root": "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/",
    },
    "Bengalese Finch Calls": {
        "originals_column": "local_path",
        "data_root": "gs://esp-ml-datasets/bengalese_finch/v0.1.0/raw/",
    },
    "birdeep": {
        "originals_column": "audio_path",
        "presampled_columns": ["16khz_path"],
        "data_root": "gs://esp-ml-datasets/birdeep/",
    },
    "birdset": {
        "originals_column": "audio_path",
        "presampled_columns": ["16khz_path", "32khz_path"],
        "data_root": "gs://esp-ml-datasets/birdset/v0.1.0/raw/",
    },
    "chiffchaff_id": {
        "originals_column": "local_path",
        "data_root": "gs://esp-ml-datasets/chiffchaff_id/v0.1.0/raw/",
    },
    "corvid_wascher": {
        "originals_column": "audio_path",
        "data_root": "gs://esp-ml-datasets/wascher_corvid_comparison/",
    },
    "dclde2026": {
        "originals_column": "audio_path",
        "presampled_columns": ["16khz_path", "32khz_path"],
        "presampled_subdirs": {
            "16khz_path": "audio_16k",
            "32khz_path": "audio_32k",
        },
        "data_root": "gs://esp-ml-datasets/dclde2026/v0.1.0/raw/2026/dclde_2026_killer_whales/",
    },
    "dinardo_dolphin_whistles": {
        "originals_column": "local_path",
        "data_root": "gs://esp-ml-datasets/dinardo2023_dolphin_whistles/v0.1.0/raw/",
    },
    "esp_raincoast": {
        "originals_column": "local_path",
        "data_root": "gs://esp-raincoast/2023-2024/",
    },
    "geladas": {
        "originals_column": "local_path",
        "data_root": "gs://esp-ml-datasets/geladas/v0.1.0/raw/",
    },
    "giant_otters": {
        "originals_column": "path",
        "data_root": "gs://esp-ml-datasets/giant_otters/v0.1.0/raw/",
    },
    "gibbon_solos": {
        "originals_column": "local_path",
        "data_root": "gs://esp-ml-datasets/clink2020_gibbon_solos/v0.1.0/raw/",
    },
    "hawaiian_birds": {
        "originals_column": "audio_path",
        "data_root": "gs://esp-ml-datasets/hawaiian_birds/",
    },
    "inaturalist": {
        "originals_column": "originals_path",
        "presampled_columns": ["32khz_path", "16khz_path"],
        "data_root": "gs://esp-ml-datasets/inaturalist/v0.1.0/raw/",
    },
    "InfantMarmosetsVox": {
        "originals_column": "path",
        "derived_paths": [
            {"column": "path", "find": "audio_44k", "replace": "audio_16k"},
        ],
        "data_root": "gs://esp-ml-datasets/infant_marmosets_vox/",
    },
    "insectset_459": {
        "originals_column": "local_path",
        "data_root": "gs://esp-ml-datasets/insectset_459/v0.1.0/raw/",
    },
    "littleowl_id": {
        "originals_column": "local_path",
        "data_root": "gs://esp-ml-datasets/littleowl_id/v0.1.0/raw/",
    },
    "macaques_coo_calls": {
        "originals_column": "local_path",
        "data_root": "gs://esp-ml-datasets/macaques_coo_calls/v0.1.0/raw/",
    },
    "nocturnal_bird_migration": {
        "originals_column": "audio_path",
        "presampled_columns": ["16khz_path", "32khz_path"],
        "data_root": "gs://esp-ml-datasets/nocturnal_bird_migration/",
    },
    "pipit_id": {
        "originals_column": "local_path",
        "data_root": "gs://esp-ml-datasets/pipit_id/v0.1.0/raw/",
    },
    "powdermill": {
        "originals_column": "audio_path",
        "data_root": "gs://esp-ml-datasets/powdermill/",
    },
    "subsegmentation": {
        "originals_column": "audio_path",
        # data_root varies per split — use split.parent fallback
    },
    "superb_starling": {
        "originals_column": "Begin Path",
        "data_root": "gs://esp-ml-datasets/superb-starlings-keen/v0.1.0/organized_data/",
    },
    "voxaboxen": {
        "originals_column": "audio_fp",
        # data_root varies per split — use split.parent fallback
    },
    "voxaboxen_events": {
        "originals_column": "audio_fp",
        # data_root varies per split — use split.parent fallback
    },
    "wabad": {
        "originals_column": "audio_fp",
        "presampled_columns": ["16khz_path", "32khz_path"],
        "data_root": "gs://esp-ml-datasets/wabad/v0.1.0/raw/",
    },
    "watkins": {
        "originals_column": "audio_path",
        "presampled_columns": ["16khz_path", "32khz_path"],
        "data_root": "gs://esp-ml-datasets/watkins/v0.1.0/",
    },
    "xeno-canto": {
        "originals_column": "relative_path",
        "originals_audio_prefix": True,
        "presampled_columns": ["32khz_path", "16khz_path"],
        "presampled_subdirs": {
            "32khz_path": "audio_32k",
            "16khz_path": "audio_16k",
        },
        "data_root": "gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/",
    },
    "xeno_canto_annotated_jeantet_23": {
        "originals_column": "audio_path",
        "presampled_columns": ["16khz_path", "32khz_path"],
        "data_root": "gs://esp-ml-datasets/xeno_canto_annotated_jeantet_2023/",
    },
    "zebra_finch_julie_elie": {
        "originals_column": "local_path",
        "data_root": "gs://esp-ml-datasets/zebra_finch_julie_elie/v0.1.0/raw/",
    },
}


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def _is_null(val: object) -> bool:
    """Return True if the value is null / empty / NaN.

    Returns
    -------
    bool
        True if `val` is None, an empty string, or NaN.
    """
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    return False


def resolve_audio_paths(
    row: dict,
    registry_entry: dict,
    data_root: str,
) -> tuple[str | None, list[tuple[str, str]]]:
    """Resolve all audio GCS paths for a row.

    Returns
    -------
    (originals_gcs_path | None, [(column_name, full_gcs_path), ...])
        The first element is the originals path (for validation download).
        The second element is all paths to copy (including originals).
    """
    all_paths: list[tuple[str, str]] = []
    originals_path: str | None = None

    if not data_root.endswith("/"):
        data_root = data_root + "/"

    # --- Originals ---
    orig_col = registry_entry["originals_column"]
    orig_val = row.get(orig_col)
    if not _is_null(orig_val):
        rel = str(orig_val)
        # XenoCanto pattern: prepend "audio/" if needed
        if registry_entry.get("originals_audio_prefix") and not rel.startswith("audio/"):
            rel = "audio/" + rel
        originals_path = data_root + rel
        all_paths.append((orig_col, originals_path))

    # --- Pre-resampled columns ---
    presampled_subdirs = registry_entry.get("presampled_subdirs", {})
    for col in registry_entry.get("presampled_columns", []):
        val = row.get(col)
        if _is_null(val):
            continue
        rel = str(val)
        subdir = presampled_subdirs.get(col)
        if subdir:
            full = data_root + subdir + "/" + rel
        else:
            full = data_root + rel
        all_paths.append((col, full))

    # --- Derived paths (string replacement, e.g. InfantMarmosetsVox) ---
    for derived in registry_entry.get("derived_paths", []):
        src_val = row.get(derived["column"])
        if _is_null(src_val):
            continue
        new_rel = str(src_val).replace(derived["find"], derived["replace"], 1)
        if new_rel != str(src_val):
            full = data_root + new_rel
            all_paths.append((f"{derived['column']}[{derived['replace']}]", full))

    return originals_path, all_paths


# ---------------------------------------------------------------------------
# Dataset class introspection
# ---------------------------------------------------------------------------
def get_dataset_class(dataset_name: str) -> type:
    # Look up the dataset class by name using the registry function
    # which will raise a clear error if not found.
    from esp_data import dataset_class_from_name

    return dataset_class_from_name(dataset_name)


def get_version_configs(dataset_class: str, registry_entry: dict | None = None) -> dict[str, dict]:
    """Return ``{version: {"split_paths": ..., "data_root": ...}}``.

    For non-versioned datasets returns a single entry keyed by the
    version string from DatasetInfo. ``data_root`` is included only when
    the dataset class declares ``VERSIONS`` (per-version) or the registry
    entry sets one. When omitted, callers should fall back to
    ``split_path``-derived roots (needed for datasets like ``voxaboxen``
    where each split lives under a different prefix).

    Parameters
    ----------
    dataset_class : type
        The dataset class to inspect for version and split path information.
    registry_entry : dict, optional
        The registry entry for the dataset, used to look up a default
        data_root when the dataset does not declare VERSIONS and the
        registry entry specifies one.

    Returns
    -------
    dict[str, dict]
    """
    if hasattr(dataset_class, "VERSIONS"):
        return dict(dataset_class.VERSIONS)

    version = dataset_class.info.version
    split_paths = dict(dataset_class.info.split_paths)
    vcfg: dict = {"split_paths": split_paths}
    if registry_entry is not None and "data_root" in registry_entry:
        vcfg["data_root"] = registry_entry["data_root"]
    # Otherwise omit data_root so `derive_data_root` falls back to each
    # split's parent — required for datasets like voxaboxen where every
    # split lives under a different prefix.
    return {version: vcfg}


def derive_data_root(
    split_path: str,
    version_config: dict,
    registry_entry: dict | None = None,
) -> str:
    """Derive the data_root GCS prefix for a dataset-version.

    Priority: per-version ``data_root`` from ``version_config`` (set on
    classes that declare ``VERSIONS``) > registry-level ``data_root`` >
    parent of ``split_path``.

    Parameters
    ----------
    split_path : str
    version_config : dict
        The version config dict containing split_paths and optionally
        data_root.
    registry_entry : dict, optional
        A ``DATASET_REGISTRY`` entry. When its ``data_root`` is set, it
        is used if the version config does not specify one. Required for
        datasets whose canonical audio root is not the direct parent of
        the split file (e.g. ``zebra_finch_julie_elie``,
        ``audioset_strong``).

    Returns
    -------
    str
        The GCS prefix for the dataset-version, guaranteed to end with
        ``/``.
    """
    if "data_root" in version_config:
        root = version_config["data_root"]
    elif registry_entry is not None and "data_root" in registry_entry:
        root = registry_entry["data_root"]
    else:
        root = split_path.rsplit("/", 1)[0] + "/"
    if not root.endswith("/"):
        root = root + "/"
    return root
