"""Build the Animal Kingdom (AR) multi-label action manifests.

Given a locally-extracted Animal Kingdom action-recognition release (``--src``
with the AR video clips + annotation), this builds per-split manifest CSVs with
absolute ``gs://`` ``video_path`` columns and a ``", "``-joined multi-label
action set per clip (140 actions).

Animal Kingdom is GATED (SUTD usage agreement / MS Forms), so it cannot be
curl'd unattended — ``jobs/build_animal_kingdom.sh`` expects a pre-staged
archive on scratch.

⚠️ VERIFY-ON-FIRST-RUN: the AR annotations follow the Charades CSV format with
fields ``clip_id, clip_number, frame_number, clip_path, action_labels`` and an
action label-map ``df_action.xlsx`` (action id -> action name). The exact
filenames (train/val/test csv) and the action-id delimiter are resolved by
:func:`load_action_map` / :func:`load_split_csv`, which **fail loud** if the
real files differ — adjust once the archive is in hand.

Usage (see jobs/build_animal_kingdom.sh):
    uv run python scripts/data_preprocessing_scripts/animal_kingdom/build_animal_kingdom.py \
        --src /scratch/$USER/animal_kingdom/action_recognition \
        --out /scratch/$USER/animal_kingdom/staging --splits test
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

GCS_ROOT_DEFAULT = "gs://esp-data-ingestion/animal_kingdom/v0.1.0"

_SPLIT_FILES: dict[str, list[str]] = {
    "train": ["train.csv", "df_action_train.csv", "AR_train.csv"],
    "test": ["test.csv", "df_action_test.csv", "AR_test.csv"],
}
_OUT_COLUMNS = ["asset_id", "modality", "labels", "split", "video_path"]


def load_action_map(annot_dir: Path) -> dict[str, str]:
    """Load the action-id -> action-name map from ``df_action.xlsx``.

    Parameters
    ----------
    annot_dir : Path
        The AR ``annotation`` directory.

    Returns
    -------
    dict[str, str]
        Mapping from string action id to action name.

    Raises
    ------
    FileNotFoundError
        If ``df_action.xlsx`` (or a ``.csv`` fallback) is not found.
    RuntimeError
        If id / name columns cannot be identified.
    """
    fp = next(
        (annot_dir / n for n in ("df_action.xlsx", "df_action.csv") if (annot_dir / n).exists()),
        None,
    )
    if fp is None:
        raise FileNotFoundError(
            f"df_action.[xlsx|csv] not found under {annot_dir}; "
            f"found {sorted(p.name for p in annot_dir.glob('*'))}"
        )
    df = pd.read_excel(fp) if fp.suffix == ".xlsx" else pd.read_csv(fp)
    cols = {c.lower(): c for c in df.columns}
    id_keys = ("action", "id", "action_id", "index")
    id_col = next((cols[c] for c in id_keys if c in cols), df.columns[0])
    name_keys = ("name", "action_name", "en", "action")
    name_col = next((cols[c] for c in name_keys if c in cols and cols[c] != id_col), None)
    if name_col is None:
        raise RuntimeError(f"{fp}: could not identify an action-name column in {list(df.columns)}")
    return {str(r[id_col]).strip(): str(r[name_col]).strip() for _, r in df.iterrows()}


def _resolve_action(action_id: str, action_map: dict[str, str]) -> str:
    """Map one action id to its name, tolerating ``c``-prefixed / zero-padded ids.

    Returns
    -------
    str
        The action name, or the raw id if unmapped.
    """
    if action_id in action_map:
        return action_map[action_id]
    stripped = action_id.lstrip("c").lstrip("0") or "0"
    return action_map.get(stripped, action_id)


def _find_split_csv(annot_dir: Path, split: str) -> Path:
    """Locate the AR split CSV for ``split``.

    Returns
    -------
    Path
        The resolved split CSV.

    Raises
    ------
    FileNotFoundError
        If no candidate split CSV is found.
    """
    for name in _SPLIT_FILES[split]:
        fp = annot_dir / name
        if fp.exists():
            return fp
    raise FileNotFoundError(
        f"No AR split csv for {split!r} under {annot_dir} (tried {_SPLIT_FILES[split]}); "
        f"found {sorted(p.name for p in annot_dir.glob('*.csv'))}"
    )


def load_split_csv(
    annot_dir: Path, split: str, action_map: dict[str, str]
) -> list[tuple[str, str, list[str]]]:
    """Parse an AR split CSV into ``(asset_id, clip_path, action_names)`` rows.

    Parameters
    ----------
    annot_dir : Path
        The AR ``annotation`` directory.
    split : str
        ``train`` / ``test``.
    action_map : dict[str, str]
        Action-id -> action-name map.

    Returns
    -------
    list[tuple[str, str, list[str]]]
        ``(asset_id, clip_path, [action_name, ...])`` per clip.

    Raises
    ------
    RuntimeError
        If clip-id / clip-path / action-label columns cannot be identified.
    """
    fp = _find_split_csv(annot_dir, split)
    df = pd.read_csv(fp, keep_default_na=False, na_values=[])
    cols = {c.lower(): c for c in df.columns}
    id_keys = ("clip_id", "video_id", "original_vido_id", "id")
    id_col = next((cols[c] for c in id_keys if c in cols), None)
    path_col = next((cols[c] for c in ("clip_path", "path", "video_path") if c in cols), id_col)
    lab_col = next((cols[c] for c in ("action_labels", "labels", "label") if c in cols), None)
    if id_col is None or lab_col is None:
        raise RuntimeError(f"{fp}: missing clip-id/action-label columns in {list(df.columns)}")
    rows = []
    for _, r in df.iterrows():
        aid = str(r[id_col]).strip()
        raw = str(r[lab_col]).strip()
        ids = [t for t in re.split(r"[\s,;]+", raw) if t]
        names = sorted({_resolve_action(i, action_map) for i in ids} - {""})
        rows.append((aid, str(r[path_col]).strip(), names))
    return rows


def build_split(
    src: Path, annot_dir: Path, split: str, action_map: dict[str, str], gcs_root: str
) -> pd.DataFrame:
    """Build the manifest for one AR split.

    Parameters
    ----------
    src : Path
        Extracted AR root (containing the clip videos).
    annot_dir : Path
        The AR ``annotation`` directory.
    split : str
        ``train`` / ``test``.
    action_map : dict[str, str]
        Action-id -> action-name map.
    gcs_root : str
        GCS root for absolute ``video_path`` columns.

    Returns
    -------
    pd.DataFrame
        The manifest for the split.

    Raises
    ------
    RuntimeError
        If any referenced clip file is missing.
    """
    video_dirs = ("dataset/video", "video", "clips")
    video_root = next((src / d for d in video_dirs if (src / d).is_dir()), src)
    rows = []
    missing = 0
    for aid, clip_path, names in load_split_csv(annot_dir, split, action_map):
        candidates = [video_root / clip_path, video_root / f"{aid}.mp4", src / clip_path]
        local = next((c for c in candidates if c.exists()), None)
        if local is None:
            missing += 1
            continue
        rows.append(
            {
                "asset_id": aid,
                "modality": "video",
                "labels": ", ".join(names),
                "split": split,
                "video_path": f"{gcs_root}/video/{aid}.mp4",
            }
        )
    if missing:
        raise RuntimeError(
            f"{split}: {missing} clips referenced by the annotation are missing under "
            f"{video_root}; refusing to ship a manifest with missing files."
        )
    return pd.DataFrame(rows, columns=_OUT_COLUMNS)


def main() -> None:
    """Run the Animal Kingdom (AR) manifest build."""
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Extracted AR root dir.")
    p.add_argument("--out", required=True, help="Staging output dir.")
    p.add_argument("--gcs-root", default=GCS_ROOT_DEFAULT)
    p.add_argument("--splits", nargs="+", default=["test"])
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    annot_dir = next((src / d for d in ("annotation", "annot") if (src / d).is_dir()), src)

    print("=== 1. load action label map ===", flush=True)
    action_map = load_action_map(annot_dir)
    print(f"{len(action_map)} actions")

    print("\n=== 2. build manifests ===", flush=True)
    frames = []
    for split in args.splits:
        df = build_split(src, annot_dir, split, action_map, args.gcs_root)
        df.to_csv(out / f"animal_kingdom_{split}.csv", index=False)
        n_multi = int((df["labels"].str.contains(",")).sum())
        print(f"animal_kingdom_{split}.csv: {len(df)} clips ({n_multi} multi-action)")
        frames.append(df)
    alldf = pd.concat(frames, ignore_index=True)
    alldf.to_csv(out / "animal_kingdom_all.csv", index=False)
    print(f"animal_kingdom_all.csv: {len(alldf)} clips")
    print("\nDONE.")


if __name__ == "__main__":
    main()
