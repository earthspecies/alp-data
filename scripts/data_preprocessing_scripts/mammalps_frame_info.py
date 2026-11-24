"""Add frame sampling columns to the MammAlps split tables.

Each split CSV gets the columns that `MammAlps` uses to decode clips, filled with
values that read every clip whole at its native frame rate:

- `start_frame`: 0
- `end_frame`: `num_frames`
- `video_fps`: `average_fps`
- `frame_stride`: 1
- `fps`: `video_fps`

`MammAlps(frames_per_second=...)` overwrites `frame_stride` and `fps` at load time.

By default the tables in `SPLIT_CSVS` are overwritten in place. Re-running is
safe: existing frame columns are reset to the defaults above. Writes to GCS go
through `gsutil` (user credentials), since the default `gcsfs` credentials may
only have read access to the bucket.

Usage:
    uv run python scripts/data_preprocessing_scripts/mammalps_frame_info.py
    # or write copies elsewhere instead (local dir or gs:// prefix)
    uv run python scripts/data_preprocessing_scripts/mammalps_frame_info.py \
        --out-dir <dir>
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from alp_data.io import anypath, filesystem_from_path

_V010_ROOT = "gs://esp-data-ingestion/mammalps/v0.1.0"
SPLIT_CSVS = [
    f"{_V010_ROOT}/benchmark_1/metadata/{name}.csv" for name in ("all", "train", "val", "test")
]


def add_frame_info(df: pd.DataFrame) -> pd.DataFrame:
    """Add the frame sampling columns to a split table.

    Parameters
    ----------
    df : pd.DataFrame
        A MammAlps split table with `num_frames` and `average_fps` columns.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with `start_frame`, `end_frame`, `video_fps`,
        `frame_stride` and `fps` columns.
    """
    df = df.copy()
    df["start_frame"] = 0
    df["end_frame"] = df["num_frames"].astype(int)
    df["video_fps"] = df["average_fps"].astype(float)
    df["frame_stride"] = 1
    df["fps"] = df["video_fps"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for the updated CSVs. Defaults to overwriting each CSV in place.",
    )
    parser.add_argument(
        "--split-csvs",
        nargs="+",
        default=SPLIT_CSVS,
        help="Split CSVs to process. Defaults to the benchmark 1 tables in the bucket.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:
        for csv in args.split_csvs:
            src = anypath(csv)
            dst = src if args.out_dir is None else anypath(args.out_dir) / src.name
            # Read empty strings as NaN but keep literal values such as "None", and
            # parse floats exactly so the existing columns are written back unchanged
            with filesystem_from_path(src).open(str(src), "r") as f:
                df = pd.read_csv(
                    f, keep_default_na=False, na_values=[""], float_precision="round_trip"
                )
            df = add_frame_info(df)

            local_fp = Path(tmp_dir) / src.name
            df.to_csv(local_fp, index=False)
            if str(dst).startswith("gs://"):
                subprocess.run(["gsutil", "cp", str(local_fp), str(dst)], check=True)
            else:
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(local_fp, dst)
            print(f"{src} -> {dst} ({len(df)} rows)")


if __name__ == "__main__":
    main()
