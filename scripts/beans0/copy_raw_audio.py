"""Copy raw audio to Beans0 raw/ directory from other buckets."""

import argparse
import asyncio

import pandas as pd
from tqdm import tqdm

from esp_data.paths import AnyPath


async def send_file_async(original_path: str, file_name: str, target_dir: str):
    orig_path = AnyPath(original_path)
    target = AnyPath(target_dir) / file_name

    asyncio.sleep(0.2)
    if not AnyPath(target).exists():
        orig_path.copy_to(target)


def send_file_sync(original_path: str, file_name: str, target_dir: str):
    orig_path = AnyPath(original_path)
    target = AnyPath(target_dir) / file_name

    if not AnyPath(target).exists():
        orig_path.copy_to(target)


def main():
    parser = argparse.ArgumentParser(
        description="Copy files that are in a metadata.csv file to the raw/ folder of Beans0 from their original paths"
    )
    parser.add_argument("--metadata_file_path", type=str)
    parser.add_argument("--original_paths_file_path", type=str)
    parser.add_argument("--target_dir", type=str)

    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata_file_path)
    original_paths = pd.read_csv(args.original_paths_file_path).to_list()

    for i, p in tqdm(enumerate(original_paths)):
        send_file_sync(p, str(metadata["file_name"].iloc[i]), args.target_dir)


if __name__ == "__main__":
    main()
