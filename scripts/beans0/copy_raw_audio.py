"""Copy raw audio to Beans0 raw/ directory from other buckets."""

import argparse
import asyncio
import json

import pandas as pd
from tqdm import tqdm

from esp_data.file_io.files import GSFile
from esp_data.paths import AnyPath
from esp_data.utils import make_simple_logger

logger = make_simple_logger(name="beans0_copy_data", add_file_handler=True)


async def send_file_async(original_path: str, file_name: str, target_dir: str):
    orig_path = GSFile(original_path)
    target = AnyPath(target_dir) / file_name

    asyncio.sleep(0.2)
    if not target.exists():
        orig_path.copy_to(target)


def send_file_sync(original_path: str, file_name: str, target_dir: str):
    orig_path = GSFile(original_path)
    target = AnyPath(target_dir) / file_name

    if not target.exists():
        orig_path.copy_to(target)


def main():
    parser = argparse.ArgumentParser(
        description="Copy files that are in a metadata.csv file to the raw/ folder of Beans0 from their original paths"
    )
    parser.add_argument("--metadata_file_path", type=str)
    parser.add_argument("--original_paths_file_path", type=str)
    parser.add_argument("--target_dir", type=str)
    parser.add_argument("--state_json_file", type=str, default=None)

    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata_file_path)
    original_paths = pd.read_csv(args.original_paths_file_path)

    # state
    start = 0
    done_with = []
    if args.state_json_file:
        with open(args.state_json_file, "r") as f:
            state = json.load(f)
            start = state["start"]
            done_with = state["done_with"]

    for i in tqdm(range(start, len(original_paths)), total=len(original_paths) - start):
        try:
            p = str(original_paths["path"].iloc[i])
            send_file_sync(p, str(metadata["file_name"].iloc[i]), args.target_dir)
            done_with.append(p)

        except Exception as e:
            logger.error(f"Failed with exception {e}")
            with open("./state.json", "w") as f:
                json.dump({"start": i, "done_with": done_with}, f)
            raise RuntimeError(f"Something went wrong {e}")


if __name__ == "__main__":
    main()
