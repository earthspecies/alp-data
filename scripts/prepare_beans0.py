"""This script creates the Huggingface dataset for the Beans0 dataset."""

import argparse
import json
import logging
import os
import pdb
import pickle
from typing import Optional

import yaml
from cloudpathlib import AnyPath
from dotenv import load_dotenv
from gcsfs import GCSFileSystem
from pydantic import Field
from tqdm import tqdm

from esp_data.config import DataSample, DatasetConfig
from esp_data.dataset import HFDataset
from esp_data.file_io import GSAudioFile

# load .env from project root
load_dotenv()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(console)

file_handler = logging.FileHandler("prepare_beans0.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


CREATOR = "M.Hagiwara; masato at earthspecies dot org"
STORAGE_OPTIONS = {"project": os.getenv("GCP_DEFAULT_PROJECT")}
LOCAL_CACHE_DIR = "./temp_cache"

DATASET_JSONL_PATHS = {
    "esc50": "gs://foundation-model-data/data/esc50-animal/test_processed.jsonl",
    "watkins": "gs://foundation-model-data/data/watkins-classification/test_processed.jsonl",
    "cbi": "gs://foundation-model-data/data/cbi-classification/test_processed.jsonl",
    "humbugdb": "gs://foundation-model-data/data/cbi-classification/test_processed.jsonl",
    "enabirds": "gs://foundation-model-data/data/enabirds-detection/test_processed.jsonl",
    "hiceas": "gs://foundation-model-data/data/hiceas-detection/test_processed.jsonl",
    "rfcx": "gs://foundation-model-data/data/rfcx-detection/test_processed.jsonl",
    "gibbons": "gs://foundation-model-data/data/hainan-gibbons-detection/test_processed.jsonl",
    "lifestage": "gs://foundation-model-data/data/animalspeak-lifestage/v1.32/test_filtered_processed_license.jsonl",
    "call-type": "gs://foundation-model-data/data/animalspeak-call-type/v1.32/test_classification_subset_processed.jsonl",
}

LICENSES = {
    "esc50": "CC-BY-NC",
    "rfcx": "academic, research & non-commercial use",
    "cbi": "CC-BY-NC-SA",
    "humbugdb": "CC-BY",
    "enabirds": "CC0",
    "hiceas": "data are free to use without restriction",
    "watkins": "free for personal/academic uses",
    "gibbons": "CC-BY-NC-SA",  # rest are missing or are at individual file level
    "lifestage": "per file licenses, please see individual files",
    "call-type": "per file licenses, please see individual files",
}

ALL_DATASET_NAMES = list(DATASET_JSONL_PATHS.keys())

fs = GCSFileSystem()


def load_beans_metadata():
    # load beans dataset config for naturelm
    beans_cfg_path = "beans_datasets.yml"
    with open(beans_cfg_path, "r") as f:
        beans_cfg = yaml.safe_load(f)

    metadata = {b["name"]: b for b in beans_cfg if b["name"] in DATASET_JSONL_PATHS}
    # HACK, have to add gibbons separately because of naming inconsistency
    metadata["gibbons"] = [b for b in beans_cfg if "gibbons" in b["name"]][0]
    # HACK, check with Masato and David
    # for lifestage and call-type, there are several possible datasets in beans that match
    metadata["lifestage"] = [b for b in beans_cfg if "lifestage" in b["name"]]
    metadata["call-type"] = [b for b in beans_cfg if "call-type" in b["name"]]

    return metadata


METADATA = load_beans_metadata()


def read_jsonl_from_bucket(file_path: str | os.PathLike) -> list[dict]:
    """Reads a json file assuming top level key is 'annotation'"""
    with fs.open(str(file_path)) as f:
        try:
            return json.load(f)["annotation"]

        except Exception as e:
            logger.error(f"Error reading jsonl {e}, trying line by line")
            # read lines
            records = f.readlines()
            return [json.loads(record) for record in records]

        except Exception as e:
            logger.error(f"Error reading jsonl {e}")
            return []


class BeansSample(DataSample):
    """Defines the structure of a Beans0 sample.

    Fields inherited from DataSample:
        - source_dataset: str
        - license: str | None
        - metadata: dict | None
        - created_at: datetime
        - id: str
        - derived_from: str | None
    """

    # required
    path: str = Field(description="Audio filename")
    prompt: str = Field(min_length=1, description="Prompt for naturelm")
    audio: list[float] = Field(description="The audio array")

    # optional
    text: Optional[str] = Field(default=None, description="Some text caption")
    label: Optional[str] = Field(default=None, description="Optional label for the audio")


def make_sample(
    row: dict, source_dataset: str, license: str, replace_16k: bool = True, audio_filenames: set = ()
) -> dict:
    """Make a single BeansSample from a row in a component Beans0 dataset."""

    path_parts = AnyPath(row["path"]).parts

    # check if the audio filename already exists
    if os.path.basename(path_parts[-1]) in audio_filenames:
        logger.error(f"Audio filename {path_parts[-1]} already exists")
        return None

    ## VERY HACKY, issue is paths are local to David's VM
    pdb.set_trace()
    idx_go = path_parts.index("foundation-model-data")
    # ASSUMPTION: all paths are in the form gs://foundation-model-data/...
    path = "gs://" + "/".join(path_parts[idx_go:])

    if replace_16k:
        # ASSUMPTION: original audio is in "audio" and 16k audio is in "audio_16k"
        path = path.replace("audio_16k", "audio")

    audio_file = GSAudioFile(path)

    if not audio_file.exists:
        logger.error(f"Sample not found at {path}")
        return None

    try:
        audio, _ = audio_file.read_audio()
    except Exception as e:
        logger.error(f"Error reading audio {e}")
        return None

    # check if row has a license and recordist field
    metadata = {}
    if "license" in row:
        license = row["license"]
    if "recordist" in row:
        recordist = row["recordist"]
        metadata = {"recordist": recordist}

    try:
        sample = BeansSample(
            source_dataset=source_dataset,
            license=license,
            metadata=metadata,
            path=os.path.basename(path),
            audio=list(audio),
            text=row.get("text", None),
            label=row.get("label", None),
            prompt=row.get("prompt", "None"),  ## HACK !!!
        )
        return sample.to_dict()

    except Exception as e:
        logger.error(f"Pydantic Validation error {e}")
        return None


def make_component_dataset(
    dataset_cfg,
    output_dir: str | os.PathLike | AnyPath,
    batch_size: int = 100,
    replace_16k: bool = True,
    state: dict = None,
    use_local_cache: bool = False,
):
    """Create a Huggingface dataset from a component Beans0 dataset.

    Args:
        dataset_cfg (dict): The configuration for the component dataset.
        output_dir (str): The path to save the dataset to.
        batch_size (int): The number of samples to process at once.
        replace_16k (bool): Whether to replace 16k audio with original audio.
        state (dict): The state of the dataset creation, can be used to resume processing.
    """
    name = dataset_cfg["name"]
    # read the jsonl
    rows = read_jsonl_from_bucket(dataset_cfg["path"])
    logger.info(f"#### PROCESSING DATASET {name} with {len(rows)} samples ####\n")

    batch = []
    not_found = []
    audio_filenames = set()
    shard_idx = 0
    ds = None
    config = DatasetConfig(
        name=name,
        creator=CREATOR,
        version=dataset_cfg["version"],
        description=dataset_cfg["description"],
        sources=name,
        license=LICENSES.get(name, "unknown"),
    )

    if state is None:
        state = {
            "batch": batch,
            "not_found": not_found,
            "shard_idx": shard_idx,
            "idx": 0,
            "config": config,
            "audio_filenames": audio_filenames,
        }

    for idx, row in tqdm(enumerate(rows, start=state["idx"]), total=len(rows) - state["idx"]):
        state["idx"] = idx

        sample: dict = make_sample(
            row,
            source_dataset=name,
            license=dataset_cfg["license"],
            replace_16k=replace_16k,
            audio_filenames=state["audio_filenames"],
        )

        if sample is None:
            state["not_found"].append(row)
            continue

        state["batch"].append(sample)
        state["audio_filenames"].add(sample["path"])

        if len(batch) == batch_size:
            if use_local_cache:
                ds = HFDataset.from_samples(batch, config)
                ds.save_to_path(path=AnyPath(LOCAL_CACHE_DIR) / f"{name}/shard_{shard_idx}", save_config=False)
                ds.save_config(AnyPath(LOCAL_CACHE_DIR) / f"{name}")
            else:
                ds = HFDataset.from_samples(batch, config)
                # do not save config, we can add it later
                ds.save_to_path(
                    path=AnyPath(output_dir) / f"shard_{shard_idx}", storage_options=STORAGE_OPTIONS, save_config=False
                )

            shard_idx += 1
            batch = []
            state["batch"] = batch
            state["shard_idx"] = shard_idx

        # save state as pickle
        if idx % 100 == 0:
            with open("state.pkl", "wb") as f:
                pickle.dump(state, f)

    # save the last batch
    if len(batch) > 0:
        if use_local_cache and len(rows) > batch_size and ds is not None:
            ds = HFDataset.from_samples(batch, config)
            ds.save_to_path(path=AnyPath(LOCAL_CACHE_DIR) / f"{name}/shard_{shard_idx}", save_config=False)

        elif ds is None:
            # only one shard smaller than batch size
            ds = HFDataset.from_samples(batch, config)
            ds.save_to_path(path=AnyPath(output_dir), storage_options=STORAGE_OPTIONS, save_config=False)

        else:
            ds = HFDataset.from_samples(batch, config)
            ds.save_to_path(
                path=AnyPath(output_dir) / f"shard_{shard_idx}", storage_options=STORAGE_OPTIONS, save_config=False
            )

    ds.save_config(output_dir)

    if use_local_cache and len(rows) > batch_size:
        # concatenate all the shards
        ds = HFDataset.from_path(AnyPath(LOCAL_CACHE_DIR) / f"{name}", sharded=True)
        ds.save_to_path(path=AnyPath(output_dir), storage_options=STORAGE_OPTIONS, save_config=False)
        # remove the temp cache
        # shutil.rmtree(AnyPath(LOCAL_CACHE_DIR) / f"{name}")

    return state, ds


def main():
    parser = argparse.ArgumentParser(description="Create Beans0 dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="gs://foundation-model-data/data/beans0",
        help="The directory to save the dataset to",
    )
    parser.add_argument(
        "--use_local_cache",
        action="store_true",
        help="Whether to use local cache. This will save batches on disk locally before concatenating and uploading to GCS",
    )
    parser.add_argument(
        "--components_to_process",
        nargs="+",
        type=str,
        default=ALL_DATASET_NAMES,
        help=f"The component datasets to process one of {ALL_DATASET_NAMES}",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="The number of samples to include in one shard",
    )
    parser.add_argument(
        "--replace_16k",
        action="store_true",
        help="Replace 16k audio with original audio",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        type=str,
        default=["0.0.1"] * len(ALL_DATASET_NAMES),
        help="The version of each component dataset",
    )
    parser.add_argument(
        "--resume_from_state",
        action="store_true",
    )

    args = parser.parse_args()

    if args.resume_from_state:
        with open("state.pkl", "rb") as f:
            state = pickle.load(f)
            state["config"] = DatasetConfig(**state["config"])
    else:
        state = None

    # check that args.components_to_process are valid
    for name in args.components_to_process:
        if name not in ALL_DATASET_NAMES:
            raise ValueError(f"Invalid dataset name {name}")

    if len(args.versions) == 1:
        args.versions = args.versions * len(args.components_to_process)

    if len(args.components_to_process) != len(args.versions):
        raise ValueError("Number of components to process and versions must be the same")

    for n, name in enumerate(args.components_to_process):
        dataset_cfg = {
            "name": name,
            "path": DATASET_JSONL_PATHS[name],
            "license": LICENSES.get(name, "unknown"),
            "version": args.versions[n],
            "description": f"""This is a component dataset of the BEANS0 eval dataset.
            It has following metadata:\n\n
            {json.dumps(METADATA[name], indent=4)}
            """,
        }

        output_dir = AnyPath(args.output_dir) / f"{name}"
        _, ds = make_component_dataset(
            dataset_cfg,
            output_dir,
            args.batch_size,
            args.replace_16k,
            state,
            args.use_local_cache,
        )

        logger.info(f"Finished processing {name} with {len(ds)} samples")


if __name__ == "__main__":
    main()
