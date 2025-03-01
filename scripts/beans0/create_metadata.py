import argparse
import json
import os

import pandas as pd
from beans_cfg import ALL_DATASET_NAMES, LOCAL_PATHS, Beans0SampleNoAudio, beans0_cfg
from tqdm import tqdm

from esp_data.paths import AnyPath
from esp_data.utils import make_simple_logger

logger = make_simple_logger("prepare_beans0", add_file_handler=True)


def make_sample(
    row: dict,
    source_dataset: str,
    license: str,
    task: str = None,
    replace_16k: bool = True,
    dataset_name: str = None,
) -> dict:
    """Make a single BeansSample from a row in a component Beans0 dataset.

    Args:
        row (dict): The row from the component dataset.
        source_dataset (str): The name of the component dataset.
        license (str): The license for the component dataset.
        task (str): The task for the component dataset.
        replace_16k (bool): Whether to replace 16k audio with original audio.
        dataset_name (str): The name of the component dataset.

    Returns:
        dict: The Beans0Sample.
    """
    path_parts = AnyPath(row["path"]).parts

    # HACK, issue many paths are local to David's VM
    animal_speak = False
    if "foundation-model-data" in path_parts:
        idx_go = path_parts.index("foundation-model-data")
        path = "gs://" + "/".join(path_parts[idx_go:])

    elif "animalspeak2" in path_parts:
        # look for "animalspeak2"
        idx_go = path_parts.index("animalspeak2")
        path = "gs://" + "/".join(path_parts[idx_go:])
        animal_speak = True

    elif "BirdVox-70k" in path_parts:
        path = os.path.join("gs://foundation-model-data", row["path"])

    elif "zebra_finch_elie" in path_parts:
        path = os.path.join("gs://foundation-model-data", "/".join(path_parts[1:]))

    else:
        logger.error(f"Path {path_parts} not in expected format")
        return None, None

    if replace_16k and not animal_speak:
        # ASSUMPTION: original audio is in "audio" and 16k audio is in "audio_16k"
        # for animal_speak, the audio is already in the correct path (already 16k downsampled)
        path = path.replace("audio_16k", "audio")

    # audio_file = GSAudioFile(path)

    # if not audio_file.exists:
    #     logger.error(f"Sample not found at {path}")
    #     return None

    # create file specific metadata
    metadata = {}
    if "recordist" in row:
        recordist = row["recordist"]
        metadata = {"recordist": recordist}
    if "url" in row:
        metadata["url"] = row["url"]

    # HACK, BirdVox does not have "prompt" field
    if "prompt" not in row and "instruction" in row:
        row["prompt"] = row["instruction"]

    # HACK, text is not present in all datasets, sometimes 'answer' is present
    if "text" not in row and "answer" in row:
        row["text"] = row["answer"]
    if "text" not in row and "label" in row:
        row["text"] = row["label"]

    file_name = os.path.basename(path)

    if "source" in row:
        metadata["source"] = row["source"]
        source_dataset = row["source"]
        if source_dataset == "iNaturalist":
            metadata["exclude_from_release"] = True

    if "license" in row:
        if row["license"]:
            license = row["license"]
        else:
            license = "unknown"

    if license == "per file licenses, please see individual files":
        license = "unknown"

    try:
        sample = Beans0SampleNoAudio(
            source_dataset=source_dataset,
            license=license,
            metadata=metadata,
            file_name=file_name,
            instruction=row["prompt"],
            instruction_text=row["prompt"].replace("<Audio><AudioHere></Audio>", "").strip(),
            output=row["text"],
            task=task,
            dataset_name=dataset_name,
        )
        sample = sample.to_dict()
        # remove derived_from and version
        sample.pop("derived_from", None)
        sample.pop("version", None)

        # copy the audio file to the output_dir
        # if not F.exists(output_dir / path):
        #     audio_file.copy_to(output_dir / path)

        return sample, path

    except Exception as e:
        logger.error(f"Pydantic Validation error {e}")
        return None, None


def read_jsonl(path: str | AnyPath) -> list[dict]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
            annotation = data["annotation"]
    except (json.JSONDecodeError, KeyError):
        with open(path, "r") as f:
            annotation = [json.loads(line) for line in f]
    return annotation


def main():
    parser = argparse.ArgumentParser(
        description="Build or append to metadata.csv and original_paths.csv for Beans0 dataset"
    )
    parser.add_argument("--metadata_file_path", type=str)
    parser.add_argument("--original_paths_file_path", type=str)
    parser.add_argument(
        "--components_to_process",
        nargs="+",
        type=str,
        default=ALL_DATASET_NAMES,
        help=f"The component datasets to process one of {ALL_DATASET_NAMES}",
    )
    parser.add_argument(
        "--replace_16k",
        action="store_true",
        help="Replace 16k audio with original audio",
    )

    args = parser.parse_args()

    local_paths = [(k, v) for k, v in LOCAL_PATHS.items() if k in args.components_to_process]

    # load all jsonl files
    samples = []
    original_paths = []
    notfound = []

    for name, path in local_paths:
        dataset_idx_in_cfg = [d["name"] for d in beans0_cfg.metadata["components"]].index(name)
        rows = read_jsonl(path)

        for row in tqdm(rows, total=len(rows), desc=f"Processing {name}"):
            dataset_cfg = beans0_cfg.metadata["components"][dataset_idx_in_cfg]

            source_dataset = dataset_cfg["source_dataset_name"]

            sample, original_path = make_sample(
                row,
                source_dataset=source_dataset,
                license=dataset_cfg["license"],
                replace_16k=args.replace_16k,
                task=dataset_cfg["task"],
                dataset_name=name,
            )

            if sample is None:
                notfound.append(row)
                continue

            samples.append(sample)
            original_paths.append(original_path)

    logger.info(f"Could not process {len(notfound)} samples from {args.components_to_process}")

    write_csv = input("Append samples to existing metadata.csv and original_paths.csv? (y/n) ")

    if write_csv == "y":
        if args.metadata_file_path is not None:
            metadata_df = pd.read_csv(args.metadata_file_path)
        else:
            raise ValueError("metadata_file_path is required")

        if args.original_paths_file_path is not None:
            original_paths_df = pd.read_csv(args.original_paths_file_path)
        else:
            raise ValueError("original_paths_file_path is required")

        # append to metadata and original_paths
        metadata_df = metadata_df.append(pd.DataFrame(samples))
        original_paths_df = original_paths_df.append(pd.DataFrame(original_paths, name="path"))

    else:
        metadata_df = pd.DataFrame(samples)
        # some outputs are "None" which pandas handles as and need to be converted to string
        metadata_df["output"] = metadata_df["output"].astype(str)
        original_paths_df = pd.Series(original_paths, name="path")

    metadata_df.to_csv(args.metadata_file_path, index=False, na_rep="NULL")
    original_paths_df.to_csv(args.original_paths_file_path, index=False, na_rep="NULL")

    with open("dataset_config.json", "w") as fp:
        json.dump(beans0_cfg.to_dict(make_serializable=True), fp)


if __name__ == "__main__":
    main()
