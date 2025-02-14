"""This script creates the Huggingface dataset for the Beans0 dataset using multiprocessing."""

import argparse
import asyncio
import json
import logging
import os

from beans_cfg import ALL_DATASET_NAMES, LOCAL_PATHS, Beans0DatasetConfig, Beans0SampleNoAudio
from tqdm.asyncio import tqdm_asyncio

from esp_data.file_io import GSAudioFile
from esp_data.file_io.functional import write_rows_to_csv
from esp_data.paths import AnyPath

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def read_jsonl(path: str | AnyPath) -> list[dict]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
            annotation = data["annotation"]
    except (json.JSONDecodeError, KeyError):
        with open(path, "r") as f:
            annotation = [json.loads(line) for line in f]
    return annotation


async def make_sample(
    row: dict,
    source_dataset: str,
    license: str,
    replace_16k: bool = True,
    output_dir: str | AnyPath = None,
) -> dict:
    """Make a single BeansSample from a row in a component Beans0 dataset.

    Args:
        row (dict): The row from the component dataset.
        source_dataset (str): The name of the component dataset.
        license (str): The license for the component dataset.
        replace_16k (bool): Whether to replace 16k audio with original audio.
        audio_filenames (set): The set of audio filenames already processed.
        output_dir (str): The directory to save the audio to.

    Returns:
        dict: The Beans0Sample.
    """
    path_parts = AnyPath(row["path"]).parts

    ## VERY HACKY, issue is paths are local to David's VM
    if "foundation-model-data" in path_parts:
        idx_go = path_parts.index("foundation-model-data")
        animal_speak = False
    elif "animalspeak2" in path_parts:
        # look for "animalspeak2"
        idx_go = path_parts.index("animalspeak2")
        animal_speak = True
    else:
        logger.error(f"Path {path_parts} not in expected format")
        return None

    path = "gs://" + "/".join(path_parts[idx_go:])

    if replace_16k and not animal_speak:
        # ASSUMPTION: original audio is in "audio" and 16k audio is in "audio_16k"
        path = path.replace("audio_16k", "audio")

    audio_file = GSAudioFile(path)

    if not audio_file.exists:
        logger.error(f"Sample not found at {path}")
        return None

    # check if row has a license and recordist field
    metadata = {}
    if "license" in row:
        license = row["license"]
    if "recordist" in row:
        recordist = row["recordist"]
        metadata = {"recordist": recordist}
    if "url" in row:
        metadata["url"] = row["url"]

    # HACK, text is not present in all datasets, sometimes 'answer' is present
    # if "text" not in row and "answer" in row:
    #     row["text"] = row["answer"]

    if "source" in row:
        metadata["source"] = row["source"]
        source_dataset = row["source"]
        if source_dataset == "iNaturalist":
            metadata["exclude_from_release"] = True

    path = os.path.basename(path)

    try:
        assert output_dir is not None, "output_dir must be provided for the case of audio saved as raw file"

        sample = Beans0SampleNoAudio(
            source_dataset=source_dataset,
            license=license,
            metadata=metadata,
            file_name=path,
            prompt=row.get("prompt"),
            text=row.get("text", None),
            task=row.get("task", "None"),
        )
        sample = sample.to_dict()
        # remove derived_from
        sample.pop("derived_from", None)

        # copy the audio file to the output_dir
        await audio_file.async_copy_to(output_dir / path)

        return sample

    except Exception as e:
        logger.error(f"Pydantic Validation error {e}")
        return None


async def process_batch(
    batch_data: list[dict], dataset_cfg: dict, output_dir: str | AnyPath, replace_16k: bool, aggregate_dataset: dict
):
    """Process a batch of samples in parallel."""

    for row in batch_data:
        # check for duplicates
        path_parts = AnyPath(row["path"]).parts
        s = hash(f"{path_parts[-1]}_{row['prompt']}_{row['text']}")
        if s in aggregate_dataset["hashes"]:
            logger.warning(f"Audio filename {path_parts[-1]} already exists, skipping")
            continue

        aggregate_dataset["hashes"].add(s)

        sample = await make_sample(
            row,
            source_dataset=dataset_cfg["name"],
            license=dataset_cfg["license"],
            replace_16k=replace_16k,
            output_dir=output_dir,
        )

        if sample is None:
            aggregate_dataset["not_found"].append(row)
            continue

        aggregate_dataset["batches"].append(sample)

    # after every batch write to csv
    await write_rows_to_csv(aggregate_dataset["batches"], output_dir / "metadata.csv")


async def make_component_dataset_parallel(
    dataset_cfg,
    output_dir: str | os.PathLike | AnyPath,
    batch_size: int = 100,
    replace_16k: bool = True,
    aggregate_dataset: dict = None,
):
    """Create a dataset using parallel processing with async threads."""
    rows = read_jsonl(LOCAL_PATHS[dataset_cfg["name"]])

    # Split data into chunks for parallel processing
    chunk_size = (len(rows) // batch_size) + 1
    chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]
    logger.info(f"Processing {len(rows)} samples in {len(chunks)} chunks of chunk size {chunk_size}")

    # Create async tasks for gather
    await tqdm_asyncio.gather(
        *[
            process_batch(
                chunk,
                dataset_cfg=dataset_cfg,
                output_dir=output_dir,
                replace_16k=replace_16k,
                aggregate_dataset=aggregate_dataset,
            )
            for chunk in chunks
        ]
    )

    # Save the dataset as csv
    logger.info(f"Finished processing {dataset_cfg['name']} with {len(rows)} samples")
    return aggregate_dataset


async def main(args):
    output_dir = AnyPath(args.output_dir)

    aggregate_dataset = {
        "batches": [],
        "not_found": [],
        "hashes": set(),
    }

    # Create pool and process datasets
    # with mp.Pool(args.num_processes) as pool:
    #     process_func = partial(
    #         process_batch, dataset_cfg=dataset_cfg, output_dir=output_dir, replace_16k=replace_16k, no_audio=no_audio
    #     )

    #     results = list(tqdm(pool.imap(process_func, chunks), total=len(chunks), desc=f"Processing {name}"))

    for name in args.components_to_process:
        dataset_cfg: dict = Beans0DatasetConfig.metadata[name]

        aggregate_dataset = await make_component_dataset_parallel(
            dataset_cfg,
            output_dir,
            args.batch_size,
            args.replace_16k,
            aggregate_dataset,
        )

        logger.warning(f"Could not process {len(aggregate_dataset['not_found'])} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Beans0 dataset using parallel processing")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="gs://foundation-model-data/beans0/v0.0.0/",
        help="The directory to save the dataset to",
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

    args = parser.parse_args()

    asyncio.run(main(args))
