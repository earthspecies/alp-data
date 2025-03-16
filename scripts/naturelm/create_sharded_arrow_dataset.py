# /// script
# dependencies = [
#   "colorama",
# ]
# ///
"""Make NatureLM using a folder with jsonl files as generator."""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Callable

import colorama
import numpy as np
import pandas as pd
from colorama import Fore, Style
from naturelm_cfg import LICENSES, NatureLMSample, naturelm_cfg
from tqdm import tqdm

import esp_data.file_io.functional as F
from esp_data.dataset.shard_creator import load_checkpoint, save_checkpoint, write_shard
from esp_data.file_io.parsers import read_audio_bytes_from_path
from esp_data.paths import AnyPath, is_cloud_path
from esp_data.utils import make_simple_logger

# Initialize colorama for cross-platform colored terminal output
colorama.init()

# Define color schemes for different progress levels
BATCH_COLOR = Fore.BLUE
SHARD_COLOR = Fore.GREEN
SAMPLE_COLOR = Fore.CYAN
SUCCESS_COLOR = Fore.GREEN
ERROR_COLOR = Fore.RED
RESET_COLOR = Style.RESET_ALL


logger = make_simple_logger("sharded_arrow_dataset_creator")

# Global flag for graceful interruption
_interrupt_processing = False


def handle_interrupt(signum, frame):
    """Signal handler for graceful interruption"""
    global _interrupt_processing
    logger.warning("Received interrupt signal. Will stop after current batch...")
    _interrupt_processing = True


def read_jsonl(path: str | AnyPath) -> list[dict]:
    try:
        with F.open_file(path, "r") as f:
            data = json.load(f)
            annotation = data["annotation"]
    except (json.JSONDecodeError, KeyError):
        with F.open_file(path, "r") as f:
            annotation = [json.loads(line) for line in f]
    return annotation


def validate_sample(sample: dict) -> dict:
    assert np.sum(np.isnan(sample["audio"])) == 0
    assert len(sample["output"]) > 0
    assert sample["output"] != "nan"
    assert "Audio" in sample["instruction"]
    assert "<Audio>" not in sample["instruction_text"]
    assert isinstance(sample["metadata"], str), "Metadata is not a string"
    assert len(sample["file_name"]) > 0, "File name is empty"
    assert len(sample["license"]) > 0, "License is empty"
    assert len(sample["task"]) > 0, "Task is empty"
    return sample


def prepare_audio_sample_for_naturelm(
    row: dict, remove_inaturalist: bool = True, local_path: str = "/mnt/"
) -> dict[str, Any]:
    row["output"] = "None" if (pd.isna(row["output"]) or row["output"] == "nan") else row["output"]
    sample_data = {
        "instruction": row["prompt"],
        "output": row["output"],
        "task": row["task"],
        "instruction_text": row["prompt"].replace("<Audio><AudioHere></Audio>", "").strip(),
    }

    path = AnyPath(row["path"])

    # HACK David's comments on original jsonl's
    # skip_cond1 = any([c in str(path) for c in ["compa_r", "audiocaps", "animal-instruct"]])
    # skip_cond2 = row["task"] in ["compa_r", "audiocaps-qa", "audiocaps", "animal-instruct"]

    # if skip_cond1 or skip_cond2:
    #     raise ValueError(f"Skipping {path} as it is not a valid audio file")

    # idx_go = path_parts.index("foundation-model-data")
    # file_path = os.path.join(local_path, *path_parts[idx_go:])
    file_path = path
    sample_data["file_name"] = path.parts[-1]
    sample_data["source_dataset"] = row.get("source_dataset", "NatureLM")

    if sample_data["source_dataset"].lower() in LICENSES:
        license = LICENSES[sample_data["source_dataset"].lower()]
        sample_data["license"] = license
    else:
        lic = row.get("license", "unknown")
        if len(lic) < 1:
            lic = "unknown"
        sample_data["license"] = lic

    # make metadata
    md = {}
    for k, v in row.items():
        if k in sample_data or k == "path" or k == "prompt":
            continue
        md[k] = v

    # if we encounter iNaturalist as the 'source_dataset', then we cant add audio to the dataset
    if remove_inaturalist and sample_data["source_dataset"] == "iNaturalist":
        # Create silent audio data (array of zeros)
        audio_data = np.zeros(1)
        duration = 0
        sample_data["duration"] = duration
        sr = 0
    else:
        # Read audio file
        try:
            audio_data, sr = read_audio_bytes_from_path(file_path)
        except Exception as e:
            logger.error(f"Error reading audio file {e}")
            raise e

    md["sample_rate"] = sr
    sample_data["metadata"] = json.dumps(md)

    # validate
    sample_data = NatureLMSample(**sample_data).to_dict()
    sample_data = {"audio": audio_data, **sample_data}

    try:
        validate_sample(sample_data)
    except AssertionError as e:
        logger.error(f"Validation failed for sample {sample_data['id']}: {e}")
        raise e

    return sample_data


def create_sharded_dataset(
    jsonl_paths: list[str | AnyPath],
    output_path: str | AnyPath,
    sample_prep_function: Callable,
    num_samples_per_shard: int = 1000,
    num_workers: int = 4,
    storage_options: dict = None,
    shard_type: str = "arrow",
):
    """Create sharded dataset from information in metadata dataframe and a sample prep function,
    in parallel with checkpointing.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing metadata for samples, importantly the 'id' column.
            the "file_name" column is also required if the sample_prep_function requires it.
        data_root (str | AnyPath): Path to the root directory containing raw data (e.g. audio files)
        output_path (str | AnyPath): Path to save the sharded dataset
        sample_prep_function (Callable): Function to prepare a sample for sharding
        num_samples_per_shard (int, optional): Number of samples per shard. Defaults to 1000.
        num_workers (int, optional): Number of workers for parallel processing. Defaults to 4.
        storage_options (dict, optional): Storage options for reading and writing files. Defaults to None.
    """
    output_path = AnyPath(output_path)

    print(f"\n{BATCH_COLOR}=== Dataset Sharding Process ==={RESET_COLOR}")

    for j, path in enumerate(jsonl_paths):
        t0 = time.time()

        batch_output_path = output_path / f"batch_{j}"
        logger.info(f"Reading batch {j} jsonl...")
        batch = read_jsonl(path)

        if len(batch) == 0:
            logger.error("Data is empty. Nothing to process.")
            return

        # Load checkpoint if exists
        checkpoint_data = load_checkpoint(batch_output_path, batch, "checkpoint.json")
        completed_chunks = checkpoint_data["completed_chunks"] if checkpoint_data else {}

        # Calculate number of shards needed
        num_samples = len(batch)
        num_shards = int(np.ceil(num_samples / num_samples_per_shard))

        # Split metadata into chunks for parallel processing
        chunks = np.array_split(batch, num_shards)

        print(f"{BATCH_COLOR}Input JSONL file: {path}{RESET_COLOR}")
        print(
            f"{BATCH_COLOR}Total samples: {num_samples}, Shards: {num_shards}, Samples per shard: {num_samples_per_shard}{RESET_COLOR}\n"
        )

        # Filter out completed chunks
        # Add batch number
        chunks_to_process = [
            (idx + j, chunk) for idx, chunk in enumerate(chunks) if str(idx + j) not in completed_chunks
        ]

        if not chunks_to_process:
            print(f"{SUCCESS_COLOR}All chunks already processed. Nothing to do.{RESET_COLOR}")
            logger.info("All chunks already processed. Nothing to do.")
            return

        # Create partial function with fixed arguments
        process_chunk_partial = partial(
            write_shard,
            output_path=batch_output_path,
            sample_prep_function=sample_prep_function,
            output_format=shard_type,
        )

        # Process chunks in parallel with progress bar
        processed_samples = []
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(process_chunk_partial, batch=chunk, shard_id=chunk_id)
                    for chunk_id, chunk in chunks_to_process
                ]

                # Track progress with tqdm
                with tqdm(
                    total=len(chunks_to_process),
                    desc=f"{SHARD_COLOR}Processing shards{RESET_COLOR}",
                    bar_format="{l_bar}%s{bar}%s{r_bar}" % (SHARD_COLOR, RESET_COLOR),
                    ncols=100,
                ) as pbar:
                    for i, future in enumerate(futures):
                        result = future.result()
                        chunk_id = result["shard_id"]
                        processed_samples.extend(result["processed_ids"])

                        completed_chunks[str(chunk_id)] = result
                        save_checkpoint(batch_output_path, batch, completed_chunks)

                        # Update progress bar with shard info
                        pbar.set_postfix(
                            shard=f"{chunk_id:05d}",
                            success=len(result["processed_ids"]),
                            failed=len(result["failed_ids"]),
                        )
                        pbar.update(1)

        else:
            # Track progress with tqdm
            # Sequential processing with nested progress bars
            with tqdm(
                total=len(chunks_to_process),
                desc=f"{SHARD_COLOR}Processing shards{RESET_COLOR}",
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (SHARD_COLOR, RESET_COLOR),
                position=0,
                leave=True,
                ncols=100,
            ) as pbar_shards:
                for chunk_id, chunk in chunks_to_process:
                    result = process_chunk_partial(batch=chunk, shard_id=chunk_id)

                    processed_samples.extend(result["processed_samples"])

                    # Update checkpoint
                    completed_chunks[str(chunk_id)] = result
                    save_checkpoint(batch_output_path, batch, completed_chunks)

                    # Update progress bar with shard info
                    pbar_shards.set_postfix(
                        shard=f"{chunk_id:05d}",
                        success=len(result["processed_samples"]),
                        failed=len(result["failed_ids"]),
                    )
                    pbar_shards.update(1)

                    print(f"\n{SAMPLE_COLOR}Processed samples: {len(processed_samples)}{RESET_COLOR}")

        tend = time.time()

        # Report final statistics
        total_successful = len(processed_samples)
        total_failed = sum(len(chunk["failed_ids"]) for chunk in completed_chunks.values())
        success_rate = (
            (total_successful / (total_successful + total_failed)) * 100 if (total_successful + total_failed) > 0 else 0
        )

        tend = time.time()
        print(f"\n{BATCH_COLOR}=== Processing Summary ==={RESET_COLOR}")
        print(f"{SUCCESS_COLOR}- Total files processed successfully: {total_successful}{RESET_COLOR}")
        print(f"{ERROR_COLOR}- Total files failed: {total_failed}{RESET_COLOR}")
        print(f"{BATCH_COLOR}- Success rate: {success_rate:.2f}%{RESET_COLOR}")
        print(f"{BATCH_COLOR}- Total time taken: {tend - t0:.2f} seconds{RESET_COLOR}")

        # Create DataFrame with shard information
        shard_info_df = pd.DataFrame(processed_samples)

        # Merge shard information with original metadata
        metadata_df = pd.DataFrame(batch)
        metadata_df = metadata_df.merge(shard_info_df[["id", "shard_path", "shard_id"]], on="id", how="left")

        # drop any "file_path" column
        if "file_path" in metadata_df.columns:
            metadata_df.drop(columns=["file_path"], inplace=True)

        # Save updated metadata as parquet
        metadata_df.to_csv(str(batch_output_path / "metadata.csv"), index=False, storage_options=storage_options)


def main():
    # Builder script

    parser = argparse.ArgumentParser(description="Create NatureLM dataset")
    parser.add_argument("--path_to_jsonl_files", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--local_path_for_files", type=str, default="/mnt/")
    parser.add_argument("--num_samples_per_shard", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shard_type", type=str, default="hf")
    parser.add_argument("--remove_inaturalist", action="store_true")
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--changelog", type=str, default="")

    args = parser.parse_args()

    # Find all jsonl files in the directory
    jsonl_paths = F.yield_files(args.path_to_jsonl_files, pattern="*.jsonl")

    sample_prep_function = partial(
        prepare_audio_sample_for_naturelm,
        remove_inaturalist=args.remove_inaturalist,
        local_path=args.local_path_for_files,
    )

    output_path = os.path.join(args.output_path, args.version)
    if is_cloud_path(output_path):
        storage_options = {"project": os.getenv("GCP_DEFAULT_PROJECT")}
    else:
        storage_options = None

    create_sharded_dataset(
        jsonl_paths,
        output_path,
        sample_prep_function,
        num_samples_per_shard=args.num_samples_per_shard,
        num_workers=args.num_workers,
        storage_options=storage_options,
        shard_type=args.shard_type,
    )

    # write new dataset config file
    naturelm_cfg.version = args.version.replace("v", "")
    naturelm_cfg.update_changelog(args.changelog)
    with AnyPath(os.path.join(output_path, "dataset_config.json")).open("w") as fp:
        json.dump(naturelm_cfg.to_dict(make_serializable=True), fp)

    naturelm_cfg.generate_readme(AnyPath(output_path) / "README.md")


if __name__ == "__main__":
    main()
