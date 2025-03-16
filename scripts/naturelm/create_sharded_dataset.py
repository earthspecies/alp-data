"""Make NatureLM using a jsonl file"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import torchaudio
from datasets import load_dataset
from naturelm_cfg import LICENSES, NatureLMSample, naturelm_cfg
from tqdm import tqdm

import esp_data.file_io.functional as F
from esp_data.dataset.shard_creator import (
    BATCH_COLOR,
    RESET_COLOR,
    SHARD_COLOR,
    load_checkpoint,
    save_checkpoint,
    write_shard,
)
from esp_data.paths import AnyPath, make_storage_options

# def read_jsonl(path: str | AnyPath) -> list[dict]:
#     try:
#         with F.open_file(path, "r") as f:
#             data = json.load(f)
#             annotation = data["annotation"]
#     except (json.JSONDecodeError, KeyError):
#         with F.open_file(path, "r") as f:
#             annotation = [json.loads(line) for line in f]
#     return annotation


def load_audio(audio_path: str) -> np.ndarray:
    with F.open_file(audio_path, "rb") as f:
        audio_data, sr = torchaudio.load(f)
    audio_data = audio_data.numpy().squeeze().tolist()
    return audio_data, sr


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
    row: dict,
    remove_inaturalist: bool = True,
) -> dict[str, Any]:
    """Prepare a sample for the NatureLM dataset."""

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
    sample_data["source_dataset"] = row.get("source", "NatureLM")

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
            audio_data, sr = load_audio(file_path)
        except Exception as e:
            print(f"Error reading audio file {e}")
            raise e

    md["sample_rate"] = sr
    sample_data["metadata"] = json.dumps(md)

    # validate
    sample_data = NatureLMSample(**sample_data).to_dict()
    # drop some
    sample_data.pop("derived_from")
    sample_data.pop("version")

    sample_data = {"audio": audio_data, **sample_data}

    try:
        validate_sample(sample_data)
    except AssertionError as e:
        print(f"Validation failed for sample {sample_data['id']}, file_name {sample_data['file_name']}: {e}")
        raise e

    return sample_data


def create_sharded_dataset(
    data,
    start_idx,
    processed_samples,
    output_path,
    sample_prep_function,
    num_samples_per_shard,
    num_workers,
    shard_type,
    log_every,
    error_handling="warn",
    **sharding_kwargs,
) -> pd.DataFrame:
    """
    Create sharded dataset from information in a dataframe (or a Iterable[dict]) and a sample prep function,
    in parallel, with checkpointing.

    Arguments
    ---------
    data: Union[pd.DataFrame, Iterable[dict]]
        DataFrame or list of dictionaries containing sample data
    output_path: Union[str, AnyPath]
        Path to save the shards
    sample_prep_function: Callable
        Function to prepare a sample for the shard format. Must return a dictionary.
        For e.g., this function could be tokenizer, a feature preprocessor, a pydantic validator etc.
    num_samples_per_shard: int
        Number of samples to include in each shard
    num_workers: int
        Number of workers to use for parallel processing
    shard_type: str
        Output format for the shards ("webdataset", "arrow", "parquet", "hf")
    dataset_config: DatasetConfig
        Dataset configuration object, will create skeleton config if not provided.
    save_metadata_as: Optional[str]
        File format to save metadata as (parquet, csv, json, jsonl, tsv). This acts as an index for the shards.
    merge_data_and_metadata: bool
        Merge the shard information with the original data. If your data is a small DataFrame, you can combine the
        shard information with the original data to create a single DataFrame with shard information.
    log_every: int
        Log progress every N samples
    error_handling: str
        How to handle errors ("warn", "raise", "ignore")
    sharding_kwargs:
        Additional keyword arguments for the specific shard writer

    Returns
    -------
        pd.DataFrame: DataFrame containing metadata about the created shards

    Example
    -------
        >>> data = pd.DataFrame([
        ...     {"id": 1, "text": "Example 1"},
        ...     {"id": 2, "text": "Example 2"},
        ... ])
        >>> def prep_sample(item):
        ...     return {"length": len(item["text"]), **item}
        >>> metadata_df = create_sharded_dataset(
        ...     data,
        ...     "/tmp/output",
        ...     prep_sample,
        ...     num_samples_per_shard=2
        ... )
    """
    output_path = AnyPath(output_path)

    # Load checkpoint if exists
    checkpoint_data = load_checkpoint(output_path, data)
    completed_chunks = checkpoint_data["completed_chunks"] if checkpoint_data else {}
    print(f"Loaded checkpoint with {len(completed_chunks)} completed chunks.")

    # Calculate number of shards needed
    num_samples = len(data)
    num_shards = int(np.ceil(num_samples / num_samples_per_shard))

    # Split metadata into chunks for parallel processing
    chunks = [data[i * num_samples_per_shard : (i + 1) * num_samples_per_shard] for i in range(num_shards)]

    print(f"\n{BATCH_COLOR}=== Dataset Sharding Process ==={RESET_COLOR}")
    print(
        f"{BATCH_COLOR}Total samples: {num_samples}, Shards: {num_shards}, Samples per shard: {num_samples_per_shard}{RESET_COLOR}\n"
    )

    # Filter out completed chunks
    chunks_to_process = [
        (idx + start_idx, chunk) for idx, chunk in enumerate(chunks) if str(idx) not in completed_chunks
    ]

    if not chunks_to_process:
        print("All chunks already processed. Nothing to do.")
        return

    # Create partial function with fixed arguments
    process_chunk_partial = partial(
        write_shard,
        output_path=output_path,
        sample_prep_function=sample_prep_function,
        output_format=shard_type,
        log_every=log_every,
        error_handling=error_handling,
        **sharding_kwargs,
    )

    if num_workers > 1:
        # Parallel processing using ProcessPoolExecutor
        print(f"Processing {len(chunks_to_process)} shards using {num_workers} workers")

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
                for future in futures:
                    result = future.result()
                    chunk_id = result["shard_id"]
                    processed_samples.extend(result["processed_ids"])

                    # Update checkpoint after each chunk is completed
                    completed_chunks[str(chunk_id)] = result
                    save_checkpoint(output_path, completed_chunks, data)

                    # Update progress bar with shard info
                    pbar.set_postfix(
                        shard=f"{chunk_id:06d}",
                        success=len(result["processed_ids"]),
                        failed=len(result["failed_ids"]),
                    )
                    pbar.update(1)
    else:
        # Sequential processing
        print(f"Processing {len(chunks_to_process)} shards sequentially")

        with tqdm(
            total=len(chunks_to_process),
            desc="Processing shards",
            position=0,
            leave=True,
            ncols=100,
        ) as pbar_shards:
            for chunk_id, chunk in chunks_to_process:
                result = process_chunk_partial(batch=chunk, shard_id=chunk_id)
                processed_samples.extend(result["processed_ids"])

                # Update checkpoint after each chunk is completed
                completed_chunks[str(chunk_id)] = result
                save_checkpoint(output_path, completed_chunks, data)

                # Update progress bar with shard info
                pbar_shards.set_postfix(
                    shard=f"{chunk_id:05d}", success=len(result["processed_ids"]), failed=len(result["failed_ids"])
                )
                pbar_shards.update(1)

    return processed_samples


def main():
    # Builder script for NatureLM dataset
    parser = argparse.ArgumentParser(description="Create NatureLM dataset")
    parser.add_argument("--path_to_jsonl", type=str, required=True)
    parser.add_argument("--json_read_chunk_size", type=int, default=10_000)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--local_path_for_files", type=str, default="/mnt/")
    parser.add_argument("--num_samples_per_shard", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shard_type", type=str, default="hf")
    parser.add_argument("--remove_inaturalist", action="store_true")
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--changelog", type=str, default="")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--output_shard_pattern", type=str, default="**/*.arrow")

    args = parser.parse_args()

    sample_prep_function = partial(
        prepare_audio_sample_for_naturelm,
        remove_inaturalist=args.remove_inaturalist,
    )

    output_path = AnyPath(args.output_path) / args.version

    print("Updating NatureLM dataset config")
    naturelm_cfg.version = args.version.replace("v", "")
    naturelm_cfg.update_changelog(args.changelog)
    naturelm_cfg.write_json(output_path / "dataset_config.json")
    naturelm_cfg.generate_readme(output_path / "README.md")

    # first load metadata if exists
    metadata_path = output_path / "metadata.jsonl"
    if metadata_path.exists():
        print(f"Loading metadata from {metadata_path}")
        processed_samples = pd.read_json(metadata_path, lines=True, orient="records")
        num_chunks_processed = len(processed_samples) // args.json_read_chunk_size
    else:
        processed_samples = []
        num_chunks_processed = 0

    # print(f"Loading data in chunks of {args.json_read_chunk_size} from {args.path_to_json}")

    with pd.read_json(
        args.path_to_jsonl,
        lines=True,
        orient="records",
        chunksize=args.json_read_chunk_size,
        storage_options=make_storage_options(args.path_to_jsonl),
    ) as reader:
        for i, chunk in enumerate(reader):
            chunk_start_id = i * args.json_read_chunk_size
            if i < num_chunks_processed:
                print(f"Skipping chunk {i} as it has already been processed")
                continue

            print(f"=========== Processing chunk {i} with {len(chunk)} samples ===========")
            start = time.time()
            processed_samples = create_sharded_dataset(
                chunk,
                start_idx=chunk_start_id,
                processed_samples=processed_samples,
                output_path=output_path,
                sample_prep_function=sample_prep_function,
                num_samples_per_shard=args.num_samples_per_shard,
                num_workers=args.num_workers,
                shard_type=args.shard_type,
                log_every=args.log_every,
            )
            print(f"Processed chunk {i} in {time.time() - start:.2f} seconds")

            # save processed samples as global checkpoint
            shard_metadata_df = pd.DataFrame(processed_samples)
            shard_metadata_df.to_json(
                metadata_path, orient="records", lines=True, storage_options=make_storage_options(metadata_path)
            )

    # Few post-hoc checks
    ds = load_dataset(args.shard_type, output_path, streaming=False, file_pattern=args.output_shard_pattern)
    print(ds.columns)


if __name__ == "__main__":
    main()
