# /// script
# dependencies = [
#   "torchaudio",
#   "librosa==0.10.2",
#   "mlflow",
#   "fastparquet",
# ]
# ///
"""Make NatureLM using a jsonl file"""

import argparse
import io
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torchaudio
from naturelm_cfg import LICENSES, NatureLMSample, naturelm_cfg
from tqdm import tqdm

import esp_data.file_io.functional as F
from esp_data.dataset.shard_creator import (
    BATCH_COLOR,
    RESET_COLOR,
    SHARD_COLOR,
    infer_schema_from_sample,
    load_checkpoint,
    save_checkpoint,
    write_shard,
)
from esp_data.paths import AnyPath

AUDIO_PROMPT = "<Audio><AudioHere></Audio>"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("naturelm")


def encode_bytes(audio_data: np.ndarray, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="flac")
    buffer.seek(0)
    # Get the bytes content
    return buffer.read()


def read_jsonl(path: str | AnyPath) -> list[dict]:
    try:
        with F.open_file(path, "r") as f:
            data = json.load(f)
            annotation = data["annotation"]
    except (json.JSONDecodeError, KeyError):
        with F.open_file(path, "r") as f:
            annotation = [json.loads(line) for line in f]
    return annotation


def load_audio(audio_path: str | AnyPath) -> np.ndarray:
    with F.open_file(audio_path, "rb") as f:
        audio_data, sr = torchaudio.load(f)
    audio_data = audio_data.numpy().squeeze().tolist()
    return audio_data, sr


def validate_sample(sample: dict) -> dict:
    assert np.sum(np.isnan(sample["audio"])) == 0
    assert len(sample["output"]) > 0, "Output is empty"
    assert sample["output"] != "nan", "Output is nan"
    assert AUDIO_PROMPT in sample["instruction"], "Instruction does not contain AUDIO_PROMPT "
    assert AUDIO_PROMPT not in sample["instruction_text"], "Instruction text contains AUDIO_PROMPT"
    # assert isinstance(sample["metadata"], dif), "Metadata is not a string"
    assert len(sample["file_name"]) > 0, "File name is empty"
    assert len(sample["license"]) > 0, "License is empty"
    return sample


def prepare_audio_sample_for_naturelm(
    row: dict,
    remove_inaturalist: bool = True,
    shard_type: str = "hf",
) -> dict[str, Any]:
    """Prepare a sample for the NatureLM dataset."""

    row["output"] = "None" if (pd.isna(row["output"]) or row["output"] == "nan") else row["output"]

    if "prompt" not in row or row["prompt"] is None or row["prompt"] == "nan" or len(row["prompt"]) < 1:
        raise ValueError("Prompt missing!")

    if AUDIO_PROMPT not in row["prompt"]:
        row["prompt"] = f"{AUDIO_PROMPT} {row['prompt']}"

    sample_data = {
        "id": row["id"],
        "instruction": row["prompt"],
        "output": row["output"],
        "task": row["task"],
        "instruction_text": row["prompt"].replace(AUDIO_PROMPT, "").strip(),
    }

    path = AnyPath("/home/milad_earthspecies_org/data-migration/marius-highmem") / AnyPath(row["path"]).relative_to("/")

    sample_data["file_name"] = path.parts[-1]
    sample_data["source_dataset"] = row.get("source", "")
    if len(sample_data["source_dataset"]) < 1:
        sample_data["source_dataset"] = "NatureLM"

    if sample_data["source_dataset"].lower() in LICENSES:
        license = LICENSES[sample_data["source_dataset"].lower()]
        sample_data["license"] = license
    else:
        lic = row.get("license", "")
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
            audio_data, sr = load_audio(path)
        except Exception as e:
            logger.info(f"Error reading audio file {e}")
            raise e

        if np.std(audio_data) == 0.0:
            logger.info(f"WARNING: Audio is empty for sample {sample_data['id']}, filename {sample_data['file_name']}")

    md["sample_rate"] = sr
    md["duration"] = len(audio_data) / sr
    sample_data["metadata"] = md

    # validate
    sample_data = NatureLMSample(**sample_data).to_dict()
    # drop some
    sample_data.pop("derived_from")
    sample_data.pop("version")
    sample_data.pop("created_at")
    sample_data["metadata"] = json.dumps(md)
    sample_data = {"audio": audio_data, **sample_data}

    try:
        validate_sample(sample_data)
    except AssertionError as e:
        logger.info(f"Validation failed for sample {sample_data['id']}, file_name {sample_data['file_name']}: {e}")
        raise e

    if shard_type == "webdataset":
        # Store as bytes in memory
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data, sr, format="WAV")
        sample_data.pop("audio")
        return {
            "audio.wav": audio_buffer.getvalue(),
            "metadata.json": json.dumps(sample_data),
        }

    sample_data["audio"] = {
        "bytes": encode_bytes(audio_data, sr),
        "path": sample_data["file_name"],
    }
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
    """
    output_path = AnyPath(output_path)

    # Load checkpoint if exists
    checkpoint_data = load_checkpoint(output_path, data)
    completed_chunks = checkpoint_data["completed_chunks"] if checkpoint_data else {}
    logger.info(f"Loaded checkpoint with {len(completed_chunks)} completed chunks.")

    # Calculate number of shards needed
    num_samples = len(data)
    num_shards = int(np.ceil(num_samples / num_samples_per_shard))

    # Split metadata into chunks for parallel processing
    chunks = [data[i * num_samples_per_shard : (i + 1) * num_samples_per_shard] for i in range(num_shards)]

    logger.info(f"\n{BATCH_COLOR}=== Dataset Sharding Process ==={RESET_COLOR}")
    logger.info(
        f"{BATCH_COLOR}Total samples: {num_samples}, Shards: {num_shards}, Samples per shard: {num_samples_per_shard}{RESET_COLOR}\n"
    )

    # Filter out completed chunks
    chunks_to_process = [
        (idx + start_idx, chunk) for idx, chunk in enumerate(chunks) if str(idx + start_idx) not in completed_chunks
    ]

    if not chunks_to_process:
        logger.info("All chunks already processed. Nothing to do.")
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
        logger.info(f"Processing {len(chunks_to_process)} shards using {num_workers} workers")

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

                    mlflow.log_metric("chunk_done", chunk_id)
                    # Update progress bar with shard info
                    pbar.set_postfix(
                        shard=f"{chunk_id:06d}",
                        success=len(result["processed_ids"]),
                        failed=len(result["failed_ids"]),
                    )
                    pbar.update(1)
    else:
        # Sequential processing
        logger.info(f"Processing {len(chunks_to_process)} shards sequentially")

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
                    shard=f"{chunk_id:05d}",
                    success=len(result["processed_ids"]),
                    failed=len(result["failed_ids"]),
                )
                pbar_shards.update(1)

    return processed_samples


def main():
    # Builder script for NatureLM dataset
    parser = argparse.ArgumentParser(description="Create NatureLM dataset")
    parser.add_argument("--path_to_jsonl_files", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--json_read_chunk_size", type=int, default=100_000)
    parser.add_argument("--num_samples_per_shard", type=int, default=3000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shard_type", type=str, default="hf")
    parser.add_argument("--remove_inaturalist", action="store_true")
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--changelog", type=str, default="")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--output_shard_pattern", type=str, default="**/*.arrow")
    parser.add_argument("--error_handling", type=str, default="warn")
    parser.add_argument("--start_at_chunk", type=int, default=None)

    args = parser.parse_args()
    np.random.seed(0)

    sample_prep_function = partial(
        prepare_audio_sample_for_naturelm,
        remove_inaturalist=args.remove_inaturalist,
        shard_type=args.shard_type,
    )

    output_path = AnyPath(args.output_path)

    logger.info("Updating NatureLM dataset config")
    naturelm_cfg.version = args.version.replace("v", "")
    naturelm_cfg.update_changelog(args.changelog)
    naturelm_cfg.write_json(output_path / "dataset_config.json")
    naturelm_cfg.generate_readme(output_path / "README.md")

    processed_samples_file = AnyPath(output_path / "num_chunks_processed.txt")
    if processed_samples_file.exists() and args.start_at_chunk is None:
        with processed_samples_file.open("r") as f:
            num_chunks_processed = int(f.read())
    elif args.start_at_chunk is not None:
        num_chunks_processed = args.start_at_chunk
    else:
        num_chunks_processed = 0

    metadata_path = output_path / "metadata.jsonl"

    all_json_chunks = list(AnyPath(args.path_to_jsonl_files).rglob("*"))
    # shuffle the chunks
    np.random.shuffle(all_json_chunks)

    shards_completed = list(output_path.rglob(args.output_shard_pattern))
    logger.info(f"Found {len(shards_completed)} shards in {output_path}")
    num_shards = int(np.ceil(args.json_read_chunk_size / args.num_samples_per_shard))

    processed_samples = []
    writer = None
    schema = None

    for i, json_file in enumerate(all_json_chunks):
        chunk_start_id = i * num_shards

        if i < num_chunks_processed:
            logger.info(f"Skipping chunk {i} as it has already been processed")
            continue

        logger.info(f"Loading data from {json_file}")
        data = read_jsonl(json_file)
        data = pd.DataFrame(data)
        # shuffle
        data = data.sample(frac=1, random_state=0).reset_index(drop=True)

        logger.info(
            f"{SHARD_COLOR}=========== Processing chunk {i} with {len(data)} samples, in {num_shards} shards ==========={SHARD_COLOR}"
        )
        start = time.time()
        processed_samples: list[dict] | None = create_sharded_dataset(
            data,
            start_idx=chunk_start_id,
            processed_samples=processed_samples,
            output_path=output_path,
            sample_prep_function=sample_prep_function,
            num_samples_per_shard=args.num_samples_per_shard,
            num_workers=args.num_workers,
            shard_type=args.shard_type,
            log_every=args.log_every,
            error_handling=args.error_handling,
        )
        logger.info(f"Processed chunk {i} in {time.time() - start:.2f} seconds")

        if processed_samples:
            logger.info(f"Saving metadata for chunk {i}")
            with metadata_path.open("a") as f:
                for sample in processed_samples:
                    f.write(json.dumps(sample) + "\n")

            # Create DataFrame with shard information
            shard_info_df = pd.DataFrame(processed_samples, columns=["id", "shard_path", "shard_id"])

            # concatenate with 'data'
            data = pd.concat([data, shard_info_df], axis=1)
            if schema is None:
                sample = data.iloc[0].to_dict()
                schema = infer_schema_from_sample(sample)

            # append to a annotations.parquet file in the output path
            annotations_path = output_path / "annotations.parquet"

            if annotations_path.exists():
                file_obj = F.open_file(annotations_path, mode="ab")
                logger.info(f"Annotations file updated at {annotations_path}")
            else:
                file_obj = F.open_file(annotations_path, mode="wb")
                logger.info(f"Annotations file created at {annotations_path}")

            # Create appropriate writer
            if writer is None:
                writer = pq.ParquetWriter(file_obj, schema)
            table = pa.Table.from_pandas(data, schema=schema)
            writer.write_table(table)

            # save the number of chunks processed
            with processed_samples_file.open("w") as f:
                f.write(str(i + 1))

        logger.info(f"Num chunks processed: {i}")

        processed_samples = []

    if writer:
        writer.close()
        file_obj.close()


if __name__ == "__main__":
    import mlflow

    mlflow.set_tracking_uri(uri="http://100.89.114.62:8080")
    mlflow.set_experiment("naturelm_dataset_creation")

    with mlflow.start_run(log_system_metrics=True):
        main()
