import argparse
import json

import numpy as np
import pyarrow as pa
from tqdm import tqdm

from esp_data.dataset.web_dataset.audio_ds import AudioDataset
from esp_data.file_io.functional import open_file
from esp_data.paths import AnyPath
from esp_data.utils import make_simple_logger

logger = make_simple_logger("export_as_arrow")


def write_audio_dataset_to_arrow(
    output_path: str | AnyPath,
    data_generator,
    batch_size: int = 3000,
    total_samples: int = None,
    shard_name_template: str = "shard_{:06d}.arrow",
):
    """
    Write audio data to an Arrow dataset in batches directly to a GCS bucket.

    Args:
        bucket_path: GCS path like "gs://my-bucket/my-dataset.parquet"
        data_generator: A generator or iterable that yields dictionaries with metadata and audio arrays
        batch_size: Number of samples to write in each batch
    """
    output_path = AnyPath(output_path)

    # get metadata key
    batch_data = None  # type: dict[str, Any]
    all_keys = None
    sample_count = 0
    shard_count = 0
    audio_dtype = None

    for i, (audio, metadata) in tqdm(enumerate(data_generator), desc="Writing batches", total=total_samples):
        # Debug check for audio data
        if audio is None or (isinstance(audio, np.ndarray) and audio.size == 0):
            logger.warning(f"Skipping sample with empty audio data: {metadata.get('id', 'unknown')}")
            continue

        # Remember the audio dtype on first sample
        if audio_dtype is None and isinstance(audio, np.ndarray):
            audio_dtype = str(audio.dtype)
            logger.info(f"Audio data type: {audio_dtype}")

        # Ensure audio_data is a numpy array
        if not isinstance(audio, np.ndarray):
            logger.warning(f"Converting non-numpy audio data to numpy array for sample {metadata.get('id', 'unknown')}")
            audio = np.array(audio)

        # Debug output for first few samples
        if sample_count < 3:
            logger.info(f"Sample {sample_count} audio shape: {audio.shape}, dtype: {audio.dtype}")
            logger.info(f"Sample {sample_count} audio range: min={audio.min()}, max={audio.max()}")

        if i == 0:
            all_keys = [key for key in metadata.keys()] + ["audio"]
            batch_data = {key: [] for key in all_keys}

        for key in metadata.keys():
            batch_data[key].append(metadata[key])
        batch_data["audio"].append(audio.tolist())
        sample_count += 1

        if sample_count >= batch_size:
            # Write this batch to a shard
            shard_path = output_path / shard_name_template.format(shard_count)
            _write_arrow_shard(shard_path, batch_data)

            # Reset batch data
            shard_count += 1
            batch_data = {key: [] for key in all_keys}
            sample_count = 0

    # Write any remaining samples to a final shard
    if sample_count > 0:
        shard_path = output_path / shard_name_template.format(shard_count)
        _write_arrow_shard(shard_path, batch_data)
        shard_count += 1

    # Write metadata about the dataset
    dataset_info = {
        "num_shards": shard_count,
        "audio_dtype": audio_dtype,
        "shard_template": shard_name_template,
    }

    with open(output_path / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    logger.info(f"Successfully wrote {shard_count} Arrow shards to {output_path}")
    return shard_count


def _write_arrow_shard(shard_path: AnyPath, batch_data: dict):
    """Helper function to write a single Arrow shard in a format compatible with HuggingFace datasets."""
    # Convert to PyArrow Table
    n = len(batch_data["audio"])
    table = pa.Table.from_pydict(batch_data)
    # Write the table to file
    logger.info(f"Writing shard to {str(shard_path)} with {n} samples")
    with open_file(shard_path, mode="wb") as f:
        with pa.ipc.new_file(f, table.schema) as writer:
            writer.write_table(table)

    return table.schema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_dataset", type=str, help="Path to the WebDataset dataset")
    parser.add_argument("--output_path", type=str, help="Path to the output GCS bucket")
    parser.add_argument("--batch_size", type=int, default=3000, help="Number of samples to write in each batch")

    args = parser.parse_args()

    # Create an AudioDataset instance
    ds = AudioDataset(web_dataset_path=args.path_to_dataset)
    # Write the dataset to a new GCS bucket
    write_audio_dataset_to_arrow(
        args.output_path,
        ds,
        batch_size=args.batch_size,
        total_samples=len(ds),
    )


if __name__ == "__main__":
    main()
