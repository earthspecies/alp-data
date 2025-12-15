"""We want to export the Beans Zero dataset audio files as FLAC format.

Requires: huggingface 'datasets' library
Usage: `uv run --with datasets[Audio] esp-data/scripts/beans_zero/beans_zero_export_as_flac.py`
"""

import argparse
import json
import os
from io import BytesIO

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

from esp_data.io import AnyPathT, exists, filesystem_from_path

fs = filesystem_from_path("gs://")

os.environ["HF_DATASETS_CACHE"] = "/mnt/home/esp-ml-datasets/"


def resample_audio(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to the target sample rate.

    Parameters
    ----------
    audio : np.ndarray
        The audio data to resample.
    sr : int
        The original sample rate of the audio data.
    target_sr : int
        The target sample rate.

    Returns
    -------
    np.ndarray
        The resampled audio data.
    """
    if sr != target_sr:
        audio = librosa.resample(
            y=audio,
            orig_sr=sr,
            target_sr=target_sr,
            scale=True,
            res_type="kaiser_best",
        )

    return audio


def write_flac(audio: np.ndarray, sample_rate: int, path: AnyPathT | str) -> None:
    """Write audio data to a FLAC file.

    Parameters
    ----------
    audio : np.ndarray
        The audio data to write.
    path : AnyPathT | str
        The path to the output FLAC file.
    """
    with fs.open(str(path), "wb") as f:
        buffer = BytesIO()
        sf.write(buffer, audio, samplerate=sample_rate, format="FLAC")
        f.write(buffer.getbuffer())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Beans Zero dataset audio files as FLAC format."
    )
    parser.add_argument(
        "--sample-rates",
        type=int,
        nargs="+",
        help="Sample rates to resample audio files to. "
        "If not provided, only original sample rate is used.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Whether to load the dataset in streaming mode.",
    )

    args = parser.parse_args()

    sample_rates = args.sample_rates if args.sample_rates else []

    # Load dataset
    beans_zero_hf = load_dataset(
        "EarthSpeciesProject/BEANS-Zero", streaming=args.streaming, split="test"
    )

    destination_bucket = "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/"
    audio_dir = destination_bucket + "audio_flac/"

    df = []  # eventually a dataframe of annotations
    num_samples = len(beans_zero_hf) if not args.streaming else None

    for sample in tqdm(beans_zero_hf, desc="Processing samples", total=num_samples):
        audio = sample["audio"]
        metadata = json.loads(sample["metadata"])

        original_sample_rate = metadata["sample_rate"]
        duration = metadata["duration_seconds"]

        annotations_dict = {
            "output": sample["output"],
            "instruction_text": sample["instruction_text"],
            "instruction": sample["instruction"],
            "task": sample["task"],
            "dataset_name": sample["dataset_name"],
            "file_name": sample["file_name"],
            "license": sample["license"],
            "id": sample["id"],
            "duration_seconds": duration,
        }

        if sample_rates:
            for target_sr in sample_rates:
                resampled_audio = resample_audio(
                    audio=np.array(audio["array"]),
                    sr=original_sample_rate,
                    target_sr=target_sr,
                )
                target_sr = int(target_sr / 1000)  # for folder naming
                audio_path = (
                    audio_dir
                    + f"{annotations_dict['dataset_name']}/{target_sr}Hz/"
                    + f"{annotations_dict['file_name']}.flac"
                )
                if not exists(audio_path):
                    write_flac(
                        audio=resampled_audio,
                        sample_rate=target_sr,
                        path=audio_path,
                    )
                annotations_dict[f"audio_path_{target_sr}Hz"] = audio_path

        # Write original sample rate audio
        audio_path = (
            audio_dir
            + f"{annotations_dict['dataset_name']}/original_sample_rate/"
            + f"{annotations_dict['file_name']}.flac"
        )
        if not exists(audio_path):
            write_flac(
                audio=np.array(audio["array"]),
                sample_rate=original_sample_rate,
                path=audio_path,
            )
        annotations_dict["audio_path_original_sample_rate"] = audio_path

        df.append(annotations_dict)

    # Write out dataframs per dataset as csvs
    df = pd.DataFrame(df)
    datasets = df["dataset_name"].unique()
    for dataset in datasets:
        dataset_df = df[df["dataset_name"] == dataset]
        csv_path = destination_bucket + f"{dataset}.csv"

        dataset_df.to_csv(str(csv_path), index=False)

    # Write full dataframe as csv
    full_csv_path = destination_bucket + "test.csv"
    df.to_csv(str(full_csv_path), index=False)


if __name__ == "__main__":
    main()
