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

from esp_data.io import AnyPathT, anypath, exists, filesystem_from_path

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


def write_flac(audio: np.ndarray, sample_rate: int, path: AnyPathT) -> None:
    """Write audio data to a FLAC file.

    Parameters
    ----------
    audio : np.ndarray
        The audio data to write.
    path : AnyPathT
        The path to the output FLAC file.
    """
    # Remove the suffix from the path because soundfile adds it automatically
    path_parent = path.parent
    path_without_suffix = path_parent / path.stem
    with fs.open(str(path_without_suffix), "wb") as f:
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
    parser.add_argument(
        "--dont-write-audio", action="store_true", help="If set, audio files will not be written."
    )

    args = parser.parse_args()

    sample_rates = args.sample_rates if args.sample_rates else []
    # Validate all sample rates are divisible by 1000
    for sr in sample_rates:
        if sr % 1000 != 0:
            raise ValueError(f"Sample rate {sr} is not divisible by 1000.")

    # Load dataset
    beans_zero_hf = load_dataset(
        "EarthSpeciesProject/BEANS-Zero", streaming=args.streaming, split="test"
    )

    destination_bucket = anypath("gs://esp-ml-datasets/beans-zero/v0.1.0/raw/")
    audio_dir = destination_bucket / "audio"

    df = []  # eventually a dataframe of annotations
    num_samples = len(beans_zero_hf) if not args.streaming else None

    for sample in tqdm(beans_zero_hf, desc="Processing samples", total=num_samples):
        audio = np.array(sample["audio"])
        metadata = json.loads(sample["metadata"])

        original_sample_rate = metadata["sample_rate"]
        annotations_dict = {
            "output": sample["output"],
            "instruction_text": sample["instruction_text"],
            "instruction": sample["instruction"],
            "task": sample["task"],
            "dataset_name": sample["dataset_name"],
            "file_name": sample["file_name"],
            "license": sample["license"],
            "id": sample["id"],
            "metadata": sample["metadata"],
            "source_dataset": sample["source_dataset"],
        }

        if sample_rates:
            for target_sr in sample_rates:
                target_sr_khz = int(target_sr / 1000)  # for folder naming
                audio_path = (
                    audio_dir
                    / f"{annotations_dict['dataset_name']}/{target_sr_khz}KHz"
                    / f"{annotations_dict['file_name']}.flac"
                )
                if not args.dont_write_audio:
                    # Takes a bit longer to test for existence on GCS, so do it last
                    if not exists(audio_path):
                        resampled_audio = resample_audio(
                            audio=audio,
                            sr=original_sample_rate,
                            target_sr=target_sr,
                        )

                        write_flac(
                            audio=resampled_audio,
                            sample_rate=target_sr,
                            path=audio_path,
                        )
                annotations_dict[f"audio_path_{target_sr}KHz"] = audio_path

        # Write original sample rate audio
        audio_path = (
            audio_dir
            / f"{annotations_dict['dataset_name']}/original_sample_rate"
            / f"{annotations_dict['file_name']}.flac"
        )

        if not args.dont_write_audio:
            # Takes a bit longer to test for existence on GCS, so do it last
            if not exists(audio_path):
                write_flac(
                    audio=audio,
                    sample_rate=original_sample_rate,
                    path=audio_path,
                )
        annotations_dict["audio_path_original_sample_rate"] = audio_path

        df.append(annotations_dict)

    # Write out dataframs per dataset as jsonl
    df = pd.DataFrame(df)
    datasets = df["dataset_name"].unique()
    for dataset in datasets:
        dataset_df = df[df["dataset_name"] == dataset]
        jsonl_path = destination_bucket / f"{dataset}_test.jsonl"

        dataset_df.to_json(
            str(jsonl_path),
            orient="records",
            lines=True,
        )

    # Write full dataframe as jsonl
    full_json_path = destination_bucket / "test.jsonl"
    df.to_json(
        str(full_json_path),
        orient="records",
        lines=True,
    )


if __name__ == "__main__":
    main()
