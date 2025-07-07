"""Resample new xeno-canto files and copy them to target"""

import argparse

import librosa
import pandas as pd
import soundfile as sf

from esp_data.io import anypath, audio_stereo_to_mono, read_audio


def main() -> None:
    """Resample new xeno-canto files and copy them to target directory."""

    parser = argparse.ArgumentParser(
        description="Resample new xeno-canto files and copy them to target directory."
    )
    parser.add_argument(
        "--filepaths_to_copy",
        type=str,
        required=True,
        help="Path to the CSV file containing file paths to copy.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Target directory to copy the resampled files to.",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=16000,
        help="Sample rate to which the audio files should be resampled.",
    )

    args = parser.parse_args()
    target_dir = anypath(args.target_dir)

    # Read the CSV file containing file paths to copy
    df = pd.read_csv(args.filepaths_to_copy)
    df = df.dropna(subset=["files"])

    print(f"Found {len(df)} files to process.")
    for i, row in df.iterrows():
        file_path = row["files"]
        print(f"Processing file number {i}/{len(df)}: {file_path}")

        # Read the audio file
        data, sr = read_audio(file_path)

        data = audio_stereo_to_mono(data, mono_method="average").squeeze()

        # Resample if necessary
        if sr != args.target_sr:
            data = librosa.resample(
                y=data,
                orig_sr=sr,
                target_sr=args.target_sr,
                scale=True,
                res_type="kaiser_best",
            )

        # Define the target path
        target_path = target_dir / (anypath(file_path).stem + ".flac")

        # Write the resampled audio as flac file
        with target_path.open("wb") as f:
            sf.write(f, data, args.target_sr, format="FLAC")

        print(f"Copied and resampled {file_path} to {target_path}")


if __name__ == "__main__":
    main()
