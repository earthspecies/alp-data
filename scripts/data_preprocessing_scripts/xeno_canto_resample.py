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
    errors = []

    print(f"Found {len(df)} files to process.")
    for i, row in df.iterrows():
        file_path = row["files"]

        # Define the target path
        target_path = target_dir / (anypath(file_path).stem + ".flac")

        if target_path.exists():
            print(f"Skipping {file_path}, already exists at {target_path}")
            continue

        print(f"Processing file number {i}/{len(df)}: {file_path}")

        # Read the audio file
        try:
            data, sr = read_audio(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            errors.append((file_path, str(e)))
            continue

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

        # Write the resampled audio as flac file
        with target_path.open("wb") as f:
            sf.write(f, data, args.target_sr, format="FLAC")

        print(f"Copied and resampled {file_path} to {target_path}")

    # Save errors to a CSV file
    if errors:
        error_df = pd.DataFrame(errors, columns=["file_path", "error"])
        error_file = target_dir / "resample_and_copy_errors.csv"
        error_df.to_csv(error_file, index=False)
        print(f"Errors encountered during processing saved to {error_file}")
    else:
        print("All files processed successfully with no errors.")


if __name__ == "__main__":
    main()
