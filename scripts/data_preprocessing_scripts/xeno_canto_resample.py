"""Resample new xeno-canto files and copy them to target"""

import argparse
import warnings

import librosa
import pandas as pd
import soundfile as sf

from alp_data.io import anypath, audio_stereo_to_mono, read_audio

# Suppress warnings from librosa about resampling
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


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
    copied = []

    print(f"Found {len(df)} files to process.")
    for i, row in df.iterrows():
        file_path = anypath(row["files"])

        # Define the target path
        target_path = target_dir / (file_path.stem + ".flac")

        if target_path.exists():
            print(f"Skipping {file_path.name}, already exists at {target_path}")
            copied.append(target_path)
            continue

        print(f"Processing file number {i}/{len(df)}: {file_path.name}")

        try:
            # Read the audio file
            try:
                data, sr = read_audio(file_path)

            # if fail, try again with librosa
            except Exception:
                try:
                    data, sr = librosa.load(file_path, sr=None, mono=False)
                except Exception as e:
                    print(f"Error reading {file_path.name} with librosa: {e}")
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
            try:
                with target_path.open("wb") as f:
                    sf.write(f, data, args.target_sr, format="FLAC")
                copied.append(target_path)
            except OSError:
                # shorten the name and try again
                new_target_path = target_dir / (file_path.stem[:50] + ".flac")
                try:
                    with new_target_path.open("wb") as f:
                        sf.write(f, data, args.target_sr, format="FLAC")
                    copied.append(new_target_path)

                except Exception as err:
                    print(f"Error writing {target_path.name}: {err}")
                    errors.append((file_path, str(err)))
                    continue

            print(f"Copied and resampled {file_path}")

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            errors.append((file_path, str(e)))
            continue

    # Save errors to a CSV file
    if errors:
        error_df = pd.DataFrame(errors, columns=["file_path", "error"])
        error_file = "resample_and_copy_errors.csv"
        error_df.to_csv(error_file, index=False)
        print(f"Errors encountered during processing saved to {error_file}")
    else:
        print("All files processed successfully with no errors.")

    # Save copied files to a CSV file
    if copied:
        copied_df = pd.DataFrame(copied, columns=["file_path"])
        copied_file = "resampled_files_copied.csv"
        copied_df.to_csv(copied_file, index=False)
        print(f"Copied files saved to {copied_file}")
    else:
        print("No files were copied.")


if __name__ == "__main__":
    main()
