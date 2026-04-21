"""Resample new inaturalist files and copy them to target"""

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from esp_data.io import anypath

# Suppress warnings from librosa about resampling
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


def process_single_file(file_path_str: str, target_dir_str: str, target_sr: int) -> dict:
    """Process a single audio file - resample and copy.


    Parameters
    ----------
    file_path_str : str
        Path to the audio file to process.
    target_dir_str : str
        Directory where the processed file should be saved.
    target_sr : int
        Target sample rate to which the audio file should be resampled.

    Returns
    -------
    dict
        A dictionary containing the status of the operation, the file name,
        and any messages (e.g., error messages, skip reasons).
    """
    # Re-import inside the function to avoid pickling issues
    from esp_data.io import anypath, audio_stereo_to_mono, read_audio

    file_path = anypath(file_path_str)
    target_dir = anypath(target_dir_str)

    # Define the target path
    target_path = target_dir / (file_path.stem + ".flac")

    if target_path.exists():
        return {
            "status": "skipped",
            "file": file_path.name,
            "message": "already exists",
        }

    try:
        # Read the audio file
        try:
            data, sr = read_audio(file_path)
        except Exception:
            # If fail, try again with librosa
            try:
                data, sr = librosa.load(file_path, sr=None, mono=False)
            except Exception as e2:
                return {"status": "error", "file": str(file_path), "message": str(e2)}

        data = audio_stereo_to_mono(data, mono_method="average").squeeze()

        # Resample if necessary
        if sr != target_sr:
            data = librosa.resample(
                y=data,
                orig_sr=sr,
                target_sr=target_sr,
                scale=True,
                res_type="kaiser_best",
            )

        # Write the resampled audio as flac file
        with target_path.open("wb") as f:
            sf.write(f, data, target_sr, format="FLAC")

        return {"status": "success", "file": file_path.name, "target": str(target_path)}

    except Exception as e:
        return {"status": "error", "file": str(file_path), "message": str(e)}


def main() -> None:
    """Resample new inaturalist files and copy them to target directory."""

    parser = argparse.ArgumentParser(
        description="Resample new inaturalist files and copy them to target directory."
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
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,  # Conservative default
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress for each file.",
    )

    args = parser.parse_args()
    target_dir = anypath(args.target_dir)

    # Read the CSV file containing file paths to copy
    df = pd.read_csv(args.filepaths_to_copy)
    df = df.dropna(subset=["files"])

    print(f"Found {len(df)} files to process using {args.n_workers} workers.")

    # Get list of file paths as strings
    file_paths = df["files"].tolist()

    # Process files in parallel
    errors = []
    successful = 0
    skipped = 0

    print("Starting parallel processing...")

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                process_single_file, file_path, str(target_dir), args.target_sr
            ): file_path
            for file_path in file_paths
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(file_paths), desc="Processing files", unit="file") as pbar:
            for future in as_completed(future_to_file):
                try:
                    result = future.result()

                    if result["status"] == "success":
                        successful += 1
                        pbar.set_postfix({"✓": successful, "⚠": skipped, "✗": len(errors)})
                        if args.verbose:
                            tqdm.write(f"✓ Processed: {result['file']}")
                    elif result["status"] == "skipped":
                        skipped += 1
                        pbar.set_postfix({"✓": successful, "⚠": skipped, "✗": len(errors)})
                        if args.verbose:
                            tqdm.write(f"⚠ Skipped: {result['file']} - {result['message']}")
                    elif result["status"] == "error":
                        errors.append((result["file"], result["message"]))
                        pbar.set_postfix({"✓": successful, "⚠": skipped, "✗": len(errors)})
                        if args.verbose:
                            tqdm.write(f"✗ Error: {result['file']} - {result['message']}")
                        # Show last error in description
                        pbar.set_description(f"Processing files (last error: {result['file']})")

                except Exception as e:
                    file_path = future_to_file[future]
                    errors.append((file_path, str(e)))
                    pbar.set_postfix({"✓": successful, "⚠": skipped, "✗": len(errors)})
                    if args.verbose:
                        tqdm.write(f"✗ Unexpected error with {file_path}: {e}")

                pbar.update(1)

    print("\nProcessing complete:")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {len(errors)}")

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
