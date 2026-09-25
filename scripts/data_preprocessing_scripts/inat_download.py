import argparse
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from gcsfs import GCSFileSystem

from alp_data.io import anypath

fs = GCSFileSystem()
INAT_BUCKET = "gs://esp-ml-datasets/inaturalist/v0.1.0/raw/audio"


def get_stems_in_bucket(urls: list[str] | None = None, bucket: str = INAT_BUCKET) -> set[str]:
    """
    Get the set of file stems (filenames without extensions) already present in the
    inaturalist audio bucket.
    Returns:
        Set of file stems in the bucket
    """
    if not urls:
        # use gsutil ls to list files in the bucket
        result = subprocess.run(
            ["gsutil", "ls", INAT_BUCKET],
            capture_output=True,
            text=True,
        )
        urls = result.stdout.splitlines()
    stems = [Path(f).stem for f in urls]
    return set(stems)


def _get_fname_from_url(url: str) -> str:
    parsed = urlparse(url)
    return Path(parsed.path).stem


def which_files_to_download(
    darwin_core_path: str | Path,
    predownloaded_urls: list[str] | None = None,
    inat_bucket: str = INAT_BUCKET,
    output_jsonl: str | Path | None = None,
) -> pd.DataFrame:
    """Given a darwin core csv file and a list of already downloaded files,
    determine which files to download.
    Args:

        darwin_core_path: Path to darwin core csv file
        inat_bucket: GCS bucket path where audio files are stored
        output_jsonl: Path to output jsonl file with subset of darwin core entries to
            download

    Returns:
        DataFrame with subset of darwin core entries to download
    """
    stems_added_raw = get_stems_in_bucket(predownloaded_urls, inat_bucket)
    print(f"Num files already in bucket = {len(stems_added_raw)}")
    full_data = pd.DataFrame()

    for j, chunk in enumerate(pd.read_csv(darwin_core_path, chunksize=1_000_000)):
        print(f"Processing chunk {j}")
        chunk = chunk[chunk["type"] == "Sound"]
        dups = chunk.duplicated(subset=["identifier"])
        print(f"Num duplicate identifiers = {dups.sum()}")
        chunk = chunk.drop_duplicates(subset=["identifier"])

        ids_not_added_bool = chunk["identifier"].apply(
            lambda x: _get_fname_from_url(x) not in stems_added_raw
        )

        if ids_not_added_bool.sum() == 0:
            continue

        if full_data.empty:
            full_data = chunk.loc[ids_not_added_bool]
        else:
            full_data = pd.concat([full_data, chunk.loc[ids_not_added_bool]], axis=0)

        print(f"Current num found = {len(full_data)}")

    if not full_data.empty:
        full_data.to_json(
            output_jsonl,
            orient="records",
            lines=True,
            mode="w",
        )
    return full_data


def download_inat_audio(
    urls: list[str], save_to_bucket: bool = True, delay: float = 1.5
) -> tuple[list[Path], list[str]]:
    """
    Download audio files from inaturalist with respectful rate limiting.

    Args:
        urls: List of xeno-canto URLs
        save_to_bucket: Whether to save to GCS bucket or locally
        delay: Average seconds to wait between downloads (minimum 2 seconds recommended)

    Returns:
        Tuple of (list of successfully downloaded file paths, list of failed URLs)
    """

    # Create output directory
    if not save_to_bucket:
        Path("inat_audio_downloads").mkdir(exist_ok=True)
        output_dir = Path("inat_audio_downloads")
    else:
        output_dir = anypath("gs://esp-ml-datasets/inaturalist/v0.1.0/raw/audio")

    # Set up session with proper headers
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Audio downloader from ESP (contact: gagan@earthspecies.org)",
            "Accept": "audio/*,*/*;q=0.9",
        }
    )

    downloaded = []
    failed = []

    print(f"Preparing to download {len(urls)} files...")
    for i, url in enumerate(urls, 1):
        try:
            print(f"Downloading {i}/{len(urls)}: {url}")

            # Extract filename from URL
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)

            if not filename or not filename.endswith(
                (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mpga", ".aac", ".opus")
            ):
                # Fallback filename if we can't parse it
                filename = f"{filename}.mp3"

            filepath = output_dir / filename

            # Skip if file already exists
            if filepath.exists():
                print(f"  Skipping {filename} (already exists)")
                continue

            # Make request with timeout
            response = session.get(url, timeout=30, stream=True)
            response.raise_for_status()

            # Write file in chunks to handle large files
            file_opener = fs.open if save_to_bucket else open
            with file_opener(str(filepath), "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            downloaded.append(filepath)
            print(f"  ✓ Saved as {filename}")

            # Rate limiting - be respectful to their servers
            if i < len(urls):  # Don't sleep after the last download
                # Generate a random between delay - 0.5 and delay + 0.5
                random_delay = np.random.uniform(delay - 0.3, delay + 0.3)
                random_delay = max(random_delay, 1)  # Ensure at least 1 second delay
                print(f"  Waiting {random_delay} seconds...")
                time.sleep(random_delay)

        except requests.exceptions.RequestException as e:
            print(f"  ✗ Failed to download {url}: {e}")
            failed.append(url)
        except Exception as e:
            print(f"  ✗ Unexpected error with {url}: {e}")
            failed.append(url)

    print("\nDownload complete!")
    print(f"Successfully downloaded: {len(downloaded)} files")
    print(f"Failed downloads: {len(failed)} files")

    if failed:
        print("\nFailed URLs:")
        for url in failed:
            print(f"  {url}")

    return downloaded, failed


def main() -> None:
    """Main function to parse arguments and initiate download process."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Download audio files from iNaturalist Darwin Core CSV."
    )
    parser.add_argument(
        "-f",
        "--file_with_urls_to_download",
        type=str,
        default=None,
        help="(Optional) Path to a text file containing URLs of files to download, one "
        "per line. If not provided, will determine from Darwin Core CSV.",
    )
    parser.add_argument(
        "--darwin_core_csv",
        type=str,
        help="Path to the Darwin Core CSV file.",
    )
    parser.add_argument(
        "--file_with_predownloaded_urls",
        type=str,
        default=None,
        help="Path to a text file containing URLs of already downloaded files, one per line.",
    )
    parser.add_argument(
        "--inat_bucket",
        type=str,
        default=INAT_BUCKET,
        help="GCS bucket path to save downloaded audio files.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Average delay in seconds between download requests (minimum 1.0 recommended).",
    )
    parser.add_argument(
        "--no_bucket",
        action="store_true",
        help="If set, download files locally instead of to GCS bucket.",
    )
    args = parser.parse_args()

    if args.delay < 1.0:
        print("Warning: A delay of less than 1.0 seconds may be too aggressive.")

    if not args.file_with_urls_to_download:
        # Load pre-downloaded URLs if provided
        predownloaded_urls = None
        if args.file_with_predownloaded_urls:
            with open(args.file_with_predownloaded_urls, "r") as f:
                predownloaded_urls = [line.strip() for line in f if line.strip()]
            print(
                f"Loaded {len(predownloaded_urls)} pre-downloaded URLs"
                f"from {args.file_with_predownloaded_urls}"
            )

        # Determine which files to download
        df_to_download = which_files_to_download(
            darwin_core_path=args.darwin_core_csv,
            predownloaded_urls=predownloaded_urls,
            inat_bucket=args.inat_bucket,
            output_jsonl="darwin_core_subset_new_batch_ids.jsonl",
        )
    else:
        df_to_download = pd.read_json(args.file_with_urls_to_download, lines=True, orient="records")

    audio_urls = df_to_download["identifier"].tolist()
    print(f"Total new audio files to download: {len(audio_urls)}")
    if not audio_urls:
        print("No new files to download. Exiting.")
        exit(0)
    # Download with 3-second delay between requests
    downloaded, failed = download_inat_audio(audio_urls, not args.no_bucket, args.delay)

    # Print summary
    print("\nSummary:")
    print(f"Total files attempted: {len(audio_urls)}")
    print(f"Successfully downloaded: {len(downloaded)}")
    print(f"Failed downloads: {len(failed)}")


if __name__ == "__main__":
    main()
