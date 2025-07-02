import os
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from gcsfs import GCSFileSystem

fs = GCSFileSystem()


def download_inat_audio(urls, delay=2):
    """
    Download audio files from inaturalist with respectful rate limiting.

    Args:
        urls: List of xeno-canto URLs
        delay: Average seconds to wait between downloads (minimum 2 seconds recommended)
    """

    # Create output directory
    # Path(output_dir).mkdir(exist_ok=True)
    Path("inat_audio_downloads").mkdir(exist_ok=True)
    output_dir = Path("inat_audio_downloads")
    # output_dir = GSPath("gs://esp-ml-datasets/inaturalist/v0.1.0/raw/audio")

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
            with open(filepath, "wb") as f:
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


# Example usage
if __name__ == "__main__":
    df = pd.read_json(
        "inat_to_download.jsonl",
        lines=True,
        orient="records",
    )
    # Extract audio URLs from the DataFrame
    audio_urls = df.identifier.tolist()

    # Run a gsutil command to list existing files
    # This assumes you have gsutil installed and configured
    # os.system(
    #     "gsutil ls gs://esp-ml-datasets/inaturalist/v0.1.0/raw/audio > inat_files_downloaded.txt"
    # )
    with open("inat_files_downloaded.txt", "r") as f:
        existing_files = f.read().splitlines()
        existing_files = set([Path(url).name for url in existing_files])

    # check files downloaded to inat_audio_downloads
    existing_files_local = set(os.listdir("inat_audio_downloads"))

    # Combine existing files from both sources
    existing_files = existing_files.union(existing_files_local)

    # Filter out already downloaded files
    audio_urls = [url for url in audio_urls if os.path.basename(urlparse(url).path) not in existing_files]

    # Download with 3-second delay between requests
    downloaded, failed = download_inat_audio(audio_urls, delay=1.5)
