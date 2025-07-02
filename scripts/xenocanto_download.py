import os
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests


def download_xeno_canto_audio(urls, output_dir="audio_downloads", delay=2):
    """
    Download audio files from xeno-canto.org with respectful rate limiting.

    Args:
        urls: List of xeno-canto URLs
        output_dir: Directory to save files
        delay: Seconds to wait between downloads (minimum 2 seconds recommended)
    """

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Set up session with proper headers
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "XC downloader from ESP (contact: gagan@earthspecies.org)",
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
            if not filename or not filename.endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")):
                # Fallback filename if we can't parse it
                filename = Path(filename).stem
                filename = f"{filename}.mp3"

            filepath = os.path.join(output_dir, filename)

            # Skip if file already exists
            if os.path.exists(filepath):
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
                print(f"  Waiting {delay} seconds...")
                time.sleep(delay)

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
    df = pd.read_csv("../notebooks/dataset_prep_notebooks/xc_to_download.csv")
    audio_urls = df.identifier.tolist()

    # Check file names already downloaded
    existing_files = set(os.listdir("xeno_canto_audio"))

    audio_urls = [url for url in audio_urls if Path(url).name not in existing_files]

    # Download with 3-second delay between requests
    downloaded, failed = download_xeno_canto_audio(audio_urls, output_dir="xeno_canto_audio", delay=1.8)
