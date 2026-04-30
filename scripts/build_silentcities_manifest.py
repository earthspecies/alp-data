# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-cloud-storage",
# ]
# ///
"""Build manifest of basenames for silentcities 16kHz resampling.

Reads inference_silentcities_v1_plus_avex_filtered.csv from GCS,
optionally subtracts already-processed files, and writes a text
file of basenames (one per line, no extension).
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

from google.cloud import storage

BUCKET = "fewshot"
CSV_KEY = "data_large_clean/inference_silentcities_v1_plus_avex_filtered.csv"
DEST_PREFIX = "data_large_clean/silentcities_sampled_big_filtered_cropped_16kHz_v1_plus_avex/"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/silentcities_v1_plus_avex_manifest.txt")
    parser.add_argument("--skip-existing", action="store_true", help="Exclude files already in the 16kHz dest folder")
    parser.add_argument("--bucket", default=BUCKET)
    args = parser.parse_args()

    client = storage.Client()
    bucket = client.bucket(args.bucket)

    print(f"Downloading CSV from gs://{args.bucket}/{CSV_KEY} ...")
    csv_bytes = bucket.blob(CSV_KEY).download_as_bytes()
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8")))

    basenames = []
    for row in reader:
        gcs_path = row["gcs_path"]
        stem = Path(gcs_path).stem
        basenames.append(stem)

    print(f"Total files in filtered CSV: {len(basenames)}")

    if args.skip_existing:
        print(f"Listing existing files in gs://{args.bucket}/{DEST_PREFIX} ...")
        existing = set()
        for blob in bucket.list_blobs(prefix=DEST_PREFIX):
            existing.add(Path(blob.name).stem)
        print(f"Already processed: {len(existing)}")
        basenames = [b for b in basenames if b not in existing]
        print(f"Remaining to process: {len(basenames)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(basenames) + "\n")
    print(f"Wrote {len(basenames)} basenames to {out}")


if __name__ == "__main__":
    main()
