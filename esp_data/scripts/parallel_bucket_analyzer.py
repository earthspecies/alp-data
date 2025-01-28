import itertools
import re
import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Iterator, List, NamedTuple

from google.cloud import storage

from bucket_analyzer import print_analysis


class FileMetadata(NamedTuple):
    """Container for file metadata"""

    name: str
    size: int
    extension: str
    path: str
    updated: datetime


class BucketAnalyzer:
    def __init__(self, bucket_name: str, max_workers: int = 4, batch_size: int = 1000):
        """
        Initialize analyzer with better performance characteristics.

        Args:
            bucket_name: Name of the GCS bucket (without gs:// prefix)
            max_workers: Number of parallel workers
            batch_size: Size of batches for processing
        """
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.max_workers = max_workers
        self.batch_size = batch_size

        # Compile patterns once
        self.version_pattern = re.compile(r"v\d+(\.\d+)*")
        self.date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})|(\d{8})|(\d{2}-\d{2}-\d{4})|(\d{2}_\d{2}_\d{4})")

    def get_files_batch(self, prefix: str = "") -> Iterator[List[storage.Blob]]:
        """Stream files in batches to conserve memory."""
        iterator = self.bucket.list_blobs(prefix=prefix)
        while True:
            batch = list(itertools.islice(iterator, self.batch_size))
            if not batch:
                break
            yield batch

    def analyze_blob(self, blob: storage.Blob) -> Dict:
        """Analyze a single blob for all required metrics."""
        name = blob.name
        extension = blob.name.split(".")[-1].lower() if "." in blob.name else "no_extension"

        return {
            "metadata": FileMetadata(
                name=name,
                size=blob.size,
                extension=extension,
                path="/".join(name.split("/")[:-1]),
                updated=blob.updated,
            ),
            "is_readme": name.lower() == "readme" or name.lower().endswith("readme.md"),
            "has_doc": "doc" in name.lower(),
            "is_versioned": bool(self.version_pattern.search(name)),
            "has_date": bool(self.date_pattern.search(name)),
        }

    def parallel_analyze_batch(self, blobs: List[storage.Blob]) -> List[Dict]:
        """Analyze a batch of blobs in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.analyze_blob, blob) for blob in blobs]
            return [future.result() for future in as_completed(futures)]

    async def analyze_bucket(self) -> Dict:
        """
        Analyze bucket with improved performance.
        - Streams files in batches
        - Processes batches in parallel
        - Uses native GCS APIs for better performance
        """
        # Initialize aggregation containers
        results = {
            "total_size": 0,
            "file_count": 0,
            "extensions": defaultdict(int),
            "directories": set(),
            "readmes": [],
            "doc_files": [],
            "versioned_files": [],
            "dated_files": [],
            "name_lengths": [],
        }

        # Get bucket metadata efficiently
        bucket_metadata = self.bucket.metadata
        results["is_public"] = bool(bucket_metadata.get("iam_config", {}).get("publicAccessPrevention"))

        # Process files in streaming batches
        for batch in self.get_files_batch():
            batch_analysis = self.analyze_batch(batch)

            # Aggregate batch results
            for analysis in batch_analysis:
                metadata = analysis["metadata"]

                # Update counters
                results["total_size"] += metadata.size
                results["file_count"] += 1
                results["extensions"][metadata.extension] += 1
                results["name_lengths"].append(len(metadata.name))

                # Add directory
                if "/" in metadata.path:
                    results["directories"].add(metadata.path)

                # Track special files
                if analysis["is_readme"]:
                    results["readmes"].append(metadata.name)
                if analysis["has_doc"]:
                    results["doc_files"].append(metadata.name)
                if analysis["is_versioned"]:
                    results["versioned_files"].append(metadata.name)
                if analysis["has_date"]:
                    results["dated_files"].append(metadata.name)

        # Calculate final statistics
        name_lengths = results["name_lengths"]
        results["filename_analysis"] = {
            "min_length": min(name_lengths) if name_lengths else 0,
            "max_length": max(name_lengths) if name_lengths else 0,
            "avg_length": sum(name_lengths) / len(name_lengths) if name_lengths else 0,
        }

        results["directory_count"] = len(results["directories"])

        return results


async def analyze_buckets(bucket_names: List[str], **kwargs) -> Dict[str, Dict]:
    """Analyze multiple buckets efficiently."""
    results = {}

    for bucket_name in bucket_names:
        try:
            analyzer = BucketAnalyzer(bucket_name, **kwargs)
            results[bucket_name] = await analyzer.analyze_bucket()
        except Exception as e:
            results[bucket_name] = {"error": str(e)}

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze GCS buckets efficiently.")
    parser.add_argument("bucket_names", nargs="+", help="Names of the GCS buckets to analyze")

    args = parser.parse_args()
    # results = await analyze_buckets(args.bucket_names, max_workers=4, batch_size=1000)
    results = asyncio.run(analyze_buckets(args.bucket_names, max_workers=4, batch_size=1000))

    # Print results
    for bucket_name, result in results.items():
        print(f"Analysis for bucket: {bucket_name}\n")
        # print_analysis(result)
        print(result)
        print("--------------------")
