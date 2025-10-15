#!/usr/bin/env python3
"""Example usage of the PureGCSPath class."""

import os
import sys

# Add the parent directory to the path so we can import esp_data
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from esp_data.io.gcs_paths import PureGCSPath, gcs_path


def main():
    """Demonstrate PureGCSPath functionality."""
    print("=== PureGCSPath Examples ===\n")

    # Example 1: Basic path creation and properties
    print("1. Basic path creation and properties:")
    path = PureGCSPath("gs://my-data-bucket/datasets/images/cat.jpg")
    print(f"   Path: {path}")
    print(f"   Bucket: {path.bucket}")
    print(f"   Object path: {path.object_path}")
    print(f"   Filename: {path.name}")
    print(f"   Extension: {path.suffix}")
    print(f"   Stem: {path.stem}")
    print(f"   Parent: {path.parent}")
    print()

    # Example 2: Path manipulation
    print("2. Path manipulation:")
    new_bucket_path = path.with_bucket("new-bucket")
    print(f"   Original: {path}")
    print(f"   New bucket: {new_bucket_path}")

    relative_path = path.relative_to_bucket()
    print(f"   Relative to bucket: {relative_path}")
    print()

    # Example 3: Path joining
    print("3. Path joining:")
    base_path = PureGCSPath("gs://my-bucket/folder")
    joined_path = base_path / "subfolder" / "file.txt"
    print(f"   Base: {base_path}")
    print(f"   Joined: {joined_path}")

    # Join with absolute path
    absolute_joined = base_path / "/absolute/path/file.txt"
    print(f"   Absolute join: {absolute_joined}")
    print()

    # Example 4: Convenience function
    print("4. Convenience function:")
    simple_path = gcs_path("gs://simple-bucket/file.txt")
    print(f"   Created with gcs_path(): {simple_path}")
    print()

    # Example 5: Error handling
    print("5. Error handling:")
    try:
        invalid_path = PureGCSPath("/local/path")
    except ValueError as e:
        print(f"   Error caught: {e}")

    try:
        invalid_path = PureGCSPath("s3://bucket/file")
    except ValueError as e:
        print(f"   Error caught: {e}")

    try:
        invalid_path = PureGCSPath(123)
    except TypeError as e:
        print(f"   Error caught: {e}")
    print()

    # Example 6: Path comparison
    print("6. Path comparison:")
    path1 = PureGCSPath("gs://bucket/folder/file1.txt")
    path2 = PureGCSPath("gs://bucket/folder/file2.txt")
    path3 = PureGCSPath("gs://bucket/folder/file1.txt")

    print(f"   {path1} == {path3}: {path1 == path3}")
    print(f"   {path1} == {path2}: {path1 == path2}")
    print(f"   {path1} < {path2}: {path1 < path2}")
    print()

    # Example 7: Edge cases
    print("7. Edge cases:")
    edge_path = PureGCSPath("gs://bucket//folder///file.txt")
    print(f"   Multiple slashes: {edge_path}")
    print(f"   Normalized object path: {edge_path.object_path}")

    trailing_slash = PureGCSPath("gs://bucket/folder/")
    print(f"   Trailing slash: {trailing_slash}")
    print(f"   Object path: {trailing_slash.object_path}")
    print(f"   Name: {trailing_slash.name}")


if __name__ == "__main__":
    main()
