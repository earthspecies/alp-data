# `alp_data.io` module

## What does it do?

The `io` module provides a set of functions for reading and writing data to and from various file formats and storage systems. It supports local files, Google Cloud Storage (GCS), and Cloudflare R2 buckets.

!!! warning
    We have dropped support for `cloudpathlib` which was the basis for `anypath` until version 1.2.1. We now have our own implementation of a "cloud path" (e.g. "gs://my-bucket/file.txt"). This implementation `PureGSPath` or `PureS3Path` is more lightweight and provides **path manipulation features only**. For file operations on cloud paths (reading, writing, copying, listing files), you must use the `filesystem` interface.


You can use this module to:

1. [Path manipulation with `anypath`](#alp_dataioanypath) - join paths, get path components, change extensions, etc.
2. [Download and upload files (and folders)](#download-and-upload-files-with-filesystem) to and from cloud storage locations
3. [List the contents](#filesystem-folder-operations) of buckets
4. [Delete files](#delete-files) with filesystem operations
5. [Move files](#move-files-from-one-cloud-location-to-another-cloud-location) from one cloud location to another cloud location
6. [Copy files](#copy-from-one-cloud-location-to-another-cloud-location) from one cloud location to another cloud location
7. [Read audio](#read-audio-from-a-remote-file) from a remote file directly into memory as a numpy array. This is **NOT** the same as *streaming*, it loads the whole file into memory, so be careful with large files.
8. [Get audio info](#get-info-about-an-audio-file-before-reading) about an audio file before reading it, such as sample rate, duration, number of channels, etc.

There are two interfaces available:

-   `alp_data.io.anypath` a [path-like interface](#alp_dataioanypath), if you're familiar with `pathlib` you can use this interface for path manipulation operations, like joining paths, listing path parts, path parents etc. **Note**: For cloud paths (gs://, s3://), this interface provides path manipulation only, not file I/O. For local paths, it returns `pathlib.Path` which supports full file I/O operations.
-   `alp_data.io.filesystem` a [filesystem type interface](#filesystem-approach-with-alp_dataiofilesystem) this is the interface to use for reading and writing files on cloud storage, copying files between cloud and local locations, listing files in buckets etc.

## Why two interfaces?
The `anypath` factory function is user-friendly and returns the appropriate path type:
- For **local paths**: Returns `pathlib.Path` which supports both path manipulation AND file I/O operations (`.exists()`, `.open()`, `.unlink()`, `.is_dir()`, etc.)
- For **cloud paths** (gs://, s3://, r2://): Returns `PureCloudPath` instances (like `PureGSPath`) which support path manipulation only, NOT file I/O.

The `filesystem` interface is required for cloud file operations:

-   When you need to [copy files between cloud and local locations](#download-and-upload-files-with-filesystem) or between two cloud locations.
-   When you need to perform a search (a glob) operation for files in cloud storage. See [folder operations](#filesystem-folder-operations) for more details.
-   When you need to check if a cloud file exists, delete cloud files, or read/write cloud file contents.

## `alp_data.io.anypath`

Let's start by exploring the `anypath` function for path manipulation.

```python
from alp_data.io import anypath
```

```python
# automatically recognizes a path to a local file
path = anypath("/some/random/file.txt")
```

A local file path is a Path (PosixPath on Unix systems) and supports full file I/O operations:

```python
from pathlib import Path
isinstance(path, Path)
# Output: True

# Local paths support file I/O operations
# Check if file exists
path.exists()
# Output: False

# Create and write to a file
import json
json_path = anypath("random.json")
with json_path.open("w") as f:
    json.dump({"data": 1}, f)

# Read it back
with json_path.open("r") as f:
    data = json.load(f)

# Delete the file
json_path.unlink()
```

Cloud files are identified by their prefix: "gs://" for Google Cloud and "r2://" or "s3://" for Cloudflare

```python
from alp_data.io import PureGSPath
gs_path = anypath("gs://some-bucket/some-file")
isinstance(gs_path, PureGSPath)
# Output: True
```

!!! important
    **Cloud paths do NOT support file I/O operations**. The following will NOT work:

    ```python
    # ❌ These operations are NOT available for cloud paths
    gs_path = anypath("gs://bucket/file.txt")
    gs_path.exists()  # AttributeError: no 'exists' method
    gs_path.open("r")  # AttributeError: no 'open' method
    gs_path.unlink()  # AttributeError: no 'unlink' method
    gs_path.is_dir()  # AttributeError: no 'is_dir' method
    ```

    For file operations on cloud paths, use the [`filesystem` interface](#filesystem-approach-with-alp_dataiofilesystem) or standalone utility functions like `alp_data.io.exists()` and `alp_data.io.rm()`.

!!! remark
    **Note**: Cloudflare paths use the "r2://" prefix which gets converted to "s3://" internally because Cloudflare R2 uses the S3 API. The returned objects are `PureR2Path` (or `PureS3Path`) instances.

```python
from alp_data.io import PureR2Path
r2_path = anypath("r2://some-bucket/some-file")
isinstance(r2_path, PureR2Path)
# Output: True
print(r2_path)
# Output: s3://some-bucket/some-file
```

[Back to Top](#alp_dataio-module)

### Path manipulation methods

The `PureCloudPath` classes (and standard `Path` for local paths) provide these path manipulation methods:

#### Basic path properties

```python
path = anypath("/some/random/file.txt")

# what is its extension?
print(f"Suffix: {path.suffix}")
# Output: Suffix: .txt

# all extensions (for files like .tar.gz)
print(f"Suffixes: {path.suffixes}")
# Output: Suffixes: ['.txt']

# what is its name?
print(f"Filename: {path.name}")
# Output: Filename: file.txt

# name without extension
print(f"Stem: {path.stem}")
# Output: Stem: file

# where is it located?
print(f"Parent: {path.parent}")
# Output: Parent: /some/random

# split the path into components
print(f"Parts: {path.parts}")
# Output: Parts: ('/', 'some', 'random', 'file.txt')
```

#### Working with cloud paths

```python
gs_path = anypath("gs://my-bucket/folder/file.txt")

# get the bucket name
print(f"Bucket: {gs_path.bucket}")
# Output: Bucket: my-bucket

# get the drive (scheme + bucket)
print(f"Drive: {gs_path.drive}")
# Output: Drive: gs://my-bucket

# get the anchor (drive + root)
print(f"Anchor: {gs_path.anchor}")
# Output: Anchor: gs://my-bucket/

# get path components
print(f"Parts: {gs_path.parts}")
# Output: Parts: ('gs://my-bucket/', 'folder', 'file.txt')

# access parent directories
print(f"Parent: {gs_path.parent}")
# Output: Parent: gs://my-bucket/folder

# access multiple parent levels
print(f"Grandparent: {gs_path.parents[1]}")
# Output: Grandparent: gs://my-bucket/
```

#### Joining paths

```python
base = anypath("gs://my-bucket/folder")

# join with / operator
full_path = base / "subfolder" / "file.txt"
print(full_path)
# Output: gs://my-bucket/folder/subfolder/file.txt

# join with joinpath method
full_path = base.joinpath("subfolder", "file.txt")
print(full_path)
# Output: gs://my-bucket/folder/subfolder/file.txt
```

#### Changing path components

```python
path = anypath("gs://my-bucket/folder/file.txt")

# change the filename
new_path = path.with_name("newfile.txt")
print(new_path)
# Output: gs://my-bucket/folder/newfile.txt

# change the extension
new_path = path.with_suffix(".json")
print(new_path)
# Output: gs://my-bucket/folder/file.json

# change the stem (filename without extension)
new_path = path.with_stem("renamed")
print(new_path)
# Output: gs://my-bucket/folder/renamed.txt
```

#### Path matching

```python
path = anypath("gs://my-bucket/data/2024/file.txt")

# match against a pattern
print(path.match("*.txt"))
# Output: True

print(path.match("data/*.txt"))
# Output: False (doesn't match the full path structure)

print(path.match("**/file.txt"))
# Output: True (** matches any number of directories)
```

#### Converting to string representations

```python
path = anypath("gs://my-bucket/folder/file.txt")

# convert to string
str_path = str(path)
# Output: 'gs://my-bucket/folder/file.txt'

# get as URI (only for absolute paths)
uri = path.as_uri()
# Output: 'gs://my-bucket/folder/file.txt'

# get as posix-style string (always forward slashes)
posix_path = path.as_posix()
# Output: 'gs://my-bucket/folder/file.txt'

# check if path is absolute
print(path.is_absolute())
# Output: True
```

[Back to Top](#alp_dataio-module)

### Summary: anypath behavior

!!! summary
    **Key differences between local and cloud paths returned by `anypath`:**

    | Operation | Local Path (`Path`) | Cloud Path (`PureGSPath`, `PureS3Path`, etc.) |
    |-----------|---------------------|------------------------------------------------|
    | Path manipulation (`.name`, `.parent`, `.suffix`, `/`, `.with_name()`, etc.) | ✅ Yes | ✅ Yes |
    | File I/O (`.exists()`, `.open()`, `.unlink()`, `.is_dir()`, etc.) | ✅ Yes | ❌ No - Use `filesystem` |
    | `.bucket` property | N/A | ✅ Yes |

    **For cloud file operations**, use:
    - The [`filesystem` interface](#filesystem-approach-with-alp_dataiofilesystem) for most operations
    - `alp_data.io.exists(path)` to check if a file exists
    - `alp_data.io.rm(path)` to delete a file
    - `alp_data.io.read_audio(path)` to read audio files

!!! note
    Cloud storage systems like GCS and S3 don't have true "folders" like local filesystems. They use prefixes to simulate folder structures. When working with paths, remember that "gs://bucket/folder/file.txt" is really just a single object with a name that contains forward slashes.

[Back to Top](#alp_dataio-module)

## Filesystem approach with `alp_data.io.filesystem`

Let's now dive into the filesystem approach for file operations.

```python
from alp_data.io import filesystem, filesystem_from_path
```

You can instantiate a filesystem for a local, Google Cloud or R2 bucket with `filesystem`

Let's start with a GCS filesystem by passing in "gcs" (or "gs") to `filesystem()`.
The same API works for local filesystem as `filesystem("local")`

```python
fs = filesystem("gcs") # "gs" also works as an alias
```

!!! tip
    Once a filesystem object has been created, you don't need to type the "gs://" prefix.

You can also create the appropriate filesystem object from a path using `filesystem_from_path`
```python
fs = filesystem_from_path("gs://esp-ci-cd-tests")
```

### File operations with filesystem

#### Check if files exist

```python
# check for existence
fs.exists("esp-ci-cd-tests/delme1")
```

#### Read and write to files

```python
with fs.open("esp-ci-cd-tests/delme1", "r") as f:
    content = f.read()
```

```python
with fs.open("esp-ci-cd-tests/docstest", "w") as f:
    f.write("hello")
```

#### Delete files

```python
fs.rm("esp-ci-cd-tests/docstest")
# Check if it was deleted
fs.exists("esp-ci-cd-tests/docstest")
# Output: False
```

You can also use the standalone utility functions:

```python
from alp_data.io import exists, rm

# check existence
exists("gs://esp-ci-cd-tests/docstest")

# delete a file
rm("gs://esp-ci-cd-tests/docstest")
```

[Back to Top](#alp_dataio-module)

### Filesystem folder operations

List contents that match a pattern like '*txt' under a path

```python
fs.glob("esp-ci-cd-tests/**/*txt")
```

Output:
```
'esp-ci-cd-tests/esp-data-tests/find_files_tests/file_find_cloud.txt',
'esp-ci-cd-tests/esp-data-tests/non_empty/temp_file.txt',
'esp-ci-cd-tests/esp-data-tests/random2.txt',
'esp-ci-cd-tests/random.txt',
...
```

[Back to Top](#alp_dataio-module)

### Download and upload files with `filesystem`

Download files from a cloud location to local. 📝 `get` and `put` return `None` if the operation was successful.

```python
fs.get("esp-ci-cd-tests/foo.txt", "foofoo.txt")
# Output: [None]
```

Upload files from local to a cloud location.

```python
fs.put("foofoo.txt", "esp-ci-cd-tests/foo.txt")
# Output: [None]
```

[Back to Top](#alp_dataio-module)

### Move files from one cloud location to another cloud location

📝 In reality this just renames the file in the bucket.

```python
fs.mv("esp-ci-cd-tests/foo.txt", "esp-ci-cd-tests/foo2.txt")
fs.exists("esp-ci-cd-tests/foo.txt")
# Output: False
```

### Copy from one cloud location to another cloud location

!!! warning
    Copy is only supported when the source and destination cloud location are on the same cloud provider. For example, you cannot copy from a GCS bucket to an R2 bucket.

```python
fs.copy("esp-ci-cd-tests/foo2.txt", "esp-ci-cd-tests/foo3.txt")
fs.exists("esp-ci-cd-tests/foo3.txt")
# Output: True
```

[Back to Top](#alp_dataio-module)

## Read audio from a remote file

```python
from alp_data.io import read_audio

audio, sample_rate = read_audio("gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/nri-battlesounds.mp3")
print(audio.shape, sample_rate)
# (235008,) 44100
```

## Get info about an audio file before reading

```python
from alp_data.io import get_audio_info

info = get_audio_info("gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/nri-battlesounds.mp3")
print(info.keys())
# dict_keys(['sr', 'duration', 'num_frames', 'num_channels', 'format', 'subtype'])
```

[Back to Top](#alp_dataio-module)
