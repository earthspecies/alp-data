# `esp_data.io` module

## What does it do?

The `io` module provides a set of functions for reading and writing data to and from various file formats and storage systems. It supports local files, Google Cloud Storage (GCS), and Cloudflare R2 buckets.

You can use this module to:

1.  Conduct file operations like [ "exists", "unlink" (delete), "open" (read/write), and "glob" (list files)](#useful-pathlike-methods)
2.  [Download and upload files (and folders)](#copy-from-and-to-a-cloud-location) to and from cloud storage locations
3.  [List the contents](#folder-operations) of buckets
4.  [Delete files](#delete-a-file-with-unlink)
5.  [Move files](#move-files-from-one-cloud-location-to-another-cloud-location) from one cloud location to another cloud location
6. [Copy files](#copy-from-one-cloud-location-to-another-cloud-location) from one cloud location to another cloud location
7. [Read audio](#read-audio-from-a-remote-file) from a remote file directly into memory as a numpy array. This is **NOT** the same as *streaming*, it loads the whole file into memory, so be careful with large files.

There are two interfaces available:

-   `esp_data.io.anypath` a [path-like interface](#esp_dataioanypath), if you're familiar with `pathlib` you can use this interface to read and write files.
-   `esp_data.io.filesystem` a [filesystem type interface](#filesystem-approach-with-esp_dataiofilesystem)

## Why two interfaces?
The `anypath` interface is more user-friendly and is similar to the `pathlib` module in Python. It allows you to work with paths in a more intuitive way, especially if you're already familiar with `pathlib`. However, we've noticed some issues with the underlying `cloudpathlib` library that `anypath` uses, which can lead to unexpected behavior in certain cases.

We recommend that for the following use cases, the `filesystem` interface is more reliable and consistent:

-   When you need to [copy files between cloud and local locations](#download-and-upload-files-with-filesystem) or between two cloud locations.
-   When you need to perform a search (a glob) operation for a file pattern. See [folder operations](#filesystem-folder-operations) for more details.

## `esp_data.io.anypath`

Lets start by exploring the `anypath` function



```python
from esp_data.io import anypath
```


```python
# automatically recognizes a path to a local file
path = anypath("/some/random/file.txt")
```

A local file path is a PosixPath

```python
from pathlib import PosixPath
isinstance(path, PosixPath)
# Output: True
```

You can also check if the path is local or cloud

```python
path.is_local
# Output: True
```


Cloud files are identified by their prefix: "gs://" for google cloud and "r2://" or "s3://" for cloudflare

```python
anypath("gs://some-bucket/some-file").is_cloud
# Output: True
```

!!! remark
    **Note**: Cloudflare paths are considered `S3Path` objects which might confuse you since we don't use Amazon AWS S3 buckets. The reason is that Cloudflare uses the S3 API and we haven't found a great solution to separate S3 from R2 (yet). Please consider them equivalent for now.


```python
type(anypath("r2://some-bucket/some-file"))
# Output: esp_data.io.paths.R2Path
```

```python
print(anypath("r2://some-bucket/some-file"))
# Output: s3://some-bucket/some-file
```
[Back to Top](#esp_dataio-module)

### Useful pathlike methods

```python
path = anypath("/some/random/file.txt")

# does a file exist ?
print(f"Does {path} exist? Answer: {path.exists()}")
# Output: Does /some/random/file.txt exist? Answer: False

# what is its extension ?
print(f"Suffix: {path.suffix}")
# Output: Suffix: .txt

# what is its name ?
print(f"Filename: {path.name}")
# Output: Filename: file.txt

# where is it located ?
print(f"It's here: {path.parent}")
# Output: It's here: /some/random

# split the path
print(path.parts)
# Output: ('/', 'some', 'random', 'file.txt')
```


Read a pdf as bytes using `.open()`

```python
with path.open("rb") as f:
    text = f.read()
```

Create a json file and write to it

```python
import json

with anypath("random.json").open("w") as f:
    json.dump({"data": 1}, f)

# read it back
with anypath("random.json").open("r") as f:
    data = json.load(f)

assert data == {"data": 1}
```

#### Delete a file with `.unlink()`


```python
anypath("random.json").unlink()
```

[Back to Top](#esp_dataio-module)


#### Cloud files

The above operations work identically for files in buckets.

!!! note
    One important difference between cloudpaths like `GSPath` and `S3Path` and local `PosixPath` is that buckets dont have actual "folders" like your local filesystem. So if you try to create a *local* file remember that parent folders of that file need to exist. This is **NOT** true for a file like `gs://some_bucket/some_folder/some_file.txt`, you can directly write to it with `anypath("gs://some_bucket/some_folder/some_file.txt").open("w")`.


#### Copy from and to a cloud location

Download from a cloud location to local
```python
p = anypath("gs://esp-ci-cd-tests/esp-data-tests/random2.txt").download_to("myfile.txt")
p.exists()
# Output: True
```

Copy a local file  to the <path> in anypath(<path>)
```python
p = anypath("gs://esp-ci-cd-tests/esp-data-tests/random2.txt").upload_from("myfile.txt")
p.exists()
# Output: True
```

### Folder operations

If your path is a local or cloud "folder" you can also check if its a directory and list things in it


```python
folder = anypath("r2://esp-ci-cd-tests")
folder
```
    R2Path('s3://esp-ci-cd-tests')

[Back to Top](#esp_dataio-module)


```python
folder.is_dir()
# True
folder.is_file()
# False
```

Recursively list all files with the pattern "*txt" in the name


```python
list(folder.rglob("*txt"))
```

        S3Path('s3://esp-ci-cd-tests/esp-data-tests/find_files_tests/file_find_cloud.txt'),
        S3Path('s3://esp-ci-cd-tests/esp-data-tests/temprandomfolder/file1.txt'),
        S3Path('s3://esp-ci-cd-tests/esp-data-tests/temprandomfolder/file2.txt'),
        S3Path('s3://esp-ci-cd-tests/esp-data-tests/temprandomfolder/random.txt'),
        ...



Download all the contents from the cloud "folder" to a local folder called "tempfolder"
⚠️ This will create the folder if it does not exist. If it exists, it will overwrite the contents of the folder.

```python
folder.download_to("tempfolder")
# Output: PosixPath('tempfolder')
anypath("tempfolder").exists()
# Output: True
```

[Back to Top](#esp_dataio-module)

## Filesystem approach with `esp_data.io.filesystem`

Let's now dive into the filesystem approach


```python
from esp_data.io import filesystem, filesystem_from_path
```

You can instantiate a filesystem for a local, google cloud or r2 bucket with `filesystem`

Lets start with a GCS filesystem by passing in "gcs" (or "gs") to `filesystem()`.
The same API works for local filesystem as `filesystem("local")`


```python
fs = filesystem("gcs") # "gs" also works as an alias
```

!!! tip
    Once a filesystem object has been created, you dont need to type the "gs://" prefix.

You can also create the appropriate filesystem object from a path using `filesystem_from_path`
```python
fs = filesystem_from_path("gs://esp-ci-cd-tests")
```

File operations work like so:


```python
# check for existence
fs.exists("esp-ci-cd-tests/delme1")
```


Read and write to files


```python
with fs.open("esp-ci-cd-tests/delme1", "r") as f:
    f.read()
```


```python
with fs.open("esp-ci-cd-tests/docstest","w") as f:
    f.write("hello")
```

Delete files


```python
fs.rm("esp-ci-cd-tests/docstest")
# Check if it was deleted
fs.exists("esp-ci-cd-tests/docstest")
# Output: False
```


#### Filesystem folder operations
List contents that match a pattern like '*txt' under a path


```python
fs.glob("esp-ci-cd-tests/**/*txt")
```

        'esp-ci-cd-tests/esp-data-tests/find_files_tests/file_find_cloud.txt',
        'esp-ci-cd-tests/esp-data-tests/non_empty/temp_file.txt',
        'esp-ci-cd-tests/esp-data-tests/random2.txt',
        'esp-ci-cd-tests/random.txt',
        ...


[Back to Top](#esp_dataio-module)


#### Download and upload files with `filesystem`
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


#### Move files from one cloud location to another cloud location.
📝 In reality this just renames the file in the bucket.


```python
fs.mv("esp-ci-cd-tests/foo.txt", "esp-ci-cd-tests/foo2.txt")
fs.exists("esp-ci-cd-tests/foo.txt")
# Output: False
```


#### Copy from one cloud location to another cloud location


!!! warning
    Copy is only supported when the source and destination cloud location are on the same cloud provider. For example, you cannot copy from a GCS bucket to an R2 bucket.

```python
fs.copy("esp-ci-cd-tests/foo2.txt", "esp-ci-cd-tests/foo3.txt")
fs.exists("esp-ci-cd-tests/foo3.txt")
# Output: True
```

[Back to Top](#esp_dataio-module)

## Read audio from a remote file

```python
from esp_data.io import read_audio

audio, sample_rate = read_audio("gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/nri-battlesounds.mp3")
print(audio.shape, sample_rate)
# (235008,) 44100
```
