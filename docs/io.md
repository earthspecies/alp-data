# The `esp_data.io` module

The `io` module provides a set of functions for reading and writing data to and from various file formats and storage systems. It supports local files, Google Cloud Storage (GCS), and Cloudflare R2 buckets.

You can use this module to:

1.  Conduct file operations like "exists", "unlink" (delete), "open" (read/write), and "glob" (list files).
2.  Download and upload files (and folders) to and from cloud storage locations
3.  List the contents of buckets
4.  Delete files
5.  Read audio from a remote file directly into memory as a numpy array

There are two interfaces available:

-   `esp_data.io.anypath` a path-like interface, if you're familiar with `pathlib` you can use this interface to read and write files.
-   `esp_data.io.filesystem` a filesystem type interface

## `esp_data.io.anypath`

Lets start by exploring the `anypath` function



```python
from esp_data.io import anypath
```


```python
# automatically recognizes a path to a local file
path = anypath("/some/random/file.txt")
```


```python
from pathlib import PosixPath
isinstance(path, PosixPath)
```
    True



A local file path is a PosixPath


```python
path.is_local
```
    True


Cloud files are identified by their prefix: "gs://" for google cloud and "r2://" or "s3://" for cloudflare

```python
anypath("gs://some-bucket/some-file").is_cloud
```
    True



**Note**: Cloudflare paths are considered `S3Path` objects which might confuse you since we don't use Amazon AWS S3 buckets. The reason is that Cloudflare uses the S3 api and we haven't found a great solution for it yet.


```python
type(anypath("r2://some-bucket/some-file"))
```
    esp_data.io.paths.R2Path



### Useful pathlike methods

```python
# does a file exist ?
print(f"Does {path} exist? Answer: {path.exists()}")

# what is its extension ?
print(f"Suffix: {path.suffix}")

# what is its name ?
print(f"Filename: {path.name}")

# where is it located ?
print(f"It's here: {path.parent}")

# split the path
print(path.parts)
```
    Does /some/random/file.txt exist? Answer: False
    Suffix: .txt
    Filename: file.txt
    It's here: /some/random
    ('/', 'some', 'random', 'file.txt')


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

Delete a file with `.unlink()`


```python
anypath("random.json").unlink()
```

#### Cloud files

The above operations work identically for files in buckets.

⚠️ One important difference between cloudpaths like `GSPath` and `S3Path` and local `PosixPath` is that buckets dont have actual "folders" like your local filesystem. So if you try to create a *local* file remember that parent folders of that file need to exist. This **NOT** true for a file like `gs://some_bucket/some_folder/some_file.txt`, you can directly write to it with `anypath("gs://some_bucket/some_folder/some_file.txt").open("w")`.


#### Copy from and to a cloud location


```python
p = anypath("gs://esp-ci-cd-tests/esp-data-tests/random2.txt").download_to("myfile.txt")
p.exists()
```
    True


```python
# copy a local file  to the <path> in anypath(<path>)
p = anypath("gs://esp-ci-cd-tests/esp-data-tests/random2.txt").upload_from("myfile.txt")
```

### Folder operations

If your path is a local or cloud "folder" you can also check if its a directory and list things in it


```python
folder = anypath("r2://esp-ci-cd-tests/temprandomfolder")
folder
```
    R2Path('s3://esp-ci-cd-tests')




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
    [S3Path('s3://esp-ci-cd-tests/esp-data-tests/find_files_tests/file_find_cloud.txt'),
     S3Path('s3://esp-ci-cd-tests/esp-data-tests/temprandomfolder/file1.txt'),
     S3Path('s3://esp-ci-cd-tests/esp-data-tests/temprandomfolder/file2.txt'),
     S3Path('s3://esp-ci-cd-tests/esp-data-tests/temprandomfolder/random.txt'),
     ...]



Download all the contents in cloud "folder"


```python
folder.download_to("tempfolder")
```
    PosixPath('tempfolder')


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

**NOTE** once a filesystem object has been created, you dont need to type the "gs://" prefix.

File operations work like so:


```python
# check for existence
fs.exists("esp-ci-cd-tests/delme1")
```
    True



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
fs.exists("esp-ci-cd-tests/docstest")
```
    False



List contents that match a pattern like '*text' under a path


```python
fs.glob("esp-ci-cd-tests/**/*txt")
```
    ['esp-ci-cd-tests/esp-data-tests/find_files_tests/file_find_cloud.txt',
     'esp-ci-cd-tests/esp-data-tests/non_empty/temp_file.txt',
     'esp-ci-cd-tests/esp-data-tests/random2.txt',
     ...
     'esp-ci-cd-tests/random.txt',
     'esp-ci-cd-tests/random_for_upload_test.txt',
     'esp-ci-cd-tests/titouan.txt']



Download and upload files


```python
fs.get("esp-ci-cd-tests/foo.txt", "foofoo.txt")
```
    [None]




```python
fs.put("foofoo.txt", "esp-ci-cd-tests/foo.txt")
```
    [None]


## Read audio from a remote file

```python
from esp_data.io import read_audio

audio, sample_rate = read_audio("gs://esp-ci-cd-tests/esp-data-tests/some_subfolder/nri-battlesounds.mp3")
print(audio.shape, sample_rate)
# (235008,) 44100
```
