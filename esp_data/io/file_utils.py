from esp_data.io.filesystem import filesystem_from_path
from esp_data.io.paths import AnyPathT


def exists(path: str | AnyPathT) -> bool:
    return filesystem_from_path(path).exists(str(path))


def rm(path: str | AnyPathT, recursive: bool = False, maxdepth: int | None = None, **kwargs):
    filesystem_from_path(path).rm(str(path), recursive=recursive, maxdepth=maxdepth, **kwargs)
