from typing import Any

from esp_data.io.filesystem import filesystem_from_path
from esp_data.io.paths import AnyPathT


def exists(path: str | AnyPathT) -> bool:
    """Check if a file or directory exists.

    Parameters
    ----------
    path: str | AnyPathT
        File or directory to check.

    Returns
    -------
    bool
        True if the file or directory exists, False otherwise.
    """
    return filesystem_from_path(path).exists(str(path))


def rm(
    path: str | AnyPathT,
    recursive: bool = False,
    maxdepth: int | None = None,
    **kwargs: dict[str, Any],
) -> None:
    """Delete files.

    Parameters
    ----------
    path: str | AnyPathT
        File(s) to delete.
    recursive: bool
        If file(s) are directories, recursively delete contents and then also remove the
        directory
    maxdepth: int | None
        Depth to pass to walk for finding files to delete, if recursive. If None, there
        will be no limit and infinite recursion may be possible.
    """
    filesystem_from_path(path).rm(str(path), recursive=recursive, maxdepth=maxdepth, **kwargs)
