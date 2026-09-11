from typing import Any

import aiohttp

from alp_data.io.filesystem import filesystem_from_path
from alp_data.io.paths import AnyPathT, PureHTTPPath, PureHTTPSPath, anypath


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

    Raises
    ------
    aiohttp.ClientError
        For an HTTP(S) path the server could not answer for: the host is
        unreachable, or it replied with a status other than 404 (401, 403 and
        500 included). Such a URL is neither present nor absent, so it raises
        rather than reporting False.

    Notes
    -----
    HTTP(S) paths are checked with `HTTPFileSystem.info`, which sends a HEAD and
    reads `Content-Length`, instead of `HTTPFileSystem.exists`, which GETs the
    URL and calls any status below 400 "exists". A HEAD is roughly half the
    latency and, unlike the discarded GET, transfers no part of the body — worth
    having when checking many URLs. An object that is there costs that one HEAD;
    `info` retries with a GET before giving up, so an absent one costs two
    requests. A server that answers a missing object with a 200 "not found" page
    still reports as existing.
    """  # noqa: DOC502 - the documented error is re-raised via a variable
    fs = filesystem_from_path(path)
    path_str = str(path)

    if not isinstance(anypath(path), (PureHTTPSPath, PureHTTPPath)):
        return fs.exists(path_str)

    try:
        fs.info(path_str)
    except FileNotFoundError as e:
        # `HTTPFileSystem.info` reports every failure as FileNotFoundError, a
        # genuine 404 and an unreachable host alike, so the exception it chained
        # is what actually answers the question.
        cause = e.__cause__
        if isinstance(cause, aiohttp.ClientResponseError) and cause.status == 404:
            return False
        if isinstance(cause, aiohttp.ClientError):
            # Not a 404, so the object may well be there and we simply could not
            # look. Surface the transport error itself: the FileNotFoundError
            # wrapper reads as "absent", which is exactly what this is not.
            raise cause from None
        # Nothing recognisable to go on: settle it with a strict existence
        # check, which costs a GET but returns False only for a real 404.
        return fs.exists(path_str, strict=True)

    return True


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

    Raises
    ------
    NotImplementedError
        If `path` is an HTTP(S) URL. HTTP(S) endpoints are read-only in `alp_data`.

    Notes
    -----
    HTTP(S) paths are rejected before any request is made. `fsspec`'s
    `HTTPFileSystem` does not implement file removal, so it would raise a bare
    `NotImplementedError` with no message anyway — and with `recursive=True` it
    would first issue several HTTP requests while expanding the path.
    """
    if isinstance(anypath(path), (PureHTTPSPath, PureHTTPPath)):
        raise NotImplementedError(
            f"Cannot delete over HTTP(S): {path}. HTTP(S) endpoints are read-only in "
            "alp_data; use a 'gs://' or 'r2://' path to delete an object."
        )

    filesystem_from_path(path).rm(str(path), recursive=recursive, maxdepth=maxdepth, **kwargs)
