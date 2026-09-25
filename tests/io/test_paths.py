"""Integration tests for filesystem operations with path objects.

This module tests the integration between path objects (from paths.py) and
filesystem operations (from filesystem.py).
"""

import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import aiohttp
import pytest
from fsspec.implementations.http import HTTPFileSystem

from alp_data.io import anypath, exists, filesystem, filesystem_from_path, rm


def test_anypath_local_path_with_file_operations():
    """Test anypath with local paths and file I/O operations."""
    path = anypath("tests/samples/file1.txt")
    assert isinstance(path, Path)
    assert path.is_file()
    assert path.read_text().strip() == "hello"
    assert path.exists()


@pytest.mark.parametrize(
    "cloud_path",
    [
        f"gs://esp-ci-cd-tests/esp-data-tests/test-{uuid.uuid4()}.txt",
        f"r2://esp-ci-cd-tests/esp-data-tests/test-{uuid.uuid4()}.txt",
    ],
)
def test_cloud_filesystem_operations(cloud_path):
    """Test filesystem operations (upload, info, read, delete) with cloud paths."""
    path = anypath(cloud_path)
    fs = filesystem_from_path(path)

    fs.put("tests/samples/file1.txt", str(path))

    info = fs.info(str(path))
    assert info["size"] == 6
    assert info["type"] == "file"
    with fs.open(str(path), "rb") as f:
        assert f.read() == b"hello\n"

    fs.rm(str(path))
    assert not fs.exists(str(path))


def test_filesystem_from_path():
    """Test filesystem_from_path creates appropriate filesystem objects for different path types."""
    # Test with GCS path
    gs_path = anypath("gs://bucket/file.txt")
    fs = filesystem_from_path(gs_path)
    assert fs is not None

    # Test with R2 path
    r2_path = anypath("r2://bucket/file.txt")
    fs = filesystem_from_path(r2_path)
    assert fs is not None

    # Test with local path
    local_path = anypath("local/file.txt")
    fs = filesystem_from_path(local_path)
    assert fs is not None

    # Test with HTTPS path
    https_path = anypath("https://example.com/datasets/file.json")
    fs = filesystem_from_path(https_path)
    assert isinstance(fs, HTTPFileSystem)

    # Test with plain HTTP path
    http_path = anypath("http://example.com/datasets/file.json")
    fs = filesystem_from_path(http_path)
    assert isinstance(fs, HTTPFileSystem)


@pytest.mark.parametrize("protocol", ["http", "https"])
def test_filesystem_http_protocols(protocol):
    """Both HTTP(S) protocol strings map to an HTTPFileSystem."""
    assert isinstance(filesystem(protocol), HTTPFileSystem)


def test_filesystem_unknown_protocol():
    """An unsupported protocol raises with the supported backends listed."""
    with pytest.raises(ValueError, match="Unknown backend: ftp"):
        filesystem("ftp")


@pytest.mark.parametrize(
    "url", ["https://example.com/datasets/file.json", "http://example.com/datasets/file.json"]
)
def test_rm_rejects_http_paths(url):
    """Deleting over HTTP(S) fails with an explanatory message and no request."""
    with pytest.raises(NotImplementedError, match="read-only"):
        rm(url)


def test_rm_rejects_http_paths_before_expanding():
    """`recursive=True` must not issue requests while expanding the path first."""
    with pytest.raises(NotImplementedError, match="Cannot delete over HTTP"):
        rm("https://example.com/datasets/", recursive=True)


@pytest.fixture
def recording_server():
    """A local HTTP server that records the request methods it receives.

    Yields
    ------
    tuple[str, list[tuple[str, str]]]
        The server's base URL and the list of `(method, path)` it has served.
        `/present` answers 200 with a `Content-Length`, `/forbidden` answers 403
        and anything else answers 404.
    """
    served: list[tuple[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def respond(self):
            served.append((self.command, self.path))
            body = b"hello" if self.path == "/present" else b""
            status = {"/present": 200, "/forbidden": 403}.get(self.path, 404)
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body and self.command == "GET":
                self.wfile.write(body)

        do_GET = respond
        do_HEAD = respond

        def log_message(self, *args):
            """Silence the default stderr request log."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", served
    finally:
        server.shutdown()
        server.server_close()


def test_exists_over_http_sends_only_a_head(recording_server):
    """A present object is confirmed by HEAD alone: no GET, no body transferred."""
    base, served = recording_server

    assert exists(f"{base}/present") is True
    assert served == [("HEAD", "/present")]


def test_exists_over_http_falls_back_to_get_for_a_missing_object(recording_server):
    """An absent object costs two requests: `info` retries with a GET on 404."""
    base, served = recording_server

    assert exists(f"{base}/missing") is False
    assert served == [("HEAD", "/missing"), ("GET", "/missing")]


def test_exists_over_http_raises_on_a_status_that_is_not_404(recording_server):
    """A 403 object exists but cannot be read, so it must not report False.

    The status is recovered from the error `info` chained, so no third request
    is needed to tell this apart from a 404.
    """
    base, served = recording_server

    with pytest.raises(aiohttp.ClientResponseError) as excinfo:
        exists(f"{base}/forbidden")

    assert excinfo.value.status == 403
    assert served == [("HEAD", "/forbidden"), ("GET", "/forbidden")]


def test_exists_over_http_raises_when_the_host_is_unreachable():
    """An unreachable endpoint is not evidence that the object is absent."""
    with pytest.raises(aiohttp.ClientConnectorError):
        exists("http://127.0.0.1:9/unreachable.txt")
