"""DataRepo abstraction — separates dataset identity from physical access location.

The idea: a dataset's identity (its name, version, schema, and the relative
paths of files within its folder) is independent of *where* it's physically
hosted. A `DataRepo` declares one physical location (base URL + kind), and
datasets can list multiple repos they are available in. At read time, the
resolver picks the highest-priority accessible repo and joins its base URL
with the dataset folder + relative path to produce a fetchable URL.

This is the same pattern used by Python packaging (PyPI + mirrors), HuggingFace
Hub (repo ID + CDN), Docker (image digest + registries), Maven/Go/npm, etc.

Backward compat: `DatasetInfo.split_paths` remains the public interface that
Dataset subclasses read. When a DatasetInfo is constructed with `repos` +
`folder` + `splits` instead of `split_paths`, a Pydantic `model_validator`
auto-derives `split_paths` by running the resolver. Nothing in the 37 existing
Dataset subclasses needs to change.

See `docs/esp-data-datarepos-plan.md` in the parent monorepo for the full
design rationale.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RepoKind = Literal["gs", "s3", "https", "local"]
PathEncoding = Literal["strict", "legacy_raw", "legacy_percent"]

# Strict relative-path grammar: POSIX-style, URL-safe ASCII subset
# (RFC 3986 unreserved chars + forward slash). No leading slash,
# no ".." traversal, no trailing slash.
_STRICT_RELPATH_RE = re.compile(r"^(?!/)(?!.*(?:^|/)\.\.(?:/|$))[A-Za-z0-9._~\-/]+(?<!/)$")


class InvalidRelativePathError(ValueError):
    """Raised when a relative path fails the strict-subset grammar for a
    `path_encoding='strict'` repo."""


def validate_strict_relpath(relpath: str) -> None:
    """Assert that `relpath` conforms to the strict subset.

    Constraints:
      - allowed chars: [A-Za-z0-9._~\\-/]
      - no leading or trailing slash
      - no ".." segments (directory traversal)

    Raises
    ------
    InvalidRelativePathError
        If `relpath` violates any rule. Message identifies which.
    """
    if not isinstance(relpath, str) or not relpath:
        raise InvalidRelativePathError(f"Relative path must be a non-empty string; got {relpath!r}")
    if relpath.startswith("/"):
        raise InvalidRelativePathError(f"Relative path must not start with '/': {relpath!r}")
    if relpath.endswith("/"):
        raise InvalidRelativePathError(
            f"Relative path must not end with '/' (files only): {relpath!r}"
        )
    # Check for ".." as a path segment (simple string check is enough since
    # the char class below forbids quoted dots)
    segments = relpath.split("/")
    if ".." in segments:
        raise InvalidRelativePathError(
            f"Relative path must not contain '..' (directory traversal): {relpath!r}"
        )
    if not _STRICT_RELPATH_RE.match(relpath):
        # Identify which char is the problem for a better error message.
        # Must use ASCII-only checks — `str.isalnum()` is Unicode-aware and
        # would accept chars like `ñ`, which the strict regex rejects.
        _ASCII_ALNUM = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
        _OTHER_ALLOWED = frozenset("._~-/")
        allowed = _ASCII_ALNUM | _OTHER_ALLOWED
        for ch in relpath:
            if ch not in allowed:
                raise InvalidRelativePathError(
                    f"Relative path contains disallowed char {ch!r} (U+{ord(ch):04X}) "
                    f"in {relpath!r}. Strict subset is [A-Za-z0-9._~-/]."
                )
        # Fallback — shouldn't reach here but be safe
        raise InvalidRelativePathError(f"Relative path {relpath!r} doesn't match strict subset.")


@dataclass(frozen=True)
class DataRepo:
    """A location where one or more datasets are available.

    Parameters
    ----------
    id : str
        Unique identifier for this repo (e.g., ``"esp-internal-gcs"``,
        ``"esp-public-r2"``).
    kind : RepoKind
        Protocol used to access this repo. Determines how `base_url` is
        joined with relative paths.
    base_url : str
        Root URL prefix (e.g., ``"gs://esp-ml-datasets/"``,
        ``"https://pub-abc.r2.dev/"``). Trailing slash is optional; the
        resolver normalizes.
    path_encoding : PathEncoding, default "strict"
        How relative paths within datasets stored in this repo are
        encoded. ``"strict"`` means URL-safe ASCII subset only.
        ``"legacy_raw"`` and ``"legacy_percent"`` exist for datasets
        like xeno-canto that have filename bytes outside the strict subset.
    priority : int, default 100
        Lower numbers are preferred when multiple repos for a dataset
        are accessible. ESP-internal typically gets priority ~10; public
        mirrors get ~50.
    """

    id: str
    kind: RepoKind
    base_url: str
    path_encoding: PathEncoding = "strict"
    priority: int = 100


# In-memory repo registry. Populated at import time with ESP defaults; users
# can override via `register_repo` / `unregister_repo`.
_REPO_REGISTRY: dict[str, DataRepo] = {}

# Optional per-repo access checker. Default: accept all registered repos.
# Real deployment might check for valid credentials, network reachability,
# etc. before returning True.
_access_checker: Callable[[DataRepo], bool] | None = None


def register_repo(repo: DataRepo) -> None:
    """Register a DataRepo in the global registry."""
    _REPO_REGISTRY[repo.id] = repo


def unregister_repo(repo_id: str) -> None:
    """Remove a DataRepo from the registry by id. No-op if not present."""
    _REPO_REGISTRY.pop(repo_id, None)


def get_repo(repo_id: str) -> DataRepo | None:
    """Return the DataRepo with the given id, or None if not registered.

    Returns
    -------
    DataRepo | None
        The registered repo with that id, or None if not registered.
    """
    return _REPO_REGISTRY.get(repo_id)


def list_repos() -> list[DataRepo]:
    """Return all currently-registered DataRepos.

    Returns
    -------
    list[DataRepo]
        All registered repos, in no particular order.
    """
    return list(_REPO_REGISTRY.values())


def set_access_checker(checker: Callable[[DataRepo], bool] | None) -> None:
    """Install a function that decides whether a repo is accessible.

    When `resolve()` picks among candidate repos, only those for which the
    checker returns True are considered. Use this to gate private repos
    behind credential checks, or to restrict to offline-available repos, etc.
    """
    global _access_checker
    _access_checker = checker


def _join_relpath_parts(*parts: str) -> str:
    """Normalize + join path segments with a single forward slash.

    Strips leading/trailing slashes from each part, drops empty parts.
    Intended for relative-path joining only; caller handles base URLs.

    Returns
    -------
    str
        The joined relative path, or an empty string if all inputs were empty.
    """
    cleaned = [p.strip("/") for p in parts if p]
    return "/".join(p for p in cleaned if p)


def join_url(repo: DataRepo, *relpath_parts: str) -> str:
    """Join a repo's base URL with relative path parts, dispatching on kind.

    Parameters
    ----------
    repo : DataRepo
        The repo whose `base_url` will be the prefix.
    *relpath_parts : str
        Relative path segments (e.g. dataset folder, split file name).
        Each is normalized independently — leading/trailing slashes
        stripped, empty segments dropped.

    Returns
    -------
    str
        A URL (or filesystem path, for `kind="local"`) suitable for
        passing to `anypath()` or directly to a filesystem backend.

    Raises
    ------
    ValueError
        If `repo.kind` is not one of the supported kinds.
    """
    base = repo.base_url.rstrip("/")
    relpath = _join_relpath_parts(*relpath_parts)
    if repo.kind in ("gs", "s3", "https"):
        # All three are URL-shaped; simple string concat with single slash.
        # (urllib.parse.urljoin would fight with gs:// and s3:// schemes.)
        return f"{base}/{relpath}" if relpath else base
    elif repo.kind == "local":
        return str(Path(base) / relpath) if relpath else base
    else:
        raise ValueError(f"Unknown repo kind: {repo.kind}")


class NoAccessibleRepoError(RuntimeError):
    """Raised when resolve() can't find any registered + accessible repo for a dataset."""


def resolve(
    repos_ids: list[str],
    folder: str,
    relpath: str,
    registry: dict[str, DataRepo] | None = None,
) -> str:
    """Resolve (dataset repo list, dataset folder, relative path) to a fetchable URL.

    Picks the highest-priority accessible repo from `repos_ids` that is
    currently registered. Joins its base URL with the dataset's folder
    and the relative path.

    Parameters
    ----------
    repos_ids : list[str]
        Ordered list of repo IDs the dataset declares itself available in.
        The order is advisory; the actual choice is driven by priority
        among the accessible ones.
    folder : str
        Path within each repo to the dataset's root folder. Relative.
    relpath : str
        Path within the dataset folder to the target file. Relative.
    registry : dict[str, DataRepo] | None
        Optional explicit registry to use instead of the global one.
        For testing, mostly.

    Returns
    -------
    str
        The resolved URL.

    Raises
    ------
    NoAccessibleRepoError
        If none of the named repos are registered and accessible.
    """
    reg = registry if registry is not None else _REPO_REGISTRY
    candidates: list[DataRepo] = []
    for rid in repos_ids:
        repo = reg.get(rid)
        if repo is None:
            continue
        if _access_checker is not None and not _access_checker(repo):
            continue
        candidates.append(repo)

    if not candidates:
        raise NoAccessibleRepoError(
            f"No accessible repos for dataset. Declared: {repos_ids}. "
            f"Registered: {sorted(reg.keys())}."
        )

    candidates.sort(key=lambda r: r.priority)
    chosen = candidates[0]

    # Validate path encoding at resolution time — failure here is a
    # data-configuration error, not a runtime issue, and should surface clearly.
    if chosen.path_encoding == "strict":
        validate_strict_relpath(folder)
        validate_strict_relpath(relpath)
    # "legacy_raw" and "legacy_percent" modes: no constraint enforced yet.
    # See `docs/esp-data-datarepos-plan.md` for the planned conversion logic.

    return join_url(chosen, folder, relpath)


def _register_default_repos() -> None:
    """Register the ESP default repo(s). Called at module import."""
    register_repo(
        DataRepo(
            id="esp-internal-gcs",
            kind="gs",
            base_url="gs://esp-ml-datasets/",
            path_encoding="strict",
            priority=10,
        )
    )
    # TODO: register the public R2 mirror once we know its base_url.
    # register_repo(
    #     DataRepo(
    #         id="esp-public-r2",
    #         kind="https",
    #         base_url="https://pub-<id>.r2.dev/",
    #         path_encoding="strict",
    #         priority=50,
    #     )
    # )


_register_default_repos()
