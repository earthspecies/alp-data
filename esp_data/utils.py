import os
import re
from pathlib import Path
from datetime import datetime, timezone
import json

from cloudpathlib import AnyPath


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def utc_now_str() -> str:
    # Format will contain tzinfo (Z) for UTC
    # e.g. 2021-02-01T14:30:00.000000+00:00
    return utc_now().isoformat()


def utc_now_timestamp() -> int:
    return int(utc_now().timestamp())


# validators
def validate_json_str(v: str) -> str:
    try:
        json.loads(v)
        return v
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")


def validate_version(version: str) -> str:
    # Basic semver validation
    pattern = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    if not re.match(pattern, version):
        raise ValueError("Version must follow semantic versioning (e.g., 1.0.0)")
    return version


def validate_path_exists(p: str | os.PathLike) -> str:
    path = AnyPath(p)
    if not path.exists():
        raise ValueError(f"Path does not exist: {p}")
    return str(path)


def increment_version(version: str, mode: str = "patch") -> str:
    """Increment the version number following semantic versioning"""
    version = validate_version(version)
    if mode not in ["major", "minor", "patch"]:
        raise ValueError("Mode must be one of 'major', 'minor', or 'patch'")

    major, minor, patch = map(int, version.split("."))
    if mode == "major":
        major += 1
        minor = 0
        patch = 0
    elif mode == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1

    return f"{major}.{minor}.{patch}"


def is_gcs_path(path: str | Path | os.PathLike) -> bool:
    return str(path).startswith("gs://")


def is_s3_path(path: str | Path | os.PathLike) -> bool:
    return str(path).startswith("s3://")


def is_cloudflarer2_path(path: str | Path | os.PathLike) -> bool:
    return "r2.cloudflarestorage" in str(path)


def is_local_path(path: str | Path | os.PathLike) -> bool:
    return not (is_gcs_path(path) or is_s3_path(path) or is_cloudflarer2_path(path))


def is_cloud_path(path: str | Path | os.PathLike) -> bool:
    return is_gcs_path(path) or is_s3_path(path) or is_cloudflarer2_path(path)
