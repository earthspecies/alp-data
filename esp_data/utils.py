import asyncio
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Callable
from uuid import UUID, uuid4

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


def validate_id(v: str) -> str:
    try:
        UUID(v)
        return v
    except ValueError:
        raise ValueError("Invalid UUID format")


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


def validate_datetime(v: datetime | str) -> datetime:
    if isinstance(v, str):
        return datetime.fromisoformat(v)

    # check that tzinfo is present and is UTC
    if v.tzinfo is None or v.tzinfo.utcoffset(v) != timedelta(0):
        raise ValueError("created_at must be a datetime object with UTC timezone")
    return v


def make_id() -> str:
    return str(uuid4())


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


async def run_as_async(func: Callable, *args, **kwargs) -> Callable:
    """Run the function asynchronously.

    Args:
        func (Callable): The function to run asynchronously.

    Returns:
        Callable: The function that runs asynchronously.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args, **kwargs)
