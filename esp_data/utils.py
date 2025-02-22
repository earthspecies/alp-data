import asyncio
import concurrent.futures
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from functools import partial
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


async def run_as_async(func: Callable, new_event_loop: bool = False, **func_kwargs) -> Callable:
    """Run the function asynchronously.

    Args:
        func (Callable): The function to run asynchronously.

    Returns:
        Callable: The function that runs asynchronously.
    """
    if new_event_loop:
        loop = asyncio.new_event_loop()
    else:
        loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, partial(func, **func_kwargs))


def make_simple_logger(name: str, add_file_handler: bool = False) -> logging.Logger:
    """Create a simple logger with a stream handler.

    Args:
        name (str): Name of the logger
        add_file_handler (bool, optional): Add a file handler. Defaults to False.

    Returns:
        logging.Logger: Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if add_file_handler:
        fh = logging.FileHandler(f"{name}.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
