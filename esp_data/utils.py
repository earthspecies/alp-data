import asyncio
import concurrent.futures
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Callable
from uuid import UUID, uuid4

import google_crc32c
from google.cloud import secretmanager

logger = logging.getLogger("esp_data")


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def utc_now_str() -> str:
    # Format will contain tzinfo (Z) for UTC
    # e.g. 2021-02-01T14:30:00.000000+00:00
    return utc_now().isoformat()


def utc_now_timestamp() -> int:
    return int(utc_now().timestamp())


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


def validate_datetime(v: str) -> str:
    try:
        d = datetime.fromisoformat(v)
    except ValueError:
        raise ValueError("Invalid datetime string")

    # check that tzinfo is present and is UTC
    if d.tzinfo is None or d.tzinfo.utcoffset(d) != timedelta(0):
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


def read_gcp_secret(secret_id: str, version_id: str = "latest", project_id: str = "okapi-274503") -> str:
    """
    A function to read a secret from Google Secret Manager.

    Implementation is based on the example in official Google documentation:
    https://cloud.google.com/secret-manager/docs/samples/secretmanager-access-secret-version
    """

    client = secretmanager.SecretManagerServiceClient()

    resource_name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": resource_name})

    # Verify the payload
    crc32c = google_crc32c.Checksum()
    crc32c.update(response.payload.data)
    if response.payload.data_crc32c != int(crc32c.hexdigest(), 16):
        logger.error(f"Data corruption detected while reading secret: {secret_id}")
        raise ValueError

    payload = response.payload.data.decode("UTF-8")
    return payload


class CachedClassProperty:
    def __init__(self, method):
        self.method = method
        self.cache_attrname = f"_cached_class_attr_{method.__name__}"

    def __get__(self, instance, owner=None):
        if owner is None:
            owner = type(instance)
        if not hasattr(owner, self.cache_attrname):
            value = self.method(owner)
            setattr(owner, self.cache_attrname, value)
        return getattr(owner, self.cache_attrname)


def cached_class_property(method):
    return CachedClassProperty(method)
