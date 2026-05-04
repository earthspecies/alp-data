"""This file contains many utility functions useful for the rest of esp_data."""

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Callable
from uuid import UUID, uuid4

import google_crc32c
from google.cloud import secretmanager

logger = logging.getLogger("esp_data")

DATA_HOME = os.environ.get("DATA_HOME", "gs://esp-data-274503")


def utc_now() -> datetime:
    """Return the current time with UTC timezone information.

    Returns
    -------
    datetime
        The current datetime object, timezone-aware and set to UTC.

    Examples
    --------
    >>> now = utc_now()
    >>> isinstance(now, datetime)
    True
    """
    return datetime.now(tz=timezone.utc)


def utc_now_str() -> str:
    """Return the current UTC time as an ISO 8601 formatted string.

    The format includes timezone information (e.g., +00:00 or Z).

    Returns
    -------
    str
        The current UTC datetime as an ISO 8601 formatted string.

    Examples
    --------
    >>> now_str = utc_now_str()
    >>> isinstance(now_str, str)
    True
    >>> '+' in now_str or 'Z' in now_str # Check for timezone indicator
    True
    >>> now_str.endswith('+00:00') or now_str.endswith('Z') # Specifically UTC
    True
    """
    # Format will contain tzinfo (Z) for UTC
    # e.g. 2021-02-01T14:30:00.000000+00:00
    return utc_now().isoformat()


def utc_now_timestamp() -> int:
    """Return the current UTC time as a Unix timestamp (integer seconds).

    Returns
    -------
    int
        The number of seconds since the Unix epoch (1970-01-01 UTC).

    Examples
    --------
    >>> ts = utc_now_timestamp()
    >>> isinstance(ts, int)
    True
    >>> ts > 1600000000 # Check if it's a plausible timestamp (after year 2020)
    True
    """
    return int(utc_now().timestamp())


def create_hash(s: str | bytes) -> str:
    """Create a SHA-256 hash of the given string.

    Parameters
    ----------
    s : str | bytes
        The input string or bytes array to hash.

    Returns
    -------
    str
        The hexadecimal representation of the SHA-256 hash.

    Example
    -------
    >>> create_hash("hello")
    '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    sha256 = hashlib.sha256()
    if isinstance(s, bytes):
        sha256.update(s)
    else:
        sha256.update(s.encode("utf-8"))
    return sha256.hexdigest()


def validate_json_str(v: str) -> str:
    """Validate if a string is valid JSON.

    Parameters
    ----------
    v : str
        The string to validate.

    Returns
    -------
    str
        The original string if it is valid JSON.

    Raises
    ------
    ValueError
        If the string is not valid JSON.

    Examples
    --------
    >>> validate_json_str('{"key": "value", "number": 123}')
    '{"key": "value", "number": 123}'
    """
    try:
        json.loads(v)
        return v
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}") from e


def validate_id(v: str) -> str:
    """Validate if a string is a valid UUID.

    Parameters
    ----------
    v : str
        The string to validate.

    Returns
    -------
    str
        The original string if it is a valid UUID.

    Raises
    ------
    ValueError
        If the string is not a valid UUID format.

    Examples
    --------
    >>> validate_id("123e4567-e89b-12d3-a456-426614174000")
    '123e4567-e89b-12d3-a456-426614174000'
    """
    try:
        UUID(v)
        return v
    except ValueError as e:
        raise ValueError("Invalid UUID format") from e


def validate_version(version: str) -> str:
    """Validate if a string follows semantic versioning.

    Checks for the basic MAJOR.MINOR.PATCH format.

    Parameters
    ----------
    version : str
        The version string to validate.

    Returns
    -------
    str
        The original string if it follows SemVer format.

    Raises
    ------
    ValueError
        If the string does not follow SemVer format.

    Examples
    --------
    >>> validate_version("1.0.0")
    '1.0.0'
    >>> validate_version("2.10.3-alpha.1+build.5")
    '2.10.3-alpha.1+build.5'
    """
    # Basic semver validation
    pattern = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"  # noqa E501
    if not re.match(pattern, version):
        raise ValueError("Version must follow semantic versioning (e.g., 1.0.0)")
    return version


def validate_datetime(v: str) -> str:
    """Validate if a string is an ISO 8601 datetime with UTC timezone.

    Parameters
    ----------
    v : str
        The datetime string to validate.

    Returns
    -------
    str
        The original string if it is a valid ISO 8601 datetime string
        representing a UTC time.

    Raises
    ------
    ValueError
        If the string is not a valid ISO 8601 format, or if it does not
        contain timezone information, or if the timezone is not UTC.

    Examples
    --------
    >>> validate_datetime("2023-10-27T10:00:00+00:00")
    '2023-10-27T10:00:00+00:00'
    """
    try:
        d = datetime.fromisoformat(v)
    except ValueError as e:
        raise ValueError("Invalid datetime string") from e

    # check that tzinfo is present and is UTC
    if d.tzinfo is None or d.tzinfo.utcoffset(d) != timedelta(0):
        raise ValueError("created_at must be a datetime object with UTC timezone")
    return v


def make_id() -> str:
    """Generate a new unique identifier (UUID version 4) as a string.

    Returns
    -------
    str
        A randomly generated UUID string.

    Examples
    --------
    >>> import uuid
    >>> new_id = make_id()
    >>> isinstance(new_id, str)
    True
    >>> len(new_id) == 36 # UUIDs have a fixed string length
    True
    """
    return str(uuid4())


def increment_version(version: str, mode: str = "patch") -> str:
    """Increment a semantic version string by major, minor, or patch level.

    Parameters
    ----------
    version : str
        The semantic version string to increment (e.g., "1.2.3").
    mode : str, optional
        The part of the version to increment: "major", "minor", or "patch".
        Defaults to "patch".

    Returns
    -------
    str
        The incremented version string.

    Raises
    ------
    ValueError
        If the input version string is not valid SemVer, or if the mode
        is not one of "major", "minor", "patch".

    Examples
    --------
    >>> increment_version("1.2.3")
    '1.2.4'
    >>> increment_version("1.2.3", mode="minor")
    '1.3.0'
    >>> increment_version("1.2.3", mode="major")
    '2.0.0'
    >>> increment_version("1.9.5", mode="minor")
    '1.10.0'
    """
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


def read_gcp_secret(
    secret_id: str, version_id: str = "latest", project_id: str = "okapi-274503"
) -> str:
    """Read a secret's value from Google Secret Manager.


    Parameters
    ----------
    secret_id : str
        The ID of the secret in Google Secret Manager.
    version_id : str, optional
        The version of the secret to access (e.g., "5" or "latest").
        Defaults to "latest".
    project_id : str, optional
        The Google Cloud project ID where the secret resides.
        Defaults to "okapi-274503".

    Returns
    -------
    str
        The decoded secret payload as a UTF-8 string.

    Raises
    ------
    ValueError
        If data corruption is detected (CRC32C check fails).

    Notes
    -----
    Requires authentication with Google Cloud (e.g., through
    `gcloud auth application-default login`
    or service account credentials) with appropriate permissions for Secret Manager.
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
    """Decorator to create a cached, read-only property on a class.

    The decorated method is executed only once when the property is first
    accessed on the class. The result is stored on the class and returned
    for all subsequent accesses.

    Parameters
    ----------
    method : Callable
        The method to decorate. It must accept the class (`cls`) as its
        first argument.

    Returns
    -------
    CachedClassProperty
        An instance of the descriptor that implements the caching logic.

    Examples
    --------
    >>> class DatabaseConnection:
    ...     _connection = None
    ...
    ...     @cached_class_property
    ...     def connection(cls):
    ...         print("Establishing connection...")
    ...         # Simulate creating a connection object
    ...         cls._connection = f"Connection-{id(cls)}"
    ...         return cls._connection
    ...
    >>> conn1 = DatabaseConnection.connection
    Establishing connection...
    >>> print(conn1) # doctest: +ELLIPSIS
    Connection-...
    >>> conn2 = DatabaseConnection.connection # Should not print "Establishing..."
    >>> print(conn1 is conn2) # Should be the exact same object
    True
    """

    def __init__(self, method: Callable) -> None:
        """Initialize the CachedClassProperty descriptor.

        Parameters
        ----------
        method : Callable
            The method being decorated, which computes the value to be cached.
            It is expected to accept the owner class (`cls`) as its first argument.

        """
        self.method = method
        self.cache_attrname = f"_cached_class_attr_{method.__name__}"

    def __get__(self, instance: object, owner: type = None) -> object:
        """Retrieve the property's value, computing and caching if needed.

        Parameters
        ----------
        instance : object or None
            The instance through which the property was accessed, or None if
            accessed directly via the class.
        owner : type or None
            The class that owns the property (e.g., `MySettings` in the example).
            Python automatically supplies this.

        Returns
        -------
        Any
            The computed and cached result of the decorated method.
        """
        if owner is None:
            owner = type(instance)
        if not hasattr(owner, self.cache_attrname):
            value = self.method(owner)
            setattr(owner, self.cache_attrname, value)
        return getattr(owner, self.cache_attrname)


def cached_class_property(method: Callable) -> Callable:
    """Decorator to create a cached, read-only property on a class.

    The decorated method is executed only once when the property is first
    accessed on the class. The result is stored on the class and returned
    for all subsequent accesses.

    Parameters
    ----------
    method : Callable
        The method to decorate. It must accept the class (`cls`) as its
        first argument.

    Returns
    -------
    CachedClassProperty
        An instance of the descriptor that implements the caching logic.

    """
    return CachedClassProperty(method)
