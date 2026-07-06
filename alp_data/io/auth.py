"""Google Cloud Storage authentication helpers.

Provides access-token retrieval for authenticating GCS REST/HTTP requests
(e.g. the ffmpeg range-read path in `alp_data.io.read_utils`). Credentials are
cached at module level so a token is only fetched once per session and
refreshed when it expires.
"""

import logging

import google.auth
from google.auth.transport.requests import Request

logger = logging.getLogger("alp_data")

# Module-level cache of Google credentials.
# Reused across calls so the token is
# only refreshed once per session (and again whenever it expires).
_gcs_credentials = None

# A way to tell if a client is authenticated: Application Default
# Credentials lookup fails, we stop re-attempting it (it is relatively expensive)
# and treat the environment as credential-less for the rest of the session.
_gcs_credentials_unavailable = False


class GCSAuthError(Exception):
    """Raised when Google Cloud credentials cannot be obtained or refreshed."""


def get_gcs_token() -> str:
    """Fetch a valid access token using Google Application Default Credentials.

    Relies on the caller having authenticated with GCP, e.g. via
    `gcloud auth application-default login` or a service account. The
    credentials are cached in _gcs_credentials.

    Returns
    -------
    str
        A valid access token for authenticating requests to Google Cloud.

    Raises
    ------
    GCSAuthError
        If credentials cannot be obtained or refreshed.
    """
    global _gcs_credentials
    try:
        if _gcs_credentials is None:
            _gcs_credentials, _ = google.auth.default()
        if not _gcs_credentials.valid:
            _gcs_credentials.refresh(Request())
        return _gcs_credentials.token
    except Exception as e:
        raise GCSAuthError(
            f"Error authenticating with Google Cloud: {e}.\n"
            "Ensure you have run 'gcloud auth application-default login' "
            "and have permission to access the GCS bucket."
        ) from e


def get_gcs_token_if_available() -> str | None:
    """Return a GCS access token if ambient credentials exist, otherwise None.

    A valid token works for both public and private buckets,
    so we send one whenever credentials are
    available and fall back to anonymous access only when they are not.

    Returns
    -------
    str or None
        A valid access token, or None if no ambient credentials are available.
    """
    global _gcs_credentials_unavailable
    if _gcs_credentials_unavailable:
        return None
    try:
        return get_gcs_token()
    except GCSAuthError:
        _gcs_credentials_unavailable = True
        return None
