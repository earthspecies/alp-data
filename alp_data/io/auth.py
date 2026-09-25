"""Google Cloud Storage authentication helpers.

Provides access-token retrieval for authenticating GCS REST/HTTP requests
(e.g. the ffmpeg range-read path in `alp_data.io.read_utils`).

Tokens are *downscoped* through a GCP Credential Access Boundary: each token
only grants read-only access (`roles/storage.objectViewer`) to the single
bucket it was requested for. This limits the blast radius when a token leaves
the process (e.g. on the ffmpeg command line, visible via `ps` on shared
hosts): a leaked token gives at most short-lived read access to the one
bucket that was already being read. The full-identity source credentials
never leave the process.

Caching: the source credentials are resolved once per session; downscoped
credentials are cached per bucket and re-exchanged (one STS call) only on
expiry. A session with no ambient credentials at all is remembered as
credential-less; other auth failures back off for
`_AUTH_RETRY_BACKOFF_SECONDS` before re-attempting.
"""

import logging
import time

import google.auth
from google.auth import downscoped
from google.auth.transport.requests import Request

logger = logging.getLogger("alp_data")

# Full-identity source credentials, resolved once per session.
_source_credentials = None

# Downscoped credentials, cached per bucket.
_downscoped_credentials_by_bucket: dict[str, downscoped.Credentials] = {}

# Sticky verdict for a session with no ambient credentials at all (the ADC
# lookup is relatively expensive, so it is not re-attempted).
_gcs_credentials_unavailable = False

# Deadline (monotonic) before which failed refresh/exchange attempts are not
# retried, so hot read loops don't pay a failed network call per read.
_AUTH_RETRY_BACKOFF_SECONDS = 60.0
_auth_failure_backoff_until = 0.0


class GCSAuthError(Exception):
    """Raised when Google Cloud credentials cannot be obtained or refreshed."""


def _bucket_access_boundary(bucket: str) -> downscoped.CredentialAccessBoundary:
    """Build a boundary granting `roles/storage.objectViewer` on `bucket` only.

    Parameters
    ----------
    bucket : str
        Name of the GCS bucket the boundary should grant access to.

    Returns
    -------
    downscoped.CredentialAccessBoundary
        The single-bucket, read-only access boundary.
    """
    return downscoped.CredentialAccessBoundary(
        rules=[
            downscoped.AccessBoundaryRule(
                available_resource=f"//storage.googleapis.com/projects/_/buckets/{bucket}",
                available_permissions=["inRole:roles/storage.objectViewer"],
            )
        ]
    )


def get_gcs_token(bucket: str) -> str:
    """Fetch a valid access token downscoped to read-only access on `bucket`.

    Relies on the caller having authenticated with GCP, e.g. via
    `gcloud auth application-default login` or a service account.

    Parameters
    ----------
    bucket : str
        Name of the GCS bucket the token must grant read access to.

    Returns
    -------
    str
        A valid access token restricted to read-only access on `bucket`.

    Raises
    ------
    GCSAuthError
        If credentials cannot be obtained, downscoped, or refreshed.
    """
    global _source_credentials
    try:
        if _source_credentials is None:
            _source_credentials, _ = google.auth.default()
        credentials = _downscoped_credentials_by_bucket.get(bucket)
        if credentials is None:
            credentials = downscoped.Credentials(
                source_credentials=_source_credentials,
                credential_access_boundary=_bucket_access_boundary(bucket),
            )
            _downscoped_credentials_by_bucket[bucket] = credentials
        if not credentials.valid:
            credentials.refresh(Request())
        return credentials.token
    except Exception as e:
        raise GCSAuthError(
            f"Error authenticating with Google Cloud: {e}.\n"
            "Ensure you have run 'gcloud auth application-default login' "
            "and have permission to access the GCS bucket."
        ) from e


def get_gcs_token_if_available(bucket: str) -> str | None:
    """Return a bucket-scoped GCS token if ambient credentials exist, else None.

    A valid token works for both public and private buckets, so we send one
    whenever credentials are available and fall back to anonymous access only
    when they are not. Only a failed ADC lookup (no ambient credentials at
    all) is remembered for the session; other failures back off for
    `_AUTH_RETRY_BACKOFF_SECONDS` and are then re-attempted.

    Parameters
    ----------
    bucket : str
        Name of the GCS bucket the token must grant read access to.

    Returns
    -------
    str or None
        A valid access token restricted to read-only access on `bucket`, or
        None if no ambient credentials are available.
    """
    global _gcs_credentials_unavailable, _auth_failure_backoff_until
    if _gcs_credentials_unavailable:
        return None
    if time.monotonic() < _auth_failure_backoff_until:
        return None
    try:
        return get_gcs_token(bucket)
    except GCSAuthError:
        if _source_credentials is None:  # no ambient credentials at all
            _gcs_credentials_unavailable = True
        else:
            _auth_failure_backoff_until = time.monotonic() + _AUTH_RETRY_BACKOFF_SECONDS
        return None
