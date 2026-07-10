"""Google Cloud Storage authentication helpers.

Provides access-token retrieval for authenticating GCS REST/HTTP requests
(e.g. the ffmpeg range-read path in `alp_data.io.read_utils`).

Tokens handed out by this module are *downscoped* via a GCP Credential Access
Boundary: each token is only valid for read-only access
(`roles/storage.objectViewer`) to the single bucket it was requested for, and
is rejected by every other GCP service and bucket. This limits the blast
radius when a token leaves the process (e.g. on the ffmpeg command line,
where it is visible to co-tenants via `ps` on shared hosts): a leaked token
grants at most short-lived read access to the one bucket that was already
being read.

Source credentials are cached at module level so the Application Default
Credentials lookup happens once per session; downscoped credentials are
cached per bucket and refreshed (a token exchange with the GCP STS endpoint)
only when they expire.
"""

import logging
import time

import google.auth
from google.auth import downscoped
from google.auth.transport.requests import Request

logger = logging.getLogger("alp_data")

# Module-level cache of the source (full-identity) Google credentials.
# Reused across calls so the ADC lookup only happens once per session. The
# source credentials never leave the process; only downscoped tokens do.
_source_credentials = None

# Per-bucket cache of downscoped credentials. Each entry is refreshed
# independently when its token expires.
_downscoped_credentials_by_bucket: dict[str, downscoped.Credentials] = {}

# A way to tell if a client is authenticated: Application Default
# Credentials lookup fails, we stop re-attempting it (it is relatively expensive)
# and treat the environment as credential-less for the rest of the session.
_gcs_credentials_unavailable = False

# Backoff for failures with otherwise-present credentials (e.g. a network blip
# or blocked STS endpoint during refresh/exchange): skip re-attempts until this
# monotonic deadline, so hot read loops don't pay a failed network call per read.
_AUTH_RETRY_BACKOFF_SECONDS = 60.0
_auth_failure_backoff_until = 0.0


class GCSAuthError(Exception):
    """Raised when Google Cloud credentials cannot be obtained or refreshed."""


def _bucket_access_boundary(bucket: str) -> downscoped.CredentialAccessBoundary:
    """Build a read-only Credential Access Boundary for a single bucket.

    Parameters
    ----------
    bucket : str
        Name of the GCS bucket the boundary should grant access to.

    Returns
    -------
    downscoped.CredentialAccessBoundary
        A boundary granting `roles/storage.objectViewer` on `bucket` only.
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
    `gcloud auth application-default login` or a service account. The ambient
    credentials are exchanged (via the GCP STS endpoint) for a token that is
    only valid for `roles/storage.objectViewer` on the given bucket, so the
    returned token can safely be passed to subprocesses. Source credentials
    and per-bucket downscoped credentials are cached at module level.

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
            # Refreshes the source credentials if needed, then re-runs the
            # STS token exchange for this bucket's boundary.
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

    A valid token works for both public and private buckets,
    so we send one whenever credentials are
    available and fall back to anonymous access only when they are not.

    Parameters
    ----------
    bucket : str
        Name of the GCS bucket the token must grant read access to.

    The credential-less verdict is only cached (sticky for the session) when
    Application Default Credentials discovery itself fails — i.e. the
    environment has no ambient credentials. A failure while refreshing or
    downscoping an otherwise-working credential (e.g. a network blip or a
    blocked STS endpoint) returns None and pauses re-attempts for
    `_AUTH_RETRY_BACKOFF_SECONDS`, so transient errors recover quickly while
    persistent ones cost one failed network call per backoff window instead
    of one per read.

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
