"""Unitary tests of Google Cloud Storage authentication helpers."""

import urllib.error
import urllib.request

import pytest
from google.auth import downscoped

import alp_data.io.auth as auth
from alp_data.io.auth import GCSAuthError, get_gcs_token, get_gcs_token_if_available

TEST_BUCKET = "esp-ci-cd-tests"


def _http_status(url: str, headers: dict[str, str]) -> int:
    """Return the HTTP status of a GET request, without raising on 4xx.

    Parameters
    ----------
    url : str
        The URL to request.
    headers : dict[str, str]
        HTTP headers to send with the request.

    Returns
    -------
    int
        The HTTP status code of the response.
    """
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as e:
        return e.code


def test_get_gcs_token() -> None:
    """get_gcs_token returns a non-empty access token from ambient credentials."""
    token = get_gcs_token(TEST_BUCKET)
    assert isinstance(token, str)
    assert len(token) > 0


def test_get_gcs_token_is_downscoped_to_bucket() -> None:
    """The token works for reads on its bucket but nothing else.

    The token is exchanged through a Credential Access Boundary, so it must
    be able to read objects in `TEST_BUCKET` while being rejected by other
    GCP services — even ones the underlying identity can access.
    """
    token = get_gcs_token(TEST_BUCKET)
    headers = {"Authorization": f"Bearer {token}"}

    # In boundary: a ranged object read (what the ffmpeg path performs).
    status = _http_status(
        f"https://storage.googleapis.com/{TEST_BUCKET}"
        "/esp-data-tests/some_subfolder/nri-battlesounds.mp3",
        headers={**headers, "Range": "bytes=0-9"},
    )
    assert status in (200, 206)

    # Out of boundary: any non-storage service must reject the token.
    status = _http_status(
        "https://secretmanager.googleapis.com/v1/projects/esp-data-274503/secrets",
        headers=headers,
    )
    assert status in (401, 403)


def test_downscoped_credentials_cached_per_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each bucket gets its own cached downscoped credentials, built once."""
    built = []

    class _FakeCredentials:
        def __init__(
            self,
            source_credentials: object,
            credential_access_boundary: downscoped.CredentialAccessBoundary,
        ) -> None:
            resource = credential_access_boundary.rules[0].available_resource
            built.append(resource)
            self.token = f"token-for-{resource.rsplit('/', 1)[-1]}"
            self.valid = True

    monkeypatch.setattr(auth, "_source_credentials", object())
    monkeypatch.setattr(auth, "_downscoped_credentials_by_bucket", {})
    monkeypatch.setattr(auth.downscoped, "Credentials", _FakeCredentials)

    assert get_gcs_token("bucket-a") == "token-for-bucket-a"
    assert get_gcs_token("bucket-a") == "token-for-bucket-a"
    assert get_gcs_token("bucket-b") == "token-for-bucket-b"
    # bucket-a's credentials were reused, not rebuilt.
    assert len(built) == 2


def test_maybe_get_gcs_token_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto path returns None and caches the verdict when no credentials exist."""
    calls = {"n": 0}

    def _no_creds(bucket: str) -> str:
        calls["n"] += 1
        raise GCSAuthError("no ambient credentials")

    monkeypatch.setattr(auth, "_source_credentials", None)
    monkeypatch.setattr(auth, "_gcs_credentials_unavailable", False)
    monkeypatch.setattr(auth, "_auth_failure_backoff_until", 0.0)
    monkeypatch.setattr(auth, "get_gcs_token", _no_creds)

    assert get_gcs_token_if_available(TEST_BUCKET) is None
    # Sticky verdict: the second call must not re-attempt the ADC lookup.
    assert get_gcs_token_if_available(TEST_BUCKET) is None
    assert calls["n"] == 1


def test_maybe_get_gcs_token_transient_failure_backs_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refresh/exchange failure backs off instead of disabling auth for the session.

    When source credentials exist but a token refresh or STS exchange fails
    (e.g. a network blip or blocked STS endpoint), the auto path must not
    cache a credential-less verdict, must not re-attempt on every call (hot
    read loops would pay a failed network call per read), and must re-attempt
    once the backoff window has passed.
    """
    calls = {"n": 0}

    def _failing_exchange(bucket: str) -> str:
        calls["n"] += 1
        raise GCSAuthError("STS exchange failure")

    # Source credentials were obtained earlier in the session.
    monkeypatch.setattr(auth, "_source_credentials", object())
    monkeypatch.setattr(auth, "_gcs_credentials_unavailable", False)
    monkeypatch.setattr(auth, "_auth_failure_backoff_until", 0.0)
    monkeypatch.setattr(auth, "get_gcs_token", _failing_exchange)

    assert get_gcs_token_if_available(TEST_BUCKET) is None
    assert calls["n"] == 1
    assert auth._gcs_credentials_unavailable is False

    # Within the backoff window: no re-attempt, still None.
    assert get_gcs_token_if_available(TEST_BUCKET) is None
    assert calls["n"] == 1

    # After the backoff window passes, the next call re-attempts.
    monkeypatch.setattr(auth, "_auth_failure_backoff_until", 0.0)
    assert get_gcs_token_if_available(TEST_BUCKET) is None
    assert calls["n"] == 2
