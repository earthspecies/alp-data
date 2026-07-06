"""Unitary tests of Google Cloud Storage authentication helpers."""

import alp_data.io.auth as auth
from alp_data.io.auth import GCSAuthError, get_gcs_token, get_gcs_token_if_available


def test_get_gcs_token() -> None:
    """get_gcs_token returns a non-empty access token from ambient credentials."""
    token = get_gcs_token()
    assert isinstance(token, str)
    assert len(token) > 0


def test_maybe_get_gcs_token_no_credentials(monkeypatch) -> None:
    """Auto path returns None and caches the verdict when no credentials exist."""
    calls = {"n": 0}

    def _no_creds() -> str:
        calls["n"] += 1
        raise GCSAuthError("no ambient credentials")

    monkeypatch.setattr(auth, "_gcs_credentials_unavailable", False)
    monkeypatch.setattr(auth, "get_gcs_token", _no_creds)

    assert get_gcs_token_if_available() is None
    # Sticky verdict: the second call must not re-attempt the ADC lookup.
    assert get_gcs_token_if_available() is None
    assert calls["n"] == 1
