"""Tests for the centralised Fyers token resolver.

These assert the resolution order and the local-vs-environment split without
ever touching a real credential or the network.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import fyers_auth


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (fyers_auth.TOKEN_ENV_VAR, fyers_auth.TOKEN_FILE_ENV_VAR):
        monkeypatch.delenv(var, raising=False)


def _token_file(tmp_path, monkeypatch, contents):
    path = tmp_path / ".fyers_token"
    path.write_text(contents, encoding="utf-8")
    monkeypatch.setenv(fyers_auth.TOKEN_FILE_ENV_VAR, str(path))
    return path


def test_env_var_wins_over_file(tmp_path, monkeypatch):
    _token_file(tmp_path, monkeypatch, "file-token")
    monkeypatch.setenv(fyers_auth.TOKEN_ENV_VAR, "env-token")
    assert fyers_auth.resolve_access_token() == "env-token"


def test_file_used_when_env_var_absent(tmp_path, monkeypatch):
    _token_file(tmp_path, monkeypatch, "file-token")
    assert fyers_auth.resolve_access_token() == "file-token"


def test_surrounding_whitespace_is_stripped(tmp_path, monkeypatch):
    _token_file(tmp_path, monkeypatch, "  padded-token\n")
    assert fyers_auth.resolve_access_token() == "padded-token"
    monkeypatch.setenv(fyers_auth.TOKEN_ENV_VAR, "  env-token \n")
    assert fyers_auth.resolve_access_token() == "env-token"


def test_blank_env_var_falls_through_to_file(tmp_path, monkeypatch):
    # An env var that is present but empty/whitespace must not mask the file,
    # otherwise a misconfigured runtime silently loses its credential.
    _token_file(tmp_path, monkeypatch, "file-token")
    monkeypatch.setenv(fyers_auth.TOKEN_ENV_VAR, "   ")
    assert fyers_auth.resolve_access_token() == "file-token"


def test_missing_everything_raises_named_error(monkeypatch, tmp_path):
    monkeypatch.setenv(fyers_auth.TOKEN_FILE_ENV_VAR, str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError) as exc:
        fyers_auth.resolve_access_token()
    # The message must name both remedies, since this is the cloud-vs-local
    # decision point for whoever hits it.
    assert fyers_auth.TOKEN_ENV_VAR in str(exc.value)
    assert "fyers_login.py" in str(exc.value)


def test_empty_file_falls_through_to_named_error(tmp_path, monkeypatch):
    _token_file(tmp_path, monkeypatch, "   \n")
    with pytest.raises(FileNotFoundError):
        fyers_auth.resolve_access_token()


def test_default_token_file_is_unchanged(monkeypatch):
    """Local behaviour must stay byte-identical to the old inline read."""
    monkeypatch.delenv(fyers_auth.TOKEN_FILE_ENV_VAR, raising=False)
    assert fyers_auth.token_file_path() == Path(".fyers_token")


def test_fingerprint_never_leaks_the_token():
    token = "SUPER-SECRET-ACCESS-TOKEN"
    fp = fyers_auth.token_fingerprint(token)
    assert token not in fp
    assert "SUPER" not in fp
    assert f"len={len(token)}" in fp
    # Stable for the same input, different for different input.
    assert fp == fyers_auth.token_fingerprint(token)
    assert fp != fyers_auth.token_fingerprint(token + "x")


def test_fingerprint_is_sha256_prefix():
    import hashlib
    token = "abc123"
    expected = hashlib.sha256(b"abc123").hexdigest()[:8]
    assert f"sha256={expected}" in fyers_auth.token_fingerprint(token)
