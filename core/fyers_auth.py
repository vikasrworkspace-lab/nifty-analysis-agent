"""Centralised Fyers access-token resolution.

Single source of truth for how the project obtains a Fyers access token, so
interactive local use and the Cloud Run jobs resolve credentials identically.
Both former call sites inlined their own copy of the file read; they now
delegate here.

Resolution order:

1. ``FYERS_ACCESS_TOKEN`` environment variable
2. ``.fyers_token`` file (path overridable with ``FYERS_TOKEN_FILE``)

An access token is valid for one trading day. It is renewed out-of-band by the
``nifty-fyers-auth`` Cloud Run job (``scripts/fyers_auth_job.py``), which performs
a fully server-side TOTP login and publishes the result as a new
``FYERS_ACCESS_TOKEN`` secret version. This module deliberately does no
refresh/renewal itself: consumers only resolve the token, and the login needs
the account PIN, TOTP key and secret-write IAM that the data jobs must not hold.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

TOKEN_ENV_VAR = "FYERS_ACCESS_TOKEN"
TOKEN_FILE_ENV_VAR = "FYERS_TOKEN_FILE"
DEFAULT_TOKEN_FILE = ".fyers_token"


def token_file_path() -> Path:
    """Path of the on-disk token cache used when the env var is unset."""
    return Path(os.getenv(TOKEN_FILE_ENV_VAR) or DEFAULT_TOKEN_FILE)


def token_fingerprint(token: str) -> str:
    """Log-safe description of a token: length plus a truncated SHA-256.

    Never returns any part of the token itself, so it is safe to print.
    """
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]
    return f"len={len(token)} sha256={digest}"


def resolve_access_token() -> str:
    """Return a usable Fyers access token, or raise if none can be found.

    Raises ``FileNotFoundError`` to stay compatible with the previous inline
    behaviour, and with callers that treat a missing token as a soft failure.
    """
    env_token = (os.getenv(TOKEN_ENV_VAR) or "").strip()
    if env_token:
        return env_token

    path = token_file_path()
    if path.exists():
        file_token = path.read_text(encoding="utf-8").strip()
        if file_token:
            return file_token

    raise FileNotFoundError(
        f"No Fyers access token available. Set the {TOKEN_ENV_VAR} environment "
        f"variable (Cloud Run / CI), or run scripts/fyers_login.py to create "
        f"{path}."
    )
