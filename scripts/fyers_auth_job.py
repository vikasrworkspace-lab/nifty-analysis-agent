"""Daily Fyers access-token rotator (Cloud Run job ``nifty-fyers-auth``).

A Fyers access token is valid for one trading day. Rather than have a human mint
one every morning, this job runs shortly before the open and performs a fully
server-side TOTP login, then stores the fresh token as a new version of the
``FYERS_ACCESS_TOKEN`` secret. The data jobs bind ``FYERS_ACCESS_TOKEN:latest``,
so they pick the new version up on their next execution with no redeploy.

Why TOTP and not the refresh token
----------------------------------
Fyers disabled ``/api/v3/validate-refresh-token`` platform-wide, answering
``{"code": -16, "message": "Refresh token API is currently disabled to comply
with SEBI regulations."}``. SEBI's retail-algo framework requires 2FA once per
trading day and does not permit continuous refresh-token sessions, so the old
refresh-and-rotate design could never work regardless of coding. Fyers points
users at TOTP for automated sessions instead.

Unlike the refresh token, a TOTP secret has **no expiry cycle**: it stays valid
until it is regenerated in the Fyers account portal. This job therefore has no
periodic human step.

The five-step flow (the ``api-t2``/``vagator`` pair is Fyers' login service)::

    1. POST api-t2/vagator/v2/send_login_otp_v2  -> request_key
    2. POST api-t2/vagator/v2/verify_otp         (TOTP) -> request_key
    3. POST api-t2/vagator/v2/verify_pin_v2      -> trade-level bearer token
    4. POST .../token                             (Bearer) -> auth_code
    5. POST .../validate-authcode                 -> access_token

Steps 4 and 5 are attempted against both of Fyers' host/version pairs
(``api.fyers.in/api/v2`` and ``api-t1.fyers.in/api/v3``), which is why they go
through ``_first_working``: samples in the wild use either, and the v3 variant
returns the auth code inside a ``data.auth`` JWT rather than a redirect.

Step 5 talks to Fyers' documented endpoint directly instead of going through the
``fyers-apiv3`` SDK, so the job needs no vendor SDK at runtime and the whole
exchange is unit-testable with a fake HTTP session.

Deliberate limits:

* Tokens, the PIN and the TOTP code are never printed -- only
  ``token_fingerprint`` (length + truncated SHA-256), matching the rest of the
  project.
* Every Fyers response is parsed before its status is trusted, so a rejection
  reports Fyers' own ``code``/``message``. A bare ``raise_for_status()`` hid the
  ``-16`` above behind a generic ``HTTPError: 400``, which cost a debugging
  round-trip.
* HTTP 200 is also not trusted: the vagator endpoints answer 200 with
  ``s: "error"`` for a stale OTP, and signal success with code 1043.
* The TOTP code is generated away from the end of its 30s window, because a code
  that rolls over during validation comes back as ``-1028``/``otp might be
  expired``.
* On any failure it logs ``ALERT`` and exits non-zero; it never falls back to a
  stale token, so a broken rotation is visible instead of silent.

Required environment (bound from Secret Manager by the job):

    FYERS_APP_ID, FYERS_SECRET_KEY, FYERS_USER_ID, FYERS_TOTP_KEY, FYERS_PIN

and ``FYERS_REDIRECT_URI`` (plain env -- it is a registered callback, not a
credential). ``FYERS_REFRESH_TOKEN`` is no longer read and can be deleted.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import struct
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from core.fyers_auth import token_fingerprint

DEFAULT_API_BASE = "https://api-t1.fyers.in/api/v3"
DEFAULT_VAGATOR_BASE = "https://api-t2.fyers.in/vagator/v2"
DEFAULT_REDIRECT_URI = "https://vikasrworkspace-lab.github.io/nifty-analysis-agent/"
DEFAULT_ACCESS_SECRET = "FYERS_ACCESS_TOKEN"

# Fyers serves the auth-code exchange on two host/version pairs and working
# sample code uses both: ``api.fyers.in/api/v2`` and ``api-t1.fyers.in/api/v3``.
# Rather than bet on one and burn a debugging round-trip on a 404, try the
# candidates in order and report whichever pair actually worked.
TOKEN_URL_CANDIDATES = (
    "https://api.fyers.in/api/v2/token",
    "https://api-t1.fyers.in/api/v3/token",
)
VALIDATE_AUTHCODE_CANDIDATES = (
    "https://api-t1.fyers.in/api/v3/validate-authcode",
    "https://api.fyers.in/api/v2/validate-authcode",
)

# Fyers returns HTTP 200 for some business-level failures (``s: "error"``, e.g.
# ``-2``/``-1028`` on a stale OTP), so status alone is not enough to trust a step.
# ``send_login_otp_v2`` in particular signals success with code 1043/"user exist".
ERROR_SENTINEL = "error"

# The vagator endpoints are an internal login service that rejects requests
# without a browser-shaped User-Agent, so they are not the plain API client.
BROWSER_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def app_id_hash(app_id: str, secret_key: str) -> str:
    """``sha256("<app_id>:<secret_key>")`` hex digest, as Fyers documents it."""
    return hashlib.sha256(f"{app_id}:{secret_key}".encode("utf-8")).hexdigest()


def totp_code(secret: str, *, timestamp: float | None = None,
              step: int = 30, digits: int = 6) -> str:
    """RFC 6238 TOTP over Fyers' base32 secret.

    Implemented inline rather than via ``pyotp`` so the container image needs no
    extra dependency, and so the algorithm is verifiable against the RFC's
    published test vectors in the tests.
    """
    padded = secret.strip().upper()
    padded += "=" * ((8 - len(padded) % 8) % 8)
    key = base64.b32decode(padded, casefold=True)
    counter = int((time.time() if timestamp is None else timestamp) // step)
    digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    truncated = struct.unpack(">L", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(truncated % (10 ** digits)).zfill(digits)


def _detail(body) -> str:
    """Fyers' own error code/message, for an ALERT line."""
    if isinstance(body, dict):
        bits = [str(body.get(k)) for k in ("code", "message") if body.get(k) is not None]
        if bits:
            return "code=" + " message=".join(bits)
    return "no error detail in response"


def _require_ok(body, step: str):
    """Raise if Fyers signalled a business-level failure.

    The vagator endpoints answer HTTP 200 with ``s: "error"`` for bad or expired
    OTPs, so ``raise_for_status()`` cannot be trusted to catch them.
    """
    if isinstance(body, dict) and str(body.get("s", "")).lower() == ERROR_SENTINEL:
        raise RuntimeError(f"{step} rejected the request: {_detail(body)}")
    return body


def _jwt_claim(token: str, claim: str):
    """Read one claim out of a JWT payload without verifying its signature.

    Used only to lift the auth code out of Fyers' ``data.auth`` token; the
    signature is Fyers' to check in ``validate-authcode``.
    """
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, TypeError):
        return None
    return claims.get(claim)


def _extract_auth_code(body) -> str | None:
    """Pull the auth code out of whichever shape Fyers returned.

    Two forms are in circulation: the v2 endpoint answers HTTP 308 with a
    redirect ``Url`` carrying ``auth_code``, while the v3 endpoint answers 200
    with ``data.auth``, a JWT whose ``code`` claim is the same value.
    """
    if isinstance(body, dict):
        redirect = body.get("Url")
        if redirect:
            codes = parse_qs(urlparse(redirect).query).get("auth_code")
            if codes:
                return codes[0]
        auth = (body.get("data") or {}).get("auth")
        if auth:
            return _jwt_claim(auth, "code")
    return None


def _fresh_totp(secret: str, *, step: int = 30, digits: int = 6,
                guard_seconds: int = 3, now: float | None = None) -> str:
    """TOTP code that is not about to roll over mid-request.

    A code generated in the last few seconds of its 30s window expires while
    Fyers is still validating it, which surfaces as ``-1028``/"otp might be
    expired". Waiting out the tail of the window is the documented fix and costs
    at most ~30s once a day, before the open.
    """
    current = time.time() if now is None else now
    remaining = step - (int(current) % step)
    if remaining < guard_seconds:
        time.sleep(guard_seconds - remaining + 0.5)
    return totp_code(secret, digits=digits)


def _post_json(http, url: str, *, headers=None, json=None, data=None,
               timeout: float = 30, expect=(200,)):
    """POST, then validate status *and* body, reporting Fyers' own error text.

    ``requests`` does not follow 308, so the ``/api/v2/token`` redirect carrying
    the auth code comes back as a normal response and 308 is a success here.
    """
    response = http.post(url, headers=headers, json=json, data=data, timeout=timeout)
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code not in expect:
        raise RuntimeError(f"HTTP {response.status_code} from {url}: {_detail(body)}")
    return body


def _first_working(http, step: str, urls, *, headers=None, json=None,
                   timeout: float = 30, expect=(200,)):
    """Try each candidate endpoint in turn; return the first usable response.

    Fyers exposes the auth exchange on both ``api.fyers.in/api/v2`` and
    ``api-t1.fyers.in/api/v3``, and samples in the wild use either. Trying both
    keeps a wrong guess from becoming a debugging round-trip, while still
    surfacing the last real error if none work.
    """
    last_error = None
    for url in urls:
        try:
            body = _post_json(http, url, headers=headers, json=json,
                              timeout=timeout, expect=expect)
            _require_ok(body, f"{step} at {url}")
        except RuntimeError as exc:
            last_error = exc
            continue
        return body
    raise RuntimeError(f"{step} failed on all endpoints: {last_error}")


def request_access_token(
    user_id: str,
    totp_key: str,
    pin: str,
    app_id: str,
    secret_key: str,
    redirect_uri: str = DEFAULT_REDIRECT_URI,
    *,
    vagator_base: str = DEFAULT_VAGATOR_BASE,
    api_base: str = DEFAULT_API_BASE,
    token_url: str | None = None,
    validate_urls: tuple[str, ...] | None = None,
    timeout: float = 30,
    http=None,
) -> str:
    """Log in end to end via TOTP and return an API access token.

    ``http`` is injectable so tests can supply a fake HTTP client. ``token_url``
    and ``validate_urls`` override the endpoint candidates when given.
    """
    http = http or requests
    base = vagator_base.rstrip("/")

    # 1. Kick off the login and get a request_key to bind the factors to.
    #    Success here is code 1043/"user exist", so only `s` distinguishes it
    #    from a failure -- never the numeric code.
    body = _post_json(http, f"{base}/send_login_otp_v2", headers=BROWSER_HEADERS,
                      json={"fy_id": base64.b64encode(user_id.encode()).decode(),
                            "app_id": "2"}, timeout=timeout)
    _require_ok(body, "send_login_otp_v2")
    request_key = (body or {}).get("request_key")
    if not request_key:
        raise RuntimeError(f"send_login_otp_v2 returned no request_key: {_detail(body)}")

    # 2. Satisfy 2FA with a freshly computed TOTP.
    body = _post_json(http, f"{base}/verify_otp", headers=BROWSER_HEADERS,
                      json={"request_key": request_key,
                            "otp": _fresh_totp(totp_key)},
                      timeout=timeout)
    _require_ok(body, "verify_otp")
    request_key = (body or {}).get("request_key")
    if not request_key:
        raise RuntimeError(f"verify_otp failed: {_detail(body)}")

    # 3. Satisfy the PIN, which returns a trade-level bearer token.
    body = _post_json(http, f"{base}/verify_pin_v2", headers=BROWSER_HEADERS,
                      json={"request_key": request_key, "identity_type": "pin",
                            "identifier": base64.b64encode(pin.encode()).decode()},
                      timeout=timeout)
    _require_ok(body, "verify_pin_v2")
    bearer = ((body or {}).get("data") or {}).get("access_token")
    if not bearer:
        raise RuntimeError(f"verify_pin_v2 failed: {_detail(body)}")

    # 4. Trade that bearer for an auth code bound to our app. appType is the
    #    suffix of the client id ("...-100"), the rest is the id itself.
    body = _first_working(
        http, "token",
        (token_url,) if token_url else TOKEN_URL_CANDIDATES,
        headers={"Authorization": f"Bearer {bearer}", **BROWSER_HEADERS},
        json={"fyers_id": user_id, "app_id": app_id.rsplit("-", 1)[0],
              "redirect_uri": redirect_uri, "appType": app_id.rsplit("-", 1)[-1],
              "code_challenge": "", "state": "state", "scope": "",
              "nonce": "", "response_type": "code", "create_cookie": True},
        timeout=timeout, expect=(200, 308),
    )
    auth_code = _extract_auth_code(body)
    if not auth_code:
        raise RuntimeError(f"token step returned no auth_code: {_detail(body)}")

    # 5. Exchange the auth code for the API access token the data jobs use.
    body = _first_working(
        http, "validate-authcode", validate_urls or VALIDATE_AUTHCODE_CANDIDATES,
        headers=BROWSER_HEADERS,
        json={"grant_type": "authorization_code",
              "appIdHash": app_id_hash(app_id, secret_key),
              "code": auth_code}, timeout=timeout)
    if not isinstance(body, dict) or body.get("s") != "ok" or not body.get("access_token"):
        raise RuntimeError(f"validate-authcode rejected the auth code: {_detail(body)}")
    return body["access_token"]


def add_access_token_version(
    access_token: str, *, project: str, secret_name: str, client=None
) -> str:
    """Add ``access_token`` as a new version of ``secret_name``; return its name."""
    from google.cloud import secretmanager

    client = client or secretmanager.SecretManagerServiceClient()
    parent = f"projects/{project}/secrets/{secret_name}"
    response = client.add_secret_version(
        request={"parent": parent, "payload": {"data": access_token.encode("utf-8")}}
    )
    return response.name


def _required_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def _project() -> str:
    for name in ("GCP_PROJECT", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def main() -> int:
    try:
        app_id = _required_env("FYERS_APP_ID")
        secret_key = _required_env("FYERS_SECRET_KEY")
        user_id = _required_env("FYERS_USER_ID")
        totp_key = _required_env("FYERS_TOTP_KEY")
        pin = _required_env("FYERS_PIN")
    except RuntimeError as exc:
        print(f"[fyers-auth] ALERT: {exc}", file=sys.stderr, flush=True)
        return 1

    project = _project()
    if not project:
        print(
            "[fyers-auth] ALERT: no GCP project (set GCP_PROJECT).",
            file=sys.stderr,
            flush=True,
        )
        return 1

    secret_name = (os.getenv("FYERS_ACCESS_SECRET") or DEFAULT_ACCESS_SECRET).strip()
    redirect_uri = (os.getenv("FYERS_REDIRECT_URI") or DEFAULT_REDIRECT_URI).strip()
    vagator_base = (os.getenv("FYERS_VAGATOR_BASE") or DEFAULT_VAGATOR_BASE).strip()
    token_url = (os.getenv("FYERS_TOKEN_URL") or "").strip() or None
    validate_urls = (os.getenv("FYERS_VALIDATE_AUTHCODE_URLS") or "").strip()
    if validate_urls:
        validate_urls = tuple(u.strip() for u in validate_urls.split(",") if u.strip())
    else:
        validate_urls = None

    try:
        token = request_access_token(
            user_id, totp_key, pin, app_id, secret_key,
            redirect_uri=redirect_uri, vagator_base=vagator_base,
            token_url=token_url, validate_urls=validate_urls,
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure as an ALERT
        print(
            f"[fyers-auth] ALERT: TOTP login failed ({type(exc).__name__}: {exc}). "
            f"Check FYERS_USER_ID / FYERS_TOTP_KEY / FYERS_PIN, and that TOTP 2FA "
            f"is still enabled on the account.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(f"[fyers-auth] obtained access token: {token_fingerprint(token)}", flush=True)

    try:
        version = add_access_token_version(
            token, project=project, secret_name=secret_name
        )
    except Exception as exc:  # noqa: BLE001 - IAM/API failure must be visible
        print(
            f"[fyers-auth] ALERT: could not store a new {secret_name} version "
            f"({type(exc).__name__}: {exc}). Check secretVersionAdder IAM.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(f"[fyers-auth] stored {version}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
