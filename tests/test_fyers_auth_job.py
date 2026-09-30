"""Tests for the daily Fyers token rotator (scripts/fyers_auth_job.py).

The contract under test is the TOTP login chain and the safety rules: the token,
PIN and TOTP code must never be printed, each hop must post the body Fyers
expects, Fyers' own error code/message must survive into the exception (so a
rejection like the SEBI ``-16`` is diagnosable from the log), and any failure
must exit non-zero rather than publish a stale token. All HTTP and Secret
Manager calls are faked.
"""
import base64
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import fyers_auth_job as mod

APP_ID = "ABCDEF1234-100"
USER_ID = "user-1234"
TOTP_SECRET = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"  # RFC 6238 base32 of "1234567890..."
PIN = "1234"


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Record sleeps instead of performing them, so the OTP-rollover guard is
    testable and the suite never stalls on the current wall-clock second."""
    slept = []
    monkeypatch.setattr(mod.time, "sleep", lambda s: slept.append(s))
    return slept


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeSession:
    """Returns queued responses in order and records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, data=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json,
                           "data": data, "timeout": timeout})
        if not self.responses:
            raise AssertionError(f"unexpected extra POST to {url}")
        nxt = self.responses.pop(0)
        return nxt if isinstance(nxt, FakeResponse) else FakeResponse(nxt)


def _happy_chain(access_token="NEW.TOKEN"):
    return [
        FakeResponse({"request_key": "rk-1"}),
        FakeResponse({"request_key": "rk-2"}),
        FakeResponse({"data": {"access_token": "BEARER"}}),
        FakeResponse({"Url": "https://cb/?auth_code=CODE-123&state=state"}, status=308),
        FakeResponse({"s": "ok", "code": 200, "access_token": access_token}),
    ]


def _login(session, **kwargs):
    args = dict(user_id=USER_ID, totp_key=TOTP_SECRET, pin=PIN, app_id=APP_ID,
                secret_key="SEC", http=session)
    args.update(kwargs)
    return mod.request_access_token(**args)


def _set_env(monkeypatch):
    monkeypatch.setenv("FYERS_APP_ID", APP_ID)
    monkeypatch.setenv("FYERS_SECRET_KEY", "SEC")
    monkeypatch.setenv("FYERS_USER_ID", USER_ID)
    monkeypatch.setenv("FYERS_TOTP_KEY", TOTP_SECRET)
    monkeypatch.setenv("FYERS_PIN", PIN)
    monkeypatch.setenv("GCP_PROJECT", "proj")


# --- TOTP -------------------------------------------------------------------

def test_totp_matches_rfc6238_sha1_vectors():
    # RFC 6238 appendix B, SHA-1 column, secret "12345678901234567890". The RFC
    # vectors are 8 digits, so ask for 8: that compares the raw truncation and
    # dynamic-offset arithmetic exactly, rather than only its last 6 digits.
    expected = {59: "94287082", 1111111109: "07081804",
                1111111111: "14050471", 1234567890: "89005924",
                2000000000: "69279037", 20000000000: "65353130"}
    for timestamp, code in expected.items():
        assert mod.totp_code("GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ",
                             timestamp=timestamp, digits=8) == code


def test_totp_six_digit_form_is_the_rfc_value_truncated():
    # Fyers' login takes a 6-digit OTP, i.e. the RFC's 8-digit value's last 6.
    assert mod.totp_code("GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ", timestamp=59) == "287082"


def test_totp_tolerates_lowercase_and_missing_padding():
    assert (mod.totp_code("gezdgnbvgy3tqojqgezdgnbvgy3tqojq", timestamp=59)
            == mod.totp_code("GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ", timestamp=59))


def test_totp_is_six_digits():
    assert len(mod.totp_code(TOTP_SECRET, timestamp=59)) == 6


# --- the chain --------------------------------------------------------------

def test_login_returns_access_token():
    session = FakeSession(_happy_chain())
    assert _login(session) == "NEW.TOKEN"
    assert [c["url"].split("/vagator/v2/")[-1].split("/")[-1] for c in session.calls[:3]] == [
        "send_login_otp_v2", "verify_otp", "verify_pin_v2"]


def test_login_sends_browser_headers_and_encodes_the_identifiers():
    session = FakeSession(_happy_chain())
    _login(session)
    send_otp, verify_otp, verify_pin = session.calls[:3]

    # The vagator service rejects requests without a browser-shaped User-Agent.
    assert "Mozilla" in send_otp["headers"]["User-Agent"]
    assert "Mozilla" in verify_pin["headers"]["User-Agent"]

    # fy_id and the PIN travel base64-encoded, per Fyers' login contract.
    assert send_otp["json"]["fy_id"] == base64.b64encode(USER_ID.encode()).decode()
    assert verify_pin["json"]["identifier"] == base64.b64encode(PIN.encode()).decode()
    assert verify_pin["json"]["identity_type"] == "pin"

    # The TOTP is a fresh 6-digit code, never a stored value.
    assert len(verify_otp["json"]["otp"]) == 6
    assert verify_otp["json"]["otp"].isdigit()


def test_login_chains_request_keys_between_steps():
    session = FakeSession(_happy_chain())
    _login(session)
    assert session.calls[0]["json"] == {"fy_id": session.calls[0]["json"]["fy_id"],
                                        "app_id": "2"}
    assert session.calls[1]["json"]["request_key"] == "rk-1"
    assert session.calls[2]["json"]["request_key"] == "rk-2"


def test_token_step_splits_the_client_id_and_bears_the_trade_token():
    session = FakeSession(_happy_chain())
    _login(session)
    token_step = session.calls[3]
    assert token_step["headers"]["Authorization"] == "Bearer BEARER"
    assert token_step["json"]["app_id"] == "ABCDEF1234"
    assert token_step["json"]["appType"] == "100"
    assert token_step["json"]["redirect_uri"] == mod.DEFAULT_REDIRECT_URI


def test_final_step_posts_the_documented_validate_authcode_body():
    session = FakeSession(_happy_chain())
    _login(session)
    final = session.calls[4]
    assert final["url"].endswith("/api/v3/validate-authcode")
    assert final["json"]["grant_type"] == "authorization_code"
    assert final["json"]["code"] == "CODE-123"
    assert final["json"]["appIdHash"] == mod.app_id_hash(APP_ID, "SEC")


def test_app_id_hash_is_sha256_of_app_id_colon_secret():
    assert mod.app_id_hash(APP_ID, "SEC") == hashlib.sha256(
        f"{APP_ID}:SEC".encode()).hexdigest()


# --- failure handling -------------------------------------------------------

def test_send_otp_without_request_key_raises():
    session = FakeSession([FakeResponse({"s": "ok", "code": 1043})])
    with pytest.raises(RuntimeError, match="send_login_otp_v2 returned no request_key"):
        _login(session)


def test_send_otp_accepts_the_1043_user_exist_success_shape():
    # Success at this step is code 1043/"user exist", which is not a 200-ish
    # value, so only `s` may be used to tell success from failure.
    session = FakeSession([FakeResponse({"s": "ok", "code": 1043,
                                         "message": "user exist",
                                         "request_key": "rk-1"})]
                          + _happy_chain()[1:])
    assert _login(session) == "NEW.TOKEN"


def test_verify_otp_failure_raises():
    session = FakeSession(_happy_chain()[:1] + [FakeResponse({"s": "error", "code": -8,
                                                             "message": "invalid otp"})])
    with pytest.raises(RuntimeError, match="verify_otp rejected"):
        _login(session)


def test_verify_pin_failure_raises():
    session = FakeSession(_happy_chain()[:2] + [FakeResponse({"s": "error", "code": -442,
                                                             "message": "bad pin"})])
    with pytest.raises(RuntimeError, match="verify_pin_v2 rejected"):
        _login(session)


def test_http_200_error_with_a_request_key_is_still_rejected():
    # The dangerous shape: a 200 whose body says error but which still carries a
    # request_key. Trusting the request_key alone would push a failed 2FA step
    # forward as if it had passed.
    session = FakeSession([FakeResponse({"s": "error", "code": -2,
                                         "message": "something went wrong",
                                         "request_key": "rk-bogus"})])
    with pytest.raises(RuntimeError, match="something went wrong"):
        _login(session)


def test_token_step_without_auth_code_raises():
    session = FakeSession(_happy_chain()[:3] + [FakeResponse({"Url": "https://cb/?x=1"},
                                                             status=308)])
    with pytest.raises(RuntimeError, match="no auth_code"):
        _login(session)


def test_final_step_rejection_surfaces_fyers_code_and_message():
    # The SEBI refresh-token ban arrived as exactly this shape; the message has
    # to survive into the log or the cause is unrecoverable from the output.
    body = {"s": "error", "code": -16,
            "message": "Refresh token API is currently disabled to comply with SEBI regulations."}
    session = FakeSession(_happy_chain()[:4]
                          + [FakeResponse(body, status=400)
                             for _ in mod.VALIDATE_AUTHCODE_CANDIDATES])
    with pytest.raises(RuntimeError) as excinfo:
        _login(session)
    assert "-16" in str(excinfo.value)
    assert "SEBI" in str(excinfo.value)


def test_final_step_missing_access_token_raises():
    session = FakeSession(_happy_chain()[:4] + [FakeResponse({"s": "ok", "code": 200})])
    with pytest.raises(RuntimeError, match="validate-authcode rejected"):
        _login(session)


def test_non_json_body_does_not_crash_the_error_path():
    class NotJson(FakeResponse):
        def json(self):
            raise ValueError("not json")

    session = FakeSession([NotJson(None, status=500)])
    with pytest.raises(RuntimeError, match="HTTP 500"):
        _login(session)


# --- Fyers' two endpoint pairs and two auth-code shapes ----------------------

def test_token_step_falls_back_to_the_second_host_version_pair():
    # Fyers serves this exchange on api.fyers.in/api/v2 and api-t1.fyers.in/api/v3
    # and working samples use either, so a 404 on the first must not abort.
    session = FakeSession(_happy_chain()[:3]
                          + [FakeResponse({"message": "not found"}, status=404)]
                          + _happy_chain()[3:])
    assert _login(session) == "NEW.TOKEN"
    attempted = [c["url"] for c in session.calls if c["url"].endswith("/token")]
    assert attempted == list(mod.TOKEN_URL_CANDIDATES)


def test_token_step_accepts_the_v3_data_auth_jwt_shape():
    # The v3 variant answers 200 with data.auth (a JWT) instead of a 308 redirect;
    # its `code` claim is the same auth code.
    def _jwt(claims):
        payload = base64.urlsafe_b64encode(
            json.dumps(claims).encode()).decode().rstrip("=")
        return f"hdr.{payload}.sig"

    session = FakeSession(_happy_chain()[:3]
                          + [FakeResponse({"s": "ok", "code": "",
                                           "data": {"auth": _jwt({"code": "CODE-JWT"})}})]
                          + _happy_chain()[4:])
    assert _login(session) == "NEW.TOKEN"
    assert session.calls[4]["json"]["code"] == "CODE-JWT"


def test_all_candidates_failing_surfaces_the_last_real_error():
    session = FakeSession(_happy_chain()[:3]
                          + [FakeResponse({"code": -16, "message": "SEBI"},
                                          status=400) for _ in mod.TOKEN_URL_CANDIDATES])
    with pytest.raises(RuntimeError, match="failed on all endpoints"):
        _login(session)


# --- OTP rollover guard ------------------------------------------------------

def _at_window_second(offset: int) -> float:
    """A timestamp landing on ``offset`` seconds within a 30s TOTP window."""
    return 1_700_000_000 - (1_700_000_000 % 30) + offset


def test_fresh_totp_waits_when_the_window_is_about_to_roll_over(no_real_sleep):
    # Second 29 of a 30s window: a code generated now expires mid-validation and
    # comes back as -1028/"otp might be expired".
    mod._fresh_totp(TOTP_SECRET, now=_at_window_second(29))
    assert no_real_sleep and no_real_sleep[0] > 0


def test_fresh_totp_does_not_wait_mid_window(no_real_sleep):
    mod._fresh_totp(TOTP_SECRET, now=_at_window_second(10))
    assert no_real_sleep == []


# --- main() safety rules ----------------------------------------------------

def test_main_missing_env_alerts_without_network(monkeypatch, capsys):
    for name in ("FYERS_APP_ID", "FYERS_SECRET_KEY", "FYERS_USER_ID", "FYERS_TOTP_KEY",
                 "FYERS_PIN", "GCP_PROJECT"):
        monkeypatch.delenv(name, raising=False)
    assert mod.main() == 1
    assert "ALERT" in capsys.readouterr().err


def test_main_missing_project_alerts(monkeypatch, capsys):
    _set_env(monkeypatch)
    for name in ("GCP_PROJECT", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        monkeypatch.delenv(name, raising=False)
    assert mod.main() == 1
    assert "ALERT" in capsys.readouterr().err


def test_main_stores_new_version_and_never_prints_the_token(monkeypatch, capsys):
    _set_env(monkeypatch)
    token = "SUPER-SECRET-ACCESS-TOKEN-xyz"
    monkeypatch.setattr(mod, "request_access_token", lambda *a, **k: token)
    captured = {}

    def _store(tok, **kwargs):
        captured["token"] = tok
        captured.update(kwargs)
        return "projects/proj/secrets/FYERS_ACCESS_TOKEN/versions/3"

    monkeypatch.setattr(mod, "add_access_token_version", _store)

    assert mod.main() == 0
    assert captured["token"] == token
    assert captured["project"] == "proj"
    assert captured["secret_name"] == "FYERS_ACCESS_TOKEN"
    out = capsys.readouterr().out
    assert token not in out
    assert PIN not in out


def test_main_login_failure_is_an_alert_and_stores_nothing(monkeypatch, capsys):
    _set_env(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("verify_otp failed: code=-8 invalid otp")

    monkeypatch.setattr(mod, "request_access_token", _boom)
    stored = []
    monkeypatch.setattr(mod, "add_access_token_version", lambda *a, **k: stored.append(1))

    assert mod.main() == 1
    assert stored == []
    err = capsys.readouterr().err
    assert "ALERT" in err and "-8" in err


def test_main_store_failure_is_an_alert(monkeypatch, capsys):
    _set_env(monkeypatch)
    monkeypatch.setattr(mod, "request_access_token", lambda *a, **k: "TOK")

    def _boom(*a, **k):
        raise PermissionError("secretVersionAdder denied")

    monkeypatch.setattr(mod, "add_access_token_version", _boom)

    assert mod.main() == 1
    assert "ALERT" in capsys.readouterr().err
