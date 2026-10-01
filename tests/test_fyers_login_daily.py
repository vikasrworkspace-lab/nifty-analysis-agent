"""Tests for the daily manual Fyers login step (scripts/fyers_login.py).

Two contracts matter here:

* **No refresh token.** Fyers disabled the refresh API platform-wide, so the
  script must not write ``FYERS_REFRESH_TOKEN`` at all. It used to, which cost a
  second confirmation prompt every morning and seeded a credential nothing could
  ever use.
* **Verify after push.** A token that Fyers rejects is otherwise discovered at
  09:00, when data silently stops flowing. The morning run must fail instead.

All gcloud interaction is faked; nothing here touches the network or a project.
"""
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import fyers_login

TOKEN = "SECRET-ACCESS-TOKEN-VALUE-abc123"


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


class FakeSession:
    """Stands in for fyersModel.SessionModel."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate_authcode(self):
        return "https://api.fyers.in/auth?x=1"

    def set_token(self, auth_code):
        self.auth_code = auth_code

    def generate_token(self):
        # Fyers also returns a refresh_token in this response. It is deliberately
        # present here to prove the script ignores it.
        return {"access_token": TOKEN, "refresh_token": "REFRESH-SHOULD-BE-IGNORED"}


@pytest.fixture
def login_run(monkeypatch, tmp_path):
    """Drive main() end to end with the browser flow and gcloud stubbed out."""
    monkeypatch.setattr(fyers_login, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("FYERS_APP_ID", "ABCDEF1234-100")
    monkeypatch.setenv("FYERS_SECRET_KEY", "SEC")
    monkeypatch.setenv("FYERS_REDIRECT_URI", "https://example.test/cb")
    monkeypatch.setenv("K_SERVICE", "")  # not "inside Cloud Run"
    monkeypatch.delenv("FYERS_SECRET_NAME", raising=False)
    monkeypatch.setattr(fyers_login.fyersModel, "SessionModel", FakeSession)
    monkeypatch.setattr("builtins.input", lambda *a: "AUTH-CODE")
    monkeypatch.chdir(tmp_path)

    pushed = []
    verified = []
    monkeypatch.setattr(
        fyers_login, "push_secret",
        lambda tok, name, proj: pushed.append((tok, name, proj)) or 0,
    )
    monkeypatch.setattr(
        fyers_login, "verify_token",
        lambda tok, cid: verified.append((tok, cid)) or (True, "Fyers accepted the token"),
    )

    state = {"pushed": pushed, "verified": verified}
    return state


def _run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fyers_login.py", *argv])
    return fyers_login.main()


# --- no refresh token --------------------------------------------------------

def test_push_writes_only_the_access_token(login_run, monkeypatch):
    # Fyers returns a refresh_token in the same response. Seeding it was the old
    # behaviour and must not come back.
    assert _run(["--push-secret"], monkeypatch) == 0
    assert [name for _, name, _ in login_run["pushed"]] == ["FYERS_ACCESS_TOKEN"]


def test_refresh_secret_is_never_referenced(login_run, monkeypatch, capsys):
    assert _run(["--push-secret"], monkeypatch) == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "REFRESH-SHOULD-BE-IGNORED" not in out
    assert "FYERS_REFRESH_TOKEN" not in out


def test_no_refresh_secret_flag_exists(login_run, monkeypatch):
    # The CLI must not even offer it, so nobody can pass it back in.
    with pytest.raises(SystemExit):
        _run(["--push-secret", "--refresh-secret-name", "FYERS_REFRESH_TOKEN"], monkeypatch)


def test_secret_name_defaults_to_access_token(login_run, monkeypatch):
    assert _run(["--push-secret"], monkeypatch) == 0
    assert login_run["pushed"][0][1] == "FYERS_ACCESS_TOKEN"


def test_explicit_secret_name_still_wins(login_run, monkeypatch):
    assert _run(["--push-secret", "--secret-name", "OTHER"], monkeypatch) == 0
    assert login_run["pushed"][0][1] == "OTHER"


def test_env_secret_name_is_honoured(login_run, monkeypatch):
    monkeypatch.setenv("FYERS_SECRET_NAME", "FROM_ENV")
    assert _run(["--push-secret"], monkeypatch) == 0
    assert login_run["pushed"][0][1] == "FROM_ENV"


def test_without_push_secret_nothing_is_pushed_or_verified(login_run, monkeypatch):
    assert _run([], monkeypatch) == 0
    assert login_run["pushed"] == []
    assert login_run["verified"] == []


# --- verification ------------------------------------------------------------

def test_verification_runs_after_the_push(login_run, monkeypatch):
    assert _run(["--push-secret"], monkeypatch) == 0
    assert login_run["verified"], "token was never verified"
    assert login_run["verified"][0][0] == TOKEN


def test_verification_failure_exits_non_zero(login_run, monkeypatch, capsys):
    # The whole point: a rejected token must fail here, not silently at 09:00.
    monkeypatch.setattr(
        fyers_login, "verify_token",
        lambda tok, cid: (False, "HTTP 400: code=-209 message=Please provide the validation parameter"),
    )
    assert _run(["--push-secret"], monkeypatch) == 1
    captured = capsys.readouterr()
    assert "TOKEN FAILED" in captured.out
    # Fyers' own code/message must survive into the morning output, and the run
    # must also say out loud that the data jobs will fail.
    assert "-209" in captured.out
    assert "data jobs will fail today" in captured.err


def test_skip_verify_avoids_the_call(login_run, monkeypatch):
    assert _run(["--push-secret", "--skip-verify"], monkeypatch) == 0
    assert login_run["verified"] == []


def test_push_failure_short_circuits_before_verification(login_run, monkeypatch):
    monkeypatch.setattr(fyers_login, "push_secret", lambda *a: 1)
    assert _run(["--push-secret"], monkeypatch) == 1
    assert login_run["verified"] == [], "verified a token that was never pushed"


def test_token_is_never_printed(login_run, monkeypatch, capsys):
    assert _run(["--push-secret"], monkeypatch) == 0
    captured = capsys.readouterr()
    assert TOKEN not in captured.out
    assert TOKEN not in captured.err


# --- verify_token -----------------------------------------------------------

class _Recorder:
    def __init__(self, response=None, raises=None):
        self.response = response
        self.raises = raises
        self.kwargs = None

    def get(self, url, headers=None, timeout=None):
        self.kwargs = {"url": url, "headers": headers, "timeout": timeout}
        if self.raises is not None:
            raise self.raises
        return self.response


def test_verify_token_accepts_a_200_ok(monkeypatch):
    rec = _Recorder(FakeResponse({"s": "ok", "fy_id": "XX", "mobile_no": "98xxxxxx82"}))
    monkeypatch.setattr(requests, "get", rec.get)
    ok, detail = fyers_login.verify_token(TOKEN, "ABCDEF1234-100")
    assert ok and "accepted" in detail
    # The token travels in the Authorization header, never in the URL or argv.
    # Fyers wants "APP_ID:TOKEN", not "Bearer <token>": a Bearer header is
    # unparseable and returns HTTP 400 / -209 even for a valid token.
    assert rec.kwargs["headers"]["Authorization"] == f"ABCDEF1234-100:{TOKEN}"
    assert rec.kwargs["headers"]["version"] == "3"
    assert TOKEN not in rec.kwargs["url"]


def test_verify_token_rejects_fyers_error_body(monkeypatch):
    rec = _Recorder(FakeResponse({"s": "error", "code": -209,
                                  "message": "Please provide the validation parameter"}, 400))
    monkeypatch.setattr(requests, "get", rec.get)
    ok, detail = fyers_login.verify_token(TOKEN, "APP")
    assert not ok
    assert "-209" in detail and "validation parameter" in detail


def test_verify_token_rejects_a_200_that_is_not_ok(monkeypatch):
    # Status 200 alone is not success on Fyers; `s` decides.
    rec = _Recorder(FakeResponse({"s": "error", "code": -16, "message": "SEBI"}))
    monkeypatch.setattr(requests, "get", rec.get)
    ok, detail = fyers_login.verify_token(TOKEN, "APP")
    assert not ok and "-16" in detail


def test_verify_token_reports_network_failure(monkeypatch):
    rec = _Recorder(raises=OSError("no route to host"))
    monkeypatch.setattr(requests, "get", rec.get)
    ok, detail = fyers_login.verify_token(TOKEN, "APP")
    assert not ok
    assert "could not reach Fyers" in detail


def test_verify_token_reports_non_json_body(monkeypatch):
    class NotJson:
        status_code = 502

        def json(self):
            raise ValueError("html")

    monkeypatch.setattr(requests, "get", _Recorder(NotJson()).get)
    ok, detail = fyers_login.verify_token(TOKEN, "APP")
    assert not ok and "502" in detail


def test_verify_token_never_echoes_the_token(monkeypatch):
    rec = _Recorder(FakeResponse({"s": "error", "code": -1, "message": "nope"}, 400))
    monkeypatch.setattr(requests, "get", rec.get)
    _ok, detail = fyers_login.verify_token(TOKEN, "APP")
    assert TOKEN not in detail
