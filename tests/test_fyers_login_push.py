"""Safety tests for the opt-in ``--push-secret`` path in scripts/fyers_login.py.

The point of these tests is the *constraints*, not the happy path: the token
must never be printed, never appear in a process argument vector, never be
written anywhere, and the secret must never be created or overwritten silently.
All gcloud interaction is faked; nothing here touches the network or a real
project.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import fyers_login

TOKEN = "SECRET-ACCESS-TOKEN-VALUE-abc123"


class FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def fake_gcloud(monkeypatch):
    """Record every gcloud invocation and script the canned responses."""
    calls = []
    responses = {}

    def _run(exe, args, stdin_data=None):
        calls.append({"args": list(args), "stdin": stdin_data})
        for key, result in responses.items():
            if key in args:
                return result
        return FakeResult()

    monkeypatch.setattr(fyers_login, "_gcloud_executable", lambda: "gcloud")
    monkeypatch.setattr(fyers_login, "_gcloud", _run)
    return {"calls": calls, "responses": responses}


def _argv_blob(recorder):
    return " ".join(" ".join(c["args"]) for c in recorder["calls"])


def test_refuses_to_run_inside_cloud_run(monkeypatch):
    # Condition: never run automatically from Cloud Run.
    monkeypatch.setenv("K_SERVICE", "my-intraday-job")
    called = []
    monkeypatch.setattr(fyers_login, "_gcloud_executable",
                        lambda: called.append("exe") or "gcloud")
    assert fyers_login.push_secret(TOKEN, "my-secret", "proj") == 1
    assert called == []


def test_fails_clearly_when_gcloud_missing(monkeypatch, capsys):
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.setattr(fyers_login, "_gcloud_executable", lambda: None)
    assert fyers_login.push_secret(TOKEN, "my-secret", "proj") == 1
    assert "gcloud" in capsys.readouterr().out


def test_fails_when_no_active_project(monkeypatch, capsys):
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.setattr(fyers_login, "_gcloud_executable", lambda: "gcloud")
    monkeypatch.setattr(fyers_login, "_active_project", lambda exe: None)
    assert fyers_login.push_secret(TOKEN, "my-secret", None) == 1
    assert "no active gcloud project" in capsys.readouterr().out.lower()


def test_missing_secret_is_an_error_and_is_never_created(fake_gcloud, capsys):
    # Condition: the secret must not be created here.
    fake_gcloud["responses"]["describe"] = FakeResult(returncode=1)
    assert fyers_login.push_secret(TOKEN, "my-secret", "proj") == 1
    assert "not found" in capsys.readouterr().out
    assert "secrets create" not in _argv_blob(fake_gcloud)


def test_token_is_never_printed(fake_gcloud, monkeypatch, capsys):
    fake_gcloud["responses"]["describe"] = FakeResult()
    fake_gcloud["responses"]["list"] = FakeResult(stdout="1\n")
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    fyers_login.push_secret(TOKEN, "my-secret", "proj")
    assert TOKEN not in capsys.readouterr().out


def test_token_is_never_passed_as_a_command_argument(fake_gcloud, monkeypatch):
    # Condition: pipe over stdin (--data-file=-), never inline in argv.
    fake_gcloud["responses"]["describe"] = FakeResult()
    fake_gcloud["responses"]["list"] = FakeResult(stdout="1\n")
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    fyers_login.push_secret(TOKEN, "my-secret", "proj")

    assert TOKEN not in _argv_blob(fake_gcloud)
    add = [c for c in fake_gcloud["calls"] if "add" in c["args"]]
    assert len(add) == 1
    assert "--data-file=-" in add[0]["args"]
    assert add[0]["stdin"] == TOKEN  # the token travels only via stdin


def test_token_is_never_written_to_disk(fake_gcloud, monkeypatch, tmp_path):
    # Condition: never written to a normal file by the push path.
    fake_gcloud["responses"]["describe"] = FakeResult()
    fake_gcloud["responses"]["list"] = FakeResult(stdout="1\n")
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    monkeypatch.chdir(tmp_path)
    fyers_login.push_secret(TOKEN, "my-secret", "proj")
    assert list(tmp_path.iterdir()) == []


def test_overwrite_requires_explicit_confirmation(fake_gcloud, monkeypatch, capsys):
    # Condition: never silently overwrite.
    fake_gcloud["responses"]["describe"] = FakeResult()
    fake_gcloud["responses"]["list"] = FakeResult(stdout="1\n2\n")

    for answer in ("", "n", "no", "N", "maybe"):
        monkeypatch.setattr("builtins.input", lambda *a, _a=answer: _a)
        assert fyers_login.push_secret(TOKEN, "my-secret", "proj") == 1
        assert not [c for c in fake_gcloud["calls"] if "add" in c["args"]]

    # The new version number is shown before asking.
    out = capsys.readouterr().out
    assert "version 3" in out
    assert "existing versions: 2" in out


def test_only_affirmative_answers_proceed(fake_gcloud, monkeypatch):
    # Case-insensitive affirmation is still explicit, so it proceeds; anything
    # that is not an unambiguous yes must abort without adding a version.
    fake_gcloud["responses"]["describe"] = FakeResult()
    fake_gcloud["responses"]["list"] = FakeResult(stdout="")

    for answer in ("n", "N", "no", "", "  ", "maybe", "sure", "1", "ok"):
        monkeypatch.setattr("builtins.input", lambda *a, _a=answer: _a)
        assert fyers_login.push_secret(TOKEN, "my-secret", "proj") == 1
    assert not [c for c in fake_gcloud["calls"] if "add" in c["args"]]

    for answer in ("y", "Y", "yes", "YES", " yes "):
        monkeypatch.setattr("builtins.input", lambda *a, _a=answer: _a)
        assert fyers_login.push_secret(TOKEN, "my-secret", "proj") == 0
    assert len([c for c in fake_gcloud["calls"] if "add" in c["args"]]) == 5


def test_ctrl_c_at_prompt_aborts_without_adding(fake_gcloud, monkeypatch):
    fake_gcloud["responses"]["describe"] = FakeResult()
    fake_gcloud["responses"]["list"] = FakeResult(stdout="")

    def _raise(*_a):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise)
    assert fyers_login.push_secret(TOKEN, "my-secret", "proj") == 1
    assert not [c for c in fake_gcloud["calls"] if "add" in c["args"]]


def test_explicit_y_adds_exactly_one_version(fake_gcloud, monkeypatch):
    fake_gcloud["responses"]["describe"] = FakeResult()
    fake_gcloud["responses"]["list"] = FakeResult(stdout="1\n")
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    assert fyers_login.push_secret(TOKEN, "my-secret", "proj") == 0
    assert len([c for c in fake_gcloud["calls"] if "add" in c["args"]]) == 1


def test_project_flag_is_used_when_supplied(fake_gcloud, monkeypatch):
    fake_gcloud["responses"]["describe"] = FakeResult()
    fake_gcloud["responses"]["list"] = FakeResult(stdout="")
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    fyers_login.push_secret(TOKEN, "my-secret", "explicit-project")
    assert "--project=explicit-project" in _argv_blob(fake_gcloud)
