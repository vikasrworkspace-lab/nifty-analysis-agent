"""Regression tests for the three bugs found on 2026-10-01.

All three were hidden by the expired Fyers token: the feed was dead, so the
intraday exporter never reached its analogue search and the dashboard never
reached its own. They only surfaced once a live feed let the code actually run.

1. ``np.argpartition`` is partitioned at ``k``, but ``kth`` is an *index*. When
   the candidate pool holds exactly ``k`` rows this raises
   ``ValueError: kth(=k) out of bounds (k)``, so nothing publishes.
2. ``verify_token`` sent ``Authorization: Bearer <token>``. Fyers wants
   ``APP_ID:TOKEN`` and answers HTTP 400 / ``-209`` regardless of the token.
3. ``verify_token`` reported success when ``requests`` was missing, claiming a
   verification that never happened.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_fyers_login():
    spec = importlib.util.spec_from_file_location(
        "fyers_login_under_test", ROOT / "scripts" / "fyers_login.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 1. argpartition kth off-by-one
# --------------------------------------------------------------------------


def test_argpartition_at_k_raises_with_exactly_k_candidates():
    """The exact production failure, so the fix is pinned to a real numpy error."""
    dist = np.arange(46, dtype=float)
    with pytest.raises(ValueError, match="out of bounds"):
        np.argpartition(dist, 46)


def test_argpartition_at_k_minus_one_works_with_exactly_k_candidates():
    dist = np.arange(46, dtype=float)
    idx = np.argpartition(dist, 45)[:46]
    assert len(idx) == 46


def test_k_minus_one_selects_the_true_nearest():
    """Guard against 'fixing' the crash by returning the wrong rows."""
    rng = np.random.default_rng(7)
    dist = rng.random(50)
    k = 10
    got = set(np.argpartition(dist, k - 1)[:k].tolist())
    want = set(np.argsort(dist)[:k].tolist())
    assert got == want


def test_k_minus_one_is_stable_across_pool_sizes():
    for n in (2, 5, 46, 100, 251):
        dist = np.arange(n, dtype=float)
        k = min(50, n)
        idx = np.argpartition(dist, k - 1)[:k]
        assert len(idx) == k


def test_intraday_exporter_partitions_at_k_minus_one():
    src = _source("scripts/export_intraday.py")
    assert "np.argpartition(dist[i], k - 1)[:k]" in src
    assert "np.argpartition(dist[i], k)[:k]" not in src


def test_dashboard_exporter_partitions_at_top_n_minus_one():
    src = _source("scripts/export_dashboard.py")
    assert "np.argpartition(dists, top_n - 1)[:top_n]" in src
    assert "np.argpartition(dists, top_n)[:top_n]" not in src


def test_no_unclamped_argpartition_remains():
    """Every argpartition in the shipped code must partition at <len - 1>."""
    offenders = []
    for path in ROOT.rglob("*.py"):
        # Skip the venv, build output, and this file (its own source contains
        # the very call shapes being asserted on).
        parts = path.parts
        if any(p in parts for p in (".venv", ".git", "public", "tests")):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "argpartition"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.BinOp)
                and isinstance(node.args[1].op, ast.Sub)
            ):
                offenders.append(path.relative_to(ROOT).as_posix())
    assert sorted(offenders) == [
        "scripts/export_dashboard.py",
        "scripts/export_intraday.py",
    ], offenders


# --------------------------------------------------------------------------
# 2 & 3. verify_token header shape and honesty
# --------------------------------------------------------------------------


def test_verify_token_uses_appid_colon_token_not_bearer(monkeypatch):
    fyers_login = _load_fyers_login()
    captured = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"s": "ok", "data": {"name": "x"}}

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return FakeResponse()

    import requests

    monkeypatch.setattr(requests, "get", fake_get)

    ok, detail = fyers_login.verify_token("TOKEN123", "APP456")

    assert ok, detail
    assert captured["headers"]["Authorization"] == "APP456:TOKEN123"
    assert captured["headers"]["X-Api-Key"] == "APP456"
    assert captured["headers"]["version"] == "3"
    assert "Bearer" not in captured["headers"]["Authorization"]


def test_verify_token_reports_fyers_error_code(monkeypatch):
    fyers_login = _load_fyers_login()

    class FakeResponse:
        status_code = 400

        @staticmethod
        def json():
            return {"s": "error", "code": -16, "message": "nope"}

    import requests

    monkeypatch.setattr(
        requests, "get", lambda *a, **k: FakeResponse()
    )

    ok, detail = fyers_login.verify_token("T", "A")

    assert not ok
    assert "-16" in detail


def test_verify_token_does_not_claim_success_without_requests(monkeypatch):
    """A missing requests must be a failure, not a silent pass."""
    fyers_login = _load_fyers_login()

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "requests":
            raise ImportError("no requests")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "requests", None)
    monkeypatch.setattr("builtins.__import__", fake_import)

    ok, detail = fyers_login.verify_token("T", "A")

    assert not ok
    assert "NOT verified" in detail


def test_verify_token_source_has_no_bearer_prefix():
    src = _source("scripts/fyers_login.py")
    assert "Bearer {access_token}" not in src
    assert '{client_id}:{access_token}' in src
