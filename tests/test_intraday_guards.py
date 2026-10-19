"""Regression guards against two recurring critical bugs:

1. Timezone timestamp shift bug:
   Guards against any code shifting intraday candles by +5:30 (e.g. via
   naive tz_localize("UTC").tz_convert("Asia/Kolkata")), which caused 14:45
   afternoon candles to be erroneously stamped as 09:15 morning candles.
   Ensures export_intraday.py loads from canonical session_mod.archive_bars.

2. Repainting / forming-candle trigger bug:
   Guards against Gann V2 and Gann 90 cockpits triggering trades mid-candle
   on an unclosed 15m forming bar.
   Ensures extract15MinSessionBars flags forming buckets with `is_forming: true`,
   and both cockpit loops strictly enforce `if (b.is_forming) continue;`.
"""

import ast
from pathlib import Path
import pytest
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]


def _read_source(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


# -----------------------------------------------------------------------------
# 1. TIMEZONE & TIMESTAMP INTEGRITY GUARDS
# -----------------------------------------------------------------------------


def test_export_intraday_uses_canonical_archive_bars():
    """export_intraday.py must load canonical 5m archive, not unverified raw fyers_db."""
    src = _read_source("scripts/export_intraday.py")
    assert "session_mod.archive_bars(" in src, (
        "export_intraday.py must load canonical 5m archive using session_mod.archive_bars"
    )
    # Ensure the catastrophic double-conversion line does not exist anywhere in scripts/export_intraday.py
    assert 'tz_localize("UTC").tz_convert' not in src, (
        "Forbidden: export_intraday.py must not double-shift timestamps via tz_localize('UTC').tz_convert"
    )
    assert "tz_localize('UTC').tz_convert" not in src


def test_intraday_exported_json_session_open_integrity():
    """Every session in exported intraday JSON must start at 09:15 IST and have valid market hours."""
    for tf in ("5", "15"):
        path = ROOT / f"intraday_{tf}_nifty.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        keys = [k for k in data.keys() if not k.startswith("_")]
        days = sorted(list(set(k[:10] for k in keys)))
        for day in days:
            day_bars = [k for k in keys if k.startswith(day)]
            assert len(day_bars) > 0
            first_bar = day_bars[0]
            # First bar of the session MUST be 09:15
            assert first_bar.startswith(f"{day}T09:15:00+05:30"), (
                f"Session {day} first bar timestamp must be 09:15:00+05:30, got {first_bar}"
            )
            # Last bar must not exceed 15:30
            last_bar = day_bars[-1]
            last_time = last_bar[11:16]
            assert last_time <= "15:25", f"Bar time exceeded session close: {last_bar}"


def test_no_double_utc_tz_convert_in_repo():
    """No script in the codebase should convert a naive timestamp with UTC->IST shift."""
    for path in (ROOT / "core").rglob("*.py"):
        code = path.read_text(encoding="utf-8")
        assert 'tz_localize("UTC").tz_convert' not in code and "tz_localize('UTC').tz_convert" not in code, (
            f"Found forbidden UTC->IST double-conversion in {path}"
        )
    for path in (ROOT / "scripts").rglob("*.py"):
        code = path.read_text(encoding="utf-8")
        assert 'tz_localize("UTC").tz_convert' not in code and "tz_localize('UTC').tz_convert" not in code, (
            f"Found forbidden UTC->IST double-conversion in {path}"
        )


# -----------------------------------------------------------------------------
# 2. MID-CANDLE / FORMING 15M CANDLE GUARDS IN UI
# -----------------------------------------------------------------------------


def test_index_html_extract15min_flags_forming_candles():
    """extract15MinSessionBars in index.html must flag forming buckets with is_forming."""
    html = _read_source("index.html")
    assert "function extract15MinSessionBars(" in html
    assert "is_forming: isForming" in html or "is_forming:" in html
    assert "list.length < 3" in html, (
        "extract15MinSessionBars must require 3 5m bars before considering a 15m candle closed"
    )


def test_gann_cockpits_ignore_forming_candles():
    """Both Gann V2 and Gann 90 loops must reject unclosed forming candles."""
    html = _read_source("index.html")
    # Verify both loops have the unclosed candle guard
    forming_guard = "if (b.is_forming) continue;"
    count = html.count(forming_guard)
    assert count >= 2, (
        f"Expected at least 2 occurrences of '{forming_guard}' (one in Gann V2, one in Gann 90), found {count}"
    )


def test_upfront_execution_levels_and_multi_trade_nav():
    """UI must contain upfront execution level boxes and multi-trade switcher pills."""
    html = _read_source("index.html")
    # Upfront boxes
    assert "1. Entry Spot" in html
    assert "2. Actual SL" in html
    assert "3. Target 1" in html
    assert "4. Cost SL" in html
    assert "5. Target 2" in html

    # Multi-trade switcher containers
    assert 'id="s1-trades-nav"' in html
    assert 'id="s2-trades-nav"' in html
    assert 'id="gann90-s1-trades-nav"' in html
    assert 'id="gann90-s2-trades-nav"' in html

    # Switcher controller functions
    assert "function selectV2Trade(" in html
    assert "function selectG90Trade(" in html
