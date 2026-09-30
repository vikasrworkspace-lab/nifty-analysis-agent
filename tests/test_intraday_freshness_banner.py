"""Tests for the intraday payload freshness block and the dashboard's use of it.

Why this exists: the intraday exporter refuses to publish a stale feed, so a dead
feed leaves the *last good* JSON published and the dashboard simply looks quiet.
The only way the user learns their morning login was skipped is if the payload
carries a timestamp and the UI turns it into a banner.

The contract these tests pin:

* ``_freshness`` is a sibling of ``_meta``, never a key inside it. ``_meta`` is
  the WFO contract and is asserted field-by-field elsewhere; adding unrelated
  keys there would break that.
* ``last_bar_ts`` is the newest bar actually published, so the UI's age
  calculation reflects the data on screen rather than when the file was written.
* The UI tolerance equals the exporter's gate tolerance, or the banner and the
  gate would disagree about what "stale" means.
"""
import importlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ei = importlib.import_module("scripts.export_intraday")


# --- freshness_block ---------------------------------------------------------

def test_freshness_block_carries_the_four_fields():
    bar = pd.Timestamp("2026-09-30T14:25:00+05:30")
    block = ei.freshness_block("nifty", "5", bar)
    assert set(block) == {"symbol", "timeframe", "last_bar_ts", "generated_at"}


def test_last_bar_ts_is_the_newest_published_bar():
    bar = pd.Timestamp("2026-09-30T14:25:00+05:30")
    block = ei.freshness_block("nifty", "5", bar)
    # Must be the bar's own timestamp, not "now": a file regenerated from a
    # stale cache would otherwise look fresh purely by being rewritten.
    assert block["last_bar_ts"] == "2026-09-30T14:25:00+05:30"
    assert block["last_bar_ts"] != block["generated_at"]


def test_generated_at_is_iso_and_in_ist():
    block = ei.freshness_block("banknifty", "15", None)
    stamp = datetime.fromisoformat(block["generated_at"])
    assert stamp.tzinfo is not None
    assert stamp.utcoffset().total_seconds() == 5.5 * 3600


def test_missing_bar_yields_a_null_timestamp_rather_than_a_crash():
    block = ei.freshness_block("nifty", "60", None)
    assert block["last_bar_ts"] is None
    # The UI treats a null as "unknown", which is not the same as "stale".
    assert block["symbol"] == "nifty"
    assert block["timeframe"] == "60m"


def test_timeframe_is_labelled_with_its_minutes():
    assert ei.freshness_block("nifty", "30", None)["timeframe"] == "30m"


def test_freshness_is_a_sibling_of_meta_not_a_key_inside_it():
    # Guard the structural decision: _meta is the WFO contract.
    source = (ROOT / "scripts" / "export_intraday.py").read_text(encoding="utf-8")
    assert 'data["_freshness"]' in source
    assert not re.search(r'data\["_meta"\]\[', source), (
        "freshness must not be nested inside the _meta WFO contract"
    )


def test_freshness_is_not_counted_as_a_dated_series_entry():
    # The UI builds its date list from keys not starting with '_', so a reserved
    # block cannot become the default date selection (underscore sorts above
    # digits).
    payload = {"_freshness": ei.freshness_block("nifty", "5", None),
               "2026-09-30T09:15:00+05:30": {"signals": {}}}
    dates = [k for k in payload if not k.startswith("_")]
    assert dates == ["2026-09-30T09:15:00+05:30"]


def test_block_survives_a_json_round_trip():
    block = ei.freshness_block("nifty", "5", pd.Timestamp("2026-09-30T14:25:00+05:30"))
    assert json.loads(json.dumps({"_freshness": block}))["_freshness"] == block


# --- the UI banner -----------------------------------------------------------

HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def test_ui_defines_the_banner_and_its_tolerance():
    assert 'id="stale-banner"' in HTML
    assert "function updateStaleBanner" in HTML
    assert "MAX_STALE_MIN" in HTML


def test_ui_tolerance_matches_the_exporter_gate_tolerance():
    # auto_intraday_loop.MAX_BAR_AGE_MIN is what cloud_intraday_export.py gates
    # on. If these drift, the banner can contradict the gate.
    loop = importlib.import_module("scripts.auto_intraday_loop")
    ui_match = re.search(r"const MAX_STALE_MIN = (\d+);", HTML)
    assert ui_match, "could not read MAX_STALE_MIN out of index.html"
    assert int(ui_match.group(1)) == loop.MAX_BAR_AGE_MIN


def test_ui_uses_the_published_last_bar_not_the_file_time():
    fn = re.search(r"function updateStaleBanner\(\) \{(.*?)\n        \}", HTML, re.S)
    body = fn.group(1)
    assert "_freshness" in body
    assert "last_bar_ts" in body
    assert "generated_at" not in body, (
        "banner must key off the newest bar; generated_at would look fresh "
        "even when the cache itself is stale"
    )


def test_ui_only_warns_in_market_hours():
    fn = re.search(r"function updateStaleBanner\(\) \{(.*?)\n        \}", HTML, re.S)
    body = fn.group(1)
    # Outside market hours the previous session's data is correct, not stale.
    assert "inMarketHoursIst()" in body


def test_ui_computes_ist_independently_of_the_viewer_timezone():
    assert "timeZone: 'Asia/Kolkata'" in HTML
    assert "h23" in HTML, "hour12:false alone can yield hour 24 at midnight"


def test_banner_is_evaluated_after_every_payload_load():
    # The silent 60s refresh must re-evaluate it, or a feed that dies mid-session
    # keeps whatever the page decided on load.
    assert HTML.count("updateStaleBanner()") >= 2
    re_check = re.search(
        r"dashboardData = await res\.json\(\);\s*\n\s*updateStaleBanner\(\);", HTML
    )
    assert re_check, "banner is not refreshed after a payload fetch"


def test_banner_starts_hidden():
    assert re.search(r'id="stale-banner"[^>]*class="[^"]*\bhidden\b', HTML)
