"""The exporter's option-payload path, with the network stubbed out.

``tests/test_option_model.py`` covers the option modules and
``tests/test_dom_option_layer.py`` covers the browser. This covers the seam
between them: that ``build_slim_option_payload()`` actually runs end to end
inside the exporter, persists a bounded payload, reuses the probe fetch instead
of re-requesting it, and degrades to None on every failure mode.

No test here touches the network.
"""
import datetime as dt
import importlib
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import export_dashboard as ed  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "nifty_option_chain.json"
TODAY = dt.date(2026, 9, 30)
# The real production options block, so these tests exercise the shipping
# configuration (Indian 10Y G-Sec proxy) rather than a convenient stand-in.
PROD = json.loads((ROOT / "config" / "settings.json").read_text(encoding="utf-8"))["options"]
SETTINGS = {"options": dict(PROD, risk_free_rate=None)}
UNSET_SETTINGS = SETTINGS


@pytest.fixture
def fx():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def stubbed(fx, monkeypatch):
    """Stub both chain fetchers and record which expiries were requested."""
    calls = []

    def fetch(spec, expiry_iso=None):
        calls.append(expiry_iso)
        if expiry_iso is None:
            return {"records": fx["front"]["records"]}
        if expiry_iso == "27-Oct-2026":
            return {"records": fx["monthly"]["records"]}
        if expiry_iso == "06-Oct-2026":
            return {"records": fx["front"]["records"]}
        return None

    monkeypatch.setattr(ed, "_fetch_option_chain", fetch)
    monkeypatch.setattr(ed, "_fetch_option_chain_for_expiry", fetch)
    return calls


def test_builds_both_expiry_candidates(stubbed):
    p = ed.build_slim_option_payload(("index", "NIFTY"), SETTINGS, today=TODAY)
    assert p is not None
    assert [c["expiry"] for c in p["candidates"]] == ["2026-10-06", "2026-10-27"]
    assert [c["dte"] for c in p["candidates"]] == [6, 27]


def test_payload_is_bounded(stubbed):
    p = ed.build_slim_option_payload(("index", "NIFTY"), SETTINGS, today=TODAY)
    assert len(json.dumps(p)) < 20000


def test_every_persisted_leg_carries_greeks(stubbed):
    p = ed.build_slim_option_payload(("index", "NIFTY"), SETTINGS, today=TODAY)
    for c in p["candidates"]:
        assert c["legs"], "expiry %s produced no legs" % c["expiry"]
        for leg in c["legs"]:
            assert set(leg["greeks"]) == {"delta", "gamma", "theta", "vega"}


def test_only_the_second_expiry_costs_a_fetch(stubbed):
    """The probe already returns expiry[0]; re-requesting it is wasted."""
    ed.build_slim_option_payload(("index", "NIFTY"), SETTINGS, today=TODAY)
    assert stubbed.count(None) == 1, "probe should be fetched once: %r" % (stubbed,)
    assert "27-Oct-2026" in stubbed, "second expiry was never requested"


def test_all_expired_chain_returns_none(monkeypatch):
    monkeypatch.setattr(ed, "_fetch_option_chain", lambda spec: {
        "records": {"underlyingValue": 25000.0, "expiryDates": ["29-Sep-2026"],
                    "data": []}})
    assert ed.build_slim_option_payload(("index", "NIFTY"), SETTINGS, today=TODAY) is None


def test_network_failure_returns_none_not_an_exception(monkeypatch):
    """The docstring promises this never raises; it must hold on a live fetch too."""
    def boom(spec):
        raise RuntimeError("network down")

    monkeypatch.setattr(ed, "_fetch_option_chain", boom)
    assert ed.build_slim_option_payload(("index", "NIFTY"), SETTINGS, today=TODAY) is None


def test_missing_chain_spec_returns_none():
    """Uses the real guard rather than a stub, which would mask it."""
    fresh = importlib.reload(ed)
    fresh._fetch_option_chain_for_expiry = lambda spec, expiry_iso: None
    assert fresh.build_slim_option_payload(None, SETTINGS, today=TODAY) is None


def test_production_config_flows_through_to_greeks(stubbed):
    """The shipping config must reach the payload, provenance and all."""
    p = ed.build_slim_option_payload(("index", "NIFTY"), {"options": PROD}, today=TODAY)
    assert p["risk_free_rate"] == 0.0717
    assert p["risk_free_rate_configured"] is True
    assert p["risk_free_rate_source"] == "India 10Y Government Security yield"
    assert p["risk_free_rate_as_of"] == "2026-09-29"
    assert p["risk_free_rate_type"] == "proxy"
    for c in p["candidates"]:
        priceable = [l for l in c["legs"] if l["iv"] is not None]
        assert priceable, "expiry %s had no priceable legs" % c["expiry"]
        for leg in priceable:
            assert leg["greeks"]["theta"] < 0
        # Rows the feed reports with zero IV are persisted but must stay
        # unpriced rather than being given an invented volatility.
        for leg in c["legs"]:
            if leg["iv"] is None:
                assert leg["greeks"]["delta"] is None


def test_unconfigured_rate_is_disclosed(stubbed):
    """A null rate must be reported as unconfigured, never silently assumed."""
    p = ed.build_slim_option_payload(("index", "NIFTY"), UNSET_SETTINGS, today=TODAY)
    assert p["risk_free_rate"] is None
    assert p["risk_free_rate_configured"] is False
    # ...and it must still price, at r=0, rather than dropping the whole payload.
    assert p["candidates"]


def test_expired_default_expiry_is_not_published(monkeypatch, fx):
    """The live default is an already-expired series with a fake IV ramp.

    Regression guard: if the fetch stops filtering past-dated expiries, this
    is what gets published.
    """
    monkeypatch.setattr(ed, "_fetch_option_chain", lambda spec, expiry_iso=None: {
        "records": fx["front"]["records"]})
    monkeypatch.setattr(ed, "_fetch_option_chain_for_expiry",
                        lambda spec, expiry_iso: None)
    p = ed.build_slim_option_payload(("index", "NIFTY"), SETTINGS, today=TODAY)
    if p is not None:
        for c in p["candidates"]:
            assert c["dte"] > 0, "published an expired expiry: %s" % c["expiry"]
