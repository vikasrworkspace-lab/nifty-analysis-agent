"""Tests for the two-layer Trade Engine option model.

Two rules govern this file:

1. Assertions must be load-bearing. A test that passes when the behaviour it
   describes is removed is worse than no test, so the numeric tests perturb a
   real input and assert the OUTPUT changes by a specific amount. Tests are
   written against the sanitized fixture in tests/fixtures/nifty_option_chain.json,
   which mirrors the live feed including its defects (optionType null, sparse
   strikePrices, percent IV, zero IV on far OTM/deep ITM rows, two date formats).
   No test touches the network.

2. Missing or unpriceable data must produce None plus a reason. It must never
   silently fall back to a proxy -- in particular the option R:R must never be
   substituted with the underlying's point R:R.
"""
import datetime as dt
import json
import math
import pathlib

import pytest

from core import option_pricing as op
from core import options_chain as oc

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "nifty_option_chain.json"
SETTINGS = pathlib.Path(__file__).resolve().parents[1] / "config" / "settings.json"
TODAY = dt.date(2026, 9, 30)

# The production discount rate, read from the real config rather than restated
# here, so a config change cannot silently diverge from what the tests assume.
# This is a TEST rate: 0.06 appears only in this file and in tests/dom_option_layer.js
# as an arbitrary non-zero value for checking rate sensitivity. It is not the
# production recommendation -- see test_production_config_carries_the_gsec_proxy.
PRODUCTION_OPTIONS = json.loads(SETTINGS.read_text(encoding="utf-8"))["options"]
RATE = op.resolve_risk_free_rate(PRODUCTION_OPTIONS)


@pytest.fixture(scope="module")
def fx():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def nifty_payload(fx):
    """The bounded _options block the exporter would persist for NIFTY."""
    return oc.build_option_payload(
        {
            "2026-10-06": fx["front"]["records"]["data"],
            "2026-10-27": fx["monthly"]["records"]["data"],
        },
        25000.0,
        TODAY,
        rate_info=RATE,
        monthly_expiries={"2026-10-27"},
    )


LONG_FC = {"direction": "LONG", "expected_move_pct": 0.8, "invalidation_pct": 0.4}
SHORT_FC = {"direction": "SHORT", "expected_move_pct": 0.8, "invalidation_pct": 0.4}


# --------------------------------------------------------------------------
# Expiry selection
# --------------------------------------------------------------------------

def test_select_expiries_picks_front_and_next_monthly(fx):
    """NIFTY lists weeklies; only the last of each month is a monthly."""
    front, monthly = oc.select_expiries(fx["front"]["records"]["expiryDates"], TODAY)
    assert front == "2026-10-06"
    assert monthly == "2026-10-27"


def test_monthly_is_derived_not_assumed(fx):
    """BANK lists ONLY monthlies, so the front expiry is itself a monthly.

    A rule that returned the front expiry for both candidates would pass a
    weeklies-only test and silently break here.
    """
    front, monthly = oc.select_expiries(fx["bank_nifty"]["records"]["expiryDates"], TODAY)
    assert front == "2026-10-27"
    assert monthly == "2026-11-23"
    assert monthly != front


def test_select_expiries_skips_expired_front(fx):
    """The live default was an already-expired series with a fake IV ramp."""
    front, monthly = oc.select_expiries(
        ["29-Sep-2026", "06-Oct-2026", "27-Oct-2026"], TODAY
    )
    assert front == "2026-10-06"
    assert oc.dte(front, TODAY) > 0


def test_select_expiries_all_expired_returns_none():
    assert oc.select_expiries(["29-Sep-2026"], TODAY) == (None, None)


def test_select_expiries_no_monthly_after_front():
    """Degrades to front-only rather than inventing a second candidate."""
    assert oc.select_expiries(["06-Oct-2026"], TODAY) == ("2026-10-06", None)


def test_dte_is_signed():
    assert oc.dte("2026-10-06", TODAY) == 6
    assert oc.dte("2026-09-29", TODAY) == -1


# --------------------------------------------------------------------------
# IV normalisation -- the feed reports percent, Black-Scholes needs decimal
# --------------------------------------------------------------------------

def test_iv_percent_normalised_to_decimal(nifty_payload):
    """Feed said 13.0 meaning 13%; it must be stored as 0.13.

    Passing 13.0 straight to Black-Scholes inflates a 27-day ATM call to ~11000
    and makes every leg look risk-free, so this is the single most
    consequential normalisation in the module.
    """
    atm = next(l for l in nifty_payload["candidates"][1]["legs"]
               if l["strike"] == 25000.0 and l["leg"] == "CE")
    assert abs(atm["iv"] - 0.13) < 1e-9


def test_iv_already_decimal_is_left_alone():
    assert oc.normalise_iv(0.13) == pytest.approx(0.13)
    assert oc.normalise_iv(13.0) == pytest.approx(0.13)


def test_zero_iv_is_rejected_not_priced():
    assert oc.normalise_iv(0.0) is None
    assert oc.normalise_iv(None) is None


def test_absurd_iv_is_rejected():
    """Above 300% is a data error, not a market quote; must not be scaled."""
    assert oc.normalise_iv(450.0) is None


def test_zero_iv_legs_are_dropped_but_priceable_rows_survive(nifty_payload):
    front = nifty_payload["candidates"][0]["legs"]
    assert any(l["iv"] is None for l in front), "iv=0 rows must be present as None"
    assert all(l["iv"] is None or 0.10 < l["iv"] < 0.20 for l in front)


# --------------------------------------------------------------------------
# Strike selection
# --------------------------------------------------------------------------

def test_strike_interval_ignores_sparse_strike_prices(fx):
    """records.strikePrices is sparse (1500, 3000, 4500...).

    Using it would give a ~2000-point window; the real listing interval is 100.
    """
    sparse = fx["front"]["records"]["strikePrices"]
    real = [r["strikePrice"] for r in fx["front"]["records"]["data"]]
    assert oc.strike_interval(real, 25000.0) == 100.0
    assert oc.strike_interval(sparse, 25000.0) != 100.0


def test_strike_interval_is_modal_near_atm_not_global_median():
    """Fine near-ATM strikes with coarse wings: mode wins, median does not."""
    strikes = [15000, 20000, 25000, 30000] + list(range(24000, 26100, 50))
    assert oc.strike_interval(strikes, 25000) == 50.0


def test_strike_interval_scales_to_bank_nifty(fx):
    rows = fx["bank_nifty"]["records"]["data"]
    strikes = [r["strikePrice"] for r in rows]
    assert oc.strike_interval(strikes, 55000.0) == 500.0


def test_selected_strikes_are_atm_plus_minus_span(fx):
    rows = fx["front"]["records"]["data"]
    strikes = [r["strikePrice"] for r in rows]
    sel = oc.select_strikes(strikes, 25000.0, 100.0, span=3)
    assert sel == [24700.0, 24800.0, 24900.0, 25000.0, 25100.0, 25200.0]
    assert min(sel) <= 25000.0 <= max(sel)


def test_strike_selection_shifts_with_spot():
    """Moving spot must move the window, else ATM is not really ATM."""
    strikes = [24800, 24900, 25000, 25100, 25200, 25300]
    a = oc.select_strikes(strikes, 25000.0, 100.0, span=1)
    b = oc.select_strikes(strikes, 25100.0, 100.0, span=1)
    assert a != b
    # span=1 means ATM +/- 1, so the lowest pick is one interval BELOW spot.
    assert a == [24900.0, 25000.0, 25100.0]
    assert b == [25000.0, 25100.0, 25200.0]


# --------------------------------------------------------------------------
# Payload shape
# --------------------------------------------------------------------------

def test_payload_is_bounded(nifty_payload):
    raw = json.dumps(nifty_payload)
    assert len(raw) < 20000, "slim payload must stay small"
    assert len(nifty_payload["candidates"]) == 2
    legs = sum(len(c["legs"]) for c in nifty_payload["candidates"])
    assert legs <= 2 * 7 * 2, "ATM +/-3 strikes x 2 legs x 2 expiries"


def test_payload_carries_both_candidates(nifty_payload):
    expiries = {c["expiry"]: c["dte"] for c in nifty_payload["candidates"]}
    assert expiries == {"2026-10-06": 6, "2026-10-27": 27}


def test_payload_monthly_flag_is_not_derived_from_dte(nifty_payload):
    """Guessing is_monthly from DTE > 0 would mark every candidate monthly."""
    flags = {c["expiry"]: c["is_monthly"] for c in nifty_payload["candidates"]}
    assert flags == {"2026-10-06": False, "2026-10-27": True}


def test_payload_persists_only_required_fields(nifty_payload):
    allowed = {"strike", "expiry", "dte", "leg", "ltp", "iv", "bid", "ask",
               "bid_qty", "ask_qty", "volume", "oi", "change_oi", "greeks"}
    for c in nifty_payload["candidates"]:
        for leg in c["legs"]:
            assert set(leg) <= allowed


def test_payload_carries_server_computed_greeks(nifty_payload):
    """Greeks are a function of the chain alone, so the exporter computes them.

    Persisting them keeps a single source of truth; the browser only needs a
    price function for the scenario legs and cannot silently disagree.
    """
    leg = next(l for l in nifty_payload["candidates"][1]["legs"]
               if l["strike"] == 25000.0 and l["leg"] == "CE")
    gk = leg["greeks"]
    assert set(gk) == {"delta", "gamma", "theta", "vega"}
    assert 0 < gk["delta"] < 1
    assert gk["gamma"] > 0
    assert gk["theta"] < 0


def test_constructed_trade_carries_the_payload_greeks(nifty_payload):
    """The rendered trade's Greeks must be the persisted ones, not a recompute.

    The payload is the single source of truth. If evaluate_leg() recomputed
    them, a trade could report one delta beside a different delta in the chain
    table and nothing would fail.
    """
    trade = op.construct_option_trade(LONG_FC, nifty_payload, rate=RATE["rate"])
    payload_greeks = next(
        l["greeks"]
        for c in nifty_payload["candidates"] if c["expiry"] == trade["expiry"]
        for l in c["legs"]
        if l["strike"] == trade["strike"] and l["leg"] == trade["leg"]
    )
    assert trade["greeks"] == payload_greeks
    # The flattened greeks must agree with the nested block they came from.
    for k in ("delta", "gamma", "theta", "vega"):
        assert trade[k] == payload_greeks[k]


def test_constructed_trade_greeks_track_a_rate_change(nifty_payload):
    """Load-bearing check that the passthrough is real data, not a frozen copy.

    Changing the discount rate must change the persisted Greeks, and therefore
    the ones the trade reports.
    """
    base = op.construct_option_trade(LONG_FC, nifty_payload, rate=RATE["rate"])
    bumped = op.construct_option_trade(LONG_FC, nifty_payload, rate=0.0)
    assert base["greeks"] != bumped["greeks"]
    assert base["greeks"]["delta"] != bumped["greeks"]["delta"]


def test_payload_carries_the_configured_rate_and_its_provenance(nifty_payload):
    """The UI must be able to label the rate the exporter actually priced with."""
    assert nifty_payload["risk_free_rate"] == 0.0717
    assert nifty_payload["risk_free_rate_configured"] is True
    assert nifty_payload["risk_free_rate_source"] == "India 10Y Government Security yield"
    assert nifty_payload["risk_free_rate_as_of"] == "2026-09-29"
    assert nifty_payload["risk_free_rate_type"] == "proxy"
    assert nifty_payload["risk_free_rate_refresh"] == "manual"


def test_production_config_carries_the_gsec_proxy():
    """Pins the documented proxy so it cannot be swapped for r=0 or a US rate."""
    assert PRODUCTION_OPTIONS["risk_free_rate"] == 0.0717
    assert RATE["rate"] == 0.0717
    assert RATE["configured"] is True
    # A 10Y yield is a proxy, not a maturity-matched curve; saying so is required.
    assert RATE["type"] == "proxy"


def test_unset_rate_prices_at_zero_but_is_reported_unconfigured(fx):
    """The r=0 fallback must never be mistakable for a real yield."""
    info = op.resolve_risk_free_rate({})
    assert info["rate"] == 0.0
    assert info["declared"] is None
    assert info["configured"] is False
    p = oc.build_option_payload(
        {"2026-10-27": fx["monthly"]["records"]["data"]}, 25000.0, TODAY,
        rate_info=info, monthly_expiries={"2026-10-27"})
    assert p["risk_free_rate"] is None
    assert p["risk_free_rate_configured"] is False


def test_absurd_rate_is_rejected_not_silently_priced():
    with pytest.raises(ValueError):
        op.resolve_risk_free_rate({"risk_free_rate": -1.5})


def test_configured_rate_actually_moves_the_theta(nifty_payload):
    """Load-bearing: the configured rate must reach the persisted Greeks."""
    zero = op.greeks(25000, 25000, 27 / 365, 0.13, 0.0, "CE")["theta"]
    live = op.greeks(25000, 25000, 27 / 365, 0.13, RATE["rate"], "CE")["theta"]
    assert live != zero
    # Select the ATM 27d call explicitly rather than relying on leg ordering.
    leg = next(l for l in nifty_payload["candidates"][1]["legs"]
               if l["strike"] == 25000.0 and l["leg"] == "CE")
    assert leg["greeks"]["theta"] == pytest.approx(live, rel=1e-9)


# --------------------------------------------------------------------------
# Black-Scholes
# --------------------------------------------------------------------------

def test_atm_call_delta_near_half():
    g = op.greeks(25000, 25000, 27 / 365, 0.13, 0.0, "CE")
    assert 0.45 < g["delta"] < 0.55


def test_otm_call_delta_below_atm():
    at_m = op.greeks(25000, 25000, 27 / 365, 0.13, 0.0, "CE")["delta"]
    otm = op.greeks(25000, 25200, 27 / 365, 0.13, 0.0, "CE")["delta"]
    assert otm < at_m


def test_put_delta_is_negative():
    assert op.greeks(25000, 25000, 27 / 365, 0.13, 0.0, "PE")["delta"] < 0


def test_theta_is_negative_for_both_legs():
    """Time decay costs the buyer money; a positive theta would be a sign error."""
    assert op.greeks(25000, 25000, 27 / 365, 0.13, 0.0, "CE")["theta"] < 0
    assert op.greeks(25000, 25000, 27 / 365, 0.13, 0.0, "PE")["theta"] < 0


@pytest.mark.parametrize("leg", ["CE", "PE"])
@pytest.mark.parametrize("rate", [0.0, 0.06])
def test_theta_matches_finite_difference(leg, rate):
    """Pin theta to the definition: theta = -dV/dT, per calendar day.

    This is deliberately independent of the closed-form expression. A sign flip
    on the ``r*K`` term, or dropping the ``r`` factor, is INVISIBLE at r=0
    because that whole term vanishes -- mutation testing showed such a mutant
    surviving. Differencing the price function catches it at any rate, and
    verifies the per-day conversion as well.
    """
    yrs = 27 / 365.0
    h = 1.0 / 365.0  # one day, in years
    spot, strike, vol = 25000.0, 25000.0, 0.13
    v_up = op.price(spot, strike, yrs + h, vol, rate, leg)
    v_dn = op.price(spot, strike, yrs - h, vol, rate, leg)
    numeric_daily = -(v_up - v_dn) / (2 * h) / 365.0
    reported = op.greeks(spot, strike, yrs, vol, rate, leg)["theta"]
    assert numeric_daily < 0
    assert reported == pytest.approx(numeric_daily, rel=2e-3)


def test_theta_responds_to_the_risk_free_rate():
    """A non-zero rate must change the strike leg of theta.

    Guards the ``r`` factor that the r=0 parity check cannot see.
    """
    a = op.greeks(25000, 24000, 27 / 365, 0.13, 0.0, "CE")["theta"]
    b = op.greeks(25000, 24000, 27 / 365, 0.13, 0.06, "CE")["theta"]
    assert a != b


def test_gamma_positive_and_peaks_at_atm():
    at_m = op.greeks(25000, 25000, 27 / 365, 0.13, 0.0, "CE")["gamma"]
    wing = op.greeks(25000, 25500, 27 / 365, 0.13, 0.0, "CE")["gamma"]
    assert at_m > 0 and at_m > wing


def test_price_increases_with_vol():
    lo = op.price(25000, 25000, 27 / 365, 0.11, 0.0, "CE")
    mid = op.price(25000, 25000, 27 / 365, 0.13, 0.0, "CE")
    hi = op.price(25000, 25000, 27 / 365, 0.15, 0.0, "CE")
    assert lo < mid < hi


def test_price_never_negative_and_respects_bounds():
    """A call must sit between its intrinsic floor and spot, or the model is broken."""
    p = op.price(25000, 20000, 27 / 365, 0.13, 0.0, "CE")
    assert 5000 <= p <= 25000, "deep ITM call must exceed intrinsic 5000"
    deep_otm = op.price(25000, 40000, 27 / 365, 0.13, 0.0, "CE")
    assert 0 <= deep_otm < 1.0


def test_put_call_parity_approximately_holds():
    """C - P = S - K e^(-rT). At r=0 and ATM, S-K = 0, so C == P."""
    c = op.price(25000, 25000, 27 / 365, 0.13, 0.0, "CE")
    p = op.price(25000, 25000, 27 / 365, 0.13, 0.0, "PE")
    assert (c - p) == pytest.approx(0.0, abs=1e-6)
    # Non-zero rate: the K term must be discounted.
    c2 = op.price(25000, 25000, 27 / 365, 0.13, 0.06, "CE")
    p2 = op.price(25000, 25000, 27 / 365, 0.13, 0.06, "PE")
    assert (c2 - p2) == pytest.approx(25000 - 25000 * pow(2.718281828, -0.06 * 27 / 365), abs=1e-6)


def test_unpriceable_inputs_return_none():
    assert op.price(25000, 25000, 0, 0.13, 0.0, "CE") is None
    assert op.price(25000, 25000, 27 / 365, 0, 0.0, "CE") is None
    assert op.price(0, 25000, 27 / 365, 0.13, 0.0, "CE") is None
    assert op.price(25000, 25000, 27 / 365, 0.13, 0.0, "XX") is None


# --------------------------------------------------------------------------
# Option R:R -- must be in option points, never underlying points
# --------------------------------------------------------------------------

def test_option_rr_is_in_option_points(nifty_payload):
    t = op.construct_option_trade(LONG_FC, nifty_payload, rate=0.0)
    assert t["rr_basis"] == "option_points"
    assert t["option_risk"] > 0
    assert t["rr"] == pytest.approx(t["option_reward"] / t["option_risk"], rel=1e-9)


def test_option_rr_is_not_the_underlying_rr(nifty_payload):
    """The bug this whole layer exists to prevent.

    The underlying R:R for 0.8% target / 0.4% stop is 2.0. If the option R:R
    merely echoed that, Layer B would be decorative.
    """
    t = op.construct_option_trade(LONG_FC, nifty_payload, rate=0.0)
    assert t["rr"] != pytest.approx(2.0, abs=0.01)


def test_missing_iv_gives_no_rr_and_a_reason():
    leg = {"strike": 25000.0, "expiry": "2026-10-27", "dte": 27, "leg": "CE",
           "ltp": 352.6, "iv": None}
    r = op.evaluate_leg(leg, 25000.0, 0.0, 0.8, 0.4)
    assert r["rr"] is None
    assert "volatility" in r["reason"]


def test_zero_iv_gives_no_rr_and_a_reason():
    leg = {"strike": 25000.0, "expiry": "2026-10-27", "dte": 27, "leg": "CE",
           "ltp": 352.6, "iv": 0.0}
    r = op.evaluate_leg(leg, 25000.0, 0.0, 0.8, 0.4)
    assert r["rr"] is None and r["reason"]


def test_missing_ltp_gives_no_rr():
    leg = {"strike": 25000.0, "expiry": "2026-10-27", "dte": 27, "leg": "CE",
           "ltp": None, "iv": 0.13}
    assert op.evaluate_leg(leg, 25000.0, 0.0, 0.8, 0.4)["rr"] is None


def test_expired_expiry_gives_no_rr():
    leg = {"strike": 25000.0, "expiry": "2026-09-29", "dte": -1, "leg": "CE",
           "ltp": 352.6, "iv": 0.13}
    r = op.evaluate_leg(leg, 25000.0, 0.0, 0.8, 0.4)
    assert r["rr"] is None and "time" in r["reason"]


def test_no_option_data_returns_no_trade():
    assert op.construct_option_trade(LONG_FC, None) is None
    assert op.construct_option_trade(LONG_FC, {"candidates": []}) is None
    assert op.construct_option_trade(LONG_FC, {"underlying": 25000.0, "candidates": []}) is None


def test_missing_underlying_returns_no_trade(nifty_payload):
    stripped = dict(nifty_payload)
    stripped.pop("underlying")
    assert op.construct_option_trade(LONG_FC, stripped) is None


def test_missing_forecast_move_returns_no_trade(nifty_payload):
    bad = {"direction": "LONG"}
    assert op.construct_option_trade(bad, nifty_payload) is None


def test_unknown_direction_returns_no_trade(nifty_payload):
    bad = {"direction": "FLAT", "expected_move_pct": 0.8, "invalidation_pct": 0.4}
    assert op.construct_option_trade(bad, nifty_payload) is None


# --------------------------------------------------------------------------
# Sensitivity -- perturbing an input must change the output
# --------------------------------------------------------------------------

def test_expected_move_increases_rr(nifty_payload):
    """A bigger expected move must be worth more of the option, not less."""
    small = op.construct_option_trade(
        {"direction": "LONG", "expected_move_pct": 0.4, "invalidation_pct": 0.4},
        nifty_payload, rate=0.0)
    big = op.construct_option_trade(
        {"direction": "LONG", "expected_move_pct": 1.5, "invalidation_pct": 0.4},
        nifty_payload, rate=0.0)
    assert big["rr"] > small["rr"]


def test_wider_invalidation_lowers_rr(nifty_payload):
    tight = op.construct_option_trade(
        {"direction": "LONG", "expected_move_pct": 0.8, "invalidation_pct": 0.2},
        nifty_payload, rate=0.0)
    wide = op.construct_option_trade(
        {"direction": "LONG", "expected_move_pct": 0.8, "invalidation_pct": 0.8},
        nifty_payload, rate=0.0)
    assert wide["rr"] < tight["rr"]


def test_spot_perturbation_anchors_scenarios_to_live_spot(nifty_payload):
    """The scenario legs must follow the live spot, not a hardcoded 25000.

    Which strike ends up on top is a separate question (the score rewards
    delta, so it can favour a deeper ITM leg), but every scenario price and
    underlying level must be anchored to whatever spot was supplied.
    """
    levels = set()
    for spot in (24800.0, 25000.0, 25200.0):
        t = op.construct_option_trade(LONG_FC, nifty_payload, rate=0.0, spot=spot)
        assert t["underlying_at_target"] == pytest.approx(spot * 1.008, rel=1e-9)
        assert t["underlying_at_stop"] == pytest.approx(spot * 0.996, rel=1e-9)
        levels.add(round(t["underlying_at_target"], 2))
    assert len(levels) == 3, "scenario anchor ignored the supplied spot"


def test_evaluate_leg_delta_responds_to_spot():
    """A 25000 call is OTM below the strike and ITM above it."""
    base = {"strike": 25000.0, "expiry": "2026-10-27", "dte": 27, "leg": "CE",
            "ltp": 352.6, "iv": 0.13}
    below = op.evaluate_leg(base, 24000.0, 0.0, 0.8, 0.4)
    above = op.evaluate_leg(base, 26000.0, 0.0, 0.8, 0.4)
    assert above["delta"] > 0.5 > below["delta"]


def test_scenario_prices_are_rebased_to_actual_ltp():
    """Entry must be the traded price, not the model's opinion of it.

    With a deliberately stale LTP well below the model price, the un-rebased
    form (risk = ltp - model(stop)) goes negative and reports "no risk".
    Rebasing anchors the entry leg to the LTP and keeps risk positive.
    """
    leg = {"strike": 25000.0, "expiry": "2026-10-27", "dte": 27, "leg": "CE",
           "ltp": 200.0, "iv": 0.13}
    r = op.evaluate_leg(leg, 25000.0, 0.0, 0.8, 0.4)
    assert r["option_risk"] > 0
    # Model is far above the stale print; the gap is disclosed, not hidden.
    assert r["model_vs_ltp_gap_pct"] > 30
    assert r["rr"] is not None


def test_iv_perturbation_changes_rr():
    """Holding entry fixed, a different vol must move the option R:R.

    Uses a single leg so the comparison is exact rather than filtered through
    candidate ranking.
    """
    base = {"strike": 25000.0, "expiry": "2026-10-27", "dte": 27, "leg": "CE", "ltp": 352.6}
    lo = op.evaluate_leg(dict(base, iv=0.11), 25000.0, 0.0, 0.8, 0.4)
    hi = op.evaluate_leg(dict(base, iv=0.17), 25000.0, 0.0, 0.8, 0.4)
    assert abs(hi["rr"] - lo["rr"]) > 0.01


def test_dte_perturbation_changes_theta_and_rr():
    base = {"strike": 25000.0, "expiry": "2026-10-27", "leg": "CE", "ltp": 352.6, "iv": 0.13}
    near = op.evaluate_leg(dict(base, dte=6), 25000.0, 0.0, 0.8, 0.4)
    far = op.evaluate_leg(dict(base, dte=27), 25000.0, 0.0, 0.8, 0.4)
    assert near["theta"] != far["theta"]
    assert near["rr"] != far["rr"]


# --------------------------------------------------------------------------
# Direction handling
# --------------------------------------------------------------------------

def test_long_takes_a_call_and_short_takes_a_put(nifty_payload):
    long_t = op.construct_option_trade(LONG_FC, nifty_payload, rate=0.0)
    short_t = op.construct_option_trade(SHORT_FC, nifty_payload, rate=0.0)
    assert long_t["leg"] == "CE"
    assert short_t["leg"] == "PE"


def test_short_scenario_signs_are_inverted():
    """A SHORT's target is BELOW spot and its stop is ABOVE.

    Hardcoding the LONG convention puts the put's stop above entry, inverts the
    risk term and yields a negative R:R.
    """
    tgt, stop = op.scenario_spots(25000.0, 0.8, 0.4, "SHORT")
    assert tgt < 25000.0 < stop


def test_long_scenario_signs():
    tgt, stop = op.scenario_spots(25000.0, 0.8, 0.4, "LONG")
    assert stop < 25000.0 < tgt


def test_short_produces_a_usable_rr(nifty_payload):
    t = op.construct_option_trade(SHORT_FC, nifty_payload, rate=0.0)
    assert t["rr"] is not None and t["rr"] > 0
    assert t["option_risk"] > 0


# --------------------------------------------------------------------------
# Expiry choice under BTST
# --------------------------------------------------------------------------

def _dte_scores(trade):
    out = {}
    for c in trade["all_candidates"]:
        out.setdefault(c["expiry"], c["score"])
    return out


def test_btst_widens_the_penalty_on_very_short_dte(nifty_payload):
    """The load-bearing BTST claim.

    It is NOT "BTST picks a different expiry" -- the swing case may already
    prefer the monthly. The guarantee is that holding a near-expiry option
    overnight is penalised, so the monthly's lead must grow under BTST.
    """
    swing = _dte_scores(op.construct_option_trade(LONG_FC, nifty_payload, rate=0.0, btst=False))
    btst = _dte_scores(op.construct_option_trade(LONG_FC, nifty_payload, rate=0.0, btst=True))
    gap_swing = swing["2026-10-27"] - swing["2026-10-06"]
    gap_btst = btst["2026-10-27"] - btst["2026-10-06"]
    assert gap_btst > gap_swing
    assert gap_btst - gap_swing >= 4.0, "overnight hold must cost the short-DTE leg"


def test_btst_never_keeps_a_sub_minimum_dte(nifty_payload):
    """A DTE below the floor must be rejected outright, not merely demoted."""
    leg = {"strike": 25000.0, "expiry": "2026-09-30", "dte": 1, "leg": "CE",
           "ltp": 180.0, "iv": 0.30}
    score, notes = op._score_candidate(
        op.evaluate_leg(leg, 25000.0, 0.0, 0.8, 0.4), LONG_FC,
        btst=True, btst_min_dte=3)
    assert any("below BTST minimum" in n for n in notes)
    assert score < -50


def test_both_candidates_are_always_considered(nifty_payload):
    t = op.construct_option_trade(LONG_FC, nifty_payload, rate=0.0)
    expiries = {c["expiry"] for c in t["all_candidates"]}
    assert expiries == {"2026-10-06", "2026-10-27"}
    assert t["candidates_considered"] == 12  # 2 expiries x 6 strikes


def test_expiry_role_is_derived_from_dte_not_list_position(nifty_payload):
    """Reversing the payload order must not relabel front/monthly."""
    rev = dict(nifty_payload)
    rev["candidates"] = list(reversed(nifty_payload["candidates"]))
    t = op.construct_option_trade(LONG_FC, rev, rate=0.0)
    roles = {c["expiry"]: c["expiry_role"] for c in t["all_candidates"]}
    assert roles["2026-10-06"] == "front"
    assert roles["2026-10-27"] == "next_monthly"


# --------------------------------------------------------------------------
# Stale data
# --------------------------------------------------------------------------

def test_stale_chain_is_detectable():
    """A chain whose DTE has gone negative must be refused, not priced."""
    stale = {
        "expiry": "2026-09-29",
        "dte": -1,
        "legs": [{"strike": 25000.0, "expiry": "2026-09-29", "dte": -1,
                  "leg": "CE", "ltp": 210.0, "iv": 0.14}],
    }
    payload = {"underlying": 25000.0, "candidates": [stale]}
    t = op.construct_option_trade(LONG_FC, payload, rate=0.0)
    assert t is None or t["rr"] is None


def test_all_candidates_expose_a_reason_when_unpriceable():
    payload = {"underlying": 25000.0, "candidates": [
        {"expiry": "2026-09-29", "dte": -1, "is_monthly": False,
         "legs": [{"strike": 25000.0, "expiry": "2026-09-29", "dte": -1,
                   "leg": "CE", "ltp": 210.0, "iv": 0.14}]}]}
    t = op.construct_option_trade(LONG_FC, payload, rate=0.0)
    assert t["rr"] is None
    assert t["reason"]


# --------------------------------------------------------------------------
# Fixture fidelity -- the feed's defects must stay represented
# --------------------------------------------------------------------------

def test_fixture_preserves_feed_defects(fx):
    """If someone 'tidies' the fixture, these regressions come back."""
    rows = fx["front"]["records"]["data"]
    assert all(r["CE"]["optionType"] is None for r in rows)
    assert all(r["PE"]["optionType"] is None for r in rows)
    assert any(r["CE"]["impliedVolatility"] == 0 for r in rows)
    assert all(isinstance(s, str) for s in fx["front"]["records"]["strikePrices"])
    # Two date formats on purpose.
    assert fx["front"]["records"]["data"][0]["expiryDate"] == "06-10-2026"
    assert fx["front"]["records"]["expiryDates"][0] == "06-Oct-2026"


def test_fixture_strike_prices_never_breach_intrinsic(fx):
    """A sub-intrinsic LTP is an arbitrage violation and wrecks theta/gamma."""
    for key in ("front", "monthly", "bank_nifty"):
        block = fx[key]
        spot = block["records"]["underlyingValue"]
        for row in block["records"]["data"]:
            k = row["strikePrice"]
            assert row["CE"]["lastPrice"] >= max(spot - k, 0) - 1e-9
            assert row["PE"]["lastPrice"] >= max(k - spot, 0) - 1e-9


def test_fixture_respects_european_no_arbitrage_bounds(fx):
    """European bounds use the DISCOUNTED strike, not the face strike.

    The face-strike bound above is a weaker, always-safe check. This is the
    tight one: max(Ke^-rT - S, 0) <= P <= Ke^-rT and max(S - Ke^-rT, 0) <= C <= S.
    """
    dte = {"front": 6, "monthly": 27, "bank_nifty": 27}
    for key, days in dte.items():
        block = fx[key]
        spot = block["records"]["underlyingValue"]
        disc = math.exp(-RATE["rate"] * days / 365.0)
        for row in block["records"]["data"]:
            k = row["strikePrice"]
            kd = k * disc
            c = row["CE"]["lastPrice"]
            p = row["PE"]["lastPrice"]
            assert max(0.0, kd - spot) - 1e-6 <= p <= kd + 1e-6, (key, k, p, kd)
            assert max(0.0, spot - kd) - 1e-6 <= c <= spot + 1e-6, (key, k, c)


def test_fixture_call_falls_and_put_rises_with_strike(fx):
    """A call must fall and a put must RISE as the strike increases.

    Written the wrong way round (put falling) this test still passes on a chain
    whose puts are mirrored copies of its calls, so the direction is the point.
    """
    for key in ("front", "monthly", "bank_nifty"):
        rows = sorted(fx[key]["records"]["data"], key=lambda r: float(r["strikePrice"]))
        calls = [r["CE"]["lastPrice"] for r in rows]
        puts = [r["PE"]["lastPrice"] for r in rows]
        assert all(b <= a + 1e-9 for a, b in zip(calls, calls[1:])), key
        assert all(b >= a - 1e-9 for a, b in zip(puts, puts[1:])), key


def test_fixture_legs_satisfy_put_call_parity_at_the_production_rate(fx):
    """C - P == S - K*e^(-rT), within one tick.

    This is what catches a chain generated at a DIFFERENT discount rate: at r=0
    a call and its put are equal at every strike, which silently desynchronises
    every discounted model price from the quoted LTP once production uses 7.17%.
    """
    dte = {"front": 6, "monthly": 27, "bank_nifty": 27}
    for key, days in dte.items():
        block = fx[key]
        spot = block["records"]["underlyingValue"]
        disc = math.exp(-RATE["rate"] * days / 365.0)
        checked = 0
        for row in block["records"]["data"]:
            k = float(row["strikePrice"])
            iv_c = row["CE"]["impliedVolatility"]
            iv_p = row["PE"]["impliedVolatility"]
            if not iv_c or not iv_p:
                continue  # the feed's zero-IV rows carry no usable quote
            c = row["CE"]["lastPrice"]
            p = row["PE"]["lastPrice"]
            assert abs((c - p) - (spot - k * disc)) <= 1.0, (key, k, c, p)
            checked += 1
        assert checked >= 5, (key, checked)


def test_fixture_parity_would_fail_if_the_chain_were_priced_at_zero_rate(fx):
    """Guards the guard: the parity test must actually be load-bearing.

    Feeding the same quotes through a zero-rate parity check must FAIL. If this
    ever passes, the parity assertion above has been weakened to a tautology.
    """
    dte = {"front": 6, "monthly": 27, "bank_nifty": 27}
    mismatches = 0
    for key, days in dte.items():
        block = fx[key]
        spot = block["records"]["underlyingValue"]
        for row in block["records"]["data"]:
            k = float(row["strikePrice"])
            if not row["CE"]["impliedVolatility"] or not row["PE"]["impliedVolatility"]:
                continue
            residual = ((row["CE"]["lastPrice"] - row["PE"]["lastPrice"])
                        - (spot - k))  # the r=0 identity
            if abs(residual) > 1.0:
                mismatches += 1
    assert mismatches >= 10, mismatches


def test_fixture_supplies_no_greeks(fx):
    """Greeks must be computed, never read from the feed."""
    leg = fx["front"]["records"]["data"][0]["CE"]
    for k in ("delta", "gamma", "theta", "vega"):
        assert k not in leg
