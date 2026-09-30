"""Black-Scholes pricing and the Layer B option-trade construction.

Layer A (the underlying forecast) produces a directional view on the index.
Layer B turns that into a specific option position and prices its risk in
OPTION space, never in index points.

Why this module exists: a NIFTY stop being hit does not mean the option goes to
zero. An option retains substantial value past an underlying invalidation level
depending on strike, expiry, IV and moneyness. Symmetrically, a far-OTM
near-expiry option can lose most of its value on a modest underlying move. Any
R:R computed from index point distances is therefore wrong for the option trade,
so the R:R here is always:

    risk   = option entry price - option stop price
    reward = option target price - option entry price

The live NSE chain does not supply delta/gamma/theta/vega, so they are computed
here from spot, strike, time to expiry, volatility and the risk-free rate.
"""
from __future__ import annotations

import math

# 365-day year. Exchanges quote Indian option expiries in calendar days and
# the live chain carries no trading-day calendar, so this is the standard
# convention. It is stated here rather than hidden because it sets theta.
DAYS_PER_YEAR = 365.0

SQRT_2PI = math.sqrt(2.0 * math.pi)

# Config keys carrying the discount rate and its provenance. The rate is a
# PROXY (Indian 10Y G-Sec), not a maturity-matched curve, so the metadata is
# persisted alongside it: a 7.17% 10-year yield is the wrong discount rate for a
# 6-day option, and that basis error has to be visible rather than implied.
RATE_KEY = "risk_free_rate"
RATE_META_KEYS = (
    "risk_free_rate_source",
    "risk_free_rate_as_of",
    "risk_free_rate_type",
    "risk_free_rate_refresh",
)


def resolve_risk_free_rate(options_cfg=None):
    """Read the configured discount rate and its provenance out of the config.

    Returns a dict with:
        rate       -- the float to actually price with
        declared   -- the raw config value (None when unset), for display
        configured -- whether a real value was supplied
        source/as_of/type/refresh -- provenance, verbatim from config

    Kept deliberately separate from :func:`price` and :func:`greeks`, which stay
    pure functions of their arguments. Config parsing, defaults and
    documentation live here, so the pricing math has exactly one definition and
    the tests can drive it with any rate without touching the filesystem.

    When unset, pricing falls back to r=0.0 and ``configured`` is False, so a
    caller can never mistake the fallback for a real yield.
    """
    cfg = options_cfg or {}
    raw = cfg.get(RATE_KEY)
    meta = {k[len(RATE_KEY) + 1:]: cfg.get(k) for k in RATE_META_KEYS}
    if raw is None:
        return {"rate": 0.0, "declared": None, "configured": False, **meta}
    rate = float(raw)
    # Below -1.0 the year fraction is effectively undefined; catching it here
    # beats emitting a NaN premium that would silently propagate to the UI.
    if rate <= -1.0:
        raise ValueError("risk_free_rate must be > -1.0, got %r" % (raw,))
    return {"rate": rate, "declared": rate, "configured": True, **meta}


def _norm_cdf(x):
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / SQRT_2PI


def intrinsic_value(spot, strike, leg):
    d = spot - strike
    return max(d, 0.0) if leg == "CE" else max(-d, 0.0)


def price(spot, strike, years, vol, rate, leg, dividend=0.0):
    """Black-Scholes European option price.

    Returns None when the inputs cannot support a price (non-positive spot or
    strike, non-positive time/vol). Callers must treat None as "unpriceable"
    rather than substituting a fallback.
    """
    if spot is None or strike is None or spot <= 0 or strike <= 0:
        return None
    if years is None or years <= 0 or vol is None or vol <= 0:
        return None

    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * years) / (vol * math.sqrt(years))
    d2 = d1 - vol * math.sqrt(years)
    df = math.exp(-rate * years)

    if leg == "CE":
        return spot * math.exp(-dividend * years) * _norm_cdf(d1) - strike * df * _norm_cdf(d2)
    if leg == "PE":
        return strike * df * _norm_cdf(-d2) - spot * math.exp(-dividend * years) * _norm_cdf(-d1)
    return None


def greeks(spot, strike, years, vol, rate, leg, dividend=0.0):
    """delta, gamma, theta (per calendar day), vega (per 1 vol point).

    Returns a dict with None values when unpriceable, so the UI can show
    "unavailable" instead of a fabricated number.
    """
    out = {"delta": None, "gamma": None, "theta": None, "vega": None}
    if spot is None or strike is None or spot <= 0 or strike <= 0:
        return out
    if years is None or years <= 0 or vol is None or vol <= 0:
        return out

    sqrt_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    df = math.exp(-rate * years)
    pdf = _norm_pdf(d1)

    if leg == "CE":
        out["delta"] = math.exp(-dividend * years) * _norm_cdf(d1)
    elif leg == "PE":
        out["delta"] = math.exp(-dividend * years) * (_norm_cdf(d1) - 1.0)
    else:
        return out

    out["gamma"] = (math.exp(-dividend * years) * pdf) / (spot * vol * sqrt_t)
    # Standard Black-Scholes theta (annualised), then converted to per calendar
    # day so it is directly comparable to an option LTP:
    #   call: -S e^-qT N'(d1) sigma/(2 sqrt(T)) - r K e^-rT N(d2)  + q S e^-qT N(d1)
    #   put:  -S e^-qT N'(d1) sigma/(2 sqrt(T)) + r K e^-rT N(-d2) - q S e^-qT N(-d1)
    # Note the strike term carries the rate factor r AND the call sign is
    # negative. Dropping r and/or flipping that sign returns a POSITIVE theta
    # for a long option, i.e. it would claim time decay pays the buyer.
    common = spot * math.exp(-dividend * years) * pdf * vol / (2.0 * sqrt_t)
    rk_df = rate * strike * df
    if leg == "CE":
        theta_annual = (-common - rk_df * _norm_cdf(d2)
                        + dividend * spot * math.exp(-dividend * years) * _norm_cdf(d1))
    else:
        theta_annual = (-common + rk_df * _norm_cdf(-d2)
                        - dividend * spot * math.exp(-dividend * years) * _norm_cdf(-d1))
    out["theta"] = theta_annual / DAYS_PER_YEAR
    out["vega"] = spot * math.exp(-dividend * years) * pdf * sqrt_t / 100.0
    return out


def option_response(leg_data, spot_move_pct, years, vol, rate):
    """Expected option price after a given underlying move, holding IV constant.

    This is the number that must drive the option R:R. It answers "what is this
    option worth if the underlying moves as the forecast expects", which is not
    the same as the underlying's own point move scaled by delta, because gamma
    and theta both act over the holding period.
    """
    if leg_data is None:
        return None
    strike = leg_data.get("strike")
    leg = leg_data.get("leg")
    iv = leg_data.get("iv")
    if iv is None or iv <= 0 or spot_move_pct is None:
        return None
    new_spot = spot_move_pct.get("from", 0.0) * (1.0 + spot_move_pct.get("pct", 0.0) / 100.0) \
        if isinstance(spot_move_pct, dict) else None
    if new_spot is None:
        return None
    return price(new_spot, strike, years, iv, rate, leg)


def _years(dte_days):
    if dte_days is None or dte_days <= 0:
        return None
    return dte_days / DAYS_PER_YEAR


def scenario_spots(spot, target_pct, stop_pct, direction="LONG"):
    """Underlying spot at the forecast target and at the invalidation level.

    Direction matters and is not cosmetic. For a LONG the target is above spot
    and the stop below; for a SHORT it is the reverse. Hardcoding the LONG
    convention makes a SHORT put's "stop" sit ABOVE entry, which inverts the
    risk term and reports a negative R:R. Magnitudes are always the absolute
    expected move and invalidation distance.
    """
    if direction == "SHORT":
        return spot * (1.0 - target_pct / 100.0), spot * (1.0 + stop_pct / 100.0)
    return spot * (1.0 + target_pct / 100.0), spot * (1.0 - stop_pct / 100.0)


def evaluate_leg(leg_data, spot, rate, target_pct, stop_pct, direction="LONG"):
    """Full evaluation of a single option leg against an underlying forecast.

    ``target_pct`` / ``stop_pct`` are the underlying's expected move and
    invalidation, in percent (magnitudes). Both are translated into option
    space via :func:`scenario_spots`, which honours ``direction``.

    The R:R returned here is in option points. If any input is unpriceable the
    result carries ``rr = None`` and a reason -- it never falls back to
    underlying point distances.
    """
    res = {
        "strike": leg_data.get("strike"),
        "expiry": leg_data.get("expiry"),
        "dte": leg_data.get("dte"),
        "leg": leg_data.get("leg"),
        "ltp": leg_data.get("ltp"),
        "iv": leg_data.get("iv"),
        "rr": None,
        "rr_basis": "option_points",
        "reason": None,
    }
    if leg_data.get("iv") is None or leg_data.get("iv") <= 0:
        res["reason"] = "no usable implied volatility"
        return res
    if leg_data.get("ltp") is None or leg_data["ltp"] <= 0:
        res["reason"] = "no option LTP"
        return res

    years = _years(leg_data.get("dte"))
    if years is None:
        res["reason"] = "expiry has no positive time remaining"
        return res

    strike = leg_data["strike"]
    res["intrinsic"] = intrinsic_value(spot, strike, leg_data["leg"])
    res["time_value"] = leg_data["ltp"] - res["intrinsic"]
    res.update(greeks(spot, strike, years, leg_data["iv"], rate, leg_data["leg"]))

    iv = leg_data["iv"]
    leg = leg_data["leg"]
    model_spot = price(spot, strike, years, iv, rate, leg)
    res["model_price_at_spot"] = model_spot
    if model_spot is None:
        res["reason"] = "option price could not be computed at spot"
        return res

    # The market LTP is what is actually paid, and it will not equal the model
    # price (different IV surface, spread, stale print). Scenario prices are
    # therefore RE-BASED so that the entry leg equals the LTP exactly. Without
    # this the risk term is ``ltp - model(stop)``, which goes negative whenever
    # the model price sits above the LTP and silently reports "no risk".
    if model_spot > 0:
        res["model_vs_ltp_gap_pct"] = (model_spot - leg_data["ltp"]) / leg_data["ltp"] * 100.0
    else:
        res["model_vs_ltp_gap_pct"] = None

    def _rebased(scenario_spot):
        raw = price(scenario_spot, strike, years, iv, rate, leg)
        if raw is None:
            return None
        return max(0.0, leg_data["ltp"] + (raw - model_spot))

    tgt_spot, stop_spot = scenario_spots(spot, target_pct, stop_pct, direction)
    res["option_at_target"] = _rebased(tgt_spot)
    res["option_at_stop"] = _rebased(stop_spot)
    res["underlying_at_target"] = tgt_spot
    res["underlying_at_stop"] = stop_spot

    if res["option_at_target"] is None or res["option_at_stop"] is None:
        res["reason"] = "option price could not be computed at the scenario legs"
        return res

    # Carry the exporter's persisted Greeks through, rather than recomputing
    # them here. The payload is the single source of truth, so a trade reported
    # to the user and the Greeks they see beside it cannot disagree.
    res["greeks"] = leg_data.get("greeks")

    risk = leg_data["ltp"] - res["option_at_stop"]
    reward = res["option_at_target"] - leg_data["ltp"]
    res["option_risk"] = risk
    res["option_reward"] = reward
    if risk <= 0:
        res["reason"] = "option is worth more at the underlying stop than at entry (risk <= 0)"
        res["rr"] = None
    else:
        res["rr"] = reward / risk
    return res


def _score_candidate(ev, forecast, btst=False, btst_min_dte=3):
    """Rank a candidate expiry on DTE, theta drag and forecast horizon.

    Both candidates are always scored; the caller picks. A very short DTE under
    a BTST (next-session) horizon is penalised rather than accepted blindly,
    because gamma/theta dominate that close to expiry and a near-expiry option
    can lose most of its value on a modest underlying move.
    """
    dte_days = ev.get("dte")
    theta = ev.get("theta")
    score = 0.0
    notes = []

    if dte_days is None:
        return -1e9, ["no DTE"]
    if btst and dte_days < btst_min_dte:
        score -= 100.0
        notes.append("DTE %d below BTST minimum %d" % (dte_days, btst_min_dte))
    elif btst and dte_days < 7:
        score -= 5.0
        notes.append("DTE %d is very short for an overnight hold" % dte_days)

    # Reward a delta that actually expresses the forecast without being
    # so far OTM that the payoff is mostly optionality the model cannot price.
    delta = ev.get("delta")
    if delta is not None:
        score += min(abs(delta), 0.7) * 10.0
        if delta is not None and delta != 0 and ev.get("leg") == "CE" and delta < 0:
            notes.append("call delta sign inconsistent")
    if theta is not None:
        ltp = ev.get("ltp") or 0.0
        if ltp > 0:
            # Daily theta as a fraction of premium: bigger is worse.
            drag = abs(theta) / ltp
            score -= drag * 20.0
            if drag > 0.05:
                notes.append("theta drag %.1f%%/day of premium" % (drag * 100))
    if ev.get("rr") is not None:
        score += min(ev["rr"], 3.0)
    return score, notes


def construct_option_trade(forecast, options_payload, rate=0.0, btst=False,
                           btst_min_dte=3, spot=None):
    """Build the option trade for a Layer A forecast.

    ``forecast`` supplies ``direction`` ("LONG"/"SHORT"), ``expected_move_pct``
    and ``invalidation_pct``. Both expiry candidates are evaluated and returned;
    the recommended one is chosen by :func:`_score_candidate` rather than by
    blindly taking the nearest expiry.

    Returns None when there is no usable option data -- the caller must then
    show no option trade rather than substituting underlying levels.
    """
    if not options_payload or not options_payload.get("candidates"):
        return None
    if not forecast or not forecast.get("direction"):
        return None

    direction = forecast["direction"]
    if direction not in ("LONG", "SHORT"):
        return None
    want_leg = "CE" if direction == "LONG" else "PE"

    base_spot = spot if spot is not None else options_payload.get("underlying")
    if base_spot is None:
        return None

    target_pct = forecast.get("expected_move_pct")
    stop_pct = forecast.get("invalidation_pct")
    if target_pct is None or stop_pct is None:
        return None

    # Rank expiries by DTE so "front" is the nearest and "next_monthly" is the
    # one after it. Derived from DTE rather than list position, because the
    # payload may be ordered any way and position would silently mislabel.
    candidates = sorted(
        options_payload["candidates"],
        key=lambda c: (c.get("dte") is None, c.get("dte")),
    )
    roles = {}
    for idx, cand in enumerate(candidates):
        roles[id(cand)] = "front" if idx == 0 else "next_monthly"

    evaluations = []
    for cand in candidates:
        # Nearest ATM strike first, so the top-ranked candidate is a sane one.
        legs = [l for l in cand.get("legs", []) if l.get("leg") == want_leg]
        if not legs:
            continue
        legs.sort(key=lambda l: abs((l.get("strike") or 0) - base_spot))
        for l in legs:
            ev = evaluate_leg(l, base_spot, rate, target_pct, stop_pct,
                              direction=direction)
            ev["expiry_role"] = roles[id(cand)]
            ev["is_monthly_candidate"] = cand.get("is_monthly", False)
            evaluations.append(ev)

    if not evaluations:
        return None

    scored = []
    for ev in evaluations:
        s, notes = _score_candidate(ev, forecast, btst=btst, btst_min_dte=btst_min_dte)
        scored.append((s, ev, notes))
    scored.sort(key=lambda t: t[0], reverse=True)
    best_score, best, best_notes = scored[0]

    out = dict(best)
    out["score"] = best_score
    out["notes"] = best_notes
    out["direction"] = direction
    out["candidates_considered"] = len(evaluations)
    out["all_candidates"] = [
        {
            "expiry": e.get("expiry"),
            "dte": e.get("dte"),
            "strike": e.get("strike"),
            "leg": e.get("leg"),
            "expiry_role": e.get("expiry_role"),
            "rr": e.get("rr"),
            "score": s,
            "reason": e.get("reason"),
            "notes": n,
        }
        for s, e, n in scored
    ]
    return out
