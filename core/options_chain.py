"""Option-chain selection and slimming for the dashboard payload.

The exporter already fetches a live chain (jugaad_data) but historically kept
only three OI-derived scalars (support / resistance / max pain). Layer B needs
actual option prices, so this module selects a small, explicit slice of the
chain and reduces it to the fields the risk model consumes.

Findings from the live endpoint that this module is written against
(verified 2026-09-30, NIFTY + NIFTY BANK):

  * ``optionType`` is ``None`` on every leg. The CE/PE distinction exists only
    in the row key (``row["CE"]`` / ``row["PE"]``), so it must be read there.
  * ``impliedVolatility`` is ``0`` on far out-of-the-money rows. Pricing off a
    zero IV produces a deterministic zero, so those rows are dropped.
  * Expiry appears in two formats: ``dd-MMM-yyyy`` in ``records["expiryDates"]``
    but ``dd-mm-yyyy`` on each leg. Both are normalised to ISO here.
  * ``records["strikePrices"]`` is a sparse, unevenly spaced list
    (1500, 3000, 4500 ...) and cannot be used to infer the strike interval.
    The interval is derived from the strikes actually present in ``data``.
  * A single fetch returns ONE expiry, and with no explicit ``expiry`` the
    default is ``expiryDates[0]`` -- which was already past-dated, carrying a
    perfectly linear fake-IV ramp (1.02, 2.54, 3.96 ...). Expiry must therefore
    be requested explicitly and past-dated expiries skipped.
  * Greeks are not provided. delta/gamma/theta/vega are computed downstream in
    :mod:`core.option_pricing`.

No function here invents a field. Anything the live response does not supply is
either absent from the slim payload or explicitly ``None``.
"""
from __future__ import annotations

import datetime as _dt
import math
from collections import Counter

from core.option_pricing import greeks, resolve_risk_free_rate

# Legs are keyed by row key, not by the (always-None) ``optionType`` field.
LEGS = ("CE", "PE")

# Fields worth keeping from each leg. Anything not in this list is dropped.
_SLIM_FIELDS = {
    "lastPrice": "ltp",
    "buyPrice1": "bid",
    "sellPrice1": "ask",
    "buyQuantity1": "bid_qty",
    "sellQuantity1": "ask_qty",
    "totalTradedVolume": "volume",
    "openInterest": "oi",
    "changeinOpenInterest": "change_oi",
    "impliedVolatility": "iv",
}


def parse_expiry(value):
    """Normalise an NSE expiry string to a ``date``.

    Accepts ``dd-MMM-yyyy`` (records.expiryDates) and ``dd-mm-yyyy`` (leg
    fields). Returns None for anything unparseable rather than guessing.
    """
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%d-%B-%Y", "%Y-%m-%d"):
        try:
            return _dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def dte(expiry, today):
    """Calendar days to expiry. Negative once expired."""
    d = parse_expiry(expiry)
    if d is None or today is None:
        return None
    return (d - today).days


def select_expiries(expiry_dates, today, min_dte=1):
    """Front expiry plus the next monthly expiry after it.

    ``front`` is the nearest expiry that has not expired. ``monthly`` is the
    first expiry strictly after ``front`` that is the last one in its own
    calendar month.

    Monthly is *derived* from the calendar rather than assumed from a weekly
    series. That matters: NIFTY lists weeklies (06/13/19/27 Oct) with only the
    last one each month being a monthly, while NIFTY BANK lists only monthlies
    -- so for BANK the front expiry is itself a monthly and the rule must
    advance past it rather than returning the same date twice.

    Returns ``(front, monthly)`` as ISO strings, either of which may be None.
    """
    if not expiry_dates or today is None:
        return None, None

    valid = []
    for raw in expiry_dates:
        d = parse_expiry(raw)
        if d is not None and (d - today).days >= min_dte:
            valid.append(d)
    if not valid:
        return None, None
    valid = sorted(set(valid))

    # Last expiry within each (year, month) bucket == the monthly.
    last_of_month = {}
    for d in valid:
        key = (d.year, d.month)
        if key not in last_of_month or d > last_of_month[key]:
            last_of_month[key] = d
    monthlies = sorted(last_of_month.values())

    front = valid[0]
    monthly = next((m for m in monthlies if m > front), None)
    return front.isoformat(), (monthly.isoformat() if monthly else None)


def _looks_percent(iv):
    """Heuristic guard for a feed that reports IV in percent rather than decimal.

    On the verified 27-Oct-2026 NIFTY chain, at-the-money IV was ~0.13 in
    decimal terms. A value above 3.0 is therefore far outside any plausible
    annualised volatility and must be percent.
    """
    return iv is not None and iv > 3.0


def normalise_iv(raw, min_iv=0.01):
    """Return IV as a decimal, or None when unusable.

    The live feed reports percent (e.g. ``13.0`` meaning 13%); some sources
    report decimal (``0.13``). Both are accepted, but a value above 3.0 is
    rejected outright rather than divided, because 300% annualised volatility
    is not a real market quote -- it is a data error that would otherwise flow
    silently into the risk model.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and math.isnan(raw):
        return None
    iv = float(raw)
    if _looks_percent(iv) and iv > 300.0:
        return None
    if _looks_percent(iv):
        iv = iv / 100.0
    if iv < min_iv:
        return None
    return iv


def strike_interval(strikes, spot=None, near_bands=6):
    """Modal strike gap, preferring gaps observed near the money.

    A plain median is wrong here. A real chain lists fine 50-point strikes near
    ATM and coarse 1000/1300/1500-point strikes far out, so the median lands
    somewhere between the two regimes and the selection window becomes arbitrary.
    The mode over near-ATM gaps is the exchange's actual listing interval.

    Returns None if fewer than two strikes exist.
    """
    vals = sorted({float(s) for s in strikes if s is not None})
    if len(vals) < 2:
        return None

    if spot is not None:
        near = [s for s in vals if abs(s - spot) <= 1000]
        # A handful of near strikes may not span two gaps; widen until it does.
        for band in (near_bands * 50, near_bands * 200, near_bands * 500, None):
            near = [s for s in vals if abs(s - spot) <= band] if band else vals
            gaps = [b - a for a, b in zip(near, near[1:]) if b > a]
            if gaps:
                return Counter(gaps).most_common(1)[0][0]

    gaps = [b - a for a, b in zip(vals, vals[1:]) if b > a]
    if not gaps:
        return None
    return Counter(gaps).most_common(1)[0][0]


def select_strikes(strikes, spot, interval, span=3):
    """ATM +/- ``span`` strikes, snapped to the chain's own interval.

    Returns the selected strikes ascending. Falls back to the strikes nearest
    ATM when the interval cannot be determined.
    """
    vals = sorted({float(s) for s in strikes if s is not None})
    if not vals or spot is None:
        return []
    if not interval:
        k = min(len(vals), 2 * span + 1)
        return sorted(vals, key=lambda s: abs(s - spot))[:k]

    step = float(interval)
    centre = round(spot / step) * step
    wanted = [centre + i * step for i in range(-span, span + 1)]

    selected = []
    for target in wanted:
        # Snap to the nearest actually-listed strike.
        best = min(vals, key=lambda s: abs(s - target))
        if best not in selected:
            selected.append(best)
    return sorted(selected)


def _slim_leg(leg_data, leg, strike, expiry_iso, today, min_iv):
    """Reduce one leg to the persisted fields, or None if unusable."""
    if not leg_data:
        return None

    ltp = leg_data.get("lastPrice")
    if ltp is None or (isinstance(ltp, (int, float)) and math.isnan(ltp)):
        return None

    # The feed reports IV in PERCENT (a valid 27-Oct NIFTY chain returns ~13.0
    # for 13%), while Black-Scholes requires a decimal. Verified against the
    # live response: passing 13.0 straight through inflates an ATM 27-day call
    # to ~11000 and makes every leg look risk-free. Normalised here so the
    # persisted ``iv`` is decimal and consistent for the risk model.
    iv = normalise_iv(leg_data.get("impliedVolatility"), min_iv=min_iv)

    out = {
        "strike": strike,
        "expiry": expiry_iso,
        "dte": dte(expiry_iso, today),
        "leg": leg,
        "ltp": float(ltp),
        "iv": None if iv is None else float(iv),
    }
    for src, dst in _SLIM_FIELDS.items():
        if src == "lastPrice" or src == "impliedVolatility":
            continue
        val = leg_data.get(src)
        if isinstance(val, (int, float)) and math.isnan(val):
            val = None
        out[dst] = val
    return out


def slim_chain(rows, spot, today, expiry, strike_span=3, min_iv=0.01):
    """Slim one expiry's rows to ATM +/- span strikes, both legs.

    ``rows`` is the ``records["data"]`` list for a single expiry.
    """
    if not rows or spot is None or expiry is None:
        return []

    interval = None
    all_strikes = []
    for row in rows:
        s = row.get("strikePrice")
        if s is not None:
            all_strikes.append(s)
    interval = strike_interval(all_strikes, spot)
    wanted = select_strikes(all_strikes, spot, interval, strike_span)

    out = []
    for row in rows:
        strike = row.get("strikePrice")
        if strike is None or float(strike) not in wanted:
            continue
        strike = float(strike)
        for leg in LEGS:
            slim = _slim_leg(row.get(leg), leg, strike, expiry, today, min_iv)
            if slim is not None:
                out.append(slim)
    return out


def build_option_payload(chain_by_expiry, spot, today, strike_span=3, min_iv=0.01,
                         rate_info=None, source="live", monthly_expiries=None):
    """Assemble the persisted ``_options`` block.

    ``chain_by_expiry`` maps ISO expiry -> rows for that expiry, because the
    endpoint returns one expiry per call and each must be requested explicitly.

    ``monthly_expiries`` is the set of ISO expiries :func:`select_expiries`
    identified as monthlies. It is passed in rather than re-derived because
    "is monthly" is a calendar property of the expiry list, not of a single
    chain, and guessing it from DTE (DTE > 0 is true for every valid expiry)
    would mark all of them monthly.

    ``rate_info`` is the dict from :func:`option_pricing.resolve_risk_free_rate`.
    It carries both the rate used for pricing and its provenance, all of which
    is persisted so the UI can label the rate instead of inventing a label.

    Every expiry passed in has already been through :func:`select_expiries`, so
    past-dated series never reach this function.
    """
    rate_info = dict(rate_info or resolve_risk_free_rate(None))
    rate = rate_info["rate"]
    monthlies = set(monthly_expiries or ())
    candidates = []
    for expiry in sorted(chain_by_expiry):
        legs = slim_chain(chain_by_expiry[expiry], spot, today, expiry,
                          strike_span=strike_span, min_iv=min_iv)
        if not legs:
            continue
        d = dte(expiry, today)
        # Greeks are a function of spot/strike/time/vol only -- NOT of any
        # forecast -- so they are computed here and persisted. The browser then
        # only needs a price function for the scenario legs, instead of
        # re-deriving delta/gamma/theta/vega and risking a second
        # implementation disagreeing with this one.
        for leg in legs:
            years = (leg["dte"] / 365.0) if (leg["dte"] or 0) > 0 else None
            leg["greeks"] = greeks(spot, leg["strike"], years,
                                   leg["iv"], rate, leg["leg"])
        candidates.append({
            "expiry": expiry,
            "dte": d,
            "is_monthly": expiry in monthlies,
            "underlying": spot,
            "legs": legs,
        })

    if not candidates:
        return None
    # front = smallest DTE; the other is the next monthly.
    candidates.sort(key=lambda c: (c["dte"] is None, c["dte"]))
    return {
        "source": source,
        "fetched_at": today.isoformat() if hasattr(today, "isoformat") else str(today),
        "underlying": spot,
        # Explicit translation: rate_info uses short internal names, the payload
        # uses the config key names so the UI can read them without a lookup
        # table. A blind **rate_info would silently persist None for every field.
        "risk_free_rate": rate_info.get("declared"),
        "risk_free_rate_configured": rate_info.get("configured"),
        "risk_free_rate_source": rate_info.get("source"),
        "risk_free_rate_as_of": rate_info.get("as_of"),
        "risk_free_rate_type": rate_info.get("type"),
        "risk_free_rate_refresh": rate_info.get("refresh"),
        "candidates": candidates,
    }
