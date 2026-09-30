import json
import os
import sys
import itertools
from pathlib import Path

import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
import yfinance
import requests
import re
import yfinance as yf

# Run as `python scripts/export_dashboard.py` from anywhere: make the project
# root importable so `core` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.data as core_data
import core.settings as core_settings

try:
    from nselib import capital_market
    NSELIB_AVAILABLE = True
except ImportError:
    NSELIB_AVAILABLE = False

def build_wfo_meta(best_final_edge, best_final_stats, oos_edge, oos_win_rate,
                   oos_trades, stability, val_window, oos_window,
                   ui_boxes=None, symbol=None, val_period=None,
                   oos_period=None, oos_predictions=None):
    """Build the BTST ``_meta`` block for one symbol's walk-forward.

    Kept as a pure function so the "SIGNAL" / "NO SIGNAL" contract can be unit
    tested without running the full walk-forward pass (which is minutes of
    feature work per instrument).

    Contract:
      * SIGNAL is only ever emitted when the OOS edge is strictly positive and
        at least one feature was selected. Never force it.
      * NO SIGNAL is a legitimate, fully-populated result: it reports the same
        edge/trade fields as SIGNAL plus a human-readable ``reason`` naming the
        gate that failed, and an empty feature list. The UI relies on this to
        explain itself instead of rendering silent blanks.
      * ``wfo_candidate_features`` carries the combination the optimizer
        actually selected, so the UI can compute a forecast from it even when
        the OOS gate rejected it. It is NOT a validated feature set: it must
        never be promoted into ``wfo_optimal_features`` and must never be used
        to qualify a trade. An empty list means no combination cleared the
        5-trade floor, which is distinct from the field being absent.
      * ``oos_predictions`` holds one record per out-of-sample observation
        (see ``run_wfo``). It is what lets the Historical view answer "what did
        the model predict on this date?". It is deliberately SEPARATE from
        ``status``: the status is a property of the whole evaluation and must
        never be presented as an individual day's verdict. A date absent from
        this list has no per-date OOS prediction, and the UI must say so rather
        than substituting the global status.
      * INSUFFICIENT_DATA is produced by ``build_insufficient_meta`` instead,
        for instruments with too little clean history to test.
    """
    # val_* comes from the validation window (used for feature selection);
    # oos_* comes from the strictly-later out-of-sample window. The two are
    # disjoint by construction (see wfo_window_bounds), so the UI can present
    # them as a genuine validation -> OOS generalisation check.
    common = {
        "symbol": symbol,
        "val_edge": round(float(best_final_edge), 4) if best_final_edge != -999 else 0,
        "val_win_rate": round(float(best_final_stats[1] * 100), 1) if best_final_stats else 0,
        "val_trades": int(best_final_stats[0]) if best_final_stats else 0,
        "oos_edge": round(float(oos_edge), 4),
        "oos_win_rate": round(float(oos_win_rate * 100), 1) if oos_trades > 0 else 0,
        "oos_trades": int(oos_trades),
        "stability": stability,
        "val_window": val_window,
        "oos_window": oos_window,
        "val_period": val_period,
        "oos_period": oos_period,
        "oos_predictions": list(oos_predictions) if oos_predictions else [],
    }

    if best_final_edge > 0 and oos_edge > 0 and ui_boxes:
        meta = dict(common)
        meta["wfo_optimal_features"] = list(ui_boxes)
        meta["wfo_candidate_features"] = list(ui_boxes)
        meta["status"] = "SIGNAL"
        meta["reason"] = "OOS edge > 0"
        return meta

    if best_final_stats is None:
        reason = "no feature combination cleared the 5-trade floor"
    elif best_final_edge <= 0:
        reason = "validation edge <= 0"
    else:
        reason = "OOS edge <= 0"

    meta = dict(common)
    meta["wfo_optimal_features"] = []
    meta["wfo_candidate_features"] = list(ui_boxes) if ui_boxes else []
    meta["status"] = "NO SIGNAL"
    meta["reason"] = reason
    return meta

    if best_final_stats is None:
        reason = "no feature combination cleared the 5-trade floor"
    elif best_final_edge <= 0:
        reason = "validation edge <= 0"
    else:
        reason = "OOS edge <= 0"

    meta = dict(common)
    meta["wfo_optimal_features"] = []
    meta["wfo_candidate_features"] = list(ui_boxes) if ui_boxes else []
    meta["status"] = "NO SIGNAL"
    meta["reason"] = reason
    return meta


def wfo_window_bounds(n_days, val_window, oos_window):
    """Half-open index bounds for the BTST walk-forward pass.

    Returns ``(val_start, val_end, oos_start, oos_end)`` where each range is
    ``[start, end)``. The validation window used for feature selection is
    strictly earlier than, and disjoint from, the out-of-sample window, so the
    selected combination is never scored on the data it is evaluated on. With
    the configured 252/126 split the two windows sit back to back:
    ``val_end == oos_start``.

    Kept as a pure function so the "selection never sees OOS" property is unit
    tested without running the ~90s network-backed walk-forward.
    """
    oos_start = n_days - oos_window - 1
    oos_end = n_days - 1
    val_start = oos_start - val_window
    val_end = oos_start
    return val_start, val_end, oos_start, oos_end


# --- Shared walk-forward methodology ---------------------------------------
# These are METHOD parameters, identical for every symbol. Each instrument is
# validated independently over its own history and selects its own features;
# nothing is inherited from NIFTY.

WFO_FEATURES = ['z_rsi', 'z_stochrsi', 'z_ema_diff', 'z_price_ema', 'z_vol']
# z_atr is intentionally excluded: no BTST checkbox and no calculateTopK branch,
# so it could only be selected and then misreported in the UI.

WFO_FEATURE_UI = {
    'z_rsi': 'chk-rsi',
    'z_stochrsi': 'chk-stochrsi',
    'z_ema_diff': 'chk-ema59',
    'z_price_ema': 'chk-ema20',
    'z_vol': 'chk-deltaoi',
    'z_vix': 'chk-vix',
    'z_bn_rel': 'chk-divergence',
}

WFO_VAL_WINDOW = 252
WFO_OOS_WINDOW = 126
WFO_EMBARGO = 5          # days excluded from the analogue search per step
WFO_K = 50               # analogues per prediction
WFO_LONG_BAND = 55       # prob_up >= this -> long
WFO_SHORT_BAND = 45      # prob_up <= this -> short
WFO_MIN_TRADES = 5       # below this a window is treated as unvalidated
WFO_MIN_FEATURES = 2
WFO_MAX_FEATURES = 5

# 0.01 PERCENTAGE POINTS per feature. All edge arithmetic below runs in
# percent, which is the unit the original "0.01% per feature" comment intended
# and the unit the intraday walk-forward already uses (export_intraday.py
# computes Forward_Return * 100).
#
# The pre-correction daily code fed Close.pct_change() FRACTIONS into this same
# 0.01, so the penalty was effectively 1 percentage point per feature -- about
# 100x its stated size. That suppressed edge estimates near zero and biased
# selection toward the smallest (2-feature) combinations. It is a selection-side
# error only: the OOS edge definition itself is unchanged, and the
# `OOS edge <= 0 -> NO SIGNAL` gate is untouched.
WFO_COMPLEXITY_PENALTY = 0.01


def _wfo_direction(prob_up):
    """Map a prob_up percentage onto a trade direction."""
    if prob_up is None:
        return "NO_PREDICTION"
    if prob_up >= WFO_LONG_BAND:
        return "LONG"
    if prob_up <= WFO_SHORT_BAND:
        return "SHORT"
    return "FLAT"


def run_wfo(df, symbol, features=None, val_window=WFO_VAL_WINDOW,
            oos_window=WFO_OOS_WINDOW, embargo=WFO_EMBARGO, k=WFO_K,
            min_trades=WFO_MIN_TRADES,
            complexity_penalty=WFO_COMPLEXITY_PENALTY,
            return_scale=100.0, verbose=True):
    """Independent walk-forward validation for one instrument.

    Returns ``(meta, diagnostics)``. ``meta`` is the ``_meta`` block for the
    symbol's dashboard file; ``diagnostics`` carries the unrounded figures and
    period bounds for offline inspection.

    Methodology (unchanged from the original walk-forward, except for the
    complexity-penalty unit fix):

      * One prediction per evaluation day: the 50 nearest prior days on the
        candidate feature subset, measured with a 5-day embargo so the match set
        can never contain the target day or its immediate future. ``prob_up`` is
        the share of those analogues whose NEXT day closed up.
      * Selection: for each OOS day, every 2-5 feature combination is scored on
        the trailing ``val_window`` days and the best is frozen BEFORE that day's
        OOS observation is scored. The OOS day never influences its own
        selection.
      * The final feature set for the *live* forecast is chosen on the
        ``val_window`` days immediately preceding the OOS window -- disjoint from
        it, so the reported validation metrics are not OOS metrics in disguise.
      * Aggregation: edge = sum(signed next-day returns over trades) x win rate.
        Validation edge carries the complexity penalty; OOS edge is raw.

    ``return_scale`` converts the fraction returns into percent. The gate is
    sign-based and therefore scale-invariant, but the penalty is an absolute
    subtraction, so the scale must match the penalty's unit.
    """
    features = list(features or WFO_FEATURES)
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]

    if missing:
        if verbose:
            print(f"  WFO [{symbol}]: missing features {missing}; skipping.")
        return build_insufficient_meta(
            symbol,
            f"instruments lacks required feature(s): {', '.join(missing)}",
        ), {"symbol": symbol, "ok": False, "reason": "missing_features"}

    # dropna on the feature columns only; the frame must keep 'Return', which
    # is the next-day outcome the walk-forward scores against.
    df_wfo = df.dropna(subset=available)
    n_days = len(df_wfo)
    eval_window = val_window + oos_window

    if n_days <= eval_window:
        if verbose:
            print(f"  WFO [{symbol}]: not enough history "
                  f"({n_days} rows, need > {eval_window}).")
        return build_insufficient_meta(
            symbol,
            f"only {n_days} clean rows of history; "
            f"needs more than {eval_window} for a "
            f"{val_window}-day validation + {oos_window}-day OOS split",
        ), {"symbol": symbol, "ok": False, "reason": "insufficient_history",
            "n_days": n_days}

    feature_matrix = df_wfo[available].values
    returns_wfo = df_wfo['Return'].values * return_scale
    dates = df_wfo.index

    if verbose:
        print(f"WFO [{symbol}]: {n_days} clean rows, "
              f"val={val_window} oos={oos_window} (features={len(available)})")

    combos = []
    for r in range(WFO_MIN_FEATURES, min(WFO_MAX_FEATURES, len(available)) + 1):
        combos.extend(itertools.combinations(range(len(available)), r))

    # predictions[c_idx][i] = (prob_up, expected_return, actual_next_return)
    # Precomputed over the whole eval window so no rolling step hits a gap.
    predictions = {c_idx: {} for c_idx in range(len(combos))}
    for c_idx, comb in enumerate(combos):
        mat = feature_matrix[:, comb]
        for i in range(n_days - eval_window - 1, n_days - 1):
            search_mat = mat[:i - embargo]
            if len(search_mat) < k:
                continue
            diffs = search_mat - mat[i]
            dists = np.sum(diffs ** 2, axis=1)
            top_n = min(k, len(dists))
            top_idx = np.argpartition(dists, top_n)[:top_n]
            next_rets = returns_wfo[top_idx + 1]
            prob_up = (np.sum(next_rets > 0) / top_n) * 100
            expected_ret = float(np.mean(next_rets))
            actual_next_ret = float(returns_wfo[i + 1])
            predictions[c_idx][i] = (prob_up, expected_ret, actual_next_ret)

    def calc_edge(c_idx, start_i, end_i):
        win_count, total_trades, cumulative_ret = 0, 0, 0.0
        for i in range(start_i, end_i):
            rec = predictions[c_idx].get(i)
            if rec is None:
                continue
            prob_up, _, actual_ret = rec
            if prob_up >= WFO_LONG_BAND:
                total_trades += 1
                if actual_ret > 0:
                    win_count += 1
                cumulative_ret += actual_ret
            elif prob_up <= WFO_SHORT_BAND:
                total_trades += 1
                if actual_ret < 0:
                    win_count += 1
                cumulative_ret -= actual_ret
        if total_trades < min_trades:
            return -999.0, 0, 0.0, 0.0
        win_rate = win_count / total_trades
        avg_ret = cumulative_ret / total_trades
        edge = cumulative_ret * win_rate
        adj_edge = edge - (len(combos[c_idx]) * complexity_penalty)
        return adj_edge, total_trades, win_rate, avg_ret

    val_sel_start, val_sel_end, oos_start, oos_end = wfo_window_bounds(
        n_days, val_window, oos_window
    )

    # --- Rolling OOS: freeze a config per day, then score that day -----------
    oos_trades, oos_wins, oos_cum_ret = 0, 0, 0.0
    selected_combs_freq = {c_idx: 0 for c_idx in range(len(combos))}
    oos_predictions = []

    for today_i in range(oos_start, oos_end):
        val_start = today_i - val_window
        val_end = today_i

        best_val_edge, best_c_idx = -999.0, None
        for c_idx in range(len(combos)):
            adj_edge, _, _, _ = calc_edge(c_idx, val_start, val_end)
            if adj_edge > best_val_edge:
                best_val_edge = adj_edge
                best_c_idx = c_idx

        date_str = pd.Timestamp(dates[today_i]).strftime('%Y-%m-%d')
        actual_ret = float(returns_wfo[today_i + 1])

        if best_c_idx is None or best_val_edge <= 0:
            # No combination cleared the validation gate, so the model held no
            # position. Recorded explicitly rather than dropped, so a per-date
            # lookup can distinguish "no position" from "not evaluated".
            oos_predictions.append({
                "date": date_str,
                "direction": "NO_MODEL",
                "prob_up": None,
                "expected_return": None,
                "actual_return": round(actual_ret, 4),
                "correct": None,
                "features": [],
                "designation": "OOS",
                "note": "no feature combination cleared the validation gate",
            })
            continue

        selected_combs_freq[best_c_idx] += 1
        rec = predictions[best_c_idx].get(today_i)
        if rec is None:
            # Too little prior history to build a match set for this day. Left
            # absent on purpose: the UI reports "no per-date OOS prediction".
            continue

        prob_up, expected_ret, actual_ret = rec
        direction = _wfo_direction(prob_up)
        correct = None
        if direction == "LONG":
            oos_trades += 1
            correct = actual_ret > 0
            if correct:
                oos_wins += 1
            oos_cum_ret += actual_ret
        elif direction == "SHORT":
            oos_trades += 1
            correct = actual_ret < 0
            if correct:
                oos_wins += 1
            oos_cum_ret -= actual_ret

        oos_predictions.append({
            "date": date_str,
            "direction": direction,
            "prob_up": round(float(prob_up), 2),
            "expected_return": round(float(expected_ret), 4),
            "actual_return": round(float(actual_ret), 4),
            "correct": correct,
            "features": [available[idx] for idx in combos[best_c_idx]],
            "designation": "OOS",
        })

    oos_win_rate = (oos_wins / oos_trades) if oos_trades > 0 else 0.0
    oos_avg_ret = (oos_cum_ret / oos_trades) if oos_trades > 0 else 0.0
    oos_edge = oos_cum_ret * oos_win_rate

    # --- Final config for the live forecast, chosen on the pre-OOS window ---
    best_final_edge, best_final_c_idx, best_final_stats = -999.0, None, None
    for c_idx in range(len(combos)):
        adj_edge, t, wr, ar = calc_edge(c_idx, val_sel_start, val_sel_end)
        if adj_edge > best_final_edge:
            best_final_edge = adj_edge
            best_final_c_idx = c_idx
            best_final_stats = (t, wr, ar)

    # Feature stability: share of OOS steps in which a feature was selected.
    # Denominator is the number of steps that actually took a model, not the
    # window length, so a run of NO_MODEL days cannot deflate every feature.
    stability = {}
    feature_freq = {f: 0 for f in available}
    n_selected_steps = sum(selected_combs_freq.values())
    for c_idx, count in selected_combs_freq.items():
        if count > 0:
            for f_idx in combos[c_idx]:
                feature_freq[available[f_idx]] += count
    if n_selected_steps > 0:
        for f, freq in feature_freq.items():
            # Only mapped features. An unmapped name (e.g. "z_atr") would reach
            # the UI as though it were a selected checkbox id.
            if f in WFO_FEATURE_UI:
                stability[WFO_FEATURE_UI[f]] = round(
                    (freq / n_selected_steps) * 100, 1
                )

    ui_boxes = None
    if best_final_c_idx is not None:
        best_comb = [available[idx] for idx in combos[best_final_c_idx]]
        ui_boxes = [WFO_FEATURE_UI[f] for f in best_comb if f in WFO_FEATURE_UI]

    val_period = {
        "start": pd.Timestamp(dates[val_sel_start]).strftime('%Y-%m-%d'),
        "end": pd.Timestamp(dates[val_sel_end - 1]).strftime('%Y-%m-%d'),
    }
    oos_period = {
        "start": pd.Timestamp(dates[oos_start]).strftime('%Y-%m-%d'),
        "end": pd.Timestamp(dates[oos_end - 1]).strftime('%Y-%m-%d'),
    }

    meta = build_wfo_meta(
        best_final_edge=best_final_edge,
        best_final_stats=best_final_stats,
        oos_edge=oos_edge,
        oos_win_rate=oos_win_rate,
        oos_trades=oos_trades,
        stability=stability,
        val_window=val_window,
        oos_window=oos_window,
        ui_boxes=ui_boxes,
        symbol=symbol,
        val_period=val_period,
        oos_period=oos_period,
        oos_predictions=oos_predictions,
    )

    diagnostics = {
        "symbol": symbol,
        "ok": True,
        "n_days": n_days,
        "features_evaluated": available,
        "combinations": len(combos),
        "val_period": val_period,
        "oos_period": oos_period,
        "val_edge_raw": (round(best_final_edge + len(combos[best_final_c_idx]) * complexity_penalty, 6)
                         if best_final_c_idx is not None else None),
        "val_edge_penalised": round(float(best_final_edge), 6),
        "val_trades": best_final_stats[0] if best_final_stats else 0,
        "val_win_rate": round(best_final_stats[1], 6) if best_final_stats else None,
        "val_avg_return": round(best_final_stats[2], 6) if best_final_stats else None,
        "oos_edge": round(float(oos_edge), 6),
        "oos_trades": oos_trades,
        "oos_win_rate": round(oos_win_rate, 6),
        "oos_avg_return": round(oos_avg_ret, 6),
        "oos_cumulative_return": round(float(oos_cum_ret), 6),
        "oos_records": len(oos_predictions),
        "oos_no_model_days": sum(1 for p in oos_predictions
                                 if p["direction"] == "NO_MODEL"),
        "selected_features": [available[idx] for idx in combos[best_final_c_idx]]
        if best_final_c_idx is not None else [],
        "complexity_penalty_per_feature": complexity_penalty,
        "return_scale": return_scale,
    }

    if verbose:
        print(f"  WFO [{symbol}]: {meta['status']} ({meta['reason']}) | "
              f"val_edge={meta['val_edge']} oos_edge={meta['oos_edge']} "
              f"oos_trades={meta['oos_trades']} oos_win={meta['oos_win_rate']}%")
    return meta, diagnostics


def build_insufficient_meta(symbol, reason):
    """A symbol-level ``_meta`` that refuses to imply validation it lacks.

    Emitted instead of a fabricated walk-forward result when an instrument has
    too little clean history, or lacks a feature the methodology requires. The
    UI must treat this as UNVALIDATED: no edge figures, no features, and the
    trade gate closed, because nothing was actually tested.
    """
    return {
        "symbol": symbol,
        "val_edge": 0,
        "val_win_rate": 0,
        "val_trades": 0,
        "oos_edge": 0,
        "oos_win_rate": 0,
        "oos_trades": 0,
        "stability": {},
        "val_window": WFO_VAL_WINDOW,
        "oos_window": WFO_OOS_WINDOW,
        "val_period": None,
        "oos_period": None,
        "oos_predictions": [],
        "wfo_optimal_features": [],
        "wfo_candidate_features": [],
        "status": "INSUFFICIENT_DATA",
        "reason": reason,
    }


def fetch_jugaad_backfill(spec, from_date, to_date):
    """Recent-date OHLC for one instrument from NSE via jugaad-data.

    ``spec`` is ``(kind, symbol)`` with kind in {"index", "stock"}, or None to
    skip the backfill for that instrument entirely. Returns a tz-naive
    Open/High/Low/Close frame indexed by date, or None on any failure.

    Every failure path returns None rather than raising, so a wrong symbol
    name or an NSE outage degrades to "no backfill" instead of aborting the
    export part-way through and leaving a half-written dashboard file.
    """
    if not spec:
        return None
    kind, symbol = spec
    try:
        if kind == "stock":
            from jugaad_data.nse import stock_df as _fetch
        else:
            from jugaad_data.nse import index_df as _fetch
        ns = _fetch(symbol=symbol, from_date=from_date, to_date=to_date)
    except Exception as exc:
        print(f"[jugaad] {kind} fetch failed for {symbol}: {exc}")
        return None

    if ns is None or getattr(ns, "empty", True):
        return None

    # index_df() dates the session under HistoricalDate; stock_df() under DATE.
    date_col = "HistoricalDate" if "HistoricalDate" in ns.columns else "DATE"
    if date_col not in ns.columns:
        print(f"[jugaad] {symbol}: no date column, skipping backfill")
        return None

    out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(ns[date_col].to_numpy())))
    for src, dst in (("OPEN", "Open"), ("HIGH", "High"), ("LOW", "Low"), ("CLOSE", "Close")):
        if src not in ns.columns:
            print(f"[jugaad] {symbol}: missing {src}, skipping backfill")
            return None
        # .to_numpy() is required: assigning a Series would align it against
        # out's DatetimeIndex by label while ns still carries a RangeIndex, so
        # every value would land as NaN and the concat would append an all-NaN
        # session instead of real OHLC.
        out[dst] = pd.to_numeric(ns[src], errors="coerce").to_numpy()
    out.index.name = None
    out = out.dropna()
    # jugaad's stock_df() returns a handful of Sunday-stamped rows that are not
    # NSE sessions. Appending them invents non-trading days in the history, so
    # keep weekdays only. This only ever removes rows (NSE publishes no
    # Saturday/Sunday index sessions), and it leaves the master-DB path -- whose
    # own stray weekend rows are pre-existing and out of scope here -- untouched.
    out = out[out.index.dayofweek < 5]
    if out.empty:
        print(f"[jugaad] {symbol}: no usable rows, skipping backfill")
        return None
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)
    return out.sort_index()


def _fetch_option_chain(chain_spec):
    """Option chain for one instrument, or None if unavailable.

    ``chain_spec`` is (kind, symbol) with kind in {"index", "stock"}. Previously
    this was hardcoded to the NIFTY index chain and run for every instrument, so
    Bank Nifty was published Nifty's option strikes against a spot ~32,000
    points away. Returns None rather than raising so a failed fetch degrades to
    "no options data" (the card then shows N/A) instead of falling back to
    another instrument's levels.
    """
    if not chain_spec:
        return None
    kind, symbol = chain_spec
    from jugaad_data.nse import NSELive
    n = NSELive()
    if kind == "stock":
        return n.equities_option_chain(symbol)
    return n.index_option_chain(symbol)


def _strike_spacing(pe_data, ce_data):
    """Median gap between consecutive strikes, or None if undeterminable.

    The 300/1000-point windows this exporter used to hardcode are Nifty's
    scale. Bank Nifty lists strikes 500 apart around 55,000, so a +/-300 window
    contains no strikes at all and support/resistance silently come back None.
    Deriving the window from the chain's own strike spacing keeps Nifty on
    exactly 300/1000 (50-point strikes -> 6x and 20x) while scaling to whatever
    the instrument actually trades.
    """
    strikes = sorted({x['strikePrice'] for x in pe_data + ce_data if x.get('strikePrice') is not None})
    if len(strikes) < 2:
        return None
    gaps = [b - a for a, b in zip(strikes, strikes[1:]) if b > a]
    if not gaps:
        return None
    return sorted(gaps)[len(gaps) // 2]


def _strike_bands(pe_data, ce_data):
    """(support, max_pain, momentum) strike windows for this chain, or None.

    Each is a multiple of the chain's own strike spacing, so NIFTY (50-pt
    strikes) resolves to exactly the 300 / 1000 / 500 this file used to
    hardcode -- Nifty's output is unchanged -- while NIFTY BANK (500-pt
    strikes) gets 3000 / 10000 / 5000 instead of windows too narrow to hold a
    single strike, which is what silently produced empty support/resistance.
    """
    spacing = _strike_spacing(pe_data, ce_data)
    if spacing is None:
        return None
    return 6 * spacing, 20 * spacing, 10 * spacing


def _fetch_option_chain_for_expiry(chain_spec, expiry_iso):
    """Option chain for ONE explicit expiry, or None.

    The live endpoint returns a single expiry per call and defaults to the FIRST
    listed expiry when none is requested. On 2026-09-30 that default was
    29-Sep-2026, already past-dated, and its IV column was a perfectly linear
    ramp (1.02, 2.54, 3.96 ...) rather than a market smile. Every expiry is
    therefore requested explicitly, and past-dated ones are filtered upstream by
    core.options_chain.select_expiries.
    """
    if not chain_spec:
        return None
    kind, symbol = chain_spec
    from jugaad_data.nse import NSELive
    n = NSELive()
    try:
        if kind == "stock":
            return n.equities_option_chain(symbol, expiry=expiry_iso)
        return n.index_option_chain(symbol, expiry=expiry_iso)
    except Exception as exc:
        print(f"[options] chain fetch failed for {expiry_iso}: {exc}")
        return None


def build_slim_option_payload(chain_spec, settings, today=None):
    """Fetch and slim the front + next-monthly chains into a bounded payload.

    Returns the ``_options`` block or None. This never raises: a failed fetch
    degrades to "no option data" so the UI shows N/A rather than substituting
    another instrument's levels.
    """
    from core import options_chain as oc
    from core.option_pricing import resolve_risk_free_rate

    opts = settings.get("options", {}) if settings else {}
    strike_span = opts.get("expiry_strike_span", 3)
    min_dte = opts.get("min_dte", 1)
    min_iv = opts.get("min_iv", 0.01)
    # Resolution (including the documented Indian 10Y G-Sec proxy value and its
    # provenance) happens once, here, and is carried into the payload verbatim so
    # the UI labels the rate the exporter actually priced with.
    rate_info = resolve_risk_free_rate(opts)

    try:
        probe = _fetch_option_chain(chain_spec)
    except Exception as exc:
        print(f"[options] chain probe failed: {exc}")
        return None
    if probe is None:
        return None
    records = probe.get("records") or {}
    spot = records.get("underlyingValue")
    expiry_labels = records.get("expiryDates")
    if not spot or not expiry_labels:
        return None

    today = today or datetime.now().date()
    front, monthly = oc.select_expiries(expiry_labels, today, min_dte=min_dte)
    wanted = [e for e in (front, monthly) if e]
    if not wanted:
        print("[options] no non-expired expiry available; skipping option payload")
        return None

    label_to_iso = {}
    for lbl in expiry_labels:
        d = oc.parse_expiry(lbl)
        if d is not None:
            label_to_iso[d.isoformat()] = lbl

    by_expiry = {}
    for iso in wanted:
        lbl = label_to_iso.get(iso)
        if lbl is None:
            continue
        chain = probe if lbl == expiry_labels[0] else _fetch_option_chain_for_expiry(chain_spec, lbl)
        if chain is None:
            continue
        rows = (chain.get("records") or {}).get("data")
        if rows:
            by_expiry[iso] = rows

    payload = oc.build_option_payload(
        by_expiry, spot, today, strike_span=strike_span, min_iv=min_iv,
        rate_info=rate_info,
        monthly_expiries={monthly} if monthly else set(),
    )
    if payload is None:
        print("[options] chain contained no usable near-ATM legs; skipping option payload")
    return payload


_EXTERNAL_CLOSE_CACHE: dict[str, pd.Series] = {}


def _external_close(ticker: str) -> pd.Series:
    """Last-close series for an external ticker, downloaded once per run.

    ^INDIAVIX, INR=X and ^GSPC carry the same 10y window for every instrument,
    so fetching them inside process_symbol() repeated the identical download
    three times. A Cloud Run execution is a fresh process, so the cache cannot
    serve stale data across runs.
    """
    if ticker not in _EXTERNAL_CLOSE_CACHE:
        _EXTERNAL_CLOSE_CACHE[ticker] = yf.Ticker(ticker).history(period="10y")["Close"]
    return _EXTERNAL_CLOSE_CACHE[ticker]


def process_symbol(symbol_name, cache_key, yf_symbol, jugaad_spec=None, chain_spec=None):
    data_file = f"dashboard_data.json" if symbol_name == "nifty" else f"dashboard_data_{symbol_name}.json"
    data = {}
    
    # Load from our new unified Yahoo+Fyers Database. Resolved via
    # core.data.ohlcv_path() so this can never drift from the cache writer.
    csv_path = str(core_data.ohlcv_path(cache_key, core_settings.load_settings()))
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        # Standardize column names
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        print(f"Loaded master DB for {symbol_name}: {len(df)} rows from {csv_path}")
    else:
        print(f"Master DB {csv_path} missing. Fallback to yfinance...")
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period="10y")
    
    # Fetch Institutional Data (VIX & USDINR). Memoised per run so the identical
    # ^INDIAVIX / INR=X / ^GSPC windows are downloaded once, not once per symbol.
    try:
        df["VIX"] = _external_close("^INDIAVIX")
        df["USDINR"] = _external_close("INR=X")
        df["SP500"] = _external_close("^GSPC")
        df["BANK_NIFTY"] = _external_close("^NSEBANK")
    except Exception as e:
        print("Failed to fetch institutional data:", e)

    # --- NEW: Fetch missing recent dates using jugaad-data (more reliable) ---
    # Now per instrument. This used to be gated on `symbol_name == "nifty"`
    # ("keep it simple and clean"), so bank_nifty was never repaired while
    # nifty was -- the two dashboards drifted apart on the same trading day and
    # bank_nifty was left short of sessions the cache already had.
    to_date = datetime.now().date()
    from_date = to_date - timedelta(days=30)
    # The master DB is tz-naive, but the yfinance fallback path builds a
    # tz-aware index. isin/concat then raise "Cannot compare tz-naive and
    # tz-aware timestamps" and kill the whole export, so normalise first.
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    ns = fetch_jugaad_backfill(jugaad_spec, from_date, to_date)
    if ns is not None and not ns.empty:
        # Both indexes are tz-naive here: HistoricalDate/DATE is already a plain
        # IST date, and localizing it to Asia/Kolkata made it tz-aware, so the
        # isin and the concat below raised "Cannot compare tz-naive and
        # tz-aware timestamps" and the whole backfill silently died.
        missing = ns[~ns.index.isin(df.index)]
        if not missing.empty:
            df = pd.concat([df, missing]).sort_index()
            print(f"Appended {len(missing)} missing days from jugaad_data ({jugaad_spec[1]}).")


    df["Return"] = df["Close"].pct_change()
    
    if "USDINR" in df.columns:
        df["USDINR_Return"] = df["USDINR"].pct_change()
    else:
        df["USDINR"] = 80.0
        df["USDINR_Return"] = 0.0
        
    if "VIX" not in df.columns:
        df["VIX"] = 15.0
        
    if "SP500" in df.columns:
        # Note: SP500 close yesterday provides the overnight cue for today's Nifty open
        df["SP500_Return"] = df["SP500"].pct_change().shift(1)
    else:
        df["SP500_Return"] = 0.0
        
    if "BANK_NIFTY" in df.columns:
        df["BANK_NIFTY_Return"] = df["BANK_NIFTY"].pct_change()
    else:
        df["BANK_NIFTY_Return"] = 0.0
    
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.ema(length=5, append=True)
    df.ta.ema(length=9, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.stochrsi(length=14, rsi_length=14, k=3, d=3, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.supertrend(length=7, multiplier=3.0, append=True)
    df.ta.atr(length=14, append=True)
    
    # Calculate continuous differentials
    df['EMA5_9_diff'] = df['EMA_5'] - df['EMA_9']
    df['Price_20EMA_diff'] = df['Close'] - df['EMA_20']
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, 1)
    df['BN_Rel'] = df['BANK_NIFTY_Return'] - df['Return']

    # Apply Rolling Z-Score Normalization (prevent lookahead bias)
    rolling_window = 252
    features_to_normalize = {
        'RSI_14': 'z_rsi',
        'STOCHRSIk_14_14_3_3': 'z_stochrsi',
        'EMA5_9_diff': 'z_ema_diff',
        'Price_20EMA_diff': 'z_price_ema',
        'ATRr_14': 'z_atr',
        'Vol_Ratio': 'z_vol',
        'VIX': 'z_vix',
        'BN_Rel': 'z_bn_rel'
    }
    
    for col, z_name in features_to_normalize.items():
        if col in df.columns:
            r_mean = df[col].rolling(rolling_window).mean()
            r_std = df[col].rolling(rolling_window).std().replace(0, 1e-5)
            df[z_name] = (df[col] - r_mean) / r_std
    

    # Fetch FII data
    mc_fii_dict = {}
    try:
        from jugaad_data.nse import NSELive
        n = NSELive()
        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        res = n.s.get(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.5", "Accept": "*/*"})
        if res.status_code == 200:
            for item in res.json():
                if item['category'] == 'FII/FPI':
                    # Parse date "24-Sep-2026" to "2026-09-24"
                    dt_obj = datetime.strptime(item['date'], '%d-%b-%Y')
                    d_str = dt_obj.strftime('%Y-%m-%d')
                    mc_fii_dict[d_str] = float(item['netValue'])
    except Exception as e:
        print(f"NSE FII fetch failed: {e}")
        pass

    try:
        url = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.php"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        matches = re.findall(r'\{"date":"([^"]+)".*?"fiiCM":"([^"]+)"', response.text)
        for d, val in matches:
            if d not in mc_fii_dict:
                mc_fii_dict[d] = float(val.replace(',', ''))
    except Exception as e:
        pass

    fii_df = None
    if NSELIB_AVAILABLE:
        try:
            fii_df = capital_market.fii_dii_trading_activity()
        except Exception as e:
            pass

    # Fetch Options Data for Support and Resistance globally (only valid for current time)
    global_options_support = None
    global_options_resistance = None
    global_options_max_pain = None
    global_options_momentum = None
    slim_options_payload = None
    try:
        oc = _fetch_option_chain(chain_spec)
        if oc is None:
            raise ValueError("no option chain for %s" % (chain_spec,))
        current_price = oc['records']['underlyingValue']

        # Calculate Support and Resistance from Option Chain (Max OI)
        pe_data = []
        ce_data = []
        for item in oc['records']['data']:
            if 'PE' in item:
                pe_data.append(item['PE'])
            if 'CE' in item:
                ce_data.append(item['CE'])

        # Strike windows scale with the instrument's own strike spacing. On
        # NIFTY (50-pt strikes) these resolve to exactly the 300 / 1000 / 500
        # this file previously hardcoded, so Nifty's output is unchanged, while
        # NIFTY BANK (500-pt strikes) gets 3000 / 10000 / 5000 instead of
        # windows too narrow to contain a single strike.
        bands = _strike_bands(pe_data, ce_data)
        if bands is None:
            print("[options] cannot infer strike spacing, skipping OI levels")
            raise ValueError("option chain has no usable strike spacing")
        support_band, pain_band, momentum_band = bands

        # Find Intraday Support (Max Put OI within support_band below current price)
        puts_below = [x for x in pe_data if (current_price - support_band) <= x['strikePrice'] < current_price]
        if puts_below:
            max_put = max(puts_below, key=lambda x: x['openInterest'])
            global_options_support = max_put['strikePrice']

        # Find Intraday Resistance (Max Call OI within support_band above current price)
        calls_above = [x for x in ce_data if current_price < x['strikePrice'] <= (current_price + support_band)]
        if calls_above:
            max_call = max(calls_above, key=lambda x: x['openInterest'])
            global_options_resistance = max_call['strikePrice']

        # Calculate Intraday Option Momentum (Delta OI)
        # Sum of changeinOpenInterest for strikes within +/- momentum_band
        put_delta = sum([x.get('changeinOpenInterest', 0) for x in pe_data if abs(x['strikePrice'] - current_price) <= momentum_band])
        call_delta = sum([x.get('changeinOpenInterest', 0) for x in ce_data if abs(x['strikePrice'] - current_price) <= momentum_band])
        
        global_options_momentum = None
        if call_delta > 0 or put_delta > 0:
            if put_delta > call_delta:
                global_options_momentum = "bullish"
            else:
                global_options_momentum = "bearish"

        # Calculate Max Pain
        all_strikes = sorted(list(set([x['strikePrice'] for x in pe_data + ce_data])))
        check_strikes = [s for s in all_strikes if current_price - pain_band <= s <= current_price + pain_band]
        
        min_loss = float('inf')
        for expiry_price in check_strikes:
            total_loss = 0
            for ce in ce_data:
                if expiry_price > ce['strikePrice']:
                    total_loss += (expiry_price - ce['strikePrice']) * ce['openInterest']
            for pe in pe_data:
                if expiry_price < pe['strikePrice']:
                    total_loss += (pe['strikePrice'] - expiry_price) * pe['openInterest']
            if total_loss < min_loss:
                min_loss = total_loss
                global_options_max_pain = expiry_price

        # Layer B needs actual option prices, not just OI levels. Reuses the
        # expiry calendar already fetched above; only the second expiry costs
        # an extra call, since one fetch returns one expiry.
        try:
            slim_options_payload = build_slim_option_payload(
                chain_spec, core_settings.load_settings()
            )
            if slim_options_payload is not None:
                cands = slim_options_payload["candidates"]
                print("[options] slim payload: %d expiries (%s), %d legs"
                      % (len(cands),
                         ", ".join("%s DTE %s" % (c["expiry"], c["dte"]) for c in cands),
                         sum(len(c["legs"]) for c in cands)))
        except Exception as e:
            print(f"[options] slim payload skipped: {e}")
    except Exception as e:
        print(f"Option Chain fetch failed: {e}")
        pass

    # Process all dates where we have at least 200 days of history
    trading_days = df.index[200:]
    returns_arr = df['Return'].values
    
    for date_obj in trading_days:
        date_str = date_obj.strftime("%Y-%m-%d")
        fii = "neutral"
        row = df.loc[date_obj]
        
        close = float(row["Close"])
        ema20 = float(row.get("EMA_20", close))
        ema200 = float(row.get("EMA_200", close))
        ema5 = float(row.get("EMA_5", close))
        ema9 = float(row.get("EMA_9", close))
        rsi = float(row.get("RSI_14", 50))
        stochrsi_k = float(row.get("STOCHRSIk_14_14_3_3", 50))
        stochrsi_d = float(row.get("STOCHRSId_14_14_3_3", 50))
        macd = float(row.get("MACD_12_26_9", 0))
        macd_signal = float(row.get("MACDs_12_26_9", 0))
        st_dir = row.get("SUPERTd_7_3.0", 0)
        
        import math

        
        high_val = float(row.get("High", close))

        
        if math.isnan(high_val): high_val = close

        
        low_val = float(row.get("Low", close))

        
        if math.isnan(low_val): low_val = close
        
        vix_val = float(row.get("VIX", 15.0))
        if math.isnan(vix_val): vix_val = 15.0
        
        usdinr_ret = float(row.get("USDINR_Return", 0.0))
        if math.isnan(usdinr_ret): usdinr_ret = 0.0
        
        sp500_ret = float(row.get("SP500_Return", 0.0))
        if math.isnan(sp500_ret): sp500_ret = 0.0
        
        bank_nifty_ret = float(row.get("BANK_NIFTY_Return", 0.0))
        if math.isnan(bank_nifty_ret): bank_nifty_ret = 0.0
        nifty_ret = float(row.get("Return", 0.0))
        if math.isnan(nifty_ret): nifty_ret = 0.0
        
        # Z-scores
        z_rsi = float(row.get("z_rsi", 0.0))
        z_stochrsi = float(row.get("z_stochrsi", 0.0))
        z_ema_diff = float(row.get("z_ema_diff", 0.0))
        z_price_ema = float(row.get("z_price_ema", 0.0))
        z_atr = float(row.get("z_atr", 0.0))
        z_vol = float(row.get("z_vol", 0.0))
        z_vix = float(row.get("z_vix", 0.0))
        z_bn_rel = float(row.get("z_bn_rel", 0.0))
        
        intermarket_div = "neutral"
        if nifty_ret > 0 and bank_nifty_ret < 0:
            intermarket_div = "bearish_divergence"
        elif nifty_ret < 0 and bank_nifty_ret > 0:
            intermarket_div = "bullish_divergence"


        # For historical dates, we don't have historical option chain, so we apply the live one (or None).
        # A more advanced script would only use this for the latest day.
        options_support = global_options_support if date_obj == trading_days[-1] else None
        options_resistance = global_options_resistance if date_obj == trading_days[-1] else None
        options_max_pain = global_options_max_pain if date_obj == trading_days[-1] else None
        options_momentum = global_options_momentum if date_obj == trading_days[-1] else None
        signals = {

        
            "close": round(close, 2),

        
            "high": round(high_val, 2),

        
            "low": round(low_val, 2),
            "daily_return_pct": round(row.get("Return", 0) * 100, 2),
        "options_support": options_support,
        "options_resistance": options_resistance,
        "options_max_pain": options_max_pain,
        "options_momentum": options_momentum,
        "india_vix": round(vix_val, 2),
        "vix_regime": "high_vol" if vix_val > 15 else "low_vol",
        "usdinr_trend": "bearish_for_nifty" if usdinr_ret > 0.002 else ("bullish_for_nifty" if usdinr_ret < -0.002 else "neutral"),
        "sp500_return_pct": round(sp500_ret * 100, 2),
        "sp500_cue": "bullish" if sp500_ret > 0.003 else ("bearish" if sp500_ret < -0.003 else "neutral"),
        "intermarket_divergence": intermarket_div,

        
            "ema20_signal": "bullish" if close > ema20 else "bearish",
            "ema200_signal": "bullish" if close > ema200 else "bearish",
            "ema20_diff_pct": round(((close - ema20) / ema20) * 100, 2),
            "ema200_diff_pct": round(((close - ema200) / ema200) * 100, 2),
            "ema5_signal": "bullish" if ema5 > ema9 else "bearish",
            "rsi_value": round(rsi, 2),
            "stochrsi_k": round(stochrsi_k, 2),
            "stochrsi_d": round(stochrsi_d, 2),
            "stochrsi_signal": "overbought" if stochrsi_k > 80 else ("oversold" if stochrsi_k < 20 else ("bullish crossover" if stochrsi_k > stochrsi_d else "bearish crossover")),
            "rsi_signal": "overbought" if rsi > 70 else ("oversold" if rsi < 30 else "neutral"),
            "macd_signal": "bullish" if macd > macd_signal else "bearish",
            "supertrend_signal": "bullish" if st_dir == 1 else "bearish",
            "z_rsi": round(z_rsi, 3) if not math.isnan(z_rsi) else 0.0,
            "z_stochrsi": round(z_stochrsi, 3) if not math.isnan(z_stochrsi) else 0.0,
            "z_ema_diff": round(z_ema_diff, 3) if not math.isnan(z_ema_diff) else 0.0,
            "z_price_ema": round(z_price_ema, 3) if not math.isnan(z_price_ema) else 0.0,
            "z_atr": round(z_atr, 3) if not math.isnan(z_atr) else 0.0,
            "z_vol": round(z_vol, 3) if not math.isnan(z_vol) else 0.0,
            "z_vix": round(z_vix, 3) if not math.isnan(z_vix) else 0.0,
            "z_bn_rel": round(z_bn_rel, 3) if not math.isnan(z_bn_rel) else 0.0
        }

        


        idx_long = df.index.get_loc(date_obj)
        if idx_long >= 15:
            current_returns = returns_arr[idx_long-4:idx_long+1]
            search_space = returns_arr[:idx_long - 5]
            if len(search_space) >= 5:
                hist_windows = np.lib.stride_tricks.sliding_window_view(search_space, window_shape=5)
                diffs = hist_windows - current_returns
                dists = np.sum(diffs**2, axis=1)
                best_window_idx = np.argmin(dists)
                best_match_idx = best_window_idx + 4 # because window ends at index + 4
                
                analogue_date_obj = df.index[best_match_idx]
                analogue_date_str = analogue_date_obj.strftime("%d %b %Y")
                next_day_ret = returns_arr[best_match_idx + 1]
                next_day_dir = "UP" if next_day_ret > 0 else "DOWN"
                signals["analogue_match"] = f"Similar to {analogue_date_str} (Next day went {next_day_dir})"
            else:
                signals["analogue_match"] = "No match found"
        else:
            signals["analogue_match"] = "Not enough data"


        # MC FII logic
        fii_value = None
        if date_str in mc_fii_dict:
            net_val = mc_fii_dict[date_str]
            fii_value = net_val
            if net_val > 500: fii = "buying"
            elif net_val < -500: fii = "selling"
        elif fii == "neutral":
            # Fallback to tighter gap heuristic
            try:
                if idx_long > 0:
                    prev_close = df["Close"].iloc[idx_long - 1]
                    open_price = row["Open"]
                    gap_pct = ((open_price - prev_close) / prev_close) * 100
                    change_pct = ((close - open_price) / open_price) * 100
                    if gap_pct > 0.15 and change_pct > -0.2: fii = "buying"
                    elif gap_pct < -0.15 and change_pct < 0.2: fii = "selling"
            except:
                pass

        # Legacy VIX block removed as it is handled historically now
        data[date_str] = {
            "fii": fii,
            "fii_value": fii_value,
            "signals": signals
        }
        
    # Walk-forward validation runs for EVERY instrument, each over its own
    # history and selecting its own features. It used to be gated on
    # `symbol_name == "nifty"`, so Bank Nifty and Reliance published with no
    # `_meta` at all -- which the UI read as "no verdict" and therefore left
    # their trade card ungated. A missing verdict is not a passing verdict.
    print("Running walk-forward validation...")
    try:
        data["_meta"], _wfo_diag = run_wfo(df, symbol_name)
    except Exception as e:
        # Never let a walk-forward failure silently drop the block: an absent
        # `_meta` is read downstream as "unvalidated", which is a stronger and
        # less honest claim than "we tried and it failed".
        print("WFO failed:", e)
        data["_meta"] = build_insufficient_meta(
            symbol_name, f"walk-forward failed: {e}"
        )
            
    # Layer B: the slim option snapshot. Attached at the top level rather than
    # per-date because it describes the chain as of the export, not a series of
    # historical states. Absent key means "no chain available" and the UI must
    # show N/A rather than fall back to underlying levels.
    if slim_options_payload is not None:
        data["_options"] = slim_options_payload

    with open(data_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[{symbol_name}] Successfully updated {data_file}")

def main():
    # Process multiple instruments.
    #
    # The second element is the *cache key* used by core.data.save_series(),
    # not a hand-written filename. It used to be hardcoded as "banknifty.csv"
    # while the cache writes "bank_nifty.csv", so Bank Nifty never found its
    # master DB and silently fell back to a 10y yfinance window. Resolving the
    # path through core.data.ohlcv_path() keeps writer and reader in lockstep.
    #
    # "reliance" is not a configured symbol (see config/settings.json
    # -> symbols), so it has no cache series and legitimately uses the fallback.
    #
    # The fourth element is the jugaad-data backfill spec, (kind, symbol), so the
    # recent-date repair runs for every instrument rather than nifty alone.
    # kind is "index" for index_df() and "stock" for stock_df().
    #
    # The fifth is the option-chain spec (see _option_chain_spec): without it
    # every instrument was handed NIFTY's strikes, so Bank Nifty was published
    # Nifty's option levels against a spot ~32k points away.
    #
    # Note the two APIs spell Bank Nifty differently and both spellings are
    # required: index_df() wants "NIFTY BANK", while NSELive.index_option_chain
    # wants "BANKNIFTY" and returns an empty payload for "NIFTY BANK".
    instruments = [
        ("nifty", "nifty", "^NSEI", ("index", "NIFTY 50"), ("index", "NIFTY")),
        ("banknifty", "bank_nifty", "^NSEBANK", ("index", "NIFTY BANK"), ("index", "BANKNIFTY")),
        ("reliance", "reliance", "RELIANCE.NS", ("stock", "RELIANCE"), ("stock", "RELIANCE")),
    ]

    for sym, cache_key, yf_symbol, jugaad_spec, chain_spec in instruments:
        print(f"\n--- Processing {sym.upper()} ---")
        process_symbol(sym, cache_key, yf_symbol, jugaad_spec, chain_spec)

if __name__ == "__main__":
    main()
