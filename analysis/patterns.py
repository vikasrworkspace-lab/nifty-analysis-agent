"""Similar-historical-day detection and next-day outcome analysis.

Everything here is point-in-time:
  * features at day D only use data up to the close of D,
  * analogues for D are drawn strictly from days BEFORE D,
  * z-scoring uses a trailing lookback window (never full-sample stats).
No future information is ever used.
"""
import numpy as np
import pandas as pd

from analysis import technical


def build_features(df: pd.DataFrame, cfg: dict, ma_periods=None) -> pd.DataFrame:
    """Work frame: indicators + feature columns + next-day outcome columns."""
    ind = dict(cfg["indicators"])
    if ma_periods:
        ind["ma_periods"] = list(ma_periods)
    sub_cfg = dict(cfg)
    sub_cfg["indicators"] = ind
    out = technical.add_indicators(df, sub_cfg)

    close, high, low, open_ = out["close"], out["high"], out["low"], out["open"]
    for n in ind["ma_periods"]:
        out[f"dist_sma{n}"] = (close / out[f"sma_{n}"] - 1.0) * 100.0
    out["macd_hist_pct"] = out["macd_hist"] / close * 100.0

    # Calendar / session meta features (point-in-time by construction).
    idx = out.index
    out["dow_monday"] = (idx.dayofweek == 0).astype(int)
    dom = idx.day
    dim = idx.days_in_month
    out["days_to_month_end"] = dim - dom
    out["expiry_zone"] = (out["days_to_month_end"] <= 5).astype(int)

    out["nxt_close"] = close.shift(-1)
    out["nxt_open_gap"] = (open_.shift(-1) / close - 1.0) * 100.0
    out["nxt_ret"] = (close.shift(-1) / close - 1.0) * 100.0
    out["nxt_high_pct"] = (high.shift(-1) / close - 1.0) * 100.0
    out["nxt_low_pct"] = (low.shift(-1) / close - 1.0) * 100.0
    out["nxt_maf"] = out["nxt_high_pct"].clip(lower=0.0)
    out["nxt_mad"] = out["nxt_low_pct"].clip(upper=0.0)
    return out


def zscore_point_in_time(frame: pd.DataFrame, features, lookback: int) -> pd.DataFrame:
    """Point-in-time z-scores: each row is standardized by the trailing window
    ending at that row (rolling mean/std, never full-sample statistics)."""
    minp = min(lookback, 60)
    z = pd.DataFrame(index=frame.index)
    for f in features:
        col = frame[f].astype(float)
        mean = col.rolling(lookback, min_periods=minp).mean()
        std = col.rolling(lookback, min_periods=minp).std(ddof=0)
        z[f] = (col - mean) / std.replace(0, np.nan)
    return z


def find_analogues(
    frame: pd.DataFrame,
    z: pd.DataFrame,
    features,
    weights,
    target_pos: int,
    k: int,
    standardize_lookback: int,
    session_col=None,
):
    """Top-k most similar historical days strictly BEFORE target_pos.

    Similarity = weighted Euclidean distance on point-in-time z-scored features.
    Returns (positions, distances).

    ``session_col`` enables the intraday no-lookahead guard. In a bar-level
    frame every bar is labelled with a forward outcome that runs to its own
    session close (``rest_of_session`` / ``fwd_ret_1``). A candidate from the
    SAME session as the target therefore carries an outcome that extends past
    the target bar -- the target's own future would leak into the analogue
    cohort. Passing the session column drops those candidates, so analogues are
    drawn only from strictly earlier sessions. Daily frames pass None: their
    next-day outcome is already strictly after the target day.
    """
    feats = np.array([z[f].to_numpy(dtype=float) for f in features]).T
    w = np.array([weights.get(f, 1.0) for f in features], dtype=float)
    target = feats[target_pos]
    mask = ~np.isnan(feats).any(axis=1)
    cand_idx = np.flatnonzero(mask)
    cand_idx = cand_idx[cand_idx < target_pos]
    if session_col is not None and session_col in frame.columns:
        sessions = frame[session_col].to_numpy()
        cand_idx = cand_idx[sessions[cand_idx] != sessions[target_pos]]
    if cand_idx.size == 0:
        return [], []
    cand = feats[cand_idx]
    diff = (cand - target) * w
    dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    order = np.argsort(dist)[:k]
    return cand_idx[order].tolist(), dist[order].tolist()


OUTCOME_METRICS = ["nxt_open_gap", "nxt_ret", "nxt_high_pct", "nxt_low_pct", "nxt_mad", "nxt_maf"]


def outcome_stats(sub: pd.DataFrame) -> dict:
    """Aggregate distribution of next-day outcomes for a cohort of analogues."""
    stats = {}
    for col in OUTCOME_METRICS:
        s = sub[col].dropna()
        if s.empty:
            stats[col] = {"count": 0}
            continue
        stats[col] = {
            "count": int(s.size),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std(ddof=0)),
            "p5": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
        }
    r = sub["nxt_ret"].dropna()
    if r.empty:
        stats["direction"] = {"up": 0, "down": 0, "flat": 0, "prob_up": float("nan")}
    else:
        stats["direction"] = {
            "up": int((r > 0).sum()),
            "down": int((r < 0).sum()),
            "flat": int((r == 0).sum()),
            "prob_up": float((r > 0).mean()),
        }
    stats["dates"] = [d.strftime("%Y-%m-%d") for d in sub.index]
    return stats


def _mad(s: pd.Series) -> float:
    med = s.median()
    return float((s - med).abs().median())


def divergence(frame: pd.DataFrame, sub: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """Current day value vs analogue median/robust spread for each feature (raw scale)."""
    rows = []
    for f in feature_cols:
        today = float(frame[f].iloc[-1]) if pd.notna(frame[f].iloc[-1]) else float("nan")
        med = float(sub[f].median()) if sub[f].notna().any() else float("nan")
        mad = _mad(sub[f]) if sub[f].notna().any() else float("nan")
        rows.append({"feature": f, "today": round(today, 3), "analogue_median": round(med, 3), "analogue_mad": round(mad, 3)})
    return pd.DataFrame(rows)


def scenarios(stats: dict, sentiment_cfg: dict) -> dict:
    """Derive bullish / neutral / bearish scenario bands strictly from the
    analogue distribution. Interpretation (news, internals) is added later by
    the orchestrating agent, not here."""
    nxt = stats.get("nxt_ret", {})
    if not nxt or not nxt.get("count"):
        return {}
    prob_up = stats["direction"]["prob_up"]
    bull_t = sentiment_cfg["bull_threshold"]
    bear_t = sentiment_cfg["bear_threshold"]
    if prob_up >= bull_t:
        bias = "Bullish"
    elif prob_up <= bear_t:
        bias = "Bearish"
    else:
        bias = "Neutral"

    r = stats["dates"]  # noqa: F841 - kept for symmetry; not needed below
    return {
        "bias": bias,
        "prob_up": round(prob_up, 3),
        "prob_down": round(1.0 - prob_up, 3),
        "central_case": nxt["median"],
        "median_band": [nxt["p25"], nxt["p75"]],
        "full_range": [nxt["p5"], nxt["p95"]],
        "bullish_scenario": {"prob": round(max(prob_up, 1 - prob_up), 3), "target_median_ret": nxt["p75"]},
        "bearish_scenario": {"prob": round(min(prob_up, 1 - prob_up), 3), "target_median_ret": nxt["p25"]},
    }


def auto_select_features(
    frame: pd.DataFrame,
    z: pd.DataFrame,
    available_features: list,
    weights: dict,
    target_pos: int,
    k: int,
    standardize_lookback: int,
    backtest_days: int = 90,
    top_n: int = 5,
) -> tuple:
    """
    Evaluates individual features using robust quantitative scoring:
    1. Regime Tagging (ADX) to filter candidate features.
    2. Out-of-Sample Validation (Train/Val split).
    3. Information Coefficient (Edge over Base Rate).
    4. Expectancy (Risk-Adjusted Return).
    Returns the top N feature names and their validation edge scores.
    """
    current_adx = float(frame["adx_14"].iloc[target_pos]) if "adx_14" in frame.columns else 22.0
    
    trend_features = {"dist_sma20", "dist_sma50", "dist_sma200", "macd_hist_pct", "ret_20d", "ret_5d"}
    range_features = {"rsi_14", "range_pct", "atr_pct", "vol_20d", "gap_pct", "ret_1d"}
    
    allowed_features = available_features.copy()
    if current_adx > 25:
        allowed_features = [f for f in allowed_features if f in trend_features or f not in range_features]
    elif current_adx < 20:
        allowed_features = [f for f in allowed_features if f in range_features or f not in trend_features]

    train_start = max(1, target_pos - backtest_days)
    val_start = target_pos - (backtest_days // 3)
    
    def evaluate_window(f, start, end):
        if end <= start:
            return 0.0, 0.0
        correct = 0
        valid = 0
        strat_returns = []
        for pos in range(start, end):
            idx, _ = find_analogues(frame, z, [f], weights, pos, k, standardize_lookback)
            if not idx:
                continue
            
            prob_up = (frame["nxt_ret"].iloc[idx] > 0).mean()
            pred_up = prob_up > 0.5
            actual_ret = frame["nxt_ret"].iloc[pos]
            actual_up = actual_ret > 0
            
            if pred_up == actual_up:
                correct += 1
            valid += 1
            
            trade_ret = actual_ret if pred_up else -actual_ret
            strat_returns.append(trade_ret)
            
        acc = correct / valid if valid > 0 else 0.0
        base_rate = (frame["nxt_ret"].iloc[start:end] > 0).mean()
        edge = acc - base_rate
        exp = float(np.mean(strat_returns)) if strat_returns else 0.0
        return edge, exp

    scores = {}
    for f in allowed_features:
        t_edge, t_exp = evaluate_window(f, train_start, val_start)
        v_edge, v_exp = evaluate_window(f, val_start, target_pos)
        
        # Only select if it showed positive edge and expectancy in BOTH windows
        if t_edge > 0.0 and t_exp > 0.0 and v_edge > 0.0 and v_exp > 0.0:
            scores[f] = (t_edge + v_edge) / 2.0
            
    # Fallback if strict validation eliminates all features:
    if not scores:
        for f in allowed_features:
            _, t_exp = evaluate_window(f, train_start, target_pos)
            scores[f] = t_exp
            
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_features = [f for f, score in ranked[:top_n]]
    return best_features, scores