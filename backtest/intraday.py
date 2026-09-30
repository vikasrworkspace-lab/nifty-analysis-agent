"""Walk-forward validation for the intraday analogue engine.

Same no-lookahead contract as the daily backtest: at each test bar p, analogues
are drawn strictly from bars before p, predictions use only information
available at p, and outcomes are the same-session forward returns.

Reuses patterns.zscore_point_in_time / find_analogues on the bar frame.
"""
import time

import numpy as np
import pandas as pd

from analysis import patterns


def backtest_bars(frame: pd.DataFrame, settings: dict, features, weights,
                  k: int = None, lookback: int = None, target: str = "rest_of_session") -> dict:
    k = k or settings["intraday"]["k"]
    lookback = lookback or settings["intraday"]["standardize_lookback"]
    t0 = time.time()

    z = patterns.zscore_point_in_time(frame, features, lookback)
    target_col = frame[target]

    anal_pos = []
    pos = np.flatnonzero(frame[target].notna().to_numpy())
    rows = []

    for p in pos:
        sim, _ = patterns.find_analogues(
            frame, z, features, weights, int(p), k, lookback, session_col="session"
        )
        if not sim:
            continue
        cohort = target_col.iloc[sim].dropna()
        pred = float(cohort.median()) if cohort.size else np.nan
        actual = float(target_col.iloc[p])
        if not np.isnan(pred - actual):
            rows.append((frame.index[p], pred, actual))
            anal_pos.append(sim)

    if not rows:
        return {"ok": False, "n_days": 0}

    rec = pd.DataFrame(rows, columns=["ts", "pred", "actual"]).set_index("ts")
    sgn = np.sign(rec["pred"]) == np.sign(rec["actual"])
    hit_rate = float(sgn.mean())
    baseline = float((rec["actual"] > 0).mean())
    edge = (hit_rate - baseline) * 100.0
    corr = float(rec["pred"].corr(rec["actual"])) if len(rec) > 2 else float("nan")
    mae = float((rec["pred"] - rec["actual"]).abs().mean())
    mean_actual = float(rec["actual"].mean())
    mean_pred = float(rec["pred"].mean())

    return {
        "ok": True,
        "n_days": len(rec),
        "hit_rate": hit_rate,
        "always_up_acc": float((rec["actual"] >= 0).mean()),
        "edge": edge,
        "corr": corr,
        "mae": mae,
        "baseline_mean": mean_actual,
        "mean_pred": mean_pred,
        "records": rec,
        "seconds": round(time.time() - t0, 1),
        "analogue_positions": anal_pos,
    }