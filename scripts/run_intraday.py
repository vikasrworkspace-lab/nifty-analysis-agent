"""Intraday analogue pipeline: load archive -> features -> analogues for the
latest bar -> walk-forward backtest gate -> HTML report + console summary.

Usage:
    python scripts/run_intraday.py [--tf 5|15|30|60] [--k 100]
                                   [--lookback 500] [--no-backtest]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import core.data as data
import core.settings as settings_mod
from analysis import intraday, patterns, report
from backtest import intraday as bt_intraday
from core import session


def load_for_tf(settings: dict, minutes: int) -> pd.DataFrame:
    bars = session.archive_bars("nifty", minutes, settings)
    if bars.empty:
        raise SystemExit(f"No intraday archive for {minutes}m. Run scripts/update_data.py --intraday first.")
    return bars


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=int, default=5, choices=[5, 15, 30, 60])
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--lookback", type=int, default=None)
    ap.add_argument("--no-backtest", action="store_true")
    args = ap.parse_args(argv)

    settings = settings_mod.load_settings()
    t0 = time.time()
    k = args.k or settings["intraday"]["k"]
    lookback = args.lookback or settings["intraday"]["standardize_lookback"]
    features = settings["intraday"]["features"]
    weights = settings["intraday"]["weights"]
    target = settings["intraday"]["primary_target"]

    bars = load_for_tf(settings, args.tf)
    daily = data.load_series("nifty", settings)
    if daily.empty:
        raise SystemExit("Daily nifty series missing (run scripts/update_data.py).")

    frame = intraday.build_features(bars, daily, settings)
    valid = intraday.select_valid(frame, features)
    if valid.empty:
        raise SystemExit("No candidate bars with valid features+outcomes yet.")

    z = patterns.zscore_point_in_time(frame, features, lookback)
    target_pos = len(frame) - 1
    sim, dist = patterns.find_analogues(
        frame, z, features, weights, target_pos, k, lookback, session_col="session"
    )
    if not sim:
        raise SystemExit("No analogues found.")

    target_ts = frame.index[target_pos]
    cohort = frame.iloc[sim]
    stats = intraday.cohort_metrics(cohort, settings["intraday"]["horizon_bars"], target)
    stats["raw"] = cohort[target].dropna().to_numpy()
    stats["analogue_count"] = len(cohort)

    bt_res = None
    if not args.no_backtest:
        bt_res = bt_intraday.backtest_bars(frame, settings, features, weights, k=k, lookback=lookback, target=target)

    ctx = {
        "asof": target_ts,
        "target_ts": target_ts,
        "tf": args.tf,
        "k": k,
        "features": features,
        "stats": stats,
        "scenarios": {},
        "analogues": cohort,
        "backtest": bt_res,
        "settings": settings,
        "target": target,
        "images": {},
    }
    ctx["images"] = {
        "session": report.intraday_session_chart(frame, settings),
        "hist": report.intraday_histogram(stats, target),
    }
    html_path, pdf_path = report.write_intraday_report(ctx, settings, include_pdf=False)

    print("=" * 70)
    print(f"INTRADAY ({args.tf}m) - NIFTY 50 at {target_ts:%Y-%m-%d %H:%M}")
    print(f"  analogues: {len(cohort)}  similar bars")
    print(f"  prob rest-up: {stats['direction']['prob_up']*100:.1f}%  "
          f"median rest: {stats[target]['median']:.2f}%  "
          f"P5-P95: {stats[target]['p5']:.2f}% to {stats[target]['p95']:.2f}%")
    print(f"  next-bar median: {stats.get('fwd_ret_1', {}).get('median', float('nan')):.2f}%")
    if bt_res and bt_res["ok"]:
        print("  Backtest (walk-forward):")
        print(f"    bars={bt_res['n_days']}  hit={bt_res['hit_rate']*100:.1f}%  "
              f"always-rest-up={bt_res['always_up_acc']*100:.1f}%  edge={bt_res['edge']:+.2f}pp  "
              f"corr={bt_res['corr']:.3f}  MAE={bt_res['mae']:.2f}%")
    print(f"  report -> {html_path}   (total {time.time()-t0:.1f}s)")
    print("=" * 70)


if __name__ == "__main__":
    main()