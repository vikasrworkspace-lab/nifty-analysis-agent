"""Per-symbol walk-forward independence and complexity-penalty units.

Two properties are pinned here:

  1. Each instrument is validated over its OWN history and selects its OWN
     features. Nothing may be inherited from NIFTY. The walk-forward used to be
     gated on ``symbol_name == "nifty"``, which left Bank Nifty and Reliance
     published with no ``_meta`` -- read downstream as "no verdict", i.e. the
     trade card ungated.
  2. The complexity penalty is subtracted in the same unit as the edge. The
     original code fed ``pct_change()`` fractions into a penalty documented as
     "0.01%", making it ~100x too large. With edges in percent, the penalty for
     an n-feature combination must be exactly ``n * 0.01``.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import export_dashboard as ed  # noqa: E402

N_ROWS = 700
FEATURES = ed.WFO_FEATURES
PERSISTENCE = 0.9


def _frame(slope=0.8, seed=7):
    """Synthetic daily frame with a known feature -> next-day-return link.

    The features are AR(1) with high persistence, because a walk-forward can
    only forecast the next session if today's state carries into tomorrow's.
    (With iid features nothing is predictable and every combination ties at
    chance -- which would make an "independence" test pass vacuously.)

    ``slope`` > 0 makes z_rsi genuinely predictive, so the optimizer finds
    trades; ``slope`` < 0 makes it reliably wrong, so no combination clears the
    validation gate. ``Return`` is in PERCENT, matching the corrected scale.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=N_ROWS)
    innovation = np.sqrt(1.0 - PERSISTENCE ** 2)
    series = {}
    for f in FEATURES:
        x = np.zeros(N_ROWS)
        for t in range(1, N_ROWS):
            x[t] = PERSISTENCE * x[t - 1] + innovation * rng.normal()
        series[f] = x
    df = pd.DataFrame(series, index=dates)
    df["Return"] = slope * df["z_rsi"] + rng.normal(0, 0.4, N_ROWS)
    return df


def _noise_frame(seed=11):
    """Same persistent features, but returns unrelated to them.

    This is the control: a symbol with no learnable relationship must be
    refused by the validation gate rather than validated on chance.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=N_ROWS)
    series = {}
    for f in FEATURES:
        x = np.zeros(N_ROWS)
        for t in range(1, N_ROWS):
            x[t] = PERSISTENCE * x[t - 1] + np.sqrt(1.0 - PERSISTENCE ** 2) * rng.normal()
        series[f] = x
    df = pd.DataFrame(series, index=dates)
    df["Return"] = rng.normal(0, 0.8, N_ROWS)
    return df


class TestInsufficientData:
    def test_short_history_is_refused_not_faked(self):
        df = _frame().iloc[:200]  # below 252 + 126
        meta, diag = ed.run_wfo(df, "reliance", verbose=False)
        assert meta["status"] == "INSUFFICIENT_DATA"
        assert diag["ok"] is False
        assert "252" in meta["reason"]

    def test_missing_feature_is_refused(self):
        df = _frame().drop(columns=["z_vol"])
        meta, diag = ed.run_wfo(df, "banknifty", verbose=False)
        assert meta["status"] == "INSUFFICIENT_DATA"
        assert "z_vol" in meta["reason"]
        assert diag["reason"] == "missing_features"

    def test_refusal_publishes_no_edge_evidence(self):
        meta, _ = ed.run_wfo(_frame().iloc[:100], "reliance", verbose=False)
        assert meta["oos_edge"] == 0
        assert meta["oos_trades"] == 0
        assert meta["stability"] == {}
        assert meta["wfo_optimal_features"] == []
        assert meta["oos_predictions"] == []


class TestPenaltyUnits:
    def test_penalty_is_n_features_times_0_01_percent(self):
        meta, diag = ed.run_wfo(_frame(slope=0.8), "nifty", verbose=False)
        assert diag["return_scale"] == 100.0
        assert diag["complexity_penalty_per_feature"] == 0.01
        raw, penalised = diag["val_edge_raw"], diag["val_edge_penalised"]
        assert raw is not None and penalised is not None
        n_selected = len(diag["selected_features"])
        assert 2 <= n_selected <= 5
        # The whole point of the fix: the gap is a few basis points, not a few
        # percentage points.
        assert raw - penalised == pytest.approx(n_selected * 0.01)

    def test_penalty_measured_in_percent_not_fraction(self):
        # In the old unit system a 3-feature penalty was 0.03 on an edge scale
        # where a typical edge was ~0.04 -- i.e. it dominated the signal. In
        # percent the same 0.03 is negligible against an edge of ~4.
        _, diag = ed.run_wfo(_frame(slope=0.8), "nifty", verbose=False)
        assert abs(diag["val_edge_penalised"]) > 1.0


class TestPerSymbolIndependence:
    def test_predictable_symbol_validates(self):
        meta, diag = ed.run_wfo(_frame(slope=0.8, seed=7), "nifty",
                                verbose=False)
        assert meta["oos_trades"] > 0
        assert meta["oos_edge"] > 0
        assert meta["status"] == "SIGNAL"

    def test_unpredictable_symbol_is_refused(self):
        # The control: no relationship between features and next-day returns.
        # The gate must reject it rather than validate it on chance.
        meta, _ = ed.run_wfo(_noise_frame(), "banknifty", verbose=False)
        assert meta["status"] == "NO SIGNAL"
        assert meta["oos_edge"] <= 0
        assert meta["wfo_optimal_features"] == []
        # Hit rate near chance, not a confident-but-wrong forecast.
        assert 35.0 <= meta["oos_win_rate"] <= 65.0

    def test_symbols_are_evaluated_independently(self):
        good, good_diag = ed.run_wfo(_frame(slope=0.8, seed=7), "nifty",
                                     verbose=False)
        bad, bad_diag = ed.run_wfo(_noise_frame(), "banknifty", verbose=False)
        # Same history length, same candidate pool, different verdicts -- so
        # nothing can have been inherited from NIFTY.
        assert good_diag["n_days"] == bad_diag["n_days"]
        assert good_diag["features_evaluated"] == bad_diag["features_evaluated"]
        assert good["status"] != bad["status"]
        assert good["oos_edge"] > bad["oos_edge"]
        assert good["symbol"] == "nifty"
        assert bad["symbol"] == "banknifty"

    def test_symbol_is_stamped_into_meta(self):
        meta, diag = ed.run_wfo(_frame(), "banknifty", verbose=False)
        assert meta["symbol"] == "banknifty"
        assert diag["symbol"] == "banknifty"

    def test_candidate_pool_is_shared_but_selection_is_not(self):
        # The candidate feature pool is the UI checkbox set and is identical by
        # design; what must differ is each symbol's own OOS evidence.
        a, da = ed.run_wfo(_frame(slope=0.8, seed=1), "nifty", verbose=False)
        b, db = ed.run_wfo(_frame(slope=0.8, seed=2), "banknifty",
                           verbose=False)
        assert da["features_evaluated"] == db["features_evaluated"] == FEATURES
        assert a["wfo_candidate_features"] != [] or b["wfo_candidate_features"] != []


class TestOOSRecordIntegrity:
    def test_records_lie_inside_the_oos_period_only(self):
        meta, _ = ed.run_wfo(_frame(slope=0.8), "nifty", verbose=False)
        start, end = meta["oos_period"]["start"], meta["oos_period"]["end"]
        assert meta["val_period"]["end"] < start
        for r in meta["oos_predictions"]:
            assert start <= r["date"] <= end

    def test_no_model_days_are_not_counted_as_trades(self):
        meta, diag = ed.run_wfo(_frame(slope=0.8), "nifty", verbose=False)
        directional = [r for r in meta["oos_predictions"]
                       if r["direction"] in ("LONG", "SHORT")]
        assert diag["oos_no_model_days"] >= 0
        assert len(directional) == meta["oos_trades"]

    def test_correctness_matches_direction(self):
        meta, _ = ed.run_wfo(_frame(slope=0.8), "nifty", verbose=False)
        for r in meta["oos_predictions"]:
            if r["direction"] == "LONG":
                assert r["correct"] == (r["actual_return"] > 0)
            elif r["direction"] == "SHORT":
                assert r["correct"] == (r["actual_return"] < 0)
            else:
                assert r["correct"] is None

    def test_flat_direction_emits_no_trade(self):
        meta, _ = ed.run_wfo(_frame(slope=0.8), "nifty", verbose=False)
        for r in meta["oos_predictions"]:
            if r["direction"] == "FLAT":
                assert ed.WFO_SHORT_BAND < r["prob_up"] < ed.WFO_LONG_BAND

    def test_trade_floor_is_enforced(self):
        # Fewer than 5 directional trades can never be called validated.
        meta, _ = ed.run_wfo(_frame(slope=-0.8), "nifty", verbose=False)
        if meta["oos_trades"] < ed.WFO_MIN_TRADES:
            assert meta["status"] != "SIGNAL"
