"""BTST daily walk-forward contract tests.

The BTST panel is only trustworthy if ``_meta`` never advertises a signal the
OOS evidence does not support. These tests pin that contract:

  1. SIGNAL requires a strictly positive OOS edge and a non-empty feature set.
  2. NO SIGNAL is a first-class result: same edge/trade fields, a populated
     ``reason`` naming the failing gate, and an empty feature list.
  3. Both branches stay internally consistent (win rate only when trades > 0).

The window widths are also asserted, because the 30/30 -> 252/126 change is
what made the OOS figures meaningful in the first place.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_dashboard import build_wfo_meta

VAL_WINDOW = 252
OOS_WINDOW = 126

BASE = dict(
    stability={"chk-rsi": 100.0, "chk-ema20": 87.3},
    val_window=VAL_WINDOW,
    oos_window=OOS_WINDOW,
)


def _meta(**overrides):
    kwargs = dict(
        best_final_edge=0.0181,
        best_final_stats=(72, 0.55, 0.004),
        oos_edge=0.0312,
        oos_win_rate=0.61,
        oos_trades=79,
        ui_boxes=["chk-rsi", "chk-ema20"],
        **BASE,
    )
    kwargs.update(overrides)
    return build_wfo_meta(**kwargs)


class TestSignal:
    def test_positive_oos_edge_emits_signal(self):
        m = _meta()
        assert m["status"] == "SIGNAL"
        assert m["reason"] == "OOS edge > 0"

    def test_signal_advertises_features(self):
        assert _meta()["wfo_optimal_features"] == ["chk-rsi", "chk-ema20"]

    def test_signal_carries_oos_evidence(self):
        m = _meta()
        assert m["oos_edge"] > 0
        assert m["oos_trades"] == 79
        assert m["oos_win_rate"] == 61.0

    def test_signal_reports_validation_window(self):
        m = _meta()
        assert m["val_window"] == VAL_WINDOW
        assert m["oos_window"] == OOS_WINDOW


class TestNoSignal:
    def test_non_positive_oos_edge_blocks_signal(self):
        m = _meta(oos_edge=-0.0078, oos_win_rate=0.468)
        assert m["status"] == "NO SIGNAL"
        assert m["reason"] == "OOS edge <= 0"

    def test_zero_oos_edge_blocks_signal(self):
        m = _meta(oos_edge=0.0, oos_win_rate=0.5)
        assert m["status"] == "NO SIGNAL"
        assert m["reason"] == "OOS edge <= 0"

    def test_negative_validation_edge_blocks_signal(self):
        m = _meta(best_final_edge=-0.004, best_final_stats=(30, 0.4, -0.002))
        assert m["status"] == "NO SIGNAL"
        assert m["reason"] == "validation edge <= 0"

    def test_trade_floor_failure_is_explained(self):
        m = _meta(best_final_edge=-999, best_final_stats=None, oos_edge=0.0,
                  oos_trades=0, oos_win_rate=0.0, ui_boxes=None)
        assert m["status"] == "NO SIGNAL"
        assert "5-trade floor" in m["reason"]

    def test_empty_features_never_yield_signal(self):
        # Positive edges but nothing to select must not render a signal.
        m = _meta(ui_boxes=[])
        assert m["status"] == "NO SIGNAL"
        assert m["wfo_optimal_features"] == []

    def test_no_signal_still_reports_oos_figures(self):
        m = _meta(oos_edge=-0.0078, oos_win_rate=0.468)
        assert m["oos_trades"] == 79
        assert m["oos_edge"] == -0.0078
        assert m["oos_win_rate"] == 46.8
        assert m["val_edge"] == 0.0181

    def test_no_signal_has_no_features(self):
        assert _meta(oos_edge=-0.01)["wfo_optimal_features"] == []

    def test_reason_is_always_present(self):
        for oos in (0.0, -0.01, 0.5):
            assert _meta(oos_edge=oos)["reason"]


class TestCandidateFeatures:
    """The candidate set drives the forecast but never the trade gate.

    Forecast availability must be independent of OOS qualification: the UI
    needs to know which combination the optimizer picked even when that
    combination failed to validate, otherwise a legitimate BTST forecast gets
    suppressed by the very gate meant only to qualify trades.
    """

    def test_no_signal_publishes_candidate(self):
        # val_edge is positive and a combination cleared the floor, so the
        # optimizer did select one -- only the OOS edge rejected it.
        m = _meta(oos_edge=-0.0078, oos_win_rate=0.468)
        assert m["wfo_candidate_features"] == ["chk-rsi", "chk-ema20"]

    def test_candidate_is_never_promoted_on_no_signal(self):
        m = _meta(oos_edge=-0.0078)
        assert m["status"] == "NO SIGNAL"
        assert m["wfo_optimal_features"] == []
        assert m["wfo_candidate_features"] != []

    def test_candidate_present_on_signal(self):
        m = _meta()
        assert m["wfo_candidate_features"] == m["wfo_optimal_features"]

    def test_empty_candidate_when_floor_not_cleared(self):
        m = _meta(best_final_edge=-999, best_final_stats=None, oos_edge=0.0,
                  oos_trades=0, oos_win_rate=0.0, ui_boxes=None)
        assert m["wfo_candidate_features"] == []
        assert m["wfo_optimal_features"] == []

    def test_empty_candidate_is_reported_as_floor_failure(self):
        # Distinguishes "no qualifying candidate" from "field missing", so the
        # UI can label a fallback feature set honestly.
        m = _meta(best_final_edge=-999, best_final_stats=None, oos_edge=0.0,
                  oos_trades=0, oos_win_rate=0.0, ui_boxes=None)
        assert "5-trade floor" in m["reason"]

    def test_candidate_present_but_never_qualifies_trade(self):
        # Positive validation edge and a real candidate, yet still NO SIGNAL:
        # the candidate alone must not promote the status.
        m = _meta(oos_edge=0.0, oos_win_rate=0.5)
        assert m["wfo_candidate_features"] != []
        assert m["status"] == "NO SIGNAL"
        assert m["wfo_optimal_features"] == []


class TestConsistency:
    def test_win_rate_zero_when_no_trades(self):
        m = _meta(oos_trades=0, oos_win_rate=0.0, oos_edge=0.0)
        assert m["oos_trades"] == 0
        assert m["oos_win_rate"] == 0.0

    def test_signal_never_has_empty_trade_count(self):
        m = _meta()
        assert m["status"] == "SIGNAL"
        assert m["oos_trades"] > 0

    def test_both_branches_expose_same_statistical_fields(self):
        keys = {"val_edge", "val_win_rate", "val_trades", "oos_edge",
                "oos_win_rate", "oos_trades", "stability", "val_window",
                "oos_window", "wfo_optimal_features", "wfo_candidate_features",
                "status", "reason"}
        assert set(_meta()) == keys
        assert set(_meta(oos_edge=-0.01)) == keys

    def test_stability_keys_are_mapped_checkbox_ids(self):
        # Unmapped raw feature names (e.g. "z_atr") must never reach the UI.
        for k in _meta()["stability"]:
            assert k.startswith("chk-")
