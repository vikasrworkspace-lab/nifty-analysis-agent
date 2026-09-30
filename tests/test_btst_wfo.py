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

# Every branch -- SIGNAL, NO SIGNAL and INSUFFICIENT_DATA -- must publish the
# same field set, so the UI never has to special-case a missing key.
META_KEYS = {
    "symbol", "val_edge", "val_win_rate", "val_trades", "oos_edge",
    "oos_win_rate", "oos_trades", "stability", "val_window", "oos_window",
    "val_period", "oos_period", "oos_predictions", "wfo_optimal_features",
    "wfo_candidate_features", "status", "reason",
}


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
        assert set(_meta()) == META_KEYS
        assert set(_meta(oos_edge=-0.01)) == META_KEYS

    def test_insufficient_data_exposes_the_same_fields(self):
        # A symbol that cannot be tested must still publish a full, uniformly
        # shaped block so the UI can render it without special-casing a missing
        # key -- and must not imply any edge was measured.
        from scripts.export_dashboard import build_insufficient_meta

        m = build_insufficient_meta("reliance", "not enough history")
        assert set(m) == META_KEYS
        assert m["status"] == "INSUFFICIENT_DATA"
        assert m["oos_edge"] == 0
        assert m["oos_trades"] == 0
        assert m["wfo_optimal_features"] == []
        assert m["oos_predictions"] == []

    def test_stability_keys_are_mapped_checkbox_ids(self):
        # Unmapped raw feature names (e.g. "z_atr") must never reach the UI.
        for k in _meta()["stability"]:
            assert k.startswith("chk-")


class TestPerDatePredictionsAreSeparate:
    """``status`` is a whole-evaluation verdict; ``oos_predictions`` is the
    per-date record. The UI must be able to answer "what did the model predict
    on this date?" without ever falling back to the global status.
    """

    RECORD = {
        "date": "2026-09-15",
        "direction": "LONG",
        "prob_up": 62.4,
        "expected_return": 0.31,
        "actual_return": -0.12,
        "correct": False,
        "features": ["z_rsi", "z_vol"],
        "designation": "OOS",
    }

    def test_records_survive_on_both_branches(self):
        assert _meta(oos_predictions=[self.RECORD])["oos_predictions"] == [self.RECORD]
        assert _meta(oos_edge=-0.01,
                     oos_predictions=[self.RECORD])["oos_predictions"] == [self.RECORD]

    def test_absent_records_default_to_empty_not_missing(self):
        # "no per-date prediction" must be representable, so the UI can say so
        # rather than substituting the global status.
        assert _meta()["oos_predictions"] == []

    def test_record_carries_every_field_the_ui_needs(self):
        r = _meta(oos_predictions=[self.RECORD])["oos_predictions"][0]
        assert set(r) == {"date", "direction", "prob_up", "expected_return",
                          "actual_return", "correct", "features",
                          "designation"}
        assert r["designation"] == "OOS"

    def test_no_model_day_is_distinguishable_from_no_prediction(self):
        # A day the model sat out is a real observation and must be recorded;
        # a day outside the evaluated range is simply absent.
        no_model = {"date": "2026-09-16", "direction": "NO_MODEL",
                    "prob_up": None, "expected_return": None,
                    "actual_return": 0.04, "correct": None, "features": [],
                    "designation": "OOS", "note": "no combination cleared the gate"}
        m = _meta(oos_predictions=[self.RECORD, no_model])
        assert len(m["oos_predictions"]) == 2
        assert m["oos_predictions"][1]["direction"] == "NO_MODEL"
        assert m["oos_predictions"][1]["correct"] is None

    def test_period_bounds_are_published(self):
        m = _meta(val_period={"start": "2025-01-01", "end": "2025-12-31"},
                  oos_period={"start": "2026-01-01", "end": "2026-06-30"})
        assert m["val_period"]["end"] < m["oos_period"]["start"]
        assert m["oos_period"] is not None

    def test_symbol_is_recorded(self):
        assert _meta(symbol="banknifty")["symbol"] == "banknifty"


class TestPublishedPayloads:
    """Validate the committed dashboard JSON, not just the builder.

    ``renderPerDateOOS()`` calls ``.toFixed()`` on the numeric fields and
    ``.find(r => r.date === ...)`` on the record list, so a single ``null`` or
    a duplicated/unsorted date would break the panel for real. These run
    against the tracked artifacts that GitHub Pages actually serves.
    """

    FILES = {
        "nifty": "dashboard_data.json",
        "banknifty": "dashboard_data_banknifty.json",
        "reliance": "dashboard_data_reliance.json",
    }

    def _load(self, filename):
        import json
        root = Path(__file__).resolve().parents[1]
        return json.loads((root / filename).read_text(encoding="utf-8"))

    def test_every_symbol_publishes_meta(self):
        # Previously only NIFTY had a _meta, so the sibling symbols silently
        # ran ungated. Each must now carry its own independent verdict.
        for symbol, filename in self.FILES.items():
            meta = self._load(filename)["_meta"]
            assert meta["symbol"] == symbol, filename
            assert meta["status"] in {"SIGNAL", "NO SIGNAL", "INSUFFICIENT_DATA"}

    def test_status_matches_the_oos_edge(self):
        for filename in self.FILES.values():
            meta = self._load(filename)["_meta"]
            if meta["status"] == "SIGNAL":
                assert meta["oos_edge"] > 0, filename
                assert meta["wfo_optimal_features"], filename
            elif meta["status"] == "NO SIGNAL":
                assert meta["oos_edge"] <= 0, filename

    def test_per_date_records_are_ui_safe(self):
        for filename in self.FILES.values():
            payload = self._load(filename)
            dates = set(payload) - {"_meta"}
            records = payload["_meta"]["oos_predictions"]
            assert records, filename
            for r in records:
                seated_out = r["direction"] in {"FLAT", "NO_MODEL"}
                # A day the model sat out carries no forecast, so its numeric
                # fields are legitimately null -- the UI must guard them.
                for k in ("prob_up", "expected_return", "actual_return"):
                    if seated_out and r[k] is None:
                        continue
                    assert isinstance(r[k], (int, float)), (filename, r["date"], k)
                assert r["correct"] is None or isinstance(r["correct"], bool)
                # A traded day must name the features it used.
                if not seated_out:
                    assert r["features"], (filename, r["date"])
                # The date must be selectable in the UI.
                assert r["date"] in dates, (filename, r["date"])

    def test_days_the_model_sat_out_are_unscored(self):
        # FLAT (no-trade band) and NO_MODEL (gate rejected everything) are both
        # non-events. Scoring either as a loss would understate the model.
        for filename in self.FILES.values():
            for r in self._load(filename)["_meta"]["oos_predictions"]:
                seated_out = r["direction"] in {"FLAT", "NO_MODEL"}
                assert seated_out == (r["correct"] is None), \
                    (filename, r["date"], r["direction"], r["correct"])

    def test_records_are_unique_and_chronological(self):
        # renderPerDateOOS uses .find(), so duplicates would silently shadow.
        for filename in self.FILES.values():
            meta = self._load(filename)["_meta"]
            ds = [r["date"] for r in meta["oos_predictions"]]
            assert ds == sorted(ds), filename
            assert len(ds) == len(set(ds)), filename
            # And the record span must match the advertised OOS window.
            assert ds[0] == meta["oos_period"]["start"], filename
            assert ds[-1] == meta["oos_period"]["end"], filename

    def test_oos_predictions_never_leak_past_the_written_dates(self):
        # Guards the AGENTS.md hazard: a stale local cache could let a
        # prediction reference a session the payload does not contain.
        for filename in self.FILES.values():
            payload = self._load(filename)
            dates = set(payload) - {"_meta"}
            meta = payload["_meta"]
            assert meta["oos_period"]["end"] <= max(dates), filename
            assert meta["val_period"]["end"] < meta["oos_period"]["start"], filename

    def test_recorded_returns_reconcile_with_the_published_prices(self):
        """``actual_return`` must be the real next-session return in the file.

        This is what makes it safe to publish an ``_meta`` alongside the
        payload: the numbers are not from some other vintage of the data. A
        regenerated-but-diverged cache would show up here immediately.
        """
        for filename in self.FILES.values():
            payload = self._load(filename)
            dates = sorted(k for k in payload if k != "_meta")
            pos = {d: i for i, d in enumerate(dates)}
            checked = 0
            for r in payload["_meta"]["oos_predictions"]:
                if r["direction"] in {"FLAT", "NO_MODEL"} or r["actual_return"] is None:
                    continue
                i = pos[r["date"]]
                assert i + 1 < len(dates), (filename, r["date"], "no next session")
                c0 = payload[dates[i]]["signals"]["close"]
                c1 = payload[dates[i + 1]]["signals"]["close"]
                expected = (c1 / c0 - 1) * 100
                assert abs(expected - r["actual_return"]) < 0.02, \
                    (filename, r["date"], expected, r["actual_return"])
                checked += 1
            assert checked > 0, filename
