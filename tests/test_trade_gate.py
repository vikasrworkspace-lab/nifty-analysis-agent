"""BTST forecast / trade-qualification separation.

The BTST panel must never show a blank forecast just because the WFO
out-of-sample gate rejected the trade, and must never show an actionable
trade setup when that gate rejected it.

These are static assertions on ``index.html``. They verify the gate exists,
is scoped correctly, and is wired to the right elements.

TESTING GAP -- read this before trusting these tests:
Cases A / B / C are NOT live DOM-tested in this repo. There is no jsdom
dependency (adding one was declined) and the currently published
``dashboard_data.json`` predates the ``wfo_candidate_features`` field, so
the Case B forecast path cannot be exercised end to end. These tests cover
structure and the metadata contract (see tests/test_btst_wfo.py); the
behavioural confirmation happens by hand once the data is refreshed.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INDEX = (ROOT / "index.html").read_text(encoding="utf-8")


def _body(func_name):
    """Return the source of a function body by brace matching."""
    m = re.search(r"function\s+%s\s*\(" % re.escape(func_name), INDEX)
    assert m, "%s not found in index.html" % func_name
    start = INDEX.index("{", m.end() - 1)
    depth, i = 0, start
    while i < len(INDEX):
        if INDEX[i] == "{":
            depth += 1
        elif INDEX[i] == "}":
            depth -= 1
            if depth == 0:
                return INDEX[start:i + 1]
        i += 1
    raise AssertionError("unbalanced braces in %s" % func_name)


def _case_c_block():
    """The `if (!topK) { ... }` early-return block inside renderPrediction.

    Sliced by brace match rather than by the next `return;`, because the
    function has earlier `return;` statements that would truncate the slice.
    """
    body = _body("renderPrediction")
    marker = body.index("if (!topK)")
    start = body.index("{", marker)
    depth, i = 0, start
    while i < len(body):
        if body[i] == "{":
            depth += 1
        elif body[i] == "}":
            depth -= 1
            if depth == 0:
                return body[start:i + 1]
        i += 1
    raise AssertionError("unbalanced braces in the !topK block")


class TestTradeGateExists:
    def test_helper_defined(self):
        assert "function isTradeQualified()" in INDEX

    def test_gate_scoped_to_daily(self):
        # Intraday must keep its existing ungated behaviour.
        body = _body("isTradeQualified")
        assert "currentTimeframe !== 'daily'" in body
        assert re.search(r"currentTimeframe\s*!==\s*'daily'\)\s*return true", body)

    def test_gate_reads_wfo_status_signal(self):
        body = _body("isTradeQualified")
        assert '_meta' in body
        assert 'status === "SIGNAL"' in body

    def test_missing_meta_is_not_treated_as_rejection(self):
        # The per-symbol sibling payloads (banknifty, reliance) carry no _meta.
        # Absence of a WFO verdict must not strip their Trade Engine.
        body = _body("isTradeQualified")
        assert re.search(r"if\s*\(\s*!meta\s*\|\|\s*meta\.status === undefined", body)
        assert re.search(r"meta\.status === null\s*\)\s*return true", body)

    def test_gate_used_in_render_path(self):
        assert "isTradeQualified()" in _body("renderPrediction")


class TestForecastNotGated:
    """The OOS gate must not decide whether a forecast is computed."""

    def test_no_signal_branch_keeps_features_checked(self):
        body = _body("runAutoSelect")
        branch = body[body.index('"NO SIGNAL"'):]
        branch = branch[:branch.index("renderPrediction()")]
        assert "cb.checked = false" in branch, "NO SIGNAL branch should not blanket-uncheck"
        assert "cb.checked = true" in branch, "NO SIGNAL branch must still select a forecast feature set"

    def test_forecast_prefers_wfo_candidate(self):
        body = _body("forecastFeatureIds")
        assert "wfo_candidate_features" in body

    def test_fallback_only_when_no_candidate(self):
        body = _body("forecastFeatureIds")
        assert "usedFallback: true" in body
        # Fallback must be guarded by an emptiness check, not unconditional.
        assert re.search(r"if\s*\(ids\.length\s*>\s*0\)", body)
        assert re.search(r"usedFallback:\s*false", body)

    def test_candidate_is_distinct_from_optimal(self):
        # The UI must read the candidate set, not the validated one, when the
        # gate rejected the trade.
        body = _body("forecastFeatureIds")
        assert "wfo_optimal_features" not in body


class TestTradeEngineGating:
    def test_actionable_wrapper_exists(self):
        assert 'id="te-actionable"' in INDEX
        assert 'id="te-actionable-rr"' in INDEX

    def test_trade_status_element_exists(self):
        assert 'id="te-trade-status"' in INDEX
        assert "TRADE STATUS: NO VALIDATED SIGNAL" in INDEX

    def test_unqualified_hides_actionable_levels(self):
        body = _body("renderPrediction")
        assert "actionableEl.classList.add('hidden')" in body
        assert "actionableRrEl.classList.add('hidden')" in body

    def test_qualified_shows_actionable_levels(self):
        body = _body("renderPrediction")
        assert "actionableEl.classList.remove('hidden')" in body
        assert "actionableRrEl.classList.remove('hidden')" in body

    def test_unqualified_surfaces_oos_reason(self):
        body = _body("renderPrediction")
        assert "oosReasonEl" in body
        assert "meta.reason" in body
        assert "meta.oos_edge" in body

    def test_oos_reason_never_renders_a_bare_placeholder(self):
        # A legacy payload may carry oos_edge but not oos_trades. That must
        # degrade to prose, not to a lone '?' that reads like a broken value.
        # Strip // comments first so prose about the placeholder cannot
        # satisfy (or trip) the assertion.
        body = re.sub(r"//[^\n]*", "", _body("renderPrediction"))
        assert "ternary ? meta.oos_trades" not in body
        assert "'?'" not in body
        assert '"?"' not in body
        assert "trade counts not recorded" in body
        # The counts, when present, are still rendered.
        assert "meta.oos_trades" in body
        assert "meta.oos_win_rate" in body

    def test_insufficient_data_is_labelled_not_silently_unqualified(self):
        # A symbol the walk-forward could not test must say so. Without an
        # explicit branch it falls through to the SIGNAL path and renders a
        # blank panel, which reads as "no data" rather than "never validated".
        body = re.sub(r"//[^\n]*", "", _body("runAutoSelect"))
        assert "INSUFFICIENT_DATA" in body
        assert "UNVALIDATED" in body

    def test_insufficient_data_never_qualifies_a_trade(self):
        body = re.sub(r"//[^\n]*", "", _body("isTradeQualified"))
        # Only an explicit SIGNAL qualifies. INSUFFICIENT_DATA falls through
        # to False, which is the honest outcome: nothing was validated.
        assert 'meta.status === "SIGNAL"' in body
        assert "INSUFFICIENT_DATA" not in body

    def test_hidden_fields_are_the_actionable_ones(self):
        # Entry / Stop / Targets / R:R must live inside a gated wrapper.
        for field in ("te-entry", "te-stop", "te-t1", "te-t2", "te-rr"):
            idx = INDEX.index('id="%s"' % field)
            cands = [INDEX.rfind('id="te-actionable"', 0, idx),
                     INDEX.rfind('id="te-actionable-rr"', 0, idx)]
            wrapper = max(cands)
            assert wrapper != -1, "%s has no enclosing gated wrapper" % field
            closer = INDEX.index("</div>", idx)
            assert wrapper < idx < closer, "%s is not inside a gated wrapper" % field


class TestPerDateVsGlobalSeparation:
    """The symbol-level verdict must never stand in for a date's own result."""

    def test_per_date_panel_exists(self):
        assert 'id="oos-per-date"' in INDEX

    def test_per_date_renderer_reads_only_the_matching_record(self):
        body = re.sub(r"//[^\n]*", "", _body("renderPerDateOOS"))
        # The lookup must be an exact match on the selected date.
        assert "meta.oos_predictions.find(r => r.date === dateStr)" in body
        # And it must not reach for the global verdict to fill the gap.
        assert "meta.status" not in body
        assert "meta.oos_edge" not in body

    def test_missing_date_uses_the_exact_fallback_text(self):
        body = re.sub(r"//[^\n]*", "", _body("renderPerDateOOS"))
        assert "No per-date OOS prediction is available for this date." in body

    def test_absent_record_array_is_handled(self):
        body = re.sub(r"//[^\n]*", "", _body("renderPerDateOOS"))
        # A legacy payload with no oos_predictions must degrade to the same
        # honest message rather than throwing.
        assert "Array.isArray(meta.oos_predictions)" in body

    def test_per_date_panel_renders_on_every_prediction_pass(self):
        body = re.sub(r"//[^\n]*", "", _body("renderPrediction"))
        assert "renderPerDateOOS(dateStr)" in body
        # It must be called before the top-K gate, so a date with a stored OOS
        # record still renders it even when no analogue match is found.
        assert body.index("renderPerDateOOS(dateStr)") < body.index("calculateTopK(dateStr)")

    def test_global_badges_are_labelled_model_level(self):
        # The Auto-Select badge is a whole-window verdict. Label it so it
        # cannot be read as the call for the selected date.
        assert INDEX.count("MODEL-LEVEL:") >= 3
        for label in ("MODEL-LEVEL: WALK-FORWARD OPTIMIZED",
                      "MODEL-LEVEL: NO RELIABLE OOS SIGNAL DETECTED",
                      "MODEL-LEVEL: UNVALIDATED — INSUFFICIENT DATA"):
            assert label in INDEX

    def test_flat_days_are_not_scored(self):
        # A FLAT day is not a trade, so `correct` is null and must not be
        # rendered as a win or a loss.
        body = re.sub(r"//[^\n]*", "", _body("renderPerDateOOS"))
        assert "rec.direction === 'FLAT'" in body
        assert "rec.correct ? 'Correct' : 'Incorrect'" in body

    def test_no_model_days_are_not_scored(self):
        # A NO_MODEL day is the model sitting out because no combination
        # cleared the gate. Scoring it as a loss would understate the record,
        # and it has no feature set -- so the panel must branch on it too.
        body = re.sub(r"//[^\n]*", "", _body("renderPerDateOOS"))
        assert "rec.direction === 'NO_MODEL'" in body
        assert "isFlat || isNoModel" in body
        # Null numeric fields must be guarded, not passed to .toFixed().
        assert "rec.prob_up !== null" in body
        assert "rec.expected_return !== null" in body
        # An empty feature list must not render a dangling "Model used:".
        assert "rec.features && rec.features.length" in body


class TestStaleState:
    """Both boxes must be explicitly set on every render path."""

    def test_case_c_hides_trade_engine(self):
        early = _case_c_block()
        assert "trade-engine-box" in early
        assert "classList.add('hidden')" in early

    def test_case_c_restores_old_strategy_box(self):
        early = _case_c_block()
        assert "old-strategy-box" in early
        assert "classList.remove('hidden')" in early

    def test_trade_engine_always_visibility_set(self):
        # add('hidden') must exist for the trade engine, not just remove(),
        # otherwise a previously rendered actionable card survives.
        body = _body("renderPrediction")
        assert re.search(r"teBox\.classList\.add\('hidden'\)", body)

    def test_old_strategy_box_always_visibility_set(self):
        body = _body("renderPrediction")
        assert re.search(r"oldBox\.classList\.remove\('hidden'\)", body)
        assert re.search(r"getElementById\('old-strategy-box'\)\.classList\.add\('hidden'\)", body)


class TestDivBalance:
    def test_markup_is_balanced(self):
        body = INDEX[:INDEX.index("</body>")]
        body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
        assert len(re.findall(r"<div\b", body)) == len(re.findall(r"</div>", body))


class TestIntradayUnregressed:
    def test_no_intraday_horizon_added(self):
        assert "next_week" not in INDEX
        assert "next_month" not in INDEX

    def test_legacy_fallback_preserved(self):
        # The hardcoded set must still exist for the no-candidate case.
        assert "['chk-rsi', 'chk-ema59', 'chk-stochrsi']" in INDEX
