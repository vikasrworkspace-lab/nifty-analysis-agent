"""Intraday analogue search must not draw from the target's own session.

In a bar-level frame every bar's forward outcome (``rest_of_session`` /
``fwd_ret_1``) runs to its own session close. A candidate bar from the same
session as the target therefore carries an outcome that extends past the
target bar -- the target's own future. ``find_analogues(session_col="session")``
drops those candidates so analogues come only from strictly earlier sessions.

Without the guard, the nearest-neighbour cohort for a mid- or late-session bar
is dominated by the bars immediately before it in the same session, whose
"forward" return is largely the move the model is supposed to predict.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import patterns

TZ = "Asia/Kolkata"
INDEX = (ROOT / "index.html").read_text(encoding="utf-8")


def _frame():
    # Session A: bars 0,1,2 (2026-09-28). Session B: bars 3,4,5 (2026-09-29).
    idx = pd.DatetimeIndex(
        [
            "2026-09-28 09:15",
            "2026-09-28 09:20",
            "2026-09-28 09:25",
            "2026-09-29 09:15",
            "2026-09-29 09:20",
            "2026-09-29 09:25",
        ],
        tz=TZ,
    )
    session = pd.to_datetime(idx.date)
    # Session-B bars are engineered to sit right on top of the target so that,
    # with the guard removed, they would be the closest candidates.
    feat = [0.0, 0.0, 0.0, 0.99, 1.00, 1.00]
    frame = pd.DataFrame({"session": session, "feat": feat}, index=idx)
    z = pd.DataFrame({"feat": feat}, index=idx)
    return frame, z


def test_same_session_candidates_are_excluded():
    frame, z = _frame()
    target = 5  # last bar of session B
    pos, _ = patterns.find_analogues(
        frame, z, ["feat"], {"feat": 1.0}, target, k=10,
        standardize_lookback=3, session_col="session",
    )
    sessions = frame["session"].to_numpy()
    assert pos, "guard must not empty the candidate pool for this fixture"
    assert all(sessions[p] != sessions[target] for p in pos), (
        "analogue drawn from the target's own session: %s" % pos
    )
    assert set(pos) == {0, 1, 2}


def test_without_guard_same_session_can_leak():
    # Demonstrates why the guard exists: omitting session_col lets the two
    # same-session bars (3 and 4) into the cohort.
    frame, z = _frame()
    target = 5
    pos, _ = patterns.find_analogues(
        frame, z, ["feat"], {"feat": 1.0}, target, k=10,
        standardize_lookback=3, session_col=None,
    )
    sessions = frame["session"].to_numpy()
    assert any(sessions[p] == sessions[target] for p in pos)


def test_default_behaviour_unchanged_for_sessionless_frames():
    # Daily frames have no session column and must behave exactly as before.
    frame, z = _frame()
    frame = frame.drop(columns=["session"])
    pos, _ = patterns.find_analogues(
        frame, z, ["feat"], {"feat": 1.0}, 5, k=10, standardize_lookback=3,
    )
    assert pos


def test_js_matcher_applies_the_same_guard():
    # The shipped dashboard still computes intraday Top-K in the browser; until
    # that matcher is retired it must apply the same same-session exclusion.
    assert "isIntraday && histDate.slice(0, 10) === dateStr.slice(0, 10)" in INDEX


# --- IntradayWFO fold boundaries -------------------------------------------
# Forward_Return is measured to each bar's own session close, so a walk-forward
# fold edge that lands mid-session lets the final validation bar's label reach
# into the OOS window. The optimizer must snap fold edges to session starts.

SESSION_BARS = 10
SESSIONS = 8


def _bar_frame():
    days = pd.date_range("2026-09-01", periods=SESSIONS, freq="B", tz=TZ)
    offsets = np.tile(np.arange(SESSION_BARS) * 5, SESSIONS)
    idx = days.repeat(SESSION_BARS) + pd.to_timedelta(offsets, unit="m")
    n = len(idx)
    return pd.DataFrame(
        {
            "a": np.linspace(0.0, 1.0, n),
            "b": np.linspace(1.0, 0.0, n),
            "Forward_Return": np.full(n, 0.05),
        },
        index=pd.DatetimeIndex(idx),
    )


def test_wfo_fold_edges_land_on_session_boundaries():
    import importlib

    ei = importlib.import_module("scripts.export_intraday")
    frame = _bar_frame()

    # Sizes deliberately not multiples of SESSION_BARS, so unsnapped raw index
    # arithmetic would produce mid-session fold edges.
    wfo = ei.IntradayWFO(
        frame, ["a", "b"], k=3, train_size=22, val_size=25, oos_size=25
    )

    seen = []
    original = wfo._evaluate_combo

    def spy(X_train, y_train, X_eval, y_eval, features):
        seen.append(X_eval.index)
        return original(X_train, y_train, X_eval, y_eval, features)

    wfo._evaluate_combo = spy
    wfo.run()

    assert seen, "walk-forward produced no folds"

    days = frame.index.normalize()
    for window in seen:
        first = window[0]
        same_day = days == first.normalize()
        assert first == days[same_day][0], (
            "fold edge at %s is mid-session; its label reaches the next window"
            % first
        )


def test_wfo_session_snapping_helpers():
    import importlib

    ei = importlib.import_module("scripts.export_intraday")
    wfo = ei.IntradayWFO(_bar_frame(), ["a"], k=3, val_size=25, oos_size=25)
    starts = wfo._session_start_positions()

    assert starts[0] == 0
    assert list(starts) == list(range(0, SESSION_BARS * SESSIONS, SESSION_BARS))
    # forward snaps up to the next session start, back snaps down to the last
    assert wfo._snap_forward(starts, 7) == 10
    assert wfo._snap_back(starts, 17) == 10
    assert wfo._snap_forward(starts, 10) == 10
