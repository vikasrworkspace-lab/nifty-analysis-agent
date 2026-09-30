"""BTST walk-forward window contract.

The model-selection window must never overlap the out-of-sample window,
otherwise the reported "validation" edge is really an in-sample figure and the
UI's validation-vs-OOS comparison is meaningless. ``wfo_window_bounds`` is the
single source of truth for that split; the exporter's final feature selection
scores combinations on ``[val_start, val_end)`` only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_dashboard import wfo_window_bounds


class TestWindowBounds:
    def test_validation_is_disjoint_from_oos(self):
        val_start, val_end, oos_start, oos_end = wfo_window_bounds(3000, 252, 126)
        assert val_start < val_end
        assert oos_start < oos_end
        # Back to back, no shared index: selection cannot see any OOS bar.
        assert val_end == oos_start

    def test_window_widths_match_configuration(self):
        val_start, val_end, oos_start, oos_end = wfo_window_bounds(3000, 252, 126)
        assert val_end - val_start == 252
        assert oos_end - oos_start == 126

    def test_no_selection_index_reaches_an_oos_index(self):
        _, val_end, oos_start, _ = wfo_window_bounds(2500, 252, 126)
        assert val_end <= oos_start

    def test_oos_stops_one_bar_early(self):
        # The last bar's forward outcome is unknown, so OOS must exclude it.
        _, _, _, oos_end = wfo_window_bounds(1000, 252, 126)
        assert oos_end == 999

    def test_bounds_are_zero_at_minimum_usable_history(self):
        val_start, _, oos_start, _ = wfo_window_bounds(379, 252, 126)
        assert val_start == 0
        assert oos_start == 252

    def test_scales_for_intraday_sized_history(self):
        val_start, val_end, oos_start, oos_end = wfo_window_bounds(4000, 300, 300)
        assert val_start == 4000 - 300 - 300 - 1
        assert val_end == oos_start
        assert oos_end == 3999
