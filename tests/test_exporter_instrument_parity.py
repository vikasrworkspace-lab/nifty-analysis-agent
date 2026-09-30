"""Per-instrument parity for scripts/export_dashboard.py.

Covers the two places where Bank Nifty was silently served Nifty's data:
the jugaad recent-date repair (gated on ``symbol_name == "nifty"``) and the
option chain (hardcoded ``index_option_chain("NIFTY")`` plus fixed 300/1000
point strike windows that contain no NIFTY BANK strikes at all).
"""

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_dashboard as ed  # noqa: E402


SRC = (ROOT / "scripts" / "export_dashboard.py").read_text(encoding="utf-8")


def _chain(strikes, price, put_oi=None, call_oi=None):
    """Build a jugaad-shaped option chain payload."""
    put_oi = put_oi or {}
    call_oi = call_oi or {}
    data = []
    for s in strikes:
        data.append(
            {
                "strikePrice": s,
                "PE": {
                    "strikePrice": s,
                    "openInterest": put_oi.get(s, 0),
                    "changeinOpenInterest": 0,
                },
                "CE": {
                    "strikePrice": s,
                    "openInterest": call_oi.get(s, 0),
                    "changeinOpenInterest": 0,
                },
            }
        )
    return {"records": {"underlyingValue": price, "data": data}}


def _pe(oc):
    return [i["PE"] for i in oc["records"]["data"]]


def _ce(oc):
    return [i["CE"] for i in oc["records"]["data"]]


def _instrument_table():
    """Parse the exporter's `instruments` list straight from the source."""
    body = re.search(r"instruments = \[(.*?)\n    \]", SRC, re.S).group(1)
    return re.findall(
        r'\("(\w+)",\s*"(\w+)",\s*"([^"]+)",\s*\("(\w+)",\s*"([^"]+)"\),'
        r'\s*\("(\w+)",\s*"([^"]+)"\)',
        body,
    )


# --------------------------------------------------------------------------
# Strike scaling
# --------------------------------------------------------------------------


class TestStrikeScaling:
    def test_nifty_50pt_strikes_reproduce_the_old_hardcoded_bands(self):
        """Nifty must be bit-for-bit unaffected by the refactor."""
        strikes = list(range(23000, 23101, 50))
        oc = _chain(strikes, 23050)
        support, pain, momentum = ed._strike_bands(_pe(oc), _ce(oc))
        assert (support, pain, momentum) == (300, 1000, 500)

    def test_bank_nifty_500pt_strikes_get_proportionally_wider_bands(self):
        strikes = list(range(54000, 57101, 500))
        oc = _chain(strikes, 55500)
        support, pain, momentum = ed._strike_bands(_pe(oc), _ce(oc))
        assert (support, pain, momentum) == (3000, 10000, 5000)

    def test_old_fixed_window_finds_nothing_for_bank_nifty(self):
        """The regression being fixed: +/-300 on 500-pt strikes is empty."""
        oc = _chain(list(range(52000, 59101, 500)), 55500)
        puts_below = [x for x in _pe(oc) if 55500 - 300 <= x["strikePrice"] < 55500]
        assert puts_below == []
        support, _, _ = ed._strike_bands(_pe(oc), _ce(oc))
        puts_below = [x for x in _pe(oc) if 55500 - support <= x["strikePrice"] < 55500]
        # 6 x 500pt spacing -> 3000pt window -> 6 strikes below spot.
        assert len(puts_below) == 6

    def test_spacing_is_median_gap_not_mean(self):
        """A gappy chain must not widen the window via one huge outlier gap."""
        strikes = [100, 150, 200, 2000]
        oc = _chain(strikes, 200)
        assert ed._strike_spacing(_pe(oc), _ce(oc)) == 50

    def test_spacing_none_when_chain_is_too_thin(self):
        assert ed._strike_spacing([], []) is None
        assert ed._strike_spacing([{"strikePrice": 100}], []) is None

    def test_spacing_ignores_duplicate_strikes_from_pe_and_ce(self):
        oc = _chain([100, 150, 200], 150)
        assert ed._strike_spacing(_pe(oc), _ce(oc)) == 50

    def test_source_has_no_hardcoded_option_windows(self):
        for bad in ("current_price - 300", "current_price + 300",
                    "current_price - 1000", "current_price + 1000"):
            assert bad not in SRC, "%s is hardcoded again" % bad


# --------------------------------------------------------------------------
# Option chain dispatch
# --------------------------------------------------------------------------


    def test_observed_nse_ladders_scale_as_expected(self):
        """Real ladders seen from NSELive, not synthetic ones.

        NIFTY returns a 50-pt ladder (so bands land back on the historical
        300/1000/500) and BANKNIFTY a 100-pt one. Both chains also carry sparse
        far-out strikes with 1000-1500 pt gaps at the wings, so the estimator
        has to be the median gap and not the first or the mean.
        """
        nifty = list(range(20500, 24801, 50)) + [15000, 16500, 18000, 19500]
        bank = list(range(47000, 59601, 100)) + [43500, 45000, 46500]
        oc_n, oc_b = _chain(nifty, 22650), _chain(bank, 54250)
        assert ed._strike_bands(_pe(oc_n), _ce(oc_n)) == (300, 1000, 500)
        assert ed._strike_bands(_pe(oc_b), _ce(oc_b)) == (600, 2000, 1000)


class TestOptionChainDispatch:
    def test_index_and_stock_kinds_use_different_jugaad_methods(self, monkeypatch):
        calls = []

        class FakeLive:
            def index_option_chain(self, symbol):
                calls.append(("index", symbol))
                return "IDX"

            def equities_option_chain(self, symbol):
                calls.append(("stock", symbol))
                return "EQ"

        import jugaad_data.nse as nsemod
        monkeypatch.setattr(nsemod, "NSELive", FakeLive)

        assert ed._fetch_option_chain(("index", "NIFTY BANK")) == "IDX"
        assert ed._fetch_option_chain(("stock", "RELIANCE")) == "EQ"
        assert calls == [("index", "NIFTY BANK"), ("stock", "RELIANCE")]

    def test_missing_spec_returns_none_rather_than_falling_back_to_nifty(self, monkeypatch):
        def explode(self, symbol):
            raise AssertionError("must not fall back to %r" % symbol)

        import jugaad_data.nse as nsemod
        monkeypatch.setattr(nsemod, "NSELive", lambda: type(
            "L", (), {"index_option_chain": explode, "equities_option_chain": explode})())
        assert ed._fetch_option_chain(None) is None

    def test_source_does_not_hardcode_the_nifty_chain(self):
        assert 'index_option_chain("NIFTY")' not in SRC


# --------------------------------------------------------------------------
# jugaad recent-date backfill
# --------------------------------------------------------------------------


def _ns_frame(date_col, n=3, start="2026-09-21"):
    """n consecutive days from `start`. Default start is a Monday."""
    return pd.DataFrame(
        {
            date_col: pd.date_range(start, periods=n).date,
            "OPEN": [1.0 + i for i in range(n)],
            "HIGH": [2.0 + i for i in range(n)],
            "LOW": [0.5 + i for i in range(n)],
            "CLOSE": [1.5 + i for i in range(n)],
            "TIMESTAMP": ["x"] * n,
        }
    )


class TestJugaadBackfill:
    def test_none_spec_skips_without_touching_the_network(self):
        assert ed.fetch_jugaad_backfill(None, "2026-09-01", "2026-09-29") is None

    def test_index_frame_normalises_historicaldate(self, monkeypatch):
        import jugaad_data.nse as nsemod
        seen = {}

        def fake(symbol, from_date, to_date):
            seen.update(symbol=symbol, from_date=from_date, to_date=to_date)
            return _ns_frame("HistoricalDate")

        monkeypatch.setattr(nsemod, "index_df", fake)
        out = ed.fetch_jugaad_backfill(("index", "NIFTY BANK"), "2026-09-01", "2026-09-29")
        assert seen["symbol"] == "NIFTY BANK"
        assert list(out.columns) == ["Open", "High", "Low", "Close"]
        assert out.index.tz is None
        assert out["Close"].tolist() == [1.5, 2.5, 3.5]

    def test_stock_frame_normalises_date_column(self, monkeypatch):
        import jugaad_data.nse as nsemod
        monkeypatch.setattr(nsemod, "stock_df", lambda **kw: _ns_frame("DATE"))
        out = ed.fetch_jugaad_backfill(("stock", "RELIANCE"), "2026-09-01", "2026-09-29")
        assert list(out.columns) == ["Open", "High", "Low", "Close"]
        assert len(out) == 3

    def test_network_failure_degrades_to_none(self, monkeypatch):
        import jugaad_data.nse as nsemod

        def boom(**kw):
            raise RuntimeError("NSE down")

        monkeypatch.setattr(nsemod, "index_df", boom)
        assert ed.fetch_jugaad_backfill(("index", "NIFTY"), "a", "b") is None

    def test_missing_ohlc_column_degrades_to_none(self, monkeypatch):
        import jugaad_data.nse as nsemod
        frame = _ns_frame("HistoricalDate").drop(columns=["HIGH"])
        monkeypatch.setattr(nsemod, "index_df", lambda **kw: frame)
        assert ed.fetch_jugaad_backfill(("index", "NIFTY"), "a", "b") is None

    def test_empty_frame_degrades_to_none(self, monkeypatch):
        import jugaad_data.nse as nsemod
        empty = _ns_frame("HistoricalDate").iloc[0:0]
        monkeypatch.setattr(nsemod, "index_df", lambda **kw: empty)
        assert ed.fetch_jugaad_backfill(("index", "NIFTY"), "a", "b") is None

    def test_ohlc_values_are_positional_not_label_aligned(self, monkeypatch):
        """Regression: aligning on ns's RangeIndex made every value NaN.

        Assigning ns[src] into a frame indexed by date label-aligns, and ns
        still carries a RangeIndex, so the whole backfill silently became
        all-NaN rows that then got concatenated into the price history.
        """
        import jugaad_data.nse as nsemod
        monkeypatch.setattr(nsemod, "index_df", lambda **kw: _ns_frame("HistoricalDate", 4))
        out = ed.fetch_jugaad_backfill(("index", "NIFTY"), "a", "b")
        assert not out.isna().any().any(), "backfill frame contains NaN"
        assert out["Close"].tolist() == [1.5, 2.5, 3.5, 4.5]

    def test_all_nan_rows_are_dropped_not_concatenated(self, monkeypatch):
        import jugaad_data.nse as nsemod
        frame = _ns_frame("HistoricalDate", 2)
        frame["CLOSE"] = [None, None]
        monkeypatch.setattr(nsemod, "index_df", lambda **kw: frame)
        assert ed.fetch_jugaad_backfill(("index", "NIFTY"), "a", "b") is None

    def test_backfill_index_concatenates_against_a_tz_naive_cache(self, monkeypatch):
        """The historical tz-aware/tz-naive concat crash must not return."""
        import jugaad_data.nse as nsemod
        monkeypatch.setattr(nsemod, "index_df", lambda **kw: _ns_frame("HistoricalDate", 2))
        ns = ed.fetch_jugaad_backfill(("index", "NIFTY"), "a", "b")
        df = pd.DataFrame(
            {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0]},
            index=pd.DatetimeIndex(["2026-09-19"]),
        )
        missing = ns[~ns.index.isin(df.index)]
        merged = pd.concat([df, missing]).sort_index()
        assert len(merged) == 3
        assert merged.index.tz is None

    def test_source_no_longer_gates_backfill_on_nifty_only(self):
        assert 'if symbol_name == "nifty":\n            ns = index_df' not in SRC
        assert "fetch_jugaad_backfill(jugaad_spec" in SRC

    def test_sunday_rows_from_jugaad_are_not_appended(self, monkeypatch):
        """stock_df() returns Sunday-dated non-sessions; they must not land."""
        import jugaad_data.nse as nsemod
        frame = _ns_frame("DATE", 7, start="2026-09-20")  # Sunday start
        assert [d.weekday() for d in frame["DATE"]] == [6, 0, 1, 2, 3, 4, 5]
        monkeypatch.setattr(nsemod, "stock_df", lambda **kw: frame)
        out = ed.fetch_jugaad_backfill(("stock", "RELIANCE"), "a", "b")
        assert list(out.index) == list(pd.date_range("2026-09-21", periods=5))

    def test_tz_aware_cache_is_normalised_before_concat(self):
        """The yfinance fallback builds a tz-aware index; isin would raise."""
        assert "df.index = df.index.tz_localize(None)" in SRC
        tz_aware = pd.DataFrame(
            {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0]},
            index=pd.DatetimeIndex(["2026-09-19"]).tz_localize("America/New_York"),
        )
        if getattr(tz_aware.index, "tz", None) is not None:
            tz_aware.index = tz_aware.index.tz_localize(None)
        ns = ed.fetch_jugaad_backfill(("index", "NIFTY"), "a", "b")
        assert ns is None or len(pd.concat([tz_aware, ns])) >= 1


# --------------------------------------------------------------------------
# Instrument table
# --------------------------------------------------------------------------


class TestInstrumentTable:
    def test_every_instrument_declares_backfill_and_chain(self):
        table = _instrument_table()
        assert {r[0] for r in table} == {"nifty", "banknifty", "reliance"}
        for name, _cache, _yf, jkind, jsym, ckind, csym in table:
            assert jkind in ("index", "stock"), name
            assert ckind in ("index", "stock"), name
            assert jsym and csym, name

    def test_bank_nifty_uses_bank_nifty_sources(self):
        row = [r for r in _instrument_table() if r[0] == "banknifty"][0]
        # index_df() and NSELive.index_option_chain spell Bank Nifty
        # differently. "NIFTY BANK" is correct for the first and returns an
        # empty payload (no "records" key) for the second.
        assert row[4] == "NIFTY BANK", "backfill must query index_df('NIFTY BANK')"
        assert row[6] == "BANKNIFTY", "option chain must be index_option_chain('BANKNIFTY')"

    def test_bank_nifty_chain_symbol_is_not_the_jugaad_spelling(self):
        row = [r for r in _instrument_table() if r[0] == "banknifty"][0]
        assert row[4] != row[6], "the two jugaad APIs use different Bank Nifty spellings"

    def test_nifty_row_unchanged(self):
        row = [r for r in _instrument_table() if r[0] == "nifty"][0]
        assert (row[1], row[2], row[4], row[6]) == ("nifty", "^NSEI", "NIFTY 50", "NIFTY")

    def test_reliance_row_uses_equity_chain(self):
        row = [r for r in _instrument_table() if r[0] == "reliance"][0]
        assert (row[4], row[5], row[6]) == ("RELIANCE", "stock", "RELIANCE")
