"""Intraday collectors must never write one symbol's bars into another's archive.

Regression cover for the Bank Nifty contamination bug.

``update_from_yfinance`` read ``load_bars("nifty", ...)`` regardless of the
``name`` it was asked to refresh. So refreshing bank_nifty built
``data/intraday/5m/bank_nifty.csv`` as *Nifty's entire archive* with the fetched
60-day window laid over the tail (``drop_duplicates(..., keep="last")``). Every
bar outside that window kept Nifty's price, which is how 375 Bank Nifty bars
ended up byte-identical to Nifty.

``update_from_fyers`` already passed ``name`` correctly, so the two collectors
disagreed; the yfinance one was wrong.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NIFTY_PRICE = 22000.0
BANK_PRICE = 55000.0
TZ = "Asia/Kolkata"


def _settings(tmp_path):
    return {
        "symbols": {"nifty": "^NSEI", "bank_nifty": "^NSEBANK"},
        "intraday": {
            "base_timeframe": "5m",
            "session": {"open": "09:15", "close": "15:30", "tz": TZ},
            "yfinance_depth_days": 60,
        },
        "paths": {"intraday_archive_dir": str(tmp_path)},
    }


def _frame(prices, start="2026-07-01 09:15", periods=6, freq="5min"):
    idx = pd.date_range(start=start, periods=periods, freq=freq, tz=TZ)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": 0.0,
        },
        index=idx,
    )


class _FakeTicker:
    """Stands in for yf.Ticker; returns a fixed frame."""

    frame = None

    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, **kwargs):
        return type(self).frame.copy()


def _install_fake_yfinance(monkeypatch, frame):
    _FakeTicker.frame = frame
    import core.data_intraday as di
    monkeypatch.setitem(sys.modules, "yfinance", type("M", (), {"Ticker": _FakeTicker}))


def test_bank_refresh_does_not_write_nifty_bars(monkeypatch, tmp_path):
    import core.data_intraday as di

    settings = _settings(tmp_path)
    # Nifty archive already holds history; Bank Nifty's holds only one old day.
    di._save_bars(_frame(NIFTY_PRICE, start="2026-07-01 09:15", periods=10), "nifty", settings, 5)
    di._save_bars(_frame(BANK_PRICE, start="2026-07-01 09:15", periods=2), "bank_nifty", settings, 5)

    # The fetch returns fresh Bank Nifty bars on overlapping + new timestamps.
    _install_fake_yfinance(monkeypatch, _frame(BANK_PRICE, start="2026-07-01 09:15", periods=10))

    di.update_from_yfinance("bank_nifty", "^NSEBANK", settings)

    bank = di.load_bars("bank_nifty", 5, settings)
    assert not bank.empty
    # Every row must be Bank Nifty's price. Reading nifty's archive instead
    # leaves the two pre-fetch rows at 22000.
    assert set(bank["close"].round(2)) == {BANK_PRICE}, (
        "bank_nifty archive contains foreign prices: %s"
        % sorted(set(bank["close"].round(2)))
    )
    assert len(bank) == 10


def test_nifty_archive_untouched_by_bank_refresh(monkeypatch, tmp_path):
    import core.data_intraday as di

    settings = _settings(tmp_path)
    di._save_bars(_frame(NIFTY_PRICE, start="2026-07-01 09:15", periods=10), "nifty", settings, 5)
    di._save_bars(_frame(BANK_PRICE, start="2026-07-01 09:15", periods=2), "bank_nifty", settings, 5)

    _install_fake_yfinance(monkeypatch, _frame(BANK_PRICE, start="2026-07-01 09:15", periods=10))
    di.update_from_yfinance("bank_nifty", "^NSEBANK", settings)

    nifty = di.load_bars("nifty", 5, settings)
    assert set(nifty["close"].round(2)) == {NIFTY_PRICE}
    assert len(nifty) == 10, "nifty's archive was modified by a bank_nifty refresh"


def test_empty_fetch_returns_named_archive(monkeypatch, tmp_path):
    """The empty-fetch path must also return the *named* archive."""
    import core.data_intraday as di

    settings = _settings(tmp_path)
    di._save_bars(_frame(BANK_PRICE, start="2026-07-01 09:15", periods=4), "bank_nifty", settings, 5)
    di._save_bars(_frame(NIFTY_PRICE, start="2026-07-01 09:15", periods=4), "nifty", settings, 5)

    _install_fake_yfinance(monkeypatch, pd.DataFrame())

    out = di.update_from_yfinance("bank_nifty", "^NSEBANK", settings)
    assert set(out["close"].round(2)) == {BANK_PRICE}, (
        "empty fetch returned the wrong symbol's archive"
    )


def test_collectors_agree_on_named_archive():
    """Both collectors must read the named archive, not a hardcoded one."""
    import inspect

    import core.data_intraday as di

    for fn in (di.update_from_yfinance, di.update_from_fyers):
        src = inspect.getsource(fn)
        assert 'load_bars("nifty"' not in src, (
            "%s still hardcodes the nifty archive" % fn.__name__
        )
        assert "load_bars(name" in src, (
            "%s does not read the named archive" % fn.__name__
        )


class TestNoCrossContaminationInRealArchive:
    """Guard the actual shipped archives, when they are present."""

    def test_bank_5m_does_not_equal_nifty_5m(self):
        nifty_p = ROOT / "data" / "intraday" / "5m" / "nifty.csv"
        bank_p = ROOT / "data" / "intraday" / "5m" / "bank_nifty.csv"
        if not (nifty_p.exists() and bank_p.exists()):
            pytest.skip("5m archives not generated in this checkout")

        nifty = pd.read_csv(nifty_p, index_col=0, parse_dates=True)
        bank = pd.read_csv(bank_p, index_col=0, parse_dates=True)
        merged = nifty.join(bank, how="inner", lsuffix="_n", rsuffix="_b")
        if merged.empty:
            pytest.skip("archives share no timestamps")

        identical = merged[
            merged["close_n"].round(2) == merged["close_b"].round(2)
        ]
        assert identical.empty, (
            "%d bar(s) in bank_nifty 5m are identical to nifty 5m (first: %s). "
            "Bank Nifty and Nifty cannot trade the same price -- this means the "
            "collector wrote one symbol's bars into the other's archive."
            % (len(identical), identical.index[0])
        )
