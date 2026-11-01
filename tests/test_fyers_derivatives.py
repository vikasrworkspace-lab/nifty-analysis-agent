import datetime
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.derivatives_fyers import (
    FYERS_SPECS,
    fetch_fyers_derivatives,
)


def test_fyers_specs_coverage():
    assert "nifty" in FYERS_SPECS
    assert "banknifty" in FYERS_SPECS
    assert FYERS_SPECS["nifty"]["index_symbol"] == "NSE:NIFTY50-INDEX"
    assert FYERS_SPECS["banknifty"]["index_symbol"] == "NSE:NIFTYBANK-INDEX"


def test_fetch_fyers_derivatives_unknown_symbol():
    assert fetch_fyers_derivatives("UNKNOWN_SYM") is None


def test_fetch_fyers_derivatives_mock_client():
    class MockFyers:
        def optionchain(self, data=None):
            return {
                "s": "ok",
                "data": {
                    "callOi": 10000000,
                    "putOi": 15000000,
                    "indiavixData": {"ltp": 14.5},
                    "expiryData": [{"date": "03-11-2026"}],
                    "optionsChain": [
                        {
                            "strike_price": -1,
                            "ltp": 23000.0,
                            "fp": 23040.0,
                        },
                        {
                            "strike_price": 23200,
                            "option_type": "CE",
                            "oi": 5000000,
                            "oich": 500000,
                            "ltp": 25.0,
                            "greeks": {"delta": 0.2, "iv": 13.0},
                        },
                        {
                            "strike_price": 22800,
                            "option_type": "PE",
                            "oi": 6000000,
                            "oich": 800000,
                            "ltp": 30.0,
                            "greeks": {"delta": -0.25, "iv": 14.0},
                        },
                    ],
                },
            }

    mock = MockFyers()
    res = fetch_fyers_derivatives("nifty", client=mock)

    assert res is not None
    assert res["source"] == "fyers"
    assert res["symbol"] == "NIFTY"
    assert res["futures"]["spot"] == 23000.0
    assert res["futures"]["ltp"] == 23040.0
    assert res["futures"]["basis"] == 40.0
    assert res["futures"]["status"] == "PREMIUM"
    assert res["options"]["pcr"] == 1.5
    assert res["options"]["vix"] == 14.5
    assert res["options"]["heaviest_ce"]["strike"] == 23200
    assert res["options"]["heaviest_pe"]["strike"] == 22800
    assert res["options"]["active_support_pe"]["strike"] == 22800
    assert res["options"]["active_resistance_ce"]["strike"] == 23200
    assert res["options"]["active_corridor"]["floor"] == 22800
    assert res["options"]["active_corridor"]["ceiling"] == 23200
    assert res["directional_bias"] == "BULLISH"
    assert res["options"]["atm"]["strike"] == 23000
    assert res["options"]["expiry"] == "03-11-2026"
    assert len(res["strikes"]) == 2
