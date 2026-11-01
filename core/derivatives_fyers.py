"""Live Fyers Derivatives (Options Chain, Greeks & Futures) collector for Institutional Radar.

Fetches real-time Options Chain (strike-wise OI, OI change, Greeks, PCR, heaviest strikes)
and current Futures quote (basis, premium/discount) via Fyers API v3.
"""
from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo
from typing import Any
from pathlib import Path
import sys

from core.data_fyers import get_fyers_client

IST = ZoneInfo("Asia/Kolkata")

FYERS_SPECS = {
    "nifty": {
        "index_symbol": "NSE:NIFTY50-INDEX",
        "strike_step": 50,
        "strike_count": 15,
    },
    "banknifty": {
        "index_symbol": "NSE:NIFTYBANK-INDEX",
        "strike_step": 100,
        "strike_count": 15,
    },
}


def fetch_fyers_derivatives(symbol_name: str, client=None) -> dict[str, Any] | None:
    """Fetch live options chain, Greeks, and futures positioning from Fyers API v3.

    Returns a structured dictionary matching the dashboard contract, or None on failure.
    Never raises an unhandled exception so calling exporters remain robust.
    """
    key = symbol_name.lower().replace("-", "").replace("_", "")
    spec = FYERS_SPECS.get(key)
    if not spec:
        return None

    try:
        if client is None:
            client = get_fyers_client()
    except Exception as e:
        print(f"[{symbol_name}] Warning: Cannot initialize Fyers client: {e}")
        return None

    now = datetime.datetime.now(IST)
    time_str = now.strftime("%H:%M IST")

    try:
        res = client.optionchain(data={
            "symbol": spec["index_symbol"],
            "strikecount": spec["strike_count"],
            "greeks": "1",
        })
        if res.get("s") != "ok" or not res.get("data"):
            print(f"[{symbol_name}] Fyers optionchain warning: {res.get('message', res)}")
            return None

        data = res["data"]
        chain = data.get("optionsChain", [])
        vix_data = data.get("indiavixData", {})
        india_vix = vix_data.get("ltp")

        # 1. Spot and Futures price from the master index element (strike_price == -1)
        spot_price = None
        fut_price = None
        for item in chain:
            if item.get("strike_price") == -1:
                spot_price = item.get("ltp")
                fut_price = item.get("fp")
                break

        basis = round(fut_price - spot_price, 2) if (fut_price is not None and spot_price is not None) else None

        # 2. Options aggregation
        call_oi = data.get("callOi", 0)
        put_oi = data.get("putOi", 0)
        pcr = round(put_oi / call_oi, 2) if call_oi > 0 else 0.0
        total_oi = call_oi + put_oi
        pe_ratio = round((put_oi / total_oi) * 100, 1) if total_oi > 0 else 50.0

        ce_added = 0
        pe_added = 0
        heaviest_ce = None
        max_ce_oi = -1
        heaviest_pe = None
        max_pe_oi = -1

        by_strike: dict[float, dict[str, Any]] = {}

        for item in chain:
            opt_type = item.get("option_type")
            if opt_type not in ("CE", "PE"):
                continue

            strike = item.get("strike_price", 0)
            oi = int(item.get("oi") or 0)
            oich = int(item.get("oich") or 0)
            raw_oichp = item.get("oichp")
            if raw_oichp is not None and str(raw_oichp).strip() != "":
                try:
                    oichp = float(raw_oichp)
                except (ValueError, TypeError):
                    oichp = 0.0
            elif oi > 0 and oich != 0 and (oi - oich) > 0:
                oichp = round((oich / (oi - oich)) * 100, 2)
            else:
                oichp = 0.0
            ltp = item.get("ltp")
            greeks = item.get("greeks", {})

            if strike not in by_strike:
                by_strike[strike] = {"strike": strike}

            if opt_type == "CE":
                ce_added += oich
                by_strike[strike]["ce_oi"] = oi
                by_strike[strike]["ce_oich"] = oich
                by_strike[strike]["ce_oichp"] = oichp
                by_strike[strike]["ce_ltp"] = ltp
                by_strike[strike]["ce_iv"] = greeks.get("iv")
                by_strike[strike]["ce_delta"] = greeks.get("delta")
                if oi > max_ce_oi:
                    max_ce_oi = oi
                    heaviest_ce = {
                        "strike": strike,
                        "oi": oi,
                        "oich": oich,
                        "oichp": oichp,
                        "ltp": ltp,
                        "delta": greeks.get("delta"),
                    }
            elif opt_type == "PE":
                pe_added += oich
                by_strike[strike]["pe_oi"] = oi
                by_strike[strike]["pe_oich"] = oich
                by_strike[strike]["pe_oichp"] = oichp
                by_strike[strike]["pe_ltp"] = ltp
                by_strike[strike]["pe_iv"] = greeks.get("iv")
                by_strike[strike]["pe_delta"] = greeks.get("delta")
                if oi > max_pe_oi:
                    max_pe_oi = oi
                    heaviest_pe = {
                        "strike": strike,
                        "oi": oi,
                        "oich": oich,
                        "oichp": oichp,
                        "ltp": ltp,
                        "delta": greeks.get("delta"),
                    }

        # Front expiry date from expiryData
        expiry_str = None
        exp_list = data.get("expiryData", [])
        if exp_list and isinstance(exp_list, list) and len(exp_list) > 0:
            expiry_str = exp_list[0].get("date")

        payload = {
            "timestamp": time_str,
            "source": "fyers",
            "symbol": symbol_name.upper(),
            "generated_at": now.isoformat(),
            "futures": {
                "ltp": fut_price,
                "spot": spot_price,
                "basis": basis,
                "status": "PREMIUM" if (basis and basis > 0) else ("DISCOUNT" if (basis and basis < 0) else "PAR"),
            },
            "options": {
                "pcr": pcr,
                "call_oi": call_oi,
                "put_oi": put_oi,
                "ce_added": ce_added,
                "pe_added": pe_added,
                "pe_ratio": pe_ratio,
                "spot": spot_price,
                "vix": india_vix,
                "heaviest_ce": heaviest_ce,
                "heaviest_pe": heaviest_pe,
                "expiry": expiry_str,
            },
            "strikes": sorted(by_strike.values(), key=lambda x: x["strike"]),
        }
        return payload

    except Exception as e:
        print(f"[{symbol_name}] Error fetching Fyers derivatives: {e}")
        return None
