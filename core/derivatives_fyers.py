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

        # 3. Near-The-Money (NTM) Active Corridor & 1-Lot Capital Safeguard
        # Far-OTM strikes like 22,500 / 23,500 carry accumulated monthly OI from weeks ago.
        # Intraday traders need immediate active walls near spot (±250 pts Nifty, ±600 pts BankNifty).
        near_window = 250 if key == "nifty" else 600
        lot_size = 75 if key == "nifty" else 30
        strike_step = spec.get("strike_step", 50)

        active_support_pe = None
        active_resistance_ce = None
        atm_strike = None
        atm_ce = None
        atm_pe = None

        if spot_price:
            atm_strike = int(round(spot_price / strike_step) * strike_step)
            atm_data = by_strike.get(atm_strike, {})
            atm_ce = {
                "strike": atm_strike,
                "ltp": atm_data.get("ce_ltp") or 0.0,
                "delta": atm_data.get("ce_delta") or 0.50,
                "lot_size": lot_size,
                "capital_required": round((atm_data.get("ce_ltp") or 0.0) * lot_size),
            }
            atm_pe = {
                "strike": atm_strike,
                "ltp": atm_data.get("pe_ltp") or 0.0,
                "delta": atm_data.get("pe_delta") or -0.50,
                "lot_size": lot_size,
                "capital_required": round((atm_data.get("pe_ltp") or 0.0) * lot_size),
            }

            # Near-the-money PE candidates (Support floor)
            pe_near = [
                s for s in by_strike.values()
                if (spot_price - near_window) <= s["strike"] <= (spot_price + 25) and "pe_oi" in s
            ]
            if pe_near:
                # Rank by open interest (with bonus if today's change is strongly positive)
                pe_near.sort(key=lambda s: (s.get("pe_oi") or 0) + max(0, (s.get("pe_oich") or 0) * 0.5), reverse=True)
                top_pe = pe_near[0]
                active_support_pe = {
                    "strike": top_pe["strike"],
                    "oi": top_pe.get("pe_oi") or 0,
                    "oich": top_pe.get("pe_oich") or 0,
                    "oichp": top_pe.get("pe_oichp") or 0.0,
                    "ltp": top_pe.get("pe_ltp") or 0.0,
                    "delta": top_pe.get("pe_delta"),
                    "dist_pts": round(spot_price - top_pe["strike"], 1),
                }

            # Near-the-money CE candidates (Resistance ceiling)
            ce_near = [
                s for s in by_strike.values()
                if (spot_price - 25) <= s["strike"] <= (spot_price + near_window) and "ce_oi" in s
            ]
            if ce_near:
                ce_near.sort(key=lambda s: (s.get("ce_oi") or 0) + max(0, (s.get("ce_oich") or 0) * 0.5), reverse=True)
                top_ce = ce_near[0]
                active_resistance_ce = {
                    "strike": top_ce["strike"],
                    "oi": top_ce.get("ce_oi") or 0,
                    "oich": top_ce.get("ce_oich") or 0,
                    "oichp": top_ce.get("ce_oichp") or 0.0,
                    "ltp": top_ce.get("ce_ltp") or 0.0,
                    "delta": top_ce.get("ce_delta"),
                    "dist_pts": round(top_ce["strike"] - spot_price, 1),
                }

        # Fallback to global heaviest if near window produced nothing
        if not active_support_pe:
            active_support_pe = heaviest_pe
        if not active_resistance_ce:
            active_resistance_ce = heaviest_ce

        # 4. Synthesize Real-Time Institutional Direction Verdict
        dir_verdict = "SIDEWAYS"
        dir_strength = "NEUTRAL"
        if (basis and basis > 15) and pe_ratio >= 52:
            dir_verdict = "STRONG BULLISH"
            dir_strength = "BULLISH"
        elif (basis and basis < -15) and pe_ratio <= 48:
            dir_verdict = "STRONG BEARISH"
            dir_strength = "BEARISH"
        elif pe_ratio >= 58:
            dir_verdict = "BULLISH (PUT WRITING DEFENSE)"
            dir_strength = "BULLISH"
        elif pe_ratio <= 42:
            dir_verdict = "BEARISH (CALL RESISTANCE CAPPING)"
            dir_strength = "BEARISH"
        else:
            dir_verdict = "SIDEWAYS (RANGEBOUND CHOP)"
            dir_strength = "SIDEWAYS"

        corridor_floor = active_support_pe.get("strike") if active_support_pe else (atm_strike - 100 if atm_strike else 0)
        corridor_ceil = active_resistance_ce.get("strike") if active_resistance_ce else (atm_strike + 100 if atm_strike else 0)

        payload = {
            "timestamp": time_str,
            "source": "fyers",
            "symbol": symbol_name.upper(),
            "generated_at": now.isoformat(),
            "directional_verdict": dir_verdict,
            "directional_bias": dir_strength,
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
                "active_support_pe": active_support_pe,
                "active_resistance_ce": active_resistance_ce,
                "active_corridor": {
                    "floor": corridor_floor,
                    "ceiling": corridor_ceil,
                    "width": corridor_ceil - corridor_floor if (corridor_ceil and corridor_floor) else 0,
                },
                "atm": {
                    "strike": atm_strike,
                    "ce": atm_ce,
                    "pe": atm_pe,
                },
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
