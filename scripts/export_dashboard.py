import json
import os
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
import yfinance
import requests
import re
import yfinance as yf

try:
    from nselib import capital_market
    NSELIB_AVAILABLE = True
except ImportError:
    NSELIB_AVAILABLE = False

def build_wfo_meta(best_final_edge, best_final_stats, oos_edge, oos_win_rate,
                   oos_trades, stability, val_window, oos_window,
                   ui_boxes=None):
    """Build the BTST ``_meta`` block for the daily WFO.

    Kept as a pure function so the "SIGNAL" / "NO SIGNAL" contract can be unit
    tested without running the full walk-forward pass (which takes ~90s and
    needs network-backed data).

    Contract:
      * SIGNAL is only ever emitted when the OOS edge is strictly positive and
        at least one feature was selected. Never force it.
      * NO SIGNAL is a legitimate, fully-populated result: it reports the same
        edge/trade figures as SIGNAL plus a human-readable ``reason`` naming
        the gate that failed, and an empty feature list. The UI relies on this
        to explain itself instead of rendering silent blanks.
    """
    common = {
        "val_edge": round(float(best_final_edge), 4) if best_final_edge != -999 else 0,
        "val_win_rate": round(float(best_final_stats[1] * 100), 1) if best_final_stats else 0,
        "val_trades": int(best_final_stats[0]) if best_final_stats else 0,
        "oos_edge": round(float(oos_edge), 4),
        "oos_win_rate": round(float(oos_win_rate * 100), 1) if oos_trades > 0 else 0,
        "oos_trades": int(oos_trades),
        "stability": stability,
        "val_window": val_window,
        "oos_window": oos_window,
    }

    if best_final_edge > 0 and oos_edge > 0 and ui_boxes:
        meta = dict(common)
        meta["wfo_optimal_features"] = list(ui_boxes)
        meta["status"] = "SIGNAL"
        meta["reason"] = "OOS edge > 0"
        return meta

    if best_final_stats is None:
        reason = "no feature combination cleared the 5-trade floor"
    elif best_final_edge <= 0:
        reason = "validation edge <= 0"
    else:
        reason = "OOS edge <= 0"

    meta = dict(common)
    meta["wfo_optimal_features"] = []
    meta["status"] = "NO SIGNAL"
    meta["reason"] = reason
    return meta


def process_symbol(symbol_name, db_filename):
    data_file = f"dashboard_data.json" if symbol_name == "nifty" else f"dashboard_data_{symbol_name}.json"
    data = {}
    
    # Load from our new unified Yahoo+Fyers Database
    csv_path = f"data/historical/{db_filename}"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        # Standardize column names
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        print(f"Loaded master DB for {symbol_name}: {len(df)} rows.")
    else:
        print(f"Master DB {csv_path} missing. Fallback to yfinance...")
        ticker_map = {"nifty": "^NSEI", "banknifty": "^NSEBANK", "reliance": "RELIANCE.NS"}
        ticker_sym = ticker_map.get(symbol_name, "^NSEI")
        ticker = yf.Ticker(ticker_sym)
        df = ticker.history(period="10y")
    
    # Fetch Institutional Data (VIX & USDINR)
    try:
        vix = yf.Ticker("^INDIAVIX")
        vix_history = vix.history(period="10y")
        df["VIX"] = vix_history["Close"]
        
        usdinr = yf.Ticker("INR=X")
        inr_history = usdinr.history(period="10y")
        df["USDINR"] = inr_history["Close"]
        
        sp500 = yf.Ticker("^GSPC")
        sp500_history = sp500.history(period="10y")
        df["SP500"] = sp500_history["Close"]
        
        bank_nifty = yf.Ticker("^NSEBANK")
        bank_history = bank_nifty.history(period="10y")
        df["BANK_NIFTY"] = bank_history["Close"]
    except Exception as e:
        print("Failed to fetch institutional data:", e)

    # --- NEW: Fetch missing recent dates using jugaad-data (more reliable) ---
    try:
        from jugaad_data.nse import index_df
        from datetime import datetime, timedelta
        
        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=30)
        # Skip jugaad_data fallback for non-nifty to keep it simple and clean
        if symbol_name == "nifty":
            ns = index_df(symbol="NIFTY 50", from_date=from_date, to_date=to_date)
            
            if not ns.empty:
                ns['Date'] = pd.to_datetime(ns['HistoricalDate'])
                ns = ns.set_index('Date')
                ns = ns.sort_index()
                ns.index = ns.index.tz_localize('Asia/Kolkata')
                
                ns['Open'] = pd.to_numeric(ns['OPEN'])
                ns['High'] = pd.to_numeric(ns['HIGH'])
                ns['Low'] = pd.to_numeric(ns['LOW'])
                ns['Close'] = pd.to_numeric(ns['CLOSE'])
                ns = ns[['Open', 'High', 'Low', 'Close']]
                
                # Find which dates from ns are missing in df
                missing = ns[~ns.index.isin(df.index)]
                if not missing.empty:
                    df = pd.concat([df, missing]).sort_index()
                    print(f"Appended {len(missing)} missing days from jugaad_data.")
    except Exception as e:
        print("Fallback jugaad_data fetch failed:", e)


    df["Return"] = df["Close"].pct_change()
    
    if "USDINR" in df.columns:
        df["USDINR_Return"] = df["USDINR"].pct_change()
    else:
        df["USDINR"] = 80.0
        df["USDINR_Return"] = 0.0
        
    if "VIX" not in df.columns:
        df["VIX"] = 15.0
        
    if "SP500" in df.columns:
        # Note: SP500 close yesterday provides the overnight cue for today's Nifty open
        df["SP500_Return"] = df["SP500"].pct_change().shift(1)
    else:
        df["SP500_Return"] = 0.0
        
    if "BANK_NIFTY" in df.columns:
        df["BANK_NIFTY_Return"] = df["BANK_NIFTY"].pct_change()
    else:
        df["BANK_NIFTY_Return"] = 0.0
    
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.ema(length=5, append=True)
    df.ta.ema(length=9, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.stochrsi(length=14, rsi_length=14, k=3, d=3, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.supertrend(length=7, multiplier=3.0, append=True)
    df.ta.atr(length=14, append=True)
    
    # Calculate continuous differentials
    df['EMA5_9_diff'] = df['EMA_5'] - df['EMA_9']
    df['Price_20EMA_diff'] = df['Close'] - df['EMA_20']
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, 1)
    df['BN_Rel'] = df['BANK_NIFTY_Return'] - df['Return']

    # Apply Rolling Z-Score Normalization (prevent lookahead bias)
    rolling_window = 252
    features_to_normalize = {
        'RSI_14': 'z_rsi',
        'STOCHRSIk_14_14_3_3': 'z_stochrsi',
        'EMA5_9_diff': 'z_ema_diff',
        'Price_20EMA_diff': 'z_price_ema',
        'ATRr_14': 'z_atr',
        'Vol_Ratio': 'z_vol',
        'VIX': 'z_vix',
        'BN_Rel': 'z_bn_rel'
    }
    
    for col, z_name in features_to_normalize.items():
        if col in df.columns:
            r_mean = df[col].rolling(rolling_window).mean()
            r_std = df[col].rolling(rolling_window).std().replace(0, 1e-5)
            df[z_name] = (df[col] - r_mean) / r_std
    

    # Fetch FII data
    mc_fii_dict = {}
    try:
        from jugaad_data.nse import NSELive
        n = NSELive()
        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        res = n.s.get(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.5", "Accept": "*/*"})
        if res.status_code == 200:
            for item in res.json():
                if item['category'] == 'FII/FPI':
                    # Parse date "24-Sep-2026" to "2026-09-24"
                    dt_obj = datetime.strptime(item['date'], '%d-%b-%Y')
                    d_str = dt_obj.strftime('%Y-%m-%d')
                    mc_fii_dict[d_str] = float(item['netValue'])
    except Exception as e:
        print(f"NSE FII fetch failed: {e}")
        pass

    try:
        url = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.php"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        matches = re.findall(r'\{"date":"([^"]+)".*?"fiiCM":"([^"]+)"', response.text)
        for d, val in matches:
            if d not in mc_fii_dict:
                mc_fii_dict[d] = float(val.replace(',', ''))
    except Exception as e:
        pass

    fii_df = None
    if NSELIB_AVAILABLE:
        try:
            fii_df = capital_market.fii_dii_trading_activity()
        except Exception as e:
            pass

    # Fetch Options Data for Support and Resistance globally (only valid for current time)
    global_options_support = None
    global_options_resistance = None
    global_options_max_pain = None
    global_options_momentum = None
    try:
        from jugaad_data.nse import NSELive
        n = NSELive()
        oc = n.index_option_chain("NIFTY")

        # Get current price
        current_price = oc['records']['underlyingValue']

        # Calculate Support and Resistance from Option Chain (Max OI)
        pe_data = []
        ce_data = []
        for item in oc['records']['data']:
            if 'PE' in item:
                pe_data.append(item['PE'])
            if 'CE' in item:
                ce_data.append(item['CE'])

        # Find Intraday Support (Max Put OI strictly within 300 pts below current price)
        puts_below = [x for x in pe_data if (current_price - 300) <= x['strikePrice'] < current_price]
        if puts_below:
            max_put = max(puts_below, key=lambda x: x['openInterest'])
            global_options_support = max_put['strikePrice']

        # Find Intraday Resistance (Max Call OI strictly within 300 pts above current price)
        calls_above = [x for x in ce_data if current_price < x['strikePrice'] <= (current_price + 300)]
        if calls_above:
            max_call = max(calls_above, key=lambda x: x['openInterest'])
            global_options_resistance = max_call['strikePrice']

        # Calculate Intraday Option Momentum (Delta OI)
        # Sum of changeinOpenInterest for strikes within +/- 500 points
        put_delta = sum([x.get('changeinOpenInterest', 0) for x in pe_data if abs(x['strikePrice'] - current_price) <= 500])
        call_delta = sum([x.get('changeinOpenInterest', 0) for x in ce_data if abs(x['strikePrice'] - current_price) <= 500])
        
        global_options_momentum = None
        if call_delta > 0 or put_delta > 0:
            if put_delta > call_delta:
                global_options_momentum = "bullish"
            else:
                global_options_momentum = "bearish"

        # Calculate Max Pain
        all_strikes = sorted(list(set([x['strikePrice'] for x in pe_data + ce_data])))
        check_strikes = [s for s in all_strikes if current_price - 1000 <= s <= current_price + 1000]
        
        min_loss = float('inf')
        for expiry_price in check_strikes:
            total_loss = 0
            for ce in ce_data:
                if expiry_price > ce['strikePrice']:
                    total_loss += (expiry_price - ce['strikePrice']) * ce['openInterest']
            for pe in pe_data:
                if expiry_price < pe['strikePrice']:
                    total_loss += (pe['strikePrice'] - expiry_price) * pe['openInterest']
            if total_loss < min_loss:
                min_loss = total_loss
                global_options_max_pain = expiry_price
    except Exception as e:
        print(f"Option Chain fetch failed: {e}")
        pass

    # Process all dates where we have at least 200 days of history
    trading_days = df.index[200:]
    returns_arr = df['Return'].values
    
    for date_obj in trading_days:
        date_str = date_obj.strftime("%Y-%m-%d")
        fii = "neutral"
        row = df.loc[date_obj]
        
        close = float(row["Close"])
        ema20 = float(row.get("EMA_20", close))
        ema200 = float(row.get("EMA_200", close))
        ema5 = float(row.get("EMA_5", close))
        ema9 = float(row.get("EMA_9", close))
        rsi = float(row.get("RSI_14", 50))
        stochrsi_k = float(row.get("STOCHRSIk_14_14_3_3", 50))
        stochrsi_d = float(row.get("STOCHRSId_14_14_3_3", 50))
        macd = float(row.get("MACD_12_26_9", 0))
        macd_signal = float(row.get("MACDs_12_26_9", 0))
        st_dir = row.get("SUPERTd_7_3.0", 0)
        
        import math

        
        high_val = float(row.get("High", close))

        
        if math.isnan(high_val): high_val = close

        
        low_val = float(row.get("Low", close))

        
        if math.isnan(low_val): low_val = close
        
        vix_val = float(row.get("VIX", 15.0))
        if math.isnan(vix_val): vix_val = 15.0
        
        usdinr_ret = float(row.get("USDINR_Return", 0.0))
        if math.isnan(usdinr_ret): usdinr_ret = 0.0
        
        sp500_ret = float(row.get("SP500_Return", 0.0))
        if math.isnan(sp500_ret): sp500_ret = 0.0
        
        bank_nifty_ret = float(row.get("BANK_NIFTY_Return", 0.0))
        if math.isnan(bank_nifty_ret): bank_nifty_ret = 0.0
        nifty_ret = float(row.get("Return", 0.0))
        if math.isnan(nifty_ret): nifty_ret = 0.0
        
        # Z-scores
        z_rsi = float(row.get("z_rsi", 0.0))
        z_stochrsi = float(row.get("z_stochrsi", 0.0))
        z_ema_diff = float(row.get("z_ema_diff", 0.0))
        z_price_ema = float(row.get("z_price_ema", 0.0))
        z_atr = float(row.get("z_atr", 0.0))
        z_vol = float(row.get("z_vol", 0.0))
        z_vix = float(row.get("z_vix", 0.0))
        z_bn_rel = float(row.get("z_bn_rel", 0.0))
        
        intermarket_div = "neutral"
        if nifty_ret > 0 and bank_nifty_ret < 0:
            intermarket_div = "bearish_divergence"
        elif nifty_ret < 0 and bank_nifty_ret > 0:
            intermarket_div = "bullish_divergence"


        # For historical dates, we don't have historical option chain, so we apply the live one (or None).
        # A more advanced script would only use this for the latest day.
        options_support = global_options_support if date_obj == trading_days[-1] else None
        options_resistance = global_options_resistance if date_obj == trading_days[-1] else None
        options_max_pain = global_options_max_pain if date_obj == trading_days[-1] else None
        options_momentum = global_options_momentum if date_obj == trading_days[-1] else None
        signals = {

        
            "close": round(close, 2),

        
            "high": round(high_val, 2),

        
            "low": round(low_val, 2),
            "daily_return_pct": round(row.get("Return", 0) * 100, 2),
        "options_support": options_support,
        "options_resistance": options_resistance,
        "options_max_pain": options_max_pain,
        "options_momentum": options_momentum,
        "india_vix": round(vix_val, 2),
        "vix_regime": "high_vol" if vix_val > 15 else "low_vol",
        "usdinr_trend": "bearish_for_nifty" if usdinr_ret > 0.002 else ("bullish_for_nifty" if usdinr_ret < -0.002 else "neutral"),
        "sp500_return_pct": round(sp500_ret * 100, 2),
        "sp500_cue": "bullish" if sp500_ret > 0.003 else ("bearish" if sp500_ret < -0.003 else "neutral"),
        "intermarket_divergence": intermarket_div,

        
            "ema20_signal": "bullish" if close > ema20 else "bearish",
            "ema200_signal": "bullish" if close > ema200 else "bearish",
            "ema20_diff_pct": round(((close - ema20) / ema20) * 100, 2),
            "ema200_diff_pct": round(((close - ema200) / ema200) * 100, 2),
            "ema5_signal": "bullish" if ema5 > ema9 else "bearish",
            "rsi_value": round(rsi, 2),
            "stochrsi_k": round(stochrsi_k, 2),
            "stochrsi_d": round(stochrsi_d, 2),
            "stochrsi_signal": "overbought" if stochrsi_k > 80 else ("oversold" if stochrsi_k < 20 else ("bullish crossover" if stochrsi_k > stochrsi_d else "bearish crossover")),
            "rsi_signal": "overbought" if rsi > 70 else ("oversold" if rsi < 30 else "neutral"),
            "macd_signal": "bullish" if macd > macd_signal else "bearish",
            "supertrend_signal": "bullish" if st_dir == 1 else "bearish",
            "z_rsi": round(z_rsi, 3) if not math.isnan(z_rsi) else 0.0,
            "z_stochrsi": round(z_stochrsi, 3) if not math.isnan(z_stochrsi) else 0.0,
            "z_ema_diff": round(z_ema_diff, 3) if not math.isnan(z_ema_diff) else 0.0,
            "z_price_ema": round(z_price_ema, 3) if not math.isnan(z_price_ema) else 0.0,
            "z_atr": round(z_atr, 3) if not math.isnan(z_atr) else 0.0,
            "z_vol": round(z_vol, 3) if not math.isnan(z_vol) else 0.0,
            "z_vix": round(z_vix, 3) if not math.isnan(z_vix) else 0.0,
            "z_bn_rel": round(z_bn_rel, 3) if not math.isnan(z_bn_rel) else 0.0
        }

        


        idx_long = df.index.get_loc(date_obj)
        if idx_long >= 15:
            current_returns = returns_arr[idx_long-4:idx_long+1]
            search_space = returns_arr[:idx_long - 5]
            if len(search_space) >= 5:
                hist_windows = np.lib.stride_tricks.sliding_window_view(search_space, window_shape=5)
                diffs = hist_windows - current_returns
                dists = np.sum(diffs**2, axis=1)
                best_window_idx = np.argmin(dists)
                best_match_idx = best_window_idx + 4 # because window ends at index + 4
                
                analogue_date_obj = df.index[best_match_idx]
                analogue_date_str = analogue_date_obj.strftime("%d %b %Y")
                next_day_ret = returns_arr[best_match_idx + 1]
                next_day_dir = "UP" if next_day_ret > 0 else "DOWN"
                signals["analogue_match"] = f"Similar to {analogue_date_str} (Next day went {next_day_dir})"
            else:
                signals["analogue_match"] = "No match found"
        else:
            signals["analogue_match"] = "Not enough data"


        # MC FII logic
        fii_value = None
        if date_str in mc_fii_dict:
            net_val = mc_fii_dict[date_str]
            fii_value = net_val
            if net_val > 500: fii = "buying"
            elif net_val < -500: fii = "selling"
        elif fii == "neutral":
            # Fallback to tighter gap heuristic
            try:
                if idx_long > 0:
                    prev_close = df["Close"].iloc[idx_long - 1]
                    open_price = row["Open"]
                    gap_pct = ((open_price - prev_close) / prev_close) * 100
                    change_pct = ((close - open_price) / open_price) * 100
                    if gap_pct > 0.15 and change_pct > -0.2: fii = "buying"
                    elif gap_pct < -0.15 and change_pct < 0.2: fii = "selling"
            except:
                pass

        # Legacy VIX block removed as it is handled historically now
        data[date_str] = {
            "fii": fii,
            "fii_value": fii_value,
            "signals": signals
        }
        
    if symbol_name == "nifty":
        # Replace WFO block in export_dashboard.py
        print("Running Phase 3B Walk-Forward Optimizer to find current best features...")
        try:
            import itertools
            df_wfo = df.copy()
            # z_atr is intentionally excluded: it has no BTST checkbox and no
            # calculateTopK branch, so it could only ever be selected and then
            # misreported (it was aliased to chk-stochrsi in the UI).
            available_features = ['z_rsi', 'z_stochrsi', 'z_ema_diff', 'z_price_ema', 'z_vol']
            df_wfo.dropna(subset=available_features, inplace=True)
            
            feature_matrix = df_wfo[available_features].values
            returns_wfo = df_wfo['Return'].values
            
            n_days = len(feature_matrix)
            # Daily/BTST sampling is ~1 observation per trading day, so the old
            # 30+30 split gave the optimiser ~30 points and it almost never
            # cleared the 5-trade floor. Use 1y validation + 6m OOS instead.
            val_window = 252
            oos_window = 126
            eval_window = val_window + oos_window
            # Days of embargo excluded from the analogue search so the match set
            # never contains the target day or its immediate future.
            embargo = 5
            print(f"WFO n_days after dropna: {n_days}")
            print(f"WFO window: val={val_window} oos={oos_window} (features={len(available_features)})")
            
            if n_days > val_window + oos_window:
                test_combs = []
                for r in range(2, 6): # Min 2, max 5 features
                    test_combs.extend(list(itertools.combinations(range(len(available_features)), r)))
                
                # Precompute predictions across the whole eval window so the
                # rolling val and OOS passes never hit a missing day.
                predictions = {c_idx: {} for c_idx in range(len(test_combs))}
                for c_idx, comb in enumerate(test_combs):
                    mat = feature_matrix[:, comb]
                    for i in range(n_days - eval_window - 1, n_days - 1):
                        target_vec = mat[i]
                        search_mat = mat[:i-embargo]
                        if len(search_mat) < 50: continue
                            
                        diffs = search_mat - target_vec
                        dists = np.sum(diffs**2, axis=1)
                        top_n = min(50, len(dists))
                        top_idx = np.argpartition(dists, top_n)[:top_n]
                            
                        next_rets = returns_wfo[top_idx + 1]
                        up_count = np.sum(next_rets > 0)
                        prob_up = (up_count / top_n) * 100
                        actual_next_ret = returns_wfo[i + 1]
                        predictions[c_idx][i] = (prob_up, actual_next_ret)
        
                def calc_edge(c_idx, start_i, end_i):
                    win_count, total_trades, cumulative_ret = 0, 0, 0
                    for i in range(start_i, end_i):
                        if i not in predictions[c_idx]: continue
                        prob_up, actual_ret = predictions[c_idx][i]
                        if prob_up >= 55:
                            total_trades += 1
                            if actual_ret > 0: win_count += 1
                            cumulative_ret += actual_ret
                        elif prob_up <= 45:
                            total_trades += 1
                            if actual_ret < 0: win_count += 1
                            cumulative_ret -= actual_ret
                            
                    if total_trades < 5: return -999, 0, 0, 0
                    win_rate = win_count / total_trades
                    avg_ret = cumulative_ret / total_trades
                    edge = cumulative_ret * win_rate
                    # Complexity Penalty: 0.01% per feature
                    adj_edge = edge - (len(test_combs[c_idx]) * 0.01)
                    return adj_edge, total_trades, win_rate, avg_ret
        
                oos_start = n_days - oos_window - 1
                oos_end = n_days - 1
                
                oos_trades, oos_wins, oos_cum_ret = 0, 0, 0
                selected_combs_freq = {c_idx: 0 for c_idx in range(len(test_combs))}
                
                # Rolling OOS Loop
                for today_i in range(oos_start, oos_end):
                    val_start = today_i - val_window
                    val_end = today_i
                    
                    best_val_edge, best_c_idx = -999, None
                    for c_idx in range(len(test_combs)):
                        adj_edge, _, _, _ = calc_edge(c_idx, val_start, val_end)
                        if adj_edge > best_val_edge:
                            best_val_edge = adj_edge
                            best_c_idx = c_idx
                            
                    if best_c_idx is not None and best_val_edge > 0:
                        selected_combs_freq[best_c_idx] += 1
                        prob_up, actual_ret = predictions[best_c_idx].get(today_i, (50, 0))
                        if prob_up >= 55:
                            oos_trades += 1
                            if actual_ret > 0: oos_wins += 1
                            oos_cum_ret += actual_ret
                        elif prob_up <= 45:
                            oos_trades += 1
                            if actual_ret < 0: oos_wins += 1
                            oos_cum_ret -= actual_ret
        
                oos_win_rate = (oos_wins / oos_trades) if oos_trades > 0 else 0
                oos_avg_ret = (oos_cum_ret / oos_trades) if oos_trades > 0 else 0
                oos_edge = oos_avg_ret * oos_win_rate * oos_trades # Total Edge
        
                # Final Model Selection for Tomorrow
                best_final_edge, best_final_c_idx, best_final_stats = -999, None, None
                for c_idx in range(len(test_combs)):
                    adj_edge, t, wr, ar = calc_edge(c_idx, oos_start, oos_end) # Val is the OOS period
                    if adj_edge > best_final_edge:
                        best_final_edge = adj_edge
                        best_final_c_idx = c_idx
                        best_final_stats = (t, wr, ar)
                
                ui_mapping = {
                    'z_rsi': 'chk-rsi', 'z_stochrsi': 'chk-stochrsi',
                    'z_ema_diff': 'chk-ema59', 'z_price_ema': 'chk-ema20',
                    'z_vol': 'chk-deltaoi', 'z_vix': 'chk-vix', 'z_bn_rel': 'chk-divergence'
                }
                
                stability = {}
                feature_freq = {f: 0 for f in available_features}
                for c_idx, count in selected_combs_freq.items():
                    if count > 0:
                        for f_idx in test_combs[c_idx]:
                            feature_freq[available_features[f_idx]] += count
                for f, freq in feature_freq.items():
                    # Only mapped features. Previously an unmapped f leaked its
                    # raw name (e.g. "z_atr") into _meta, which the UI then
                    # displayed as though it were a selected checkbox id.
                    if f in ui_mapping:
                        stability[ui_mapping[f]] = round((freq / oos_window) * 100, 1)
        
                ui_boxes = None
                if best_final_c_idx is not None:
                    best_comb = [available_features[idx] for idx in test_combs[best_final_c_idx]]
                    ui_boxes = [ui_mapping[f] for f in best_comb if f in ui_mapping]

                data["_meta"] = build_wfo_meta(
                    best_final_edge=best_final_edge,
                    best_final_stats=best_final_stats,
                    oos_edge=oos_edge,
                    oos_win_rate=oos_win_rate,
                    oos_trades=oos_trades,
                    stability=stability,
                    val_window=val_window,
                    oos_window=oos_window,
                    ui_boxes=ui_boxes,
                )
                if data["_meta"]["status"] == "SIGNAL":
                    print(f"WFO Phase 3B Optimal Set: {ui_boxes}")
                else:
                    print(f"WFO Phase 3B: NO SIGNAL detected ({data['_meta']['reason']}).")
                    
        except Exception as e:
            print("WFO Phase 3B Failed:", e)
            
    with open(data_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[{symbol_name}] Successfully updated {data_file}")

def main():
    # Process multiple instruments
    instruments = [
        ("nifty", "nifty.csv"),
        ("banknifty", "banknifty.csv"),
        ("reliance", "reliance.csv")
    ]
    
    for sym, filename in instruments:
        print(f"\n--- Processing {sym.upper()} ---")
        process_symbol(sym, filename)

if __name__ == "__main__":
    main()
