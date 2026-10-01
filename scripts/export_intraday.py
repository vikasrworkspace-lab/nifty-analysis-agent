import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_ta as ta
import math
import itertools
from collections import defaultdict

import core.settings as settings_mod
from core import session as session_mod

class IntradayWFO:
    """Walk-Forward Optimizer for intraday bar-level feature selection.

    Finalized spec: Train 252+ bars → Validation 300 bars → OOS 300 bars,
    test 1–5 feature combinations, complexity penalty, OOS edge gate,
    feature stability tracking, NO SIGNAL if OOS edge isn't positive.
    """

    def __init__(self, df, all_features, k=50, val_size=300, oos_size=300, train_size=252):
        self.df = df
        self.all_features = all_features
        self.k = k
        self.val_size = val_size
        self.oos_size = oos_size
        self.train_size = train_size

        # 1–5 feature combinations (finalized spec)
        self.feature_combinations = []
        for r in range(1, min(6, len(all_features) + 1)):
            self.feature_combinations.extend(list(itertools.combinations(all_features, r)))

    def _session_start_positions(self):
        """Bar positions that open a new trading session.

        ``Forward_Return`` is measured to each bar's own session close, so a
        fold boundary landing mid-session lets the last validation bar's label
        reach into the OOS window -- the same class of leak as intraday
        analogues drawn from the target's own session. Snapping fold edges to
        session starts keeps train/validation/OOS label horizons disjoint.
        """
        day = self.df.index.normalize()
        is_start = np.empty(len(day), dtype=bool)
        is_start[0] = True
        is_start[1:] = day[1:].to_numpy() != day[:-1].to_numpy()
        return np.flatnonzero(is_start)

    def _snap_forward(self, starts, pos):
        """First session start at or after ``pos``."""
        i = np.searchsorted(starts, pos)
        return int(starts[i]) if i < len(starts) else len(self.df)

    def _snap_back(self, starts, pos):
        """Last session start at or before ``pos``."""
        i = np.searchsorted(starts, pos, side="right") - 1
        return int(starts[i]) if i >= 0 else 0

    def _evaluate_combo(self, X_train, y_train_ret, X_eval, y_eval_ret, features):
        """Score a feature combo: nearest-analogue prediction on eval window."""
        feat_list = list(features)
        X_tr = X_train[feat_list].values
        X_ev = X_eval[feat_list].values
        y_tr_r = y_train_ret.values
        y_ev_r = y_eval_ret.values

        # Vectorised pairwise squared-distance matrix
        dist = np.sum((X_ev[:, np.newaxis, :] - X_tr[np.newaxis, :, :]) ** 2, axis=2)

        edge_sum = 0.0
        trades = 0
        wins = 0

        k = min(self.k, len(X_tr))
        for i in range(len(X_ev)):
            # kth must be a valid index, so partition at k-1 rather than k.
            # When the candidate pool holds exactly k rows, k is out of range and
            # numpy raises "kth(=k) out of bounds (k)". k is already clamped to
            # the pool above, so k-1 is valid for every pool size and [:k] still
            # selects the k nearest analogues.
            nearest_idx = np.argpartition(dist[i], k - 1)[:k]
            nearest_returns = y_tr_r[nearest_idx]

            pred_return = np.mean(nearest_returns)
            actual_return = y_ev_r[i]

            if pred_return > 0.02:       # Long signal
                edge_sum += actual_return
                trades += 1
                if actual_return > 0:
                    wins += 1
            elif pred_return < -0.02:     # Short signal
                edge_sum -= actual_return
                trades += 1
                if actual_return < 0:
                    wins += 1

        # Complexity penalty: penalise higher-dimensional combos (-0.01% per feature)
        # Note: Since edge_sum is in percentage points (e.g. 0.44 means 0.44%), 0.01% is 0.01
        complexity_penalty = len(features) * 0.01
        score = (edge_sum - complexity_penalty) if trades > 0 else -999.0
        return score, edge_sum, trades, wins

    def run(self):
        total_bars = len(self.df)
        min_required = self.train_size + self.val_size + self.oos_size
        if total_bars < min_required:
            print(f"  WFO: not enough bars ({total_bars} < {min_required}), skipping.")
            return None

        optimal_history = []
        oos_edge_total = 0.0
        oos_trades_total = 0
        oos_wins_total = 0
        step_size = self.oos_size

        session_starts = self._session_start_positions()
        # OOS folds are advanced by whole OOS blocks, so consecutive OOS
        # windows never overlap. Each start is snapped forward to a session
        # boundary and the train/validation edge is snapped back to one, so no
        # validation label can extend into the OOS window.
        cursor = self.train_size + self.val_size
        while cursor < total_bars:
            start_oos = self._snap_forward(session_starts, cursor)
            if start_oos >= total_bars:
                break
            end_oos = min(start_oos + step_size, total_bars)
            start_val = self._snap_back(session_starts, start_oos - self.val_size)
            cursor = end_oos
            if start_val <= 0:
                continue

            # --- Phase 1: find best combo on Validation window ---
            X_train = self.df.iloc[:start_val]
            y_train_ret = self.df['Forward_Return'].iloc[:start_val]

            X_val = self.df.iloc[start_val:start_oos]
            y_val_ret = self.df['Forward_Return'].iloc[start_val:start_oos]

            best_val_score = -9999
            best_combo = None

            for combo in self.feature_combinations:
                score, _, _, _ = self._evaluate_combo(
                    X_train, y_train_ret, X_val, y_val_ret, combo
                )
                if score > best_val_score:
                    best_val_score = score
                    best_combo = combo

            # Only proceed if val edge is positive
            if best_combo is None or best_val_score <= 0:
                continue

            optimal_history.append(best_combo)

            # --- Phase 2: OOS evaluation with val-selected combo ---
            # Train for OOS includes everything up to OOS start (train + val)
            X_train_oos = self.df.iloc[:start_oos]
            y_train_oos_ret = self.df['Forward_Return'].iloc[:start_oos]

            X_oos = self.df.iloc[start_oos:end_oos]
            y_oos_ret = self.df['Forward_Return'].iloc[start_oos:end_oos]

            _, oos_edge, oos_t, oos_w = self._evaluate_combo(
                X_train_oos, y_train_oos_ret, X_oos, y_oos_ret, best_combo
            )
            oos_edge_total += oos_edge
            oos_trades_total += oos_t
            oos_wins_total += oos_w

        # --- Feature stability across all WFO folds ---
        stability = defaultdict(int)
        for combo in optimal_history:
            for f in combo:
                stability[f] += 1
        n_folds = max(len(optimal_history), 1)
        stability_pct = {k: round((v / n_folds) * 100) for k, v in stability.items()}

        # --- NO SIGNAL gate: OOS edge must be positive ---
        if not optimal_history or oos_edge_total <= 0:
            return {
                "wfo_optimal_features": [],
                "stability": stability_pct,
                "oos_edge": round(oos_edge_total, 4),
                "oos_win_rate": round((oos_wins_total / oos_trades_total) * 100, 1) if oos_trades_total > 0 else 0,
                "oos_trades": oos_trades_total,
                "status": "NO SIGNAL"
            }

        latest_combo = optimal_history[-1]

        return {
            "wfo_optimal_features": list(latest_combo),
            "stability": stability_pct,
            "oos_edge": round(oos_edge_total, 4),
            "oos_win_rate": round((oos_wins_total / oos_trades_total) * 100, 1) if oos_trades_total > 0 else 0,
            "oos_trades": oos_trades_total,
            "status": "ACTIVE"
        }

_RESAMPLE_COLS = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}


def resample_session(df: pd.DataFrame, minutes: int, settings: dict) -> pd.DataFrame:
    """Session-anchored OHLCV frame at the requested timeframe.

    Delegates to core.session.resample_ohlcv so both intraday invariants hold:
    bins anchor at the session open (09:15) and truncated end-of-session bins
    are dropped. A default-origin resample instead lands 30m/60m bars on the
    :00/:30 midnight grid, which does not line up with a 09:15 session open and
    leaves a short final bar at the close that would contaminate the features.
    """
    if minutes == 5:
        return df.copy()

    lower = df.rename(columns=str.lower)
    out = session_mod.resample_ohlcv(lower, minutes, settings)
    out = out.drop(columns=["_session"], errors="ignore")
    return out.rename(columns=_RESAMPLE_COLS)


def process_intraday_symbol(symbol_name, db_filename):
    print(f"[{symbol_name}] Starting Intraday Engine Processing...")
    
    from pathlib import Path
    import pytz
    IST = pytz.timezone("Asia/Kolkata")

    csv_path = f"data/fyers_db/5/{db_filename}"
    df = None
    
    # 1. Load Fyers DB if present
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df.columns = [c.lower() for c in df.columns]
        except Exception as e:
            print(f"[{symbol_name}] Error reading {csv_path}: {e}")
            
    # 2. Check for yfinance archive data (e.g. data/intraday/5m/nifty.csv or bank_nifty.csv)
    yf_name_map = {
        "nifty": "nifty",
        "banknifty": "bank_nifty",
        "reliance": "reliance"
    }
    yf_filename = yf_name_map.get(symbol_name, symbol_name)
    yf_path = f"data/intraday/5m/{yf_filename}.csv"
    
    if os.path.exists(yf_path):
        try:
            yf_df = pd.read_csv(yf_path, index_col=0, parse_dates=True)
            yf_df.columns = [c.lower() for c in yf_df.columns]
            
            if df is not None and not df.empty:
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC").tz_convert(IST)
                else:
                    df.index = df.index.tz_convert(IST)
                if yf_df.index.tz is None:
                    yf_df.index = yf_df.index.tz_localize("UTC").tz_convert(IST)
                else:
                    yf_df.index = yf_df.index.tz_convert(IST)
                    
                df = pd.concat([df, yf_df])
                df = df[~df.index.duplicated(keep="last")].sort_index()
                Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csv_path)
            else:
                df = yf_df
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC").tz_convert(IST)
                else:
                    df.index = df.index.tz_convert(IST)
                Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csv_path)
        except Exception as e:
            print(f"[{symbol_name}] Error merging yfinance archive: {e}")
            
    if df is None or df.empty:
        print(f"[{symbol_name}] No intraday data available for {symbol_name}.")
        return

    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})

    settings = settings_mod.load_settings()

    # Drop pre-open / closing-auction rows. The Fyers 5m archive carries
    # 09:05 and 09:10 bars, so a raw pass-through would otherwise see 77 bars
    # on some sessions instead of 75. Same filter the archive loader applies,
    # so both intraday pipelines share one definition of the session.
    df = session_mod.filter_session(df, settings)

    # Cap the RAW 5m window once, BEFORE any resampling, so every timeframe
    # (5/15/30/60m) covers the same calendar span. Capping each timeframe
    # independently would silently give them different histories.
    # 5000 bars ~= 67 trading days; enough for multiple WFO folds at 5m/15m.
    df = session_mod.cap_raw_bars(df, settings)

    timeframes = {
        "5": 5,
        "15": 15,
        "30": 30,
        "60": 60
    }
    
    # We only process the last 3 unique trading days for the UI JSON payload
    unique_dates = np.unique(df.index.date)
    if len(unique_dates) < 1:
        return
    last_3_days = unique_dates[-3:]
    
    for tf_str, minutes in timeframes.items():
        data_file = f"intraday_{tf_str}_{symbol_name}.json"

        tf_df = resample_session(df, minutes, settings)
        if tf_df.empty:
            print(f"[{symbol_name}] {tf_str}m resample produced no bars, skipping.")
            continue
            
        tf_df.ta.ema(length=200, append=True)
        tf_df.ta.ema(length=20, append=True)
        tf_df.ta.ema(length=5, append=True)
        tf_df.ta.ema(length=9, append=True)
        tf_df.ta.rsi(length=14, append=True)
        tf_df.ta.stochrsi(length=14, rsi_length=14, k=3, d=3, append=True)
        tf_df.ta.atr(length=14, append=True)
        tf_df.ta.macd(fast=12, slow=26, signal=9, append=True)
        tf_df.ta.supertrend(length=10, multiplier=3, append=True)
        
        tf_df['EMA5_9_diff'] = tf_df['EMA_5'] - tf_df['EMA_9']
        tf_df['Price_20EMA_diff'] = tf_df['Close'] - tf_df['EMA_20']
        tf_df['Vol_Ratio'] = tf_df['Volume'] / tf_df['Volume'].rolling(20).mean().replace(0, 1)
        
        # Calculate Forward Return (rest of session)
        tf_df['Date'] = tf_df.index.date
        tf_df['Session_Close'] = tf_df.groupby('Date')['Close'].transform('last')
        tf_df['Forward_Return'] = (tf_df['Session_Close'] - tf_df['Close']) / tf_df['Close'] * 100

        # Daily return = this bar's close vs the previous session's close. Not
        # bar open->close, which understated the day's move and made the Intraday
        # tab disagree with the daily/BTST tabs.
        session_closes = tf_df.groupby('Date')['Close'].last()
        tf_df['Prev_Session_Close'] = tf_df['Date'].map(session_closes.shift(1))
        tf_df['Daily_Return'] = (
            (tf_df['Close'] - tf_df['Prev_Session_Close'])
            / tf_df['Prev_Session_Close'] * 100
        )
        
        # Calculate Forward Max Up (MFE) and Max Down (MAE) for the rest of the session
        reversed_df = tf_df.iloc[::-1]
        tf_df['Session_Max_High'] = reversed_df.groupby('Date')['High'].cummax().iloc[::-1]
        tf_df['Session_Min_Low'] = reversed_df.groupby('Date')['Low'].cummin().iloc[::-1]
        
        tf_df['Session_Max_High_Next'] = tf_df.groupby('Date')['Session_Max_High'].shift(-1)
        tf_df['Session_Min_Low_Next'] = tf_df.groupby('Date')['Session_Min_Low'].shift(-1)
        
        tf_df['Forward_Max_Up'] = (tf_df['Session_Max_High_Next'] - tf_df['Close']) / tf_df['Close'] * 100
        tf_df['Forward_Max_Down'] = (tf_df['Session_Min_Low_Next'] - tf_df['Close']) / tf_df['Close'] * 100
        tf_df['Forward_Max_Up'] = tf_df['Forward_Max_Up'].fillna(0)
        tf_df['Forward_Max_Down'] = tf_df['Forward_Max_Down'].fillna(0)
        
        rolling_window = 252 # About ~3.5 days of 5m bars
        if minutes != 5:
            rolling_window = max(20, 252 // (minutes // 5))
            
        features_map = {
            'RSI_14': 'z_rsi',
            'STOCHRSIk_14_14_3_3': 'z_stochrsi',
            'EMA5_9_diff': 'z_ema_diff',
            'Price_20EMA_diff': 'z_price_ema',
            'ATRr_14': 'z_atr',
            'Vol_Ratio': 'z_vol'
        }
        
        for raw, z in features_map.items():
            if raw in tf_df.columns:
                r_mean = tf_df[raw].rolling(rolling_window).mean()
                r_std = tf_df[raw].rolling(rolling_window).std().replace(0, 1e-5)
                tf_df[z] = (tf_df[raw] - r_mean) / r_std
                
        # Drop rows with NaN features
        tf_df = tf_df.dropna(subset=list(features_map.values()) + ['Forward_Return'])
        
        # Mask each session's own final bar from the WFO target: Forward_Return
        # is 0 there by construction, so those rows carry no signal. Derived
        # per day rather than from a single global clock time, so a partial
        # latest session cannot shift which bar is excluded.
        tf_df['Time'] = tf_df.index.time
        session_last = tf_df.groupby('Date')['Time'].transform('last')
        valid_wfo_bars = tf_df[tf_df['Time'] != session_last].copy()
        valid_wfo_bars['y_target'] = (valid_wfo_bars['Forward_Return'] > 0).astype(int)
        
        data = {}
        
        # Run Intraday WFO with features that map directly to UI checkboxes
        wfo_features = ['z_rsi', 'z_stochrsi', 'z_ema_diff', 'z_price_ema', 'z_vol']
        total_valid = len(valid_wfo_bars)
        
        # Dynamically scale WFO windows if we don't have enough history for the full 852-bar spec
        if total_valid >= 100:
            if total_valid > 252 + 300 + 300:
                t_size, v_size, o_size = 252, 300, 300
            else:
                v_size = int(total_valid * 0.25)
                o_size = int(total_valid * 0.25)
                t_size = total_valid - v_size - o_size
                
            wfo_engine = IntradayWFO(valid_wfo_bars, wfo_features, k=50, val_size=v_size, oos_size=o_size, train_size=t_size)
            meta = wfo_engine.run()
            if meta:
                data["_meta"] = meta
        
        # UI Payload: Include the last 15 days for robust analogue matching
        last_days = unique_dates[-15:] if len(unique_dates) >= 15 else unique_dates
        ui_df = tf_df[tf_df.index.date >= last_days[0]]
        last_bar = ui_df.index[-1] if len(ui_df) else None
        
        for dt, row in ui_df.iterrows():
            date_str = dt.isoformat()
            
            c = float(row["Close"])
            o = float(row["Open"])
            e200 = float(row["EMA_200"]) if "EMA_200" in row and not pd.isna(row["EMA_200"]) else c
            e20 = float(row["EMA_20"]) if "EMA_20" in row and not pd.isna(row["EMA_20"]) else c
            e5 = float(row["EMA_5"]) if "EMA_5" in row and not pd.isna(row["EMA_5"]) else c
            e9 = float(row["EMA_9"]) if "EMA_9" in row and not pd.isna(row["EMA_9"]) else c
            rsi = float(row["RSI_14"]) if "RSI_14" in row and not pd.isna(row["RSI_14"]) else 50.0
            stoch_k = float(row["STOCHRSIk_14_14_3_3"]) if "STOCHRSIk_14_14_3_3" in row and not pd.isna(row["STOCHRSIk_14_14_3_3"]) else 50.0
            stoch_d = float(row["STOCHRSId_14_14_3_3"]) if "STOCHRSId_14_14_3_3" in row and not pd.isna(row["STOCHRSId_14_14_3_3"]) else 50.0
            macd_val = float(row["MACD_12_26_9"]) if "MACD_12_26_9" in row and not pd.isna(row["MACD_12_26_9"]) else 0.0
            macd_sig = float(row["MACDs_12_26_9"]) if "MACDs_12_26_9" in row and not pd.isna(row["MACDs_12_26_9"]) else 0.0
            st_dir = float(row["SUPERTd_10_3"]) if "SUPERTd_10_3" in row and not pd.isna(row["SUPERTd_10_3"]) else 1.0

            signals = {
                "open": round(o, 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(c, 2),
                "volume": float(row["Volume"]),
                "daily_return_pct": round(float(row["Daily_Return"]), 2) if pd.notna(row.get("Daily_Return")) else 0.0,
                "forward_return_pct": round(float(row["Forward_Return"]), 2),
                "forward_max_up_pct": round(float(row.get("Forward_Max_Up", 0.0)), 2),
                "forward_max_down_pct": round(float(row.get("Forward_Max_Down", 0.0)), 2),
                "z_rsi": round(float(row.get("z_rsi", 0.0)), 3),
                "z_stochrsi": round(float(row.get("z_stochrsi", 0.0)), 3),
                "z_ema_diff": round(float(row.get("z_ema_diff", 0.0)), 3),
                "z_price_ema": round(float(row.get("z_price_ema", 0.0)), 3),
                "z_atr": round(float(row.get("z_atr", 0.0)), 3),
                "z_vol": round(float(row.get("z_vol", 0.0)), 3),
                "z_vix": 0.0,
                "z_bn_rel": 0.0,
                "ema200_signal": "bullish" if c >= e200 else "bearish",
                "ema200_diff_pct": round(((c - e200) / e200) * 100, 2) if e200 else 0.0,
                "ema20_signal": "bullish" if c >= e20 else "bearish",
                "ema20_diff_pct": round(((c - e20) / e20) * 100, 2) if e20 else 0.0,
                "supertrend_signal": "bullish" if st_dir == 1 else "bearish",
                "rsi_value": round(rsi, 1),
                "rsi_signal": "overbought" if rsi >= 70 else ("oversold" if rsi <= 30 else "neutral"),
                "macd_signal": "bullish" if macd_val >= macd_sig else "bearish",
                "ema5_signal": "bullish" if e5 >= e9 else "bearish",
                "stochrsi_k": round(stoch_k, 1),
                "stochrsi_signal": "overbought" if stoch_k >= 80 else ("oversold" if stoch_k <= 20 else ("bullish crossover" if stoch_k >= stoch_d else "bearish crossover")),
                "india_vix": 12.5,
                "options_max_pain": round(c / 50) * 50
            }
            data[date_str] = {"signals": signals}
            
        # Freshness travels in its own block rather than inside _meta: _meta is
        # the WFO contract and is asserted field-by-field in tests, so it is not
        # the place to bolt on unrelated keys.
        data["_freshness"] = freshness_block(symbol_name, tf_str, last_bar)

        with open(data_file, "w") as f:
            json.dump(data, f)
            
        print(f"[{symbol_name}] Exported {data_file} ({tf_str}m) with TRUE INTRADAY features and WFO.")

def freshness_block(symbol_name: str, tf_str: str, last_bar) -> dict:
    """The ``_freshness`` block stamped into an intraday payload.

    Its own block rather than a key inside ``_meta``, which is the WFO contract
    and is asserted field-by-field in tests. The UI reads this to warn that the
    feed has stopped -- the one failure mode a skipped morning login causes, and
    the one the user would otherwise only notice by eye.
    """
    return {
        "symbol": symbol_name,
        "timeframe": f"{tf_str}m",
        "last_bar_ts": last_bar.isoformat() if last_bar is not None else None,
        "generated_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
    }


def main():
    instruments = [
        ("nifty", "NSE_NIFTY50-INDEX.csv"),
        ("banknifty", "NSE_NIFTYBANK-INDEX.csv"),
        ("reliance", "NSE_RELIANCE-EQ.csv")
    ]
    
    for sym, filename in instruments:
        process_intraday_symbol(sym, filename)

if __name__ == "__main__":
    main()
