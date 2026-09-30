import os
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

from core.fyers_auth import resolve_access_token

# Setup timezone
import pytz
IST = pytz.timezone("Asia/Kolkata")

def get_fyers_client():
    """Initializes and returns the Fyers API client."""
    load_dotenv()
    client_id = os.getenv("FYERS_APP_ID")
    access_token = resolve_access_token()
    return fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")

def get_db_path(symbol: str, resolution: str, settings: dict = None) -> Path:
    """Returns the local CSV database path for the given symbol and timeframe.

    The root comes from settings (``paths.fyers_db_dir``) so there is a single
    source of truth shared with the cloud cache layer.
    """
    import core.settings as settings_mod

    if settings is None:
        settings = settings_mod.load_settings()
    # Clean symbol for filename (e.g. NSE:NIFTY50-INDEX -> NSE_NIFTY50-INDEX)
    safe_sym = symbol.replace(":", "_")
    db_dir = settings_mod.rel(settings, "fyers_db_dir") / resolution
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / f"{safe_sym}.csv"

def fetch_chunk(fyers, symbol: str, resolution: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetches a specific date range chunk from Fyers."""
    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": start_date.strftime("%Y-%m-%d"),
        "range_to": end_date.strftime("%Y-%m-%d"),
        "cont_flag": "1"
    }
    
    response = fyers.history(data=data)
    if response.get("s") != "ok" or "candles" not in response or not response["candles"]:
        return pd.DataFrame()
        
    candles = response["candles"]
    df = pd.DataFrame(candles, columns=["epoch", "open", "high", "low", "close", "volume"])
    
    # Convert epochs to tz-aware IST datetime index
    df["ts"] = pd.to_datetime(df["epoch"], unit="s")
    df = df.set_index("ts").drop(columns=["epoch"])
    df.index = df.index.tz_localize("UTC").tz_convert(IST)
    
    return df

def build_historical_database(symbol: str, resolution: str, days_back: int = 300, settings: dict = None):
    """
    Builds or updates the local historical database for a given instrument.
    Fyers allows max 100 days per request for intraday. We chunk backwards.
    """
    fyers = get_fyers_client()
    db_path = get_db_path(symbol, resolution, settings)
    
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days_back)
    
    # Fyers API max chunk size is 100 days for intraday data
    CHUNK_DAYS = 90
    
    all_chunks = []
    
    # If the DB already exists, load it to avoid re-fetching everything
    existing_df = pd.DataFrame()
    if db_path.exists():
        existing_df = pd.read_csv(db_path, index_col=0, parse_dates=True)
        print(f"[{symbol} | {resolution}] Found existing DB with {len(existing_df)} rows.")
        # Only fetch data from the last available timestamp
        if not existing_df.empty:
            last_ts = existing_df.index[-1]
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize(IST)
            # Fetch from 2 days before the last timestamp just to be safe (overlapping)
            start_dt = last_ts.replace(tzinfo=None) - timedelta(days=2)
            
    print(f"[{symbol} | {resolution}] Fetching data from {start_dt.date()} to {end_dt.date()}...")
    
    current_end = end_dt
    while current_end > start_dt:
        current_start = max(start_dt, current_end - timedelta(days=CHUNK_DAYS))
        
        print(f"   -> Fetching chunk: {current_start.date()} to {current_end.date()}")
        chunk_df = fetch_chunk(fyers, symbol, resolution, current_start, current_end)
        
        if not chunk_df.empty:
            all_chunks.append(chunk_df)
            
        current_end = current_start - timedelta(days=1)
        time.sleep(0.5) # Prevent rate limiting
        
    if not all_chunks and existing_df.empty:
        print(f"[{symbol} | {resolution}] No data found at all.")
        return
        
    if all_chunks:
        new_data = pd.concat(all_chunks)
        if not existing_df.empty:
            # Combine existing DB with new chunks
            final_df = pd.concat([existing_df, new_data])
        else:
            final_df = new_data
            
        # Deduplicate and sort
        final_df = final_df[~final_df.index.duplicated(keep="last")].sort_index()
        
        # Save to Local DB
        final_df.to_csv(db_path)
        print(f"[{symbol} | {resolution}] DB Updated! Total rows: {len(final_df)}.")
    else:
        print(f"[{symbol} | {resolution}] Database is already entirely up to date.")

if __name__ == "__main__":
    # Test building the database for multiple timeframes
    symbol = "NSE:NIFTY50-INDEX"
    print("--- BUILDING LOCAL FYERS DATABASE ---")
    build_historical_database(symbol, "5", days_back=200)   # 5-minute data (last 200 days)
    build_historical_database(symbol, "15", days_back=200)  # 15-minute data
    build_historical_database(symbol, "60", days_back=200)  # Hourly data
    build_historical_database(symbol, "1D", days_back=1000) # Daily data (last ~3 years)
