import os
import time
import alpaca_trade_api as tradeapi
import pandas as pd
from dotenv import load_dotenv
from strategy import compute_signal_and_score
from logger_config import get_logger

load_dotenv()
logger = get_logger()

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
DATA_API_URL = os.getenv("DATA_API_URL", "https://data.alpaca.markets")

# NEW: Minimum volume for screening
MIN_VOLUME_FOR_SCREENING = int(os.getenv("MIN_VOLUME_FOR_SCREENING", "100000"))

data_api = tradeapi.REST(API_KEY, API_SECRET, DATA_API_URL)

# Alpaca Data API rate limit management
ALPACA_DATA_API_LAST_CALL_TIME = 0
ALPACA_DATA_API_RATE_LIMIT_SECONDS = 0.5 # Alpaca free tier: 200 req/min = 1 req every 0.3 seconds. Use 0.5s for safety.

def enforce_alpaca_data_rate_limit():
    global ALPACA_DATA_API_LAST_CALL_TIME
    elapsed = time.time() - ALPACA_DATA_API_LAST_CALL_TIME
    if elapsed < ALPACA_DATA_API_RATE_LIMIT_SECONDS:
        sleep_time = ALPACA_DATA_API_RATE_LIMIT_SECONDS - elapsed
        # logger.debug(f"Alpaca Data API rate limit hit. Sleeping for {sleep_time:.2f} seconds.")
        time.sleep(sleep_time)
    ALPACA_DATA_API_LAST_CALL_TIME = time.time()

def get_latest_df_for_screener(symbol, limit=60):
    """Retrieves and formats minute-based stock data from Alpaca for screening."""
    enforce_alpaca_data_rate_limit() # Enforce rate limit before each call
    try:
        bars = data_api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=limit).df
        if bars.empty:
            return None
        
        bars.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return bars
    except Exception:
        return None

def screen_for_opportunities(symbols_to_scan):
    """
    Screens for potential trading opportunities from a list of symbols
    and returns a sorted list of the best candidates based on technical score.
    """
    logger.info(f"Starting market screener for {len(symbols_to_scan)} symbols...")
    opportunities = []

    for symbol in symbols_to_scan:
        df = get_latest_df_for_screener(symbol) # Use the rate-limited data fetcher
        if df is None or df.empty or len(df) < 20: # Ensure enough data for SMA20
            continue
        
        # NEW: Volume confirmation check
        if df['Volume'].iloc[-1] < MIN_VOLUME_FOR_SCREENING:
            logger.debug(f"Screening: {symbol} skipped due to low volume ({df['Volume'].iloc[-1]})")
            continue

        signal, score = compute_signal_and_score(df)
        
        # Only consider buy signals for initial screening
        if signal == 1 and score > 0:
            last_price = df['Close'].iloc[-1]
            opportunities.append({
                'symbol': symbol,
                'signal': signal,
                'score': score,
                'last_price': last_price
            })
            # logger.debug(f"Screening: {symbol} scored {score:.4f}") # Too verbose for many stocks

    # Sort opportunities by score in descending order
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    
    return opportunities

