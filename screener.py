import os
import time
import alpaca_trade_api as tradeapi
import pandas as pd
from dotenv import load_dotenv
from strategy import compute_signal_and_score
from logger_config import get_logger

load_dotenv()
logger = get_logger()

# Alpaca API configuration
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
DATA_API_URL = os.getenv("DATA_API_URL", "https://data.alpaca.markets")

data_api = tradeapi.REST(API_KEY, API_SECRET, DATA_API_URL)

def get_latest_df(symbol, limit=390):
    """Retrieves and formats minute-based stock data from Alpaca."""
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
    and returns a sorted list of the best candidates.
    """
    logger.info("Starting market screener...")
    opportunities = []

    for symbol in symbols_to_scan:
        df = get_latest_df(symbol)
        if df is None or df.empty:
            continue
        
        signal, score = compute_signal_and_score(df)
        
        if signal != -1 and score > 0:
            last_price = df['Close'].iloc[-1]
            opportunities.append({
                'symbol': symbol,
                'signal': signal,
                'score': score,
                'last_price': last_price
            })
            logger.info(f"Found opportunity for {symbol} with score {score:.4f}.")

    # Sort opportunities by score in descending order
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    
    return opportunities

