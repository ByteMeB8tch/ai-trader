import os
import time
import alpaca_trade_api as tradeapi
import pandas as pd
from dotenv import load_dotenv
from strategy import compute_signal_and_score, add_indicators
from logger_config import get_logger

load_dotenv()
logger = get_logger()

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
DATA_API_URL = os.getenv("DATA_API_URL", "https://data.alpaca.markets")

# Minimum volume for screening
MIN_VOLUME_FOR_SCREENING = int(os.getenv("MIN_VOLUME_FOR_SCREENING", "100000"))

data_api = tradeapi.REST(API_KEY, API_SECRET, DATA_API_URL)

# Alpaca Data API rate limit management
ALPACA_DATA_API_LAST_CALL_TIME = 0
ALPACA_DATA_API_RATE_LIMIT_SECONDS = 0.5

def enforce_alpaca_data_rate_limit():
    global ALPACA_DATA_API_LAST_CALL_TIME
    elapsed = time.time() - ALPACA_DATA_API_LAST_CALL_TIME
    if elapsed < ALPACA_DATA_API_RATE_LIMIT_SECONDS:
        sleep_time = ALPACA_DATA_API_RATE_LIMIT_SECONDS - elapsed
        time.sleep(sleep_time)
    ALPACA_DATA_API_LAST_CALL_TIME = time.time()

def get_latest_df_for_screener(symbol, limit=100):
    """Retrieves and formats minute-based stock data from Alpaca for screening."""
    enforce_alpaca_data_rate_limit()
    try:
        bars = data_api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=limit).df
        if bars.empty:
            return None
        
        bars.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return bars
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

def screen_for_opportunities(symbols_to_scan):
    """
    Screens for potential trading opportunities from a list of symbols
    and returns a sorted list of the best candidates based on technical score.
    """
    logger.info(f"Starting market screener for {len(symbols_to_scan)} symbols...")
    opportunities = []

    for symbol in symbols_to_scan:
        df = get_latest_df_for_screener(symbol, limit=100)
        if df is None or df.empty or len(df) < 50:
            continue
        
        # Volume confirmation check
        if df['Volume'].iloc[-1] < MIN_VOLUME_FOR_SCREENING:
            continue
            
        # Add indicators
        df = add_indicators(df)
        df = df.dropna()
        
        if df.empty:
            continue
            
        signal, score = compute_signal_and_score(df)
        
        # Get current data for additional filtering
        current_data = df.iloc[-1]
        
        # Additional filters for better quality signals
        is_quality_signal = True
        
        # Filter out low volatility stocks (ATR < 0.5% of price)
        if current_data['ATR'] / current_data['Close'] < 0.005:
            is_quality_signal = False
            
        # Filter out stocks with very wide Bollinger Bands (high volatility)
        if current_data['BB_Width'] > 0.1:  # More than 10% width
            is_quality_signal = False
            
        # Only consider buy signals for initial screening
        if signal == 1 and score > 0 and is_quality_signal:
            last_price = df['Close'].iloc[-1]
            opportunities.append({
                'symbol': symbol,
                'signal': signal,
                'score': score,
                'last_price': last_price,
                'rsi': current_data['RSI'],
                'atr': current_data['ATR'],
                'volume_ratio': current_data['Volume_Ratio']
            })

    # Sort opportunities by score in descending order
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    
    return opportunities