import os, time
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST
from strategy import compute_signal
from logger_config import get_logger

load_dotenv()
logger = get_logger()

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_BASE_URL", "https://paper-api.alpaca.markets")
SYMBOL = os.getenv("SYMBOL", "AAPL")
POSITION_SIZE_USD = float(os.getenv("POSITION_SIZE_USD", "500"))
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "3600"))

if not API_KEY or not API_SECRET:
    logger.error("Missing Alpaca API key/secret. Fill .env and restart.")
    raise SystemExit(1)

api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def test_api_connection():
    """Test if Alpaca API connection works"""
    try:
        account = api.get_account()
        logger.info(f"Connected to Alpaca. Account status: {account.status}")
        return True
    except Exception as e:
        logger.error(f"Alpaca API connection failed: {e}")
        return False

def is_market_open():
    """Check if the market is open using Alpaca SDK"""
    try:
        clock = api.get_clock()
        logger.info(f"Market status: {'OPEN' if clock.is_open else 'CLOSED'}")
        return clock.is_open
    except Exception as e:
        logger.error(f"Error checking market status with Alpaca SDK: {e}")
        # Fallback: check if it's a weekday during market hours
        now = time.localtime()
        if now.tm_wday >= 5:  # Weekend (0=Monday, 6=Sunday)
            return False
        # Check if between 9:30 AM and 4:00 PM ET (approx)
        hour, minute = now.tm_hour, now.tm_min
        return (9 <= hour < 16) and not (hour == 9 and minute < 30)

def get_mock_data(symbol, days=120):
    """Generate realistic mock data for testing"""
    try:
        dates = [datetime.now() - timedelta(days=x) for x in range(days, 0, -1)]
        base_price = 180.0
        
        returns = np.random.normal(0.001, 0.02, days)
        prices = base_price * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            'Open': prices * 0.998,
            'High': prices * 1.015,
            'Low': prices * 0.985, 
            'Close': prices,
            'Volume': np.random.randint(50000000, 100000000, days)
        }, index=dates)
        
        logger.warning("Using MOCK DATA - Yahoo Finance unavailable")
        return df
    except Exception as e:
        logger.error(f"Mock data failed: {e}")
        return None

def get_alpha_vantage_data(symbol, api_key, days=120):
    """Get stock data from Alpha Vantage with error handling"""
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&apikey={api_key}&outputsize=full"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # Check for error message in the response
        if 'Error Message' in data:
            logger.error(f"Alpha Vantage error: {data['Error Message']}")
            return None
            
        if 'Time Series (Daily)' in data:
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
            df = df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High', 
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.index = pd.to_datetime(df.index)
            df = df.sort_index().iloc[-days:]
            logger.info(f"Successfully got data from Alpha Vantage for {symbol}")
            return df
    except Exception as e:
        logger.error(f"Alpha Vantage failed: {e}")
    return None

def get_latest_df(symbol, days=120):
    """Comprehensive data fetching with multiple fallbacks"""
    # Try Yahoo Finance first
    logger.info("Attempting to get data from Yahoo Finance...")
    try:
        df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)
        if df is not None and not df.empty:
            logger.info("Successfully got data from Yahoo Finance.")
            return df
    except Exception as e:
        logger.error(f"Yahoo Finance failed: {e}")
    
    # Fallback 1: Alpha Vantage (requires free API key)
    alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if alpha_vantage_key:
        logger.info("Yahoo Finance failed. Attempting to get data from Alpha Vantage...")
        df = get_alpha_vantage_data(symbol, alpha_vantage_key, days)
        if df is not None:
            return df
    
    # Fallback 2: Mock data for development
    logger.info("Both external data sources failed. Using mock data...")
    df = get_mock_data(symbol, days)
    return df

def safe_place_order(symbol, qty, side):
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side=side, type='market', time_in_force='gtc')
        logger.info(f"Order submitted: {side} {qty} {symbol}")
        return order
    except Exception as e:
        logger.error(f"Order failed: {e}")
        return None

def get_position_qty(symbol):
    try:
        pos = api.get_position(symbol)
        return int(float(pos.qty))
    except Exception:
        return 0

def main_loop():
    logger.info("Starting paper trading loop")
    
    # Test API connection first
    if not test_api_connection():
        logger.error("Cannot connect to Alpaca API. Check your .env file and API keys.")
        return
    
    while True:
        try:
            if is_market_open():
                df = get_latest_df(SYMBOL, days=120)
                if df is None or df.empty:
                    logger.warning("No data; sleeping for 60 seconds")
                    time.sleep(60)
                    continue
                
                signal = compute_signal(df)
                current_qty = get_position_qty(SYMBOL)
                last_price = df['Close'].iloc[-1]
                qty_to_trade = max(1, int(POSITION_SIZE_USD / last_price))

                if signal == 1 and current_qty == 0:
                    safe_place_order(SYMBOL, qty_to_trade, 'buy')
                elif signal == 0 and current_qty > 0:
                    safe_place_order(SYMBOL, current_qty, 'sell')
                else:
                    logger.info("No trade action required")
                    
                logger.info(f"Next check in {POLL_INTERVAL_SECONDS} seconds.")
                time.sleep(POLL_INTERVAL_SECONDS)
            else:
                logger.info("Market closed - waiting 15 minutes before next check")
                time.sleep(900)
                
        except Exception as e:
            logger.exception("Error in main loop. Sleeping for 5 minutes before retry.")
            time.sleep(300)

if __name__ == "__main__":
    main_loop()