
import os, time
import yfinance as yf
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

def get_latest_df(symbol, days=120):
    df = yf.download(symbol, period=f"{days}d", interval="1d")
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
    while True:
        try:
            df = get_latest_df(SYMBOL, days=120)
            if df is None or df.empty:
                logger.warning("No data; sleeping")
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
        except Exception as e:
            logger.exception("Error in main loop")
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    main_loop()