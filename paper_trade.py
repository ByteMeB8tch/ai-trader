import os
import time
import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST

from strategy import compute_signal_and_score
from screener import screen_for_opportunities
from news_analyzer import fetch_news_for_symbol, analyze_with_gemini, analyze_with_perplexity, analyze_news_for_symbol
from ml_predictor import train_model, predict_next_price_movement
from logger_config import get_logger

load_dotenv()
logger = get_logger()

# Bot config from .env
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_BASE_URL", "https://paper-api.alpaca.markets")

POSITION_SIZE_USD = float(os.getenv("POSITION_SIZE_USD", "500"))
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
BASE_STOP_LOSS_PERCENT = float(os.getenv("BASE_STOP_LOSS_PERCENT", "8.0"))
BASE_TAKE_PROFIT_PERCENT = float(os.getenv("BASE_TAKE_PROFIT_PERCENT", "2.0"))
MIN_STOP_LOSS_PERCENT = float(os.getenv("MIN_STOP_LOSS_PERCENT", "1.0"))
MIN_TAKE_PROFIT_PERCENT = float(os.getenv("MIN_TAKE_PROFIT_PERCENT", "1.0"))

BATCH_SIZE = 50
SCREENING_POOL_SIZE = int(os.getenv("SCREENING_POOL_SIZE", "1000"))
TOP_CANDIDATES_COUNT = int(os.getenv("TOP_CANDIDATES_COUNT", "10"))
RISK_PER_TRADE_PERCENT = float(os.getenv("RISK_PER_TRADE_PERCENT", "1.0"))
ML_TRAINING_DATA_LIMIT = int(os.getenv("ML_TRAINING_DATA_LIMIT", "1000"))

data_api = tradeapi.REST(API_KEY, API_SECRET, os.getenv("DATA_API_URL", "https://data.alpaca.markets"))
api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

if not API_KEY or not API_SECRET:
    logger.error("Missing Alpaca API key/secret. Fill .env and restart.")
    raise SystemExit(1)

open_positions_metadata = {}
trained_ml_model = None

ALPACA_TRADING_API_LAST_CALL_TIME = 0
ALPACA_TRADING_API_RATE_LIMIT_SECONDS = 0.3

ALPACA_DATA_API_LAST_CALL_TIME = 0
ALPACA_DATA_API_RATE_LIMIT_SECONDS = 0.5

def enforce_alpaca_trading_rate_limit():
    global ALPACA_TRADING_API_LAST_CALL_TIME
    elapsed = time.time() - ALPACA_TRADING_API_LAST_CALL_TIME
    if elapsed < ALPACA_TRADING_API_RATE_LIMIT_SECONDS:
        time.sleep(ALPACA_TRADING_API_RATE_LIMIT_SECONDS - elapsed)
    ALPACA_TRADING_API_LAST_CALL_TIME = time.time()

def enforce_alpaca_data_rate_limit_direct():
    global ALPACA_DATA_API_LAST_CALL_TIME
    elapsed = time.time() - ALPACA_DATA_API_LAST_CALL_TIME
    if elapsed < ALPACA_DATA_API_RATE_LIMIT_SECONDS:
        time.sleep(ALPACA_DATA_API_RATE_LIMIT_SECONDS - elapsed)
    ALPACA_DATA_API_LAST_CALL_TIME = time.time()

def test_api_connection():
    enforce_alpaca_trading_rate_limit()
    try:
        account = api.get_account()
        logger.info(f"Connected to Alpaca. Account status: {account.status}")
        return True
    except Exception as e:
        logger.error(f"Alpaca API connection failed: {e}")
        return False

def is_market_open():
    enforce_alpaca_trading_rate_limit()
    try:
        clock = api.get_clock()
        logger.info(f"Market status: {'OPEN' if clock.is_open else 'CLOSED'}")
        return clock.is_open
    except Exception as e:
        logger.error(f"Error checking market status with Alpaca SDK: {e}")
        now = datetime.now()
        is_weekday = 0 <= now.weekday() <= 4
        is_market_hour_et = 9 <= now.hour < 16 and not (now.hour == 9 and now.minute < 30)
        return is_weekday and is_market_hour_et

def get_latest_df(symbol, limit=390):
    enforce_alpaca_data_rate_limit_direct()
    try:
        bars = data_api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=limit).df
        if bars.empty:
            return None
        bars.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return bars
    except Exception as e:
        logger.error(f"Failed to get data from Alpaca for {symbol}: {e}")
        return None

def get_all_active_assets():
    enforce_alpaca_trading_rate_limit()
    try:
        assets = api.list_assets(status='active', asset_class='us_equity')
        return [asset.symbol for asset in assets if asset.tradable]
    except Exception as e:
        logger.error(f"Failed to get asset list from Alpaca: {e}")
        return []

def get_position_qty(symbol):
    enforce_alpaca_trading_rate_limit()
    try:
        pos = api.get_position(symbol)
        return int(float(pos.qty))
    except Exception:
        return 0

def get_position_info(symbol):
    enforce_alpaca_trading_rate_limit()
    try:
        pos = api.get_position(symbol)
        return pos
    except Exception:
        return None

def safe_place_order(symbol, qty, side, suggested_sl, suggested_tp):
    enforce_alpaca_trading_rate_limit()
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side=side, type='market', time_in_force='gtc')
        logger.info(f"Order submitted: {side} {qty} {symbol}")
        if side == 'buy' and order:
            filled_price = None
            if hasattr(order, 'filled_avg_price') and order.filled_avg_price:
                filled_price = float(order.filled_avg_price)
            else:
                time.sleep(0.5)
                pos_info = get_position_info(symbol)
                if pos_info:
                    filled_price = float(pos_info.avg_entry_price)
            entry_price_for_meta = filled_price if filled_price is not None else open_positions_metadata.get(symbol, {}).get('last_price', 0)
            open_positions_metadata[symbol] = {
                'entry_price': entry_price_for_meta,
                'dynamic_sl_percent': suggested_sl,
                'dynamic_tp_percent': suggested_tp,
                'qty': qty
            }
            logger.info(f"Stored dynamic SL/TP for {symbol}: Entry={entry_price_for_meta:.2f}, SL={suggested_sl:.2f}%, TP={suggested_tp:.2f}%")
        return order
    except Exception as e:
        logger.error(f"Order failed for {symbol}: {e}")
        return None

def get_account_info():
    enforce_alpaca_trading_rate_limit()
    try:
        account = api.get_account()
        return {
            'equity': float(account.equity),
            'buying_power': float(account.buying_power)
        }
    except Exception as e:
        logger.error(f"Failed to get account info: {e}")
        return {'equity': 0.0, 'buying_power': 0.0}

def initialize_open_positions_metadata():
    logger.info("Initializing open positions metadata from Alpaca...")
    try:
        positions = api.list_positions()
        for pos in positions:
            symbol = pos.symbol
            entry_price = float(pos.avg_entry_price)
            qty = int(float(pos.qty))
            open_positions_metadata[symbol] = {
                'entry_price': entry_price,
                'dynamic_sl_percent': BASE_STOP_LOSS_PERCENT,
                'dynamic_tp_percent': BASE_TAKE_PROFIT_PERCENT,
                'qty': qty
            }
            logger.info(f"Initialized position for {symbol}: Entry={entry_price:.2f}, Qty={qty}, SL={BASE_STOP_LOSS_PERCENT:.2f}%, TP={BASE_TAKE_PROFIT_PERCENT:.2f}%")
    except Exception as e:
        logger.error(f"Error initializing open positions metadata: {e}")

def close_all_positions_at_ist_time():
    # Stub implementation: does nothing for now
    logger.info("close_all_positions_at_ist_time called (stub)")

def main_loop():
    logger.info("Starting paper trading loop")
    if not test_api_connection():
        logger.error("Cannot connect to Alpaca API. Check your .env file and API keys.")
        return

    initialize_open_positions_metadata()

    all_symbols = get_all_active_assets()
    if not all_symbols:
        logger.error("No tradable assets found. Exiting.")
        return

    logger.info(f"Found {len(all_symbols)} tradable symbols. Starting scan in batches of {BATCH_SIZE}.")

    global trained_ml_model
    logger.info("Fetching historical data to train ML model...")
    ml_training_symbol = 'SPY'
    historical_df_for_ml = get_latest_df(ml_training_symbol, limit=ML_TRAINING_DATA_LIMIT)

    if historical_df_for_ml is not None and not historical_df_for_ml.empty:
        trained_ml_model = train_model(historical_df_for_ml)
        if trained_ml_model:
            logger.info("ML model successfully trained.")
        else:
            logger.warning("ML model training failed. ML predictions will be skipped.")
    else:
        logger.warning("Could not fetch historical data for ML model training. ML predictions will be skipped.")

    risk_multiplier = 1.0
    max_multiplier = 5.0
    cycles_without_trade = 0
    no_trade_threshold = 3

    while True:
        try:
            if is_market_open():
                account_info = get_account_info()
                current_equity = float(account_info['equity'])
                current_buying_power = float(account_info['buying_power'])

                if current_equity <= 0 or current_buying_power <= 0:
                    logger.error("Account equity or buying power is zero or negative. Cannot trade. Sleeping.")
                    time.sleep(300)
                    continue

                max_risk_amount = current_equity * (RISK_PER_TRADE_PERCENT / 100)
                logger.info(f"Current Equity: ${current_equity:.2f}, Max Risk per Trade: ${max_risk_amount:.2f}")

                close_all_positions_at_ist_time()

                import random
                random.shuffle(all_symbols)
                symbols_to_screen = all_symbols[:SCREENING_POOL_SIZE]

                trades_executed = 0

                for i in range(0, len(symbols_to_screen), BATCH_SIZE):
                    batch = symbols_to_screen[i:i + BATCH_SIZE]
                    logger.info(f"Scanning batch {int(i / BATCH_SIZE) + 1}/{int(len(symbols_to_screen) / BATCH_SIZE) + 1} for {len(batch)} symbols...")

                    opportunities = screen_for_opportunities(batch)
                    if not opportunities:
                        continue

                    # Select the best candidate (highest score)
                    try:
                        best_candidate = max(opportunities, key=lambda x: x['score'])
                    except Exception as e:
                        logger.error(f"Error selecting best candidate from opportunities: {e}. Opportunities: {opportunities}")
                        continue
                    symbol = best_candidate['symbol']
                    technical_signal = best_candidate['signal']
                    technical_score = best_candidate['score']
                    last_price = best_candidate['last_price']

                    # Further check with Gemini, Perplexity, and news
                    articles = fetch_news_for_symbol(symbol)
                    recent_df = get_latest_df(symbol, limit=20)
                    gemini_result = analyze_with_gemini(symbol, articles, recent_df)
                    news_text = "\n".join([f"Headline: {a['headline']}\nSummary: {a['summary']}" for a in articles])
                    perplexity_result = analyze_with_perplexity(news_text)
                    news_result = analyze_news_for_symbol(symbol, recent_df)

                    gemini_ok = gemini_result.get('confidence', 0) > 0.5 and gemini_result.get('sentiment', '').lower() == 'positive'
                    perplexity_ok = perplexity_result.get('confidence', 0) > 0.5 and perplexity_result.get('sentiment', '').lower() == 'positive'
                    news_ok = news_result and news_result.get('confidence', 0) > 0.5 and news_result.get('sentiment', '').lower() == 'positive'

                    if gemini_ok and perplexity_ok and news_ok:
                        account = get_account_info()
                        price = last_price if last_price > 0 else 1.0
                        qty = 1  # Always buy 1 share for testing
                        sl = gemini_result.get('suggested_stop_loss_percent', BASE_STOP_LOSS_PERCENT)
                        tp = gemini_result.get('suggested_take_profit_percent', BASE_TAKE_PROFIT_PERCENT)
                        open_positions_metadata[symbol] = {
                            'entry_price': price,
                            'qty': qty,
                            'dynamic_sl': sl,
                            'dynamic_tp': tp
                        }
                        safe_place_order(symbol, qty, 'buy', sl, tp)
                        logger.info(f"Placed buy order for {symbol}: qty={qty}, SL={sl}%, TP={tp}% (BEST TRADE WITH NEWS/GEMINI/PERPLEXITY)")
                        trades_executed += 1
                    else:
                        logger.info(f"Skipped {symbol} due to news/Gemini/Perplexity filter.")

                # Risk escalation
                if trades_executed == 0:
                    cycles_without_trade += 1
                    if cycles_without_trade >= no_trade_threshold and risk_multiplier < max_multiplier:
                        risk_multiplier = min(max_multiplier, risk_multiplier * 1.3)
                        logger.info(f"No trades for {cycles_without_trade} cycles, increased risk multiplier to {risk_multiplier:.2f}")
                else:
                    cycles_without_trade = 0
                    risk_multiplier = 1.0

                logger.info(f"Completed a full scan cycle. Next scan in {POLL_INTERVAL_SECONDS} seconds.")
                time.sleep(POLL_INTERVAL_SECONDS)
            else:
                logger.info("Market closed. Waiting 15 minutes to reopen check.")
                time.sleep(900)
        except Exception as e:
            logger.exception("Error in main loop, sleeping 5 minutes before retry.")
            time.sleep(300)

if __name__ == "__main__":
    main_loop()
