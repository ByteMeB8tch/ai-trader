import os
import time
import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST
from strategy import compute_signal_and_score
from screener import screen_for_opportunities
from news_analyzer import fetch_news_for_symbol, analyze_with_gemini, analyze_with_perplexity
from ml_predictor import train_model, predict_next_price_movement 
from logger_config import get_logger

load_dotenv()
logger = get_logger()

# Bot configuration from .env
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_BASE_URL", "https://paper-api.alpaca.markets")
POSITION_SIZE_USD = float(os.getenv("POSITION_SIZE_USD", "500")) # Base amount to allocate per trade
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
BASE_STOP_LOSS_PERCENT = float(os.getenv("BASE_STOP_LOSS_PERCENT", "8.0"))  # Updated to 8.0%
BASE_TAKE_PROFIT_PERCENT = float(os.getenv("BASE_TAKE_PROFIT_PERCENT", "2.0"))

# NEW: Minimum thresholds for Gemini's suggested SL/TP
MIN_STOP_LOSS_PERCENT = float(os.getenv("MIN_STOP_LOSS_PERCENT", "1.0")) # Minimum stop-loss %
MIN_TAKE_PROFIT_PERCENT = float(os.getenv("MIN_TAKE_PROFIT_PERCENT", "1.0")) # Minimum take-profit %

BATCH_SIZE = 50 
SCREENING_POOL_SIZE = int(os.getenv("SCREENING_POOL_SIZE", "1000")) 
TOP_CANDIDATES_COUNT = int(os.getenv("TOP_CANDIDATES_COUNT", "10")) 
RISK_PER_TRADE_PERCENT = float(os.getenv("RISK_PER_TRADE_PERCENT", "1.0")) 

# NEW: Machine Learning configuration
ML_TRAINING_DATA_LIMIT = int(os.getenv("ML_TRAINING_DATA_LIMIT", "1000")) 

# Alpaca API clients
DATA_API_URL = os.getenv("DATA_API_URL", "https://data.alpaca.markets")
data_api = tradeapi.REST(API_KEY, API_SECRET, DATA_API_URL)
api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

if not API_KEY or not API_SECRET:
    logger.error("Missing Alpaca API key/secret. Fill .env and restart.")
    raise SystemExit(1)

# --- Global storage for dynamic SL/TP per position (resets on bot restart) ---
# For persistent storage across restarts, a database (e.g., Firestore) would be needed.
open_positions_metadata = {} 

# Store the trained ML model globally (resets on bot restart)
trained_ml_model = None

# Alpaca API rate limit management (200 req/min for free tier for trading, 200 req/min for data)
ALPACA_TRADING_API_LAST_CALL_TIME = 0
ALPACA_TRADING_API_RATE_LIMIT_SECONDS = 0.3 # 200 req/min = 1 req every 0.3 seconds

# Data API rate limit is managed in screener.py, but for direct calls, apply here.
ALPACA_DATA_API_LAST_CALL_TIME = 0
ALPACA_DATA_API_RATE_LIMIT_SECONDS = 0.5 # Alpaca free tier: 200 req/min = 1 req every 0.3 seconds. Use 0.5s for safety.

def enforce_alpaca_trading_rate_limit():
    global ALPACA_TRADING_API_LAST_CALL_TIME
    elapsed = time.time() - ALPACA_TRADING_API_LAST_CALL_TIME
    if elapsed < ALPACA_TRADING_API_RATE_LIMIT_SECONDS:
        sleep_time = ALPACA_TRADING_API_RATE_LIMIT_SECONDS - elapsed
        logger.info(f"Alpaca Trading API rate limit hit. Sleeping for {sleep_time:.2f} seconds.")
        time.sleep(sleep_time)
    ALPACA_TRADING_API_LAST_CALL_TIME = time.time()

def enforce_alpaca_data_rate_limit_direct(): # New rate limit for direct data calls in paper_trade.py
    global ALPACA_DATA_API_LAST_CALL_TIME
    elapsed = time.time() - ALPACA_DATA_API_LAST_CALL_TIME
    if elapsed < ALPACA_DATA_API_RATE_LIMIT_SECONDS:
        sleep_time = ALPACA_DATA_API_RATE_LIMIT_SECONDS - elapsed
        logger.info(f"Alpaca Data API rate limit hit (direct call). Sleeping for {sleep_time:.2f} seconds.")
        time.sleep(sleep_time)
    ALPACA_DATA_API_LAST_CALL_TIME = time.time()


def test_api_connection():
    """Tests if Alpaca API connection works."""
    enforce_alpaca_trading_rate_limit() 
    try:
        account = api.get_account()
        logger.info(f"Connected to Alpaca. Account status: {account.status}")
        return True
    except Exception as e:
        logger.error(f"Alpaca API connection failed: {e}")
        return False

def is_market_open():
    """Checks if the market is open using the Alpaca SDK."""
    enforce_alpaca_trading_rate_limit() 
    try:
        clock = api.get_clock()
        logger.info(f"Market status: {'OPEN' if clock.is_open else 'CLOSED'}")
        return clock.is_open
    except Exception as e:
        logger.error(f"Error checking market status with Alpaca SDK: {e}")
        now = datetime.now() 
        is_weekday = 0 <= now.weekday() <= 4
        is_market_hour_et = 9 <= now.hour < 16 and not (now.tm_hour == 9 and now.tm_min < 30)
        return is_weekday and is_market_hour_et

def get_market_close_time():
    """Gets the market close time for today in UTC."""
    enforce_alpaca_trading_rate_limit() 
    try:
        clock = api.get_clock()
        if clock and clock.next_close:
            return clock.next_close
        else:
            logger.warning("Could not get next_close from Alpaca clock. Estimating market close.")
            market_close_et = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
            return market_close_et.replace(tzinfo=timezone.utc) + timedelta(hours=4) 
    except Exception as e:
        logger.error(f"Error getting market close time: {e}")
        return None

def close_all_positions_at_ist_time():
    """
    Closes all open positions before 11 PM IST (5:30 PM UTC) to ensure no positions
    are held overnight.
    """
    current_time_utc = datetime.now(timezone.utc)
    
    # Target time: 5:30 PM UTC, which is 11:00 PM IST
    square_off_target_hour_utc = 17
    square_off_target_minute_utc = 30
    
    today_square_off_utc = current_time_utc.replace(
        hour=square_off_target_hour_utc, 
        minute=square_off_target_minute_utc, 
        second=0, 
        microsecond=0
    )
    
    if current_time_utc >= today_square_off_utc:
        square_off_time_utc = today_square_off_utc + timedelta(days=1)
    else:
        square_off_time_utc = today_square_off_utc
        
    time_to_close_buffer = timedelta(minutes=5)
    
    if current_time_utc >= (square_off_time_utc - time_to_close_buffer) and current_time_utc < square_off_time_utc:
        logger.info(f"Approaching 11 PM IST ({square_off_time_utc.strftime('%H:%M UTC')}). Attempting to close all open positions.")
        try:
            positions = api.list_positions()
            if positions:
                for pos in positions:
                    logger.info(f"Squaring off position for {pos.symbol} (qty: {pos.qty}).")
                    enforce_alpaca_trading_rate_limit() 
                    api.close_position(pos.symbol) 
                    if pos.symbol in open_positions_metadata:
                        del open_positions_metadata[pos.symbol]
            else:
                logger.info("No open positions to square off.")
        except Exception as e:
            logger.error(f"Error closing all positions before 11 PM IST: {e}")
    else:
        logger.info(f"Not yet time to square off positions for 11 PM IST. Next square off window: {(square_off_time_utc - time_to_close_buffer).strftime('%Y-%m-%d %H:%M UTC')} to {square_off_time_utc.strftime('%Y-%m-%d %H:%M UTC')}.")


def get_latest_df(symbol, limit=390):
    """Retrieves and formats minute-based stock data from Alpaca."""
    enforce_alpaca_data_rate_limit_direct() # Enforce rate limit for direct data calls
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
    """Gets a list of all tradable assets on Alpaca."""
    enforce_alpaca_trading_rate_limit() 
    try:
        assets = api.list_assets(status='active', asset_class='us_equity')
        return [asset.symbol for asset in assets if asset.tradable]
    except Exception as e:
        logger.error(f"Failed to get asset list from Alpaca: {e}")
        return []

def get_position_qty(symbol):
    """Gets the current position quantity for a given symbol."""
    enforce_alpaca_trading_rate_limit() 
    try:
        pos = api.get_position(symbol)
        return int(float(pos.qty))
    except Exception:
        return 0

def get_position_info(symbol):
    """Gets detailed information about a position."""
    enforce_alpaca_trading_rate_limit() 
    try:
        pos = api.get_position(symbol)
        return pos
    except Exception:
        return None

def safe_place_order(symbol, qty, side, suggested_sl, suggested_tp):
    """Safely places a market order and updates local position metadata."""
    enforce_alpaca_trading_rate_limit() 
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side=side, type='market', time_in_force='gtc')
        logger.info(f"Order submitted: {side} {qty} {symbol}")
        
        # Store dynamic SL/TP with the position if it's a buy order
        if side == 'buy' and order:
            filled_price = None
            if order.filled_avg_price:
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
            logger.info(f"Stored dynamic SL/TP for {symbol}: Entry={entry_price_for_meta:.2f}, SL={open_positions_metadata[symbol]['dynamic_sl_percent']:.2f}%, TP={open_positions_metadata[symbol]['dynamic_tp_percent']:.2f}%")

        return order
    except Exception as e:
        logger.error(f"Order failed for {symbol}: {e}")
        return None

def get_account_info():
    """Gets essential account information for risk management."""
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
    """
    Populates open_positions_metadata with existing positions from Alpaca at startup.
    This ensures the bot "remembers" positions opened in previous runs.
    """
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
    no_trade_threshold = 3  # Increase risk after this many cycles of no trades

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

                max_risk_amount_per_trade = current_equity * (RISK_PER_TRADE_PERCENT / 100)
                logger.info(f"Current Equity: ${current_equity:.2f}, Max Risk per Trade: ${max_risk_amount_per_trade:.2f}")

                positions_to_close = []
                try:
                    positions = api.list_positions()
                    for pos in positions:
                        symbol = pos.symbol
                        df_current = get_latest_df(symbol, limit=10)
                        if df_current is not None and not df_current.empty:
                            current_price = float(df_current['Close'].iloc[-1])
                            pos_meta = open_positions_metadata.get(symbol, {})
                            entry_price = pos_meta.get('entry_price', float(pos.avg_entry_price))
                            current_stop_loss_threshold = max(pos_meta.get('dynamic_sl_percent', BASE_STOP_LOSS_PERCENT), MIN_STOP_LOSS_PERCENT)
                            current_take_profit_threshold = max(pos_meta.get('dynamic_tp_percent', BASE_TAKE_PROFIT_PERCENT), MIN_TAKE_PROFIT_PERCENT)
                            unrealized_pl_percent = ((current_price - entry_price) / entry_price) * 100
                            logger.info(f"Monitoring {symbol} (Held). P/L: {unrealized_pl_percent:.2f}% (SL: {-current_stop_loss_threshold:.2f}%, TP: {current_take_profit_threshold:.2f}%)")
                            if unrealized_pl_percent < -current_stop_loss_threshold:
                                logger.warning(f"Stop-loss triggered for {symbol}. P/L: {unrealized_pl_percent:.2f}% - Selling.")
                                positions_to_close.append(pos)
                            elif unrealized_pl_percent > current_take_profit_threshold:
                                logger.info(f"Take-profit triggered for {symbol}. P/L: {unrealized_pl_percent:.2f}% - Selling.")
                                positions_to_close.append(pos)
                            else:
                                logger.info(f"Position for {symbol} within SL/TP. Asking Gemini for hold/sell advice...")
                                news_articles = fetch_news_for_symbol(symbol)
                                gemini_analysis = analyze_with_gemini(symbol, news_articles, df_current)
                                confidence = gemini_analysis.get('confidence', 0.0)
                                sentiment = gemini_analysis.get('sentiment', 'neutral')
                                if sentiment == 'Negative' or (sentiment == 'Neutral' and confidence < 0.5):
                                    logger.warning(f"Gemini suggests selling {symbol} (Sentiment: {sentiment}, Confidence: {confidence:.2f}). Selling position.")
                                    positions_to_close.append(pos)
                        else:
                            logger.warning(f"Could not get current price for {pos.symbol} to check held position. P/L unknown.")
                except Exception as e:
                    logger.error(f"Error checking held positions: {e}")

                for pos in positions_to_close:
                    safe_place_order(pos.symbol, pos.qty, 'sell', pos_meta.get('dynamic_sl_percent', BASE_STOP_LOSS_PERCENT), pos_meta.get('dynamic_tp_percent', BASE_TAKE_PROFIT_PERCENT))
                    if pos.symbol in open_positions_metadata:
                        del open_positions_metadata[pos.symbol]

                close_all_positions_at_ist_time()

                import random
                random.shuffle(all_symbols)
                symbols_to_screen_this_cycle = all_symbols[:SCREENING_POOL_SIZE]

                trades_executed_this_cycle = 0

                for i in range(0, len(symbols_to_screen_this_cycle), BATCH_SIZE):
                    batch = symbols_to_screen_this_cycle[i:i + BATCH_SIZE]
                    logger.info(f"Screening batch {int(i/BATCH_SIZE) + 1}/{int(len(symbols_to_screen_this_cycle)/BATCH_SIZE) + 1} for {len(batch)} symbols...")
                    opportunities = screen_for_opportunities(batch)

                    if opportunities:
                        top_candidates = opportunities[:TOP_CANDIDATES_COUNT]
                        logger.info(f"Found {len(top_candidates)} top technical candidates in this batch for deeper analysis.")

                        for candidate in top_candidates:
                            symbol = candidate['symbol']
                            technical_signal = candidate['signal']
                            technical_score = candidate['score']
                            last_price = candidate['last_price']
                            current_qty = get_position_qty(symbol)
                            if technical_signal == 1 and current_qty == 0:
                                logger.info(f"Performing deep analysis for NEW BUY candidate {symbol} (Technical Score: {technical_score:.4f})...")
                                ml_prediction = -1
                                ml_probability_of_up = 0.0

                                if trained_ml_model:
                                    current_df_for_ml = get_latest_df(symbol, limit=ML_TRAINING_DATA_LIMIT)
                                    if current_df_for_ml is not None and not current_df_for_ml.empty and len(current_df_for_ml) >= 20:
                                        ml_prediction, ml_probability_of_up = predict_next_price_movement(trained_ml_model, current_df_for_ml)
                                        logger.info(f"ML Prediction for {symbol}: {ml_prediction} (Prob Up: {ml_probability_of_up:.2f})")
                                    else:
                                        logger.warning(f"Insufficient data for ML prediction for {symbol}. Skipping ML.")
                                else:
                                    logger.warning("ML model not trained. Skipping ML prediction.")

                                if ml_prediction == 1 and ml_probability_of_up > 0.15:
                                    articles = fetch_news_for_symbol(symbol)
                                    if not articles:
                                        logger.info(f"No news articles for {symbol}, skipping.")
                                        continue
                                    recent_prices = get_latest_df(symbol, limit=10)
                                    if recent_prices is None or recent_prices.empty:
                                        logger.info(f"No recent prices for {symbol}, skipping.")
                                        continue
                                    news_text = "\n".join(f"Headline: {a['headline']}\nSummary: {a['summary']}" for a in articles)
                                    gemini_result = analyze_with_gemini(symbol, articles, recent_prices)
                                    perplexity_result = analyze_with_perplexity(news_text)
                                    logger.info(f"{symbol} Gemini: Sentiment={gemini_result['sentiment']} Confidence={gemini_result['confidence']:.2f}")
                                    logger.info(f"{symbol} Perplexity: Sentiment={perplexity_result['sentiment']} Confidence={perplexity_result['confidence']:.2f}")

                                    min_confidence = 0.3
                                    buy_signal = (gemini_result['sentiment'] == 'Positive' and gemini_result['confidence'] >= min_confidence) or \
                                                 (perplexity_result['sentiment'] == 'Positive' and perplexity_result['confidence'] >= min_confidence)
                                    max_confidence = max(gemini_result['confidence'], perplexity_result['confidence'])

                                    if buy_signal:
                                        # Increase trade size by multiplier during inactivity
                                        account = get_account_info()
                                        price = recent_prices['Close'].iloc[-1]
                                        max_risk_amount = account['equity'] * RISK_PER_TRADE_PERCENT / 100
                                        desired_value = min(risk_multiplier * POSITION_SIZE_USD, max_risk_amount, account['buying_power'])
                                        qty = max(1, int(desired_value / price))
                                        if qty * price > account['buying_power']:
                                            logger.warning(f"Not enough buying power for {symbol} to buy {qty} shares.")
                                            continue

                                        sl = max(gemini_result.get('suggested_stop_loss_percent', MIN_STOP_LOSS_PERCENT), MIN_STOP_LOSS_PERCENT)
                                        tp = max(gemini_result.get('suggested_take_profit_percent', MIN_TAKE_PROFIT_PERCENT), MIN_TAKE_PROFIT_PERCENT)
                                        open_positions_metadata[symbol] = {
                                            'entry_price': price,
                                            'qty': qty,
                                            'dynamic_sl': sl,
                                            'dynamic_tp': tp
                                        }

                                        safe_place_order(symbol, qty, 'buy', sl, tp)
                                        logger.info(f"Placed buy order for {symbol}: qty={qty}, SL={sl}%, TP={tp}% (risk_multiplier={risk_multiplier:.2f})")
                                        trades_executed_this_cycle += 1
                                    else:
                                        logger.info(f"Insufficient confidence for {symbol}. Skipping trade.")
                                else:
                                    logger.info(f"ML predictions do not support buying for {symbol}.")
                            elif technical_signal == 0 and current_qty > 0:
                                logger.info(f"Technical sell signal for {symbol}, but position is open. Relying on dynamic SL/TP or market close.")
                            else:
                                logger.info(f"No new buy signal for {symbol} or position already open.")
                    else:
                        logger.info("No technically profitable opportunities found in this batch.")

                # Risk escalation logic
                if trades_executed_this_cycle == 0:
                    cycles_without_trade += 1
                    if cycles_without_trade >= no_trade_threshold and risk_multiplier < max_multiplier:
                        risk_multiplier *= 1.3
                        risk_multiplier = min(risk_multiplier, max_multiplier)
                        logger.info(f"No trades for {cycles_without_trade} cycles, increasing risk multiplier to {risk_multiplier:.2f}")
                else:
                    cycles_without_trade = 0
                    risk_multiplier = 1.0

                logger.info(f"Completed a full market scan cycle. Next check in {POLL_INTERVAL_SECONDS} seconds.")
                time.sleep(POLL_INTERVAL_SECONDS)
            else:
                logger.info("Market closed - waiting 15 minutes before next check")
                time.sleep(900)
        except Exception as e:
            logger.exception("Error in main loop. Sleeping for 5 minutes before retry.")
            time.sleep(300)

if __name__ == "__main__":
    main_loop()
