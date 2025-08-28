import os
import time
import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST
from strategy import compute_signal_and_score
from screener import screen_for_opportunities
from news_analyzer import fetch_news_for_symbol, analyze_with_gemini # NEW IMPORT
from logger_config import get_logger

load_dotenv()
logger = get_logger()

# Bot configuration from .env
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_BASE_URL", "https://paper-api.alpaca.markets")
POSITION_SIZE_USD = float(os.getenv("POSITION_SIZE_USD", "500"))
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
BASE_STOP_LOSS_PERCENT = float(os.getenv("BASE_STOP_LOSS_PERCENT", "3.0"))  # Base stop-loss
BASE_TAKE_PROFIT_PERCENT = float(os.getenv("BASE_TAKE_PROFIT_PERCENT", "6.0")) # Base take-profit
BATCH_SIZE = 100 
SCREENING_POOL_SIZE = int(os.getenv("SCREENING_POOL_SIZE", "1000")) # How many stocks to initially screen
TOP_CANDIDATES_COUNT = int(os.getenv("TOP_CANDIDATES_COUNT", "10")) # How many top candidates to analyze with news/Gemini

# Alpaca API clients
DATA_API_URL = os.getenv("DATA_API_URL", "https://data.alpaca.markets")
data_api = tradeapi.REST(API_KEY, API_SECRET, DATA_API_URL)
api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

if not API_KEY or not API_SECRET:
    logger.error("Missing Alpaca API key/secret. Fill .env and restart.")
    raise SystemExit(1)

# Alpaca Trading API rate limit management (200 req/min for free tier)
ALPACA_TRADING_API_LAST_CALL_TIME = 0
ALPACA_TRADING_API_RATE_LIMIT_SECONDS = 0.3 # 200 req/min = 1 req every 0.3 seconds

def enforce_alpaca_trading_rate_limit():
    global ALPACA_TRADING_API_LAST_CALL_TIME
    elapsed = time.time() - ALPACA_TRADING_API_LAST_CALL_TIME
    if elapsed < ALPACA_TRADING_API_RATE_LIMIT_SECONDS:
        sleep_time = ALPACA_TRADING_API_RATE_LIMIT_SECONDS - elapsed
        logger.info(f"Alpaca Trading API rate limit hit. Sleeping for {sleep_time:.2f} seconds.")
        time.sleep(sleep_time)
    ALPACA_TRADING_API_LAST_CALL_TIME = time.time()


def test_api_connection():
    """Tests if Alpaca API connection works."""
    enforce_alpaca_trading_rate_limit() # Apply rate limit
    try:
        account = api.get_account()
        logger.info(f"Connected to Alpaca. Account status: {account.status}")
        return True
    except Exception as e:
        logger.error(f"Alpaca API connection failed: {e}")
        return False

def is_market_open():
    """Checks if the market is open using the Alpaca SDK."""
    enforce_alpaca_trading_rate_limit() # Apply rate limit
    try:
        clock = api.get_clock()
        logger.info(f"Market status: {'OPEN' if clock.is_open else 'CLOSED'}")
        return clock.is_open
    except Exception as e:
        logger.error(f"Error checking market status with Alpaca SDK: {e}")
        # Simple fallback for market hours
        now = datetime.now() # Use datetime.now() for timezone-aware comparison
        # Assuming ET for market hours, convert local time to approximate ET
        # This is a basic fallback, proper timezone handling is more complex
        is_weekday = 0 <= now.weekday() <= 4
        is_market_hour_et = 9 <= now.hour < 16 and not (now.hour == 9 and now.minute < 30)
        return is_weekday and is_market_hour_et

def get_market_close_time():
    """Gets the market close time for today in UTC."""
    enforce_alpaca_trading_rate_limit() # Apply rate limit
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

def close_all_positions_at_market_close():
    """Closes all open positions if market close is imminent."""
    market_close_time_utc = get_market_close_time()
    if market_close_time_utc:
        time_to_close_buffer = timedelta(minutes=5)
        current_time_utc = datetime.now(timezone.utc)

        if current_time_utc >= (market_close_time_utc - time_to_close_buffer) and current_time_utc < market_close_time_utc:
            logger.info("Market close is imminent. Attempting to close all open positions.")
            try:
                positions = api.list_positions()
                if positions:
                    for pos in positions:
                        logger.info(f"Squaring off position for {pos.symbol} (qty: {pos.qty}).")
                        enforce_alpaca_trading_rate_limit() # Apply rate limit
                        api.close_position(pos.symbol) 
                else:
                    logger.info("No open positions to square off.")
            except Exception as e:
                logger.error(f"Error closing all positions at market close: {e}")
        else:
            logger.info("Not yet time to square off positions for market close.")
    else:
        logger.warning("Could not determine market close time. Skipping market close square off.")


def get_latest_df(symbol, limit=390):
    """Retrieves and formats minute-based stock data from Alpaca."""
    # Rate limit for data API is handled in screener.py, but for direct calls, apply here.
    # For simplicity, we'll assume screener.py's rate limit is sufficient for initial calls
    # or that this function is called less frequently for individual symbols.
    try:
        bars = data_api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=limit).df
        if bars.empty:
            logger.warning(f"No data returned for {symbol}. Check if symbol exists.")
            return None
        
        bars.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return bars
    except Exception as e:
        logger.error(f"Failed to get data from Alpaca for {symbol}: {e}")
        return None

def get_all_active_assets():
    """Gets a list of all tradable assets on Alpaca."""
    enforce_alpaca_trading_rate_limit() # Apply rate limit
    try:
        assets = api.list_assets(status='active', asset_class='us_equity')
        return [asset.symbol for asset in assets if asset.tradable]
    except Exception as e:
        logger.error(f"Failed to get asset list from Alpaca: {e}")
        return []

def safe_place_order(symbol, qty, side):
    """Safely places a market order."""
    enforce_alpaca_trading_rate_limit() # Apply rate limit
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side=side, type='market', time_in_force='gtc')
        logger.info(f"Order submitted: {side} {qty} {symbol}")
        return order
    except Exception as e:
        logger.error(f"Order failed for {symbol}: {e}")
        return None

def get_position_qty(symbol):
    """Gets the current position quantity for a given symbol."""
    enforce_alpaca_trading_rate_limit() # Apply rate limit
    try:
        pos = api.get_position(symbol)
        return int(float(pos.qty))
    except Exception:
        return 0

def get_position_info(symbol):
    """Gets detailed information about a position."""
    enforce_alpaca_trading_rate_limit() # Apply rate limit
    try:
        pos = api.get_position(symbol)
        return pos
    except Exception:
        return None

def main_loop():
    logger.info("Starting paper trading loop")
    
    if not test_api_connection():
        logger.error("Cannot connect to Alpaca API. Check your .env file and API keys.")
        return

    all_symbols = get_all_active_assets()
    if not all_symbols:
        logger.error("No tradable assets found. Exiting.")
        return

    logger.info(f"Found {len(all_symbols)} tradable symbols. Starting scan in batches of {BATCH_SIZE}.")

    while True:
        try:
            if is_market_open():
                # Check all current positions for stop-loss and take-profit
                try:
                    positions = api.list_positions()
                    for pos in positions:
                        df_current = get_latest_df(pos.symbol, limit=1) 
                        if df_current is not None and not df_current.empty:
                            current_price = float(df_current['Close'].iloc[-1])
                            entry_price = float(pos.avg_entry_price)
                            unrealized_pl_percent = ((current_price - entry_price) / entry_price) * 100
                            
                            # Use dynamic SL/TP if available, otherwise base
                            current_stop_loss_threshold = BASE_STOP_LOSS_PERCENT
                            current_take_profit_threshold = BASE_TAKE_PROFIT_PERCENT

                            # Check if the position has associated dynamic SL/TP from Gemini (if stored)
                            # This would require storing SL/TP with the position when it's opened,
                            # which is an advanced feature not yet implemented. For now, we use base.

                            if unrealized_pl_percent < -current_stop_loss_threshold:
                                logger.warning(f"Stop-loss triggered for {pos.symbol}. Unrealized P/L: {unrealized_pl_percent:.2f}%")
                                safe_place_order(pos.symbol, pos.qty, 'sell')
                            elif unrealized_pl_percent > current_take_profit_threshold:
                                logger.info(f"Take-profit triggered for {pos.symbol}. Unrealized P/L: {unrealized_pl_percent:.2f}%")
                                safe_place_order(pos.symbol, pos.qty, 'sell')
                        else:
                            logger.warning(f"Could not get current price for {pos.symbol} to check stop-loss/take-profit.")
                except Exception as e:
                    logger.error(f"Error checking positions for stop-loss/take-profit: {e}")

                # Check if it's time to square off all positions before market close
                close_all_positions_at_market_close()

                # Shuffle symbols to ensure fair screening across runs
                import random
                random.shuffle(all_symbols)

                # Process stocks in batches for screening
                # Limit to SCREENING_POOL_SIZE for efficiency
                symbols_to_screen_this_cycle = all_symbols[:SCREENING_POOL_SIZE]
                
                for i in range(0, len(symbols_to_screen_this_cycle), BATCH_SIZE):
                    batch = symbols_to_screen_this_cycle[i:i + BATCH_SIZE]
                    logger.info(f"Screening batch {int(i/BATCH_SIZE) + 1}/{int(len(symbols_to_screen_this_cycle)/BATCH_SIZE) + 1} for {len(batch)} symbols...")

                    # Screen for and sort candidates based on technical score
                    # screener.py will return a sorted list of opportunities based on technical score
                    opportunities = screen_for_opportunities(batch)
                    
                    if opportunities:
                        # Select top candidates for deeper analysis with news/Gemini
                        top_candidates = opportunities[:TOP_CANDIDATES_COUNT]
                        logger.info(f"Found {len(top_candidates)} top technical candidates in this batch for deeper analysis.")

                        for candidate in top_candidates:
                            symbol = candidate['symbol']
                            technical_signal = candidate['signal']
                            technical_score = candidate['score']
                            last_price = candidate['last_price']

                            current_qty = get_position_qty(symbol)

                            # Only consider buying if we don't already have a position
                            if technical_signal == 1 and current_qty == 0:
                                logger.info(f"Performing deep analysis for {symbol} (Technical Score: {technical_score:.4f})...")
                                
                                # Fetch news
                                news_articles = fetch_news_for_symbol(symbol)
                                
                                # Get recent DF for Gemini context (last 10 minutes)
                                recent_df = get_latest_df(symbol, limit=10) 

                                # Analyze with Gemini API
                                gemini_analysis = analyze_with_gemini(symbol, news_articles, recent_df)
                                
                                confidence = gemini_analysis.get('confidence', 0.0)
                                sentiment = gemini_analysis.get('sentiment', 'neutral')
                                rationale = gemini_analysis.get('rationale', 'N/A')
                                suggested_sl = gemini_analysis.get('suggested_stop_loss_percent', BASE_STOP_LOSS_PERCENT)
                                suggested_tp = gemini_analysis.get('suggested_take_profit_percent', BASE_TAKE_PROFIT_PERCENT)

                                logger.info(f"Gemini Analysis for {symbol}: Sentiment={sentiment}, Confidence={confidence:.2f}, SL={suggested_sl:.2f}%, TP={suggested_tp:.2f}% - Rationale: {rationale}")

                                # Dynamic position sizing and trading decision based on Gemini's confidence
                                trade_amount_multiplier = 0.0 # Default to no trade

                                if sentiment == 'Positive':
                                    if confidence > 0.8: # Very high confidence for "guaranteed" profit
                                        trade_amount_multiplier = 2.0 # Trade a larger amount
                                        logger.info(f"Very high confidence BUY for {symbol}. Multiplier: {trade_amount_multiplier}x")
                                    elif confidence > 0.6: # High confidence
                                        trade_amount_multiplier = 1.0 # Trade normal amount
                                        logger.info(f"High confidence BUY for {symbol}. Multiplier: {trade_amount_multiplier}x")
                                    elif confidence > 0.4: # Medium confidence
                                        trade_amount_multiplier = 0.5 # Trade a smaller amount
                                        logger.info(f"Medium confidence BUY for {symbol}. Multiplier: {trade_amount_multiplier}x")
                                
                                if trade_amount_multiplier > 0:
                                    qty_to_trade = max(1, int((POSITION_SIZE_USD * trade_amount_multiplier) / last_price))
                                    safe_place_order(symbol, qty_to_trade, 'buy')
                                    # For a truly dynamic SL/TP, you'd need to store these with the position
                                    # and update the stop-loss checking logic to retrieve them.
                                    # For now, the continuous stop-loss/take-profit check uses BASE values
                                    # unless you implement a way to store and retrieve these per position.
                                else:
                                    logger.info(f"Insufficient confidence or non-positive sentiment for {symbol}. Skipping trade.")

                            elif technical_signal == 0 and current_qty > 0:
                                # A technical sell signal. If position is open, rely on SL/TP or market close.
                                logger.info(f"Technical sell signal for {symbol}, but position is open. Relying on SL/TP or market close.")
                            else:
                                logger.info(f"No new buy signal for {symbol} or position already open.")
                    else:
                        logger.info("No technically profitable opportunities found in this batch.")
                
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

