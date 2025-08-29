import os
import time
import alpaca_trade_api as tradeapi
import pandas as pd
import random
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST
from strategy import compute_signal_and_score
from screener import screen_for_opportunities
from news_analyzer import fetch_news_for_symbol, analyze_with_gemini
from ml_predictor import train_model, predict_next_price_movement 
from logger_config import get_logger

load_dotenv()
logger = get_logger()

# Bot configuration from .env
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_BASE_URL", "https://paper-api.alpaca.markets")
POSITION_SIZE_USD = float(os.getenv("POSITION_SIZE_USD", "500"))
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
BASE_STOP_LOSS_PERCENT = float(os.getenv("BASE_STOP_LOSS_PERCENT", "3.0"))  
BASE_TAKE_PROFIT_PERCENT = float(os.getenv("BASE_TAKE_PROFIT_PERCENT", "9.0")) 
MIN_STOP_LOSS_PERCENT = float(os.getenv("MIN_STOP_LOSS_PERCENT", "1.5")) 
MIN_TAKE_PROFIT_PERCENT = float(os.getenv("MIN_TAKE_PROFIT_PERCENT", "2.0")) 
BATCH_SIZE = 100 
SCREENING_POOL_SIZE = int(os.getenv("SCREENING_POOL_SIZE", "1000")) 
TOP_CANDIDATES_COUNT = int(os.getenv("TOP_CANDIDATES_COUNT", "5")) 
RISK_PER_TRADE_PERCENT = float(os.getenv("RISK_PER_TRADE_PERCENT", "1.0")) 
ML_TRAINING_DATA_LIMIT = int(os.getenv("ML_TRAINING_DATA_LIMIT", "1000")) 
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "5"))
MAX_DAILY_LOSS_PERCENT = float(os.getenv("MAX_DAILY_LOSS_PERCENT", "5.0"))

# Alpaca API clients
DATA_API_URL = os.getenv("DATA_API_URL", "https://data.alpaca.markets")
data_api = tradeapi.REST(API_KEY, API_SECRET, DATA_API_URL)
api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

if not API_KEY or not API_SECRET:
    logger.error("Missing Alpaca API key/secret. Fill .env and restart.")
    raise SystemExit(1)

# Global variables
open_positions_metadata = {}
trained_ml_model = None
ALPACA_TRADING_API_LAST_CALL_TIME = 0
ALPACA_TRADING_API_RATE_LIMIT_SECONDS = 0.3
ALPACA_DATA_API_LAST_CALL_TIME = 0
ALPACA_DATA_API_RATE_LIMIT_SECONDS = 0.5
initial_equity = None
daily_equity_high = None
consecutive_losses = 0
max_consecutive_losses = 5

def enforce_alpaca_trading_rate_limit():
    global ALPACA_TRADING_API_LAST_CALL_TIME
    elapsed = time.time() - ALPACA_TRADING_API_LAST_CALL_TIME
    if elapsed < ALPACA_TRADING_API_RATE_LIMIT_SECONDS:
        sleep_time = ALPACA_TRADING_API_RATE_LIMIT_SECONDS - elapsed
        time.sleep(sleep_time)
    ALPACA_TRADING_API_LAST_CALL_TIME = time.time()

def enforce_alpaca_data_rate_limit_direct():
    global ALPACA_DATA_API_LAST_CALL_TIME
    elapsed = time.time() - ALPACA_DATA_API_LAST_CALL_TIME
    if elapsed < ALPACA_DATA_API_RATE_LIMIT_SECONDS:
        sleep_time = ALPACA_DATA_API_RATE_LIMIT_SECONDS - elapsed
        time.sleep(sleep_time)
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
        logger.error(f"Error checking market status: {e}")
        now = datetime.now()
        return 0 <= now.weekday() <= 4 and 9 <= now.hour < 16

def get_market_close_time():
    enforce_alpaca_trading_rate_limit()
    try:
        clock = api.get_clock()
        return clock.next_close if clock and clock.next_close else None
    except Exception as e:
        logger.error(f"Error getting market close time: {e}")
        return None

def close_all_positions_at_market_close():
    current_time_utc = datetime.now(timezone.utc)
    square_off_target_hour_utc = 18
    square_off_target_minute_utc = 30
    
    today_square_off_utc = current_time_utc.replace(
        hour=square_off_target_hour_utc, minute=square_off_target_minute_utc, second=0, microsecond=0
    )
    
    if current_time_utc >= today_square_off_utc:
        square_off_time_utc = today_square_off_utc + timedelta(days=1)
    else:
        square_off_time_utc = today_square_off_utc
        
    time_to_close_buffer = timedelta(minutes=5)
    
    if current_time_utc >= (square_off_time_utc - time_to_close_buffer) and current_time_utc < square_off_time_utc:
        logger.info(f"Approaching 12 AM IST. Closing all positions.")
        try:
            positions = api.list_positions()
            for pos in positions:
                logger.info(f"Squaring off {pos.symbol} (qty: {pos.qty})")
                enforce_alpaca_trading_rate_limit()
                api.close_position(pos.symbol)
                if pos.symbol in open_positions_metadata:
                    del open_positions_metadata[pos.symbol]
        except Exception as e:
            logger.error(f"Error closing positions: {e}")

def get_latest_df(symbol, limit=390):
    enforce_alpaca_data_rate_limit_direct()
    try:
        bars = data_api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=limit).df
        if bars.empty:
            return None
        bars.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return bars
    except Exception as e:
        logger.error(f"Failed to get data for {symbol}: {e}")
        return None

def get_all_active_assets():
    enforce_alpaca_trading_rate_limit()
    try:
        assets = api.list_assets(status='active', asset_class='us_equity')
        return [asset.symbol for asset in assets if asset.tradable]
    except Exception as e:
        logger.error(f"Failed to get asset list: {e}")
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
        return api.get_position(symbol)
    except Exception:
        return None

def wait_for_order_fill(order_id, symbol, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            order = api.get_order(order_id)
            if order.filled_at:
                position = get_position_info(symbol)
                if position:
                    return float(position.avg_entry_price)
        except:
            time.sleep(1)
    return None

def safe_place_order(symbol, qty, side):
    enforce_alpaca_trading_rate_limit()
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side=side, type='market', time_in_force='gtc')
        logger.info(f"Order submitted: {side} {qty} {symbol}")
        
        if side == 'buy' and order:
            filled_price = wait_for_order_fill(order.id, symbol)
            if filled_price is None:
                filled_price = open_positions_metadata.get(symbol, {}).get('last_price', 0)
            
            open_positions_metadata[symbol] = {
                'entry_price': filled_price,
                'dynamic_sl_percent': BASE_STOP_LOSS_PERCENT,
                'dynamic_tp_percent': BASE_TAKE_PROFIT_PERCENT,
                'qty': qty
            }
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
            'buying_power': float(account.buying_power),
            'cash': float(account.cash)
        }
    except Exception as e:
        logger.error(f"Failed to get account info: {e}")
        return {'equity': 0.0, 'buying_power': 0.0, 'cash': 0.0}

def initialize_open_positions_metadata():
    logger.info("Initializing open positions metadata...")
    try:
        positions = api.list_positions()
        for pos in positions:
            symbol = pos.symbol
            open_positions_metadata[symbol] = {
                'entry_price': float(pos.avg_entry_price),
                'dynamic_sl_percent': BASE_STOP_LOSS_PERCENT,
                'dynamic_tp_percent': BASE_TAKE_PROFIT_PERCENT,
                'qty': int(float(pos.qty))
            }
    except Exception as e:
        logger.error(f"Error initializing positions: {e}")

def train_market_model():
    logger.info("Training market model on multiple ETFs...")
    symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
    all_data = []
    for symbol in symbols:
        df = get_latest_df(symbol, limit=ML_TRAINING_DATA_LIMIT//len(symbols))
        if df is not None:
            all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data)
        return train_model(combined_df)
    return None

def can_open_new_position(current_positions_count):
    return current_positions_count < MAX_POSITIONS

def emergency_stop_conditions(account_info):
    global initial_equity, daily_equity_high, consecutive_losses
    
    if initial_equity is None:
        initial_equity = account_info['equity']
        daily_equity_high = account_info['equity']
    
    current_equity = account_info['equity']
    daily_equity_high = max(daily_equity_high, current_equity)
    
    # 20% total drawdown
    if current_equity < initial_equity * 0.8:
        logger.error("EMERGENCY STOP: 20% total drawdown!")
        return True
        
    # 5% daily drawdown
    if current_equity < daily_equity_high * (1 - MAX_DAILY_LOSS_PERCENT/100):
        logger.error(f"EMERGENCY STOP: {MAX_DAILY_LOSS_PERCENT}% daily drawdown!")
        return True
        
    # Too many consecutive losses
    if consecutive_losses >= max_consecutive_losses:
        logger.error(f"EMERGENCY STOP: {consecutive_losses} consecutive losses!")
        return True
        
    return False

def update_trade_outcome(symbol, profit_loss):
    global consecutive_losses
    if profit_loss < 0:
        consecutive_losses += 1
    else:
        consecutive_losses = max(0, consecutive_losses - 1)

def main_loop():
    global initial_equity, daily_equity_high, consecutive_losses
    
    logger.info("Starting enhanced trading bot")
    
    if not test_api_connection():
        return

    initialize_open_positions_metadata()
    all_symbols = get_all_active_assets()
    
    if not all_symbols:
        logger.error("No tradable assets found.")
        return

    # Train improved ML model
    global trained_ml_model
    trained_ml_model = train_market_model()
    if trained_ml_model:
        logger.info("Market model trained successfully")
    else:
        logger.warning("ML model training failed")

    while True:
        try:
            if is_market_open():
                account_info = get_account_info()
                current_equity = account_info['equity']
                
                if emergency_stop_conditions(account_info):
                    logger.error("Trading halted - emergency conditions")
                    time.sleep(3600)
                    continue

                # Check and manage existing positions
                try:
                    positions = api.list_positions()
                    for pos in positions:
                        symbol = pos.symbol
                        df = get_latest_df(symbol, limit=10)
                        if df is not None and not df.empty:
                            current_price = float(df['Close'].iloc[-1])
                            entry_price = float(pos.avg_entry_price)
                            pl_percent = ((current_price - entry_price) / entry_price) * 100
                            
                            sl_percent = max(BASE_STOP_LOSS_PERCENT, MIN_STOP_LOSS_PERCENT)
                            tp_percent = max(BASE_TAKE_PROFIT_PERCENT, MIN_TAKE_PROFIT_PERCENT)
                            
                            if pl_percent < -sl_percent:
                                logger.warning(f"Stop-loss triggered for {symbol}")
                                safe_place_order(symbol, pos.qty, 'sell')
                                update_trade_outcome(symbol, pl_percent)
                            elif pl_percent > tp_percent:
                                logger.info(f"Take-profit triggered for {symbol}")
                                safe_place_order(symbol, pos.qty, 'sell')
                                update_trade_outcome(symbol, pl_percent)
                except Exception as e:
                    logger.error(f"Error monitoring positions: {e}")

                close_all_positions_at_market_close()
                random.shuffle(all_symbols)
                
                # Screen for opportunities in batches
                for i in range(0, min(SCREENING_POOL_SIZE, len(all_symbols)), BATCH_SIZE):
                    batch = all_symbols[i:i + BATCH_SIZE]
                    opportunities = screen_for_opportunities(batch)
                    
                    if opportunities and can_open_new_position(len(api.list_positions())):
                        for candidate in opportunities[:TOP_CANDIDATES_COUNT]:
                            symbol = candidate['symbol']
                            if get_position_qty(symbol) == 0:
                                # ML prediction
                                ml_prediction, ml_prob = -1, 0.0
                                if trained_ml_model:
                                    df_ml = get_latest_df(symbol, limit=ML_TRAINING_DATA_LIMIT)
                                    if df_ml is not None:
                                        ml_prediction, ml_prob = predict_next_price_movement(trained_ml_model, df_ml)
                                
                                if ml_prediction == 1 and ml_prob > 0.3:
                                    # Gemini analysis and trade execution
                                    news = fetch_news_for_symbol(symbol)
                                    recent_data = get_latest_df(symbol, limit=10)
                                    gemini_analysis = analyze_with_gemini(symbol, news, recent_data)
                                    
                                    if gemini_analysis.get('sentiment') == 'Positive':
                                        # Calculate position size with risk management
                                        risk_amount = current_equity * (RISK_PER_TRADE_PERCENT/100)
                                        qty = max(1, int(risk_amount / candidate['last_price']))
                                        safe_place_order(symbol, qty, 'buy')
                
                time.sleep(POLL_INTERVAL_SECONDS)
            else:
                logger.info("Market closed - waiting 15 minutes")
                time.sleep(900)
                
        except Exception as e:
            logger.exception("Error in main loop")
            time.sleep(300)

if __name__ == "__main__":
    main_loop()