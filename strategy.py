import pandas as pd
import numpy as np

def add_indicators(df):
    """Adds multiple technical indicators to the DataFrame."""
    # Existing SMAs
    df['SMA9'] = df['Close'].rolling(window=9).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()

    # Exponential Moving Averages
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()

    # Bollinger Bands
    bb_length = 20
    bb_std = 2
    df['BB_Middle'] = df['Close'].rolling(window=bb_length).mean()
    bb_std_dev = df['Close'].rolling(window=bb_length).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * bb_std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * bb_std_dev)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

    # RSI (Relative Strength Index)
    rsi_length = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_length).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # Volume indicators
    df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA20'].replace(0, 1) # Avoid division by zero

    # ATR (Average True Range) for volatility measurement
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()

    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14).replace(0, 1)) # Avoid division by zero
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return df

def compute_signal_and_score(df):
    """
    Computes a trading signal and profitability score based on multiple indicators.
    Returns:
    tuple: (signal, score) where signal is 1 (buy), 0 (sell), or -1 (hold),
    and score is a float representing the potential profitability.
    """
    df = add_indicators(df)
    # Drop rows with NaN values
    df = df.dropna()
    if df.empty or len(df) < 2:
        return None, 0.0

    current_data = df.iloc[-1]
    previous_data = df.iloc[-2]

    signal = -1
    score = 0.0

    # Calculate a composite score based on multiple indicators
    indicator_score = 0
    max_possible_score = 0

    # 1. SMA Crossover (weight: 30%)
    max_possible_score += 30
    if previous_data['SMA9'] < previous_data['SMA20'] and current_data['SMA9'] > current_data['SMA20']:
        indicator_score += 30
    elif previous_data['SMA9'] > previous_data['SMA20'] and current_data['SMA9'] < current_data['SMA20']:
        indicator_score -= 30

    # 2. MACD (weight: 20%)
    max_possible_score += 20
    if current_data['MACD'] > current_data['MACD_Signal'] and previous_data['MACD'] <= previous_data['MACD_Signal']:
        indicator_score += 20
    elif current_data['MACD'] < current_data['MACD_Signal'] and previous_data['MACD'] >= previous_data['MACD_Signal']:
        indicator_score -= 20

    # 3. RSI (weight: 15%)
    max_possible_score += 15
    if current_data['RSI'] < 30: # Oversold
        indicator_score += 15
    elif current_data['RSI'] > 70: # Overbought
        indicator_score -= 15
    elif 30 <= current_data['RSI'] <= 70:
        # Neutral RSI adds nothing
        max_possible_score -= 15 # Adjust max since neutral doesn't contribute

    # 4. Bollinger Band Position (weight: 15%)
    max_possible_score += 15
    bb_range = current_data['BB_Upper'] - current_data['BB_Lower']
    bb_range = bb_range if bb_range != 0 else 1  # Prevent division by zero
    bb_position = (current_data['Close'] - current_data['BB_Lower']) / bb_range

    if bb_position < 0.2: # Near lower band (potential buy)
        indicator_score += 15
    elif bb_position > 0.8: # Near upper band (potential sell)
        indicator_score -= 15

    # 5. Volume confirmation (weight: 10%)
    max_possible_score += 10
    if current_data['Volume_Ratio'] > 1.5: # High volume confirms move
        if indicator_score > 0: # If other indicators are bullish
            indicator_score += 10
        elif indicator_score < 0: # If other indicators are bearish
            indicator_score -= 10

    # 6. Stochastic (weight: 10%)
    max_possible_score += 10
    if current_data['Stoch_K'] < 20 and current_data['Stoch_D'] < 20: # Oversold
        indicator_score += 10
    elif current_data['Stoch_K'] > 80 and current_data['Stoch_D'] > 80: # Overbought
        indicator_score -= 10

    # Determine final signal and score
    if indicator_score >= 20: # Strong buy signal
        signal = 1
        # Normalize score to 0-1 range
        score = min(1.0, max(0.0, indicator_score / max_possible_score))
        # Adjust for how far price is from SMA20
        price_deviation = (current_data['Close'] - current_data['SMA20']) / current_data['SMA20']
        score = score * (1 + price_deviation) if price_deviation > 0 else score
    elif indicator_score <= -20: # Strong sell signal
        signal = 0
        # Normalize score to 0-1 range
        score = min(1.0, max(0.0, abs(indicator_score) / max_possible_score))
        # Adjust for how far price is from SMA20
        price_deviation = (current_data['SMA20'] - current_data['Close']) / current_data['SMA20']
        score = score * (1 + price_deviation) if price_deviation > 0 else score
    else: # Hold signal
        signal = -1
        score = 0.0

    return signal, score
