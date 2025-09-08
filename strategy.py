import pandas as pd
import numpy as np

def add_indicators(df, adx_period=14, cci_period=20, keltner_period=20, keltner_mult=2):
    # Simple Moving Averages
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

    # RSI
    rsi_length = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_length).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # Volume
    df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA20'].replace(0, 1)  # avoid div by zero

    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()

    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    denom = (high_14 - low_14).replace(0, 1)
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / denom)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # ADX
    tr = pd.concat([
        df['High'] - df['Low'],
        np.abs(df['High'] - df['Close'].shift()),
        np.abs(df['Low'] - df['Close'].shift())
    ], axis=1).max(axis=1)

    atr = tr.rolling(window=adx_period).mean()
    up_move = df['High'].diff()
    down_move = df['Low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * pd.Series(plus_dm).rolling(window=adx_period).sum() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=adx_period).sum() / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
    df['ADX'] = pd.Series(dx).rolling(window=adx_period).mean()

    # CCI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    cci_denom = tp.rolling(window=cci_period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).replace(0, 1)
    df['CCI'] = (tp - tp.rolling(window=cci_period).mean()) / (0.015 * cci_denom)

    # Keltner Channel
    ema = df['Close'].ewm(span=keltner_period).mean()
    kc_upper = ema + keltner_mult * df['ATR']
    kc_lower = ema - keltner_mult * df['ATR']
    df['KC_Upper'] = kc_upper
    df['KC_Lower'] = kc_lower

    return df


def compute_signal_and_score(df):
    df = add_indicators(df)
    df = df.dropna()

    if df.empty or len(df) < 2:
        return None, 0.0

    current_data = df.iloc[-1]
    previous_data = df.iloc[-2]

    signal = -1
    score = 0.0
    indicator_score = 0
    max_possible_score = 0

    # SMA cross
    max_possible_score += 30
    if previous_data['SMA9'] < previous_data['SMA20'] and current_data['SMA9'] > current_data['SMA20']:
        indicator_score += 30
    elif previous_data['SMA9'] > previous_data['SMA20'] and current_data['SMA9'] < current_data['SMA20']:
        indicator_score -= 30

    # MACD cross
    max_possible_score += 20
    if current_data['MACD'] > current_data['MACD_Signal'] and previous_data['MACD'] <= previous_data['MACD_Signal']:
        indicator_score += 20
    elif current_data['MACD'] < current_data['MACD_Signal'] and previous_data['MACD'] >= previous_data['MACD_Signal']:
        indicator_score -= 20

    # RSI levels
    max_possible_score += 15
    if current_data['RSI'] < 30:
        indicator_score += 15
    elif current_data['RSI'] > 70:
        indicator_score -= 15
    elif 30 <= current_data['RSI'] <= 70:
        max_possible_score -= 15

    # Bollinger Bands
    max_possible_score += 15
    bb_range = current_data['BB_Upper'] - current_data['BB_Lower']
    bb_range = bb_range if bb_range != 0 else 1
    bb_position = (current_data['Close'] - current_data['BB_Lower']) / bb_range
    if bb_position < 0.2:
        indicator_score += 15
    elif bb_position > 0.8:
        indicator_score -= 15

    # Volume ratio
    max_possible_score += 10
    if current_data['Volume_Ratio'] > 1.5:
        if indicator_score > 0:
            indicator_score += 10
        elif indicator_score < 0:
            indicator_score -= 10

    # Stochastic oscillator
    max_possible_score += 10
    if current_data['Stoch_K'] < 20 and current_data['Stoch_D'] < 20:
        indicator_score += 10
    elif current_data['Stoch_K'] > 80 and current_data['Stoch_D'] > 80:
        indicator_score -= 10

    # ADX and MACD trend confirmation
    max_possible_score += 10
    if current_data['ADX'] > 25:
        indicator_score += 5
    if current_data['MACD'] > current_data['MACD_Signal']:
        indicator_score += 5

    # CCI momentum indicator
    max_possible_score += 10
    if current_data['CCI'] > 100:
        indicator_score += 10
    elif current_data['CCI'] < -100:
        indicator_score -= 10

    # Keltner Channel breakout
    max_possible_score += 10
    if current_data['Close'] > current_data['KC_Upper']:
        indicator_score += 10
    elif current_data['Close'] < current_data['KC_Lower']:
        indicator_score -= 10

    # Determine signal and score with lowered threshold
    if indicator_score >= 15:
        signal = 1
        score = min(1.0, max(0.0, indicator_score / max_possible_score))
        price_deviation = (current_data['Close'] - current_data['SMA20']) / current_data['SMA20']
        score = score * (1 + price_deviation) if price_deviation > 0 else score
    elif indicator_score <= -15:
        signal = 0
        score = min(1.0, max(0.0, abs(indicator_score) / max_possible_score))
        price_deviation = (current_data['SMA20'] - current_data['Close']) / current_data['SMA20']
        score = score * (1 + price_deviation) if price_deviation > 0 else score
    else:
        signal = -1
        score = 0.0

    return signal, score
