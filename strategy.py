
import pandas as pd

def add_indicators(df):
    df['sma20'] = df['Close'].rolling(20).mean()
    df['sma50'] = df['Close'].rolling(50).mean()
    return df

def compute_signal(df):
    df = add_indicators(df)
    if len(df) < 50 or df['sma20'].isna().any() or df['sma50'].isna().any():
        return 0
    return 1 if df['sma20'].iloc[-1] > df['sma50'].iloc[-1] else 0