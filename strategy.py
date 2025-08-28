import pandas as pd
import numpy as np

def add_indicators(df):
    """Adds a 9-period and 20-period Simple Moving Average to the DataFrame."""
    # Calculate a 9-period SMA for short-term trend
    df['SMA9'] = df['Close'].rolling(window=9).mean()
    # Calculate a 20-period SMA for long-term trend
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    return df

def compute_signal_and_score(df):
    """
    Computes a trading signal and a profitability score based on a crossover.
    Returns:
        tuple: (signal, score) where signal is 1 (buy), 0 (sell), or -1 (hold),
               and score is a float representing the potential profitability.
    """
    df = add_indicators(df)
    
    # Drop rows with NaN values (from rolling mean calculations)
    df = df.dropna()

    if df.empty or len(df) < 2:
        return None, 0.0

    current_data = df.iloc[-1]
    previous_data = df.iloc[-2]
    
    signal = -1
    score = 0.0

    # Buy signal: 9-period SMA crosses above 20-period SMA
    if previous_data['SMA9'] < previous_data['SMA20'] and current_data['SMA9'] > current_data['SMA20']:
        signal = 1
        # Calculate a simple score based on how far the price is from the SMA
        score = (current_data['Close'] - current_data['SMA20']) / current_data['SMA20']
        if score < 0: score = 0 # Ensure score is non-negative
    
    # Sell signal: 9-period SMA crosses below 20-period SMA
    elif previous_data['SMA9'] > previous_data['SMA20'] and current_data['SMA9'] < current_data['SMA20']:
        signal = 0
        # Calculate a simple score for selling
        score = (current_data['SMA20'] - current_data['Close']) / current_data['SMA20']
        if score < 0: score = 0
        
    return signal, score
