

import yfinance as yf
from strategy import add_indicators
import matplotlib.pyplot as plt

def backtest(symbol="AAPL", period="2y"):
    df = yf.download(symbol, period=period, interval="1d")
    df = add_indicators(df).dropna()
    df['signal'] = 0
    df.loc[df.sma20 > df.sma50, 'signal'] = 1
    df['returns'] = df['Close'].pct_change().shift(-1)
    df['strategy'] = df['signal'] * df['returns']
    cum_strategy = (1 + df['strategy'].dropna()).cumprod()
    cum_bh = (1 + df['returns'].dropna()).cumprod()
    print(f"Strategy cumulative return: {cum_strategy.iloc[-1]:.4f}")
    print(f"Buy & Hold cumulative return: {cum_bh.iloc[-1]:.4f}")
    plt.figure(figsize=(10,5))
    cum_bh.plot(label='Buy & Hold')
    cum_strategy.plot(label='SMA Strategy')
    plt.legend(); plt.title(f"{symbol} Backtest")
    plt.show()

if __name__ == "__main__":
    backtest()