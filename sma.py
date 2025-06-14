import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Style for plots
sns.set(style='darkgrid')

# Technical Indicators
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0).flatten()
    loss = np.where(delta < 0, -delta, 0).flatten()
    avg_gain = pd.Series(gain).rolling(window).mean()
    avg_loss = pd.Series(loss).rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data):
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def analyze_stock(ticker, period='6mo'):
    # Fetch data
    data = yf.download(ticker, period=period)
    
    if data.empty:
        print(f"No data found for {ticker}")
        return

    # Calculate indicators
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['Signal'] = calculate_macd(data)

    # Plotting
    plt.figure(figsize=(14, 10))

    # Price + MA
    plt.subplot(3, 1, 1)
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.plot(data['SMA20'], label='SMA 20', color='red', linestyle='--')
    plt.plot(data['SMA50'], label='SMA 50', color='green', linestyle='--')
    plt.title(f'{ticker} Price with Moving Averages')
    plt.legend()

    # RSI
    plt.subplot(3, 1, 2)
    plt.plot(data['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
    plt.title('Relative Strength Index (RSI)')
    plt.legend()

    # MACD
    plt.subplot(3, 1, 3)
    plt.plot(data['MACD'], label='MACD', color='blue')
    plt.plot(data['Signal'], label='Signal Line', color='orange')
    plt.title('MACD')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage
analyze_stock('AAPL')  # You can replace 'AAPL' with any ticker like 'TSLA', 'GOOG', 'NIFTY.NS', etc.
