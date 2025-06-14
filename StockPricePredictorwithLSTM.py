import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Get User Input
ticker = input("Enter stock symbol (e.g. AAPL): ")
period = input("Enter historical period (e.g. 6mo, 1y, 2y): ")

# Step 2: Fetch historical stock data
df = yf.download(ticker, period=period)
if df.empty:
    print("No data found for this ticker.")
    exit()

# Step 3: Use only closing price
data = df['Close'].values.reshape(-1, 1)

# Step 4: Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Step 5: Create sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

seq_len = 60
X, y = create_sequences(scaled_data, seq_len)

# Step 6: Train/test split
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]

# Step 7: Define and train LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Step 8: Predict next 15 days
last_seq = scaled_data[-seq_len:]
predictions = []

current_seq = last_seq.copy()
for _ in range(15):
    pred = model.predict(current_seq.reshape(1, seq_len, 1), verbose=0)
    predictions.append(pred[0, 0])
    current_seq = np.append(current_seq[1:], pred, axis=0)

# Step 9: Inverse transform predictions
future_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Step 10: Plot results
plt.figure(figsize=(12, 6))
plt.plot(df.index[-100:], df['Close'].values[-100:], label='Past Prices')
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=15)
plt.plot(future_dates, future_prices, label='Predicted Prices (Next 15 Days)', color='orange')
plt.title(f'{ticker} Price Forecast')
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
