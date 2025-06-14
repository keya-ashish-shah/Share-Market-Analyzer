import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Page title
st.title("📈 Stock Price Predictor (LSTM Model)")

# Sidebar inputs
ticker = st.text_input("Enter stock symbol (e.g. AAPL, TSLA, NIFTY.NS)", "AAPL")
start_date = st.date_input("Select start date", pd.to_datetime("2023-01-01"))
end_date = st.date_input("Select end date", pd.to_datetime("2025-06-01"))


# Button to start prediction
if st.button("Predict Future Prices"):

    st.info(f"Fetching data for **{ticker}** from **{start_date}** to **{end_date}**...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        st.error("No data found. Please check the stock symbol.")
    else:
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        # Create sequences
        def create_sequences(data, seq_len):
            X, y = [], []
            for i in range(seq_len, len(data)):
                X.append(data[i-seq_len:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        seq_len = 60
        X, y = create_sequences(scaled_data, seq_len)
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]

        # Define LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        # Predict next 15 days
        last_seq = scaled_data[-seq_len:]
        future_predictions = []
        current_seq = last_seq.copy()
        for _ in range(15):
            pred = model.predict(current_seq.reshape(1, seq_len, 1), verbose=0)
            future_predictions.append(pred[0, 0])
            current_seq = np.append(current_seq[1:], pred, axis=0)

        future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=15)

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index[-100:], df['Close'].values[-100:], label='Past Prices')
        ax.plot(future_dates, future_prices, label='Predicted (Next 15 Days)', color='orange')
        ax.set_title(f'{ticker} Price Forecast')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

                # Create DataFrame for download
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': future_prices.flatten()
        })

        st.subheader("📥 Download Predicted Prices")

        # CSV download
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"{ticker}_15_day_forecast.csv",
            mime='text/csv'
        )

        # Excel download
        from io import BytesIO
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            forecast_df.to_excel(writer, index=False, sheet_name="Forecast")
        st.download_button(
            label="Download as Excel",
            data=excel_buffer.getvalue(),
            file_name=f"{ticker}_15_day_forecast.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

