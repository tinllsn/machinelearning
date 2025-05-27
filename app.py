import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime, timedelta

def download_stock_data(symbol, start_date, end_date, max_retries=3, retry_delay=5):
    """
    Download stock data with retry logic and caching
    """
    # Create cache directory if it doesn't exist
    cache_dir = "stock_data_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Create cache filename
    cache_file = os.path.join(cache_dir, f"{symbol}_{start_date}_{end_date}.csv")
    
    # Check if we have cached data
    if os.path.exists(cache_file):
        # Check if cache is recent (less than 1 day old)
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
        if cache_age < timedelta(days=1):
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
    
    # If no cache or cache is old, download new data
    for attempt in range(max_retries):
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            # Save to cache
            data.to_csv(cache_file)
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                st.error(f"Failed to download data after {max_retries} attempts: {str(e)}")
                return None

model = load_model('D:\Hoc May\Stock_Market_Prediction_ML (1)\Stock Predictions Model.keras')

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'AAPL')
start = '2023-01-01'
end = '2024-12-21'

# Download data with retry logic
data = download_stock_data(stock, start, end)

if data is not None:
    st.subheader('Stock Data')
    st.write(data)

    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    st.subheader('Price vs MA50')
    ma_50_days = data.Close.rolling(50).mean()
    fig1 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, 'r')
    plt.plot(data.Close, 'g')
    plt.show()
    st.pyplot(fig1)

    st.subheader('Price vs MA50 vs MA100')
    ma_100_days = data.Close.rolling(100).mean()
    fig2 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, 'r')
    plt.plot(ma_100_days, 'b')
    plt.plot(data.Close, 'g')
    plt.show()
    st.pyplot(fig2)

    st.subheader('Price vs MA100 vs MA200')
    ma_200_days = data.Close.rolling(200).mean()
    fig3 = plt.figure(figsize=(8,6))
    plt.plot(ma_100_days, 'r')
    plt.plot(ma_200_days, 'b')
    plt.plot(data.Close, 'g')
    plt.show()
    st.pyplot(fig3)

    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i,0])

    x,y = np.array(x), np.array(y)

    predict = model.predict(x)

    scale = 1/scaler.scale_

    predict = predict * scale
    y = y * scale

    st.subheader('Original Price vs Predicted Price')
    fig4 = plt.figure(figsize=(8,6))
    plt.plot(predict, 'r', label='Original Price')
    plt.plot(y, 'g', label='Predicted Price')
    plt.legend()
    plt.show()
    st.pyplot(fig4)
else:
    st.error("Unable to load stock data. Please try again later.")