# Keep all imports the same as before
import streamlit as st
import mysql.connector
from mysql.connector import Error
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto Predictor", layout="wide")

@st.cache_resource
def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='crypto',
            user='root',
            password='root'
        )
        if connection.is_connected():
            db_Info = connection.get_server_info()
            st.success(f"✅ Connected to MySQL Server version {db_Info}")
            cursor = connection.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()
            st.info(f"Using database: `{record[0]}`")
        return connection
    except Error as e:
        st.error(f"❌ MySQL connection error: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_crypto_data(api_key):
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    parameters = {'start': '1', 'limit': '100', 'convert': 'USD'}
    headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': api_key}
    try:
        response = requests.get(url, headers=headers, params=parameters)
        response.raise_for_status()
        return response.json()['data']
    except Exception as e:
        st.error(f"API error: {e}")
        return []

def get_historical_data(symbol):
    dates = pd.date_range(end=datetime.today(), periods=100).to_pydatetime().tolist()
    prices = np.random.uniform(low=20, high=200, size=len(dates))
    return pd.DataFrame({'date': dates, 'price': prices})

def get_current_price_data(symbol):
    dates = pd.date_range(end=datetime.today(), periods=24, freq='H').to_pydatetime().tolist()
    prices = np.random.uniform(low=20, high=200, size=len(dates))
    return pd.DataFrame({'date': dates, 'price': prices})

def predict_prices_lstm(prices):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices['price'].values.reshape(-1, 1))
    sequence_length = 60
    X_train, y_train = [], []
    for i in range(sequence_length, len(scaled_data)):
        X_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    test_data = scaled_data[-sequence_length:]
    test_data = np.reshape(test_data, (1, test_data.shape[0], 1))
    predictions = []
    for i in range(30):
        pred = model.predict(test_data, verbose=0)
        predictions.append(pred[0, 0])
        test_data = np.append(test_data[:, 1:, :], [[[pred[0, 0]]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_dates = [datetime.today() + timedelta(days=i) for i in range(1, 31)]
    pred_df = pd.DataFrame({'date': future_dates, 'predicted_price_lstm': predictions.flatten()})
    return pred_df, model, X_train, y_train, scaler

def display_crypto_data():
    st.title("💰 Cryptocurrency Price Prediction (LSTM)")
    create_connection()

    api_key ='d5963f6a-b053-4e45-b9ba-de22d7a61e84'
    crypto_data = fetch_crypto_data(api_key)

    if not crypto_data:
        return

    coin_names = [coin['name'] for coin in crypto_data]
    selected_coin = st.sidebar.selectbox("🪙 Choose a cryptocurrency", coin_names)

    if selected_coin:
        coin = next(coin for coin in crypto_data if coin['name'] == selected_coin)
        st.header(f"📊 {coin['name']} ({coin['symbol']})")
        st.metric("Current Price", f"${coin['quote']['USD']['price']:.2f}")

        historical = get_historical_data(coin['symbol'])
        current = get_current_price_data(coin['symbol'])
        prediction, model, X_train, y_train, scaler = predict_prices_lstm(historical)

        # Error Metrics
        preds_train = model.predict(X_train, verbose=0)
        preds_train = scaler.inverse_transform(preds_train)
        y_true = scaler.inverse_transform(y_train.reshape(-1, 1))
        mse = mean_squared_error(y_true, preds_train)
        mae = mean_absolute_error(y_true, preds_train)
        rmse = math.sqrt(mse)

        st.subheader("📏 Model Accuracy")
        st.info(f"MSE: {mse:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

        # Combine data for future plot
        full_data = pd.concat([historical, prediction]).reset_index(drop=True)
        full_data['predicted_price_lstm'][:len(historical)] = np.nan

        # 📈 Section 1: Historical Price
        with st.expander("📈 Historical Prices"):
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=historical['date'], y=historical['price'],
                                      mode='lines+markers', name='Historical Price',
                                      line=dict(color='cyan', width=2)))
            fig1.update_layout(template='plotly_dark', xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig1, use_container_width=True)

        # 📉 Section 2: Current Price (24-Hour)
        with st.expander("📉 Current Price (24-Hour Window)"):
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=current['date'], y=current['price'],
                                      mode='lines+markers', name='Current Price',
                                      line=dict(color='orange', width=2)))
            fig2.update_layout(template='plotly_dark', xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig2, use_container_width=True)

        # 🔮 Section 3: Future Price Prediction
        with st.expander("🔮 Future Price Prediction (30 Days)"):
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=full_data['date'], y=full_data['price'],
                                      mode='lines', name='Actual Price', line=dict(color='lightblue')))
            fig3.add_trace(go.Scatter(x=full_data['date'], y=full_data['predicted_price_lstm'],
                                      mode='lines+markers', name='Predicted Price',
                                      line=dict(color='red', dash='dash')))
            fig3.update_layout(template='plotly_dark', xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig3, use_container_width=True)

# Run app
if __name__ == "__main__":
    display_crypto_data()
