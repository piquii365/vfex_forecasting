import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="VFEX Stock Forecaster", layout="wide")

# Title
st.title("🇿🇼 VFEX Stock Price Forecasting System")

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.selectbox("Select Stock", ['SEED.VFEX', 'PPCZ.VFEX', 'ZBFH.VFEX'])
model_type = st.sidebar.selectbox("Model", ['Random Forest', 'XGBoost'])


# Load data
@st.cache_data
def load_data(ticker):
    df = pd.read_csv(f'data/processed/{ticker}_features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df


df = load_data(ticker.replace('.', '_'))

# Display data
st.subheader("📊 Historical Data")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
col2.metric("Daily Change", f"{df['simple_return'].iloc[-1]:.2%}")
col3.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
col4.metric("20-Day Volatility", f"{df['volatility_20d'].iloc[-1]:.2%}")

# Price chart
st.subheader("Price History")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='Price'
))
fig.update_layout(xaxis_title='Date', yaxis_title='Price (USD)', height=500)
st.plotly_chart(fig, use_container_width=True)


# Load model
@st.cache_resource
def load_model(model_name):
    return joblib.load(f'models/{model_name.lower().replace(" ", "_")}.pkl')


if st.sidebar.button("Generate Forecast"):
    with st.spinner("Generating forecast..."):
        model = load_model(model_type)

        # Prepare latest features
        latest_features = df.iloc[-1:].drop(['Date', 'Close', 'Open', 'High', 'Low'], axis=1)

        # Predict
        prediction = model.predict(latest_features)[0]

        # Display forecast
        st.subheader("🔮 Forecast")
        col1, col2 = st.columns(2)

        direction = "📈 UP" if prediction > 0 else "📉 DOWN"
        col1.metric("Predicted Direction", direction)
        col2.metric("Predicted Return", f"{prediction:.2%}")

        # Confidence
        st.info("⚠️ This is a prediction based on historical patterns. Always do your own research before investing.")

# Footer
st.markdown("---")
st.markdown("Built for VFEX investor decision-making | Data as of " + df['Date'].max().strftime('%Y-%m-%d'))
