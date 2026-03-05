import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from tensorflow.keras.models import load_model
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Trend AI - LSTM Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Stock Trend AI - LSTM Dashboard")
st.markdown("---")

# Paths
MODELS_DIR = '../models'
DATA_PATH = '../data/NIFTY 500-10-02-2025-to-10-02-2026.csv'

# Load saved model and config
@st.cache_resource
def load_lstm_model():
    model_path = os.path.join(MODELS_DIR, 'lstm_model.keras')
    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    return None

@st.cache_resource
def load_scaler():
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None

@st.cache_data
def load_model_config():
    config_path = os.path.join(MODELS_DIR, 'model_config.pkl')
    if os.path.exists(config_path):
        return joblib.load(config_path)
    return None

# Load data from file
@st.cache_data
def load_data_from_path(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # Try multiple date formats
    for date_format in ['%d-%b-%y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format=date_format)
            break
        except:
            continue
    else:
        df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Next_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    return df

# Find date column in dataframe
def find_date_column(df):
    # Common date column names
    date_names = ['Date', 'date', 'DATE', 'Datetime', 'datetime', 'DATETIME', 
                  'Timestamp', 'timestamp', 'TIMESTAMP', 'Time', 'time', 'TIME',
                  'trade_date', 'Trade_Date', 'trading_date']
    
    for col in date_names:
        if col in df.columns:
            return col
    
    # Check first column if it looks like a date
    first_col = df.columns[0]
    try:
        pd.to_datetime(df[first_col].head())
        return first_col
    except:
        pass
    
    # Check if index is datetime
    if df.index.name and 'date' in df.index.name.lower():
        return None  # Already indexed by date
    
    return None

# Load data from uploaded file
def load_data_from_upload(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    
    # Find and process date column
    date_col = find_date_column(df)
    
    if date_col is None:
        # Try using first column as index
        df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        df.columns = df.columns.str.strip()
        df.index.name = 'Date'
    else:
        # Try multiple date formats
        parsed = False
        for date_format in ['%d-%b-%y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']:
            try:
                df[date_col] = pd.to_datetime(df[date_col], format=date_format)
                parsed = True
                break
            except:
                continue
        
        if not parsed:
            df[date_col] = pd.to_datetime(df[date_col])
        
        df.set_index(date_col, inplace=True)
        df.index.name = 'Date'
    
    # Standardize column names for OHLC
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'open' in col_lower and 'Open' not in df.columns:
            column_mapping[col] = 'Open'
        elif 'high' in col_lower and 'High' not in df.columns:
            column_mapping[col] = 'High'
        elif 'low' in col_lower and 'Low' not in df.columns:
            column_mapping[col] = 'Low'
        elif 'close' in col_lower and 'Close' not in df.columns:
            column_mapping[col] = 'Close'
    
    df.rename(columns=column_mapping, inplace=True)
    
    # Verify required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Next_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    return df

# Create sequences for LSTM
def create_sequences(X, y, lookback=10):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

# Load resources
lstm_model = load_lstm_model()
scaler = load_scaler()
model_config = load_model_config()

# Sidebar - Dataset Upload
st.sidebar.header("📁 Dataset")
data_source = st.sidebar.radio("Select Data Source:", ["Default Dataset", "Upload CSV"])

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df = load_data_from_upload(uploaded_file)
            st.sidebar.success(f"✅ Loaded {len(df)} records")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
            df = load_data_from_path(DATA_PATH)
    else:
        st.sidebar.info("CSV should have: Date/date column + Open, High, Low, Close")
        df = load_data_from_path(DATA_PATH)
else:
    df = load_data_from_path(DATA_PATH)

st.sidebar.markdown("---")

# Sidebar - Model Info
st.sidebar.header("🤖 Model Information")
if model_config:
    st.sidebar.metric("Lookback Period", model_config['lookback'])
    st.sidebar.metric("MSE", f"{model_config['mse']:.6f}")
    st.sidebar.metric("R²", f"{model_config['r2']:.4f}")
    st.sidebar.metric("Direction Accuracy", f"{model_config['accuracy']*100:.2f}%")
    st.sidebar.markdown("---")
    st.sidebar.write("**Features:**")
    for feat in model_config['feature_cols']:
        st.sidebar.write(f"• {feat}")
else:
    st.sidebar.warning("Model config not found")

# Check if model is loaded
if lstm_model is None:
    st.error("❌ LSTM model not found in models directory. Please train and export the model first.")
    st.stop()

st.sidebar.success("✅ Model loaded successfully")

# Data Overview Section
st.header("📊 Data Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", len(df))
with col2:
    st.metric("Latest Close", f"₹{df['Close'].iloc[-1]:,.2f}")
with col3:
    st.metric("Avg Daily Return", f"{df['Next_Return'].mean()*100:.3f}%")
with col4:
    st.metric("Return Std Dev", f"{df['Next_Return'].std()*100:.3f}%")

st.dataframe(df.tail(10), use_container_width=True)

# Price Charts
st.header("📈 Price Analysis")
col1, col2 = st.columns(2)

with col1:
    # Candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    )])
    fig.update_layout(title='Stock Price Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Close price with MA20
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='20-Day MA', line=dict(dash='dash')))
    fig.update_layout(title='Close Price with 20-Day Moving Average', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)

# LSTM Predictions Section
st.header("🤖 LSTM Model Predictions")
st.markdown("---")

if st.button("🔮 Generate Predictions", type="primary"):
    with st.spinner("Generating predictions with loaded LSTM model..."):
        # Prepare data
        feature_cols = model_config['feature_cols'] if model_config else ['Open', 'High', 'Low', 'Close', 'MA20']
        lookback = model_config['lookback'] if model_config else 10
        
        df_reg = df.dropna()
        X_reg = df_reg[feature_cols].values
        y_reg = df_reg['Next_Return'].values
        
        # Use same train/test split as training (80/20)
        split_idx = int(len(X_reg) * 0.8)
        X_test = X_reg[split_idx:]
        y_test = y_reg[split_idx:]
        
        # Scale data
        X_test_scaled = scaler.transform(X_test)
        
        # Create sequences
        X_test_lstm, y_test_lstm = create_sequences(X_test_scaled, y_test, lookback)
        
        st.info(f"Test data shape: {X_test_lstm.shape}")
        
        # Make predictions
        y_pred = lstm_model.predict(X_test_lstm, verbose=0).flatten()
        
        # Metrics
        mse = mean_squared_error(y_test_lstm, y_pred)
        r2 = r2_score(y_test_lstm, y_pred)
        
        # Direction accuracy
        y_test_dir = (y_test_lstm > 0).astype(int)
        y_pred_dir = (y_pred > 0).astype(int)
        accuracy = accuracy_score(y_test_dir, y_pred_dir)
        
        st.success("✅ Predictions generated!")
        
        # Display metrics
        st.subheader("📊 Model Performance Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Squared Error (MSE)", f"{mse:.6f}")
        with col2:
            st.metric("R-squared (R²)", f"{r2:.4f}")
        with col3:
            st.metric("Direction Accuracy", f"{accuracy*100:.2f}%")
        
        # Prediction vs Actual plot
        st.subheader("📉 Actual vs Predicted Returns")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_test_lstm, mode='markers', name='Actual', marker=dict(size=5, color='blue')))
            fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted', line=dict(color='red')))
            fig.update_layout(title='Actual vs Predicted Returns', xaxis_title='Sample', yaxis_title='Return')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter plot - actual vs predicted
            fig = px.scatter(x=y_test_lstm, y=y_pred, labels={'x': 'Actual Return', 'y': 'Predicted Return'})
            fig.add_trace(go.Scatter(x=[-0.1, 0.1], y=[-0.1, 0.1], mode='lines', name='Perfect Prediction', line=dict(dash='dash', color='green')))
            fig.update_layout(title='Prediction Accuracy Scatter')
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution of predictions
        st.subheader("📊 Return Distributions")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(x=y_test_lstm, nbins=50, title='Actual Returns Distribution')
            fig.update_layout(xaxis_title='Return', yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(x=y_pred, nbins=50, title='Predicted Returns Distribution')
            fig.update_layout(xaxis_title='Return', yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
        
        # Direction Prediction Analysis
        st.subheader("🎯 Direction Prediction Analysis")
        
        correct_up = np.sum((y_test_dir == 1) & (y_pred_dir == 1))
        correct_down = np.sum((y_test_dir == 0) & (y_pred_dir == 0))
        wrong_up = np.sum((y_test_dir == 0) & (y_pred_dir == 1))
        wrong_down = np.sum((y_test_dir == 1) & (y_pred_dir == 0))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("True Positive (Up→Up)", correct_up)
        with col2:
            st.metric("True Negative (Down→Down)", correct_down)
        with col3:
            st.metric("False Positive (Down→Up)", wrong_up)
        with col4:
            st.metric("False Negative (Up→Down)", wrong_down)
        
        # Confusion matrix visualization
        confusion_data = [[correct_down, wrong_up], [wrong_down, correct_up]]
        fig = px.imshow(confusion_data, 
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Down', 'Up'],
                        y=['Down', 'Up'],
                        text_auto=True,
                        color_continuous_scale='Blues')
        fig.update_layout(title='Confusion Matrix')
        st.plotly_chart(fig, use_container_width=True)

# Latest Prediction Section
st.header("🔮 Latest Market Prediction")
st.markdown("---")

if st.button("📊 Predict Next Return", type="secondary"):
    with st.spinner("Predicting..."):
        feature_cols = model_config['feature_cols'] if model_config else ['Open', 'High', 'Low', 'Close', 'MA20']
        lookback = model_config['lookback'] if model_config else 10
        
        # Get latest data
        df_latest = df.dropna().tail(lookback)
        X_latest = df_latest[feature_cols].values
        X_latest_scaled = scaler.transform(X_latest)
        X_latest_seq = X_latest_scaled.reshape(1, lookback, len(feature_cols))
        
        # Predict
        prediction = lstm_model.predict(X_latest_seq, verbose=0)[0][0]
        direction = "📈 UP" if prediction > 0 else "📉 DOWN"
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Next Return", f"{prediction*100:.4f}%")
        with col2:
            st.metric("Predicted Direction", direction)
        
        st.info(f"Based on the last {lookback} days of data ending {df_latest.index[-1].strftime('%Y-%m-%d')}")

# Footer
st.markdown("---")
st.markdown("**Stock Trend AI** | Built with Streamlit & TensorFlow | Model loaded from `models/lstm_model.keras`")
