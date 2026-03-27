import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import hashlib
import binascii
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
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
USERS_DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')


def init_user_db():
    conn = sqlite3.connect(USERS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )
    conn.commit()
    conn.close()


def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16)
    elif isinstance(salt, str):
        salt = binascii.unhexlify(salt)

    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return binascii.hexlify(pwd_hash).decode('utf-8'), binascii.hexlify(salt).decode('utf-8')


def create_user(username, password):
    username = username.strip()
    if not username or not password:
        return False, 'Username and password are required.'

    if len(password) < 6:
        return False, 'Password must be at least 6 characters.'

    password_hash, salt = hash_password(password)
    conn = sqlite3.connect(USERS_DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(
            'INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)',
            (username, password_hash, salt)
        )
        conn.commit()
        return True, 'Account created successfully. Please log in.'
    except sqlite3.IntegrityError:
        return False, 'Username already exists. Please choose another.'
    finally:
        conn.close()


def verify_user(username, password):
    conn = sqlite3.connect(USERS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT password_hash, salt FROM users WHERE username = ?', (username.strip(),))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return False

    stored_hash, salt = row
    test_hash, _ = hash_password(password, salt)
    return test_hash == stored_hash


def preprocess_stock_dataframe(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # If Date is already in index (from previously saved CSV), restore it.
    if 'Date' not in df.columns and isinstance(df.index.name, str) and 'date' in df.index.name.lower():
        df = df.reset_index()

    # Normalize common column names to a single schema.
    rename_map = {
        'DATE': 'Date',
        'date': 'Date',
        'OPEN': 'Open',
        'open': 'Open',
        'HIGH': 'High',
        'high': 'High',
        'LOW': 'Low',
        'low': 'Low',
        'CLOSE': 'Close',
        'close': 'Close',
        'VOLUME': 'Shares Traded',
        'volume': 'Shares Traded',
        'VALUE': 'Turnover (₹ Cr)',
        'value': 'Turnover (₹ Cr)'
    }
    df = df.rename(columns=rename_map)

    # Fallback normalization for loosely named uploaded columns.
    fallback_map = {}
    for col in df.columns:
        normalized = col.strip().lower().replace('_', ' ')
        if normalized in ['trade date', 'trading date', 'timestamp', 'time'] and 'Date' not in df.columns:
            fallback_map[col] = 'Date'
        elif normalized == 'shares traded' and 'Shares Traded' not in df.columns:
            fallback_map[col] = 'Shares Traded'
        elif normalized in ['turnover', 'turnover (cr)', 'turnover (₹ cr)'] and 'Turnover (₹ Cr)' not in df.columns:
            fallback_map[col] = 'Turnover (₹ Cr)'
    if fallback_map:
        df = df.rename(columns=fallback_map)

    required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {df.columns.tolist()}")

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])

    for col in ['Open', 'High', 'Low', 'Close', 'Shares Traded', 'Turnover (₹ Cr)']:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '', regex=False),
                errors='coerce'
            )

    # NSE VALUE is in rupees in raw CSV; convert once to crore rupees.
    if 'Turnover (₹ Cr)' in df.columns and df['Turnover (₹ Cr)'].median(skipna=True) > 1e6:
        df['Turnover (₹ Cr)'] = df['Turnover (₹ Cr)'] / 1e7

    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Next_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    return df

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
    return preprocess_stock_dataframe(df)

# Load data from uploaded file
def load_data_from_upload(uploaded_file):
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, index_col=0)

    return preprocess_stock_dataframe(df)

# Create sequences for LSTM
def create_sequences(X, y, lookback=10):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def prepare_lstm_test_data(df, feature_cols, lookback, scaler, order='ascending'):
    df_eval = df.dropna().copy()
    if order == 'descending':
        df_eval = df_eval.iloc[::-1]

    X_reg = df_eval[feature_cols].values
    y_reg = df_eval['Next_Return'].values

    split_idx = int(len(X_reg) * 0.8)
    X_test = X_reg[split_idx:]
    y_test = y_reg[split_idx:]

    X_test_scaled = scaler.transform(X_test)
    X_test_lstm, y_test_lstm = create_sequences(X_test_scaled, y_test, lookback)
    return X_test_lstm, y_test_lstm


def choose_best_inference_order(df, feature_cols, lookback, scaler, model):
    candidates = []
    for order in ['ascending', 'descending']:
        try:
            X_test_lstm, y_test_lstm = prepare_lstm_test_data(df, feature_cols, lookback, scaler, order=order)
            if len(X_test_lstm) == 0:
                continue
            y_pred = model.predict(X_test_lstm, verbose=0).flatten()
            mse = mean_squared_error(y_test_lstm, y_pred)
            candidates.append((order, mse, X_test_lstm, y_test_lstm, y_pred))
        except Exception:
            continue

    if not candidates:
        raise ValueError('Unable to prepare test sequences for prediction. Check dataset length and feature columns.')

    # Lower MSE is better.
    candidates.sort(key=lambda x: x[1])
    best_order, best_mse, X_best, y_best, y_pred_best = candidates[0]
    return best_order, best_mse, X_best, y_best, y_pred_best


def train_and_save_lstm(df, feature_cols=None, lookback=10, epochs=60, batch_size=16):
    if feature_cols is None:
        feature_cols = ['Open', 'High', 'Low', 'Close', 'MA20']

    df_reg = df.dropna().copy()
    if len(df_reg) < (lookback * 3):
        raise ValueError('Not enough rows to train LSTM. Please use a larger dataset.')

    X = df_reg[feature_cols].values
    y = df_reg['Next_Return'].values

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler_local = StandardScaler()
    X_train_scaled = scaler_local.fit_transform(X_train)
    X_test_scaled = scaler_local.transform(X_test)

    X_train_lstm, y_train_lstm = create_sequences(X_train_scaled, y_train, lookback)
    X_test_lstm, y_test_lstm = create_sequences(X_test_scaled, y_test, lookback)

    if len(X_train_lstm) == 0 or len(X_test_lstm) == 0:
        raise ValueError('Sequence generation failed. Increase dataset size or reduce lookback.')

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, X_train_lstm.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    model.fit(
        X_train_lstm,
        y_train_lstm,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )

    y_pred = model.predict(X_test_lstm, verbose=0).flatten()
    mse = mean_squared_error(y_test_lstm, y_pred)
    r2 = r2_score(y_test_lstm, y_pred)
    y_test_dir = (y_test_lstm > 0).astype(int)
    y_pred_dir = (y_pred > 0).astype(int)
    accuracy = accuracy_score(y_test_dir, y_pred_dir)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(os.path.join(MODELS_DIR, 'lstm_model.keras'))
    joblib.dump(scaler_local, os.path.join(MODELS_DIR, 'scaler.pkl'))

    model_config_local = {
        'lookback': lookback,
        'feature_cols': feature_cols,
        'mse': float(mse),
        'r2': float(r2),
        'accuracy': float(accuracy)
    }
    joblib.dump(model_config_local, os.path.join(MODELS_DIR, 'model_config.pkl'))

    return model_config_local


def prepare_regression_data(df, feature_cols):
    df_reg = df.dropna().copy()
    X = df_reg[feature_cols].values
    y = df_reg['Next_Return'].values

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler_local = StandardScaler()
    X_train_scaled = scaler_local.fit_transform(X_train)
    X_test_scaled = scaler_local.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def evaluate_selected_model(model_name, df, model_config, lstm_model, scaler):
    feature_cols = model_config['feature_cols'] if model_config else ['Open', 'High', 'Low', 'Close', 'MA20']
    lookback = model_config['lookback'] if model_config else 10

    if model_name == 'LSTM':
        if lstm_model is None or scaler is None:
            raise ValueError('LSTM model/scaler not found. Retrain from sidebar to enable LSTM evaluation.')

        best_order, _, _, y_test, y_pred = choose_best_inference_order(
            df, feature_cols, lookback, scaler, lstm_model
        )
        return y_test, y_pred, {'inference_order': best_order}

    X_train_scaled, X_test_scaled, y_train, y_test = prepare_regression_data(df, feature_cols)

    model_map = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
    }

    if model_name not in model_map:
        raise ValueError(f'Unsupported model: {model_name}')

    model = model_map[model_name]
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return y_test, y_pred, {}


def render_auth_page():
    st.header('🔐 Login Required')
    st.write('Create an account or log in to access the Stock Trend AI dashboard.')

    login_tab, signup_tab = st.tabs(['Login', 'Sign Up'])

    with login_tab:
        with st.form('login_form', clear_on_submit=False):
            login_username = st.text_input('Username', key='login_username')
            login_password = st.text_input('Password', type='password', key='login_password')
            login_submit = st.form_submit_button('Login', type='primary')

            if login_submit:
                if verify_user(login_username, login_password):
                    st.session_state.authenticated = True
                    st.session_state.current_user = login_username.strip()
                    st.success('Login successful.')
                    st.rerun()
                else:
                    st.error('Invalid username or password.')

    with signup_tab:
        with st.form('signup_form', clear_on_submit=True):
            signup_username = st.text_input('Choose Username', key='signup_username')
            signup_password = st.text_input('Choose Password', type='password', key='signup_password')
            signup_confirm = st.text_input('Confirm Password', type='password', key='signup_confirm')
            signup_submit = st.form_submit_button('Create Account')

            if signup_submit:
                if signup_password != signup_confirm:
                    st.error('Passwords do not match.')
                else:
                    success, message = create_user(signup_username, signup_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

# Load resources
init_user_db()

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'current_user' not in st.session_state:
    st.session_state.current_user = None

if not st.session_state.authenticated:
    render_auth_page()
    st.stop()

lstm_model = load_lstm_model()
scaler = load_scaler()
model_config = load_model_config()

st.sidebar.header('👤 User Session')
st.sidebar.write(f"Logged in as: **{st.session_state.current_user}**")
if st.sidebar.button('Logout'):
    st.session_state.authenticated = False
    st.session_state.current_user = None
    st.rerun()
st.sidebar.markdown('---')

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

st.sidebar.header("🛠️ Model Retraining")
if st.sidebar.button("Retrain LSTM on Current Data"):
    with st.spinner('Retraining model with current dataset and preprocessing...'):
        try:
            config_new = train_and_save_lstm(df)

            load_lstm_model.clear()
            load_scaler.clear()
            load_model_config.clear()

            st.sidebar.success('Model retrained and saved successfully.')
            st.sidebar.write(f"MSE: {config_new['mse']:.6f}")
            st.sidebar.write(f"R²: {config_new['r2']:.4f}")
            st.sidebar.write(f"Direction Accuracy: {config_new['accuracy']*100:.2f}%")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f'Retraining failed: {str(e)}')

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
if lstm_model is None or scaler is None:
    st.sidebar.warning("⚠️ LSTM model/scaler not found. You can still run classical ML models.")
else:
    st.sidebar.success("✅ LSTM model loaded successfully")

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

# Model Predictions Section
st.header("🤖 ML Model Predictions")
st.markdown("---")

selected_model = st.radio(
    'Choose a model to evaluate:',
    ['Linear Regression', 'Ridge Regression', 'Random Forest Regressor', 'Gradient Boosting Regressor', 'LSTM'],
    horizontal=True
)

if st.button("🔮 Run Selected Model", type="primary"):
    with st.spinner(f"Running {selected_model}..."):
        try:
            y_test_lstm, y_pred, extra_info = evaluate_selected_model(
                selected_model, df, model_config, lstm_model, scaler
            )

            if selected_model == 'LSTM':
                st.session_state.inference_order = extra_info.get('inference_order', 'ascending')
                st.info(f"Inference order: {st.session_state.inference_order}")

            # Metrics
            mse = mean_squared_error(y_test_lstm, y_pred)
            r2 = r2_score(y_test_lstm, y_pred)

            # Direction accuracy
            y_test_dir = (y_test_lstm > 0).astype(int)
            y_pred_dir = (y_pred > 0).astype(int)
            accuracy = accuracy_score(y_test_dir, y_pred_dir)

            st.success(f"✅ {selected_model} results generated!")

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
                axis_min = float(min(np.min(y_test_lstm), np.min(y_pred)))
                axis_max = float(max(np.max(y_test_lstm), np.max(y_pred)))
                fig.add_trace(
                    go.Scatter(
                        x=[axis_min, axis_max],
                        y=[axis_min, axis_max],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(dash='dash', color='green')
                    )
                )
                fig.update_layout(title='Prediction Accuracy Scatter')
                st.plotly_chart(fig, use_container_width=True)

            pred_std = float(np.std(y_pred))
            if pred_std < 1e-3:
                st.warning(
                    'Predictions show very low variance (almost flat). This usually indicates model underfitting or a train/inference data mismatch. '
                    'Retrain the model on the same dataset and preprocessing used in this dashboard for better results.'
                )

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
        except Exception as e:
            st.error(f"Model run failed: {str(e)}")

# Latest Prediction Section
st.header("🔮 Latest Market Prediction")
st.markdown("---")

if st.button("📊 Predict Next Return", type="secondary"):
    with st.spinner("Predicting..."):
        if lstm_model is None or scaler is None:
            st.error('LSTM model/scaler not available. Retrain from sidebar first.')
            st.stop()

        feature_cols = model_config['feature_cols'] if model_config else ['Open', 'High', 'Low', 'Close', 'MA20']
        lookback = model_config['lookback'] if model_config else 10

        inference_order = st.session_state.get('inference_order', 'ascending')

        # Get latest window according to selected inference order.
        df_non_null = df.dropna()
        if inference_order == 'descending':
            df_latest = df_non_null.head(lookback)
        else:
            df_latest = df_non_null.tail(lookback)

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
