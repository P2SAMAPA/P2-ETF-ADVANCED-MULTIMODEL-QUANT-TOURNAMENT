import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import requests
import plotly.graph_objects as go
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pytz

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Advanced Alpha Tournament", layout="wide")

TARGET_ETFS = ['TLT', 'TBT', 'VNQ', 'GLD', 'SLV']
YAHOO_MACRO = ['^VIX', '^TNX', 'DX-Y.NYB']
FRED_API_KEY = st.secrets.get("FRED_API_KEY")

# --- 2. DATA ENGINE ---
def get_next_market_date():
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    next_date = now if now.hour < 16 else now + timedelta(days=1)
    while next_date.weekday() >= 5:
        next_date += timedelta(days=1)
    return next_date.strftime('%A, %b %d, %Y')

def fetch_fred_yield(api_key):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=T10Y2Y&api_key={api_key}&file_type=json"
    try:
        r = requests.get(url, timeout=10).json()
        df = pd.DataFrame(r['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df.set_index('date')[['value']].rename(columns={'value': 'T10Y2Y'})
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_master_data(api_key):
    all_tickers = TARGET_ETFS + YAHOO_MACRO
    raw = yf.download(all_tickers, start="2010-01-01", auto_adjust=True)
    prices = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    fred_df = fetch_fred_yield(api_key)
    combined = pd.concat([prices, fred_df], axis=1).ffill().dropna()
    return combined

# --- 3. REINFORCEMENT LEARNING ENV ---
class TradingEnv(gym.Env):
    def __init__(self, df, etfs, feature_cols):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.etfs = etfs
        self.feature_cols = feature_cols
        self.action_space = gym.spaces.Discrete(len(etfs))
        # Hard-coded observation space to match features exactly
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(feature_cols),), 
            dtype=np.float32
        )
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.df[self.feature_cols].iloc[0].values.astype(np.float32)
        return obs, {}

    def step(self, action):
        reward = self.df[self.etfs[action]].iloc[self.current_step]
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self.df[self.feature_cols].iloc[self.current_step].values.astype(np.float32)
        return obs, reward, done, False, {}

# --- 4. DEEP LEARNING ARCHITECTURES ---
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.cnn = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, output_dim)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.cnn(x)).transpose(1, 2)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.d_model = 64 
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.enc = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.enc, num_layers=2)
        self.fc = nn.Linear(self.d_model, output_dim)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

# --- 5. TOURNAMENT ENGINE ---
def run_tournament(data):
    rets = data[TARGET_ETFS].pct_change().dropna()
    features = data.shift(1).dropna()
    idx = rets.index.intersection(features.index)
    X, y = features.loc[idx], rets.loc[idx]
    
    # Save the exact list of columns to ensure consistency
    feature_cols = X.columns.tolist()
    
    seq_len = 10
    X_s, y_s = [], []
    for i in range(len(X) - seq_len):
        X_s.append(X.iloc[i:i+seq_len].values)
        y_s.append(y.iloc[i+seq_len].values)
    X_s, y_s = np.array(X_s), np.array(y_s)

    split = int(len(X_s) * 0.8)
    X_train, X_live = X_s[:split], X_s[split:]
    y_train, y_live = y_s[:split], y_s[split:]

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)
    
    def scale_seq(seq_data):
        flat = seq_data.reshape(-1, seq_data.shape[-1])
        scaled = scaler.transform(flat).reshape(seq_data.shape)
        return np.nan_to_num(scaled).astype(np.float32)

    X_train_sc = scale_seq(X_train)
    X_live_sc = scale_seq(X_live)

    results = {}
    
    # RL MODELS
    train_obs = X_train_sc[:, -1, :]
    # Create training DF with explicit columns
    train_env_df = pd.DataFrame(train_obs, columns=feature_cols)
    for i, col in enumerate(TARGET_ETFS):
        train_env_df[col] = y_train[:, i]

    def make_env(): return TradingEnv(train_
