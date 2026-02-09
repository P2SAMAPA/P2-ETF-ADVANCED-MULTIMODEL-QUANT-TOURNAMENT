import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from datetime import datetime

# --- 1. SETTINGS & STATE ---
st.set_page_config(page_title="Quant Tournament", layout="wide")

# Ensure results persist across clicks
if 'results' not in st.session_state:
    st.session_state.results = None
if 'forecast' not in st.session_state:
    st.session_state.forecast = None

TARGET_ETFS = ['TLT', 'TBT', 'VNQ', 'GLD', 'SLV']
MACRO = ['^VIX', '^TNX', 'DX-Y.NYB']

# --- 2. MODEL ARCHITECTURES ---
class CNN_LSTM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, 32, 3, padding=1)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.fc = nn.Linear(64, out_dim)
    def forward(self, x):
        x = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# --- 3. RL ENVIRONMENT ---
class TradingEnv(gym.Env):
    def __init__(self, df, etfs):
        super().__init__()
        self.df = df
        self.etfs = etfs
        self.action_space = gym.spaces.Discrete(len(etfs))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (df.shape[1]-len(etfs),), dtype=np.float32)
        self.current_step = 0
    def reset(self, seed=None):
        self.current_step = 0
        return self.df.iloc[0, :-len(self.etfs)].values.astype(np.float32), {}
    def step(self, action):
        reward = float(self.df[self.etfs[action]].iloc[self.current_step])
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self.df.iloc[self.current_step, :-len(self.etfs)].values.astype(np.float32)
        return obs, reward, done, False, {}

# --- 4. ENGINE ---
def run_full_tournament(data):
    # Prep data
    rets = data[TARGET_ETFS].pct_change().dropna()
    feats = data.shift(1).dropna()
    idx = rets.index.intersection(feats.index)
    X, y = feats.loc[idx], rets.loc[idx]
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X).astype(np.float32)
    split = int(len(X_sc) * 0.8)
    
    # RL Models (PPO & A2C)
    train_env_df = pd.DataFrame(X_sc[:split], columns=X.columns)
    for i, etf in enumerate(TARGET_ETFS): train_env_df[etf] = y.iloc[:split, i].values
    
    env = DummyVecEnv([lambda: TradingEnv(train_env_df, TARGET_ETFS)])
    ppo = PPO("MlpPolicy", env).learn(1000)
    a2c = A2C("MlpPolicy", env).learn(1000)
    
    # Deep Learning (CNN-LSTM)
    X_t = torch.tensor(X_sc[:split]).unsqueeze(1) # Add seq dim
    y_t = torch.tensor(y.iloc[:split].values).float()
    model = CNN_LSTM(X.shape[1], len(TARGET_ETFS))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(20):
        opt.zero_grad()
        nn.MSELoss()(model(X_t), y_t).backward()
        opt.step()

    # Inference for Leaderboard
    results = {"PPO": [], "A2C": [], "CNN-LSTM": []}
    X_live = X_sc[split:]
    y_live = y.iloc[split:].values
    
    for i in range(len(X_live)):
        p_act, _ = ppo.predict(X_live[i])
        a_act, _ = a2c.predict(X_live[i])
        with torch.no_grad():
            c_preds = model(torch.tensor(X_live[i]).reshape(1, 1, -1))
            c_act = torch.argmax(c_preds).item()
            
        results["PPO"].append(y_live[i, p_act])
        results["A2C"].append(y_live[i, a_act])
        results["CNN-LSTM"].append(y_live[i, c_act])

    # FINAL FORECAST (Latest data point)
    latest_obs = X_sc[-1:]
    p_final, _ = ppo.predict(latest_obs[0])
    forecast = {"Recommended ETF": TARGET_ETFS[p_final], "Confidence": "Regime-Adaptive"}
    
    return results, idx[split:], forecast

# --- 5. UI ---
st.title("🏆 Multimodel Quant Tournament")

@st.cache_data
def get_data():
    d = yf.download(TARGET_ETFS + MACRO, start="2018-01-01", progress=False)['Close']
    return d.ffill().dropna()

df = get_data()

if st.button("🚀 Start Tournament"):
    with st.status("Training RL & Deep Learning Models...") as status:
        res, dates, forecast = run_full_tournament(df)
        st.session_state.results = (res, dates)
        st.session_state.forecast = forecast
        status.update(label="Tournament Complete!", state="complete")

# Display Results if they exist in state
if st.session_state.results:
    res, dates = st.session_state.results
    fc = st.session_state.forecast

    # 1. THE FORECAST
    st.header(f"🎯 Trade Forecast: {fc['Recommended ETF']}")
    st.info(f"The PPO Model suggests allocating to **{fc['Recommended ETF']}** for the next market session.")

    # 2. THE LEADERBOARD
    st.subheader("📊 Model Performance (Out-of-Sample)")
    cols = st.columns(len(res))
    for i, (name, r) in enumerate(res.items()):
        total_ret = (np.prod(1 + np.array(r)) - 1)
        cols[i].metric(name, f"{total_ret:.2%}")

    # 3. THE CHART
    fig = go.Figure()
    for name, r in res.items():
        fig.add_trace(go.Scatter(x=dates, y=np.cumprod(1 + np.array(r)), name=name))
    fig.update_layout(title="Equity Curves", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
