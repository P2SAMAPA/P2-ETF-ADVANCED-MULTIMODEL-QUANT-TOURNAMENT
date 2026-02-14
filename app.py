import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import pandas_market_calendars as mcal
from huggingface_hub import hf_hub_download, HfApi
import os
from io import StringIO

# --- 1. SETTINGS & STATE ---
st.set_page_config(page_title="Alpha Tournament Pro", layout="wide")

if 'results' not in st.session_state: st.session_state.results = None

TARGET_ETFS = ['TLT', 'TBT', 'VNQ', 'GLD', 'SLV']
MACRO = ['^VIX', '^TNX', 'DX-Y.NYB']

# Get secrets from HF Spaces
FRED_API_KEY = os.environ.get("FRED_API_KEY")
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY")
HF_TOKEN = os.environ.get("HF_KEY")
HF_DATASET_REPO = "P2SAMAPA/my-etf-data"

# --- 2. MODEL ARCHITECTURES ---
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, output_dim)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv(x)).transpose(1, 2)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, output_dim)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

# --- 3. UTILITIES ---
def get_sofr_rate(api_key):
    if not api_key: return 0.053 
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=SOFR&api_key={api_key}&file_type=json"
    try:
        r = requests.get(url, timeout=10).json()
        return float(r['observations'][-1]['value']) / 100
    except: return 0.053

def get_next_trading_day():
    try:
        nyse = mcal.get_calendar('NYSE')
        today = pd.Timestamp.now(tz='America/New_York').normalize()
        schedule = nyse.schedule(start_date=today, end_date=today + timedelta(days=10))
        valid_days = mcal.date_range(schedule, frequency='1D')
        for day in valid_days:
            if day.normalize() > today:
                return day.strftime('%Y-%m-%d')
    except:
        pass
    return (pd.Timestamp.now() + timedelta(days=1)).strftime('%Y-%m-%d')

def load_data_from_hf(start_year, hf_token, dataset_repo):
    """Load data from HuggingFace dataset"""
    try:
        # Download the dataset file
        file_path = hf_hub_download(
            repo_id=dataset_repo,
            filename=f"etf_data_{start_year}.parquet",
            repo_type="dataset",
            token=hf_token
        )
        data = pd.read_parquet(file_path)
        data.index = pd.to_datetime(data.index)
        return data, "HuggingFace Dataset"
    except Exception as e:
        st.warning(f"Could not load from HF dataset: {str(e)[:100]}")
        return None, None

def fetch_alpha_vantage_data(tickers, start_date, api_key):
    """Fetch data from Alpha Vantage as fallback"""
    if not api_key:
        return None
    
    all_data = {}
    import time
    for ticker in tickers:
        try:
            clean_ticker = ticker.replace('^', '')
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={clean_ticker}&outputsize=full&apikey={api_key}"
            r = requests.get(url, timeout=15).json()
            
            if 'Time Series (Daily)' not in r:
                continue
                
            ts_data = r['Time Series (Daily)']
            df = pd.DataFrame.from_dict(ts_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df[df.index >= start_date]
            all_data[ticker] = df['5. adjusted close'].astype(float)
            time.sleep(13)
        except:
            continue
    
    if len(all_data) > 0:
        return pd.DataFrame(all_data)
    return None

def fetch_market_data(start_year, hf_token, av_key, dataset_repo):
    """Fetch data with HF primary, Alpha Vantage fallback"""
    # Try HF dataset first
    data, data_source = load_data_from_hf(start_year, hf_token, dataset_repo)
    
    if data is not None and len(data) >= 100:
        return data, data_source
    
    # Fallback to Alpha Vantage
    st.warning("HF dataset not available, trying Alpha Vantage fallback...")
    data_source = "Alpha Vantage (fallback)"
    data = fetch_alpha_vantage_data(TARGET_ETFS + MACRO, f"{start_year}-01-01", av_key)
    
    if data is None or len(data) < 100:
        raise Exception("Unable to fetch sufficient data from both HF and Alpha Vantage. Please try again later.")
    
    return data, data_source

def calculate_momentum_features(data, lookback_periods=[30, 45, 60]):
    """Calculate momentum features for multiple lookback periods"""
    momentum_features = {}
    
    for period in lookback_periods:
        momentum = data.pct_change(period)
        momentum_features[f'momentum_{period}d'] = momentum
    
    return momentum_features

class TradingEnv(gym.Env):
    def __init__(self, features, returns, etfs, tcost_bps):
        super().__init__()
        self.features, self.returns, self.etfs = features, returns, etfs
        self.tcost = tcost_bps / 10000 
        self.action_space = gym.spaces.Discrete(len(etfs))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(features.shape[1],), dtype=np.float32)
        self.current_step = 0
        self.last_action = None
    def reset(self, seed=None):
        self.current_step = 0
        self.last_action = None
        return self.features[0], {}
    def step(self, action):
        raw_reward = float(self.returns[self.current_step, action])
        penalty = 0
        if self.last_action is not None and action != self.last_action:
            penalty = self.tcost
        reward = raw_reward - penalty
        self.last_action = action
        self.current_step += 1
        done = self.current_step >= len(self.features) - 1
        return self.features[self.current_step], reward, done, False, {}

# --- 4. ENGINE ---
@st.cache_resource(ttl=604800)
def run_tournament_engine(data_json, rf_rate, tcost_bps, start_year, data_source):
    data = pd.read_json(StringIO(data_json))
    
    # Track data transformations for diagnostics
    raw_start = data.index[0].strftime('%Y-%m-%d')
    raw_end = data.index[-1].strftime('%Y-%m-%d')
    raw_rows = len(data)
    
    # Calculate momentum features for different lookback periods
    lookback_periods = [30, 45, 60]
    momentum_dict = calculate_momentum_features(data, lookback_periods)
    
    # Test each lookback period to find best performing one
    best_lookback = None
    best_score = -np.inf
    
    for period in lookback_periods:
        momentum_data = momentum_dict[f'momentum_{period}d']
        combined_data = pd.concat([data, momentum_data.add_suffix(f'_mom{period}')], axis=1).dropna()
        
        if len(combined_data) < 100:
            continue
            
        # Quick validation score using simple correlation
        rets = combined_data[TARGET_ETFS].pct_change().dropna()
        mom_cols = [col for col in combined_data.columns if f'_mom{period}' in col]
        if len(rets) > 0 and len(mom_cols) > 0:
            score = np.abs(combined_data[mom_cols].corrwith(rets[TARGET_ETFS[0]])).mean()
            if score > best_score:
                best_score = score
                best_lookback = period
    
    # Use best lookback period or default to 45
    if best_lookback is None:
        best_lookback = 45
    
    # Build final dataset with best lookback
    momentum_data = momentum_dict[f'momentum_{best_lookback}d']
    data_with_momentum = pd.concat([data, momentum_data.add_suffix(f'_mom{best_lookback}')], axis=1).dropna()
    
    rets_df = data_with_momentum[TARGET_ETFS].pct_change().dropna()
    feats_df = data_with_momentum.shift(1).dropna()
    common_idx = rets_df.index.intersection(feats_df.index)
    
    # More diagnostics
    after_processing_start = common_idx[0].strftime('%Y-%m-%d')
    after_processing_rows = len(common_idx)
    
    X, y = feats_df.loc[common_idx].values, rets_df.loc[common_idx].values
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X).astype(np.float32)
    split = int(len(X_sc) * 0.8)
    split_date = common_idx[split].strftime('%Y-%m-%d')
    
    # Create sequences for deep learning models
    seq_len = best_lookback
    X_seq = []
    y_seq = []
    for i in range(seq_len, len(X_sc)):
        X_seq.append(X_sc[i-seq_len:i])
        y_seq.append(y[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Adjust split for sequences
    split_seq = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_seq], X_seq[split_seq:]
    y_train, y_test = y_seq[:split_seq], y_seq[split_seq:]
    
    # For RL models, use flattened current features
    X_train_flat = X_sc[:split]
    X_test_flat = X_sc[split:]
    y_train_rl = y[:split]
    y_test_rl = y[split:]
    
    env = DummyVecEnv([lambda: TradingEnv(X_train_flat, y_train_rl, TARGET_ETFS, tcost_bps)])
    ppo = PPO("MlpPolicy", env, verbose=0).learn(5000)
    a2c = A2C("MlpPolicy", env, verbose=0).learn(5000)
    
    dl_models = {}
    for name, m_class in [("CNN-LSTM", CNN_LSTM_Model), ("Transformer", TransformerModel)]:
        model = m_class(X.shape[1], len(TARGET_ETFS), seq_len)
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        X_t, y_t = torch.tensor(X_train).float(), torch.tensor(y_train).float()
        for _ in range(50): 
            opt.zero_grad()
            nn.MSELoss()(model(X_t), y_t).backward()
            opt.step()
        dl_models[name] = model

    results = {"PPO": [], "A2C": [], "CNN-LSTM": [], "Transformer": []}
    test_dates = common_idx[split:]
    tcost_dec = tcost_bps / 10000

    # PPO and A2C predictions
    for name in ["PPO", "A2C"]:
        last_pick = None
        for i in range(len(X_test_flat)):
            if name == "PPO": 
                act, _ = ppo.predict(X_test_flat[i], deterministic=True)
            else:
                act, _ = a2c.predict(X_test_flat[i], deterministic=True)
            
            day_ret = y_test_rl[i, act]
            if last_pick is not None and act != last_pick: 
                day_ret -= tcost_dec
            results[name].append(day_ret)
            last_pick = act
    
    # Deep learning model predictions
    for name in ["CNN-LSTM", "Transformer"]:
        last_pick = None
        for i in range(len(X_test)):
            with torch.no_grad():
                out = dl_models[name](torch.tensor(X_test[i]).unsqueeze(0).float())
                act = torch.argmax(out).item()
            
            day_ret = y_test[i, act]
            if last_pick is not None and act != last_pick:
                day_ret -= tcost_dec
            results[name].append(day_ret)
            last_pick = act

    # OOS period calculation using actual test dates
    oos_start_year = test_dates[0].year
    oos_end_year = test_dates[-1].year
    oos_years = f"{oos_start_year}-{oos_end_year}" if oos_start_year != oos_end_year else str(oos_start_year)

    # Logic for ranking
    recency_window = 15
    recency_scores = {n: np.sum(np.array(r[-recency_window:]) > 0) / recency_window for n, r in results.items()}
    perf = {k: ((np.prod(1 + np.array(results[k])) - 1) * 0.7) + (recency_scores[k] * 0.3) for k in results.keys()}
    
    # Sort to find Champion and Runner-Up
    sorted_models = sorted(perf.items(), key=lambda x: x[1], reverse=True)
    champ, runner_up = sorted_models[0][0], sorted_models[1][0]
    
    # Forecasts for both
    forecasts = {}
    latest_feat_flat = X_sc[-1:]
    latest_feat_seq = X_seq[-1:]
    
    for m in [champ, runner_up]:
        if m == "PPO": 
            act, _ = ppo.predict(latest_feat_flat[0], deterministic=True)
        elif m == "A2C": 
            act, _ = a2c.predict(latest_feat_flat[0], deterministic=True)
        else:
            with torch.no_grad():
                f_out = dl_models[m](torch.tensor(latest_feat_seq).float())
                act = torch.argmax(f_out).item()
        forecasts[m] = TARGET_ETFS[act]

    # Process Table (Champion only)
    champ_series = pd.Series(results[champ], index=test_dates)
    monthly_rets = champ_series.groupby([champ_series.index.year, champ_series.index.month]).apply(lambda x: np.prod(1+x)-1)
    m_table = monthly_rets.unstack().fillna(0)
    m_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(m_table.columns)]
    m_table['Yearly Total'] = m_table.apply(lambda row: np.prod(1 + row) - 1, axis=1)

    # Diagnostics info
    diagnostics = {
        'requested_start': f"{start_year}-01-01",
        'raw_start': raw_start,
        'raw_end': raw_end,
        'raw_rows': raw_rows,
        'actual_start': after_processing_start,
        'processed_rows': after_processing_rows,
        'split_date': split_date,
        'data_source': data_source,
        'best_lookback': best_lookback
    }

    return results, test_dates, forecasts, champ, runner_up, m_table, recency_scores, oos_years, diagnostics

# --- 5. UI ---
st.title("Alpha Tournament Pro: Multi-model ETF Forecast")

with st.sidebar:
    st.header("Tournament Configuration")
    start_year = st.selectbox("Select Training Start Year", options=["2007", "2010", "2015", "2019", "2021"], index=0)
    t_cost = st.slider("Transaction Cost (bps)", min_value=0, max_value=100, value=10, step=5)
    run_btn = st.button("🚀 Execute Alpha Tournament")

if run_btn:
    with st.status(f"Training Tournament Models...") as status:
        try:
            rf = get_sofr_rate(FRED_API_KEY)
            raw_data, data_src = fetch_market_data(start_year, HF_TOKEN, ALPHA_VANTAGE_KEY, HF_DATASET_REPO)
            res, dates, fcasts, champ, runner, m_table, r_scores, oos_years, diag = run_tournament_engine(raw_data.to_json(), rf, t_cost, start_year, data_src)
            next_trade_day = get_next_trading_day()
            st.session_state.results = {
                "res": res, "dates": dates, "fcasts": fcasts, "champ": champ, "runner": runner, 
                "rf": rf, "monthly": m_table, "recency": r_scores, "t_cost": t_cost, 
                "oos_years": oos_years, "next_day": next_trade_day, "diagnostics": diag
            }
            status.update(label=f"Tournament Complete!", state="complete")
        except Exception as e:
            status.update(label=f"Error: {str(e)}", state="error")
            st.error(f"Failed to run tournament: {str(e)}")
            st.stop()
    st.rerun()

if st.session_state.results:
    s = st.session_state.results
    
    # --- CHAMPION ROW ---
    st.subheader(f"🏆 Champion: {s['champ']}")
    c1, c2, c3, c4 = st.columns(4)
    c_rets = np.array(s['res'][s['champ']])
    c1.metric(f"PREDICTION", s['fcasts'][s['champ']], delta=f"Valid: {s['next_day']}")
    c2.metric("Total Return (Net)", f"{(np.prod(1+c_rets)-1):.2%}", delta=f"OOS: {s['oos_years']}")
    c3.metric("Sharpe (Annualized)", f"{((np.mean(c_rets)-(s['rf']/252))/np.std(c_rets)*np.sqrt(252)):.2f}", delta=f"SOFR: {s['rf']:.2%}", delta_color="normal")
    c4.metric("Recency Score (15d)", f"{s['recency'][s['champ']]:.0%}")

    # --- RUNNER UP ROW ---
    st.subheader(f"🥈 Runner-Up: {s['runner']}")
    r1, r2, r3, r4 = st.columns(4)
    r_rets = np.array(s['res'][s['runner']])
    r1.metric(f"PREDICTION", s['fcasts'][s['runner']], delta=f"Valid: {s['next_day']}")
    r2.metric("Total Return (Net)", f"{(np.prod(1+r_rets)-1):.2%}", delta=f"OOS: {s['oos_years']}")
    r3.metric("Sharpe (Annualized)", f"{((np.mean(r_rets)-(s['rf']/252))/np.std(r_rets)*np.sqrt(252)):.2f}", delta=f"SOFR: {s['rf']:.2%}", delta_color="normal")
    r4.metric("Recency Score (15d)", f"{s['recency'][s['runner']]:.0%}")

    st.divider()
    # Charts and Tables
    fig = go.Figure()
    for name, r in s['res'].items(): fig.add_trace(go.Scatter(x=s['dates'], y=np.cumprod(1 + np.array(r)), name=name))
    fig.update_layout(title="Net Return Performance", template="plotly_dark", height=400)
    st.plotly_chart(fig, width='stretch')

    st.subheader(f"📅 Monthly Matrix ({s['champ']})")
    st.dataframe(s['monthly'].style.format("{:.2%}"), width='stretch')

    st.divider()
    st.header("🔍 Methodology")
    st.info("""
    **Recency Score (15d):** The 'Hit Rate' of a model over the last 15 trading sessions (% of positive days). 
    The engine blends this (30%) with long-term OOS performance (70%) to rank the models.
    """)
    
    st.subheader("🤖 Model Descriptions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**PPO (Proximal Policy Optimization)**")
        st.caption("A state-of-the-art reinforcement learning algorithm that learns optimal trading policies through trial and error. PPO uses an actor-critic architecture and clips policy updates to ensure stable learning. Trained for 5,000 timesteps on historical data.")
        
        st.markdown("**CNN-LSTM**")
        st.caption("A hybrid deep learning architecture combining Convolutional Neural Networks (for pattern extraction) with Long Short-Term Memory networks (for sequential dependencies). Captures both spatial and temporal features in market data. Trained for 50 epochs.")
    
    with col2:
        st.markdown("**A2C (Advantage Actor-Critic)**")
        st.caption("A reinforcement learning algorithm that simultaneously learns a value function (critic) and policy (actor). More sample-efficient than standard policy gradients and learns to maximize risk-adjusted returns. Trained for 5,000 timesteps.")
        
        st.markdown("**Transformer**")
        st.caption("An attention-based neural network architecture that weighs the importance of different time steps in the input sequence. Uses multi-head self-attention mechanisms to capture complex temporal relationships in market data. Trained for 50 epochs.")
    
    # Data Diagnostics
    if 'diagnostics' in s:
        st.divider()
        st.subheader("📊 Data Diagnostics")
        diag = s['diagnostics']
        
        if diag['requested_start'] != diag['actual_start']:
            st.warning(f"⚠️ **Data Availability Notice:** Data requested from {diag['requested_start']}, but actual usable data starts from {diag['actual_start']} due to instrument availability and momentum calculation requirements.")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Data Source", diag['data_source'])
            st.metric("Requested Start", diag['requested_start'])
        with col_b:
            st.metric("Actual Data Start", diag['actual_start'])
            st.metric("Training/OOS Split", diag['split_date'])
        with col_c:
            st.metric("Total Data Rows", f"{diag['processed_rows']:,}")
            st.metric("Optimal Lookback", f"{diag['best_lookback']} days")
