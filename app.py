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
from datetime import datetime, timedelta
import requests

# --- 1. SETTINGS & STATE ---
st.set_page_config(page_title="Alpha Tournament Pro", layout="wide")

if 'results' not in st.session_state: st.session_state.results = None

TARGET_ETFS = ['TLT', 'TBT', 'VNQ', 'GLD', 'SLV']
MACRO = ['^VIX', '^TNX', 'DX-Y.NYB']
FRED_API_KEY = st.secrets.get("FRED_API_KEY")

# --- 2. MODEL ARCHITECTURES ---
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
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
    def __init__(self, input_dim, output_dim):
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
def run_tournament_engine(data_json, rf_rate, tcost_bps):
    data = pd.read_json(data_json)
    rets_df = data[TARGET_ETFS].pct_change().dropna()
    feats_df = data.shift(1).dropna()
    common_idx = rets_df.index.intersection(feats_df.index)
    X, y = feats_df.loc[common_idx].values, rets_df.loc[common_idx].values
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X).astype(np.float32)
    split = int(len(X_sc) * 0.8) 
    X_train, X_test, y_train, y_test = X_sc[:split], X_sc[split:], y[:split], y[split:]

    env = DummyVecEnv([lambda: TradingEnv(X_train, y_train, TARGET_ETFS, tcost_bps)])
    ppo = PPO("MlpPolicy", env, verbose=0).learn(2000)
    a2c = A2C("MlpPolicy", env, verbose=0).learn(2000)
    
    dl_models = {}
    for name, m_class in [("CNN-LSTM", CNN_LSTM_Model), ("Transformer", TransformerModel)]:
        model = m_class(X.shape[1], len(TARGET_ETFS))
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        X_t, y_t = torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train).float()
        for _ in range(25): 
            opt.zero_grad()
            nn.MSELoss()(model(X_t), y_t).backward()
            opt.step()
        dl_models[name] = model

    results = {"PPO": [], "A2C": [], "CNN-LSTM": [], "Transformer": []}
    picks = {"PPO": [], "A2C": [], "CNN-LSTM": [], "Transformer": []}
    dates = common_idx[split:]
    tcost_dec = tcost_bps / 10000

    for name in results.keys():
        last_pick = None
        for i in range(len(X_test)):
            if name == "PPO": act, _ = ppo.predict(X_test[i], deterministic=True)
            elif name == "A2C": act, _ = a2c.predict(X_test[i], deterministic=True)
            else:
                with torch.no_grad():
                    x_in = torch.tensor(X_test[i]).reshape(1, 1, -1)
                    out = dl_models[name](x_in)
                    act = torch.argmax(out).item()
            day_ret = y_test[i, act]
            if last_pick is not None and act != last_pick:
                day_ret -= tcost_dec
            results[name].append(day_ret)
            picks[name].append(TARGET_ETFS[act])
            last_pick = act

    recency_window = 15
    recency_scores = {name: np.sum(np.array(rets[-recency_window:]) > 0) / recency_window for name, rets in results.items()}
    perf = {k: ( (np.prod(1 + np.array(results[k])) - 1) * 0.7) + (recency_scores[k] * 0.3) for k in results.keys()}
    champ = max(perf, key=perf.get)
    
    champ_series = pd.Series(results[champ], index=dates)
    monthly_rets = champ_series.groupby([champ_series.index.year, champ_series.index.month]).apply(lambda x: np.prod(1+x)-1)
    m_table = monthly_rets.unstack().fillna(0)
    m_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(m_table.columns)]
    m_table['Yearly Total'] = m_table.apply(lambda row: np.prod(1 + row) - 1, axis=1)

    latest_feat = X_sc[-1:]
    if champ == "PPO": f_act, _ = ppo.predict(latest_feat[0], deterministic=True)
    elif champ == "A2C": f_act, _ = a2c.predict(latest_feat[0], deterministic=True)
    else: 
        with torch.no_grad():
            f_act = torch.argmax(dl_models[champ](torch.tensor(latest_feat).reshape(1, 1, -1))).item()

    audit_list = []
    for j in range(len(X_test)-15, len(X_test)):
        audit_list.append({'Date': dates[j].strftime('%Y-%m-%d'), 'ETF Picked': picks[champ][j], 'Outcome Return': results[champ][j]})

    return results, dates, TARGET_ETFS[f_act], champ, pd.DataFrame(audit_list), m_table, recency_scores[champ]

# --- 5. UI ---
st.title("Alpha Tournament Pro: Multi-model ETF Forecast")

with st.sidebar:
    st.header("Tournament Configuration")
    start_year = st.selectbox("Select Training Start Year", options=["2007", "2010", "2015", "2019", "2021"], index=0)
    t_cost = st.slider("Transaction Cost (bps)", min_value=0, max_value=100, value=10, step=5)
    run_btn = st.button("🚀 Execute Alpha Tournament")

now = datetime.now()
target_date = now if now.hour < 16 else now + timedelta(days=1)
while target_date.weekday() >= 5: target_date += timedelta(days=1)

if run_btn:
    with st.status(f"Training Tournament Models...") as status:
        rf = get_sofr_rate(FRED_API_KEY)
        raw_data = yf.download(TARGET_ETFS + MACRO, start=f"{start_year}-01-01", progress=False)['Close'].ffill().dropna()
        res, dates, ticker, champ, audit, m_table, r_score = run_tournament_engine(raw_data.to_json(), rf, t_cost)
        st.session_state.results = {"res": res, "dates": dates, "ticker": ticker, "champ": champ, "audit": audit, "rf": rf, "start": start_year, "monthly": m_table, "recency": r_score, "t_cost": t_cost}
        status.update(label=f"Champion Identified: {champ}", state="complete")
    st.rerun()

if st.session_state.results:
    s = st.session_state.results
    st.header(f"🎯 Forecast for {target_date.strftime('%b %d')}: BUY {s['ticker']}")
    
    m1, m2, m3, m4 = st.columns(4)
    champ_rets_arr = np.array(s['res'][s['champ']])
    m1.metric("Winner Total Return (Net)", f"{(np.prod(1+champ_rets_arr)-1):.2%}", delta=s['champ'])
    
    # Sharpe Metric with Live SOFR Display
    m2.metric("Sharpe (Annualized)", 
              f"{((np.mean(champ_rets_arr)-(s['rf']/252))/np.std(champ_rets_arr)*np.sqrt(252)):.2f}",
              delta=f"Rf (SOFR): {s['rf']:.2%}", delta_color="normal")
    
    m3.metric("Recency Score (15d)", f"{s['recency']:.0%}")
    m4.metric("Friction Applied", f"{s['t_cost']} bps")

    fig = go.Figure()
    for name, r in s['res'].items():
        fig.add_trace(go.Scatter(x=s['dates'], y=np.cumprod(1 + np.array(r)), name=name))
    fig.update_layout(title=f"Cumulative Net Return (Friction: {s['t_cost']}bps)", template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"📅 Monthly Performance Matrix ({s['champ']})")
    def heatmap_style(val):
        alpha = min(abs(val)*6, 0.9)
        text_color = "white" if alpha > 0.45 else "black"
        if val > 0: return f'background-color: rgba(0, 128, 0, {alpha}); color: {text_color};'
        elif val < 0: return f'background-color: rgba(255, 0, 0, {alpha}); color: {text_color};'
        return 'color: black;'
    st.dataframe(s['monthly'].style.applymap(heatmap_style).format("{:.2%}"), use_container_width=True)

    st.subheader(f"📊 15-Day Audit Table ({s['champ']})")
    def audit_style(val):
        color = '#d1f2eb' if val > 0 else '#fcdedc'
        text_color = '#0e6251' if val > 0 else '#943126'
        return f'background-color: {color}; color: {text_color}; border-radius: 8px; font-weight: bold;'
    st.table(s['audit'].sort_values('Date', ascending=False).style.format({'Outcome Return': '{:.2%}'}).applymap(audit_style, subset=['Outcome Return']))

    st.divider()
    st.header("🔍 Methodology & Model Architecture")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Selection Logic: 15-Day Recency Score")
        st.info("""
        **Recency Score (15d):** This metric represents the 'Hit Rate' of a model over the most recent 15 trading sessions. 
        It is calculated as the percentage of days where the model's chosen ETF produced a positive return. 
        The final Champion selection uses a weighted blend (70% long-term OOS performance / 30% Recency Score) 
        to ensure the winner is both historically robust and currently in sync with market momentum.
        """)
        st.subheader("Transaction Friction")
        st.write(f"Models are trained to chase pure cumulative returns. A friction cost of **{s['t_cost']} bps** is applied to every trade change.")
    with col_b:
        st.subheader("The Competing Models")
        st.markdown("1. **PPO & A2C:** RL agents optimized for growth. \n 2. **CNN-LSTM:** Spatial-temporal pattern learner. \n 3. **Transformer:** Attention-based correlation finder.")
