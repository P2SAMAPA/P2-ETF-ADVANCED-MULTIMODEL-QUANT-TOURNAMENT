---
title: P2 ETF Multimodel Quant Tournament
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.54.0"
app_file: app.py
pinned: false
license: mit
---

# Alpha Tournament Pro: Multi-model ETF Forecast

Advanced quantitative trading system using multiple AI models (PPO, A2C, CNN-LSTM, Transformer) to predict optimal ETF allocations.

## Features
- 4 competing AI models
- Automated momentum feature engineering
- Transaction cost modeling
- Live SOFR risk-free rate integration
- Comprehensive backtesting with OOS validation

## Models
- **PPO**: Proximal Policy Optimization (RL)
- **A2C**: Advantage Actor-Critic (RL)
- **CNN-LSTM**: Hybrid deep learning
- **Transformer**: Attention-based architecture

## Data
Uses HuggingFace dataset: `P2SAMAPA/my-etf-data` with fallback to Alpha Vantage API.
```

### 4. **`.gitignore`**
```
__pycache__/
*.py[cod]
*$py.class
.env
*.log
.DS_Store
.streamlit/secrets.toml
