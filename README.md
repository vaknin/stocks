# AI-Powered Multi-Sector Trading System

> **State-of-the-art AI models for systematic multi-sector stock trading with 8-15% monthly return targets**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-ee4c2c.svg)](https://pytorch.org/)
[![Google Colab](https://img.shields.io/badge/Colab-Optimized-orange.svg)](https://colab.research.google.com/)

## 🚀 Quick Start (Google Colab)

1. **Upload and extract**: Upload `colab.zip` to Colab and extract
2. **Setup environment**: Run the setup cell in `Colab.ipynb`
3. **Validate models**: `!python colab_setup/model_validator.py`
4. **Start trading**: `!python trading_advisor.py`

**Time to first prediction**: ~3 minutes after setup

## 📊 Project Overview

This system combines cutting-edge AI models to generate profitable trading signals across multiple market sectors:

- **🧠 Primary AI**: TimesFM (Google's 500M parameter time series foundation model)
- **🔀 Pattern Recognition**: TSMamba (State space models for stock patterns)
- **🕸️ Correlation Analysis**: SAMBA (Graph-Mamba for inter-stock relationships)
- **🎯 Uncertainty Quantification**: MAPIE conformal prediction for confidence intervals
- **🔄 Adaptive Learning**: Dynamic ensemble weighting based on recent performance

### Diversified Portfolio (75+ Stocks)
- **Technology**: 15 stocks (NVDA, AMD, AAPL, MSFT, etc.)
- **Healthcare**: 10 stocks (JNJ, PFE, UNH, etc.)
- **Financials**: 10 stocks (JPM, BAC, V, MA, etc.)
- **Consumer Discretionary**: 10 stocks (AMZN, TSLA, HD, etc.)
- **Consumer Staples**: 8 stocks (KO, PG, WMT, etc.)
- **Energy**: 8 stocks (XOM, CVX, COP, etc.)
- **Utilities**: 6 stocks (NEE, DUK, SO, etc.)
- **Industrial**: 6 stocks (CAT, BA, GE, etc.)
- **International**: 6 stocks (SHOP, SAP, ADBE, etc.)

**Capital**: $2,500 test capital with scaling potential  
**Strategy**: Multi-timeframe (intraday, daily, weekly) with regime-adaptive allocation

## 🎯 Performance Targets

| Metric | Target Range |
|--------|-------------|
| **Monthly Returns** | 1-3% |
| **Win Rate** | 65-75% |
| **Sharpe Ratio** | >1.5 |
| **Max Drawdown** | 10-15% |
| **Volatility** | 15-20% annualized |

## 🔧 Installation & Usage

### Requirements
- **Python**: 3.10+ (required for MAPIE compatibility)
- **Environment**: Google Colab (recommended) or local with CUDA 12.5+
- **Memory**: 12GB+ RAM for model loading

### Google Colab Setup (Recommended)
```bash
# In Colab notebook cell:
!unzip colab.zip
!sudo apt-get update -y
!sudo apt-get install python3.10 python3.10-distutils -y
!sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
!curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
!bash colab_setup/install_from_wheels.sh
```

### Daily Trading Workflow
```bash
# Generate daily recommendations
python trading_advisor.py

# View portfolio status
python trading_advisor.py --portfolio

# Run backtesting
python backtest_engine.py --days 252
```

## 🛡️ Risk Management

### Position Sizing & Allocation
- **Kelly Criterion**: 0.25x Kelly with confidence scaling
- **Maximum Position**: 16% per stock ($300-400 for $2.5k capital)
- **Active Positions**: 3-5 maximum
- **Correlation Limit**: Max 0.7 between positions

### Regime-Adaptive Allocation
- **Bull Market**: 75% allocation (momentum following)
- **Bear Market**: 40% allocation (short bias + hedging)
- **High Volatility**: 25% allocation (reduced exposure)
- **Sideways Market**: 65% allocation (mean reversion)

## 📊 Key Features

- ✅ **Real AI Models**: TimesFM, TSMamba, SAMBA (no mock modes)
- ✅ **Multi-Sector Diversification**: 75+ stocks across 8 major sectors
- ✅ **Uncertainty Quantification**: MAPIE confidence intervals
- ✅ **Multi-Timeframe**: Intraday, daily, weekly signal fusion
- ✅ **Regime Detection**: HMM-based market state identification
- ✅ **Dynamic Ensemble**: Adaptive model weight optimization
- ✅ **Comprehensive Backtesting**: Monte Carlo validation
- ✅ **Risk Management**: Kelly sizing with correlation limits
- ✅ **Google Colab Optimized**: Fast setup with wheel caching

## 📋 Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Dependencies** | ✅ Working | PyTorch 2.7.1 + mamba_ssm compatible |
| **AI Models** | ✅ Validated | All models pass validation tests |
| **Data Pipeline** | ✅ Enhanced | Advanced validation (market hours, corporate actions) |
| **Trading Engine** | ✅ Working | Signal generation and execution |
| **Risk Management** | ✅ Enhanced | Kelly sizing + correlation limits + transaction costs |
| **Backtesting** | ✅ Institutional-Grade | Monte Carlo validation + performance attribution |
| **Multi-Sector Support** | ✅ Working | 75+ stocks across 8 sectors |
| **Transaction Costs** | ✅ Integrated | Realistic modeling (commission, spread, slippage) |
| **Performance Attribution** | ✅ Complete | Multi-factor analysis with market beta separation |

**System is production-ready with institutional-grade features and realistic 1-3% monthly return targeting.**

## 🔧 Configuration

### Trading Thresholds
```python
CONFIDENCE_THRESHOLD = 0.75  # Minimum confidence for trades
PROFIT_TARGET = 0.02         # 2% profit target
STOP_LOSS = 0.015           # 1.5% stop loss
MONTHLY_RETURN_TARGET = 0.01-0.03  # 1-3% monthly target
```

Edit `src/config/settings.py` for custom stock selection and parameters.

## 📝 License

This project is for educational and research purposes. Please ensure compliance with local financial regulations before live trading.

---

**⚡ Ready to generate 1-3% monthly returns with state-of-the-art AI models across multiple market sectors?**

Start with: `python trading_advisor.py`