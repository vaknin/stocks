# AI-Powered Investment Recommendation System

> **State-of-the-art AI models for intelligent investment recommendations across multiple sectors**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-ee4c2c.svg)](https://pytorch.org/)
[![Google Colab](https://img.shields.io/badge/Colab-Optimized-orange.svg)](https://colab.research.google.com/)

## 🚀 Quick Start (Google Colab)

1. **Upload and extract**: Upload `colab.zip` to Colab and extract
2. **Setup environment**: Run the setup cell in `Colab.ipynb`
3. **Validate models**: `!python colab_setup/model_validator.py`
4. **Get recommendations**: `!python trading_advisor.py`

**Time to first prediction**: ~3 minutes after setup

## 📊 Project Overview

This system combines cutting-edge AI models to generate intelligent investment recommendations across multiple market sectors:

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

**Analysis**: Multi-timeframe (intraday, daily, weekly) recommendations with regime-adaptive insights  
**Output**: CLI-based investment recommendations with confidence scores and risk analysis

## 🎯 Performance Targets

| Metric | Target Range |
|--------|-------------|
| **Recommendation Accuracy** | 65-75% |
| **Confidence Calibration** | 95%+ reliability |
| **Return Prediction Error** | <15% MAPE |
| **Risk Assessment Accuracy** | >90% |
| **Multi-timeframe Consistency** | >85% agreement |

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

### Daily Investment Analysis Workflow
```bash
# Generate investment recommendations
python trading_advisor.py

# Analyze current portfolio
python trading_advisor.py --portfolio

# Backtest recommendation accuracy
python backtest_engine.py --days 252
```

## 🛡️ Risk Assessment & Portfolio Analysis

### Recommendation Quality Controls
- **Confidence Filtering**: Only recommendations above 75% confidence threshold
- **Multi-model Consensus**: Ensemble agreement validation
- **Historical Accuracy**: Track record of recommendation performance
- **Uncertainty Quantification**: MAPIE confidence intervals for all predictions

### Portfolio Risk Analysis
- **Diversification Assessment**: Sector and correlation analysis
- **Position Size Recommendations**: Kelly criterion-based suggestions
- **Risk-Adjusted Return Predictions**: Sharpe ratio optimization
- **Regime-Adaptive Insights**: Market condition awareness

## 📊 Key Features

- ✅ **Real AI Models**: TimesFM, TSMamba, SAMBA (no mock modes)
- ✅ **Multi-Sector Analysis**: 75+ stocks across 8 major sectors
- ✅ **Uncertainty Quantification**: MAPIE confidence intervals for all recommendations
- ✅ **Multi-Timeframe Analysis**: Intraday, daily, weekly prediction fusion
- ✅ **Regime Detection**: HMM-based market state identification
- ✅ **Dynamic Ensemble**: Adaptive model weight optimization
- ✅ **Comprehensive Backtesting**: Historical recommendation accuracy validation
- ✅ **Portfolio Risk Assessment**: Advanced correlation and diversification analysis
- ✅ **Google Colab Optimized**: Fast setup with wheel caching

## 📋 Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Dependencies** | ✅ Working | PyTorch 2.7.1 + mamba_ssm compatible |
| **AI Models** | ✅ Validated | All models pass validation tests |
| **Data Pipeline** | ✅ Enhanced | Advanced validation (market hours, corporate actions) |
| **Recommendation Engine** | ✅ Working | Multi-model ensemble with confidence scoring |
| **Risk Assessment** | ✅ Enhanced | Portfolio analysis with correlation limits |
| **Backtesting** | ✅ Institutional-Grade | Historical accuracy validation + performance attribution |
| **Multi-Sector Support** | ✅ Working | 75+ stocks across 8 sectors |
| **Uncertainty Quantification** | ✅ Integrated | MAPIE confidence intervals for all predictions |
| **Performance Attribution** | ✅ Complete | Multi-factor analysis with market beta separation |

**System is production-ready for intelligent investment recommendation generation via CLI interface.**

## 🔧 Configuration

### Recommendation Parameters
```python
CONFIDENCE_THRESHOLD = 0.75  # Minimum confidence for recommendations
RETURN_PREDICTION_HORIZON = [1, 5, 20]  # Days ahead for predictions
RISK_TOLERANCE = 'moderate'  # Conservative, moderate, aggressive
PORTFOLIO_MAX_CORRELATION = 0.7  # Maximum position correlation
```

Edit `src/config/settings.py` for custom stock selection and parameters.

## 📝 License

This project is for educational and research purposes. All investment decisions should be made independently based on your own research and risk tolerance.

---

**⚡ Ready to get intelligent investment recommendations with state-of-the-art AI models across multiple market sectors?**

Start with: `python trading_advisor.py`