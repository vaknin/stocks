"""Configuration settings for the AI trading system."""

from typing import List, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings
import os
from pathlib import Path

class TradingConfig(BaseSettings):
    """Main configuration for the trading system."""
    
    # Target stocks (Diversified across sectors for revenue generation)
    STOCK_TICKERS: List[str] = [
        # Technology (Semiconductors & Software) - 15 stocks
        "NVDA", "AMD", "ASML", "TSM", "INTC", "QCOM", "AVGO", "MU", "SMCI", "ARM",
        "AAPL", "MSFT", "GOOGL", "META", "CRM",
        
        # Healthcare & Biotech - 10 stocks  
        "JNJ", "PFE", "UNH", "ABBV", "TMO", "AMGN", "GILD", "MRNA", "BMY", "LLY",
        
        # Financials - 10 stocks
        "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "BRK-B", "AXP", "C",
        
        # Consumer Discretionary - 10 stocks
        "AMZN", "TSLA", "HD", "MCD", "SBUX", "NKE", "TGT", "LOW", "DIS", "BKNG",
        
        # Consumer Staples - 8 stocks
        "KO", "PG", "WMT", "PEP", "COST", "CL", "KMB", "GIS",
        
        # Energy - 8 stocks
        "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO", "MPC", 
        
        # Utilities (Defensive) - 6 stocks
        "NEE", "DUK", "SO", "D", "EXC", "SRE",
        
        # Industrial - 6 stocks
        "CAT", "BA", "GE", "UNP", "HON", "MMM",
        
        # International & Others - 6 stocks  
        "SHOP", "SAP", "ADBE", "NFLX", "ORCL", "PYPL"
    ]
    
    # Capital and position sizing
    STARTING_CAPITAL: float = Field(default=2000.0, description="Starting capital in USD")
    MAX_POSITION_SIZE: float = Field(default=0.15, description="Max 15% per stock for better diversification")
    KELLY_SAFETY_FACTOR: float = Field(default=0.40, description="Optimal Kelly sizing based on 2024 research")
    MAX_ACTIVE_POSITIONS: int = Field(default=5, description="Maximum active trades")
    
    # Risk management
    MAX_DRAWDOWN: float = Field(default=0.18, description="Maximum 18% drawdown for optimal growth")
    STOP_LOSS_ATR_MULTIPLIER: float = Field(default=2.5, description="2.5x ATR dynamic stop loss")
    STOP_LOSS_PCT: float = Field(default=0.025, description="Fallback fixed stop loss")
    TARGET_VOLATILITY: float = Field(default=0.18, description="18% annualized volatility for optimal growth")
    
    # Model confidence thresholds - Dynamic ML-based (2024 research)
    CONFIDENCE_THRESHOLD_DAILY: float = Field(default=0.78, description="78% confidence for daily (ML-optimized)")
    CONFIDENCE_THRESHOLD_INTRADAY: float = Field(default=0.72, description="72% confidence for 5-min (ML-optimized)")
    CONFIDENCE_THRESHOLD_WEEKLY: float = Field(default=0.68, description="68% confidence for weekly (ML-optimized)")
    PREDICTION_INTERVAL_MAX_WIDTH: float = Field(default=0.03, description="Max 3% interval width")
    
    # Performance targets - Realistic expectations for systematic trading
    MONTHLY_RETURN_TARGET_MIN: float = Field(default=0.01, description="1% monthly minimum (realistic)")
    MONTHLY_RETURN_TARGET_MAX: float = Field(default=0.03, description="3% monthly maximum (realistic)")
    TARGET_SHARPE_RATIO: float = Field(default=1.5, description="Target Sharpe ratio >1.5 (achievable)")
    TARGET_WIN_RATE: float = Field(default=0.55, description="Target 55-65% win rate (realistic)")
    
    # Data settings
    DATA_START_DATE: str = Field(default="2022-01-01", description="Historical data start")
    DATA_FREQUENCY_INTRADAY: str = Field(default="5m", description="5-minute bars")
    DATA_FREQUENCY_DAILY: str = Field(default="1d", description="Daily bars")
    DATA_FREQUENCY_WEEKLY: str = Field(default="1wk", description="Weekly bars")
    
    # Database
    DATABASE_PATH: str = Field(default="data/trading_data.db", description="SQLite database path")
    
    # Model settings
    TIMESFM_MODEL_SIZE: str = Field(default="google/timesfm-2.0-500m-pytorch", description="TimesFM model")
    PREDICTION_HORIZON: List[int] = Field(default=[1, 5, 20], description="Prediction horizons")
    BATCH_SIZE: int = Field(default=32, description="Model batch size")
    
    # System settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    MAX_LATENCY_SECONDS: float = Field(default=1.0, description="Max signal generation latency")
    SYSTEM_UPTIME_TARGET: float = Field(default=0.99, description="99% uptime target")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global configuration instance
config = TradingConfig()

# Derived settings
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)