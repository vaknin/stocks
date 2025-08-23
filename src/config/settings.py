"""Configuration settings for the AI trading system."""

from typing import List, Dict, Any
from pydantic import BaseSettings, Field
import os
from pathlib import Path

class TradingConfig(BaseSettings):
    """Main configuration for the trading system."""
    
    # Target stocks (AI/Semiconductor focus)
    STOCK_TICKERS: List[str] = [
        "NVDA", "AMD", "ASML", "TSM", "INTC", 
        "QCOM", "AVGO", "MU", "SMCI", "ARM"
    ]
    
    # Capital and position sizing
    STARTING_CAPITAL: float = Field(default=2000.0, description="Starting capital in USD")
    MAX_POSITION_SIZE: float = Field(default=0.20, description="Max 20% per stock")
    KELLY_SAFETY_FACTOR: float = Field(default=0.25, description="Quarter Kelly sizing")
    MAX_ACTIVE_POSITIONS: int = Field(default=5, description="Maximum active trades")
    
    # Risk management
    MAX_DRAWDOWN: float = Field(default=0.10, description="Maximum 10% drawdown")
    STOP_LOSS_PCT: float = Field(default=0.025, description="2.5% stop loss")
    TARGET_VOLATILITY: float = Field(default=0.15, description="15% annualized volatility")
    
    # Model confidence thresholds
    CONFIDENCE_THRESHOLD_DAILY: float = Field(default=0.75, description="75% confidence for daily")
    CONFIDENCE_THRESHOLD_INTRADAY: float = Field(default=0.70, description="70% confidence for 5-min")
    PREDICTION_INTERVAL_MAX_WIDTH: float = Field(default=0.03, description="Max 3% interval width")
    
    # Performance targets
    MONTHLY_RETURN_TARGET_MIN: float = Field(default=0.08, description="8% monthly minimum")
    MONTHLY_RETURN_TARGET_MAX: float = Field(default=0.15, description="15% monthly maximum")
    TARGET_SHARPE_RATIO: float = Field(default=2.0, description="Target Sharpe ratio >2.0")
    TARGET_WIN_RATE: float = Field(default=0.65, description="Target 65-75% win rate")
    
    # Data settings
    DATA_START_DATE: str = Field(default="2022-01-01", description="Historical data start")
    DATA_FREQUENCY_INTRADAY: str = Field(default="5m", description="5-minute bars")
    DATA_FREQUENCY_DAILY: str = Field(default="1d", description="Daily bars")
    DATA_FREQUENCY_WEEKLY: str = Field(default="1wk", description="Weekly bars")
    
    # Database
    DATABASE_PATH: str = Field(default="data/trading_data.db", description="SQLite database path")
    
    # Model settings
    TIMESFM_MODEL_SIZE: str = Field(default="google/timesfm-1.0-500m", description="TimesFM model")
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