"""Weekly Predictor for Long-Term Trends and Seasonal Pattern Recognition.

This module implements seasonal decomposition neural networks optimized for weekly trading patterns.
Designed for 5-20 day prediction horizons with focus on:
- Weekly seasonal patterns and cycles
- Earnings cycle effects and quarterly patterns
- Sector rotation indicators
- Macroeconomic trend features and long-term momentum
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from loguru import logger
import warnings
from scipy import signal

from ...config.settings import config


class SeasonalDecompositionNetwork(nn.Module):
    """Neural network for seasonal decomposition and long-term pattern recognition."""
    
    def __init__(
        self,
        input_dim: int = 20,  # Extended features for weekly data
        hidden_dim: int = 256,
        seasonal_periods: List[int] = [5, 22, 66],  # Weekly, monthly, quarterly patterns
        dropout: float = 0.2,
        output_horizons: List[int] = [5, 10, 20]  # 5d, 10d, 20d predictions
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seasonal_periods = seasonal_periods
        self.output_horizons = output_horizons
        
        # Trend extraction network
        self.trend_network = TrendExtractionNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim // 2,
            dropout=dropout
        )
        
        # Seasonal pattern networks for different periods
        self.seasonal_networks = nn.ModuleDict()
        for period in seasonal_periods:
            self.seasonal_networks[f'seasonal_{period}'] = SeasonalPatternNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim // len(seasonal_periods),
                seasonal_period=period,
                dropout=dropout
            )
        
        # Cycle detection network (for earnings and market cycles)
        self.cycle_network = CycleDetectionNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim // 2,
            dropout=dropout
        )
        
        # Long-term memory component (LSTM for trend persistence)
        self.long_term_memory = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature fusion layer
        total_features = (
            hidden_dim // 2 +  # Trend
            sum(hidden_dim // len(seasonal_periods) for _ in seasonal_periods) +  # Seasonal
            hidden_dim // 2 +  # Cycle
            hidden_dim  # LSTM
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Horizon-specific prediction heads with increasing complexity for longer horizons
        self.prediction_heads = nn.ModuleDict()
        for horizon in output_horizons:
            # More complex networks for longer horizons
            complexity_factor = min(horizon // 5, 4)  # Scale complexity with horizon
            
            head_layers = []
            current_dim = hidden_dim
            
            for i in range(complexity_factor + 1):
                next_dim = max(hidden_dim // (2 ** i), 64)
                head_layers.extend([
                    nn.Linear(current_dim, next_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                current_dim = next_dim
            
            # Final prediction layer
            head_layers.append(nn.Linear(current_dim, 5))  # [return, confidence, volatility, trend, cycle_phase]
            
            self.prediction_heads[f'horizon_{horizon}'] = nn.Sequential(*head_layers)
        
        # Sector rotation prediction head
        self.sector_rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 11),  # 11 GICS sectors
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for weekly prediction.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary with predictions for each horizon
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Extract trend components
        trend_features = self.trend_network(x)
        
        # Extract seasonal components for different periods
        seasonal_features = []
        for period in self.seasonal_periods:
            seasonal_key = f'seasonal_{period}'
            seasonal_feat = self.seasonal_networks[seasonal_key](x)
            seasonal_features.append(seasonal_feat)
        
        # Detect cycles
        cycle_features = self.cycle_network(x)
        
        # Long-term memory processing
        lstm_output, (hidden, cell) = self.long_term_memory(x)
        # Use the final LSTM hidden state
        lstm_features = lstm_output[:, -1, :]  # (batch_size, hidden_dim)
        
        # Combine all features
        combined_features = torch.cat([
            trend_features,
            *seasonal_features,
            cycle_features,
            lstm_features
        ], dim=1)
        
        # Fuse features
        fused_features = self.fusion_layer(combined_features)
        
        # Generate predictions for each horizon
        outputs = {}
        for horizon in self.output_horizons:
            head_output = self.prediction_heads[f'horizon_{horizon}'](fused_features)
            
            outputs[f'horizon_{horizon}'] = {
                'prediction': head_output[:, 0],                      # Return prediction
                'confidence': torch.sigmoid(head_output[:, 1]),      # Confidence [0,1]
                'volatility': F.softplus(head_output[:, 2]),         # Volatility estimate
                'trend_strength': torch.tanh(head_output[:, 3]),     # Trend strength [-1,1]
                'cycle_phase': torch.sigmoid(head_output[:, 4])      # Cycle phase [0,1]
            }
        
        # Sector rotation probabilities
        sector_probs = self.sector_rotation_head(fused_features)
        
        # Add additional outputs to all horizons
        for horizon_key in outputs.keys():
            outputs[horizon_key]['sector_rotation_probs'] = sector_probs
            outputs[horizon_key]['trend_component'] = trend_features
            outputs[horizon_key]['seasonal_components'] = seasonal_features
            outputs[horizon_key]['cycle_component'] = cycle_features
        
        return outputs


class TrendExtractionNetwork(nn.Module):
    """Network for extracting long-term trend components."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        
        # Trend extraction using dilated convolutions
        self.trend_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, dilation=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, dilation=2, padding=6),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, dilation=4, padding=12),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Global average pooling for trend summary
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim) -> (batch_size, input_dim, seq_len)
        x_conv = x.transpose(1, 2)
        
        # Extract trend
        trend = self.trend_conv(x_conv)
        
        # Global pooling
        trend_pooled = self.global_pool(trend).squeeze(-1)  # (batch_size, hidden_dim)
        
        return trend_pooled


class SeasonalPatternNetwork(nn.Module):
    """Network for extracting seasonal patterns with specific periods."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seasonal_period: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.seasonal_period = seasonal_period
        
        # Fourier feature extraction for seasonality
        self.fourier_layers = nn.ModuleList()
        n_harmonics = min(seasonal_period // 2, 8)  # Limit number of harmonics
        
        for harmonic in range(1, n_harmonics + 1):
            fourier_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // n_harmonics),
                nn.Tanh()  # Tanh for periodic activation
            )
            self.fourier_layers.append(fourier_layer)
        
        # Seasonal pattern refinement
        self.seasonal_refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_dim = x.shape
        
        # Generate Fourier features for seasonality
        fourier_features = []
        
        for i, fourier_layer in enumerate(self.fourier_layers):
            harmonic = i + 1
            
            # Create periodic position encoding
            positions = torch.arange(seq_len, dtype=torch.float32, device=x.device)
            sin_encoding = torch.sin(2 * np.pi * harmonic * positions / self.seasonal_period)
            cos_encoding = torch.cos(2 * np.pi * harmonic * positions / self.seasonal_period)
            
            # Apply to features
            sin_features = fourier_layer(x) * sin_encoding.unsqueeze(0).unsqueeze(-1)
            cos_features = fourier_layer(x) * cos_encoding.unsqueeze(0).unsqueeze(-1)
            
            # Average over sequence length
            sin_avg = sin_features.mean(dim=1)  # (batch_size, hidden_dim // n_harmonics)
            cos_avg = cos_features.mean(dim=1)
            
            fourier_features.extend([sin_avg, cos_avg])
        
        # Concatenate all Fourier features
        seasonal_features = torch.cat(fourier_features, dim=1)  # (batch_size, hidden_dim)
        
        # Refine seasonal patterns
        refined_features = self.seasonal_refine(seasonal_features)
        
        return refined_features


class CycleDetectionNetwork(nn.Module):
    """Network for detecting and modeling market cycles."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        
        # Cycle detection using spectral analysis-inspired layers
        self.cycle_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=11, padding=5),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=21, padding=10),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Attention mechanism for cycle importance
        self.cycle_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Cycle summary
        self.cycle_summary = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim) -> (batch_size, input_dim, seq_len)
        x_conv = x.transpose(1, 2)
        
        # Detect cycles
        cycles = self.cycle_conv(x_conv)
        
        # Back to sequence format for attention
        cycles_seq = cycles.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
        
        # Apply attention to find important cycle patterns
        attended_cycles, _ = self.cycle_attention(cycles_seq, cycles_seq, cycles_seq)
        
        # Summarize cycles
        cycle_summary = attended_cycles.mean(dim=1)  # (batch_size, hidden_dim)
        refined_cycles = self.cycle_summary(cycle_summary)
        
        return refined_cycles


class WeeklyFeatureExtractor:
    """Extract weekly-resolution features from daily OHLCV data."""
    
    def __init__(self, lookback_weeks: int = 26):  # 6 months of weekly data
        """
        Initialize feature extractor.
        
        Args:
            lookback_weeks: Number of weeks to look back for feature computation
        """
        self.lookback_weeks = lookback_weeks
        self.lookback_days = lookback_weeks * 7  # Convert to days
        
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract weekly features from daily OHLCV data.
        
        Args:
            df: DataFrame with daily OHLCV data
            
        Returns:
            Feature array of shape (n_samples, n_features)
        """
        if len(df) < self.lookback_days:
            raise ValueError(f"Need at least {self.lookback_days} data points")
        
        # Resample to weekly data (end-of-week)
        weekly_df = self._resample_to_weekly(df)
        
        if len(weekly_df) < self.lookback_weeks:
            raise ValueError(f"Not enough weekly data: {len(weekly_df)} weeks")
        
        features = []
        
        # Basic weekly OHLCV features
        features.extend([
            weekly_df['open'].values,
            weekly_df['high'].values,
            weekly_df['low'].values,
            weekly_df['close'].values,
            weekly_df['volume'].values
        ])
        
        # Weekly returns and volatility
        weekly_returns = weekly_df['close'].pct_change().fillna(0).values
        weekly_volatility = weekly_df['close'].rolling(window=4).std().fillna(0).values  # Monthly volatility
        
        features.extend([weekly_returns, weekly_volatility])
        
        # Long-term technical indicators
        features.extend([
            self._calculate_weekly_rsi(weekly_df['close'], period=14),
            self._calculate_weekly_macd(weekly_df['close']),
            self._calculate_weekly_momentum(weekly_df['close']),
            self._calculate_earnings_cycle_position(weekly_df),
            self._calculate_sector_rotation_indicator(weekly_df),
            self._calculate_market_breadth(weekly_df),
            self._calculate_volatility_regime(weekly_df),
            self._calculate_trend_persistence(weekly_df['close']),
            self._calculate_seasonal_strength(weekly_df),
            self._calculate_quarterly_pattern(weekly_df),
            self._calculate_macro_trend_alignment(weekly_df),
            self._calculate_relative_strength(weekly_df)
        ])
        
        # Stack features
        feature_array = np.stack(features, axis=1)  # (n_samples, n_features)
        
        # Create rolling windows
        windowed_features = []
        for i in range(self.lookback_weeks, len(feature_array)):
            window = feature_array[i-self.lookback_weeks:i]
            windowed_features.append(window)
        
        if not windowed_features:
            raise ValueError("Not enough data for windowing")
        
        return np.array(windowed_features)  # (n_windows, lookback_weeks, n_features)
    
    def _resample_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample daily data to weekly (Friday close)."""
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Resample to weekly (Friday close)
        weekly_df = df.resample('W-FRI').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return weekly_df
    
    def _calculate_weekly_rsi(self, prices: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate RSI on weekly data."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values
    
    def _calculate_weekly_macd(self, prices: pd.Series) -> np.ndarray:
        """Calculate MACD on weekly data."""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        return macd.fillna(0).values
    
    def _calculate_weekly_momentum(self, prices: pd.Series) -> np.ndarray:
        """Calculate momentum indicators."""
        # 4-week and 12-week momentum
        momentum_4w = prices.pct_change(periods=4).fillna(0)
        momentum_12w = prices.pct_change(periods=12).fillna(0)
        return (momentum_4w + momentum_12w).values / 2
    
    def _calculate_earnings_cycle_position(self, df: pd.DataFrame) -> np.ndarray:
        """Estimate position within earnings cycle."""
        # Simplified earnings cycle (quarterly)
        weeks_in_quarter = 13
        cycle_position = []
        
        for i, idx in enumerate(df.index):
            week_of_year = idx.isocalendar()[1] if hasattr(idx, 'isocalendar') else 1
            quarter_week = week_of_year % weeks_in_quarter
            normalized_position = quarter_week / weeks_in_quarter
            cycle_position.append(normalized_position)
        
        return np.array(cycle_position)
    
    def _calculate_sector_rotation_indicator(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate sector rotation indicator (simplified)."""
        # Use price volatility as proxy for sector rotation intensity
        volatility = df['close'].rolling(window=8).std()
        sector_rotation = volatility / volatility.rolling(window=26).mean()
        return sector_rotation.fillna(1).values
    
    def _calculate_market_breadth(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate market breadth indicator."""
        # Use volume patterns as proxy for market breadth
        volume_trend = df['volume'].rolling(window=8).mean() / df['volume'].rolling(window=26).mean()
        return volume_trend.fillna(1).values
    
    def _calculate_volatility_regime(self, df: pd.DataFrame) -> np.ndarray:
        """Identify volatility regime."""
        # Calculate rolling volatility and regime
        vol = df['close'].rolling(window=4).std()
        vol_long = vol.rolling(window=26).mean()
        vol_regime = vol / vol_long
        return vol_regime.fillna(1).values
    
    def _calculate_trend_persistence(self, prices: pd.Series) -> np.ndarray:
        """Calculate trend persistence measure."""
        # Hurst exponent approximation
        returns = prices.pct_change().dropna()
        persistence = []
        
        for i in range(len(prices)):
            start_idx = max(0, i - 12)  # 3 months
            window_returns = returns.iloc[start_idx:i+1]
            
            if len(window_returns) < 8:
                persistence.append(0.5)
                continue
            
            # Simple trend persistence: correlation with time
            time_index = np.arange(len(window_returns))
            if len(time_index) > 1:
                correlation = np.corrcoef(time_index, window_returns.values)[0, 1]
                persistence.append(abs(correlation) if not np.isnan(correlation) else 0.5)
            else:
                persistence.append(0.5)
        
        return np.array(persistence)
    
    def _calculate_seasonal_strength(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate seasonal pattern strength."""
        # Use week-of-year patterns
        seasonal_strength = []
        
        for i, idx in enumerate(df.index):
            week_of_year = idx.isocalendar()[1] if hasattr(idx, 'isocalendar') else 1
            
            # Simple seasonal pattern based on week number
            seasonal_factor = np.sin(2 * np.pi * week_of_year / 52)
            seasonal_strength.append(seasonal_factor)
        
        return np.array(seasonal_strength)
    
    def _calculate_quarterly_pattern(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate quarterly pattern indicator."""
        quarterly_pattern = []
        
        for i, idx in enumerate(df.index):
            week_of_year = idx.isocalendar()[1] if hasattr(idx, 'isocalendar') else 1
            quarter = (week_of_year - 1) // 13  # 0, 1, 2, 3
            
            # Quarterly pattern (Q4 and Q1 often show different behavior)
            if quarter == 0 or quarter == 3:  # Q1 and Q4
                quarterly_pattern.append(1.0)
            else:  # Q2 and Q3
                quarterly_pattern.append(-1.0)
        
        return np.array(quarterly_pattern)
    
    def _calculate_macro_trend_alignment(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate macro trend alignment."""
        # Long-term trend alignment
        short_ma = df['close'].rolling(window=4).mean()   # 1 month
        long_ma = df['close'].rolling(window=26).mean()   # 6 months
        
        trend_alignment = (short_ma - long_ma) / long_ma
        return trend_alignment.fillna(0).values
    
    def _calculate_relative_strength(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate relative strength indicator."""
        # Relative to its own long-term average
        current_level = df['close']
        long_term_avg = current_level.rolling(window=52).mean()  # 1 year average
        
        relative_strength = (current_level - long_term_avg) / long_term_avg
        return relative_strength.fillna(0).values


class WeeklyPredictor:
    """Main predictor class for weekly long-term pattern recognition."""
    
    def __init__(
        self,
        lookback_weeks: int = 26,  # 6 months
        hidden_dim: int = 256,
        prediction_horizons: List[int] = [5, 10, 20],  # 5d, 10d, 20d
        device: str = "auto"
    ):
        """
        Initialize weekly predictor.
        
        Args:
            lookback_weeks: Weeks of history to use for prediction
            hidden_dim: Hidden dimension for neural network
            prediction_horizons: List of prediction horizons in days
            device: Computation device ('auto', 'cpu', 'cuda')
        """
        self.lookback_weeks = lookback_weeks
        self.prediction_horizons = prediction_horizons
        self.hidden_dim = hidden_dim
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize components
        self.feature_extractor = WeeklyFeatureExtractor(lookback_weeks)
        
        # Initialize model (will be created after first data sample)
        self.model = None
        self.is_trained = False
        
        logger.info(f"WeeklyPredictor initialized on {self.device}")
    
    def _initialize_model(self, input_dim: int):
        """Initialize the neural network model."""
        self.model = SeasonalDecompositionNetwork(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            seasonal_periods=[5, 22, 66],  # Weekly, monthly, quarterly
            dropout=0.2,
            output_horizons=self.prediction_horizons
        ).to(self.device)
        
        logger.info(f"Initialized SeasonalDecompositionNetwork with input_dim={input_dim}")
    
    def predict(
        self,
        df: pd.DataFrame,
        ticker: str,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Generate weekly predictions.
        
        Args:
            df: DataFrame with daily OHLCV data
            ticker: Stock ticker symbol
            return_confidence: Whether to return confidence estimates
            
        Returns:
            Dictionary with predictions for each horizon
        """
        try:
            # Check minimum data requirement
            min_required = self.lookback_weeks * 7 + 50  # Extra buffer for weekly resampling
            if len(df) < min_required:
                logger.warning(f"Insufficient data for weekly prediction {ticker}: {len(df)} points")
                return self._fallback_prediction(ticker)
            
            # Extract features
            features = self.feature_extractor.extract_features(df)
            
            # Initialize model if needed
            if self.model is None:
                input_dim = features.shape[-1]
                self._initialize_model(input_dim)
            
            # Use only the latest window for prediction
            latest_features = features[-1:]  # (1, lookback_weeks, n_features)
            
            # Convert to tensor
            x = torch.FloatTensor(latest_features).to(self.device)
            
            # Generate prediction
            self.model.eval()
            with torch.no_grad():
                model_outputs = self.model(x)
            
            # Process outputs
            predictions = {}
            for horizon in self.prediction_horizons:
                horizon_key = f"horizon_{horizon}"
                model_out = model_outputs[horizon_key]
                
                # Extract predictions and convert to numpy
                prediction = float(model_out['prediction'][0].cpu().numpy())
                confidence = float(model_out['confidence'][0].cpu().numpy()) if return_confidence else 0.7
                volatility = float(model_out['volatility'][0].cpu().numpy())
                trend_strength = float(model_out['trend_strength'][0].cpu().numpy())
                cycle_phase = float(model_out['cycle_phase'][0].cpu().numpy())
                
                # Sector rotation probabilities
                sector_probs = model_out['sector_rotation_probs'][0].cpu().numpy()
                
                # Create prediction intervals using volatility
                interval_width = volatility * 1.645  # 90% confidence intervals for longer horizons
                lower_bound = prediction - interval_width
                upper_bound = prediction + interval_width
                
                predictions[horizon_key] = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'prediction_interval': [lower_bound, upper_bound],
                    'volatility_forecast': volatility,
                    'trend_strength': trend_strength,
                    'cycle_phase': cycle_phase,
                    'sector_rotation_probs': sector_probs.tolist(),
                    'model_type': 'weekly_seasonal_decomposition'
                }
            
            # Add metadata
            predictions['weekly_meta'] = {
                'ticker': ticker,
                'data_points': len(df),
                'lookback_weeks': self.lookback_weeks,
                'features_used': features.shape[-1],
                'seasonal_decomposition': True,
                'sector_rotation_analysis': True,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.debug(f"Generated weekly predictions for {ticker}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error in weekly prediction for {ticker}: {e}")
            return self._fallback_prediction(ticker)
    
    def _fallback_prediction(self, ticker: str) -> Dict[str, Any]:
        """Generate fallback predictions when model fails."""
        predictions = {}
        
        for horizon in self.prediction_horizons:
            horizon_key = f"horizon_{horizon}"
            
            # Simple trend-following prediction with horizon adjustment
            base_return = 0.002 * np.log(horizon)  # Logarithmic scaling with horizon
            prediction = np.random.normal(base_return, 0.01)
            prediction = np.clip(prediction, -0.1, 0.1)  # Clip to ±10%
            
            predictions[horizon_key] = {
                'prediction': float(prediction),
                'confidence': 0.5,
                'prediction_interval': [float(prediction - 0.05), float(prediction + 0.05)],
                'volatility_forecast': 0.03,
                'trend_strength': 0.0,
                'cycle_phase': 0.5,
                'sector_rotation_probs': [0.09] * 11,  # Equal probability for 11 sectors
                'model_type': 'weekly_fallback'
            }
        
        predictions['weekly_meta'] = {
            'ticker': ticker,
            'fallback_used': True,
            'timestamp': datetime.now().isoformat()
        }
        
        return predictions
    
    def train_model(
        self,
        training_data: List[pd.DataFrame],
        tickers: List[str],
        epochs: int = 20,
        batch_size: int = 8,
        learning_rate: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Train the weekly prediction model.
        
        Args:
            training_data: List of DataFrames with daily OHLCV data
            tickers: Corresponding ticker symbols
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            
        Returns:
            Training statistics and performance metrics
        """
        logger.info(f"Training weekly model on {len(training_data)} datasets")
        
        try:
            # Prepare training data
            X_train, y_train = self._prepare_training_data(training_data, tickers)
            
            if len(X_train) < 3:
                logger.warning("Insufficient training data for weekly model")
                return {'error': 'insufficient_data'}
            
            # Initialize model if needed
            if self.model is None:
                input_dim = X_train.shape[-1]
                self._initialize_model(input_dim)
            
            # Training setup
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=1e-3
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.5, verbose=True
            )
            criterion = nn.MSELoss()
            
            # Training loop
            training_losses = []
            self.model.train()
            
            for epoch in range(epochs):
                epoch_losses = []
                
                # Create batches
                indices = torch.randperm(len(X_train))
                for i in range(0, len(X_train), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    X_batch = X_train[batch_indices]
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(X_batch)
                    
                    # Calculate loss for each horizon
                    total_loss = 0
                    for horizon in self.prediction_horizons:
                        horizon_key = f"horizon_{horizon}"
                        y_batch = y_train[horizon_key][batch_indices]
                        
                        # Prediction loss
                        pred_loss = criterion(outputs[horizon_key]['prediction'], y_batch)
                        total_loss += pred_loss
                    
                    # Backward pass
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_losses.append(total_loss.item())
                
                avg_loss = np.mean(epoch_losses)
                training_losses.append(avg_loss)
                scheduler.step(avg_loss)
                
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            self.is_trained = True
            logger.info("Weekly model training completed")
            
            return {
                'training_samples': len(X_train),
                'epochs': epochs,
                'final_loss': training_losses[-1],
                'training_losses': training_losses,
                'model_trained': True
            }
            
        except Exception as e:
            logger.error(f"Error training weekly model: {e}")
            return {'error': str(e)}
    
    def _prepare_training_data(
        self,
        training_data: List[pd.DataFrame],
        tickers: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare training data from historical daily data."""
        X_all = []
        y_all = {f"horizon_{h}": [] for h in self.prediction_horizons}
        
        for df, ticker in zip(training_data, tickers):
            try:
                min_required = self.lookback_weeks * 7 + max(self.prediction_horizons) + 50
                if len(df) < min_required:
                    continue
                
                # Extract features
                features = self.feature_extractor.extract_features(df)
                
                # Resample target data to align with weekly features
                weekly_df = self.feature_extractor._resample_to_weekly(df)
                
                # Create targets for each horizon
                targets = {}
                for horizon in self.prediction_horizons:
                    # Calculate forward returns (convert daily horizon to weekly)
                    weekly_horizon = max(1, horizon // 7)  # Convert to weeks
                    forward_returns = weekly_df['close'].pct_change(periods=weekly_horizon).shift(-weekly_horizon)
                    
                    # Align with features
                    start_idx = self.lookback_weeks
                    end_idx = start_idx + len(features)
                    
                    if end_idx <= len(forward_returns):
                        valid_targets = forward_returns.iloc[start_idx:end_idx]
                        valid_targets = valid_targets.dropna()
                        
                        # Ensure alignment
                        min_len = min(len(features), len(valid_targets))
                        if min_len > 0:
                            features_aligned = features[:min_len]
                            targets[f"horizon_{horizon}"] = valid_targets.iloc[:min_len].values
                        else:
                            continue
                    else:
                        continue
                
                # Add to training set
                if len(features_aligned) > 0 and all(len(t) > 0 for t in targets.values()):
                    X_all.append(features_aligned)
                    for horizon in self.prediction_horizons:
                        y_all[f"horizon_{horizon}"].append(targets[f"horizon_{horizon}"])
                
            except Exception as e:
                logger.debug(f"Error processing weekly training data for {ticker}: {e}")
                continue
        
        if not X_all:
            raise ValueError("No valid training data found")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(np.concatenate(X_all, axis=0)).to(self.device)
        y_tensors = {}
        for horizon in self.prediction_horizons:
            horizon_key = f"horizon_{horizon}"
            y_tensors[horizon_key] = torch.FloatTensor(np.concatenate(y_all[horizon_key])).to(self.device)
        
        return X_tensor, y_tensors
    
    def save_model(self, path: str) -> bool:
        """Save the trained model."""
        try:
            if self.model is None:
                logger.warning("No model to save")
                return False
                
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'lookback_weeks': self.lookback_weeks,
                'prediction_horizons': self.prediction_horizons,
                'hidden_dim': self.hidden_dim,
                'is_trained': self.is_trained
            }, path)
            
            logger.info(f"Weekly model saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving weekly model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load a trained model."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.lookback_weeks = checkpoint['lookback_weeks']
            self.prediction_horizons = checkpoint['prediction_horizons']
            self.hidden_dim = checkpoint.get('hidden_dim', 256)
            self.is_trained = checkpoint['is_trained']
            
            # Initialize model structure
            if self.model is None:
                # Dummy input to get dimensions
                dummy_features = np.random.randn(1, self.lookback_weeks, 20)
                input_dim = dummy_features.shape[-1]
                self._initialize_model(input_dim)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"Weekly model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading weekly model: {e}")
            return False
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'model_type': 'WeeklyPredictor',
            'lookback_weeks': self.lookback_weeks,
            'prediction_horizons': self.prediction_horizons,
            'hidden_dim': self.hidden_dim,
            'device': str(self.device),
            'is_loaded': self.is_loaded,
            'is_trained': self.is_trained,
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            'seasonal_decomposition': True,
            'sector_analysis': True,
            'cycle_detection': True
        }