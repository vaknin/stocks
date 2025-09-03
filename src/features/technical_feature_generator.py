"""Technical feature generator with regime-aware indicator selection."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import warnings
from loguru import logger
from collections import deque
from dataclasses import dataclass
from enum import Enum

from ..config.settings import config


class TechnicalIndicatorType(Enum):
    """Types of technical indicators."""
    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"
    OSCILLATOR = "oscillator"


class MarketRegimeType(Enum):
    """Market regime types for indicator selection."""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


@dataclass
class TechnicalIndicator:
    """Technical indicator configuration."""
    name: str
    indicator_type: TechnicalIndicatorType
    calculation_func: callable
    parameters: Dict[str, Any]
    regime_weights: Dict[MarketRegimeType, float]
    lookback_periods: List[int]
    normalization_method: str = "z_score"
    feature_importance_weight: float = 1.0


class TechnicalFeatureGenerator:
    """
    Advanced technical feature generator with regime-aware indicator selection.
    
    Features:
    - Regime-aware indicator selection and weighting
    - Dynamic parameter optimization based on market conditions
    - Multi-timeframe technical indicators
    - Feature normalization and scaling
    - Automated feature importance ranking
    """
    
    def __init__(
        self,
        lookback_window: int = 252,  # 1 year of daily data
        regime_detection_window: int = 20,
        feature_selection_threshold: float = 0.05,
        enable_dynamic_parameters: bool = True,
        enable_multi_timeframe: bool = True
    ):
        """
        Initialize technical feature generator.
        
        Args:
            lookback_window: Historical data window for calculations
            regime_detection_window: Window for regime detection
            feature_selection_threshold: Minimum importance threshold for feature selection
            enable_dynamic_parameters: Enable dynamic parameter optimization
            enable_multi_timeframe: Enable multi-timeframe indicators
        """
        self.lookback_window = lookback_window
        self.regime_detection_window = regime_detection_window
        self.feature_selection_threshold = feature_selection_threshold
        self.enable_dynamic_parameters = enable_dynamic_parameters
        self.enable_multi_timeframe = enable_multi_timeframe
        
        # Feature cache for performance
        self.feature_cache = {}
        self.cache_timestamps = {}
        self.cache_ttl_minutes = 15
        
        # Feature importance tracking
        self.feature_importance_history = deque(maxlen=1000)
        self.indicator_performance_tracking = {}
        
        # Initialize technical indicators
        self.indicators = self._initialize_indicators()
        
        # Current market regime cache
        self.current_regime_cache = {}
        
        logger.info(f"TechnicalFeatureGenerator initialized with {len(self.indicators)} indicators")
    
    def _initialize_indicators(self) -> Dict[str, TechnicalIndicator]:
        """Initialize all technical indicators with regime weights."""
        indicators = {}
        
        # Momentum Indicators
        indicators['rsi'] = TechnicalIndicator(
            name='rsi',
            indicator_type=TechnicalIndicatorType.MOMENTUM,
            calculation_func=self._calculate_rsi,
            parameters={'period': 14},
            regime_weights={
                MarketRegimeType.TRENDING_BULL: 0.8,
                MarketRegimeType.TRENDING_BEAR: 0.9,
                MarketRegimeType.SIDEWAYS: 1.0,
                MarketRegimeType.HIGH_VOLATILITY: 0.7,
                MarketRegimeType.LOW_VOLATILITY: 0.8,
                MarketRegimeType.BREAKOUT: 0.6,
                MarketRegimeType.REVERSAL: 1.0
            },
            lookback_periods=[14, 21, 30],
            feature_importance_weight=0.9
        )
        
        indicators['macd'] = TechnicalIndicator(
            name='macd',
            indicator_type=TechnicalIndicatorType.MOMENTUM,
            calculation_func=self._calculate_macd,
            parameters={'fast': 12, 'slow': 26, 'signal': 9},
            regime_weights={
                MarketRegimeType.TRENDING_BULL: 1.0,
                MarketRegimeType.TRENDING_BEAR: 1.0,
                MarketRegimeType.SIDEWAYS: 0.6,
                MarketRegimeType.HIGH_VOLATILITY: 0.8,
                MarketRegimeType.LOW_VOLATILITY: 0.9,
                MarketRegimeType.BREAKOUT: 0.9,
                MarketRegimeType.REVERSAL: 0.8
            },
            lookback_periods=[12, 26],
            feature_importance_weight=0.85
        )
        
        indicators['stochastic'] = TechnicalIndicator(
            name='stochastic',
            indicator_type=TechnicalIndicatorType.OSCILLATOR,
            calculation_func=self._calculate_stochastic,
            parameters={'k_period': 14, 'd_period': 3},
            regime_weights={
                MarketRegimeType.TRENDING_BULL: 0.7,
                MarketRegimeType.TRENDING_BEAR: 0.7,
                MarketRegimeType.SIDEWAYS: 1.0,
                MarketRegimeType.HIGH_VOLATILITY: 0.8,
                MarketRegimeType.LOW_VOLATILITY: 0.9,
                MarketRegimeType.BREAKOUT: 0.6,
                MarketRegimeType.REVERSAL: 1.0
            },
            lookback_periods=[14],
            feature_importance_weight=0.7
        )
        
        # Trend Indicators
        indicators['sma_ratios'] = TechnicalIndicator(
            name='sma_ratios',
            indicator_type=TechnicalIndicatorType.TREND,
            calculation_func=self._calculate_sma_ratios,
            parameters={'periods': [5, 10, 20, 50, 200]},
            regime_weights={
                MarketRegimeType.TRENDING_BULL: 1.0,
                MarketRegimeType.TRENDING_BEAR: 1.0,
                MarketRegimeType.SIDEWAYS: 0.5,
                MarketRegimeType.HIGH_VOLATILITY: 0.6,
                MarketRegimeType.LOW_VOLATILITY: 0.8,
                MarketRegimeType.BREAKOUT: 0.9,
                MarketRegimeType.REVERSAL: 0.7
            },
            lookback_periods=[5, 10, 20, 50, 200],
            feature_importance_weight=0.8
        )
        
        indicators['ema_ratios'] = TechnicalIndicator(
            name='ema_ratios',
            indicator_type=TechnicalIndicatorType.TREND,
            calculation_func=self._calculate_ema_ratios,
            parameters={'periods': [5, 10, 20, 50]},
            regime_weights={
                MarketRegimeType.TRENDING_BULL: 1.0,
                MarketRegimeType.TRENDING_BEAR: 1.0,
                MarketRegimeType.SIDEWAYS: 0.6,
                MarketRegimeType.HIGH_VOLATILITY: 0.8,
                MarketRegimeType.LOW_VOLATILITY: 0.7,
                MarketRegimeType.BREAKOUT: 1.0,
                MarketRegimeType.REVERSAL: 0.8
            },
            lookback_periods=[5, 10, 20, 50],
            feature_importance_weight=0.75
        )
        
        indicators['adx'] = TechnicalIndicator(
            name='adx',
            indicator_type=TechnicalIndicatorType.TREND,
            calculation_func=self._calculate_adx,
            parameters={'period': 14},
            regime_weights={
                MarketRegimeType.TRENDING_BULL: 1.0,
                MarketRegimeType.TRENDING_BEAR: 1.0,
                MarketRegimeType.SIDEWAYS: 0.4,
                MarketRegimeType.HIGH_VOLATILITY: 0.9,
                MarketRegimeType.LOW_VOLATILITY: 0.6,
                MarketRegimeType.BREAKOUT: 1.0,
                MarketRegimeType.REVERSAL: 0.7
            },
            lookback_periods=[14],
            feature_importance_weight=0.9
        )
        
        # Volatility Indicators
        indicators['bollinger_bands'] = TechnicalIndicator(
            name='bollinger_bands',
            indicator_type=TechnicalIndicatorType.VOLATILITY,
            calculation_func=self._calculate_bollinger_bands,
            parameters={'period': 20, 'std_dev': 2},
            regime_weights={
                MarketRegimeType.TRENDING_BULL: 0.7,
                MarketRegimeType.TRENDING_BEAR: 0.7,
                MarketRegimeType.SIDEWAYS: 1.0,
                MarketRegimeType.HIGH_VOLATILITY: 1.0,
                MarketRegimeType.LOW_VOLATILITY: 0.8,
                MarketRegimeType.BREAKOUT: 0.9,
                MarketRegimeType.REVERSAL: 1.0
            },
            lookback_periods=[20],
            feature_importance_weight=0.8
        )
        
        indicators['atr'] = TechnicalIndicator(
            name='atr',
            indicator_type=TechnicalIndicatorType.VOLATILITY,
            calculation_func=self._calculate_atr,
            parameters={'period': 14},
            regime_weights={
                MarketRegimeType.TRENDING_BULL: 0.8,
                MarketRegimeType.TRENDING_BEAR: 0.8,
                MarketRegimeType.SIDEWAYS: 0.9,
                MarketRegimeType.HIGH_VOLATILITY: 1.0,
                MarketRegimeType.LOW_VOLATILITY: 0.7,
                MarketRegimeType.BREAKOUT: 1.0,
                MarketRegimeType.REVERSAL: 0.9
            },
            lookback_periods=[14],
            feature_importance_weight=0.85
        )
        
        # Volume Indicators
        indicators['obv'] = TechnicalIndicator(
            name='obv',
            indicator_type=TechnicalIndicatorType.VOLUME,
            calculation_func=self._calculate_obv,
            parameters={},
            regime_weights={
                MarketRegimeType.TRENDING_BULL: 1.0,
                MarketRegimeType.TRENDING_BEAR: 1.0,
                MarketRegimeType.SIDEWAYS: 0.6,
                MarketRegimeType.HIGH_VOLATILITY: 0.8,
                MarketRegimeType.LOW_VOLATILITY: 0.7,
                MarketRegimeType.BREAKOUT: 1.0,
                MarketRegimeType.REVERSAL: 0.8
            },
            lookback_periods=[20],
            feature_importance_weight=0.7
        )
        
        indicators['volume_sma_ratio'] = TechnicalIndicator(
            name='volume_sma_ratio',
            indicator_type=TechnicalIndicatorType.VOLUME,
            calculation_func=self._calculate_volume_sma_ratio,
            parameters={'periods': [10, 20, 50]},
            regime_weights={
                MarketRegimeType.TRENDING_BULL: 0.8,
                MarketRegimeType.TRENDING_BEAR: 0.8,
                MarketRegimeType.SIDEWAYS: 0.5,
                MarketRegimeType.HIGH_VOLATILITY: 1.0,
                MarketRegimeType.LOW_VOLATILITY: 0.6,
                MarketRegimeType.BREAKOUT: 1.0,
                MarketRegimeType.REVERSAL: 0.9
            },
            lookback_periods=[10, 20, 50],
            feature_importance_weight=0.6
        )
        
        # Support/Resistance Indicators
        indicators['pivot_points'] = TechnicalIndicator(
            name='pivot_points',
            indicator_type=TechnicalIndicatorType.SUPPORT_RESISTANCE,
            calculation_func=self._calculate_pivot_points,
            parameters={'method': 'standard'},
            regime_weights={
                MarketRegimeType.TRENDING_BULL: 0.6,
                MarketRegimeType.TRENDING_BEAR: 0.6,
                MarketRegimeType.SIDEWAYS: 1.0,
                MarketRegimeType.HIGH_VOLATILITY: 0.7,
                MarketRegimeType.LOW_VOLATILITY: 0.9,
                MarketRegimeType.BREAKOUT: 0.8,
                MarketRegimeType.REVERSAL: 1.0
            },
            lookback_periods=[1],
            feature_importance_weight=0.7
        )
        
        return indicators
    
    def detect_market_regime(
        self,
        df: pd.DataFrame,
        cache_key: Optional[str] = None
    ) -> MarketRegimeType:
        """
        Detect current market regime based on price action and volatility.
        
        Args:
            df: OHLCV price data
            cache_key: Optional cache key for regime detection
            
        Returns:
            Detected market regime type
        """
        try:
            # Check cache if available
            if cache_key and cache_key in self.current_regime_cache:
                cache_time, cached_regime = self.current_regime_cache[cache_key]
                if (datetime.now() - cache_time).seconds < 300:  # 5-minute cache
                    return cached_regime
            
            if len(df) < self.regime_detection_window:
                regime = MarketRegimeType.SIDEWAYS  # Default regime
            else:
                regime = self._classify_market_regime(df)
            
            # Cache result
            if cache_key:
                self.current_regime_cache[cache_key] = (datetime.now(), regime)
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegimeType.SIDEWAYS
    
    def _classify_market_regime(self, df: pd.DataFrame) -> MarketRegimeType:
        """Classify market regime using technical analysis."""
        try:
            recent_df = df.tail(self.regime_detection_window)
            returns = recent_df['close'].pct_change().dropna()
            
            if len(returns) < 10:
                return MarketRegimeType.SIDEWAYS
            
            # Calculate regime indicators
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            trend_strength = self._calculate_trend_strength(recent_df)
            momentum = self._calculate_momentum_strength(recent_df)
            volume_trend = self._calculate_volume_trend(recent_df)
            
            # Regime classification logic
            if volatility > 0.35:  # High volatility threshold
                return MarketRegimeType.HIGH_VOLATILITY
            elif volatility < 0.15:  # Low volatility threshold
                return MarketRegimeType.LOW_VOLATILITY
            elif abs(trend_strength) > 0.6:  # Strong trend
                if trend_strength > 0:
                    return MarketRegimeType.TRENDING_BULL
                else:
                    return MarketRegimeType.TRENDING_BEAR
            elif momentum > 0.7 and volume_trend > 0.5:  # Potential breakout
                return MarketRegimeType.BREAKOUT
            elif abs(momentum) > 0.8:  # Strong momentum change - potential reversal
                return MarketRegimeType.REVERSAL
            else:
                return MarketRegimeType.SIDEWAYS
                
        except Exception as e:
            logger.error(f"Error classifying market regime: {e}")
            return MarketRegimeType.SIDEWAYS
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength indicator."""
        try:
            if len(df) < 20:
                return 0.0
            
            # Linear regression trend on close prices
            close_prices = df['close'].values
            x = np.arange(len(close_prices))
            slope = np.polyfit(x, close_prices, 1)[0]
            
            # Normalize by current price
            normalized_slope = slope / close_prices[-1]
            
            # R-squared for trend consistency
            p = np.poly1d(np.polyfit(x, close_prices, 1))
            ss_res = np.sum((close_prices - p(x)) ** 2)
            ss_tot = np.sum((close_prices - np.mean(close_prices)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-8))
            
            # Combine slope direction with trend consistency
            trend_strength = normalized_slope * r_squared * 100
            
            return max(-1.0, min(1.0, trend_strength))
            
        except Exception as e:
            logger.debug(f"Error calculating trend strength: {e}")
            return 0.0
    
    def _calculate_momentum_strength(self, df: pd.DataFrame) -> float:
        """Calculate momentum strength indicator."""
        try:
            if len(df) < 10:
                return 0.0
            
            # Price momentum
            price_momentum = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
            
            # Volume momentum (if available)
            volume_momentum = 0.0
            if 'volume' in df.columns and len(df) >= 10:
                recent_volume = df['volume'].tail(5).mean()
                historical_volume = df['volume'].head(-5).tail(5).mean()
                if historical_volume > 0:
                    volume_momentum = (recent_volume - historical_volume) / historical_volume
            
            # Combined momentum
            momentum_strength = price_momentum + 0.3 * volume_momentum
            
            return max(-1.0, min(1.0, momentum_strength))
            
        except Exception as e:
            logger.debug(f"Error calculating momentum strength: {e}")
            return 0.0
    
    def _calculate_volume_trend(self, df: pd.DataFrame) -> float:
        """Calculate volume trend indicator."""
        try:
            if 'volume' not in df.columns or len(df) < 10:
                return 0.5  # Neutral
            
            # Volume trend using linear regression
            volume_values = df['volume'].values
            x = np.arange(len(volume_values))
            slope = np.polyfit(x, volume_values, 1)[0]
            
            # Normalize by mean volume
            mean_volume = np.mean(volume_values)
            if mean_volume > 0:
                normalized_slope = slope / mean_volume
                return max(0.0, min(1.0, normalized_slope + 0.5))
            else:
                return 0.5
                
        except Exception as e:
            logger.debug(f"Error calculating volume trend: {e}")
            return 0.5
    
    def generate_features(
        self,
        df: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        ticker: str,
        regime_override: Optional[MarketRegimeType] = None
    ) -> np.ndarray:
        """
        Generate technical features for a given stock.
        
        Args:
            df: OHLCV price data (single DataFrame or dict for multi-asset)
            ticker: Stock ticker symbol
            regime_override: Optional regime override for testing
            
        Returns:
            Array of technical features
        """
        try:
            # Handle input data format
            if isinstance(df, dict):
                if ticker not in df:
                    logger.warning(f"Ticker {ticker} not found in multi-asset data")
                    return self._get_default_features()
                primary_df = df[ticker]
                multi_asset_data = df
            else:
                primary_df = df
                multi_asset_data = {ticker: df}
            
            # Check minimum data requirements
            if len(primary_df) < 50:
                logger.warning(f"Insufficient data for {ticker}: {len(primary_df)} rows")
                return self._get_default_features()
            
            # Generate cache key
            cache_key = f"{ticker}_{len(primary_df)}_{primary_df.index[-1]}"
            
            # Check feature cache
            if self._is_cache_valid(cache_key):
                logger.debug(f"Using cached features for {ticker}")
                return self.feature_cache[cache_key]
            
            # Detect market regime
            regime = regime_override or self.detect_market_regime(primary_df, cache_key)
            
            # Generate features for each indicator
            all_features = []
            feature_names = []
            
            for indicator_name, indicator in self.indicators.items():
                try:
                    # Get regime weight for this indicator
                    regime_weight = indicator.regime_weights.get(regime, 0.5)
                    
                    # Skip indicators with very low regime weights
                    if regime_weight < self.feature_selection_threshold:
                        continue
                    
                    # Calculate indicator values
                    indicator_features = indicator.calculation_func(
                        primary_df, 
                        **indicator.parameters
                    )
                    
                    # Apply regime weighting and normalization
                    if indicator_features is not None and len(indicator_features) > 0:
                        weighted_features = self._apply_regime_weighting(
                            indicator_features, regime_weight, indicator.normalization_method
                        )
                        all_features.extend(weighted_features)
                        feature_names.extend([f"{indicator_name}_{i}" for i in range(len(weighted_features))])
                    
                except Exception as e:
                    logger.debug(f"Error calculating {indicator_name} for {ticker}: {e}")
                    continue
            
            # Add multi-timeframe features if enabled
            if self.enable_multi_timeframe and len(primary_df) >= 100:
                multi_tf_features = self._generate_multi_timeframe_features(primary_df, regime)
                all_features.extend(multi_tf_features)
                feature_names.extend([f"multi_tf_{i}" for i in range(len(multi_tf_features))])
            
            # Add cross-asset features if multiple assets available
            if len(multi_asset_data) > 1:
                cross_asset_features = self._generate_cross_asset_features(multi_asset_data, ticker, regime)
                all_features.extend(cross_asset_features)
                feature_names.extend([f"cross_asset_{i}" for i in range(len(cross_asset_features))])
            
            # Convert to numpy array and handle edge cases
            features_array = np.array(all_features, dtype=np.float32)
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Clip extreme values
            features_array = np.clip(features_array, -5.0, 5.0)
            
            # Ensure minimum feature count (pad with zeros if necessary)
            min_features = 50
            if len(features_array) < min_features:
                features_array = np.pad(features_array, (0, min_features - len(features_array)))
            
            # Cache the result
            self._cache_features(cache_key, features_array)
            
            logger.debug(f"Generated {len(features_array)} technical features for {ticker} (regime: {regime.value})")
            
            return features_array
            
        except Exception as e:
            logger.error(f"Error generating technical features for {ticker}: {e}")
            return self._get_default_features()
    
    def _apply_regime_weighting(
        self, 
        features: List[float], 
        regime_weight: float, 
        normalization_method: str
    ) -> List[float]:
        """Apply regime weighting and normalization to features."""
        try:
            features = np.array(features)
            
            # Apply normalization
            if normalization_method == "z_score":
                if np.std(features) > 0:
                    features = (features - np.mean(features)) / np.std(features)
            elif normalization_method == "min_max":
                if np.max(features) - np.min(features) > 0:
                    features = (features - np.min(features)) / (np.max(features) - np.min(features))
            elif normalization_method == "tanh":
                features = np.tanh(features)
            
            # Apply regime weighting
            weighted_features = features * regime_weight
            
            return weighted_features.tolist()
            
        except Exception as e:
            logger.debug(f"Error applying regime weighting: {e}")
            return features if isinstance(features, list) else [0.0]
    
    def _generate_multi_timeframe_features(
        self, 
        df: pd.DataFrame, 
        regime: MarketRegimeType
    ) -> List[float]:
        """Generate multi-timeframe technical features."""
        try:
            features = []
            
            # Different timeframes (using rolling windows on daily data)
            timeframes = {
                'short': 5,    # ~1 week
                'medium': 20,  # ~1 month
                'long': 60     # ~3 months
            }
            
            for tf_name, window in timeframes.items():
                if len(df) < window:
                    features.extend([0.0] * 3)  # Default features
                    continue
                
                tf_data = df.tail(window)
                
                # Trend consistency across timeframe
                trend_consistency = self._calculate_trend_strength(tf_data)
                features.append(trend_consistency)
                
                # Volatility ratio (short vs long term)
                returns = tf_data['close'].pct_change().dropna()
                if len(returns) > 0:
                    volatility = returns.std() * np.sqrt(252)
                    features.append(volatility)
                else:
                    features.append(0.0)
                
                # Price momentum
                momentum = (tf_data['close'].iloc[-1] - tf_data['close'].iloc[0]) / tf_data['close'].iloc[0]
                features.append(momentum)
            
            return features[:9]  # 3 timeframes * 3 features each
            
        except Exception as e:
            logger.debug(f"Error generating multi-timeframe features: {e}")
            return [0.0] * 9
    
    def _generate_cross_asset_features(
        self, 
        multi_asset_data: Dict[str, pd.DataFrame], 
        primary_ticker: str, 
        regime: MarketRegimeType
    ) -> List[float]:
        """Generate cross-asset correlation and relative strength features."""
        try:
            features = []
            
            if primary_ticker not in multi_asset_data:
                return [0.0] * 5  # Default cross-asset features
            
            primary_df = multi_asset_data[primary_ticker]
            other_tickers = [t for t in multi_asset_data.keys() if t != primary_ticker]
            
            if not other_tickers or len(primary_df) < 20:
                return [0.0] * 5
            
            # Calculate cross-correlations
            primary_returns = primary_df['close'].pct_change().dropna().tail(20)
            correlations = []
            
            for other_ticker in other_tickers[:5]:  # Limit to 5 other assets
                try:
                    other_df = multi_asset_data[other_ticker]
                    other_returns = other_df['close'].pct_change().dropna().tail(20)
                    
                    # Align data
                    min_length = min(len(primary_returns), len(other_returns))
                    if min_length >= 10:
                        corr = np.corrcoef(
                            primary_returns.tail(min_length).values,
                            other_returns.tail(min_length).values
                        )[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                except Exception as e:
                    logger.debug(f"Error calculating correlation with {other_ticker}: {e}")
                    continue
            
            if correlations:
                features.append(np.mean(correlations))     # Mean correlation
                features.append(np.std(correlations))      # Correlation volatility
                features.append(np.max(correlations))      # Max correlation
                features.append(np.min(correlations))      # Min correlation
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            # Relative strength vs other assets
            primary_return = (primary_df['close'].iloc[-1] - primary_df['close'].iloc[-20]) / primary_df['close'].iloc[-20]
            other_returns = []
            
            for other_ticker in other_tickers[:5]:
                try:
                    other_df = multi_asset_data[other_ticker]
                    if len(other_df) >= 20:
                        other_return = (other_df['close'].iloc[-1] - other_df['close'].iloc[-20]) / other_df['close'].iloc[-20]
                        other_returns.append(other_return)
                except Exception as e:
                    continue
            
            if other_returns:
                relative_strength = primary_return - np.mean(other_returns)
                features.append(relative_strength)
            else:
                features.append(0.0)
            
            return features[:5]
            
        except Exception as e:
            logger.debug(f"Error generating cross-asset features: {e}")
            return [0.0] * 5
    
    # Technical indicator calculation methods
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> List[float]:
        """Calculate RSI indicator."""
        try:
            if len(df) < period + 1:
                return [50.0]  # Neutral RSI
            
            close_prices = df['close']
            price_changes = close_prices.diff()
            
            gains = price_changes.where(price_changes > 0, 0)
            losses = -price_changes.where(price_changes < 0, 0)
            
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
            
            rs = avg_gains / (avg_losses + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            
            # Return multiple values: current, trend, volatility
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
            rsi_trend = (rsi.iloc[-1] - rsi.iloc[-5]) if len(rsi) >= 5 else 0.0
            rsi_volatility = rsi.tail(10).std() if len(rsi) >= 10 else 0.0
            
            return [
                (current_rsi - 50) / 50,  # Normalized RSI
                rsi_trend / 50,           # RSI trend
                min(rsi_volatility / 10, 1.0)  # RSI volatility (capped)
            ]
            
        except Exception as e:
            logger.debug(f"Error calculating RSI: {e}")
            return [0.0, 0.0, 0.0]
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> List[float]:
        """Calculate MACD indicator."""
        try:
            if len(df) < slow + signal:
                return [0.0, 0.0, 0.0]
            
            close_prices = df['close']
            
            # Calculate EMAs
            ema_fast = close_prices.ewm(span=fast).mean()
            ema_slow = close_prices.ewm(span=slow).mean()
            
            # MACD line
            macd_line = ema_fast - ema_slow
            
            # Signal line
            signal_line = macd_line.ewm(span=signal).mean()
            
            # MACD histogram
            histogram = macd_line - signal_line
            
            # Normalize by current price for meaningful comparison
            current_price = close_prices.iloc[-1]
            
            return [
                (macd_line.iloc[-1] / current_price) * 100,    # MACD line normalized
                (signal_line.iloc[-1] / current_price) * 100,  # Signal line normalized
                (histogram.iloc[-1] / current_price) * 100     # Histogram normalized
            ]
            
        except Exception as e:
            logger.debug(f"Error calculating MACD: {e}")
            return [0.0, 0.0, 0.0]
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> List[float]:
        """Calculate Stochastic oscillator."""
        try:
            if len(df) < k_period:
                return [50.0, 50.0]
            
            high_prices = df['high'] if 'high' in df.columns else df['close']
            low_prices = df['low'] if 'low' in df.columns else df['close']
            close_prices = df['close']
            
            # %K calculation
            lowest_lows = low_prices.rolling(window=k_period).min()
            highest_highs = high_prices.rolling(window=k_period).max()
            
            k_percent = 100 * (close_prices - lowest_lows) / (highest_highs - lowest_lows + 1e-8)
            
            # %D calculation (SMA of %K)
            d_percent = k_percent.rolling(window=d_period).mean()
            
            current_k = k_percent.iloc[-1] if not pd.isna(k_percent.iloc[-1]) else 50.0
            current_d = d_percent.iloc[-1] if not pd.isna(d_percent.iloc[-1]) else 50.0
            
            return [
                (current_k - 50) / 50,  # Normalized %K
                (current_d - 50) / 50   # Normalized %D
            ]
            
        except Exception as e:
            logger.debug(f"Error calculating Stochastic: {e}")
            return [0.0, 0.0]
    
    def _calculate_sma_ratios(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50, 200]) -> List[float]:
        """Calculate SMA ratios (price/SMA - 1)."""
        try:
            close_prices = df['close']
            current_price = close_prices.iloc[-1]
            features = []
            
            for period in periods:
                if len(df) >= period:
                    sma = close_prices.rolling(window=period).mean().iloc[-1]
                    ratio = (current_price / sma - 1) if sma > 0 else 0.0
                    features.append(ratio)
                else:
                    features.append(0.0)
            
            return features
            
        except Exception as e:
            logger.debug(f"Error calculating SMA ratios: {e}")
            return [0.0] * len(periods)
    
    def _calculate_ema_ratios(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50]) -> List[float]:
        """Calculate EMA ratios (price/EMA - 1)."""
        try:
            close_prices = df['close']
            current_price = close_prices.iloc[-1]
            features = []
            
            for period in periods:
                if len(df) >= period:
                    ema = close_prices.ewm(span=period).mean().iloc[-1]
                    ratio = (current_price / ema - 1) if ema > 0 else 0.0
                    features.append(ratio)
                else:
                    features.append(0.0)
            
            return features
            
        except Exception as e:
            logger.debug(f"Error calculating EMA ratios: {e}")
            return [0.0] * len(periods)
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> List[float]:
        """Calculate Average Directional Index (ADX)."""
        try:
            if len(df) < period * 2:
                return [25.0]  # Neutral ADX
            
            high_prices = df['high'] if 'high' in df.columns else df['close']
            low_prices = df['low'] if 'low' in df.columns else df['close']
            close_prices = df['close']
            
            # True Range calculation
            high_low = high_prices - low_prices
            high_close = np.abs(high_prices - close_prices.shift(1))
            low_close = np.abs(low_prices - close_prices.shift(1))
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            
            # Directional Movement
            plus_dm = high_prices.diff()
            minus_dm = low_prices.diff() * -1
            
            plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
            minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
            
            # Smoothed values
            atr = pd.Series(true_range).rolling(window=period).mean()
            plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
            
            # ADX calculation
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
            adx = dx.rolling(window=period).mean()
            
            current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25.0
            
            return [current_adx / 100.0]  # Normalize to [0, 1]
            
        except Exception as e:
            logger.debug(f"Error calculating ADX: {e}")
            return [0.25]
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> List[float]:
        """Calculate Bollinger Bands indicators."""
        try:
            if len(df) < period:
                return [0.0, 0.0, 0.0]
            
            close_prices = df['close']
            
            # Middle band (SMA)
            middle_band = close_prices.rolling(window=period).mean()
            
            # Standard deviation
            std = close_prices.rolling(window=period).std()
            
            # Upper and lower bands
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            current_price = close_prices.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            current_middle = middle_band.iloc[-1]
            
            # Bollinger Band position
            bb_position = (current_price - current_lower) / (current_upper - current_lower + 1e-8)
            
            # Band width (normalized)
            band_width = (current_upper - current_lower) / current_middle if current_middle > 0 else 0
            
            # Price vs middle band
            price_to_middle = (current_price / current_middle - 1) if current_middle > 0 else 0
            
            return [
                bb_position - 0.5,      # BB position centered around 0
                band_width,             # Band width
                price_to_middle         # Price deviation from middle
            ]
            
        except Exception as e:
            logger.debug(f"Error calculating Bollinger Bands: {e}")
            return [0.0, 0.0, 0.0]
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> List[float]:
        """Calculate Average True Range (ATR)."""
        try:
            if len(df) < period:
                return [0.02]  # Default 2% ATR
            
            high_prices = df['high'] if 'high' in df.columns else df['close']
            low_prices = df['low'] if 'low' in df.columns else df['close']
            close_prices = df['close']
            
            # True Range calculation
            high_low = high_prices - low_prices
            high_close = np.abs(high_prices - close_prices.shift(1))
            low_close = np.abs(low_prices - close_prices.shift(1))
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            
            # ATR (smoothed true range)
            atr = pd.Series(true_range).rolling(window=period).mean()
            
            # Normalize by current price
            current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else close_prices.iloc[-1] * 0.02
            current_price = close_prices.iloc[-1]
            
            normalized_atr = current_atr / current_price if current_price > 0 else 0.02
            
            return [min(normalized_atr, 0.5)]  # Cap at 50%
            
        except Exception as e:
            logger.debug(f"Error calculating ATR: {e}")
            return [0.02]
    
    def _calculate_obv(self, df: pd.DataFrame) -> List[float]:
        """Calculate On-Balance Volume (OBV)."""
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return [0.0, 0.0]
            
            close_prices = df['close']
            volume = df['volume']
            
            # OBV calculation
            price_changes = close_prices.diff()
            volume_direction = np.where(price_changes > 0, volume, 
                              np.where(price_changes < 0, -volume, 0))
            
            obv = volume_direction.cumsum()
            
            # Normalize OBV by calculating trend and relative level
            if len(obv) >= 20:
                obv_sma = obv.rolling(window=20).mean()
                current_obv = obv.iloc[-1]
                current_sma = obv_sma.iloc[-1]
                
                # OBV relative to its moving average
                obv_relative = (current_obv - current_sma) / (abs(current_sma) + 1e-8)
                
                # OBV trend
                obv_trend = (obv.iloc[-1] - obv.iloc[-10]) / (abs(obv.iloc[-10]) + 1e-8) if len(obv) >= 10 else 0
                
                return [
                    max(-1.0, min(1.0, obv_relative)),
                    max(-1.0, min(1.0, obv_trend))
                ]
            else:
                return [0.0, 0.0]
                
        except Exception as e:
            logger.debug(f"Error calculating OBV: {e}")
            return [0.0, 0.0]
    
    def _calculate_volume_sma_ratio(self, df: pd.DataFrame, periods: List[int] = [10, 20, 50]) -> List[float]:
        """Calculate volume to SMA ratios."""
        try:
            if 'volume' not in df.columns:
                return [1.0] * len(periods)
            
            current_volume = df['volume'].iloc[-1]
            features = []
            
            for period in periods:
                if len(df) >= period:
                    volume_sma = df['volume'].rolling(window=period).mean().iloc[-1]
                    ratio = current_volume / (volume_sma + 1e-8)
                    # Log transform and cap to reasonable range
                    log_ratio = np.log(ratio + 1e-8)
                    features.append(max(-2.0, min(2.0, log_ratio)))
                else:
                    features.append(0.0)
            
            return features
            
        except Exception as e:
            logger.debug(f"Error calculating volume SMA ratios: {e}")
            return [0.0] * len(periods)
    
    def _calculate_pivot_points(self, df: pd.DataFrame, method: str = 'standard') -> List[float]:
        """Calculate pivot points and support/resistance levels."""
        try:
            if len(df) < 2:
                return [0.0, 0.0, 0.0, 0.0]
            
            # Use previous day's OHLC (or latest available)
            high = df['high'].iloc[-2] if 'high' in df.columns else df['close'].iloc[-2]
            low = df['low'].iloc[-2] if 'low' in df.columns else df['close'].iloc[-2]
            close = df['close'].iloc[-2]
            current_price = df['close'].iloc[-1]
            
            if method == 'standard':
                # Standard pivot points
                pivot = (high + low + close) / 3
                r1 = 2 * pivot - low
                s1 = 2 * pivot - high
                r2 = pivot + (high - low)
                s2 = pivot - (high - low)
            else:
                # Fibonacci pivot points
                pivot = (high + low + close) / 3
                r1 = pivot + 0.382 * (high - low)
                r2 = pivot + 0.618 * (high - low)
                s1 = pivot - 0.382 * (high - low)
                s2 = pivot - 0.618 * (high - low)
            
            # Calculate relative positions
            pivot_distance = (current_price - pivot) / pivot if pivot > 0 else 0
            r1_distance = (current_price - r1) / r1 if r1 > 0 else 0
            s1_distance = (current_price - s1) / s1 if s1 > 0 else 0
            
            # Determine position relative to key levels
            if current_price > r1:
                level_position = 1.0  # Above resistance
            elif current_price < s1:
                level_position = -1.0  # Below support
            else:
                level_position = 0.0  # Between support and resistance
            
            return [
                max(-0.5, min(0.5, pivot_distance)),    # Distance from pivot (capped)
                max(-0.5, min(0.5, r1_distance)),       # Distance from R1 (capped)
                max(-0.5, min(0.5, s1_distance)),       # Distance from S1 (capped)
                level_position                          # Position relative to key levels
            ]
            
        except Exception as e:
            logger.debug(f"Error calculating pivot points: {e}")
            return [0.0, 0.0, 0.0, 0.0]
    
    def _get_default_features(self) -> np.ndarray:
        """Get default features when calculation fails."""
        return np.zeros(50, dtype=np.float32)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached features are still valid."""
        if cache_key not in self.feature_cache:
            return False
        
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[cache_key]
        elapsed_minutes = (datetime.now() - cache_time).seconds / 60
        
        return elapsed_minutes < self.cache_ttl_minutes
    
    def _cache_features(self, cache_key: str, features: np.ndarray) -> None:
        """Cache calculated features."""
        self.feature_cache[cache_key] = features
        self.cache_timestamps[cache_key] = datetime.now()
        
        # Clean old cache entries
        self._clean_feature_cache()
    
    def _clean_feature_cache(self) -> None:
        """Remove expired cache entries."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            elapsed_minutes = (current_time - timestamp).seconds / 60
            if elapsed_minutes >= self.cache_ttl_minutes:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.feature_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all generated features."""
        names = []
        
        for indicator_name, indicator in self.indicators.items():
            # Add indicator-specific feature names based on what each function returns
            if indicator_name == 'rsi':
                names.extend([f"{indicator_name}_normalized", f"{indicator_name}_trend", f"{indicator_name}_volatility"])
            elif indicator_name == 'macd':
                names.extend([f"{indicator_name}_line", f"{indicator_name}_signal", f"{indicator_name}_histogram"])
            elif indicator_name == 'stochastic':
                names.extend([f"{indicator_name}_k", f"{indicator_name}_d"])
            elif indicator_name in ['sma_ratios', 'ema_ratios']:
                periods = indicator.parameters.get('periods', [5, 10, 20, 50])
                names.extend([f"{indicator_name}_{p}" for p in periods])
            elif indicator_name == 'adx':
                names.append(indicator_name)
            elif indicator_name == 'bollinger_bands':
                names.extend([f"{indicator_name}_position", f"{indicator_name}_width", f"{indicator_name}_deviation"])
            elif indicator_name == 'atr':
                names.append(indicator_name)
            elif indicator_name == 'obv':
                names.extend([f"{indicator_name}_relative", f"{indicator_name}_trend"])
            elif indicator_name == 'volume_sma_ratio':
                periods = indicator.parameters.get('periods', [10, 20, 50])
                names.extend([f"{indicator_name}_{p}" for p in periods])
            elif indicator_name == 'pivot_points':
                names.extend([f"{indicator_name}_pivot_dist", f"{indicator_name}_r1_dist", 
                            f"{indicator_name}_s1_dist", f"{indicator_name}_level_pos"])
        
        # Add multi-timeframe feature names
        if self.enable_multi_timeframe:
            for tf in ['short', 'medium', 'long']:
                names.extend([f"multi_tf_{tf}_trend", f"multi_tf_{tf}_volatility", f"multi_tf_{tf}_momentum"])
        
        # Add cross-asset feature names
        names.extend(['cross_asset_mean_corr', 'cross_asset_corr_vol', 'cross_asset_max_corr', 
                     'cross_asset_min_corr', 'cross_asset_relative_strength'])
        
        return names
    
    def get_regime_feature_weights(self, regime: MarketRegimeType) -> Dict[str, float]:
        """Get feature weights for a specific market regime."""
        weights = {}
        
        for indicator_name, indicator in self.indicators.items():
            weights[indicator_name] = indicator.regime_weights.get(regime, 0.5)
        
        return weights
    
    def update_indicator_performance(
        self,
        indicator_name: str,
        performance_score: float,
        regime: MarketRegimeType
    ) -> None:
        """Update performance tracking for an indicator in a specific regime."""
        if indicator_name not in self.indicator_performance_tracking:
            self.indicator_performance_tracking[indicator_name] = {}
        
        if regime not in self.indicator_performance_tracking[indicator_name]:
            self.indicator_performance_tracking[indicator_name][regime] = deque(maxlen=100)
        
        self.indicator_performance_tracking[indicator_name][regime].append(performance_score)
        
        # Adapt regime weights based on performance
        if self.enable_dynamic_parameters:
            self._adapt_regime_weights(indicator_name, regime)
    
    def _adapt_regime_weights(self, indicator_name: str, regime: MarketRegimeType) -> None:
        """Dynamically adapt regime weights based on performance."""
        try:
            if (indicator_name in self.indicator_performance_tracking and 
                regime in self.indicator_performance_tracking[indicator_name]):
                
                performance_history = self.indicator_performance_tracking[indicator_name][regime]
                
                if len(performance_history) >= 10:
                    recent_performance = np.mean(list(performance_history)[-10:])
                    
                    # Update regime weight based on performance
                    current_weight = self.indicators[indicator_name].regime_weights[regime]
                    
                    # Smoothed adaptation
                    adaptation_rate = 0.1
                    new_weight = current_weight + adaptation_rate * (recent_performance - 0.5)
                    new_weight = max(0.1, min(1.0, new_weight))  # Bounds: [0.1, 1.0]
                    
                    self.indicators[indicator_name].regime_weights[regime] = new_weight
                    
                    logger.debug(f"Adapted {indicator_name} weight for {regime.value}: {current_weight:.3f} → {new_weight:.3f}")
        
        except Exception as e:
            logger.debug(f"Error adapting regime weights for {indicator_name}: {e}")
    
    def get_generator_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the feature generator."""
        return {
            'total_indicators': len(self.indicators),
            'cache_size': len(self.feature_cache),
            'cache_hit_ratio': getattr(self, '_cache_hits', 0) / max(1, getattr(self, '_cache_requests', 1)),
            'performance_tracking_size': len(self.indicator_performance_tracking),
            'regime_cache_size': len(self.current_regime_cache),
            'settings': {
                'lookback_window': self.lookback_window,
                'regime_detection_window': self.regime_detection_window,
                'feature_selection_threshold': self.feature_selection_threshold,
                'enable_dynamic_parameters': self.enable_dynamic_parameters,
                'enable_multi_timeframe': self.enable_multi_timeframe
            }
        }