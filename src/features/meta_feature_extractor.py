"""Meta-feature extraction for neural meta-learning system."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import warnings
from loguru import logger
from collections import deque

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("TA-Lib not available. Using simplified technical indicators.")

from ..config.settings import config

class MetaFeatureExtractor:
    """Extract meta-features for neural ensemble weight optimization."""
    
    def __init__(
        self,
        lookback_window: int = 60,
        performance_window: int = 20,
        volatility_window: int = 20
    ):
        """
        Initialize meta-feature extractor.
        
        Args:
            lookback_window: Historical data window for feature calculation
            performance_window: Window for model performance tracking
            volatility_window: Window for volatility calculations
        """
        self.lookback_window = lookback_window
        self.performance_window = performance_window
        self.volatility_window = volatility_window
        
        # Model performance tracking
        self.model_performance_history = {
            'timesfm': deque(maxlen=performance_window),
            'tsmamba': deque(maxlen=performance_window),
            'samba': deque(maxlen=performance_window),
            'ensemble': deque(maxlen=performance_window)
        }
        
        # Feature cache for efficiency
        self.feature_cache = {}
        self.cache_timestamps = {}
        
        logger.info(f"MetaFeatureExtractor initialized with lookback={lookback_window}, performance_window={performance_window}")
    
    def extract_regime_features(
        self,
        price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        regime_state: Optional[Any] = None
    ) -> np.ndarray:
        """
        Extract market regime indicators.
        
        Args:
            price_data: OHLCV price data (single or multiple stocks)
            regime_state: Optional regime detector state
            
        Returns:
            Array of regime features
        """
        try:
            # Handle different input types
            if isinstance(price_data, dict):
                # Use first stock for market proxy, or find major stock
                primary_ticker = next(iter(price_data.keys()))
                df = price_data[primary_ticker]
                multi_stock_data = price_data
            else:
                df = price_data
                multi_stock_data = None
            
            if len(df) < self.lookback_window:
                # Return default features if insufficient data
                return self._get_default_regime_features()
            
            # Recent data for feature calculation
            recent_df = df.tail(self.lookback_window)
            features = []
            
            # 1. Trend Features
            close_prices = recent_df['close'].values
            returns = np.diff(np.log(close_prices))
            
            # Trend strength (linear regression slope)
            x = np.arange(len(close_prices))
            trend_slope = np.polyfit(x, close_prices, 1)[0] / close_prices[-1]  # Normalized
            features.append(trend_slope)
            
            # Trend consistency (R-squared of linear fit)
            p = np.poly1d(np.polyfit(x, close_prices, 1))
            ss_res = np.sum((close_prices - p(x)) ** 2)
            ss_tot = np.sum((close_prices - np.mean(close_prices)) ** 2)
            trend_consistency = 1 - (ss_res / (ss_tot + 1e-8))
            features.append(trend_consistency)
            
            # 2. Volatility Features
            volatility_short = np.std(returns[-10:]) * np.sqrt(252)  # 10-day volatility
            volatility_long = np.std(returns[-30:]) * np.sqrt(252)   # 30-day volatility
            volatility_ratio = volatility_short / (volatility_long + 1e-8)
            features.extend([volatility_short, volatility_long, volatility_ratio])
            
            # 3. Momentum Features
            momentum_5d = (close_prices[-1] - close_prices[-6]) / close_prices[-6] if len(close_prices) > 5 else 0
            momentum_10d = (close_prices[-1] - close_prices[-11]) / close_prices[-11] if len(close_prices) > 10 else 0
            momentum_20d = (close_prices[-1] - close_prices[-21]) / close_prices[-21] if len(close_prices) > 20 else 0
            features.extend([momentum_5d, momentum_10d, momentum_20d])
            
            # 4. Volume Features (if available)
            if 'volume' in recent_df.columns:
                volume_ma_10 = recent_df['volume'].rolling(10).mean().iloc[-1]
                volume_ma_30 = recent_df['volume'].rolling(30).mean().iloc[-1]
                volume_ratio = recent_df['volume'].iloc[-1] / (volume_ma_10 + 1e-8)
                volume_trend = recent_df['volume'].iloc[-1] / (volume_ma_30 + 1e-8)
                features.extend([volume_ratio, volume_trend])
            else:
                features.extend([1.0, 1.0])  # Neutral volume features
            
            # 5. Technical Indicators
            if TA_AVAILABLE and len(recent_df) >= 14:
                # RSI
                rsi = ta.momentum.RSIIndicator(recent_df['close'], window=14).rsi().iloc[-1]
                rsi_normalized = (rsi - 50) / 50  # Normalize to [-1, 1]
                
                # MACD
                macd_line = ta.trend.MACD(recent_df['close']).macd().iloc[-1]
                macd_signal = ta.trend.MACD(recent_df['close']).macd_signal().iloc[-1]
                macd_histogram = macd_line - macd_signal
                
                # Bollinger Bands position
                bb = ta.volatility.BollingerBands(recent_df['close'], window=20)
                bb_position = (recent_df['close'].iloc[-1] - bb.bollinger_mavg().iloc[-1]) / bb.bollinger_wband().iloc[-1]
                
                features.extend([rsi_normalized, macd_histogram, bb_position])
            else:
                # Simplified technical indicators
                # Simple RSI approximation
                price_changes = recent_df['close'].pct_change().dropna()
                gains = price_changes.where(price_changes > 0, 0)
                losses = -price_changes.where(price_changes < 0, 0)
                avg_gain = gains.rolling(14).mean().iloc[-1]
                avg_loss = losses.rolling(14).mean().iloc[-1]
                rs = avg_gain / (avg_loss + 1e-8)
                rsi_approx = 100 - (100 / (1 + rs))
                rsi_normalized = (rsi_approx - 50) / 50
                
                # Simple moving average convergence
                ma_short = recent_df['close'].rolling(12).mean().iloc[-1]
                ma_long = recent_df['close'].rolling(26).mean().iloc[-1]
                macd_approx = (ma_short - ma_long) / recent_df['close'].iloc[-1]
                
                # Bollinger position approximation
                ma_20 = recent_df['close'].rolling(20).mean().iloc[-1]
                std_20 = recent_df['close'].rolling(20).std().iloc[-1]
                bb_position_approx = (recent_df['close'].iloc[-1] - ma_20) / (2 * std_20 + 1e-8)
                
                features.extend([rsi_normalized, macd_approx, bb_position_approx])
            
            # 6. Cross-Asset Features (if multi-stock data available)
            if multi_stock_data and len(multi_stock_data) > 1:
                correlation_features = self._extract_correlation_features(multi_stock_data)
                features.extend(correlation_features)
            else:
                # Default correlation features
                features.extend([0.0, 0.0, 1.0])  # correlation_mean, correlation_std, relative_strength
            
            # 7. Regime State Features (if regime detector available)
            if regime_state is not None:
                regime_features = self._extract_regime_state_features(regime_state)
                features.extend(regime_features)
            else:
                # Default regime features
                features.extend([0.5, 0.5, 0.5])  # regime_confidence, regime_duration_norm, regime_stability
            
            # 8. Market Microstructure Approximations
            # Gap analysis
            if 'open' in recent_df.columns:
                overnight_returns = (recent_df['open'] - recent_df['close'].shift(1)) / recent_df['close'].shift(1)
                gap_magnitude = np.abs(overnight_returns.iloc[-1]) if not pd.isna(overnight_returns.iloc[-1]) else 0
                gap_frequency = np.sum(np.abs(overnight_returns.dropna()) > 0.01) / len(overnight_returns.dropna())
                features.extend([gap_magnitude, gap_frequency])
            else:
                features.extend([0.0, 0.0])
            
            # Intraday volatility (high-low range)
            if 'high' in recent_df.columns and 'low' in recent_df.columns:
                daily_ranges = (recent_df['high'] - recent_df['low']) / recent_df['close']
                avg_range = daily_ranges.rolling(10).mean().iloc[-1]
                range_expansion = daily_ranges.iloc[-1] / (avg_range + 1e-8)
                features.extend([avg_range, range_expansion])
            else:
                features.extend([0.02, 1.0])  # Default 2% range, no expansion
            
            # Ensure all features are finite
            features = np.array(features, dtype=np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Clip extreme values for stability
            features = np.clip(features, -5.0, 5.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting regime features: {e}")
            return self._get_default_regime_features()
    
    def _extract_correlation_features(self, multi_stock_data: Dict[str, pd.DataFrame]) -> List[float]:
        """Extract cross-asset correlation features."""
        try:
            tickers = list(multi_stock_data.keys())
            if len(tickers) < 2:
                return [0.0, 0.0, 1.0]
            
            # Get recent returns for all stocks
            returns_data = {}
            min_length = float('inf')
            
            for ticker, df in multi_stock_data.items():
                if len(df) >= 20:  # Need minimum data
                    returns = df['close'].pct_change().dropna()
                    if len(returns) > 0:
                        returns_data[ticker] = returns.tail(20)  # Last 20 days
                        min_length = min(min_length, len(returns_data[ticker]))
            
            if len(returns_data) < 2 or min_length < 5:
                return [0.0, 0.0, 1.0]
            
            # Align returns data
            aligned_returns = pd.DataFrame()
            for ticker, returns in returns_data.items():
                aligned_returns[ticker] = returns.tail(min_length).values
            
            # Calculate correlation matrix
            correlation_matrix = aligned_returns.corr()
            
            # Extract upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(correlation_matrix), k=1).astype(bool)
            correlations = correlation_matrix.values[mask]
            
            # Correlation features
            correlation_mean = np.mean(correlations)
            correlation_std = np.std(correlations)
            
            # Relative strength (performance vs others)
            primary_ticker = tickers[0]
            primary_return = aligned_returns[primary_ticker].iloc[-1] if len(aligned_returns) > 0 else 0
            other_returns = aligned_returns.drop(columns=primary_ticker).iloc[-1] if len(aligned_returns) > 0 else pd.Series([0])
            relative_strength = primary_return - other_returns.mean()
            
            return [
                float(correlation_mean),
                float(correlation_std),
                float(relative_strength)
            ]
            
        except Exception as e:
            logger.debug(f"Error extracting correlation features: {e}")
            return [0.0, 0.0, 1.0]
    
    def _extract_regime_state_features(self, regime_state: Any) -> List[float]:
        """Extract features from regime detector state."""
        try:
            features = []
            
            # Regime confidence
            confidence = getattr(regime_state, 'confidence', 0.5)
            features.append(confidence)
            
            # Regime duration (normalized)
            duration = getattr(regime_state, 'duration', 10)
            duration_norm = min(duration / 100.0, 1.0)  # Normalize to [0, 1]
            features.append(duration_norm)
            
            # Regime stability (inverse of recent changes)
            stability = getattr(regime_state, 'stability', 0.5)
            features.append(stability)
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting regime state features: {e}")
            return [0.5, 0.5, 0.5]
    
    def extract_model_performance_features(
        self,
        recent_predictions: Dict[str, List[float]],
        recent_actuals: List[float],
        horizon: int
    ) -> np.ndarray:
        """
        Extract model performance features.
        
        Args:
            recent_predictions: Recent predictions from each model
            recent_actuals: Recent actual returns
            horizon: Prediction horizon
            
        Returns:
            Array of performance features
        """
        try:
            features = []
            
            # Update performance history
            if len(recent_actuals) > 0:
                self._update_performance_history(recent_predictions, recent_actuals)
            
            # For each model, calculate performance metrics
            for model_name in ['timesfm', 'tsmamba', 'samba']:
                if model_name in recent_predictions and recent_predictions[model_name]:
                    # Recent accuracy
                    model_preds = recent_predictions[model_name][-self.performance_window:]
                    actual_vals = recent_actuals[-len(model_preds):] if recent_actuals else [0] * len(model_preds)
                    
                    if len(actual_vals) > 0:
                        mae = np.mean(np.abs(np.array(model_preds) - np.array(actual_vals)))
                        mse = np.mean((np.array(model_preds) - np.array(actual_vals)) ** 2)
                        
                        # Performance score (inverse of error)
                        performance_score = 1.0 / (1.0 + mae)
                        
                        # Prediction consistency (inverse of std)
                        pred_std = np.std(model_preds) if len(model_preds) > 1 else 0
                        consistency = 1.0 / (1.0 + pred_std)
                        
                        features.extend([performance_score, consistency])
                    else:
                        features.extend([0.5, 0.5])  # Neutral performance
                else:
                    features.extend([0.5, 0.5])  # Default performance
            
            # Ensemble disagreement features
            if len(recent_predictions) >= 2:
                # Calculate disagreement between models
                model_keys = list(recent_predictions.keys())[:3]  # Top 3 models
                last_predictions = []
                
                for key in model_keys:
                    if recent_predictions[key]:
                        last_predictions.append(recent_predictions[key][-1])
                
                if len(last_predictions) >= 2:
                    pred_array = np.array(last_predictions)
                    disagreement = np.std(pred_array)
                    max_deviation = np.max(np.abs(pred_array - np.mean(pred_array)))
                    
                    features.extend([disagreement, max_deviation])
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
            
            # Historical performance trends
            performance_trends = self._calculate_performance_trends()
            features.extend(performance_trends)
            
            # Ensure all features are finite
            features = np.array(features, dtype=np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting model performance features: {e}")
            return self._get_default_performance_features()
    
    def _update_performance_history(
        self,
        predictions: Dict[str, List[float]],
        actuals: List[float]
    ) -> None:
        """Update model performance tracking."""
        try:
            min_length = min(len(actuals), min(len(preds) for preds in predictions.values() if preds))
            
            for model_name, preds in predictions.items():
                if preds and model_name in self.model_performance_history:
                    # Calculate recent performance
                    model_preds = preds[-min_length:]
                    actual_vals = actuals[-min_length:]
                    
                    if len(model_preds) == len(actual_vals) and len(actual_vals) > 0:
                        mae = np.mean(np.abs(np.array(model_preds) - np.array(actual_vals)))
                        performance_score = 1.0 / (1.0 + mae)
                        
                        self.model_performance_history[model_name].append(performance_score)
                        
        except Exception as e:
            logger.debug(f"Error updating performance history: {e}")
    
    def _calculate_performance_trends(self) -> List[float]:
        """Calculate performance trends for each model."""
        trends = []
        
        for model_name in ['timesfm', 'tsmamba', 'samba']:
            history = self.model_performance_history[model_name]
            
            if len(history) >= 5:
                # Calculate trend (linear regression slope)
                x = np.arange(len(history))
                y = np.array(history)
                trend_slope = np.polyfit(x, y, 1)[0]
                
                # Recent vs historical performance
                recent_avg = np.mean(history[-5:])
                historical_avg = np.mean(history[:-5]) if len(history) > 5 else recent_avg
                relative_performance = recent_avg - historical_avg
                
                trends.extend([trend_slope, relative_performance])
            else:
                trends.extend([0.0, 0.0])  # Neutral trends
        
        return trends
    
    def extract_combined_features(
        self,
        price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        regime_state: Optional[Any] = None,
        recent_predictions: Optional[Dict[str, List[float]]] = None,
        recent_actuals: Optional[List[float]] = None,
        horizon: int = 1
    ) -> np.ndarray:
        """
        Extract complete set of meta-features for neural meta-learning.
        
        Args:
            price_data: Price data (single stock or multi-stock)
            regime_state: Market regime state
            recent_predictions: Recent model predictions
            recent_actuals: Recent actual returns
            horizon: Prediction horizon
            
        Returns:
            Combined feature array
        """
        try:
            # Cache key for efficiency
            cache_key = f"{hash(str(price_data.index[-1] if hasattr(price_data, 'index') else 'no_index'))}_{horizon}"
            current_time = datetime.now()
            
            # Check cache (valid for 1 minute)
            if (cache_key in self.feature_cache and 
                cache_key in self.cache_timestamps and
                (current_time - self.cache_timestamps[cache_key]).seconds < 60):
                return self.feature_cache[cache_key]
            
            # Extract regime features
            regime_features = self.extract_regime_features(price_data, regime_state)
            
            # Extract model performance features
            if recent_predictions and recent_actuals:
                performance_features = self.extract_model_performance_features(
                    recent_predictions, recent_actuals, horizon
                )
            else:
                performance_features = self._get_default_performance_features()
            
            # Combine all features
            combined_features = np.concatenate([regime_features, performance_features])
            
            # Normalize features to reasonable range
            combined_features = np.tanh(combined_features)  # Squash to [-1, 1]
            
            # Cache result
            self.feature_cache[cache_key] = combined_features
            self.cache_timestamps[cache_key] = current_time
            
            # Clean old cache entries
            self._clean_cache()
            
            logger.debug(f"Extracted {len(combined_features)} meta-features for horizon {horizon}")
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Error extracting combined features: {e}")
            return self._get_default_combined_features()
    
    def _get_default_regime_features(self) -> np.ndarray:
        """Get default regime features when extraction fails."""
        return np.array([
            0.0,  # trend_slope
            0.5,  # trend_consistency
            0.15, 0.15, 1.0,  # volatility features
            0.0, 0.0, 0.0,  # momentum features
            1.0, 1.0,  # volume features
            0.0, 0.0, 0.0,  # technical indicators
            0.0, 0.0, 1.0,  # correlation features
            0.5, 0.5, 0.5,  # regime state features
            0.0, 0.0,  # gap analysis
            0.02, 1.0   # intraday features
        ], dtype=np.float32)
    
    def _get_default_performance_features(self) -> np.ndarray:
        """Get default performance features when extraction fails."""
        return np.array([
            0.5, 0.5,  # timesfm performance, consistency
            0.5, 0.5,  # tsmamba performance, consistency  
            0.5, 0.5,  # samba performance, consistency
            0.0, 0.0,  # ensemble disagreement
            0.0, 0.0,  # timesfm trends
            0.0, 0.0,  # tsmamba trends
            0.0, 0.0   # samba trends
        ], dtype=np.float32)
    
    def _get_default_combined_features(self) -> np.ndarray:
        """Get default combined features when extraction fails."""
        regime_features = self._get_default_regime_features()
        performance_features = self._get_default_performance_features()
        return np.concatenate([regime_features, performance_features])
    
    def _clean_cache(self) -> None:
        """Clean old cache entries."""
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, timestamp in self.cache_timestamps.items():
            if (current_time - timestamp).seconds > 300:  # Remove entries older than 5 minutes
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.feature_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features."""
        regime_names = [
            'trend_slope', 'trend_consistency',
            'volatility_short', 'volatility_long', 'volatility_ratio',
            'momentum_5d', 'momentum_10d', 'momentum_20d',
            'volume_ratio', 'volume_trend',
            'rsi_normalized', 'macd_histogram', 'bb_position',
            'correlation_mean', 'correlation_std', 'relative_strength',
            'regime_confidence', 'regime_duration_norm', 'regime_stability',
            'gap_magnitude', 'gap_frequency',
            'avg_range', 'range_expansion'
        ]
        
        performance_names = [
            'timesfm_performance', 'timesfm_consistency',
            'tsmamba_performance', 'tsmamba_consistency',
            'samba_performance', 'samba_consistency',
            'ensemble_disagreement', 'max_deviation',
            'timesfm_trend', 'timesfm_relative_perf',
            'tsmamba_trend', 'tsmamba_relative_perf',
            'samba_trend', 'samba_relative_perf'
        ]
        
        return regime_names + performance_names
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get summary of feature importance and statistics."""
        feature_names = self.get_feature_names()
        
        # Calculate feature statistics from recent extractions
        if self.feature_cache:
            recent_features = list(self.feature_cache.values())
            feature_matrix = np.array(recent_features)
            
            feature_stats = {
                'feature_means': np.mean(feature_matrix, axis=0).tolist(),
                'feature_stds': np.std(feature_matrix, axis=0).tolist(),
                'feature_ranges': (np.max(feature_matrix, axis=0) - np.min(feature_matrix, axis=0)).tolist()
            }
            
            # Add feature names
            feature_stats['feature_names'] = feature_names
            feature_stats['n_samples'] = len(recent_features)
        else:
            feature_stats = {
                'feature_names': feature_names,
                'n_samples': 0
            }
        
        # Add performance history summary
        performance_summary = {}
        for model, history in self.model_performance_history.items():
            if history:
                performance_summary[model] = {
                    'mean_performance': np.mean(history),
                    'performance_trend': np.polyfit(range(len(history)), history, 1)[0] if len(history) > 2 else 0,
                    'n_samples': len(history)
                }
        
        feature_stats['model_performance_summary'] = performance_summary
        
        return feature_stats