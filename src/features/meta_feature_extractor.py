"""Meta-feature extraction for neural meta-learning system."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import warnings
from loguru import logger
from collections import deque

from .technical_feature_generator import TechnicalFeatureGenerator
from .microstructure_features import MicrostructureFeatureExtractor
from .cross_asset_features import CrossAssetFeatureExtractor
from .volatility_regime_features import VolatilityRegimeFeatureExtractor
from .sentiment_features import SentimentFeatureExtractor
from .feature_selection_pipeline import FeatureSelectionPipeline
from ..config.settings import config

class MetaFeatureExtractor:
    """Extract meta-features for neural ensemble weight optimization."""
    
    def __init__(
        self,
        lookback_window: int = 60,
        performance_window: int = 20,
        volatility_window: int = 20,
        enable_feature_selection: bool = True
    ):
        """
        Initialize meta-feature extractor with Phase 6 feature generators.
        
        Args:
            lookback_window: Historical data window for feature calculation
            performance_window: Window for model performance tracking
            volatility_window: Window for volatility calculations
            enable_feature_selection: Whether to use automated feature selection
        """
        self.lookback_window = lookback_window
        self.performance_window = performance_window
        self.volatility_window = volatility_window
        self.enable_feature_selection = enable_feature_selection
        
        # Initialize Phase 6 feature generators
        self.technical_generator = TechnicalFeatureGenerator(
            lookback_window=lookback_window,
            volatility_window=volatility_window
        )
        
        self.microstructure_extractor = MicrostructureFeatureExtractor(
            tick_window=1000,
            volume_profile_bins=20
        )
        
        self.cross_asset_extractor = CrossAssetFeatureExtractor(
            correlation_window=60,
            spillover_lag=5
        )
        
        self.volatility_detector = VolatilityRegimeFeatureExtractor(
            volatility_windows=[5, 10, 20, 60],
            regime_lookback=252,
            gap_threshold=0.01,
            enable_garch_modeling=True,
            enable_regime_prediction=True,
            max_regimes=4
        )
        
        self.sentiment_analyzer = SentimentAnalyzer(
            sentiment_window=20,
            momentum_window=10
        )
        
        # Feature selector for redundancy removal and optimization
        if self.enable_feature_selection:
            self.feature_selector = FeatureSelector(
                method='hybrid',
                max_features=50,
                correlation_threshold=0.95
            )
        else:
            self.feature_selector = None
        
        # Model performance tracking (including multi-resolution)
        self.model_performance_history = {
            'timesfm': deque(maxlen=performance_window),
            'tsmamba': deque(maxlen=performance_window),
            'samba': deque(maxlen=performance_window),
            'tft': deque(maxlen=performance_window),
            'multi_resolution': deque(maxlen=performance_window),
            'ensemble': deque(maxlen=performance_window)
        }
        
        # Feature cache for efficiency
        self.feature_cache = {}
        self.cache_timestamps = {}
        
        # Phase 6 feature cache (expensive to compute)
        self.phase6_feature_cache = {}
        self.phase6_cache_timestamps = {}
        
        logger.info(f"Enhanced MetaFeatureExtractor initialized with lookback={lookback_window}, performance_window={performance_window}")
        logger.info(f"Phase 6 features enabled: technical, microstructure, cross-asset, volatility regimes, sentiment")
        logger.info(f"Feature selection enabled: {enable_feature_selection}")
    
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
            
            # 5. Technical Indicators (Simplified implementations)
            if len(recent_df) >= 14:
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
            else:
                # Default values for insufficient data
                features.extend([0.0, 0.0, 0.0])
            
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
    
    def extract_multi_resolution_features(
        self,
        price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        multi_resolution_predictions: Optional[Dict[str, Any]] = None,
        adaptive_weights: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Extract meta-features specific to multi-resolution prediction system.
        
        Args:
            price_data: Price data for feature calculation
            multi_resolution_predictions: Recent multi-resolution predictions
            adaptive_weights: Current adaptive resolution weights
            
        Returns:
            Array of multi-resolution meta-features
        """
        try:
            features = []
            
            # Resolution availability features
            resolution_availability = [0.0, 0.0, 0.0]  # minute, hourly, weekly
            if multi_resolution_predictions:
                if any('minute' in str(k) for k in multi_resolution_predictions.keys()):
                    resolution_availability[0] = 1.0
                if any('hourly' in str(k) for k in multi_resolution_predictions.keys()):
                    resolution_availability[1] = 1.0  
                if any('weekly' in str(k) for k in multi_resolution_predictions.keys()):
                    resolution_availability[2] = 1.0
                    
            features.extend(resolution_availability)
            
            # Adaptive weight features
            if adaptive_weights and 'final_weights' in adaptive_weights:
                weights = adaptive_weights['final_weights']
                features.extend([
                    weights.get('minute', 0.0),
                    weights.get('hourly', 0.0),
                    weights.get('weekly', 0.0)
                ])
                
                # Weight distribution entropy (diversity measure)
                weight_values = [weights.get(res, 0.0) for res in ['minute', 'hourly', 'weekly']]
                weight_values = [w for w in weight_values if w > 0]  # Remove zeros
                if weight_values:
                    weight_entropy = -sum(w * np.log(w + 1e-8) for w in weight_values)
                    features.append(weight_entropy)
                else:
                    features.append(0.0)
                    
                # Dominant resolution indicator
                dominant_res = max(weights.items(), key=lambda x: x[1])[0] if weights else 'none'
                dominant_indicators = [0.0, 0.0, 0.0]  # minute, hourly, weekly
                if dominant_res == 'minute':
                    dominant_indicators[0] = 1.0
                elif dominant_res == 'hourly':
                    dominant_indicators[1] = 1.0
                elif dominant_res == 'weekly':
                    dominant_indicators[2] = 1.0
                features.extend(dominant_indicators)
                
            else:
                # Default values when adaptive weights not available
                features.extend([1/3, 1/3, 1/3])  # Equal weights
                features.append(np.log(3))  # Maximum entropy for equal weights
                features.extend([0.0, 0.0, 0.0])  # No dominant resolution
            
            # Multi-resolution performance features
            if hasattr(self, 'model_performance_history') and 'multi_resolution' in self.model_performance_history:
                mr_performance = list(self.model_performance_history['multi_resolution'])
                if mr_performance:
                    features.extend([
                        np.mean(mr_performance),       # Average performance
                        np.std(mr_performance),        # Performance volatility
                        mr_performance[-1] if mr_performance else 0.5,  # Recent performance
                        max(mr_performance) - min(mr_performance),  # Performance range
                    ])
                else:
                    features.extend([0.5, 0.0, 0.5, 0.0])  # Default values
            else:
                features.extend([0.5, 0.0, 0.5, 0.0])  # Default values
            
            # Cross-resolution agreement features
            if multi_resolution_predictions:
                try:
                    # Extract predictions for same horizon from different resolutions
                    horizon_predictions = {}
                    for key, pred_data in multi_resolution_predictions.items():
                        if key.startswith('horizon_') and isinstance(pred_data, dict):
                            if 'prediction' in pred_data:
                                horizon_predictions[key] = pred_data['prediction']
                    
                    if len(horizon_predictions) > 1:
                        pred_values = list(horizon_predictions.values())
                        # Cross-resolution agreement (inverse of disagreement)
                        agreement = 1.0 / (1.0 + np.std(pred_values))
                        features.append(agreement)
                        
                        # Prediction range normalized
                        pred_range = (max(pred_values) - min(pred_values)) / (abs(np.mean(pred_values)) + 1e-6)
                        features.append(min(pred_range, 1.0))  # Cap at 1.0
                    else:
                        features.extend([1.0, 0.0])  # Perfect agreement, no range
                except Exception:
                    features.extend([1.0, 0.0])  # Default values
            else:
                features.extend([1.0, 0.0])  # Default values
            
            # Time-series characteristics that affect resolution choice
            if isinstance(price_data, pd.DataFrame) and len(price_data) >= 10:
                returns = price_data['close'].pct_change().dropna()
                
                # Autocorrelation at different lags (resolution relevance indicator)
                autocorrs = []
                for lag in [1, 5, 20]:  # Daily, weekly, monthly
                    if len(returns) > lag:
                        autocorr = returns.autocorr(lag=lag)
                        autocorrs.append(autocorr if not np.isnan(autocorr) else 0.0)
                    else:
                        autocorrs.append(0.0)
                features.extend(autocorrs)
                
                # Volatility clustering (affects resolution importance)
                if len(returns) >= 20:
                    vol_short = returns.rolling(5).std().std()  # Short-term vol volatility
                    vol_long = returns.rolling(20).std().std()  # Long-term vol volatility
                    vol_clustering = vol_short / (vol_long + 1e-6)
                    features.append(min(vol_clustering, 5.0))  # Cap extreme values
                else:
                    features.append(1.0)  # Neutral clustering
                    
                # Trend persistence (weekly resolution relevance)
                if len(returns) >= 10:
                    trend_consistency = abs(returns.rolling(10).mean().mean() / (returns.std() + 1e-6))
                    features.append(min(trend_consistency, 2.0))  # Cap extreme values
                else:
                    features.append(0.0)  # No trend
                    
            else:
                # Default values for insufficient data
                features.extend([0.0, 0.0, 0.0])  # No autocorrelations
                features.append(1.0)  # Neutral vol clustering
                features.append(0.0)  # No trend persistence
            
            # Market session feature (affects minute/hourly resolution importance)
            current_hour = datetime.now().hour
            session_features = [0.0, 0.0, 0.0]  # pre_market, regular, after_hours
            if 4 <= current_hour < 9:
                session_features[0] = 1.0  # Pre-market
            elif 9 <= current_hour < 16:
                session_features[1] = 1.0  # Regular hours
            else:
                session_features[2] = 1.0  # After hours
            features.extend(session_features)
            
            # Ensure consistent feature count (pad or truncate to 32 features)
            target_length = 32
            if len(features) < target_length:
                features.extend([0.0] * (target_length - len(features)))
            elif len(features) > target_length:
                features = features[:target_length]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting multi-resolution features: {e}")
            # Return default feature vector
            return np.zeros(32, dtype=np.float32)
    
    def extract_phase6_features(
        self,
        price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        ticker: str,
        multi_asset_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract Phase 6 dynamic features (technical, microstructure, cross-asset, volatility, sentiment).
        
        Args:
            price_data: Primary stock price data
            ticker: Stock ticker symbol
            multi_asset_data: Multi-asset data for cross-asset features
            
        Returns:
            Dictionary of feature arrays from each Phase 6 component
        """
        try:
            # Use caching for expensive Phase 6 features (5-minute TTL)
            cache_key = f"phase6_{ticker}_{hash(str(price_data.index[-1] if hasattr(price_data, 'index') else 'no_index'))}"
            current_time = datetime.now()
            
            if (cache_key in self.phase6_feature_cache and 
                cache_key in self.phase6_cache_timestamps and
                (current_time - self.phase6_cache_timestamps[cache_key]).seconds < 300):
                return self.phase6_feature_cache[cache_key]
            
            phase6_features = {}
            
            # Get primary DataFrame
            if isinstance(price_data, dict):
                primary_df = price_data.get(ticker, next(iter(price_data.values())))
            else:
                primary_df = price_data
            
            # 1. Technical Features
            try:
                technical_features = self.technical_generator.generate_features(
                    price_data=primary_df,
                    regime_state=None,  # Will be provided by ensemble
                    market_context={'ticker': ticker}
                )
                phase6_features['technical'] = technical_features
                logger.debug(f"Extracted {len(technical_features)} technical features for {ticker}")
            except Exception as e:
                logger.warning(f"Technical feature extraction failed for {ticker}: {e}")
                phase6_features['technical'] = np.zeros(20, dtype=np.float32)
            
            # 2. Microstructure Features (approximated from OHLCV)
            try:
                microstructure_features = self.microstructure_extractor.extract_features(
                    ohlcv_data=primary_df,
                    tick_data=None  # Using OHLCV approximation
                )
                phase6_features['microstructure'] = microstructure_features
                logger.debug(f"Extracted {len(microstructure_features)} microstructure features for {ticker}")
            except Exception as e:
                logger.warning(f"Microstructure feature extraction failed for {ticker}: {e}")
                phase6_features['microstructure'] = np.zeros(15, dtype=np.float32)
            
            # 3. Cross-Asset Features
            try:
                if multi_asset_data and len(multi_asset_data) > 1:
                    cross_asset_features = self.cross_asset_extractor.extract_features(
                        multi_asset_data=multi_asset_data,
                        target_ticker=ticker
                    )
                else:
                    cross_asset_features = self.cross_asset_extractor.extract_features(
                        multi_asset_data={ticker: primary_df},
                        target_ticker=ticker
                    )
                phase6_features['cross_asset'] = cross_asset_features
                logger.debug(f"Extracted {len(cross_asset_features)} cross-asset features for {ticker}")
            except Exception as e:
                logger.warning(f"Cross-asset feature extraction failed for {ticker}: {e}")
                phase6_features['cross_asset'] = np.zeros(12, dtype=np.float32)
            
            # 4. Volatility Regime Features
            try:
                volatility_features = self.volatility_detector.extract_features(
                    df=primary_df,
                    ticker=ticker,
                    include_regime_features=True,
                    include_gap_features=True,
                    include_volatility_modeling=True
                )
                phase6_features['volatility_regime'] = volatility_features
                logger.debug(f"Extracted {len(volatility_features)} volatility regime features for {ticker}")
            except Exception as e:
                logger.warning(f"Volatility regime feature extraction failed for {ticker}: {e}")
                phase6_features['volatility_regime'] = np.zeros(10, dtype=np.float32)
            
            # 5. Sentiment Features
            try:
                sentiment_features = self.sentiment_analyzer.extract_features(
                    ticker=ticker,
                    price_data=primary_df
                )
                phase6_features['sentiment'] = sentiment_features
                logger.debug(f"Extracted {len(sentiment_features)} sentiment features for {ticker}")
            except Exception as e:
                logger.warning(f"Sentiment feature extraction failed for {ticker}: {e}")
                phase6_features['sentiment'] = np.zeros(18, dtype=np.float32)
            
            # Cache results
            self.phase6_feature_cache[cache_key] = phase6_features
            self.phase6_cache_timestamps[cache_key] = current_time
            
            # Clean old cache entries
            self._clean_phase6_cache()
            
            return phase6_features
            
        except Exception as e:
            logger.error(f"Error extracting Phase 6 features for {ticker}: {e}")
            return self._get_default_phase6_features()
    
    def extract_combined_features(
        self,
        price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        regime_state: Optional[Any] = None,
        recent_predictions: Optional[Dict[str, List[float]]] = None,
        recent_actuals: Optional[List[float]] = None,
        horizon: int = 1,
        ticker: str = "UNKNOWN",
        multi_asset_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> np.ndarray:
        """
        Extract complete set of meta-features for neural meta-learning including Phase 6 features.
        
        Args:
            price_data: Price data (single stock or multi-stock)
            regime_state: Market regime state
            recent_predictions: Recent model predictions
            recent_actuals: Recent actual returns
            horizon: Prediction horizon
            ticker: Stock ticker symbol
            multi_asset_data: Multi-asset data for cross-asset features
            
        Returns:
            Combined feature array with Phase 6 enhancements
        """
        try:
            # Cache key for efficiency
            cache_key = f"{ticker}_{hash(str(price_data.index[-1] if hasattr(price_data, 'index') else 'no_index'))}_{horizon}"
            current_time = datetime.now()
            
            # Check cache (valid for 1 minute)
            if (cache_key in self.feature_cache and 
                cache_key in self.cache_timestamps and
                (current_time - self.cache_timestamps[cache_key]).seconds < 60):
                return self.feature_cache[cache_key]
            
            # Extract original features
            regime_features = self.extract_regime_features(price_data, regime_state)
            
            # Extract model performance features
            if recent_predictions and recent_actuals:
                performance_features = self.extract_model_performance_features(
                    recent_predictions, recent_actuals, horizon
                )
            else:
                performance_features = self._get_default_performance_features()
            
            # Extract multi-resolution features
            multi_resolution_features = self.extract_multi_resolution_features(price_data)
            
            # Extract Phase 6 features
            phase6_feature_dict = self.extract_phase6_features(
                price_data=price_data,
                ticker=ticker,
                multi_asset_data=multi_asset_data or (price_data if isinstance(price_data, dict) else None)
            )
            
            # Concatenate Phase 6 features
            phase6_features = np.concatenate([
                phase6_feature_dict['technical'],
                phase6_feature_dict['microstructure'],
                phase6_feature_dict['cross_asset'],
                phase6_feature_dict['volatility_regime'],
                phase6_feature_dict['sentiment']
            ])
            
            # Combine all features
            all_features = [
                regime_features,
                performance_features,
                multi_resolution_features,
                phase6_features
            ]
            
            combined_features = np.concatenate(all_features)
            
            # Apply feature selection if enabled and fitted
            if (self.feature_selector is not None and 
                hasattr(self.feature_selector, 'is_fitted_') and 
                self.feature_selector.is_fitted_):
                
                try:
                    # Reshape for sklearn compatibility
                    features_2d = combined_features.reshape(1, -1)
                    selected_features = self.feature_selector.transform(features_2d)
                    combined_features = selected_features.flatten()
                    
                except Exception as e:
                    logger.debug(f"Feature selection transform failed: {e}")
                    # Continue with original features
            
            # Normalize features to reasonable range
            combined_features = np.tanh(combined_features)  # Squash to [-1, 1]
            
            # Cache result
            self.feature_cache[cache_key] = combined_features
            self.cache_timestamps[cache_key] = current_time
            
            # Clean old cache entries
            self._clean_cache()
            
            logger.debug(f"Extracted {len(combined_features)} enhanced meta-features for {ticker} horizon {horizon}")
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Error extracting enhanced combined features: {e}")
            return self._get_default_enhanced_combined_features()
    
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
        multi_resolution_features = np.zeros(32, dtype=np.float32)  # Default multi-resolution features
        return np.concatenate([regime_features, performance_features, multi_resolution_features])
    
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
    
    def _clean_phase6_cache(self) -> None:
        """Clean old Phase 6 cache entries."""
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, timestamp in self.phase6_cache_timestamps.items():
            if (current_time - timestamp).seconds > 600:  # Remove entries older than 10 minutes
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.phase6_feature_cache.pop(key, None)
            self.phase6_cache_timestamps.pop(key, None)
    
    def _get_default_phase6_features(self) -> Dict[str, np.ndarray]:
        """Get default Phase 6 features when extraction fails."""
        return {
            'technical': np.zeros(20, dtype=np.float32),
            'microstructure': np.zeros(15, dtype=np.float32),
            'cross_asset': np.zeros(12, dtype=np.float32),
            'volatility_regime': np.zeros(10, dtype=np.float32),
            'sentiment': np.zeros(18, dtype=np.float32)
        }
    
    def _get_default_enhanced_combined_features(self) -> np.ndarray:
        """Get default enhanced combined features when extraction fails."""
        regime_features = self._get_default_regime_features()
        performance_features = self._get_default_performance_features()
        multi_resolution_features = np.zeros(32, dtype=np.float32)
        
        # Default Phase 6 features
        phase6_dict = self._get_default_phase6_features()
        phase6_features = np.concatenate([
            phase6_dict['technical'],
            phase6_dict['microstructure'],
            phase6_dict['cross_asset'],
            phase6_dict['volatility_regime'],
            phase6_dict['sentiment']
        ])
        
        return np.concatenate([regime_features, performance_features, multi_resolution_features, phase6_features])
    
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
    
    def fit_feature_selector(
        self,
        training_data: List[Dict[str, pd.DataFrame]],
        target_returns: List[np.ndarray],
        tickers: List[str]
    ) -> Dict[str, Any]:
        """
        Fit the feature selector on training data.
        
        Args:
            training_data: List of multi-asset price data dictionaries
            target_returns: List of target return arrays
            tickers: List of primary ticker symbols
            
        Returns:
            Feature selection fitting results
        """
        if not self.enable_feature_selection or self.feature_selector is None:
            return {'feature_selection_enabled': False}
        
        try:
            logger.info("Fitting feature selector on training data...")
            
            # Extract features for training samples
            X_features = []
            y_targets = []
            
            for data_dict, returns, ticker in zip(training_data, target_returns, tickers):
                try:
                    # Extract enhanced features
                    features = self.extract_combined_features(
                        price_data=data_dict,
                        ticker=ticker,
                        multi_asset_data=data_dict
                    )
                    
                    if len(returns) > 0:
                        X_features.append(features)
                        y_targets.append(returns[0] if len(returns) > 0 else 0.0)  # Use first return
                        
                except Exception as e:
                    logger.debug(f"Error extracting features for {ticker}: {e}")
                    continue
            
            if len(X_features) < 10:
                logger.warning(f"Insufficient training samples for feature selection: {len(X_features)}")
                return {'error': 'insufficient_training_data'}
            
            # Convert to arrays
            X = np.array(X_features)
            y = np.array(y_targets)
            
            # Fit feature selector
            self.feature_selector.fit(X, y)
            
            # Get selection results
            selected_features = self.feature_selector.get_selected_features()
            feature_scores = self.feature_selector.get_feature_scores() if hasattr(self.feature_selector, 'get_feature_scores') else None
            
            results = {
                'feature_selection_enabled': True,
                'n_training_samples': len(X_features),
                'n_original_features': X.shape[1],
                'n_selected_features': len(selected_features),
                'selected_feature_indices': selected_features.tolist() if hasattr(selected_features, 'tolist') else selected_features,
                'feature_reduction_ratio': len(selected_features) / X.shape[1]
            }
            
            if feature_scores is not None:
                results['feature_scores'] = feature_scores.tolist() if hasattr(feature_scores, 'tolist') else feature_scores
            
            logger.info(f"Feature selector fitted: {X.shape[1]} -> {len(selected_features)} features ({results['feature_reduction_ratio']:.2%} retention)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error fitting feature selector: {e}")
            return {'error': str(e)}
    
    def get_phase6_feature_summary(self) -> Dict[str, Any]:
        """Get summary of Phase 6 feature extraction status."""
        return {
            'technical_generator': {
                'enabled': self.technical_generator is not None,
                'lookback_window': self.lookback_window if hasattr(self, 'technical_generator') else None
            },
            'microstructure_extractor': {
                'enabled': self.microstructure_extractor is not None,
                'feature_count': 15
            },
            'cross_asset_extractor': {
                'enabled': self.cross_asset_extractor is not None,
                'correlation_window': 60 if hasattr(self, 'cross_asset_extractor') else None
            },
            'volatility_detector': {
                'enabled': self.volatility_detector is not None,
                'short_window': 10 if hasattr(self, 'volatility_detector') else None,
                'long_window': 60 if hasattr(self, 'volatility_detector') else None
            },
            'sentiment_analyzer': {
                'enabled': self.sentiment_analyzer is not None,
                'sentiment_window': 20 if hasattr(self, 'sentiment_analyzer') else None
            },
            'feature_selector': {
                'enabled': self.enable_feature_selection,
                'fitted': (hasattr(self.feature_selector, 'is_fitted_') and 
                          self.feature_selector.is_fitted_) if self.feature_selector else False
            },
            'cache_stats': {
                'phase6_cache_entries': len(self.phase6_feature_cache),
                'regular_cache_entries': len(self.feature_cache)
            }
        }