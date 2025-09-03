"""Volatility regime detection and gap analysis features."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta, time
import warnings
from loguru import logger
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ..config.settings import config


class VolatilityRegime(Enum):
    """Volatility regime types."""
    LOW_VOLATILITY = "low_volatility"
    NORMAL_VOLATILITY = "normal_volatility"
    HIGH_VOLATILITY = "high_volatility"
    EXTREME_VOLATILITY = "extreme_volatility"
    VOLATILITY_SPIKE = "volatility_spike"
    VOLATILITY_COMPRESSION = "volatility_compression"


class GapType(Enum):
    """Types of price gaps."""
    OVERNIGHT_GAP = "overnight_gap"
    INTRADAY_GAP = "intraday_gap"
    EARNINGS_GAP = "earnings_gap"
    NEWS_GAP = "news_gap"
    WEEKEND_GAP = "weekend_gap"
    OPENING_GAP = "opening_gap"


class VolatilityModel(Enum):
    """Volatility modeling approaches."""
    REALIZED_VOLATILITY = "realized_volatility"
    GARCH_LIKE = "garch_like"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    RANGE_BASED = "range_based"


@dataclass
class VolatilityMetrics:
    """Volatility analysis results."""
    current_regime: VolatilityRegime
    regime_probability: float
    regime_duration: int
    transition_probability: float
    volatility_percentile: float
    persistence: float
    clustering: float
    asymmetry: float


@dataclass
class GapAnalysis:
    """Gap analysis results."""
    gap_type: GapType
    gap_size: float
    gap_direction: int  # 1 for up, -1 for down
    fill_probability: float
    time_to_fill: Optional[int]
    volume_confirmation: float
    momentum_strength: float


class VolatilityRegimeFeatureExtractor:
    """
    Advanced volatility regime detection and gap analysis feature extractor.
    
    Features:
    - Multi-timeframe volatility regime detection
    - GARCH-like volatility modeling
    - Gap detection and analysis
    - Volatility clustering and persistence
    - Regime transition predictions
    - Risk-adjusted feature scaling
    """
    
    def __init__(
        self,
        volatility_windows: List[int] = [5, 10, 20, 60],
        regime_lookback: int = 252,  # 1 year for regime detection
        gap_threshold: float = 0.01,  # 1% gap threshold
        enable_garch_modeling: bool = True,
        enable_regime_prediction: bool = True,
        max_regimes: int = 4
    ):
        """
        Initialize volatility regime feature extractor.
        
        Args:
            volatility_windows: Different windows for volatility calculation
            regime_lookback: Lookback period for regime detection
            gap_threshold: Minimum gap size to analyze
            enable_garch_modeling: Enable GARCH-like volatility modeling
            enable_regime_prediction: Enable regime transition prediction
            max_regimes: Maximum number of volatility regimes
        """
        self.volatility_windows = volatility_windows
        self.regime_lookback = regime_lookback
        self.gap_threshold = gap_threshold
        self.enable_garch_modeling = enable_garch_modeling
        self.enable_regime_prediction = enable_regime_prediction
        self.max_regimes = max_regimes
        
        # Feature caching
        self.feature_cache = {}
        self.cache_timestamps = {}
        self.cache_ttl_minutes = 20
        
        # Volatility regime tracking
        self.regime_history = deque(maxlen=100)
        self.volatility_history = defaultdict(lambda: deque(maxlen=500))
        
        # Gap analysis tracking
        self.gap_history = deque(maxlen=200)
        self.gap_statistics = defaultdict(list)
        
        # Volatility models
        self.volatility_models = {}
        self.regime_models = {}
        
        # Market hours for gap classification
        self.market_open = time(9, 30)  # 9:30 AM
        self.market_close = time(16, 0)  # 4:00 PM
        
        logger.info("VolatilityRegimeFeatureExtractor initialized")
    
    def extract_features(
        self,
        df: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        ticker: str,
        include_regime_features: bool = True,
        include_gap_features: bool = True,
        include_volatility_modeling: bool = True
    ) -> np.ndarray:
        """
        Extract comprehensive volatility regime and gap analysis features.
        
        Args:
            df: OHLCV price data
            ticker: Stock ticker symbol
            include_regime_features: Include volatility regime features
            include_gap_features: Include gap analysis features
            include_volatility_modeling: Include advanced volatility modeling
            
        Returns:
            Array of volatility regime and gap features
        """
        try:
            # Handle input data format
            if isinstance(df, dict):
                if ticker not in df:
                    logger.warning(f"Ticker {ticker} not found in multi-asset data")
                    return self._get_default_features()
                primary_df = df[ticker]
            else:
                primary_df = df
            
            # Minimum data requirement
            if len(primary_df) < min(self.volatility_windows) + 10:
                logger.warning(f"Insufficient data for {ticker}: {len(primary_df)} rows")
                return self._get_default_features()
            
            # Check cache
            cache_key = f"{ticker}_vol_{len(primary_df)}_{primary_df.index[-1]}"
            if self._is_cache_valid(cache_key):
                logger.debug(f"Using cached volatility features for {ticker}")
                return self.feature_cache[cache_key]
            
            # Extract different feature groups
            all_features = []
            
            # Volatility regime features
            if include_regime_features:
                regime_features = self._extract_volatility_regime_features(primary_df, ticker)
                all_features.extend(regime_features)
            
            # Gap analysis features
            if include_gap_features:
                gap_features = self._extract_gap_analysis_features(primary_df, ticker)
                all_features.extend(gap_features)
            
            # Advanced volatility modeling features
            if include_volatility_modeling:
                modeling_features = self._extract_volatility_modeling_features(primary_df, ticker)
                all_features.extend(modeling_features)
            
            # Regime transition features
            if self.enable_regime_prediction:
                transition_features = self._extract_regime_transition_features(primary_df, ticker)
                all_features.extend(transition_features)
            
            # Volatility risk features
            risk_features = self._extract_volatility_risk_features(primary_df, ticker)
            all_features.extend(risk_features)
            
            # Convert to numpy array and clean
            features_array = np.array(all_features, dtype=np.float32)
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
            features_array = np.clip(features_array, -10.0, 10.0)
            
            # Ensure minimum feature count
            min_features = 40
            if len(features_array) < min_features:
                features_array = np.pad(features_array, (0, min_features - len(features_array)))
            
            # Cache result
            self._cache_features(cache_key, features_array)
            
            logger.debug(f"Generated {len(features_array)} volatility regime features for {ticker}")
            
            return features_array
            
        except Exception as e:
            logger.error(f"Error extracting volatility regime features for {ticker}: {e}")
            return self._get_default_features()
    
    def _extract_volatility_regime_features(self, df: pd.DataFrame, ticker: str) -> List[float]:
        """Extract volatility regime classification features."""
        try:
            features = []
            
            # Calculate different volatility measures
            returns = df['close'].pct_change().dropna()
            
            if len(returns) < 20:
                return [0.0] * 15
            
            # 1. Realized volatility features
            for window in self.volatility_windows:
                if len(returns) >= window:
                    realized_vol = returns.rolling(window).std() * np.sqrt(252)  # Annualized
                    current_vol = realized_vol.iloc[-1] if not pd.isna(realized_vol.iloc[-1]) else 0.15
                    
                    # Volatility percentile (current vs historical)
                    if len(realized_vol.dropna()) > 10:
                        vol_percentile = stats.percentileofscore(realized_vol.dropna(), current_vol) / 100
                    else:
                        vol_percentile = 0.5
                    
                    features.append(min(current_vol, 2.0))  # Cap at 200% vol
                    features.append(vol_percentile)
                else:
                    features.extend([0.15, 0.5])  # Default vol and percentile
            
            # 2. Volatility regime classification using Gaussian Mixture
            regime_features = self._classify_volatility_regime(returns)
            features.extend(regime_features)  # Should return 4 values
            
            # 3. Volatility clustering measures
            clustering_features = self._calculate_volatility_clustering(returns)
            features.extend(clustering_features)  # Should return 3 values
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting volatility regime features: {e}")
            return [0.0] * 15
    
    def _classify_volatility_regime(self, returns: pd.Series) -> List[float]:
        """Classify current volatility regime using Gaussian Mixture Models."""
        try:
            if len(returns) < 60:
                return [0.0, 1.0, 0.0, 0.0]  # Default to normal regime
            
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=20).std().dropna()
            
            if len(rolling_vol) < 20:
                return [0.0, 1.0, 0.0, 0.0]
            
            # Fit Gaussian Mixture Model for regime detection
            vol_data = rolling_vol.values.reshape(-1, 1)
            
            # Use cached model if available and recent
            model_key = f"regime_model_{len(returns)}"
            
            if model_key not in self.regime_models or len(rolling_vol) % 50 == 0:  # Refit every 50 observations
                try:
                    # Fit with optimal number of components (up to max_regimes)
                    best_model = None
                    best_bic = float('inf')
                    
                    for n_components in range(2, min(self.max_regimes + 1, len(set(np.round(vol_data.flatten(), 4))) + 1)):
                        try:
                            model = GaussianMixture(n_components=n_components, random_state=42)
                            model.fit(vol_data)
                            bic = model.bic(vol_data)
                            
                            if bic < best_bic:
                                best_bic = bic
                                best_model = model
                        except Exception:
                            continue
                    
                    if best_model is not None:
                        self.regime_models[model_key] = best_model
                    else:
                        # Fallback to simple quantile-based classification
                        return self._simple_regime_classification(rolling_vol)
                        
                except Exception:
                    # Fallback to simple classification
                    return self._simple_regime_classification(rolling_vol)
            
            model = self.regime_models.get(model_key)
            if model is None:
                return self._simple_regime_classification(rolling_vol)
            
            # Predict current regime
            current_vol = rolling_vol.iloc[-1]
            regime_probs = model.predict_proba([[current_vol]])[0]
            
            # Map to our regime types (pad or truncate to 4 regimes)
            regime_features = np.zeros(4)
            for i, prob in enumerate(regime_probs[:4]):
                regime_features[i] = prob
            
            # If fewer than 4 components, put remaining probability in "normal" regime
            if len(regime_probs) < 4:
                regime_features[1] = max(regime_features[1], 1 - np.sum(regime_features))
            
            return regime_features.tolist()
            
        except Exception as e:
            logger.debug(f"Error in volatility regime classification: {e}")
            return self._simple_regime_classification(returns.rolling(window=20).std().dropna())
    
    def _simple_regime_classification(self, rolling_vol: pd.Series) -> List[float]:
        """Simple quantile-based regime classification fallback."""
        try:
            if len(rolling_vol) < 10:
                return [0.0, 1.0, 0.0, 0.0]
            
            current_vol = rolling_vol.iloc[-1]
            
            # Define regime thresholds using quantiles
            q25 = rolling_vol.quantile(0.25)
            q50 = rolling_vol.quantile(0.50)
            q75 = rolling_vol.quantile(0.75)
            q95 = rolling_vol.quantile(0.95)
            
            # Classify current volatility
            regime_probs = [0.0, 0.0, 0.0, 0.0]  # low, normal, high, extreme
            
            if current_vol <= q25:
                regime_probs[0] = 1.0  # Low volatility
            elif current_vol <= q50:
                regime_probs[1] = 1.0  # Normal volatility
            elif current_vol <= q75:
                regime_probs[2] = 1.0  # High volatility
            else:
                regime_probs[3] = 1.0  # Extreme volatility
            
            return regime_probs
            
        except Exception:
            return [0.0, 1.0, 0.0, 0.0]  # Default to normal
    
    def _calculate_volatility_clustering(self, returns: pd.Series) -> List[float]:
        """Calculate volatility clustering measures."""
        try:
            if len(returns) < 30:
                return [0.5, 0.0, 0.5]
            
            # Calculate absolute returns (proxy for volatility)
            abs_returns = returns.abs()
            
            # 1. Volatility persistence (autocorrelation of absolute returns)
            if len(abs_returns) > 10:
                persistence = abs_returns.autocorr(lag=1)
                persistence = persistence if not np.isnan(persistence) else 0.5
            else:
                persistence = 0.5
            
            # 2. Volatility of volatility (heteroskedasticity measure)
            rolling_vol = abs_returns.rolling(window=10).std()
            vol_of_vol = rolling_vol.std() if len(rolling_vol.dropna()) > 5 else 0.0
            
            # 3. ARCH effect test (simplified)
            arch_effect = self._calculate_arch_effect(returns)
            
            return [
                max(-1.0, min(1.0, persistence)),
                min(vol_of_vol, 1.0),
                max(0.0, min(1.0, arch_effect))
            ]
            
        except Exception as e:
            logger.debug(f"Error calculating volatility clustering: {e}")
            return [0.5, 0.0, 0.5]
    
    def _calculate_arch_effect(self, returns: pd.Series) -> float:
        """Calculate simplified ARCH effect measure."""
        try:
            if len(returns) < 20:
                return 0.5
            
            # Squared returns as proxy for volatility
            squared_returns = returns ** 2
            
            # Test for ARCH effects using autocorrelation of squared returns
            if len(squared_returns) > 10:
                arch_corr = squared_returns.autocorr(lag=1)
                arch_effect = abs(arch_corr) if not np.isnan(arch_corr) else 0.0
                return arch_effect
            
            return 0.0
            
        except Exception:
            return 0.5
    
    def _extract_gap_analysis_features(self, df: pd.DataFrame, ticker: str) -> List[float]:
        """Extract price gap analysis features."""
        try:
            features = []
            
            if len(df) < 10:
                return [0.0] * 12
            
            # 1. Detect and classify gaps
            gaps = self._detect_price_gaps(df)
            recent_gaps = gaps[-20:] if gaps else []  # Last 20 gaps
            
            # Gap frequency features
            if len(df) >= 60:  # Need reasonable history
                gap_frequency = len([g for g in recent_gaps if abs(g.gap_size) >= self.gap_threshold]) / min(60, len(df))
                up_gap_ratio = len([g for g in recent_gaps if g.gap_direction == 1]) / max(1, len(recent_gaps))
                avg_gap_size = np.mean([abs(g.gap_size) for g in recent_gaps]) if recent_gaps else 0.0
            else:
                gap_frequency = 0.0
                up_gap_ratio = 0.5
                avg_gap_size = 0.0
            
            features.extend([
                min(gap_frequency, 1.0),
                up_gap_ratio,
                min(avg_gap_size, 0.2)  # Cap at 20%
            ])
            
            # 2. Recent gap analysis
            if recent_gaps:
                latest_gap = recent_gaps[-1]
                
                # Days since last significant gap
                days_since_gap = 1  # Simplified - in real implementation would calculate actual days
                gap_momentum = min(abs(latest_gap.gap_size) * 10, 1.0)
                gap_fill_prob = latest_gap.fill_probability
                
                features.extend([
                    min(days_since_gap / 30.0, 1.0),  # Normalize to months
                    gap_momentum,
                    gap_fill_prob
                ])
            else:
                features.extend([1.0, 0.0, 0.5])  # No recent gaps
            
            # 3. Gap pattern features
            gap_pattern_features = self._analyze_gap_patterns(recent_gaps)
            features.extend(gap_pattern_features)  # Should return 6 values
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting gap analysis features: {e}")
            return [0.0] * 12
    
    def _detect_price_gaps(self, df: pd.DataFrame) -> List[GapAnalysis]:
        """Detect and analyze price gaps."""
        try:
            gaps = []
            
            if len(df) < 2:
                return gaps
            
            for i in range(1, len(df)):
                prev_close = df['close'].iloc[i - 1]
                current_open = df.get('open', df['close']).iloc[i] if 'open' in df.columns else df['close'].iloc[i]
                current_close = df['close'].iloc[i]
                current_volume = df.get('volume', 1000000).iloc[i] if 'volume' in df.columns else 1000000
                
                # Calculate gap
                gap_size = (current_open - prev_close) / prev_close
                
                if abs(gap_size) >= self.gap_threshold:
                    # Classify gap type (simplified)
                    gap_type = self._classify_gap_type(df.index[i], gap_size)
                    
                    # Calculate fill probability (simplified heuristic)
                    fill_probability = self._estimate_gap_fill_probability(gap_size, current_volume, gap_type)
                    
                    # Calculate momentum strength
                    momentum_strength = abs((current_close - current_open) / current_open) if current_open > 0 else 0.0
                    
                    # Volume confirmation
                    avg_volume = df['volume'].rolling(10).mean().iloc[i] if 'volume' in df.columns and len(df) >= i + 1 else current_volume
                    volume_confirmation = min(current_volume / (avg_volume + 1e-8), 3.0)
                    
                    gap_analysis = GapAnalysis(
                        gap_type=gap_type,
                        gap_size=gap_size,
                        gap_direction=1 if gap_size > 0 else -1,
                        fill_probability=fill_probability,
                        time_to_fill=None,  # Would be calculated with more sophisticated analysis
                        volume_confirmation=volume_confirmation,
                        momentum_strength=momentum_strength
                    )
                    
                    gaps.append(gap_analysis)
            
            return gaps
            
        except Exception as e:
            logger.debug(f"Error detecting price gaps: {e}")
            return []
    
    def _classify_gap_type(self, timestamp: pd.Timestamp, gap_size: float) -> GapType:
        """Classify the type of price gap."""
        try:
            # Simplified gap classification
            if hasattr(timestamp, 'time'):
                gap_time = timestamp.time()
                
                # Opening gap (market open)
                if self.market_open <= gap_time <= time(10, 0):
                    if timestamp.weekday() == 0:  # Monday
                        return GapType.WEEKEND_GAP
                    else:
                        return GapType.OVERNIGHT_GAP
                
                # Intraday gap (during market hours)
                elif self.market_open < gap_time < self.market_close:
                    return GapType.INTRADAY_GAP
                
                # After hours
                else:
                    return GapType.OVERNIGHT_GAP
            
            # Default classification based on gap size
            if abs(gap_size) > 0.05:  # 5% gap
                return GapType.EARNINGS_GAP  # Likely earnings or news
            else:
                return GapType.OPENING_GAP
                
        except Exception:
            return GapType.OPENING_GAP
    
    def _estimate_gap_fill_probability(
        self, 
        gap_size: float, 
        volume: float, 
        gap_type: GapType
    ) -> float:
        """Estimate probability of gap being filled."""
        try:
            base_probability = 0.7  # Base assumption that gaps tend to fill
            
            # Adjust based on gap size (larger gaps less likely to fill quickly)
            size_adjustment = max(0.0, 1.0 - abs(gap_size) * 5)  # Reduce by 5x gap size
            
            # Adjust based on gap type
            type_adjustments = {
                GapType.OVERNIGHT_GAP: 0.0,
                GapType.INTRADAY_GAP: -0.1,  # Less likely to fill same day
                GapType.EARNINGS_GAP: -0.3,  # Earnings gaps often don't fill
                GapType.NEWS_GAP: -0.2,
                GapType.WEEKEND_GAP: 0.1,   # Weekend gaps often fill
                GapType.OPENING_GAP: 0.0
            }
            
            type_adj = type_adjustments.get(gap_type, 0.0)
            
            # Volume confirmation (higher volume = more likely to sustain)
            volume_factor = max(0.5, min(1.5, np.log(volume + 1) / 15))  # Normalize volume impact
            
            fill_probability = base_probability * size_adjustment + type_adj
            fill_probability *= volume_factor
            
            return max(0.1, min(0.95, fill_probability))
            
        except Exception:
            return 0.5  # Neutral probability
    
    def _analyze_gap_patterns(self, gaps: List[GapAnalysis]) -> List[float]:
        """Analyze patterns in recent gaps."""
        try:
            if not gaps:
                return [0.0] * 6
            
            # Pattern features
            features = []
            
            # 1. Gap size trend (are gaps getting larger/smaller?)
            gap_sizes = [abs(g.gap_size) for g in gaps]
            if len(gap_sizes) >= 3:
                # Simple trend: compare first half vs second half
                mid = len(gap_sizes) // 2
                early_avg = np.mean(gap_sizes[:mid]) if mid > 0 else 0
                recent_avg = np.mean(gap_sizes[mid:])
                size_trend = (recent_avg - early_avg) / (early_avg + 1e-8) if early_avg > 0 else 0
                features.append(max(-1.0, min(1.0, size_trend)))
            else:
                features.append(0.0)
            
            # 2. Gap direction consistency
            directions = [g.gap_direction for g in gaps]
            if directions:
                direction_consistency = abs(np.mean(directions))  # How consistently up or down
                features.append(direction_consistency)
            else:
                features.append(0.0)
            
            # 3. Average fill probability
            fill_probs = [g.fill_probability for g in gaps]
            avg_fill_prob = np.mean(fill_probs) if fill_probs else 0.5
            features.append(avg_fill_prob)
            
            # 4. Volume confirmation strength
            volume_confirmations = [g.volume_confirmation for g in gaps]
            avg_volume_confirm = np.mean(volume_confirmations) if volume_confirmations else 1.0
            features.append(min(avg_volume_confirm / 2.0, 1.0))  # Normalize
            
            # 5. Gap momentum patterns
            momentum_strengths = [g.momentum_strength for g in gaps]
            avg_momentum = np.mean(momentum_strengths) if momentum_strengths else 0.0
            features.append(min(avg_momentum * 10, 1.0))  # Scale and cap
            
            # 6. Gap frequency acceleration (are gaps becoming more frequent?)
            if len(gaps) >= 6:
                # Compare recent frequency to historical
                recent_count = len(gaps[-3:])  # Last 3 gaps
                historical_count = len(gaps[:-3])  # All but last 3
                freq_acceleration = (recent_count * 2) - (historical_count / max(1, len(gaps) - 3))
                features.append(max(-1.0, min(1.0, freq_acceleration)))
            else:
                features.append(0.0)
            
            return features
            
        except Exception as e:
            logger.debug(f"Error analyzing gap patterns: {e}")
            return [0.0] * 6
    
    def _extract_volatility_modeling_features(self, df: pd.DataFrame, ticker: str) -> List[float]:
        """Extract advanced volatility modeling features."""
        try:
            features = []
            returns = df['close'].pct_change().dropna()
            
            if len(returns) < 30:
                return [0.0] * 10
            
            # 1. GARCH-like features
            if self.enable_garch_modeling:
                garch_features = self._calculate_garch_like_features(returns)
                features.extend(garch_features)  # Should return 4 values
            else:
                features.extend([0.0] * 4)
            
            # 2. Range-based volatility features
            range_vol_features = self._calculate_range_based_volatility(df)
            features.extend(range_vol_features)  # Should return 3 values
            
            # 3. Volatility surface features
            vol_surface_features = self._calculate_volatility_surface_features(returns)
            features.extend(vol_surface_features)  # Should return 3 values
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting volatility modeling features: {e}")
            return [0.0] * 10
    
    def _calculate_garch_like_features(self, returns: pd.Series) -> List[float]:
        """Calculate GARCH-like volatility features."""
        try:
            if len(returns) < 20:
                return [0.15, 0.8, 0.1, 0.0]  # Default values
            
            # Simplified GARCH(1,1) approximation
            # σ²(t) = ω + α*ε²(t-1) + β*σ²(t-1)
            
            # Calculate squared returns (innovations)
            squared_returns = returns ** 2
            
            # Exponentially weighted moving average for volatility
            lambda_decay = 0.94  # Standard RiskMetrics lambda
            ewm_vol_sq = squared_returns.ewm(alpha=1-lambda_decay).mean()
            ewm_vol = np.sqrt(ewm_vol_sq) * np.sqrt(252)  # Annualized
            
            # Current volatility estimate
            current_vol = ewm_vol.iloc[-1] if not pd.isna(ewm_vol.iloc[-1]) else 0.15
            
            # Volatility mean reversion speed (approximated)
            vol_changes = ewm_vol.pct_change().dropna()
            if len(vol_changes) > 10:
                mean_reversion = -vol_changes.autocorr(lag=1)  # Negative autocorr = mean reversion
                mean_reversion = mean_reversion if not np.isnan(mean_reversion) else 0.0
            else:
                mean_reversion = 0.0
            
            # ARCH effect strength
            arch_strength = abs(squared_returns.autocorr(lag=1)) if len(squared_returns) > 10 else 0.0
            arch_strength = arch_strength if not np.isnan(arch_strength) else 0.0
            
            # Volatility innovation (recent surprise)
            if len(returns) >= 2:
                recent_return = returns.iloc[-1]
                expected_vol = ewm_vol.iloc[-2] / np.sqrt(252)  # Daily vol
                vol_innovation = abs(recent_return) / (expected_vol + 1e-8) - 1
                vol_innovation = max(-2.0, min(2.0, vol_innovation))
            else:
                vol_innovation = 0.0
            
            return [
                min(current_vol, 2.0),  # Current volatility (capped)
                max(0.0, min(1.0, mean_reversion)),  # Mean reversion speed
                max(0.0, min(1.0, arch_strength)),   # ARCH effect strength
                vol_innovation  # Volatility innovation
            ]
            
        except Exception as e:
            logger.debug(f"Error calculating GARCH-like features: {e}")
            return [0.15, 0.8, 0.1, 0.0]
    
    def _calculate_range_based_volatility(self, df: pd.DataFrame) -> List[float]:
        """Calculate range-based volatility measures."""
        try:
            if len(df) < 10:
                return [0.15, 0.15, 1.0]
            
            # Parkinson volatility (using high-low range)
            if 'high' in df.columns and 'low' in df.columns:
                high_low_ratio = df['high'] / df['low']
                # Parkinson estimator: (1/4ln2) * ln(H/L)²
                parkinson_vol = np.sqrt(0.3606 * (np.log(high_low_ratio) ** 2).rolling(20).mean()) * np.sqrt(252)
                current_parkinson = parkinson_vol.iloc[-1] if not pd.isna(parkinson_vol.iloc[-1]) else 0.15
            else:
                current_parkinson = 0.15
            
            # Garman-Klass volatility (using OHLC)
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # Simplified GK estimator
                ln_hl = np.log(df['high'] / df['low'])
                ln_co = np.log(df['close'] / df['open'])
                gk_component = 0.5 * (ln_hl ** 2) - (2 * np.log(2) - 1) * (ln_co ** 2)
                gk_vol = np.sqrt(gk_component.rolling(20).mean()) * np.sqrt(252)
                current_gk = gk_vol.iloc[-1] if not pd.isna(gk_vol.iloc[-1]) else 0.15
            else:
                current_gk = current_parkinson
            
            # Close-to-close volatility
            cc_returns = df['close'].pct_change().dropna()
            cc_vol = cc_returns.rolling(20).std() * np.sqrt(252)
            current_cc = cc_vol.iloc[-1] if not pd.isna(cc_vol.iloc[-1]) else 0.15
            
            # Range efficiency (how much of price range is captured by close-to-close)
            if current_parkinson > 0:
                range_efficiency = current_cc / current_parkinson
                range_efficiency = max(0.1, min(2.0, range_efficiency))
            else:
                range_efficiency = 1.0
            
            return [
                min(current_parkinson, 2.0),
                min(current_gk, 2.0),
                range_efficiency
            ]
            
        except Exception as e:
            logger.debug(f"Error calculating range-based volatility: {e}")
            return [0.15, 0.15, 1.0]
    
    def _calculate_volatility_surface_features(self, returns: pd.Series) -> List[float]:
        """Calculate volatility surface-like features."""
        try:
            if len(returns) < 60:
                return [0.0, 0.0, 0.0]
            
            # Volatility term structure approximation
            short_vol = returns.rolling(5).std() * np.sqrt(252)   # 5-day vol
            medium_vol = returns.rolling(20).std() * np.sqrt(252) # 20-day vol
            long_vol = returns.rolling(60).std() * np.sqrt(252)   # 60-day vol
            
            # Term structure slope
            if not pd.isna(long_vol.iloc[-1]) and not pd.isna(short_vol.iloc[-1]) and long_vol.iloc[-1] > 0:
                term_structure_slope = (long_vol.iloc[-1] - short_vol.iloc[-1]) / long_vol.iloc[-1]
                term_structure_slope = max(-1.0, min(1.0, term_structure_slope))
            else:
                term_structure_slope = 0.0
            
            # Volatility skew approximation (using return skewness)
            recent_returns = returns.tail(60)
            if len(recent_returns) >= 30:
                vol_skew = recent_returns.skew()
                vol_skew = vol_skew if not np.isnan(vol_skew) else 0.0
                vol_skew = max(-3.0, min(3.0, vol_skew))
            else:
                vol_skew = 0.0
            
            # Volatility smile curvature (using return kurtosis)
            if len(recent_returns) >= 30:
                vol_kurtosis = recent_returns.kurtosis()
                vol_kurtosis = vol_kurtosis if not np.isnan(vol_kurtosis) else 3.0
                # Convert excess kurtosis to smile curvature indicator
                smile_curvature = max(0.0, (vol_kurtosis - 3.0) / 10.0)  # Normalize excess kurtosis
                smile_curvature = min(smile_curvature, 1.0)
            else:
                smile_curvature = 0.0
            
            return [
                term_structure_slope,
                max(-2.0, min(2.0, vol_skew / 2.0)),  # Normalized skew
                smile_curvature
            ]
            
        except Exception as e:
            logger.debug(f"Error calculating volatility surface features: {e}")
            return [0.0, 0.0, 0.0]
    
    def _extract_regime_transition_features(self, df: pd.DataFrame, ticker: str) -> List[float]:
        """Extract regime transition prediction features."""
        try:
            if not self.enable_regime_prediction or len(df) < 50:
                return [0.0] * 8
            
            features = []
            returns = df['close'].pct_change().dropna()
            
            # 1. Regime transition probability
            transition_prob = self._calculate_regime_transition_probability(returns)
            features.append(transition_prob)
            
            # 2. Regime stability measures
            stability_features = self._calculate_regime_stability(returns)
            features.extend(stability_features)  # Should return 3 values
            
            # 3. Early warning indicators
            warning_features = self._calculate_regime_warning_indicators(returns)
            features.extend(warning_features)  # Should return 4 values
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting regime transition features: {e}")
            return [0.0] * 8
    
    def _calculate_regime_transition_probability(self, returns: pd.Series) -> float:
        """Calculate probability of regime transition."""
        try:
            if len(returns) < 30:
                return 0.1  # Low transition probability by default
            
            # Use recent volatility changes as indicator
            rolling_vol = returns.rolling(10).std()
            vol_changes = rolling_vol.pct_change().abs()
            
            # High volatility changes indicate potential regime transition
            recent_vol_change = vol_changes.tail(5).mean()
            historical_vol_change = vol_changes.quantile(0.8)  # 80th percentile
            
            if historical_vol_change > 0:
                transition_signal = recent_vol_change / historical_vol_change
                transition_prob = min(transition_signal / 2.0, 0.9)  # Cap at 90%
            else:
                transition_prob = 0.1
            
            return max(0.05, transition_prob)  # Minimum 5% probability
            
        except Exception:
            return 0.1
    
    def _calculate_regime_stability(self, returns: pd.Series) -> List[float]:
        """Calculate current regime stability measures."""
        try:
            if len(returns) < 40:
                return [0.5, 0.5, 0.0]
            
            # Rolling window analysis
            window_size = 20
            rolling_vol = returns.rolling(window_size).std()
            rolling_mean = returns.rolling(window_size).mean()
            
            # Volatility stability (consistency of volatility)
            vol_stability = 1 - (rolling_vol.rolling(10).std().iloc[-1] / (rolling_vol.mean() + 1e-8))
            vol_stability = max(0.0, min(1.0, vol_stability)) if not np.isnan(vol_stability) else 0.5
            
            # Mean return stability
            mean_stability = 1 - (rolling_mean.rolling(10).std().iloc[-1] / (abs(rolling_mean.mean()) + 1e-8))
            mean_stability = max(0.0, min(1.0, mean_stability)) if not np.isnan(mean_stability) else 0.5
            
            # Regime duration (how long in current regime)
            # Simplified: use volatility regime consistency
            current_vol = rolling_vol.iloc[-1]
            vol_threshold = rolling_vol.quantile(0.7)
            
            # Count consecutive periods in same volatility regime
            is_high_vol = rolling_vol > vol_threshold
            regime_duration = 0
            
            for i in reversed(range(len(is_high_vol))):
                if is_high_vol.iloc[i] == is_high_vol.iloc[-1]:
                    regime_duration += 1
                else:
                    break
            
            duration_normalized = min(regime_duration / 50.0, 1.0)  # Normalize to ~50 days max
            
            return [vol_stability, mean_stability, duration_normalized]
            
        except Exception:
            return [0.5, 0.5, 0.0]
    
    def _calculate_regime_warning_indicators(self, returns: pd.Series) -> List[float]:
        """Calculate early warning indicators for regime changes."""
        try:
            if len(returns) < 30:
                return [0.0, 0.0, 0.0, 0.0]
            
            # 1. Volatility clustering breakdown
            abs_returns = returns.abs()
            vol_autocorr = abs_returns.rolling(20).apply(lambda x: x.autocorr(lag=1), raw=False)
            recent_autocorr = vol_autocorr.tail(5).mean()
            historical_autocorr = vol_autocorr.head(-5).mean()
            
            if not np.isnan(historical_autocorr) and historical_autocorr != 0:
                clustering_breakdown = max(0.0, historical_autocorr - recent_autocorr) / abs(historical_autocorr)
            else:
                clustering_breakdown = 0.0
            
            # 2. Return distribution changes
            recent_returns = returns.tail(20)
            historical_returns = returns.head(-20)
            
            if len(recent_returns) >= 10 and len(historical_returns) >= 10:
                # Compare skewness
                recent_skew = recent_returns.skew()
                historical_skew = historical_returns.skew()
                skew_change = abs(recent_skew - historical_skew) if not (np.isnan(recent_skew) or np.isnan(historical_skew)) else 0.0
                
                # Compare kurtosis
                recent_kurt = recent_returns.kurtosis()
                historical_kurt = historical_returns.kurtosis()
                kurt_change = abs(recent_kurt - historical_kurt) if not (np.isnan(recent_kurt) or np.isnan(historical_kurt)) else 0.0
            else:
                skew_change = 0.0
                kurt_change = 0.0
            
            # 3. Correlation structure breakdown (simplified)
            # Use autocorrelation changes as proxy
            recent_autocorr = returns.tail(20).autocorr(lag=1)
            historical_autocorr = returns.head(-20).autocorr(lag=1)
            
            if not (np.isnan(recent_autocorr) or np.isnan(historical_autocorr)):
                corr_breakdown = abs(recent_autocorr - historical_autocorr)
            else:
                corr_breakdown = 0.0
            
            return [
                min(clustering_breakdown, 1.0),
                min(skew_change / 5.0, 1.0),  # Normalize skewness change
                min(kurt_change / 10.0, 1.0), # Normalize kurtosis change
                min(corr_breakdown * 2.0, 1.0)  # Scale correlation change
            ]
            
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]
    
    def _extract_volatility_risk_features(self, df: pd.DataFrame, ticker: str) -> List[float]:
        """Extract volatility-based risk features."""
        try:
            features = []
            returns = df['close'].pct_change().dropna()
            
            if len(returns) < 20:
                return [0.0] * 6
            
            # 1. Value at Risk approximation
            var_95 = returns.quantile(0.05)  # 5% VaR
            var_99 = returns.quantile(0.01)  # 1% VaR
            
            features.extend([
                max(var_95, -0.2),  # Cap at -20%
                max(var_99, -0.3)   # Cap at -30%
            ])
            
            # 2. Expected Shortfall (Conditional VaR)
            if len(returns) >= 50:
                tail_returns = returns[returns <= var_95]
                expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var_95
                features.append(max(expected_shortfall, -0.4))  # Cap at -40%
            else:
                features.append(var_95 * 1.3)  # Approximate ES as 1.3 * VaR
            
            # 3. Volatility risk premium
            realized_vol = returns.std() * np.sqrt(252)
            # Use GARCH prediction as "expected" volatility
            garch_vol = self._simple_garch_prediction(returns)
            vol_risk_premium = garch_vol - realized_vol
            features.append(max(-0.5, min(0.5, vol_risk_premium)))
            
            # 4. Tail risk measures
            # Extreme return frequency
            extreme_threshold = 2 * returns.std()  # 2-sigma events
            extreme_frequency = (abs(returns) > extreme_threshold).sum() / len(returns)
            features.append(min(extreme_frequency, 0.2))  # Cap at 20%
            
            # 5. Downside deviation
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.0
            features.append(min(downside_deviation, 1.0))  # Cap at 100%
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting volatility risk features: {e}")
            return [0.0] * 6
    
    def _simple_garch_prediction(self, returns: pd.Series) -> float:
        """Simple GARCH(1,1) volatility prediction."""
        try:
            if len(returns) < 20:
                return 0.15  # Default 15% volatility
            
            # Simplified GARCH(1,1) parameters
            omega = 0.000002  # Long-run variance
            alpha = 0.1       # ARCH coefficient
            beta = 0.85       # GARCH coefficient
            
            # Current innovation
            last_return = returns.iloc[-1]
            last_innovation = last_return ** 2
            
            # Previous volatility estimate
            prev_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            prev_variance = (prev_vol / np.sqrt(252)) ** 2
            
            # GARCH(1,1) prediction
            next_variance = omega + alpha * last_innovation + beta * prev_variance
            next_vol = np.sqrt(next_variance) * np.sqrt(252)
            
            return min(next_vol, 2.0)  # Cap at 200%
            
        except Exception:
            return 0.15
    
    def _get_default_features(self) -> np.ndarray:
        """Get default features when extraction fails."""
        return np.zeros(40, dtype=np.float32)
    
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
        self._clean_cache()
    
    def _clean_cache(self) -> None:
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
        """Get names of all volatility regime and gap features."""
        names = []
        
        # Volatility regime features
        for window in self.volatility_windows:
            names.extend([f"vol_{window}_realized", f"vol_{window}_percentile"])
        
        names.extend([
            "regime_low_vol", "regime_normal_vol", "regime_high_vol", "regime_extreme_vol",
            "vol_persistence", "vol_of_vol", "arch_effect"
        ])
        
        # Gap analysis features
        names.extend([
            "gap_frequency", "gap_up_ratio", "gap_avg_size",
            "days_since_gap", "gap_momentum", "gap_fill_probability",
            "gap_size_trend", "gap_direction_consistency", "gap_avg_fill_prob",
            "gap_volume_confirm", "gap_avg_momentum", "gap_freq_acceleration"
        ])
        
        # Volatility modeling features
        names.extend([
            "garch_current_vol", "garch_mean_reversion", "garch_arch_strength", "garch_vol_innovation",
            "parkinson_vol", "garman_klass_vol", "range_efficiency",
            "vol_term_structure", "vol_skew", "vol_smile_curvature"
        ])
        
        # Regime transition features
        names.extend([
            "regime_transition_prob", "vol_stability", "mean_stability", "regime_duration",
            "clustering_breakdown", "skew_change", "kurt_change", "corr_breakdown"
        ])
        
        # Volatility risk features
        names.extend([
            "var_95", "var_99", "expected_shortfall", "vol_risk_premium",
            "extreme_frequency", "downside_deviation"
        ])
        
        return names
    
    def get_extractor_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the volatility regime extractor."""
        return {
            'volatility_windows': self.volatility_windows,
            'regime_lookback': self.regime_lookback,
            'gap_threshold': self.gap_threshold,
            'cache_size': len(self.feature_cache),
            'regime_history_length': len(self.regime_history),
            'gap_history_length': len(self.gap_history),
            'current_regime': self.regime_history[-1].value if self.regime_history else 'unknown',
            'volatility_models_cached': len(self.volatility_models),
            'regime_models_cached': len(self.regime_models),
            'settings': {
                'enable_garch_modeling': self.enable_garch_modeling,
                'enable_regime_prediction': self.enable_regime_prediction,
                'max_regimes': self.max_regimes,
                'cache_ttl_minutes': self.cache_ttl_minutes
            }
        }