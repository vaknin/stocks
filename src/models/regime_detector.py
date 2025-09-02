"""Hidden Markov Model-based market regime detection for adaptive trading."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from loguru import logger
import warnings
from pathlib import Path

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Regime detection will use simplified approach.")

from ..config.settings import config

class MarketRegime(Enum):
    """Different market regime types."""
    BULL_TREND = "bull_trend"          # Strong upward momentum
    BEAR_TREND = "bear_trend"          # Strong downward momentum
    SIDEWAYS = "sideways"              # Range-bound, low momentum
    HIGH_VOLATILITY = "high_volatility" # High uncertainty, choppy
    TRANSITION = "transition"          # Changing between regimes

@dataclass
class RegimeState:
    """Current regime state information."""
    regime: MarketRegime
    confidence: float
    duration: int  # Days in current regime
    probability_matrix: Dict[str, float]  # Probabilities for each regime
    features: Dict[str, float]  # Market features that led to this classification
    timestamp: datetime

class HiddenMarkovRegimeDetector:
    """Hidden Markov Model for detecting market regimes and adapting strategies."""
    
    def __init__(
        self,
        n_regimes: int = 4,
        lookback_days: int = 60,
        min_regime_duration: int = 3,
        regime_threshold: float = 0.6,
        update_frequency: str = "daily"
    ):
        """
        Initialize HMM regime detector.
        
        Args:
            n_regimes: Number of hidden states (market regimes)
            lookback_days: Days of data to use for regime detection
            min_regime_duration: Minimum days before switching regimes
            regime_threshold: Minimum confidence to declare regime change
            update_frequency: How often to update regime ('daily', 'hourly')
        """
        
        self.n_regimes = n_regimes
        self.lookback_days = lookback_days
        self.min_regime_duration = min_regime_duration
        self.regime_threshold = regime_threshold
        self.update_frequency = update_frequency
        
        # Initialize components
        if SKLEARN_AVAILABLE:
            self.hmm_model = GaussianMixture(
                n_components=n_regimes,
                covariance_type='full',
                max_iter=200,
                random_state=42
            )
            self.scaler = StandardScaler()
        else:
            self.hmm_model = None
            self.scaler = None
        
        # Regime tracking
        self.current_regime_state: Optional[RegimeState] = None
        self.regime_history: List[RegimeState] = []
        self.last_update: Optional[datetime] = None
        
        # Feature cache
        self.feature_cache = {}
        
        # Regime mapping (state index -> regime type)
        self.regime_mapping = {
            0: MarketRegime.BULL_TREND,
            1: MarketRegime.BEAR_TREND,
            2: MarketRegime.SIDEWAYS,
            3: MarketRegime.HIGH_VOLATILITY
        }
        
        logger.info(f"HMM regime detector initialized with {n_regimes} regimes")
    
    def extract_market_features(self, data_dict: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Extract market features for regime detection."""
        try:
            features_list = []
            
            # Aggregate market data
            all_returns = []
            all_volumes = []
            
            for ticker, df in data_dict.items():
                if len(df) < 20:
                    continue
                
                # Calculate returns
                returns = df['close'].pct_change().dropna()
                if len(returns) > 0:
                    all_returns.append(returns.values)
                    all_volumes.append(df['volume'].values[-len(returns):])
            
            if not all_returns:
                # Fallback: return neutral features
                return np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
            
            # Combine all stock data for market-wide features
            combined_returns = np.concatenate(all_returns)
            combined_volumes = np.concatenate(all_volumes)
            
            # Feature extraction for each day
            window_size = min(20, len(combined_returns) // len(data_dict))
            
            for i in range(window_size, len(combined_returns), len(data_dict)):
                end_idx = min(i, len(combined_returns))
                start_idx = max(0, end_idx - window_size)
                
                window_returns = combined_returns[start_idx:end_idx]
                window_volumes = combined_volumes[start_idx:end_idx]
                
                if len(window_returns) < 5:
                    continue
                
                # Calculate regime features
                features = self._calculate_regime_features(window_returns, window_volumes)
                features_list.append(features)
            
            if not features_list:
                return np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
            
            return np.array(features_list)
            
        except Exception as e:
            logger.error(f"Error extracting market features: {e}")
            return np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
    
    def _calculate_regime_features(self, returns: np.ndarray, volumes: np.ndarray) -> List[float]:
        """Calculate features that characterize market regimes."""
        try:
            # 1. Momentum (trend strength)
            momentum = np.mean(returns) * np.sqrt(252)  # Annualized
            
            # 2. Volatility
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # 3. Skewness (asymmetry of returns)
            skewness = self._calculate_skewness(returns)
            
            # 4. Volume trend (relative to recent average)
            volume_trend = (np.mean(volumes[-5:]) / np.mean(volumes[:-5])) - 1.0 if len(volumes) > 10 else 0.0
            
            # 5. Volatility clustering (GARCH-like effect)
            vol_clustering = np.corrcoef(np.abs(returns[:-1]), np.abs(returns[1:]))[0, 1]
            if np.isnan(vol_clustering):
                vol_clustering = 0.0
            
            return [momentum, volatility, skewness, volume_trend, vol_clustering]
            
        except Exception as e:
            logger.warning(f"Error calculating regime features: {e}")
            return [0.0, 0.0, 0.0, 0.0, 0.0]
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        try:
            if len(data) < 3:
                return 0.0
            
            mean = np.mean(data)
            std = np.std(data)
            
            if std == 0:
                return 0.0
            
            skew = np.mean(((data - mean) / std) ** 3)
            return skew
            
        except:
            return 0.0
    
    def detect_regime(
        self,
        data_dict: Dict[str, pd.DataFrame],
        force_update: bool = False
    ) -> RegimeState:
        """
        Detect current market regime using HMM.
        
        Args:
            data_dict: Historical OHLCV data for multiple stocks
            force_update: Force regime detection update
            
        Returns:
            Current regime state
        """
        try:
            # Check if update is needed
            now = datetime.now()
            if not force_update and self.last_update is not None:
                time_diff = now - self.last_update
                if self.update_frequency == "daily" and time_diff.days < 1:
                    return self.current_regime_state or self._fallback_regime()
                elif self.update_frequency == "hourly" and time_diff.seconds < 3600:
                    return self.current_regime_state or self._fallback_regime()
            
            # Extract market features
            features = self.extract_market_features(data_dict)
            
            if len(features) < 10:  # Need sufficient data
                logger.warning("Insufficient data for regime detection")
                return self._fallback_regime()
            
            # Detect regime using HMM or fallback method
            if self.hmm_model is not None and SKLEARN_AVAILABLE:
                regime_state = self._detect_with_hmm(features)
            else:
                regime_state = self._detect_with_heuristics(features)
            
            # Apply regime stability rules
            stable_regime = self._apply_stability_rules(regime_state)
            
            # Update tracking
            self.current_regime_state = stable_regime
            self.regime_history.append(stable_regime)
            self.last_update = now
            
            # Keep only recent history
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            logger.info(f"Detected regime: {stable_regime.regime.value} (confidence: {stable_regime.confidence:.2f})")
            
            return stable_regime
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return self._fallback_regime()
    
    def _detect_with_hmm(self, features: np.ndarray) -> RegimeState:
        """Detect regime using Hidden Markov Model."""
        try:
            # Normalize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Fit HMM model
            self.hmm_model.fit(features_scaled)
            
            # Get current regime probabilities
            current_features = features_scaled[-1:].reshape(1, -1)
            regime_probs = self.hmm_model.predict_proba(current_features)[0]
            
            # Find most likely regime
            regime_idx = np.argmax(regime_probs)
            confidence = regime_probs[regime_idx]
            
            # Map to regime type
            regime = self.regime_mapping.get(regime_idx, MarketRegime.TRANSITION)
            
            # Create probability matrix
            prob_matrix = {
                regime_type.value: float(regime_probs[idx])
                for idx, regime_type in self.regime_mapping.items()
                if idx < len(regime_probs)
            }
            
            # Extract features for interpretation
            feature_dict = {
                'momentum': float(features[-1, 0]),
                'volatility': float(features[-1, 1]),
                'skewness': float(features[-1, 2]),
                'volume_trend': float(features[-1, 3]),
                'vol_clustering': float(features[-1, 4])
            }
            
            return RegimeState(
                regime=regime,
                confidence=confidence,
                duration=self._calculate_regime_duration(regime),
                probability_matrix=prob_matrix,
                features=feature_dict,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in HMM regime detection: {e}")
            return self._detect_with_heuristics(features)
    
    def _detect_with_heuristics(self, features: np.ndarray) -> RegimeState:
        """Fallback regime detection using heuristic rules."""
        try:
            # Use latest features
            latest_features = features[-1]
            momentum, volatility, skewness, volume_trend, vol_clustering = latest_features
            
            # Heuristic regime classification
            regime = MarketRegime.SIDEWAYS  # Default
            confidence = 0.6
            
            # High volatility regime
            if volatility > 0.3:  # 30% annualized volatility
                regime = MarketRegime.HIGH_VOLATILITY
                confidence = min(0.8, volatility / 0.4)
            
            # Trend regimes
            elif abs(momentum) > 0.1:  # 10% annualized momentum
                if momentum > 0:
                    regime = MarketRegime.BULL_TREND
                else:
                    regime = MarketRegime.BEAR_TREND
                confidence = min(0.8, abs(momentum) / 0.2)
            
            # Sideways (low momentum, low volatility)
            elif abs(momentum) < 0.05 and volatility < 0.2:
                regime = MarketRegime.SIDEWAYS
                confidence = 0.7
            
            # Create probability matrix (simplified)
            prob_matrix = {
                MarketRegime.BULL_TREND.value: 0.8 if regime == MarketRegime.BULL_TREND else 0.1,
                MarketRegime.BEAR_TREND.value: 0.8 if regime == MarketRegime.BEAR_TREND else 0.1,
                MarketRegime.SIDEWAYS.value: 0.8 if regime == MarketRegime.SIDEWAYS else 0.1,
                MarketRegime.HIGH_VOLATILITY.value: 0.8 if regime == MarketRegime.HIGH_VOLATILITY else 0.1
            }
            
            # Normalize probabilities
            total_prob = sum(prob_matrix.values())
            prob_matrix = {k: v/total_prob for k, v in prob_matrix.items()}
            
            feature_dict = {
                'momentum': float(momentum),
                'volatility': float(volatility),
                'skewness': float(skewness),
                'volume_trend': float(volume_trend),
                'vol_clustering': float(vol_clustering)
            }
            
            return RegimeState(
                regime=regime,
                confidence=confidence,
                duration=self._calculate_regime_duration(regime),
                probability_matrix=prob_matrix,
                features=feature_dict,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in heuristic regime detection: {e}")
            return self._fallback_regime()
    
    def _apply_stability_rules(self, new_regime_state: RegimeState) -> RegimeState:
        """Apply stability rules to prevent regime switching too frequently."""
        try:
            # If no previous regime, return new regime
            if self.current_regime_state is None:
                new_regime_state.duration = 1
                return new_regime_state
            
            # Check if regime changed
            if new_regime_state.regime == self.current_regime_state.regime:
                # Same regime, increment duration
                new_regime_state.duration = self.current_regime_state.duration + 1
                return new_regime_state
            
            # Regime changed - check stability rules
            current_duration = self.current_regime_state.duration
            
            # Don't switch if current regime is too new
            if current_duration < self.min_regime_duration:
                # Keep current regime but reduce confidence
                stabilized_state = RegimeState(
                    regime=self.current_regime_state.regime,
                    confidence=min(self.current_regime_state.confidence * 0.9, new_regime_state.confidence),
                    duration=current_duration + 1,
                    probability_matrix=new_regime_state.probability_matrix,
                    features=new_regime_state.features,
                    timestamp=new_regime_state.timestamp
                )
                return stabilized_state
            
            # Check confidence threshold for regime change
            if new_regime_state.confidence < self.regime_threshold:
                # Not confident enough to switch
                stabilized_state = RegimeState(
                    regime=self.current_regime_state.regime,
                    confidence=self.current_regime_state.confidence * 0.95,
                    duration=current_duration + 1,
                    probability_matrix=new_regime_state.probability_matrix,
                    features=new_regime_state.features,
                    timestamp=new_regime_state.timestamp
                )
                return stabilized_state
            
            # Allow regime change
            new_regime_state.duration = 1
            return new_regime_state
            
        except Exception as e:
            logger.error(f"Error applying stability rules: {e}")
            return new_regime_state
    
    def _calculate_regime_duration(self, regime: MarketRegime) -> int:
        """Calculate how long we've been in the current regime."""
        if self.current_regime_state is None:
            return 1
        
        if self.current_regime_state.regime == regime:
            return self.current_regime_state.duration + 1
        else:
            return 1
    
    def _fallback_regime(self) -> RegimeState:
        """Return a fallback regime state when detection fails."""
        return RegimeState(
            regime=MarketRegime.SIDEWAYS,
            confidence=0.5,
            duration=1,
            probability_matrix={
                MarketRegime.BULL_TREND.value: 0.25,
                MarketRegime.BEAR_TREND.value: 0.25,
                MarketRegime.SIDEWAYS.value: 0.25,
                MarketRegime.HIGH_VOLATILITY.value: 0.25
            },
            features={
                'momentum': 0.0,
                'volatility': 0.2,
                'skewness': 0.0,
                'volume_trend': 0.0,
                'vol_clustering': 0.0
            },
            timestamp=datetime.now()
        )
    
    def get_regime_adaptation_factors(self, regime_state: RegimeState) -> Dict[str, float]:
        """
        Get adaptation factors for trading strategy based on current regime.
        
        Returns:
            Dictionary with adjustment factors for strategy parameters
        """
        regime = regime_state.regime
        confidence = regime_state.confidence
        
        # Base adaptation factors
        factors = {
            'position_size_multiplier': 1.0,
            'confidence_threshold_adjustment': 0.0,
            'profit_target_multiplier': 1.0,
            'stop_loss_multiplier': 1.0,
            'timeframe_preference': 'balanced'  # 'scalping', 'swing', 'position', 'balanced'
        }
        
        # Regime-specific adaptations
        if regime == MarketRegime.BULL_TREND:
            factors.update({
                'position_size_multiplier': 1.2 * confidence,
                'confidence_threshold_adjustment': -0.05,  # Lower threshold in bull market
                'profit_target_multiplier': 1.3,
                'stop_loss_multiplier': 0.8,
                'timeframe_preference': 'position'  # Favor longer holds in bull markets
            })
        
        elif regime == MarketRegime.BEAR_TREND:
            factors.update({
                'position_size_multiplier': 0.7,
                'confidence_threshold_adjustment': 0.1,  # Higher threshold in bear market
                'profit_target_multiplier': 0.8,
                'stop_loss_multiplier': 0.6,  # Tighter stops
                'timeframe_preference': 'scalping'  # Favor quick trades in bear markets
            })
        
        elif regime == MarketRegime.HIGH_VOLATILITY:
            factors.update({
                'position_size_multiplier': 0.6,
                'confidence_threshold_adjustment': 0.15,  # Much higher threshold
                'profit_target_multiplier': 1.5,  # Bigger targets to capture volatility
                'stop_loss_multiplier': 0.5,  # Much tighter stops
                'timeframe_preference': 'scalping'  # Quick in/out in volatile markets
            })
        
        elif regime == MarketRegime.SIDEWAYS:
            factors.update({
                'position_size_multiplier': 0.9,
                'confidence_threshold_adjustment': 0.05,
                'profit_target_multiplier': 0.9,  # Smaller targets in range-bound
                'stop_loss_multiplier': 1.1,  # Slightly wider stops
                'timeframe_preference': 'swing'  # Favor swing trading in sideways
            })
        
        return factors
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of current regime and history."""
        if self.current_regime_state is None:
            return {'current_regime': None, 'regime_history': []}
        
        # Recent regime history
        recent_history = []
        for state in self.regime_history[-10:]:
            recent_history.append({
                'regime': state.regime.value,
                'confidence': state.confidence,
                'duration': state.duration,
                'timestamp': state.timestamp.isoformat()
            })
        
        return {
            'current_regime': {
                'regime': self.current_regime_state.regime.value,
                'confidence': self.current_regime_state.confidence,
                'duration': self.current_regime_state.duration,
                'probability_matrix': self.current_regime_state.probability_matrix,
                'features': self.current_regime_state.features,
                'adaptation_factors': self.get_regime_adaptation_factors(self.current_regime_state)
            },
            'regime_history': recent_history,
            'model_info': {
                'n_regimes': self.n_regimes,
                'lookback_days': self.lookback_days,
                'min_regime_duration': self.min_regime_duration,
                'regime_threshold': self.regime_threshold,
                'sklearn_available': SKLEARN_AVAILABLE
            }
        }

# Alias for backward compatibility with model validator
RegimeDetector = HiddenMarkovRegimeDetector
