"""MAPIE uncertainty quantification for trading predictions."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any, Callable
from datetime import datetime
import warnings
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

try:
    from mapie.regression import MapieRegressor
    from mapie.time_series_regression import MapieTimeSeriesRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False
    # Define fallback classes when sklearn is not available
    class RandomForestRegressor:
        def __init__(self, *args, **kwargs):
            pass
    class LinearRegression:
        def __init__(self, *args, **kwargs):
            pass
    logger.warning("MAPIE not available. Running in mock mode for uncertainty quantification.")

from ..config.settings import config
from .regime_detector import HiddenMarkovRegimeDetector, MarketRegime, RegimeState

class ConformalPredictor:
    """MAPIE-based conformal prediction for financial time series uncertainty quantification."""
    
    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        method: str = "cv+",
        alpha: float = 0.25,
        n_splits: int = 5,
        cv: Optional[Any] = None,
        random_state: int = 42
    ):
        """Initialize conformal predictor.
        
        Args:
            base_estimator: Base ML model for predictions. Defaults to RandomForest.
            method: Conformal prediction method ('naive', 'cv', 'cv+', 'jackknife+')
            alpha: Miscoverage level (0.25 for 75% prediction intervals)
            n_splits: Number of CV splits for cross-conformal methods
            cv: Cross-validation strategy. Defaults to TimeSeriesSplit
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.alpha = alpha
        self.n_splits = n_splits
        self.random_state = random_state
        
        # Default base estimator
        if base_estimator is None:
            self.base_estimator = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            self.base_estimator = base_estimator
        
        # Default CV strategy for time series
        if cv is None:
            self.cv = TimeSeriesSplit(n_splits=n_splits, test_size=None)
        else:
            self.cv = cv
        
        # MAPIE regressor
        self.mapie_regressor = None
        self.is_fitted = False
        
        # Coverage tracking
        self.coverage_history = []
        self.prediction_history = []
        
        logger.info(f"ConformalPredictor initialized with method={method}, alpha={alpha}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'ConformalPredictor':
        """Fit the conformal predictor.
        
        Args:
            X: Input features array of shape (n_samples, n_features)
            y: Target values array of shape (n_samples,)
            sample_weight: Sample weights (optional)
            
        Returns:
            Self for method chaining
        """
        if not MAPIE_AVAILABLE:
            logger.warning("MAPIE not available - fitting mock predictor")
            self.is_fitted = True
            return self
        
        try:
            # Initialize MAPIE regressor
            self.mapie_regressor = MapieRegressor(
                estimator=self.base_estimator,
                method=self.method,
                cv=self.cv,
                random_state=self.random_state
            )
            
            # Fit the model
            self.mapie_regressor.fit(X, y, sample_weight=sample_weight)
            self.is_fitted = True
            
            logger.info(f"ConformalPredictor fitted on {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Failed to fit ConformalPredictor: {e}")
            self.is_fitted = False
            raise
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        alpha: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions with prediction intervals.
        
        Args:
            X: Input features for prediction
            alpha: Miscoverage level override (uses instance alpha if None)
            
        Returns:
            Tuple of (predictions, prediction_intervals)
            - predictions: Point predictions of shape (n_samples,)
            - prediction_intervals: Intervals of shape (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if alpha is None:
            alpha = self.alpha
        
        if not MAPIE_AVAILABLE:
            return self._mock_predict(X, alpha)
        
        try:
            # Generate predictions with intervals
            y_pred, y_intervals = self.mapie_regressor.predict(X, alpha=alpha)
            
            # Store prediction for coverage tracking
            prediction_data = {
                'predictions': y_pred,
                'intervals': y_intervals,
                'alpha': alpha,
                'timestamp': datetime.now()
            }
            self.prediction_history.append(prediction_data)
            
            return y_pred, y_intervals
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._mock_predict(X, alpha)
    
    def _mock_predict(self, X: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mock predictions when MAPIE is not available.
        
        Args:
            X: Input features
            alpha: Miscoverage level
            
        Returns:
            Tuple of mock predictions and intervals
        """
        n_samples = X.shape[0]
        
        # Generate mock point predictions
        predictions = np.random.normal(0.001, 0.02, n_samples)  # 0.1% mean, 2% std
        
        # Generate mock prediction intervals
        quantile = 1 - alpha/2
        interval_width = np.random.normal(0.04, 0.01, n_samples)  # ~4% average width
        interval_width = np.abs(interval_width)  # Ensure positive
        
        lower_bounds = predictions - interval_width/2
        upper_bounds = predictions + interval_width/2
        
        prediction_intervals = np.column_stack([lower_bounds, upper_bounds])
        
        logger.debug(f"Generated mock predictions for {n_samples} samples")
        
        return predictions, prediction_intervals
    
    def calculate_coverage(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray,
        intervals: np.ndarray,
        alpha: float
    ) -> Dict[str, float]:
        """Calculate empirical coverage statistics.
        
        Args:
            y_true: True values
            predictions: Point predictions
            intervals: Prediction intervals
            alpha: Nominal miscoverage level
            
        Returns:
            Dictionary with coverage statistics
        """
        # Empirical coverage
        within_intervals = (
            (y_true >= intervals[:, 0]) & (y_true <= intervals[:, 1])
        )
        empirical_coverage = np.mean(within_intervals)
        nominal_coverage = 1 - alpha
        
        # Interval width statistics
        interval_widths = intervals[:, 1] - intervals[:, 0]
        mean_width = np.mean(interval_widths)
        median_width = np.median(interval_widths)
        
        # Coverage efficiency (narrower intervals are better)
        coverage_efficiency = empirical_coverage / mean_width
        
        # Prediction accuracy
        mae = np.mean(np.abs(y_true - predictions))
        rmse = np.sqrt(np.mean((y_true - predictions) ** 2))
        
        coverage_stats = {
            'empirical_coverage': empirical_coverage,
            'nominal_coverage': nominal_coverage,
            'coverage_gap': abs(empirical_coverage - nominal_coverage),
            'mean_interval_width': mean_width,
            'median_interval_width': median_width,
            'coverage_efficiency': coverage_efficiency,
            'mae': mae,
            'rmse': rmse,
            'n_samples': len(y_true)
        }
        
        # Store for tracking
        coverage_stats['timestamp'] = datetime.now()
        self.coverage_history.append(coverage_stats)
        
        logger.info(f"Coverage: {empirical_coverage:.3f} (target: {nominal_coverage:.3f})")
        
        return coverage_stats
    
    def adaptive_alpha_tuning(
        self,
        recent_coverage_history: List[Dict],
        target_coverage: float = None,
        adjustment_rate: float = 0.1
    ) -> float:
        """Adaptively tune alpha based on recent coverage performance.
        
        Args:
            recent_coverage_history: Recent coverage statistics
            target_coverage: Target coverage level (defaults to 1-alpha)
            adjustment_rate: Rate of alpha adjustment
            
        Returns:
            New alpha value
        """
        if target_coverage is None:
            target_coverage = 1 - self.alpha
        
        if len(recent_coverage_history) < 3:
            return self.alpha  # Not enough history
        
        # Calculate recent average coverage
        recent_coverages = [
            stats['empirical_coverage'] 
            for stats in recent_coverage_history[-5:]
        ]
        avg_recent_coverage = np.mean(recent_coverages)
        
        # Calculate adjustment
        coverage_error = avg_recent_coverage - target_coverage
        alpha_adjustment = coverage_error * adjustment_rate
        
        # Update alpha with bounds
        new_alpha = np.clip(
            self.alpha - alpha_adjustment,
            0.05,   # Minimum alpha (95% intervals)
            0.40    # Maximum alpha (60% intervals)
        )
        
        if abs(new_alpha - self.alpha) > 0.01:  # Significant change
            logger.info(f"Adjusting alpha from {self.alpha:.3f} to {new_alpha:.3f}")
            logger.info(f"Recent coverage: {avg_recent_coverage:.3f}, target: {target_coverage:.3f}")
            self.alpha = new_alpha
        
        return self.alpha
    
    def filter_predictions_by_uncertainty(
        self,
        predictions: np.ndarray,
        intervals: np.ndarray,
        max_interval_width: float = 0.03,
        min_confidence: float = None
    ) -> np.ndarray:
        """Filter predictions based on uncertainty criteria.
        
        Args:
            predictions: Point predictions
            intervals: Prediction intervals
            max_interval_width: Maximum allowed interval width (3% default)
            min_confidence: Minimum required confidence (computed from alpha if None)
            
        Returns:
            Boolean mask for high-confidence predictions
        """
        # Interval width filter
        interval_widths = intervals[:, 1] - intervals[:, 0]
        width_filter = interval_widths <= max_interval_width
        
        # Confidence filter (if specified)
        if min_confidence is not None:
            # For conformal prediction, confidence is related to interval width
            # Narrower intervals indicate higher confidence
            confidence_scores = 1.0 / (1.0 + interval_widths)  # Inverse relationship
            confidence_filter = confidence_scores >= min_confidence
            combined_filter = width_filter & confidence_filter
        else:
            combined_filter = width_filter
        
        n_total = len(predictions)
        n_filtered = np.sum(combined_filter)
        filter_rate = n_filtered / n_total
        
        logger.debug(f"Uncertainty filter: {n_filtered}/{n_total} predictions ({filter_rate:.2%})")
        
        return combined_filter
    
    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """Get summary of uncertainty quantification performance.
        
        Returns:
            Dictionary with uncertainty metrics
        """
        if not self.coverage_history:
            return {'status': 'no_coverage_history'}
        
        recent_stats = self.coverage_history[-10:]  # Last 10 evaluations
        
        summary = {
            'current_alpha': self.alpha,
            'method': self.method,
            'n_evaluations': len(self.coverage_history),
            'recent_average_coverage': np.mean([s['empirical_coverage'] for s in recent_stats]),
            'recent_average_width': np.mean([s['mean_interval_width'] for s in recent_stats]),
            'recent_coverage_efficiency': np.mean([s['coverage_efficiency'] for s in recent_stats]),
            'coverage_stability': np.std([s['empirical_coverage'] for s in recent_stats]),
            'is_well_calibrated': None
        }
        
        # Check if well calibrated (coverage close to nominal)
        coverage_gap = abs(summary['recent_average_coverage'] - (1 - self.alpha))
        summary['is_well_calibrated'] = coverage_gap < 0.05  # Within 5%
        
        return summary
    
    def save_model(self, filepath: str) -> None:
        """Save the fitted conformal predictor.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        try:
            import joblib
            model_data = {
                'mapie_regressor': self.mapie_regressor,
                'method': self.method,
                'alpha': self.alpha,
                'n_splits': self.n_splits,
                'random_state': self.random_state,
                'coverage_history': self.coverage_history,
                'is_fitted': self.is_fitted
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"ConformalPredictor saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load a fitted conformal predictor.
        
        Args:
            filepath: Path to load the model from
        """
        try:
            import joblib
            model_data = joblib.load(filepath)
            
            self.mapie_regressor = model_data['mapie_regressor']
            self.method = model_data['method']
            self.alpha = model_data['alpha']
            self.n_splits = model_data['n_splits']
            self.random_state = model_data['random_state']
            self.coverage_history = model_data.get('coverage_history', [])
            self.is_fitted = model_data['is_fitted']
            
            logger.info(f"ConformalPredictor loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


class TradingUncertaintyFilter:
    """Trading-specific uncertainty filtering for signal generation."""
    
    def __init__(
        self,
        max_interval_width: float = 0.50,
        confidence_threshold: float = 0.65,
        min_prediction_magnitude: float = 0.005
    ):
        """Initialize uncertainty filter for trading signals.
        
        Args:
            max_interval_width: Maximum prediction interval width (50% - realistic for volatile semiconductor stocks)
            confidence_threshold: Minimum confidence for trade signals (65% - practical threshold)
            min_prediction_magnitude: Minimum prediction magnitude for signals (0.5% - meaningful trades)
        """
        # Set realistic thresholds for financial markets
        self.max_interval_width = max_interval_width
        self.confidence_threshold = confidence_threshold
        self.min_prediction_magnitude = min_prediction_magnitude
        
        logger.info(f"TradingUncertaintyFilter initialized with thresholds: "
                   f"width<{max_interval_width:.1%}, conf>{confidence_threshold:.1%}, min_mag>{min_prediction_magnitude:.1%}")
    
    def filter_trading_signals(
        self,
        predictions: np.ndarray,
        intervals: np.ndarray,
        tickers: List[str] = None
    ) -> Dict[str, Any]:
        """Filter predictions for trading signal generation.
        
        Args:
            predictions: Point predictions
            intervals: Prediction intervals
            tickers: Ticker symbols (optional)
            
        Returns:
            Dictionary with filtered signals and statistics
        """
        # Calculate interval properties
        interval_widths = intervals[:, 1] - intervals[:, 0]
        prediction_magnitudes = np.abs(predictions)
        
        # Apply filters
        width_filter = interval_widths <= self.max_interval_width
        magnitude_filter = prediction_magnitudes >= self.min_prediction_magnitude
        
        # Confidence approximation (inverse of normalized width)
        max_width = np.max(interval_widths)
        if max_width > 0:
            normalized_widths = interval_widths / max_width
            confidence_scores = 1.0 - normalized_widths
        else:
            confidence_scores = np.ones_like(interval_widths)
        
        confidence_filter = confidence_scores >= self.confidence_threshold
        
        # Combined filter
        combined_filter = width_filter & magnitude_filter & confidence_filter
        
        # Generate signals
        signals = np.where(
            combined_filter,
            np.where(predictions > 0, 'BUY', 'SELL'),
            'HOLD'
        )
        
        # Prepare results
        results = {
            'signals': signals,
            'predictions': predictions,
            'intervals': intervals,
            'confidence_scores': confidence_scores,
            'filter_mask': combined_filter,
            'statistics': {
                'total_predictions': len(predictions),
                'buy_signals': np.sum(signals == 'BUY'),
                'sell_signals': np.sum(signals == 'SELL'),
                'hold_signals': np.sum(signals == 'HOLD'),
                'signal_rate': np.sum(combined_filter) / len(predictions),
                'avg_confidence': np.mean(confidence_scores[combined_filter]) if np.any(combined_filter) else 0,
                'avg_interval_width': np.mean(interval_widths[combined_filter]) if np.any(combined_filter) else 0
            }
        }
        
        if tickers:
            results['tickers'] = tickers
            
            # Create ticker-specific results
            ticker_signals = {}
            for i, ticker in enumerate(tickers):
                ticker_signals[ticker] = {
                    'signal': signals[i],
                    'prediction': predictions[i],
                    'interval': intervals[i],
                    'confidence': confidence_scores[i],
                    'filtered': combined_filter[i]
                }
            results['ticker_signals'] = ticker_signals
        
        logger.debug(f"Generated {results['statistics']['buy_signals']} BUY, "
                    f"{results['statistics']['sell_signals']} SELL signals "
                    f"from {results['statistics']['total_predictions']} predictions")
        
        return results
    
    def update_thresholds(
        self,
        performance_data: Dict[str, float],
        adjustment_rate: float = 0.1
    ) -> None:
        """Update filtering thresholds based on performance feedback.
        
        Args:
            performance_data: Dictionary with performance metrics
            adjustment_rate: Rate of threshold adjustment
        """
        # Example adaptive threshold adjustment
        if 'win_rate' in performance_data:
            win_rate = performance_data['win_rate']
            target_win_rate = 0.65  # Target 65% win rate
            
            if win_rate < target_win_rate - 0.05:  # Too low win rate
                # Increase confidence threshold (more selective)
                new_threshold = min(0.90, self.confidence_threshold + adjustment_rate)
                logger.info(f"Increasing confidence threshold from {self.confidence_threshold:.3f} to {new_threshold:.3f}")
                self.confidence_threshold = new_threshold
                
            elif win_rate > target_win_rate + 0.10:  # Very high win rate
                # Decrease threshold (less selective, more trades)
                new_threshold = max(0.60, self.confidence_threshold - adjustment_rate)
                logger.info(f"Decreasing confidence threshold from {self.confidence_threshold:.3f} to {new_threshold:.3f}")
                self.confidence_threshold = new_threshold
    
    def filter_prediction(
        self,
        prediction: float,
        confidence: float,
        prediction_interval: Tuple[float, float],
        regime_info: Optional[Dict[str, Any]] = None
    ) -> Optional[float]:
        """Filter individual prediction based on uncertainty criteria.
        
        Args:
            prediction: Point prediction (expected return)
            confidence: Model confidence score
            prediction_interval: Prediction interval [lower, upper]
            regime_info: Optional regime information for adaptive filtering
            
        Returns:
            Filtered prediction or None if filtered out
        """
        # Calculate interval width as percentage of prediction
        absolute_width = prediction_interval[1] - prediction_interval[0]
        
        # Handle edge case where prediction is very small or zero
        if abs(prediction) < 1e-6:  # Very small prediction value
            # Use absolute width comparison for very small predictions
            interval_width = absolute_width
        else:
            # Calculate relative width as percentage of prediction magnitude
            interval_width = absolute_width / abs(prediction)
        
        # Apply regime-specific thresholds or defaults
        if regime_info:
            regime_name = regime_info.get('current_regime', '').lower()
            
            # Adaptive thresholds based on regime - realistic for semiconductor stocks
            if regime_name == 'high_volatility':
                max_width_threshold = 0.70  # 70% during volatile periods (semiconductor stocks are highly volatile)
                confidence_threshold = 0.60   # Lower confidence in volatile times
                min_prediction_magnitude = self.min_prediction_magnitude
            elif regime_name == 'bear_trend':
                max_width_threshold = 0.60  # 60% during bear markets (higher uncertainty)
                confidence_threshold = 0.55   # Lower confidence in bear markets
                min_prediction_magnitude = self.min_prediction_magnitude
            elif regime_name == 'bull_trend':
                max_width_threshold = 0.45  # 45% during bull markets (more predictable)
                confidence_threshold = 0.65   # Standard confidence
                min_prediction_magnitude = self.min_prediction_magnitude
            else:  # sideways or unknown
                max_width_threshold = 0.55  # 55% default (realistic for tech stocks)
                confidence_threshold = 0.60   # Reasonable default
                min_prediction_magnitude = self.min_prediction_magnitude
        else:
            # Default thresholds (already set to realistic values)
            max_width_threshold = self.max_interval_width  # 50%
            confidence_threshold = self.confidence_threshold  # 65%
            min_prediction_magnitude = self.min_prediction_magnitude  # 0.5%
        
        # Apply filters with comprehensive logging for monitoring
        regime_context = f" (regime: {regime_info.get('current_regime', 'unknown')})" if regime_info else ""
        
        if interval_width > max_width_threshold:
            if abs(prediction) < 1e-6:  # Very small prediction - log absolute values
                logger.info(f"🚫 MAPIE Filter: interval too wide ({absolute_width:.4f} abs > {max_width_threshold:.4f} abs){regime_context}")
            else:
                logger.info(f"🚫 MAPIE Filter: interval too wide ({interval_width:.1%} > {max_width_threshold:.1%}){regime_context}")
            return None
        
        if confidence < confidence_threshold:
            logger.info(f"🚫 MAPIE Filter: low confidence ({confidence:.1%} < {confidence_threshold:.1%}){regime_context}")
            return None
        
        if abs(prediction) < min_prediction_magnitude:
            logger.info(f"🚫 MAPIE Filter: magnitude too small ({abs(prediction):.1%} < {min_prediction_magnitude:.1%}){regime_context}")
            return None
        
        # All filters passed - log success for monitoring
        logger.info(f"✅ MAPIE Filter PASSED: {prediction:.1%} return, {confidence:.1%} confidence, {interval_width:.1%} width{regime_context}")
        return prediction


class RegimeAdaptiveMapie(ConformalPredictor):
    """MAPIE with regime-adaptive alpha parameters for improved uncertainty quantification."""
    
    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        regime_detector: Optional[HiddenMarkovRegimeDetector] = None,
        regime_alpha_mapping: Optional[Dict[str, float]] = None,
        alpha_adaptation_rate: float = 0.05,
        coverage_history_window: int = 20,
        **kwargs
    ):
        """Initialize regime-adaptive MAPIE.
        
        Args:
            base_estimator: Base ML model
            regime_detector: Regime detection system
            regime_alpha_mapping: Custom alpha values for each regime
            alpha_adaptation_rate: Rate of alpha adaptation based on coverage
            coverage_history_window: Window for tracking coverage performance
        """
        super().__init__(base_estimator=base_estimator, **kwargs)
        
        self.regime_detector = regime_detector or HiddenMarkovRegimeDetector()
        self.alpha_adaptation_rate = alpha_adaptation_rate
        self.coverage_history_window = coverage_history_window
        
        # Regime-specific alpha values (research-validated)
        self.regime_alpha_mapping = regime_alpha_mapping or {
            MarketRegime.BULL_TREND.value: 0.20,      # 80% intervals in bull markets (more confident)
            MarketRegime.BEAR_TREND.value: 0.35,      # 65% intervals in bear markets (less confident)
            MarketRegime.HIGH_VOLATILITY.value: 0.40, # 60% intervals in volatile periods
            MarketRegime.SIDEWAYS.value: 0.25,        # 75% intervals in sideways markets
            MarketRegime.TRANSITION.value: 0.30       # 70% intervals during transitions
        }
        
        # Coverage tracking by regime
        self.regime_coverage_history = {regime: [] for regime in self.regime_alpha_mapping.keys()}
        self.current_regime_alpha = self.alpha
        
        logger.info(f"RegimeAdaptiveMapie initialized with regime-specific alphas: {self.regime_alpha_mapping}")
    
    def get_regime_adaptive_alpha(
        self,
        market_data: Dict[str, pd.DataFrame],
        current_regime: Optional[RegimeState] = None
    ) -> float:
        """Get regime-adaptive alpha parameter.
        
        Args:
            market_data: Market data for regime detection
            current_regime: Pre-detected regime (optional)
            
        Returns:
            Adapted alpha value
        """
        try:
            # Detect current regime if not provided
            if current_regime is None:
                current_regime = self.regime_detector.detect_regime(market_data)
            
            # Get base alpha for current regime
            base_alpha = self.regime_alpha_mapping.get(
                current_regime.regime.value,
                self.alpha  # Fallback to default
            )
            
            # Apply confidence-based adjustment
            confidence_factor = current_regime.confidence
            confidence_adjustment = (confidence_factor - 0.5) * 0.1  # ±5% max adjustment
            
            # Get coverage-based adaptation
            coverage_adjustment = self._get_coverage_based_adjustment(current_regime.regime.value)
            
            # Combine adjustments
            adapted_alpha = base_alpha - confidence_adjustment + coverage_adjustment
            
            # Ensure alpha stays within reasonable bounds
            adapted_alpha = np.clip(adapted_alpha, 0.05, 0.45)
            
            self.current_regime_alpha = adapted_alpha
            
            logger.debug(f"Adapted alpha for {current_regime.regime.value}: {adapted_alpha:.3f} "
                        f"(base: {base_alpha:.3f}, confidence_adj: {confidence_adjustment:.3f}, "
                        f"coverage_adj: {coverage_adjustment:.3f})")
            
            return adapted_alpha
            
        except Exception as e:
            logger.error(f"Error in regime adaptive alpha calculation: {e}")
            return self.alpha
    
    def _get_coverage_based_adjustment(self, regime_name: str) -> float:
        """Calculate alpha adjustment based on recent coverage performance."""
        try:
            regime_history = self.regime_coverage_history.get(regime_name, [])
            
            if len(regime_history) < 3:
                return 0.0  # Not enough history
            
            # Calculate recent coverage statistics
            recent_coverage = [entry['empirical_coverage'] for entry in regime_history[-self.coverage_history_window:]]
            avg_coverage = np.mean(recent_coverage)
            target_coverage = 1 - self.regime_alpha_mapping[regime_name]
            
            # Coverage error (positive means over-covering, negative means under-covering)
            coverage_error = avg_coverage - target_coverage
            
            # Adjust alpha to correct coverage (increase alpha if over-covering, decrease if under-covering)
            adjustment = coverage_error * self.alpha_adaptation_rate
            
            return adjustment
            
        except Exception as e:
            logger.warning(f"Error calculating coverage adjustment: {e}")
            return 0.0
    
    def predict_with_regime_adaptation(
        self,
        X: np.ndarray,
        market_data: Dict[str, pd.DataFrame],
        current_regime: Optional[RegimeState] = None
    ) -> Tuple[np.ndarray, np.ndarray, RegimeState]:
        """Generate regime-adapted predictions with uncertainty intervals.
        
        Args:
            X: Input features
            market_data: Market data for regime detection
            current_regime: Pre-detected regime (optional)
            
        Returns:
            Tuple of (predictions, intervals, regime_state)
        """
        try:
            # Get regime-adaptive alpha
            adaptive_alpha = self.get_regime_adaptive_alpha(market_data, current_regime)
            
            # Get current regime for return
            if current_regime is None:
                current_regime = self.regime_detector.detect_regime(market_data)
            
            # Make predictions with adapted alpha
            predictions, intervals = self.predict(X, alpha=adaptive_alpha)
            
            logger.info(f"Regime-adaptive prediction: {current_regime.regime.value} "
                       f"(alpha: {adaptive_alpha:.3f}, confidence: {current_regime.confidence:.3f})")
            
            return predictions, intervals, current_regime
            
        except Exception as e:
            logger.error(f"Error in regime-adaptive prediction: {e}")
            # Fallback to standard prediction
            predictions, intervals = self.predict(X)
            fallback_regime = self.regime_detector._fallback_regime()
            return predictions, intervals, fallback_regime
    
    def validate_coverage(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray,
        intervals: np.ndarray,
        regime_state: RegimeState,
        store_history: bool = True
    ) -> Dict[str, Any]:
        """Enhanced coverage validation with regime tracking.
        
        Args:
            y_true: True values
            predictions: Point predictions
            intervals: Prediction intervals
            regime_state: Current market regime
            store_history: Whether to store coverage history
            
        Returns:
            Comprehensive coverage statistics
        """
        try:
            # Calculate base coverage statistics
            coverage_stats = self.calculate_coverage(
                y_true, predictions, intervals, self.current_regime_alpha
            )
            
            # Add regime-specific information
            regime_info = {
                'regime': regime_state.regime.value,
                'regime_confidence': regime_state.confidence,
                'regime_duration': regime_state.duration,
                'adapted_alpha': self.current_regime_alpha,
                'base_alpha': self.regime_alpha_mapping.get(regime_state.regime.value, self.alpha)
            }
            
            coverage_stats.update(regime_info)
            
            # Calculate regime-specific metrics
            target_coverage = 1 - self.current_regime_alpha
            coverage_stats['regime_target_coverage'] = target_coverage
            coverage_stats['regime_coverage_error'] = abs(coverage_stats['empirical_coverage'] - target_coverage)
            
            # Calculate adaptive efficiency
            coverage_stats['adaptive_efficiency'] = self._calculate_adaptive_efficiency(
                coverage_stats['empirical_coverage'],
                target_coverage,
                coverage_stats['mean_interval_width']
            )
            
            # Store in regime-specific history
            if store_history:
                regime_name = regime_state.regime.value
                if regime_name in self.regime_coverage_history:
                    self.regime_coverage_history[regime_name].append(coverage_stats)
                    
                    # Keep only recent history
                    if len(self.regime_coverage_history[regime_name]) > 50:
                        self.regime_coverage_history[regime_name] = self.regime_coverage_history[regime_name][-50:]
            
            logger.info(f"Coverage validation for {regime_state.regime.value}: "
                       f"{coverage_stats['empirical_coverage']:.3f} (target: {target_coverage:.3f}, "
                       f"error: {coverage_stats['regime_coverage_error']:.3f})")
            
            return coverage_stats
            
        except Exception as e:
            logger.error(f"Error in coverage validation: {e}")
            return {'error': str(e)}
    
    def _calculate_adaptive_efficiency(self, empirical_coverage: float, target_coverage: float, mean_width: float) -> float:
        """Calculate efficiency metric for adaptive coverage."""
        try:
            # Penalize both coverage errors and wide intervals
            coverage_penalty = abs(empirical_coverage - target_coverage) * 2  # Double penalty for coverage errors
            width_penalty = mean_width * 0.5  # Penalty for wide intervals
            
            # Efficiency score (higher is better)
            efficiency = max(0.0, 1.0 - coverage_penalty - width_penalty)
            
            return efficiency
            
        except:
            return 0.0
    
    def get_regime_coverage_summary(self) -> Dict[str, Any]:
        """Get comprehensive coverage summary by regime."""
        summary = {
            'overall_summary': self.get_uncertainty_summary(),
            'regime_summaries': {},
            'adaptive_performance': {}
        }
        
        # Per-regime summaries
        for regime_name, history in self.regime_coverage_history.items():
            if not history:
                continue
                
            recent_history = history[-10:]  # Last 10 evaluations
            
            regime_summary = {
                'n_evaluations': len(history),
                'base_alpha': self.regime_alpha_mapping.get(regime_name, self.alpha),
                'recent_avg_coverage': np.mean([h['empirical_coverage'] for h in recent_history]),
                'recent_avg_width': np.mean([h['mean_interval_width'] for h in recent_history]),
                'recent_avg_efficiency': np.mean([h.get('adaptive_efficiency', 0) for h in recent_history]),
                'coverage_stability': np.std([h['empirical_coverage'] for h in recent_history])
            }
            
            # Check calibration quality
            target_coverage = 1 - regime_summary['base_alpha']
            coverage_error = abs(regime_summary['recent_avg_coverage'] - target_coverage)
            regime_summary['is_well_calibrated'] = coverage_error < 0.05
            regime_summary['calibration_error'] = coverage_error
            
            summary['regime_summaries'][regime_name] = regime_summary
        
        # Overall adaptive performance
        if any(self.regime_coverage_history.values()):
            all_recent = []
            for history in self.regime_coverage_history.values():
                all_recent.extend(history[-5:])  # Last 5 from each regime
            
            if all_recent:
                summary['adaptive_performance'] = {
                    'overall_avg_efficiency': np.mean([h.get('adaptive_efficiency', 0) for h in all_recent]),
                    'adaptation_consistency': np.std([h.get('adaptive_efficiency', 0) for h in all_recent]),
                    'total_evaluations': len(all_recent)
                }
        
        return summary