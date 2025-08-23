"""MAPIE uncertainty quantification for trading predictions."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any, Callable
from datetime import datetime
import warnings
from loguru import logger

try:
    from mapie.regression import MapieRegressor
    from mapie.time_series_regression import MapieTimeSeriesRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False
    logger.warning("MAPIE not available. Running in mock mode for uncertainty quantification.")

from ..config.settings import config

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
        max_interval_width: float = 0.03,
        confidence_threshold: float = 0.75,
        min_prediction_magnitude: float = 0.005
    ):
        """Initialize uncertainty filter for trading signals.
        
        Args:
            max_interval_width: Maximum prediction interval width (3%)
            confidence_threshold: Minimum confidence for trade signals
            min_prediction_magnitude: Minimum prediction magnitude for signals
        """
        self.max_interval_width = max_interval_width
        self.confidence_threshold = confidence_threshold
        self.min_prediction_magnitude = min_prediction_magnitude
        
        logger.info(f"TradingUncertaintyFilter initialized with thresholds: "
                   f"width<{max_interval_width:.1%}, conf>{confidence_threshold:.1%}")
    
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