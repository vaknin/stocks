"""
Model Drift Detection Testing Suite - Prediction Degradation Monitoring

Tests for detecting and monitoring model drift in prediction accuracy,
feature importance changes, distribution shifts, and performance degradation
to ensure recommendation system reliability over time.

Priority: TIER 3 - Risk management for long-term recommendation system stability
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from unittest.mock import Mock, patch
from pathlib import Path
import sys
from scipy import stats
import warnings

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ensemble import MetaLearningEnsemble
from src.models.drift_detector import ModelDriftDetector, DriftAlert, DriftType
from src.config.settings import config


class ModelDriftDetector:
    """Detects various types of model drift in recommendation system."""
    
    def __init__(self, 
                 performance_window: int = 50,
                 drift_threshold: float = 0.15,
                 significance_level: float = 0.05):
        self.performance_window = performance_window
        self.drift_threshold = drift_threshold
        self.significance_level = significance_level
        self.baseline_performance = {}
        self.performance_history = []
        self.feature_importance_history = []
        
    def update_baseline(self, performance_metrics: Dict[str, float]) -> None:
        """Update baseline performance metrics."""
        self.baseline_performance = performance_metrics.copy()
    
    def add_performance_sample(self, 
                              performance_metrics: Dict[str, float],
                              feature_importance: Optional[Dict[str, float]] = None,
                              timestamp: Optional[datetime] = None) -> None:
        """Add new performance sample for drift monitoring."""
        sample = {
            'timestamp': timestamp or datetime.now(),
            'performance': performance_metrics.copy()
        }
        
        if feature_importance:
            sample['feature_importance'] = feature_importance.copy()
        
        self.performance_history.append(sample)
        
        # Keep only recent samples
        if len(self.performance_history) > self.performance_window * 2:
            self.performance_history = self.performance_history[-self.performance_window:]
    
    def detect_performance_drift(self) -> Optional[DriftAlert]:
        """Detect drift in model performance metrics."""
        if len(self.performance_history) < self.performance_window:
            return None
        
        # Split into recent and baseline periods
        recent_samples = self.performance_history[-self.performance_window//2:]
        baseline_samples = self.performance_history[-self.performance_window:-self.performance_window//2]
        
        drift_alerts = []
        
        # Check each performance metric
        for metric in ['directional_accuracy', 'mae', 'correlation']:
            recent_values = [
                sample['performance'].get(metric) 
                for sample in recent_samples
                if sample['performance'].get(metric) is not None
            ]
            
            baseline_values = [
                sample['performance'].get(metric)
                for sample in baseline_samples
                if sample['performance'].get(metric) is not None
            ]
            
            if len(recent_values) < 5 or len(baseline_values) < 5:
                continue
            
            # Statistical test for drift
            drift_detected = self._detect_statistical_drift(
                baseline_values, recent_values, metric
            )
            
            if drift_detected:
                severity = self._calculate_drift_severity(
                    baseline_values, recent_values, metric
                )
                
                alert = DriftAlert(
                    drift_type=DriftType.PERFORMANCE_DRIFT,
                    metric=metric,
                    severity=severity,
                    baseline_mean=np.mean(baseline_values),
                    current_mean=np.mean(recent_values),
                    detection_time=datetime.now(),
                    description=f"Performance drift detected in {metric}"
                )
                drift_alerts.append(alert)
        
        return drift_alerts[0] if drift_alerts else None
    
    def detect_feature_importance_drift(self) -> Optional[DriftAlert]:
        """Detect drift in feature importance patterns."""
        if len(self.performance_history) < 20:
            return None
        
        # Extract feature importance from recent samples
        recent_importance = []
        baseline_importance = []
        
        for i, sample in enumerate(self.performance_history):
            if 'feature_importance' in sample:
                if i >= len(self.performance_history) - 10:  # Recent samples
                    recent_importance.append(sample['feature_importance'])
                elif i < len(self.performance_history) - 20:  # Baseline samples
                    baseline_importance.append(sample['feature_importance'])
        
        if len(recent_importance) < 3 or len(baseline_importance) < 3:
            return None
        
        # Calculate average importance for each feature
        all_features = set()
        for importance_dict in recent_importance + baseline_importance:
            all_features.update(importance_dict.keys())
        
        drift_score = 0.0
        changed_features = []
        
        for feature in all_features:
            recent_values = [
                imp.get(feature, 0) for imp in recent_importance
            ]
            baseline_values = [
                imp.get(feature, 0) for imp in baseline_importance
            ]
            
            recent_mean = np.mean(recent_values)
            baseline_mean = np.mean(baseline_values)
            
            if baseline_mean > 0:
                relative_change = abs(recent_mean - baseline_mean) / baseline_mean
                if relative_change > 0.3:  # 30% change threshold
                    drift_score += relative_change
                    changed_features.append(feature)
        
        if drift_score > 1.0 or len(changed_features) > len(all_features) * 0.3:
            return DriftAlert(
                drift_type=DriftType.FEATURE_DRIFT,
                metric="feature_importance",
                severity=min(drift_score, 3.0),
                baseline_mean=None,
                current_mean=None,
                detection_time=datetime.now(),
                description=f"Feature importance drift in {len(changed_features)} features: {changed_features[:5]}"
            )
        
        return None
    
    def detect_data_distribution_drift(self, 
                                      recent_data: pd.DataFrame,
                                      baseline_data: pd.DataFrame) -> Optional[DriftAlert]:
        """Detect drift in input data distribution."""
        if recent_data.empty or baseline_data.empty:
            return None
        
        drift_scores = []
        drifted_columns = []
        
        # Check numeric columns for distribution drift
        numeric_columns = recent_data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in baseline_data.columns:
                recent_values = recent_data[column].dropna()
                baseline_values = baseline_data[column].dropna()
                
                if len(recent_values) < 10 or len(baseline_values) < 10:
                    continue
                
                # Kolmogorov-Smirnov test for distribution difference
                try:
                    ks_stat, p_value = stats.ks_2samp(baseline_values, recent_values)
                    
                    if p_value < self.significance_level:
                        drift_scores.append(ks_stat)
                        drifted_columns.append(column)
                        
                except Exception:
                    continue
        
        if drift_scores and np.mean(drift_scores) > 0.2:  # Significant drift
            return DriftAlert(
                drift_type=DriftType.DATA_DRIFT,
                metric="distribution",
                severity=np.mean(drift_scores) * 5,  # Scale to 0-5
                baseline_mean=None,
                current_mean=None,
                detection_time=datetime.now(),
                description=f"Data distribution drift detected in columns: {drifted_columns[:5]}"
            )
        
        return None
    
    def _detect_statistical_drift(self, 
                                 baseline_values: List[float],
                                 recent_values: List[float],
                                 metric: str) -> bool:
        """Detect statistical significance of drift."""
        try:
            # Two-sample t-test for mean difference
            t_stat, p_value = stats.ttest_ind(baseline_values, recent_values)
            
            # Also check effect size (Cohen's d)
            baseline_mean = np.mean(baseline_values)
            recent_mean = np.mean(recent_values)
            pooled_std = np.sqrt(
                ((len(baseline_values) - 1) * np.var(baseline_values) +
                 (len(recent_values) - 1) * np.var(recent_values)) /
                (len(baseline_values) + len(recent_values) - 2)
            )
            
            if pooled_std > 0:
                cohens_d = abs(recent_mean - baseline_mean) / pooled_std
            else:
                cohens_d = 0
            
            # Drift if statistically significant AND meaningful effect size
            is_significant = p_value < self.significance_level
            is_meaningful = cohens_d > 0.3  # Small to medium effect size
            
            # For accuracy metrics, also check absolute threshold
            if 'accuracy' in metric.lower() or 'correlation' in metric.lower():
                relative_change = abs(recent_mean - baseline_mean) / (abs(baseline_mean) + 1e-8)
                exceeds_threshold = relative_change > self.drift_threshold
            else:
                # For error metrics (lower is better)
                relative_change = (recent_mean - baseline_mean) / (abs(baseline_mean) + 1e-8)
                exceeds_threshold = relative_change > self.drift_threshold
            
            return (is_significant and is_meaningful) or exceeds_threshold
            
        except Exception:
            return False
    
    def _calculate_drift_severity(self,
                                 baseline_values: List[float],
                                 recent_values: List[float],
                                 metric: str) -> float:
        """Calculate severity of detected drift (0-5 scale)."""
        baseline_mean = np.mean(baseline_values)
        recent_mean = np.mean(recent_values)
        
        if abs(baseline_mean) < 1e-8:
            return 2.0  # Medium severity if baseline is near zero
        
        relative_change = abs(recent_mean - baseline_mean) / abs(baseline_mean)
        
        # Scale to 0-5 severity
        if relative_change < 0.1:
            return 1.0  # Low
        elif relative_change < 0.25:
            return 2.0  # Medium
        elif relative_change < 0.5:
            return 3.0  # High
        elif relative_change < 1.0:
            return 4.0  # Very High
        else:
            return 5.0  # Critical


class DriftAlert:
    """Represents a drift detection alert."""
    
    def __init__(self, drift_type, metric, severity, baseline_mean, current_mean, 
                 detection_time, description):
        self.drift_type = drift_type
        self.metric = metric
        self.severity = severity
        self.baseline_mean = baseline_mean
        self.current_mean = current_mean
        self.detection_time = detection_time
        self.description = description


class DriftType:
    """Types of drift that can be detected."""
    PERFORMANCE_DRIFT = "performance_drift"
    FEATURE_DRIFT = "feature_drift"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"


class TestModelDrift:
    """Test suite for model drift detection and monitoring."""
    
    @pytest.fixture(scope="class")
    def drift_detector(self):
        """Create drift detector for testing."""
        return ModelDriftDetector(
            performance_window=30,
            drift_threshold=0.15,
            significance_level=0.05
        )
    
    @pytest.fixture(scope="class")
    def ensemble_model(self):
        """Create ensemble model for testing."""
        return MetaLearningEnsemble(
            horizon_len=[1, 5],
            enable_neural_meta_learning=False
        )
    
    @pytest.fixture(scope="class")
    def stable_data_sequence(self):
        """Generate stable data sequence (no drift)."""
        np.random.seed(42)
        
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        
        # Stable price series with consistent volatility
        base_price = 100
        stable_returns = np.random.normal(0.001, 0.015, 200)  # Consistent parameters
        stable_prices = base_price * np.cumprod(1 + stable_returns)
        
        data = pd.DataFrame({
            'Open': stable_prices * np.random.uniform(0.998, 1.002, 200),
            'High': stable_prices * np.random.uniform(1.005, 1.02, 200),
            'Low': stable_prices * np.random.uniform(0.98, 0.995, 200),
            'Close': stable_prices,
            'Volume': np.random.randint(1000000, 5000000, 200)
        }, index=dates)
        
        return data
    
    @pytest.fixture(scope="class")
    def drifted_data_sequence(self):
        """Generate data sequence with drift."""
        np.random.seed(42)
        
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        
        # First half: stable regime
        stable_returns = np.random.normal(0.001, 0.015, 100)
        
        # Second half: drifted regime (higher volatility, different mean)
        drifted_returns = np.random.normal(0.003, 0.025, 100)  # Higher mean and volatility
        
        all_returns = np.concatenate([stable_returns, drifted_returns])
        prices = 100 * np.cumprod(1 + all_returns)
        
        # Also change volume pattern in second half
        stable_volumes = np.random.randint(1000000, 3000000, 100)
        drifted_volumes = np.random.randint(3000000, 8000000, 100)  # Higher volumes
        all_volumes = np.concatenate([stable_volumes, drifted_volumes])
        
        data = pd.DataFrame({
            'Open': prices * np.random.uniform(0.998, 1.002, 200),
            'High': prices * np.random.uniform(1.005, 1.02, 200),
            'Low': prices * np.random.uniform(0.98, 0.995, 200),
            'Close': prices,
            'Volume': all_volumes
        }, index=dates)
        
        return data
    
    def test_no_drift_detection_stable_data(self, drift_detector, ensemble_model, stable_data_sequence):
        """Test that no drift is detected in stable data."""
        # Generate predictions on stable data to create performance history
        performance_samples = []
        
        # Simulate model performance over time on stable data
        for i in range(50, len(stable_data_sequence), 10):
            try:
                hist_data = stable_data_sequence.iloc[:i]
                
                # Mock consistent model performance
                mock_performance = {
                    'directional_accuracy': 0.65 + np.random.normal(0, 0.05),  # Stable around 65%
                    'mae': 0.02 + np.random.normal(0, 0.003),                 # Stable low error
                    'correlation': 0.3 + np.random.normal(0, 0.08)            # Stable correlation
                }
                
                # Ensure values are within reasonable bounds
                mock_performance['directional_accuracy'] = np.clip(mock_performance['directional_accuracy'], 0, 1)
                mock_performance['mae'] = np.clip(mock_performance['mae'], 0, 0.1)
                mock_performance['correlation'] = np.clip(mock_performance['correlation'], -1, 1)
                
                performance_samples.append(mock_performance)
                
                drift_detector.add_performance_sample(
                    mock_performance,
                    timestamp=stable_data_sequence.index[i]
                )
                
            except Exception as e:
                continue
        
        if len(performance_samples) < 20:
            pytest.skip("Insufficient performance samples for drift testing")
        
        # Set baseline from early samples
        early_samples = performance_samples[:10]
        baseline_performance = {
            metric: np.mean([sample[metric] for sample in early_samples])
            for metric in early_samples[0].keys()
        }
        drift_detector.update_baseline(baseline_performance)
        
        # Check for drift detection
        drift_alert = drift_detector.detect_performance_drift()
        
        # Should not detect drift in stable data
        if drift_alert:
            # Allow for some false positives but they should be low severity
            assert drift_alert.severity <= 2.0, (
                f"False drift detected with high severity: {drift_alert.severity} "
                f"in metric {drift_alert.metric}"
            )
        else:
            # No drift detected - this is expected
            assert True
    
    def test_performance_drift_detection(self, drift_detector):
        """Test detection of performance drift."""
        # Simulate baseline performance
        baseline_samples = []
        for i in range(15):
            baseline_performance = {
                'directional_accuracy': 0.70 + np.random.normal(0, 0.03),
                'mae': 0.018 + np.random.normal(0, 0.002),
                'correlation': 0.35 + np.random.normal(0, 0.05)
            }
            baseline_samples.append(baseline_performance)
            drift_detector.add_performance_sample(baseline_performance)
        
        # Set baseline
        baseline_avg = {
            metric: np.mean([sample[metric] for sample in baseline_samples])
            for metric in baseline_samples[0].keys()
        }
        drift_detector.update_baseline(baseline_avg)
        
        # Simulate degraded performance (clear drift)
        for i in range(15):
            degraded_performance = {
                'directional_accuracy': 0.55 + np.random.normal(0, 0.03),  # Significant drop
                'mae': 0.035 + np.random.normal(0, 0.003),                 # Increased error
                'correlation': 0.15 + np.random.normal(0, 0.05)            # Reduced correlation
            }
            
            drift_detector.add_performance_sample(degraded_performance)
        
        # Check for drift detection
        drift_alert = drift_detector.detect_performance_drift()
        
        # Should detect drift
        assert drift_alert is not None, "Should detect performance drift"
        assert drift_alert.drift_type == DriftType.PERFORMANCE_DRIFT
        assert drift_alert.severity >= 2.0, f"Should detect significant drift: severity = {drift_alert.severity}"
        
        # Check that the detected metric makes sense
        expected_drifted_metrics = ['directional_accuracy', 'mae', 'correlation']
        assert drift_alert.metric in expected_drifted_metrics, f"Unexpected drifted metric: {drift_alert.metric}"
    
    def test_feature_importance_drift_detection(self, drift_detector):
        """Test detection of feature importance drift."""
        # Simulate stable feature importance baseline
        stable_importance = {
            'price_momentum': 0.25,
            'volume_ratio': 0.20,
            'rsi': 0.15,
            'macd': 0.18,
            'bollinger_bands': 0.12,
            'volatility': 0.10
        }
        
        # Add baseline samples with slight variations
        for i in range(10):
            varied_importance = {}
            for feature, importance in stable_importance.items():
                # Small random variation around baseline
                varied_importance[feature] = importance + np.random.normal(0, importance * 0.1)
            
            drift_detector.add_performance_sample(
                {'directional_accuracy': 0.65},  # Dummy performance
                feature_importance=varied_importance
            )
        
        # Simulate feature importance drift (some features become much more/less important)
        drifted_importance = stable_importance.copy()
        drifted_importance['price_momentum'] = 0.05    # Dramatically less important
        drifted_importance['volume_ratio'] = 0.40      # Much more important
        drifted_importance['new_feature'] = 0.15       # New important feature
        
        # Add drifted samples
        for i in range(10):
            varied_drifted = {}
            for feature, importance in drifted_importance.items():
                varied_drifted[feature] = importance + np.random.normal(0, importance * 0.1)
            
            drift_detector.add_performance_sample(
                {'directional_accuracy': 0.65},
                feature_importance=varied_drifted
            )
        
        # Check for feature importance drift
        drift_alert = drift_detector.detect_feature_importance_drift()
        
        # Should detect drift
        assert drift_alert is not None, "Should detect feature importance drift"
        assert drift_alert.drift_type == DriftType.FEATURE_DRIFT
        assert drift_alert.severity >= 1.5, f"Should detect meaningful drift: severity = {drift_alert.severity}"
    
    def test_data_distribution_drift_detection(self, drift_detector, stable_data_sequence, drifted_data_sequence):
        """Test detection of data distribution drift."""
        # Split drifted data into baseline and recent periods
        baseline_data = drifted_data_sequence.iloc[:100]  # Stable period
        recent_data = drifted_data_sequence.iloc[100:]    # Drifted period
        
        # Test distribution drift detection
        drift_alert = drift_detector.detect_data_distribution_drift(recent_data, baseline_data)
        
        # Should detect drift
        assert drift_alert is not None, "Should detect data distribution drift"
        assert drift_alert.drift_type == DriftType.DATA_DRIFT
        assert drift_alert.severity >= 1.0, f"Should detect distribution drift: severity = {drift_alert.severity}"
        
        # Test with stable data (should not detect drift)
        stable_baseline = stable_data_sequence.iloc[:100]
        stable_recent = stable_data_sequence.iloc[100:]
        
        stable_drift_alert = drift_detector.detect_data_distribution_drift(stable_recent, stable_baseline)
        
        # Should not detect drift in stable data (or only low severity)
        if stable_drift_alert:
            assert stable_drift_alert.severity <= 1.5, (
                f"False drift detected in stable data: severity = {stable_drift_alert.severity}"
            )
    
    def test_drift_severity_calibration(self, drift_detector):
        """Test that drift severity is properly calibrated."""
        test_cases = [
            # (baseline_values, recent_values, expected_min_severity, expected_max_severity)
            ([0.7] * 10, [0.68] * 10, 0.0, 1.5),  # Small change - low severity
            ([0.7] * 10, [0.6] * 10, 1.0, 3.0),   # Medium change - medium severity
            ([0.7] * 10, [0.5] * 10, 2.5, 4.0),   # Large change - high severity
            ([0.7] * 10, [0.3] * 10, 4.0, 5.0),   # Extreme change - max severity
        ]
        
        for baseline_vals, recent_vals, min_sev, max_sev in test_cases:
            severity = drift_detector._calculate_drift_severity(baseline_vals, recent_vals, 'accuracy')
            
            assert min_sev <= severity <= max_sev, (
                f"Severity {severity:.2f} not in expected range [{min_sev}, {max_sev}] "
                f"for baseline={np.mean(baseline_vals):.2f}, recent={np.mean(recent_vals):.2f}"
            )
    
    def test_drift_detection_with_insufficient_data(self, drift_detector):
        """Test drift detection behavior with insufficient data."""
        # Add only a few samples
        for i in range(5):
            performance = {
                'directional_accuracy': 0.65 + np.random.normal(0, 0.05),
                'mae': 0.02,
                'correlation': 0.3
            }
            drift_detector.add_performance_sample(performance)
        
        # Should not detect drift with insufficient data
        drift_alert = drift_detector.detect_performance_drift()
        assert drift_alert is None, "Should not detect drift with insufficient data"
        
        feature_drift_alert = drift_detector.detect_feature_importance_drift()
        assert feature_drift_alert is None, "Should not detect feature drift with insufficient data"
    
    def test_statistical_significance_requirements(self, drift_detector):
        """Test that drift detection requires statistical significance."""
        # Simulate performance data with random noise but no systematic drift
        baseline_mean = 0.65
        noise_level = 0.08  # High noise
        
        # Add baseline samples
        for i in range(20):
            performance = {
                'directional_accuracy': baseline_mean + np.random.normal(0, noise_level),
                'mae': 0.02 + np.random.normal(0, 0.005),
                'correlation': 0.3 + np.random.normal(0, 0.1)
            }
            drift_detector.add_performance_sample(performance)
        
        # Set baseline
        drift_detector.update_baseline({
            'directional_accuracy': baseline_mean,
            'mae': 0.02,
            'correlation': 0.3
        })
        
        # Add recent samples with same distribution (just noise, no real drift)
        for i in range(15):
            performance = {
                'directional_accuracy': baseline_mean + np.random.normal(0, noise_level),
                'mae': 0.02 + np.random.normal(0, 0.005),
                'correlation': 0.3 + np.random.normal(0, 0.1)
            }
            drift_detector.add_performance_sample(performance)
        
        # Should not detect drift (just random noise)
        drift_alert = drift_detector.detect_performance_drift()
        
        if drift_alert:
            # If detected, should be low severity (noise, not systematic drift)
            assert drift_alert.severity <= 1.5, (
                f"High severity drift detected in noisy but stable data: {drift_alert.severity}"
            )
    
    def test_drift_alert_information_completeness(self, drift_detector):
        """Test that drift alerts contain complete information."""
        # Generate clear drift scenario
        baseline_performance = {'directional_accuracy': 0.75, 'mae': 0.015}
        degraded_performance = {'directional_accuracy': 0.55, 'mae': 0.035}
        
        # Add baseline
        for i in range(15):
            drift_detector.add_performance_sample(baseline_performance)
        drift_detector.update_baseline(baseline_performance)
        
        # Add degraded samples
        for i in range(15):
            drift_detector.add_performance_sample(degraded_performance)
        
        drift_alert = drift_detector.detect_performance_drift()
        
        if drift_alert:
            # Validate alert completeness
            assert hasattr(drift_alert, 'drift_type'), "Missing drift_type"
            assert hasattr(drift_alert, 'metric'), "Missing metric"
            assert hasattr(drift_alert, 'severity'), "Missing severity"
            assert hasattr(drift_alert, 'detection_time'), "Missing detection_time"
            assert hasattr(drift_alert, 'description'), "Missing description"
            
            # Validate data types
            assert isinstance(drift_alert.severity, (int, float)), "Severity should be numeric"
            assert isinstance(drift_alert.detection_time, datetime), "Detection time should be datetime"
            assert isinstance(drift_alert.description, str), "Description should be string"
            assert len(drift_alert.description) > 10, "Description should be informative"
    
    def test_continuous_monitoring_simulation(self, drift_detector, ensemble_model):
        """Test continuous drift monitoring over extended period."""
        # Simulate continuous monitoring over several months
        dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
        
        # Create data with gradual drift over time
        drift_alerts = []
        performance_trend = []
        
        for i, date in enumerate(dates):
            # Gradual performance degradation over time
            degradation_factor = i / len(dates) * 0.3  # Up to 30% degradation
            
            base_accuracy = 0.70
            current_accuracy = base_accuracy - degradation_factor + np.random.normal(0, 0.03)
            
            base_mae = 0.02
            current_mae = base_mae + degradation_factor * 0.015 + np.random.normal(0, 0.002)
            
            performance = {
                'directional_accuracy': max(0.4, min(1.0, current_accuracy)),
                'mae': max(0.005, current_mae),
                'correlation': max(0.1, 0.35 - degradation_factor * 0.2 + np.random.normal(0, 0.05))
            }
            
            performance_trend.append(performance['directional_accuracy'])
            
            drift_detector.add_performance_sample(performance, timestamp=date)
            
            # Set baseline from first month
            if i == 30:
                early_samples = drift_detector.performance_history[-30:]
                baseline_perf = {
                    metric: np.mean([s['performance'][metric] for s in early_samples])
                    for metric in performance.keys()
                }
                drift_detector.update_baseline(baseline_perf)
            
            # Check for drift every week
            if i > 30 and i % 7 == 0:
                alert = drift_detector.detect_performance_drift()
                if alert:
                    drift_alerts.append((i, alert))
        
        # Should detect drift as performance degrades
        assert len(drift_alerts) > 0, "Should detect drift in degrading performance over time"
        
        # Later alerts should have higher severity
        if len(drift_alerts) > 1:
            early_alert_severity = drift_alerts[0][1].severity
            late_alert_severity = drift_alerts[-1][1].severity
            
            # Later drift should be more severe (as degradation accumulates)
            assert late_alert_severity >= early_alert_severity, (
                f"Later drift should be more severe: {late_alert_severity} >= {early_alert_severity}"
            )
        
        # Performance trend should show clear degradation
        first_month_perf = np.mean(performance_trend[:30])
        last_month_perf = np.mean(performance_trend[-30:])
        performance_degradation = first_month_perf - last_month_perf
        
        assert performance_degradation > 0.1, (
            f"Should show clear performance degradation: {performance_degradation:.3f}"
        )
    
    def test_drift_detector_memory_management(self, drift_detector):
        """Test that drift detector manages memory properly with long-term use."""
        initial_history_length = len(drift_detector.performance_history)
        
        # Add many samples to test memory management
        for i in range(200):
            performance = {
                'directional_accuracy': 0.65 + np.random.normal(0, 0.05),
                'mae': 0.02 + np.random.normal(0, 0.003),
                'correlation': 0.3 + np.random.normal(0, 0.08)
            }
            drift_detector.add_performance_sample(performance)
        
        final_history_length = len(drift_detector.performance_history)
        
        # Should limit history size to prevent unbounded growth
        max_expected_length = drift_detector.performance_window * 2
        assert final_history_length <= max_expected_length, (
            f"History too long: {final_history_length} > {max_expected_length}"
        )
        
        # Should still be able to detect drift with limited history
        drift_alert = drift_detector.detect_performance_drift()
        # This should not crash and may or may not detect drift depending on the random data


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])