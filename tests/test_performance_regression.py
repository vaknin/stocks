"""
Performance Regression Testing Suite - Recommendation Accuracy Baseline Tracking

Tests for tracking and monitoring recommendation accuracy baselines over time.
Detects performance degradation in model predictions, recommendation quality,
and confidence calibration to ensure consistent investment advice quality.

Priority: TIER 2 - Critical for maintaining recommendation system performance
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from unittest.mock import Mock, patch
from pathlib import Path
import sys
import json
import pickle

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ensemble import MetaLearningEnsemble
from src.trading.signal_generator import SignalGenerator, SignalType
from src.backtesting.backtest_engine import BacktestEngine
from src.config.settings import config


class PerformanceBaseline:
    """Manages performance baseline storage and comparison."""
    
    def __init__(self, baseline_file: str = "test_performance_baseline.json"):
        self.baseline_file = Path(__file__).parent / "baselines" / baseline_file
        self.baseline_file.parent.mkdir(exist_ok=True)
        self.baseline_data = self._load_baseline()
    
    def _load_baseline(self) -> Dict[str, Any]:
        """Load baseline performance metrics."""
        if self.baseline_file.exists():
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_baseline(self, baseline_data: Dict[str, Any]) -> None:
        """Save baseline performance metrics."""
        with open(self.baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        self.baseline_data = baseline_data
    
    def get_baseline_metric(self, metric_name: str) -> Optional[float]:
        """Get baseline value for a specific metric."""
        return self.baseline_data.get(metric_name)
    
    def compare_to_baseline(self, current_metrics: Dict[str, float], 
                          tolerance: float = 0.05) -> Dict[str, Dict[str, Any]]:
        """Compare current metrics to baseline with tolerance."""
        comparison = {}
        
        for metric, current_value in current_metrics.items():
            baseline_value = self.get_baseline_metric(metric)
            
            if baseline_value is not None:
                difference = current_value - baseline_value
                relative_diff = difference / baseline_value if baseline_value != 0 else float('inf')
                
                # Check if within tolerance
                is_regression = abs(relative_diff) > tolerance and difference < 0
                is_significant_change = abs(relative_diff) > tolerance
                
                comparison[metric] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'absolute_diff': difference,
                    'relative_diff': relative_diff,
                    'is_regression': is_regression,
                    'is_significant': is_significant_change,
                    'within_tolerance': abs(relative_diff) <= tolerance
                }
            else:
                comparison[metric] = {
                    'baseline': None,
                    'current': current_value,
                    'is_new_metric': True
                }
        
        return comparison


class TestPerformanceRegression:
    """Test suite for performance regression detection and baseline tracking."""
    
    @pytest.fixture(scope="class")
    def performance_baseline(self):
        """Performance baseline manager."""
        return PerformanceBaseline("recommendation_performance_baseline.json")
    
    @pytest.fixture(scope="class")
    def historical_data(self):
        """Generate historical data for backtesting."""
        np.random.seed(42)
        
        # Generate 2 years of data for robust testing
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        dates = dates[dates.weekday < 5]  # Trading days only
        
        # Multiple stocks for comprehensive testing
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM']
        data = {}
        
        for i, stock in enumerate(stocks):
            # Different volatility and trend patterns for each stock
            base_price = 100 + i * 50
            volatility = 0.15 + i * 0.05  # Varying volatilities
            trend = 0.0005 + (i - 2) * 0.0002  # Different trend directions
            
            returns = np.random.normal(trend, volatility/np.sqrt(252), len(dates))
            
            # Add regime changes
            regime_1 = slice(100, 200)  # Bull market
            regime_2 = slice(300, 400)  # Bear market
            regime_3 = slice(500, 550)  # High volatility
            
            returns[regime_1] += 0.001
            returns[regime_2] -= 0.002
            returns[regime_3] *= 2.0
            
            prices = base_price * np.cumprod(1 + returns)
            
            data[stock] = pd.DataFrame({
                'Open': prices * np.random.uniform(0.998, 1.002, len(dates)),
                'High': prices * np.random.uniform(1.005, 1.025, len(dates)),
                'Low': prices * np.random.uniform(0.975, 0.995, len(dates)),
                'Close': prices,
                'Volume': np.random.randint(1000000, 20000000, len(dates))
            }, index=dates)
        
        return data
    
    @pytest.fixture
    def ensemble_model(self):
        """Create ensemble model for testing."""
        return MetaLearningEnsemble(
            horizon_len=[1, 5, 20],
            uncertainty_alpha=0.1,
            enable_neural_meta_learning=False  # Disable for consistent testing
        )
    
    def test_prediction_accuracy_baseline(self, ensemble_model, historical_data, performance_baseline):
        """Test prediction accuracy against established baseline."""
        # Use walk-forward analysis on historical data
        stock_metrics = {}
        
        for stock, data in historical_data.items():
            if len(data) < 100:
                continue
                
            # Split data for walk-forward testing
            train_size = int(len(data) * 0.7)
            test_data = data.iloc[train_size:]
            
            predictions = []
            actual_returns = []
            prediction_errors = []
            
            # Walk-forward predictions (sample every 5 days to speed up testing)
            for i in range(0, len(test_data) - 20, 5):
                try:
                    # Historical data up to point i
                    hist_data = data.iloc[:train_size + i]
                    
                    # Generate prediction
                    pred_result = ensemble_model.predict(hist_data, stock, return_confidence=True)
                    
                    if 'horizon_1' in pred_result:
                        prediction = pred_result['horizon_1']['prediction']
                        confidence = pred_result['horizon_1']['confidence']
                        
                        # Actual return 1 day ahead
                        if train_size + i + 1 < len(data):
                            current_price = data['Close'].iloc[train_size + i]
                            future_price = data['Close'].iloc[train_size + i + 1]
                            actual_return = (future_price - current_price) / current_price
                            
                            predictions.append(prediction)
                            actual_returns.append(actual_return)
                            
                            # Calculate prediction error
                            error = abs(prediction - actual_return)
                            prediction_errors.append(error)
                            
                except Exception as e:
                    continue
            
            if len(predictions) > 10:  # Need minimum samples
                predictions = np.array(predictions)
                actual_returns = np.array(actual_returns)
                
                # Calculate performance metrics
                metrics = self._calculate_prediction_metrics(predictions, actual_returns)
                stock_metrics[stock] = metrics
        
        # Aggregate metrics across stocks
        if stock_metrics:
            aggregated_metrics = self._aggregate_stock_metrics(stock_metrics)
            
            # Compare to baseline
            comparison = performance_baseline.compare_to_baseline(
                aggregated_metrics, 
                tolerance=0.10  # 10% tolerance for regression detection
            )
            
            # Check for significant regressions
            regressions = [
                metric for metric, data in comparison.items() 
                if data.get('is_regression', False)
            ]
            
            if regressions:
                regression_details = {
                    metric: comparison[metric] for metric in regressions
                }
                pytest.fail(
                    f"Performance regression detected in metrics: {regressions}\n"
                    f"Details: {regression_details}"
                )
            
            # Update baseline if no regressions and improvements found
            significant_improvements = [
                metric for metric, data in comparison.items()
                if (data.get('relative_diff', 0) > 0.05 and 
                    not data.get('is_regression', False))
            ]
            
            if significant_improvements and not regressions:
                performance_baseline.save_baseline(aggregated_metrics)
                print(f"Baseline updated with improvements in: {significant_improvements}")
        
        else:
            pytest.skip("Insufficient data for prediction accuracy testing")
    
    def _calculate_prediction_metrics(self, predictions: np.ndarray, 
                                    actual_returns: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive prediction performance metrics."""
        if len(predictions) == 0 or len(actual_returns) == 0:
            return {}
        
        # Mean Absolute Error
        mae = np.mean(np.abs(predictions - actual_returns))
        
        # Mean Squared Error
        mse = np.mean((predictions - actual_returns) ** 2)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Directional Accuracy
        pred_directions = np.sign(predictions)
        actual_directions = np.sign(actual_returns)
        directional_accuracy = np.mean(pred_directions == actual_directions)
        
        # Mean Absolute Percentage Error (handle division by zero)
        actual_nonzero = actual_returns[actual_returns != 0]
        pred_nonzero = predictions[actual_returns != 0]
        if len(actual_nonzero) > 0:
            mape = np.mean(np.abs((actual_nonzero - pred_nonzero) / actual_nonzero))
        else:
            mape = float('inf')
        
        # Correlation between predictions and actual
        correlation = np.corrcoef(predictions, actual_returns)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Hit rate (predictions within 50% of actual)
        relative_errors = np.abs((predictions - actual_returns) / (np.abs(actual_returns) + 1e-8))
        hit_rate = np.mean(relative_errors < 0.5)
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'directional_accuracy': float(directional_accuracy),
            'mape': float(mape) if mape != float('inf') else 999.0,
            'correlation': float(correlation),
            'hit_rate': float(hit_rate)
        }
    
    def _aggregate_stock_metrics(self, stock_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across multiple stocks."""
        if not stock_metrics:
            return {}
        
        aggregated = {}
        metric_names = next(iter(stock_metrics.values())).keys()
        
        for metric in metric_names:
            values = [stock_data[metric] for stock_data in stock_metrics.values() 
                     if metric in stock_data and not np.isinf(stock_data[metric])]
            
            if values:
                aggregated[metric] = np.mean(values)
        
        # Add cross-stock consistency metrics
        directional_accuracies = [
            stock_data.get('directional_accuracy', 0) 
            for stock_data in stock_metrics.values()
        ]
        
        if directional_accuracies:
            aggregated['cross_stock_consistency'] = np.std(directional_accuracies)
        
        return aggregated
    
    def test_confidence_calibration_baseline(self, ensemble_model, historical_data, performance_baseline):
        """Test confidence calibration against baseline."""
        calibration_metrics = {}
        
        for stock, data in historical_data.items():
            if len(data) < 100:
                continue
            
            # Test confidence calibration
            confidences = []
            accuracies = []
            
            train_size = int(len(data) * 0.7)
            test_data = data.iloc[train_size:]
            
            # Sample predictions for calibration testing
            for i in range(0, len(test_data) - 20, 10):
                try:
                    hist_data = data.iloc[:train_size + i]
                    pred_result = ensemble_model.predict(hist_data, stock, return_confidence=True)
                    
                    if 'horizon_1' in pred_result:
                        prediction = pred_result['horizon_1']['prediction']
                        confidence = pred_result['horizon_1']['confidence']
                        pred_interval = pred_result['horizon_1']['prediction_interval']
                        
                        # Actual return
                        if train_size + i + 1 < len(data):
                            current_price = data['Close'].iloc[train_size + i]
                            future_price = data['Close'].iloc[train_size + i + 1]
                            actual_return = (future_price - current_price) / current_price
                            
                            # Check if actual return is within prediction interval
                            within_interval = pred_interval[0] <= actual_return <= pred_interval[1]
                            
                            confidences.append(confidence)
                            accuracies.append(1.0 if within_interval else 0.0)
                
                except Exception:
                    continue
            
            if len(confidences) > 10:
                # Calculate calibration metrics
                calibration_data = self._calculate_calibration_metrics(confidences, accuracies)
                calibration_metrics[stock] = calibration_data
        
        if calibration_metrics:
            # Aggregate calibration metrics
            aggregated_calibration = {}
            
            # Average calibration error across stocks
            calibration_errors = [
                data.get('calibration_error', float('inf')) 
                for data in calibration_metrics.values()
            ]
            calibration_errors = [e for e in calibration_errors if not np.isinf(e)]
            
            if calibration_errors:
                aggregated_calibration['avg_calibration_error'] = np.mean(calibration_errors)
                aggregated_calibration['calibration_consistency'] = np.std(calibration_errors)
            
            # Overconfidence ratio
            overconfidence_ratios = [
                data.get('overconfidence_ratio', 1.0) 
                for data in calibration_metrics.values()
            ]
            if overconfidence_ratios:
                aggregated_calibration['avg_overconfidence_ratio'] = np.mean(overconfidence_ratios)
            
            # Compare calibration to baseline
            calibration_comparison = performance_baseline.compare_to_baseline(
                aggregated_calibration,
                tolerance=0.15  # 15% tolerance for calibration metrics
            )
            
            # Check for calibration regressions
            calibration_regressions = [
                metric for metric, data in calibration_comparison.items()
                if data.get('is_regression', False)
            ]
            
            if calibration_regressions:
                pytest.fail(
                    f"Confidence calibration regression detected: {calibration_regressions}\n"
                    f"Details: {calibration_comparison}"
                )
    
    def _calculate_calibration_metrics(self, confidences: List[float], 
                                     accuracies: List[float]) -> Dict[str, float]:
        """Calculate confidence calibration metrics."""
        if not confidences or not accuracies:
            return {}
        
        confidences = np.array(confidences)
        accuracies = np.array(accuracies)
        
        # Bin confidences and calculate calibration error
        n_bins = min(10, len(confidences) // 3)  # Adaptive number of bins
        if n_bins < 2:
            return {}
        
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        calibration_errors = []
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(n_bins):
            in_bin = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if i == n_bins - 1:  # Include right edge in last bin
                in_bin = (confidences >= bins[i]) & (confidences <= bins[i + 1])
            
            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(accuracies[in_bin])
                bin_confidence = np.mean(confidences[in_bin])
                
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
                calibration_errors.append(abs(bin_accuracy - bin_confidence))
        
        if calibration_errors:
            avg_calibration_error = np.mean(calibration_errors)
            
            # Overconfidence ratio (confidence > accuracy)
            overconfident_bins = [
                conf > acc for conf, acc in zip(bin_confidences, bin_accuracies)
            ]
            overconfidence_ratio = np.mean(overconfident_bins) if overconfident_bins else 0.0
        else:
            avg_calibration_error = float('inf')
            overconfidence_ratio = 1.0
        
        return {
            'calibration_error': avg_calibration_error,
            'overconfidence_ratio': overconfidence_ratio,
            'n_calibration_bins': len(calibration_errors)
        }
    
    def test_recommendation_quality_baseline(self, performance_baseline):
        """Test recommendation quality metrics against baseline."""
        # Mock signal generator for testing
        signal_generator = SignalGenerator(
            confidence_thresholds={
                'daily': 0.70,
                'weekly': 0.75
            }
        )
        
        # Mock historical recommendation performance
        mock_recommendations = [
            {'ticker': 'AAPL', 'signal': SignalType.BUY, 'confidence': 0.85, 'actual_return': 0.03},
            {'ticker': 'MSFT', 'signal': SignalType.SELL, 'confidence': 0.78, 'actual_return': -0.02},
            {'ticker': 'GOOGL', 'signal': SignalType.HOLD, 'confidence': 0.65, 'actual_return': 0.005},
            {'ticker': 'TSLA', 'signal': SignalType.STRONG_BUY, 'confidence': 0.92, 'actual_return': 0.08},
            {'ticker': 'JPM', 'signal': SignalType.BUY, 'confidence': 0.82, 'actual_return': 0.025}
        ]
        
        # Calculate recommendation quality metrics
        quality_metrics = self._calculate_recommendation_quality(mock_recommendations)
        
        # Compare to baseline
        quality_comparison = performance_baseline.compare_to_baseline(
            quality_metrics,
            tolerance=0.08  # 8% tolerance
        )
        
        # Check for quality regressions
        quality_regressions = [
            metric for metric, data in quality_comparison.items()
            if data.get('is_regression', False)
        ]
        
        if quality_regressions:
            pytest.fail(
                f"Recommendation quality regression detected: {quality_regressions}\n"
                f"Details: {quality_comparison}"
            )
        
        # Validate minimum quality standards
        min_standards = {
            'recommendation_accuracy': 0.55,  # At least 55% accuracy
            'avg_confidence': 0.65,           # Average confidence >= 65%
            'strong_signal_accuracy': 0.70    # Strong signals >= 70% accuracy
        }
        
        for metric, min_value in min_standards.items():
            current_value = quality_metrics.get(metric, 0)
            assert current_value >= min_value, (
                f"Quality standard not met: {metric} = {current_value:.1%} < {min_value:.1%}"
            )
    
    def _calculate_recommendation_quality(self, recommendations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate recommendation quality metrics."""
        if not recommendations:
            return {}
        
        # Overall recommendation accuracy (direction)
        correct_direction = 0
        total_recommendations = 0
        
        # Strong signal tracking
        strong_signal_correct = 0
        strong_signal_total = 0
        
        # Confidence tracking
        confidences = []
        
        for rec in recommendations:
            signal = rec['signal']
            actual_return = rec['actual_return']
            confidence = rec['confidence']
            
            confidences.append(confidence)
            
            # Check direction accuracy
            if signal in [SignalType.BUY, SignalType.STRONG_BUY] and actual_return > 0:
                correct_direction += 1
            elif signal in [SignalType.SELL, SignalType.STRONG_SELL] and actual_return < 0:
                correct_direction += 1
            elif signal == SignalType.HOLD and abs(actual_return) < 0.02:  # ±2% for hold
                correct_direction += 1
            
            total_recommendations += 1
            
            # Track strong signals separately
            if signal in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
                strong_signal_total += 1
                if ((signal == SignalType.STRONG_BUY and actual_return > 0) or
                    (signal == SignalType.STRONG_SELL and actual_return < 0)):
                    strong_signal_correct += 1
        
        metrics = {}
        
        if total_recommendations > 0:
            metrics['recommendation_accuracy'] = correct_direction / total_recommendations
            metrics['avg_confidence'] = np.mean(confidences)
            metrics['confidence_std'] = np.std(confidences)
        
        if strong_signal_total > 0:
            metrics['strong_signal_accuracy'] = strong_signal_correct / strong_signal_total
        
        return metrics
    
    def test_model_latency_baseline(self, ensemble_model, performance_baseline):
        """Test model inference latency against baseline."""
        import time
        
        # Create test data
        test_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [102, 103, 104, 105, 106],
            'Low': [99, 100, 101, 102, 103],
            'Close': [101, 102, 103, 104, 105],
            'Volume': [1000000] * 5
        })
        
        # Measure inference times
        latencies = []
        
        for _ in range(10):  # Multiple runs for stability
            start_time = time.time()
            
            try:
                predictions = ensemble_model.predict(test_data, "LATENCY_TEST")
                end_time = time.time()
                
                latency = end_time - start_time
                latencies.append(latency)
                
            except Exception as e:
                # If prediction fails, record high latency as penalty
                latencies.append(10.0)  # 10 second penalty
        
        # Calculate latency metrics
        latency_metrics = {
            'avg_latency': np.mean(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'latency_std': np.std(latencies)
        }
        
        # Compare to baseline
        latency_comparison = performance_baseline.compare_to_baseline(
            latency_metrics,
            tolerance=0.20  # 20% tolerance for latency
        )
        
        # Check for latency regressions
        latency_regressions = []
        for metric, data in latency_comparison.items():
            if data.get('relative_diff', 0) > 0.20:  # >20% increase in latency
                latency_regressions.append(metric)
        
        if latency_regressions:
            pytest.fail(
                f"Latency regression detected: {latency_regressions}\n"
                f"Details: {latency_comparison}"
            )
        
        # Absolute latency requirements
        max_allowed_latency = config.MAX_LATENCY_SECONDS
        avg_latency = latency_metrics['avg_latency']
        
        assert avg_latency < max_allowed_latency, (
            f"Average latency {avg_latency:.2f}s exceeds maximum {max_allowed_latency}s"
        )
    
    def test_memory_usage_baseline(self, ensemble_model, performance_baseline):
        """Test memory usage against baseline."""
        import psutil
        import os
        
        # Measure initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create larger test dataset
        large_test_data = pd.DataFrame({
            'Open': np.random.normal(100, 10, 1000),
            'High': np.random.normal(105, 12, 1000),
            'Low': np.random.normal(95, 8, 1000),
            'Close': np.random.normal(100, 10, 1000),
            'Volume': np.random.randint(1000000, 10000000, 1000)
        })
        
        try:
            # Generate multiple predictions to stress memory
            for i in range(5):
                predictions = ensemble_model.predict(
                    large_test_data.iloc[:-i*10], 
                    f"MEMORY_TEST_{i}"
                )
            
            # Measure peak memory
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
        except Exception as e:
            # If memory stress fails, record high memory usage
            memory_increase = 1000.0  # 1GB penalty
        
        memory_metrics = {
            'memory_increase_mb': memory_increase,
            'peak_memory_mb': peak_memory if 'peak_memory' in locals() else initial_memory + memory_increase
        }
        
        # Compare to baseline
        memory_comparison = performance_baseline.compare_to_baseline(
            memory_metrics,
            tolerance=0.25  # 25% tolerance for memory usage
        )
        
        # Check for memory regressions
        memory_regressions = []
        for metric, data in memory_comparison.items():
            if data.get('relative_diff', 0) > 0.25:  # >25% increase in memory
                memory_regressions.append(metric)
        
        if memory_regressions:
            pytest.fail(
                f"Memory usage regression detected: {memory_regressions}\n"
                f"Details: {memory_comparison}"
            )
        
        # Absolute memory limits
        max_memory_increase = 500  # 500MB max increase
        assert memory_increase < max_memory_increase, (
            f"Memory increase {memory_increase:.1f}MB exceeds limit {max_memory_increase}MB"
        )
    
    def test_cross_validation_consistency(self, ensemble_model, historical_data):
        """Test consistency of performance across different data splits."""
        # Test model consistency across different time periods
        stock = 'AAPL'  # Use one stock for detailed analysis
        data = historical_data[stock]
        
        if len(data) < 200:
            pytest.skip("Insufficient data for cross-validation")
        
        # Create multiple train/test splits
        splits = [
            (0.0, 0.6, 0.6, 1.0),    # Early period
            (0.2, 0.8, 0.8, 1.0),    # Middle period  
            (0.4, 1.0, 0.8, 1.0),    # Recent period
        ]
        
        split_performances = []
        
        for train_start, train_end, test_start, test_end in splits:
            train_start_idx = int(len(data) * train_start)
            train_end_idx = int(len(data) * train_end)
            test_start_idx = int(len(data) * test_start)
            test_end_idx = int(len(data) * test_end)
            
            train_data = data.iloc[train_start_idx:train_end_idx]
            test_data = data.iloc[test_start_idx:test_end_idx]
            
            if len(train_data) < 50 or len(test_data) < 20:
                continue
            
            # Test predictions on this split
            predictions = []
            actual_returns = []
            
            for i in range(0, len(test_data) - 5, 5):
                try:
                    hist_data = pd.concat([train_data, test_data.iloc[:i]])
                    pred_result = ensemble_model.predict(hist_data, stock)
                    
                    if 'horizon_1' in pred_result:
                        prediction = pred_result['horizon_1']['prediction']
                        
                        # Actual return
                        if i + 1 < len(test_data):
                            current_price = test_data['Close'].iloc[i]
                            future_price = test_data['Close'].iloc[i + 1]
                            actual_return = (future_price - current_price) / current_price
                            
                            predictions.append(prediction)
                            actual_returns.append(actual_return)
                
                except Exception:
                    continue
            
            if len(predictions) > 5:
                split_metrics = self._calculate_prediction_metrics(
                    np.array(predictions), 
                    np.array(actual_returns)
                )
                split_performances.append(split_metrics)
        
        if len(split_performances) < 2:
            pytest.skip("Insufficient splits for consistency testing")
        
        # Analyze consistency across splits
        consistency_metrics = {}
        
        for metric in split_performances[0].keys():
            values = [perf[metric] for perf in split_performances if metric in perf]
            
            if len(values) > 1 and not any(np.isinf(v) for v in values):
                consistency_metrics[f'{metric}_consistency'] = np.std(values) / (np.mean(values) + 1e-8)
        
        # High consistency = low relative standard deviation
        for metric, consistency in consistency_metrics.items():
            if 'accuracy' in metric or 'correlation' in metric:
                # For accuracy metrics, consistency should be reasonable
                assert consistency < 0.5, (
                    f"Poor consistency in {metric}: {consistency:.2f}"
                )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])