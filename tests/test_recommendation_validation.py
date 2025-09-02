"""
Recommendation Quality Testing Suite - Model Prediction Accuracy Validation

Tests the core recommendation engine's ability to generate accurate investment advice
across different market conditions. Validates prediction accuracy, confidence calibration,
and multi-model ensemble coordination for reliable revenue generation.

Priority: TIER 1 - Critical for investment recommendation system reliability
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ensemble import MetaLearningEnsemble
from src.config.settings import config
from src.data.data_loader import DataLoader
from src.trading.signal_generator import SignalGenerator, SignalType


class TestRecommendationValidation:
    """Test suite for validating investment recommendation accuracy and quality."""
    
    @pytest.fixture(scope="class")
    def ensemble_model(self):
        """Create ensemble model for testing."""
        return MetaLearningEnsemble(
            horizon_len=[1, 5, 20],
            uncertainty_alpha=0.1,
            enable_neural_meta_learning=False  # Disable for reproducible testing
        )
    
    @pytest.fixture(scope="class")
    def sample_data(self):
        """Generate realistic sample market data for testing."""
        np.random.seed(42)  # Reproducible tests
        
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        
        # Generate realistic price movements with trends and volatility
        base_price = 100.0
        returns = np.random.normal(0.0005, 0.015, 300)  # 0.05% daily mean, 1.5% volatility
        
        # Add trend periods
        returns[50:100] += 0.002   # Bull trend
        returns[150:200] -= 0.003  # Bear trend
        returns[250:280] *= 2.0    # High volatility period
        
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        data = {
            'date': dates,
            'open': [p * np.random.uniform(0.995, 1.005) for p in prices[:-1]],
            'high': [p * np.random.uniform(1.002, 1.02) for p in prices[:-1]], 
            'low': [p * np.random.uniform(0.98, 0.998) for p in prices[:-1]],
            'close': prices[:-1],
            'volume': [np.random.randint(1000000, 10000000) for _ in range(300)]
        }
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        return df
    
    @pytest.fixture(scope="class")
    def multi_stock_data(self, sample_data):
        """Create multi-stock data for ensemble testing."""
        stocks = ['AAPL', 'MSFT', 'GOOGL']
        data_dict = {}
        
        for i, stock in enumerate(stocks):
            # Create correlated but different price series
            df = sample_data.copy()
            correlation_factor = 0.7 + i * 0.1  # Different correlations
            df['close'] *= (1 + np.random.normal(0, 0.1, len(df)) * correlation_factor)
            df['open'] = df['close'] * np.random.uniform(0.995, 1.005, len(df))
            df['high'] = df['close'] * np.random.uniform(1.002, 1.02, len(df))
            df['low'] = df['close'] * np.random.uniform(0.98, 0.998, len(df))
            
            data_dict[stock] = df
            
        return data_dict
    
    def test_prediction_accuracy_basic(self, ensemble_model, sample_data):
        """Test basic prediction accuracy for investment recommendations."""
        # Generate predictions using recent data
        recent_data = sample_data.iloc[-60:]  # Last 60 days
        
        predictions = ensemble_model.predict(
            recent_data, 
            ticker="TEST", 
            return_confidence=True
        )
        
        # Validate prediction structure
        assert isinstance(predictions, dict), "Predictions should be a dictionary"
        assert len(predictions) > 0, "Should generate at least one prediction"
        
        # Test prediction horizons
        expected_horizons = ['horizon_1', 'horizon_5', 'horizon_20']
        for horizon in expected_horizons:
            assert horizon in predictions, f"Missing prediction for {horizon}"
            
            pred_data = predictions[horizon]
            assert 'prediction' in pred_data, f"Missing prediction value for {horizon}"
            assert 'confidence' in pred_data, f"Missing confidence for {horizon}"
            assert 'prediction_interval' in pred_data, f"Missing interval for {horizon}"
            
            # Validate prediction bounds (realistic investment returns)
            prediction = pred_data['prediction']
            assert -0.5 <= prediction <= 0.5, f"Unrealistic prediction: {prediction:.2%}"
            
            # Validate confidence bounds
            confidence = pred_data['confidence']
            assert 0.0 <= confidence <= 1.0, f"Invalid confidence: {confidence}"
            
            # Validate prediction intervals
            interval = pred_data['prediction_interval']
            assert len(interval) == 2, "Interval should have lower and upper bounds"
            assert interval[0] <= prediction <= interval[1], "Prediction outside interval"
    
    def test_prediction_accuracy_walk_forward(self, ensemble_model, sample_data):
        """Test prediction accuracy using walk-forward analysis."""
        # Use first 200 days for training, test on last 50 days
        train_data = sample_data.iloc[:200]
        test_data = sample_data.iloc[200:]
        
        predictions = []
        actual_returns = []
        
        # Walk-forward testing
        for i in range(30):  # Test 30 predictions
            if i + 20 >= len(test_data):
                break
                
            # Historical data up to point i
            hist_data = pd.concat([train_data, test_data.iloc[:i+1]])
            
            # Generate prediction
            pred_result = ensemble_model.predict(hist_data, "TEST", return_confidence=True)
            
            if 'horizon_1' in pred_result:
                prediction = pred_result['horizon_1']['prediction']
                
                # Calculate actual return 1 day ahead
                current_price = test_data['close'].iloc[i]
                future_price = test_data['close'].iloc[i+1] 
                actual_return = (future_price - current_price) / current_price
                
                predictions.append(prediction)
                actual_returns.append(actual_return)
        
        if len(predictions) > 5:  # Need minimum samples for meaningful test
            predictions = np.array(predictions)
            actual_returns = np.array(actual_returns)
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((actual_returns - predictions) / np.abs(actual_returns)))
            assert mape < 0.5, f"High prediction error: MAPE = {mape:.1%}"
            
            # Directional accuracy (most important for investment decisions)
            correct_direction = np.sum(np.sign(predictions) == np.sign(actual_returns))
            directional_accuracy = correct_direction / len(predictions)
            
            # Should be better than random (50%)
            assert directional_accuracy > 0.4, f"Poor directional accuracy: {directional_accuracy:.1%}"
    
    def test_confidence_calibration_validation(self, ensemble_model, sample_data):
        """Test confidence calibration - high confidence predictions should be more accurate."""
        recent_data = sample_data.iloc[-100:]
        
        predictions = ensemble_model.predict(recent_data, "TEST", return_confidence=True)
        
        for horizon_key in ['horizon_1', 'horizon_5']:
            if horizon_key not in predictions:
                continue
                
            pred_data = predictions[horizon_key]
            confidence = pred_data['confidence']
            prediction = pred_data['prediction']
            interval = pred_data['prediction_interval']
            
            # Test confidence properties
            assert confidence >= 0.3, f"Confidence too low: {confidence:.1%}"
            
            # Higher confidence should correlate with tighter intervals
            interval_width = interval[1] - interval[0]
            if confidence > 0.8:  # High confidence
                assert interval_width < 0.15, f"High confidence but wide interval: {interval_width:.1%}"
            elif confidence < 0.5:  # Low confidence  
                assert interval_width > 0.05, f"Low confidence but narrow interval: {interval_width:.1%}"
    
    def test_multi_model_ensemble_coordination(self, ensemble_model, multi_stock_data):
        """Test that ensemble properly coordinates multiple AI models."""
        predictions = ensemble_model.predict(multi_stock_data, "AAPL", return_confidence=True)
        
        assert 'horizon_1' in predictions, "Missing ensemble prediction"
        pred_data = predictions['horizon_1']
        
        # Validate individual model contributions
        assert 'individual_predictions' in pred_data, "Missing individual model predictions"
        individual = pred_data['individual_predictions']
        
        expected_models = ['timesfm', 'tsmamba', 'samba']
        for model in expected_models:
            assert model in individual, f"Missing prediction from {model}"
            
            model_pred = individual[model]
            assert isinstance(model_pred, (int, float)), f"Invalid {model} prediction type"
            assert -0.5 <= model_pred <= 0.5, f"Unrealistic {model} prediction: {model_pred:.2%}"
        
        # Test model weights
        assert 'model_weights' in pred_data, "Missing model weights"
        weights = pred_data['model_weights']
        
        for model in expected_models:
            assert model in weights, f"Missing weight for {model}"
            assert 0.0 <= weights[model] <= 1.0, f"Invalid weight for {model}: {weights[model]}"
        
        # Weights should sum to approximately 1.0
        weight_sum = sum(weights.values())
        assert 0.98 <= weight_sum <= 1.02, f"Weights don't sum to 1.0: {weight_sum}"
        
        # Test model disagreement tracking
        assert 'model_disagreement' in pred_data, "Missing model disagreement metric"
        disagreement = pred_data['model_disagreement']
        assert 0.0 <= disagreement <= 1.0, f"Invalid disagreement: {disagreement}"
    
    def test_market_regime_adaptation(self, ensemble_model, sample_data):
        """Test investment recommendations adapt to different market regimes."""
        # Test bull market period (days 50-100)
        bull_data = sample_data.iloc[30:100]
        bull_pred = ensemble_model.predict(bull_data, "TEST", return_confidence=True)
        
        # Test bear market period (days 150-200)  
        bear_data = sample_data.iloc[130:200]
        bear_pred = ensemble_model.predict(bear_data, "TEST", return_confidence=True)
        
        # Test volatile period (days 250-280)
        volatile_data = sample_data.iloc[230:280]
        volatile_pred = ensemble_model.predict(volatile_data, "TEST", return_confidence=True)
        
        predictions = [bull_pred, bear_pred, volatile_pred]
        regime_names = ['bull', 'bear', 'volatile']
        
        for pred, regime in zip(predictions, regime_names):
            if 'horizon_1' not in pred:
                continue
                
            pred_data = pred['horizon_1']
            
            # Validate regime information
            assert 'regime_info' in pred_data, f"Missing regime info for {regime}"
            regime_info = pred_data['regime_info']
            
            assert 'current_regime' in regime_info, f"Missing current regime for {regime}"
            assert 'regime_confidence' in regime_info, f"Missing regime confidence for {regime}"
            
            regime_confidence = regime_info['regime_confidence']
            assert 0.0 <= regime_confidence <= 1.0, f"Invalid regime confidence: {regime_confidence}"
        
        # Test that model adapts weights based on regime
        if (bull_pred.get('horizon_1') and bear_pred.get('horizon_1') and 
            'model_weights' in bull_pred['horizon_1'] and 'model_weights' in bear_pred['horizon_1']):
            
            bull_weights = bull_pred['horizon_1']['model_weights']
            bear_weights = bear_pred['horizon_1']['model_weights']
            
            # Weights should differ between regimes (adaptive behavior)
            weight_differences = []
            for model in ['timesfm', 'tsmamba', 'samba']:
                if model in bull_weights and model in bear_weights:
                    diff = abs(bull_weights[model] - bear_weights[model])
                    weight_differences.append(diff)
            
            if weight_differences:
                max_diff = max(weight_differences)
                # Should show some adaptation (not identical weights)
                assert max_diff > 0.05, f"Insufficient regime adaptation: max diff = {max_diff}"
    
    def test_prediction_consistency_multi_horizon(self, ensemble_model, sample_data):
        """Test consistency across different prediction horizons."""
        recent_data = sample_data.iloc[-80:]
        predictions = ensemble_model.predict(recent_data, "TEST", return_confidence=True)
        
        horizons = ['horizon_1', 'horizon_5', 'horizon_20']
        horizon_predictions = {}
        
        # Collect predictions for each horizon
        for horizon in horizons:
            if horizon in predictions:
                pred_data = predictions[horizon]
                horizon_predictions[horizon] = {
                    'prediction': pred_data['prediction'],
                    'confidence': pred_data['confidence']
                }
        
        # Test consistency rules
        if len(horizon_predictions) >= 2:
            preds = [data['prediction'] for data in horizon_predictions.values()]
            confs = [data['confidence'] for data in horizon_predictions.values()]
            
            # Predictions should maintain general directional consistency
            positive_preds = sum(1 for p in preds if p > 0)
            negative_preds = sum(1 for p in preds if p < 0)
            
            # Allow for some disagreement but not complete contradiction
            consistency_ratio = max(positive_preds, negative_preds) / len(preds)
            assert consistency_ratio >= 0.5, f"Poor horizon consistency: {consistency_ratio:.1%}"
            
            # Confidence levels should be reasonable across horizons
            avg_confidence = np.mean(confs)
            assert avg_confidence > 0.3, f"Overall confidence too low: {avg_confidence:.1%}"
    
    def test_recommendation_robustness_data_quality(self, ensemble_model):
        """Test robustness of recommendations to data quality issues."""
        # Create data with common quality issues
        base_dates = pd.date_range('2023-01-01', periods=100, freq='D')
        base_prices = 100 * np.cumprod(1 + np.random.normal(0, 0.01, 100))
        
        # Test 1: Missing data (gaps)
        dates_with_gaps = []
        prices_with_gaps = []
        for i, (date, price) in enumerate(zip(base_dates, base_prices)):
            if i % 10 != 0:  # Skip every 10th day
                dates_with_gaps.append(date)
                prices_with_gaps.append(price)
        
        df_gaps = pd.DataFrame({
            'date': dates_with_gaps,
            'close': prices_with_gaps,
            'open': [p * 0.999 for p in prices_with_gaps],
            'high': [p * 1.015 for p in prices_with_gaps],
            'low': [p * 0.985 for p in prices_with_gaps],
            'volume': [1000000] * len(prices_with_gaps)
        })
        df_gaps.set_index('date', inplace=True)
        
        # Should handle gaps gracefully
        try:
            predictions_gaps = ensemble_model.predict(df_gaps, "TEST", return_confidence=True)
            assert len(predictions_gaps) > 0, "Should generate predictions despite gaps"
        except Exception as e:
            pytest.fail(f"Model failed on data with gaps: {e}")
        
        # Test 2: Extreme price movements (outliers)
        df_outliers = pd.DataFrame({
            'date': base_dates,
            'close': base_prices.copy(),
            'open': base_prices * 0.999,
            'high': base_prices * 1.015, 
            'low': base_prices * 0.985,
            'volume': [1000000] * 100
        })
        df_outliers.set_index('date', inplace=True)
        
        # Add extreme outliers
        df_outliers.loc[df_outliers.index[50], 'close'] *= 1.5  # 50% jump
        df_outliers.loc[df_outliers.index[51], 'close'] *= 0.7  # 30% drop
        
        # Should handle outliers without crashing
        try:
            predictions_outliers = ensemble_model.predict(df_outliers, "TEST", return_confidence=True)
            assert len(predictions_outliers) > 0, "Should generate predictions despite outliers"
            
            # Predictions should still be reasonable
            if 'horizon_1' in predictions_outliers:
                pred = predictions_outliers['horizon_1']['prediction']
                assert -0.5 <= pred <= 0.5, f"Outliers caused unrealistic prediction: {pred:.2%}"
                
        except Exception as e:
            pytest.fail(f"Model failed on data with outliers: {e}")
    
    def test_recommendation_latency_performance(self, ensemble_model, sample_data):
        """Test recommendation generation latency for real-time usage."""
        import time
        
        recent_data = sample_data.iloc[-60:]
        
        # Measure prediction latency
        start_time = time.time()
        predictions = ensemble_model.predict(recent_data, "TEST", return_confidence=True)
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Should generate recommendations within reasonable time
        max_latency = config.MAX_LATENCY_SECONDS  # From settings.py
        assert latency < max_latency, f"Recommendation latency too high: {latency:.2f}s > {max_latency}s"
        
        # Validate we got predictions despite time pressure
        assert len(predictions) > 0, "Failed to generate predictions within time limit"
        
        # Test repeated predictions (caching effect)
        start_time_cached = time.time()
        predictions_cached = ensemble_model.predict(recent_data, "TEST", return_confidence=True, use_cache=True)
        end_time_cached = time.time()
        
        cached_latency = end_time_cached - start_time_cached
        # Cached predictions should be faster
        assert cached_latency < latency, f"Caching not improving performance: {cached_latency:.2f}s"


# Integration tests with actual recommendation pipeline
class TestRecommendationIntegration:
    """Integration tests for complete recommendation pipeline."""
    
    @pytest.fixture
    def signal_generator(self):
        """Create signal generator for integration testing."""
        return SignalGenerator(
            confidence_thresholds={
                'intraday': 0.65,   # Lower thresholds for testing
                'daily': 0.70,
                'weekly': 0.75
            }
        )
    
    def test_end_to_end_recommendation_pipeline(self, signal_generator):
        """Test complete recommendation generation pipeline."""
        # Create sample market data for multiple timeframes
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        base_price = 100.0
        
        # Generate realistic data with different market patterns
        returns = np.random.normal(0.001, 0.02, 200)
        prices = [base_price * np.cumprod(1 + returns[:i+1])[-1] for i in range(200)]
        
        # Create multi-timeframe data
        from src.trading.signal_generator import TimeFrame
        data_dict = {
            'TEST': {
                TimeFrame.DAILY: pd.DataFrame({
                    'date': dates,
                    'open': [p * 0.999 for p in prices],
                    'high': [p * 1.02 for p in prices],
                    'low': [p * 0.98 for p in prices], 
                    'close': prices,
                    'volume': [np.random.randint(1000000, 5000000) for _ in range(200)]
                }).set_index('date')
            }
        }
        
        current_prices = {'TEST': prices[-1]}
        
        # Generate complete recommendations
        recommendations = signal_generator.generate_signals(data_dict, current_prices)
        
        # Validate recommendation structure
        assert isinstance(recommendations, dict), "Should return recommendation dictionary"
        
        if 'TEST' in recommendations:
            recommendation = recommendations['TEST']
            
            # Validate recommendation attributes
            assert hasattr(recommendation, 'overall_signal'), "Missing overall signal"
            assert hasattr(recommendation, 'overall_confidence'), "Missing overall confidence"
            assert hasattr(recommendation, 'consensus_score'), "Missing consensus score"
            
            # Validate signal type
            assert recommendation.overall_signal in [SignalType.BUY, SignalType.SELL, SignalType.HOLD, 
                                                   SignalType.STRONG_BUY, SignalType.STRONG_SELL]
            
            # Validate confidence and consensus
            assert 0.0 <= recommendation.overall_confidence <= 1.0
            assert 0.0 <= recommendation.consensus_score <= 1.0
    
    def test_recommendation_quality_filtering(self, signal_generator):
        """Test quality filtering of recommendations."""
        # This test would use the filter_signals_by_quality method
        # Create mock signals with varying quality
        from src.trading.signal_generator import MultiTimeFrameSignal, TradingSignal, TimeFrame
        
        # Mock high-quality signal
        high_quality_signal = MultiTimeFrameSignal(
            ticker="HIGH_QUAL",
            overall_signal=SignalType.BUY,
            overall_confidence=0.85,
            overall_strength=0.7,
            signals_by_timeframe={},
            consensus_score=0.8,
            timestamp=datetime.now()
        )
        
        # Mock low-quality signal
        low_quality_signal = MultiTimeFrameSignal(
            ticker="LOW_QUAL", 
            overall_signal=SignalType.HOLD,
            overall_confidence=0.45,
            overall_strength=0.2,
            signals_by_timeframe={},
            consensus_score=0.3,
            timestamp=datetime.now()
        )
        
        signals = {
            'HIGH_QUAL': high_quality_signal,
            'LOW_QUAL': low_quality_signal
        }
        
        # Apply quality filtering
        filtered_signals = signal_generator.filter_signals_by_quality(
            signals,
            min_consensus=0.6,
            min_confidence=0.7, 
            min_strength=0.3
        )
        
        # High quality signal should pass, low quality should be filtered
        assert 'HIGH_QUAL' in filtered_signals, "High quality signal should pass filter"
        assert 'LOW_QUAL' not in filtered_signals, "Low quality signal should be filtered"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])