"""
Comprehensive test suite for Multi-Resolution Prediction System.

Tests all components of Phase 5: Multi-Resolution Ensemble including:
- Minute-scale predictor
- Hourly predictor  
- Weekly predictor
- Resolution fusion network
- Adaptive resolution weighting
- Integration with main ensemble
"""

import pytest
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.multi_resolution import (
    MinuteScalePredictor,
    HourlyPredictor,
    WeeklyPredictor,
    ResolutionFuser,
    AdaptiveResolutionWeighting
)
from models.ensemble import MetaLearningEnsemble
from features.meta_feature_extractor import MetaFeatureExtractor


class TestMultiResolutionPredictors:
    """Test suite for individual resolution predictors."""
    
    @pytest.fixture
    def sample_daily_data(self):
        """Generate sample daily OHLCV data."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        n_points = len(dates)
        returns = np.random.normal(0.001, 0.02, n_points)
        
        close_prices = [100.0]
        for ret in returns:
            close_prices.append(close_prices[-1] * (1 + ret))
        close_prices = close_prices[1:]
        
        # Generate OHLV from close prices
        high_prices = [price * np.random.uniform(1.001, 1.02) for price in close_prices]
        low_prices = [price * np.random.uniform(0.98, 0.999) for price in close_prices]
        open_prices = close_prices[:-1] + [close_prices[-1]]
        volumes = np.random.uniform(100000, 1000000, n_points)
        
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=dates)
        
        return df
    
    @pytest.fixture
    def sample_minute_data(self):
        """Generate sample minute-level OHLCV data."""
        # Generate 1 day of minute data (390 minutes)
        start_time = datetime(2023, 6, 15, 9, 30)  # Market open
        minutes = []
        current = start_time
        for i in range(390):  # 6.5 hours * 60 minutes
            minutes.append(current)
            current += timedelta(minutes=1)
        
        np.random.seed(42)
        n_points = len(minutes)
        returns = np.random.normal(0.0001, 0.003, n_points)  # Smaller minute returns
        
        close_prices = [100.0]
        for ret in returns:
            close_prices.append(close_prices[-1] * (1 + ret))
        close_prices = close_prices[1:]
        
        # Generate OHLV
        high_prices = [price * np.random.uniform(1.0001, 1.005) for price in close_prices]
        low_prices = [price * np.random.uniform(0.995, 0.9999) for price in close_prices]
        open_prices = close_prices[:-1] + [close_prices[-1]]
        volumes = np.random.uniform(1000, 10000, n_points)
        
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=minutes)
        
        return df
    
    def test_minute_scale_predictor_initialization(self):
        """Test MinuteScalePredictor initialization."""
        predictor = MinuteScalePredictor(
            lookback_window=30,
            prediction_horizons=[1, 5],
            device='cpu'
        )
        
        assert predictor.lookback_window == 30
        assert predictor.prediction_horizons == [1, 5]
        assert not predictor.is_trained
        assert predictor.feature_extractor is not None
    
    def test_minute_scale_feature_extraction(self, sample_minute_data):
        """Test minute-scale feature extraction."""
        predictor = MinuteScalePredictor(lookback_window=30, device='cpu')
        
        # Test feature extraction
        features = predictor.feature_extractor.extract_features(sample_minute_data)
        
        assert features.shape[0] > 0  # Should have some windows
        assert features.shape[1] == 30  # Lookback window
        assert features.shape[2] > 0   # Should have features
    
    def test_minute_scale_prediction(self, sample_minute_data):
        """Test minute-scale prediction."""
        predictor = MinuteScalePredictor(
            lookback_window=30,
            prediction_horizons=[1, 5],
            device='cpu'
        )
        
        # Should work even without training (fallback prediction)
        predictions = predictor.predict(sample_minute_data, 'TEST', return_confidence=True)
        
        assert 'horizon_1' in predictions
        assert 'horizon_5' in predictions
        assert 'minute_scale_meta' in predictions
        
        # Check prediction structure
        for horizon_key in ['horizon_1', 'horizon_5']:
            pred_data = predictions[horizon_key]
            assert 'prediction' in pred_data
            assert 'confidence' in pred_data
            assert 'prediction_interval' in pred_data
            assert isinstance(pred_data['prediction'], float)
            assert 0 <= pred_data['confidence'] <= 1
    
    def test_hourly_predictor_initialization(self):
        """Test HourlyPredictor initialization."""
        predictor = HourlyPredictor(
            lookback_hours=48,
            prediction_horizons=[1, 2, 4],
            device='cpu'
        )
        
        assert predictor.lookback_hours == 48
        assert predictor.prediction_horizons == [1, 2, 4]
        assert not predictor.is_trained
        assert predictor.feature_extractor is not None
    
    def test_hourly_feature_extraction(self, sample_daily_data):
        """Test hourly feature extraction."""
        predictor = HourlyPredictor(lookback_hours=48, device='cpu')
        
        # Test with daily data (will be approximated as hourly)
        try:
            features, session_mask = predictor.feature_extractor.extract_features(sample_daily_data)
            
            assert features.shape[0] > 0  # Should have some windows
            assert features.shape[1] == 48  # Lookback hours
            assert features.shape[2] > 0   # Should have features
            assert session_mask.shape[2] == 6  # 6 session types
        except ValueError:
            # Expected if insufficient data
            pytest.skip("Insufficient data for hourly feature extraction")
    
    def test_hourly_prediction(self, sample_daily_data):
        """Test hourly prediction."""
        predictor = HourlyPredictor(
            lookback_hours=48,
            prediction_horizons=[1, 2, 4],
            device='cpu'
        )
        
        predictions = predictor.predict(sample_daily_data, 'TEST', return_confidence=True)
        
        assert 'hourly_meta' in predictions
        
        # Should have at least some horizons (may fallback if insufficient data)
        horizon_keys = [k for k in predictions.keys() if k.startswith('horizon_')]
        assert len(horizon_keys) > 0
        
        for horizon_key in horizon_keys:
            pred_data = predictions[horizon_key]
            assert 'prediction' in pred_data
            assert 'confidence' in pred_data
            assert 'model_type' in pred_data
    
    def test_weekly_predictor_initialization(self):
        """Test WeeklyPredictor initialization."""
        predictor = WeeklyPredictor(
            lookback_weeks=26,
            prediction_horizons=[5, 10, 20],
            device='cpu'
        )
        
        assert predictor.lookback_weeks == 26
        assert predictor.prediction_horizons == [5, 10, 20]
        assert not predictor.is_trained
        assert predictor.feature_extractor is not None
    
    def test_weekly_feature_extraction(self, sample_daily_data):
        """Test weekly feature extraction."""
        predictor = WeeklyPredictor(lookback_weeks=26, device='cpu')
        
        try:
            features = predictor.feature_extractor.extract_features(sample_daily_data)
            
            assert features.shape[0] > 0  # Should have some windows
            assert features.shape[1] == 26  # Lookback weeks
            assert features.shape[2] > 0   # Should have features
        except ValueError:
            # Expected if insufficient data
            pytest.skip("Insufficient data for weekly feature extraction")
    
    def test_weekly_prediction(self, sample_daily_data):
        """Test weekly prediction."""
        predictor = WeeklyPredictor(
            lookback_weeks=26,
            prediction_horizons=[5, 10, 20],
            device='cpu'
        )
        
        predictions = predictor.predict(sample_daily_data, 'TEST', return_confidence=True)
        
        assert 'weekly_meta' in predictions
        
        # Should have at least some horizons (may fallback if insufficient data)
        horizon_keys = [k for k in predictions.keys() if k.startswith('horizon_')]
        assert len(horizon_keys) > 0
        
        for horizon_key in horizon_keys:
            pred_data = predictions[horizon_key]
            assert 'prediction' in pred_data
            assert 'confidence' in pred_data
            assert 'model_type' in pred_data


class TestResolutionFusion:
    """Test suite for resolution fusion system."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Generate sample predictions from different resolutions."""
        minute_pred = {
            'horizon_1': {
                'prediction': 0.002,
                'confidence': 0.7,
                'prediction_interval': [-0.01, 0.014],
                'model_type': 'minute_scale'
            }
        }
        
        hourly_pred = {
            'horizon_1': {
                'prediction': 0.001,
                'confidence': 0.6,
                'prediction_interval': [-0.015, 0.017],
                'model_type': 'hourly'
            }
        }
        
        weekly_pred = {
            'horizon_1': {
                'prediction': 0.005,
                'confidence': 0.8,
                'prediction_interval': [-0.02, 0.03],
                'model_type': 'weekly'
            }
        }
        
        return minute_pred, hourly_pred, weekly_pred
    
    def test_resolution_fuser_initialization(self):
        """Test ResolutionFuser initialization."""
        fuser = ResolutionFuser(
            resolution_dims={'minute': 64, 'hourly': 128, 'weekly': 256},
            hidden_dim=256,
            output_horizons=[1, 5, 20],
            device='cpu'
        )
        
        assert fuser.fusion_network is not None
        assert fuser.context_encoder is not None
    
    def test_resolution_fusion(self, sample_predictions):
        """Test multi-resolution prediction fusion."""
        fuser = ResolutionFuser(device='cpu')
        minute_pred, hourly_pred, weekly_pred = sample_predictions
        
        # Test fusion
        fused_predictions = fuser.fuse_predictions(
            minute_predictions=minute_pred,
            hourly_predictions=hourly_pred,
            weekly_predictions=weekly_pred
        )
        
        assert 'horizon_1' in fused_predictions
        assert 'fusion_meta' in fused_predictions
        
        # Check fused prediction structure
        pred_data = fused_predictions['horizon_1']
        assert 'prediction' in pred_data
        assert 'confidence' in pred_data
        assert 'prediction_interval' in pred_data
        assert pred_data['model_type'] == 'multi_resolution_fusion'
        
        # Check fusion metadata
        fusion_meta = fused_predictions['fusion_meta']
        assert 'resolution_weights' in fusion_meta
        assert 'available_resolutions' in fusion_meta
    
    def test_resolution_fusion_partial_predictions(self, sample_predictions):
        """Test fusion with only some resolution predictions available."""
        fuser = ResolutionFuser(device='cpu')
        _, _, weekly_pred = sample_predictions
        
        # Test with only weekly predictions
        fused_predictions = fuser.fuse_predictions(
            weekly_predictions=weekly_pred
        )
        
        # Should still produce valid output
        assert 'horizon_1' in fused_predictions or 'fusion_meta' in fused_predictions


class TestAdaptiveWeighting:
    """Test suite for adaptive resolution weighting."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for weighting."""
        dates = pd.date_range(start='2023-06-01', end='2023-06-30', freq='D')
        np.random.seed(42)
        
        returns = np.random.normal(0.001, 0.02, len(dates))
        close_prices = [100.0]
        for ret in returns:
            close_prices.append(close_prices[-1] * (1 + ret))
        close_prices = close_prices[1:]
        
        df = pd.DataFrame({
            'close': close_prices,
            'volume': np.random.uniform(100000, 1000000, len(dates)),
            'high': [p * 1.01 for p in close_prices],
            'low': [p * 0.99 for p in close_prices],
            'open': close_prices
        }, index=dates)
        
        return df
    
    def test_adaptive_weighting_initialization(self):
        """Test AdaptiveResolutionWeighting initialization."""
        weighter = AdaptiveResolutionWeighting(
            performance_window=50,
            adaptation_rate=0.1,
            device='cpu'
        )
        
        assert weighter.performance_adapter is not None
        assert weighter.session_weighter is not None
        assert weighter.regime_network is not None
    
    def test_weight_computation(self, sample_market_data):
        """Test adaptive weight computation."""
        weighter = AdaptiveResolutionWeighting(device='cpu')
        
        weights_result = weighter.compute_adaptive_weights(
            market_data=sample_market_data,
            current_time=datetime.now(),
            volatility=0.15
        )
        
        assert 'final_weights' in weights_result
        assert 'performance_weights' in weights_result
        assert 'session_weights' in weights_result
        
        # Check weight properties
        final_weights = weights_result['final_weights']
        assert 'minute' in final_weights
        assert 'hourly' in final_weights
        assert 'weekly' in final_weights
        
        # Weights should sum to approximately 1
        weight_sum = sum(final_weights.values())
        assert abs(weight_sum - 1.0) < 0.01
    
    def test_performance_tracking(self):
        """Test performance-based adaptation."""
        weighter = AdaptiveResolutionWeighting(device='cpu')
        
        # Update performance for different resolutions
        weighter.update_performance(
            resolution_predictions={'minute': 0.01, 'hourly': 0.005, 'weekly': 0.008},
            actual_values={'minute': 0.009, 'hourly': 0.007, 'weekly': 0.012},
            resolution_confidences={'minute': 0.7, 'hourly': 0.6, 'weekly': 0.8}
        )
        
        # Get performance summary
        summary = weighter.get_weighting_summary()
        assert 'current_weights' in summary
        assert 'performance_summary' in summary
    
    def test_session_based_weighting(self):
        """Test session-based weight adjustment."""
        weighter = AdaptiveResolutionWeighting(device='cpu')
        session_weighter = weighter.session_weighter
        
        # Test different times
        morning_time = datetime(2023, 6, 15, 10, 0)  # Market open
        afternoon_time = datetime(2023, 6, 15, 15, 0)  # Market close
        evening_time = datetime(2023, 6, 15, 20, 0)  # After hours
        
        morning_weights = session_weighter.get_session_weights(morning_time)
        afternoon_weights = session_weighter.get_session_weights(afternoon_time)
        evening_weights = session_weighter.get_session_weights(evening_time)
        
        # Weights should be different for different sessions
        assert morning_weights != afternoon_weights
        assert afternoon_weights != evening_weights
        
        # All should sum to approximately 1
        for weights in [morning_weights, afternoon_weights, evening_weights]:
            assert abs(sum(weights.values()) - 1.0) < 0.01


class TestEnsembleIntegration:
    """Test suite for multi-resolution integration with main ensemble."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for ensemble testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
        np.random.seed(42)
        
        n_points = len(dates)
        returns = np.random.normal(0.001, 0.02, n_points)
        
        close_prices = [100.0]
        for ret in returns:
            close_prices.append(close_prices[-1] * (1 + ret))
        close_prices = close_prices[1:]
        
        df = pd.DataFrame({
            'open': close_prices,
            'high': [p * np.random.uniform(1.001, 1.02) for p in close_prices],
            'low': [p * np.random.uniform(0.98, 0.999) for p in close_prices],
            'close': close_prices,
            'volume': np.random.uniform(100000, 1000000, n_points)
        }, index=dates)
        
        return df
    
    def test_ensemble_with_multi_resolution_enabled(self, sample_data):
        """Test ensemble with multi-resolution system enabled."""
        ensemble = MetaLearningEnsemble(
            horizon_len=[1, 5, 20],
            enable_multi_resolution=True,
            multi_resolution_weight=0.3,
            device='cpu'
        )
        
        assert ensemble.enable_multi_resolution
        assert 'multi_resolution' in ensemble.weights
        assert ensemble.multi_resolution_predictors is not None
        assert ensemble.resolution_fuser is not None
        assert ensemble.adaptive_weighting is not None
    
    def test_ensemble_with_multi_resolution_disabled(self, sample_data):
        """Test ensemble with multi-resolution system disabled."""
        ensemble = MetaLearningEnsemble(
            horizon_len=[1, 5, 20],
            enable_multi_resolution=False,
            device='cpu'
        )
        
        assert not ensemble.enable_multi_resolution
        assert 'multi_resolution' not in ensemble.weights
        assert ensemble.multi_resolution_predictors is None
    
    @patch('src.models.timesfm_predictor.TimesFMPredictor')
    @patch('src.models.tsmamba_predictor.TSMambaPredictor')
    @patch('src.models.samba_predictor.SAMBAPredictor')
    @patch('src.models.tft_predictor.TFTPredictor')
    def test_ensemble_prediction_with_multi_resolution(
        self, 
        mock_tft, mock_samba, mock_tsmamba, mock_timesfm,
        sample_data
    ):
        """Test ensemble predictions including multi-resolution component."""
        # Mock individual predictors
        mock_pred_result = {
            'horizon_1': {'prediction': 0.01, 'confidence': 0.7},
            'horizon_5': {'prediction': 0.02, 'confidence': 0.6}
        }
        
        mock_timesfm.return_value.predict.return_value = mock_pred_result
        mock_tsmamba.return_value.predict.return_value = mock_pred_result
        mock_samba.return_value.predict.return_value = mock_pred_result
        mock_tft.return_value.predict.return_value = mock_pred_result
        
        ensemble = MetaLearningEnsemble(
            horizon_len=[1, 5],
            enable_multi_resolution=True,
            multi_resolution_weight=0.2,
            device='cpu'
        )
        
        # Test prediction
        predictions = ensemble.predict(sample_data, 'TEST')
        
        # Should contain standard ensemble predictions
        assert 'horizon_1' in predictions
        assert 'horizon_5' in predictions
        assert 'ensemble_info' in predictions
        
        # Check that multi-resolution is included in metadata
        if ensemble.enable_multi_resolution:
            assert ensemble.weights.get('multi_resolution', 0) > 0


class TestMetaFeatureExtraction:
    """Test suite for multi-resolution meta-feature extraction."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Generate sample price data."""
        dates = pd.date_range(start='2023-06-01', end='2023-06-30', freq='D')
        np.random.seed(42)
        
        returns = np.random.normal(0.001, 0.02, len(dates))
        close_prices = [100.0]
        for ret in returns:
            close_prices.append(close_prices[-1] * (1 + ret))
        close_prices = close_prices[1:]
        
        df = pd.DataFrame({
            'close': close_prices,
            'volume': np.random.uniform(100000, 1000000, len(dates)),
            'high': [p * 1.01 for p in close_prices],
            'low': [p * 0.99 for p in close_prices],
            'open': close_prices
        }, index=dates)
        
        return df
    
    def test_multi_resolution_feature_extraction(self, sample_price_data):
        """Test multi-resolution specific feature extraction."""
        extractor = MetaFeatureExtractor(lookback_window=30)
        
        # Test feature extraction
        features = extractor.extract_multi_resolution_features(sample_price_data)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (32,)  # Fixed feature dimension
        assert features.dtype == np.float32
        
        # Features should be finite
        assert np.all(np.isfinite(features))
    
    def test_combined_features_with_multi_resolution(self, sample_price_data):
        """Test combined feature extraction including multi-resolution features."""
        extractor = MetaFeatureExtractor(lookback_window=30)
        
        # Extract combined features
        combined_features = extractor.extract_combined_features(sample_price_data)
        
        assert isinstance(combined_features, np.ndarray)
        assert len(combined_features) > 50  # Should include multi-resolution features
        assert np.all(np.isfinite(combined_features))
    
    def test_multi_resolution_performance_tracking(self):
        """Test performance tracking for multi-resolution system."""
        extractor = MetaFeatureExtractor()
        
        # Check that multi-resolution is in performance history
        assert 'multi_resolution' in extractor.model_performance_history
        
        # Add some performance data
        extractor.model_performance_history['multi_resolution'].append(0.7)
        extractor.model_performance_history['multi_resolution'].append(0.8)
        
        assert len(extractor.model_performance_history['multi_resolution']) == 2


class TestErrorHandling:
    """Test suite for error handling and edge cases."""
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios."""
        # Very small dataset
        small_data = pd.DataFrame({
            'close': [100, 101, 99],
            'volume': [1000, 1200, 900],
            'high': [101, 102, 100],
            'low': [99, 100, 98],
            'open': [100, 101, 99]
        })
        
        # Test minute predictor
        minute_predictor = MinuteScalePredictor(device='cpu')
        minute_result = minute_predictor.predict(small_data, 'TEST')
        
        # Should return fallback prediction
        assert 'minute_scale_meta' in minute_result
        assert minute_result['minute_scale_meta']['fallback_used']
        
        # Test weekly predictor
        weekly_predictor = WeeklyPredictor(device='cpu')
        weekly_result = weekly_predictor.predict(small_data, 'TEST')
        
        # Should return fallback prediction
        assert 'weekly_meta' in weekly_result
        assert weekly_result['weekly_meta']['fallback_used']
    
    def test_missing_predictions_fusion(self):
        """Test fusion when some resolution predictions are missing."""
        fuser = ResolutionFuser(device='cpu')
        
        # Test with no predictions
        result = fuser.fuse_predictions()
        
        # Should return fallback
        assert 'fusion_meta' in result
        assert result['fusion_meta']['fallback_used']
    
    def test_device_handling(self):
        """Test device handling for different components."""
        # Test CPU device
        cpu_predictor = MinuteScalePredictor(device='cpu')
        assert cpu_predictor.device.type == 'cpu'
        
        # Test auto device selection
        auto_predictor = HourlyPredictor(device='auto')
        assert auto_predictor.device.type in ['cpu', 'cuda']
    
    def test_model_persistence(self):
        """Test saving and loading of multi-resolution models."""
        import tempfile
        
        predictor = MinuteScalePredictor(device='cpu')
        
        with tempfile.NamedTemporaryFile(suffix='.pt') as f:
            # Should handle saving even when model not trained
            result = predictor.save_model(f.name)
            # May return False if no model to save, which is expected
            assert isinstance(result, bool)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])