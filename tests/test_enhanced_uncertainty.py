"""Tests for Enhanced Uncertainty Quantification System (Phase 3)."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import torch
from typing import Dict, List, Any

# Import the uncertainty system components
from src.models.enhanced_uncertainty_system import EnhancedUncertaintySystem, UncertaintyPrediction
from src.models.uncertainty_extensions import (
    RegimeAdaptiveMapie, MonteCarloDropoutUncertainty, 
    DeepEnsembleUncertainty, QuantileRegressor, UncertaintyFusionNetwork
)
from src.models.regime_detector import HiddenMarkovRegimeDetector, MarketRegime, RegimeState


class TestEnhancedUncertaintySystem:
    """Test suite for Enhanced Uncertainty Quantification System."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        # Generate realistic financial features
        X = np.random.randn(n_samples, n_features)
        # Add some realistic patterns
        X[:, 0] = np.cumsum(np.random.randn(n_samples) * 0.01)  # Price-like feature
        X[:, 1] = np.random.exponential(1, n_samples)  # Volume-like feature
        
        # Generate target with some predictable patterns and noise
        y = (0.1 * X[:, 0] + 0.05 * X[:, 1] + 
             0.02 * np.sin(np.arange(n_samples) / 50) +
             np.random.randn(n_samples) * 0.02)
        
        return X, y
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for regime detection."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        market_data = {}
        for ticker in ['AAPL', 'NVDA', 'AMD']:
            # Generate realistic OHLCV data
            close_prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))
            high_prices = close_prices * (1 + np.random.exponential(0.01, 100))
            low_prices = close_prices * (1 - np.random.exponential(0.01, 100))
            volumes = np.random.lognormal(15, 1, 100)
            
            market_data[ticker] = pd.DataFrame({
                'date': dates,
                'open': np.roll(close_prices, 1),
                'high': high_prices,
                'low': low_prices,
                'close': close_prices,
                'volume': volumes
            })
        
        return market_data
    
    def test_system_initialization(self):
        """Test system initialization with different configurations."""
        # Test default initialization
        system = EnhancedUncertaintySystem(input_dim=20)
        
        assert system.input_dim == 20
        assert system.parallel_computation == True
        assert system.cache_predictions == True
        assert len(system.enable_methods) > 0
        
        # Test custom configuration
        custom_methods = {
            'regime_mapie': True,
            'mc_dropout': False,
            'deep_ensemble': True,
            'quantile_regression': False,
            'fusion_network': False
        }
        
        system_custom = EnhancedUncertaintySystem(
            input_dim=15,
            enable_methods=custom_methods,
            parallel_computation=False,
            cache_predictions=False
        )
        
        assert system_custom.input_dim == 15
        assert system_custom.enable_methods == custom_methods
        assert system_custom.parallel_computation == False
        assert system_custom.prediction_cache is None
    
    def test_system_training(self, sample_data, sample_market_data):
        """Test system training with various methods."""
        X, y = sample_data
        
        # Use smaller dataset for faster testing
        X_train, y_train = X[:200], y[:200]
        
        # Test with limited methods for faster execution
        test_methods = {
            'regime_mapie': True,
            'mc_dropout': True,
            'deep_ensemble': False,  # Disable for speed
            'quantile_regression': False,  # Disable for speed
            'fusion_network': True  # Will be enabled if 2+ methods are active
        }
        
        system = EnhancedUncertaintySystem(
            input_dim=X_train.shape[1],
            enable_methods=test_methods,
            parallel_computation=False  # Sequential for easier testing
        )
        
        # Train the system
        training_results = system.fit(
            X_train, y_train, sample_market_data,
            epochs=5,  # Reduced epochs for testing
            verbose=False
        )
        
        # Verify training results structure
        assert 'method_results' in training_results
        assert 'overall_performance' in training_results
        
        method_results = training_results['method_results']
        assert len(method_results) >= 1  # At least one method should be trained
        
        overall_performance = training_results['overall_performance']
        assert 'training_time_seconds' in overall_performance
        assert 'n_samples' in overall_performance
        assert 'active_methods' in overall_performance
        
        # Verify that trained methods are in the active methods list
        for method_name in method_results.keys():
            assert method_name in overall_performance['active_methods']
    
    def test_uncertainty_prediction_single(self, sample_data, sample_market_data):
        """Test uncertainty prediction for single sample."""
        X, y = sample_data
        X_single = X[:1]  # Single sample
        
        # Create minimal system for testing
        system = EnhancedUncertaintySystem(
            input_dim=X.shape[1],
            enable_methods={'regime_mapie': True, 'mc_dropout': False, 'deep_ensemble': False, 
                          'quantile_regression': False, 'fusion_network': False}
        )
        
        # Mock the MAPIE method to avoid actual training
        mock_mapie = Mock()
        mock_mapie.predict_with_regime_adaptation.return_value = (
            np.array([0.05]),  # predictions
            np.array([[0.02, 0.08]]),  # intervals
            RegimeState(MarketRegime.BULL_TREND, 0.8, 5, {}, {}, datetime.now())  # regime
        )
        system.methods = {'regime_mapie': mock_mapie}
        
        # Get prediction
        prediction = system.predict_with_uncertainty(X_single, sample_market_data)
        
        # Verify prediction structure
        assert isinstance(prediction, UncertaintyPrediction)
        assert hasattr(prediction, 'mean_prediction')
        assert hasattr(prediction, 'fused_prediction')
        assert hasattr(prediction, 'total_uncertainty')
        assert hasattr(prediction, 'prediction_interval_95')
        assert hasattr(prediction, 'confidence_score')
        assert hasattr(prediction, 'market_regime')
        
        # Verify data types and ranges
        assert isinstance(prediction.mean_prediction, float)
        assert isinstance(prediction.confidence_score, float)
        assert 0 <= prediction.confidence_score <= 1
        assert len(prediction.prediction_interval_95) == 2
        assert prediction.prediction_interval_95[0] < prediction.prediction_interval_95[1]
    
    def test_uncertainty_prediction_batch(self, sample_data, sample_market_data):
        """Test uncertainty prediction for multiple samples."""
        X, y = sample_data
        X_batch = X[:5]  # Small batch
        
        system = EnhancedUncertaintySystem(
            input_dim=X.shape[1],
            enable_methods={'regime_mapie': True, 'mc_dropout': False, 'deep_ensemble': False,
                          'quantile_regression': False, 'fusion_network': False}
        )
        
        # Mock the method
        mock_mapie = Mock()
        mock_mapie.predict_with_regime_adaptation.return_value = (
            np.array([0.05, 0.03, 0.01, 0.02, 0.04]),  # predictions
            np.array([[0.02, 0.08], [0.01, 0.05], [−0.01, 0.03], [0.00, 0.04], [0.01, 0.07]]),  # intervals
            RegimeState(MarketRegime.BULL_TREND, 0.8, 5, {}, {}, datetime.now())
        )
        system.methods = {'regime_mapie': mock_mapie}
        
        # Get predictions
        predictions = system.predict_with_uncertainty(X_batch, sample_market_data)
        
        # Verify batch predictions
        assert isinstance(predictions, list)
        assert len(predictions) == 5
        
        for pred in predictions:
            assert isinstance(pred, UncertaintyPrediction)
            assert isinstance(pred.mean_prediction, float)
            assert isinstance(pred.confidence_score, float)
    
    def test_prediction_caching(self, sample_data, sample_market_data):
        """Test prediction caching functionality."""
        X, y = sample_data
        X_single = X[:1]
        
        system = EnhancedUncertaintySystem(
            input_dim=X.shape[1],
            enable_methods={'regime_mapie': True, 'mc_dropout': False, 'deep_ensemble': False,
                          'quantile_regression': False, 'fusion_network': False},
            cache_predictions=True
        )
        
        # Mock method
        mock_mapie = Mock()
        mock_mapie.predict_with_regime_adaptation.return_value = (
            np.array([0.05]), np.array([[0.02, 0.08]]),
            RegimeState(MarketRegime.BULL_TREND, 0.8, 5, {}, {}, datetime.now())
        )
        system.methods = {'regime_mapie': mock_mapie}
        
        # First prediction (cache miss)
        pred1 = system.predict_with_uncertainty(X_single, sample_market_data, use_cache=True)
        assert system.cache_misses == 1
        assert system.cache_hits == 0
        
        # Second prediction with same input (cache hit)
        pred2 = system.predict_with_uncertainty(X_single, sample_market_data, use_cache=True)
        assert system.cache_hits == 1
        
        # Verify predictions are similar (timestamps will differ)
        assert abs(pred1.mean_prediction - pred2.mean_prediction) < 1e-6
    
    def test_validation_functionality(self, sample_data, sample_market_data):
        """Test prediction validation and performance metrics."""
        X, y = sample_data
        X_test, y_test = X[:50], y[:50]  # Small test set
        
        system = EnhancedUncertaintySystem(
            input_dim=X.shape[1],
            enable_methods={'regime_mapie': True, 'mc_dropout': False, 'deep_ensemble': False,
                          'quantile_regression': False, 'fusion_network': False}
        )
        
        # Mock method with reasonable predictions
        mock_mapie = Mock()
        # Create predictions close to true values with some noise
        mock_predictions = y_test + np.random.randn(len(y_test)) * 0.01
        mock_intervals = np.column_stack([mock_predictions - 0.05, mock_predictions + 0.05])
        
        mock_mapie.predict_with_regime_adaptation.return_value = (
            mock_predictions, mock_intervals,
            RegimeState(MarketRegime.BULL_TREND, 0.8, 5, {}, {}, datetime.now())
        )
        system.methods = {'regime_mapie': mock_mapie}
        
        # Validate predictions
        validation_results = system.validate_predictions(
            X_test, y_test, sample_market_data, store_history=True
        )
        
        # Verify validation results structure
        required_keys = ['prediction_accuracy', 'uncertainty_calibration', 'coverage_analysis', 
                        'method_performance', 'regime_analysis', 'summary']
        
        for key in required_keys:
            assert key in validation_results
        
        # Verify prediction accuracy metrics
        accuracy = validation_results['prediction_accuracy']
        assert 'fused_mae' in accuracy
        assert 'fused_rmse' in accuracy
        assert isinstance(accuracy['fused_mae'], float)
        assert accuracy['fused_mae'] >= 0
        
        # Verify coverage analysis
        coverage = validation_results['coverage_analysis']
        assert '95%' in coverage
        assert '80%' in coverage
        assert '50%' in coverage
        
        for level_key, level_data in coverage.items():
            assert 'empirical_coverage' in level_data
            assert 'target_coverage' in level_data
            assert 'coverage_error' in level_data
            assert 0 <= level_data['empirical_coverage'] <= 1
        
        # Verify validation history is stored
        assert len(system.validation_history) == 1
        assert system.validation_history[0] == validation_results
    
    def test_system_summary(self, sample_data, sample_market_data):
        """Test system summary and status reporting."""
        X, y = sample_data
        
        system = EnhancedUncertaintySystem(
            input_dim=X.shape[1],
            enable_methods={'regime_mapie': True, 'mc_dropout': True, 'deep_ensemble': False,
                          'quantile_regression': False, 'fusion_network': False},
            parallel_computation=True,
            cache_predictions=True
        )
        
        summary = system.get_system_summary()
        
        # Verify summary structure
        required_sections = ['system_configuration', 'cache_performance', 'method_status']
        for section in required_sections:
            assert section in summary
        
        # Verify system configuration
        config = summary['system_configuration']
        assert config['input_dim'] == X.shape[1]
        assert config['parallel_computation'] == True
        assert config['cache_enabled'] == True
        
        # Verify method status
        method_status = summary['method_status']
        for method_name, enabled in system.enable_methods.items():
            assert method_name in method_status
            assert method_status[method_name]['enabled'] == enabled
        
        # Verify cache performance (initially zero)
        cache_perf = summary['cache_performance']
        assert cache_perf['cache_hits'] == 0
        assert cache_perf['cache_misses'] == 0
        assert cache_perf['hit_rate'] == 0.0
    
    def test_error_handling(self, sample_data, sample_market_data):
        """Test system behavior with method failures."""
        X, y = sample_data
        X_single = X[:1]
        
        system = EnhancedUncertaintySystem(
            input_dim=X.shape[1],
            enable_methods={'regime_mapie': True, 'mc_dropout': True, 'deep_ensemble': False,
                          'quantile_regression': False, 'fusion_network': False}
        )
        
        # Mock methods with one failing
        mock_mapie = Mock()
        mock_mapie.predict_with_regime_adaptation.side_effect = Exception("MAPIE failed")
        
        mock_mc_dropout = Mock()
        mock_mc_dropout.predict_with_uncertainty.return_value = {
            'predictions': np.array([0.03]),
            'epistemic_uncertainty': np.array([0.01]),
            'prediction_intervals': np.array([[0.01, 0.05]]),
            'confidence': np.array([0.9])
        }
        
        system.methods = {'regime_mapie': mock_mapie, 'mc_dropout': mock_mc_dropout}
        
        # Should still work with one method failing
        prediction = system.predict_with_uncertainty(X_single, sample_market_data)
        
        assert isinstance(prediction, UncertaintyPrediction)
        assert prediction.mean_prediction is not None
        assert prediction.confidence_score >= 0
    
    def test_regime_adaptation(self, sample_data, sample_market_data):
        """Test regime-aware uncertainty adaptation."""
        X, y = sample_data
        X_single = X[:1]
        
        system = EnhancedUncertaintySystem(
            input_dim=X.shape[1],
            enable_methods={'regime_mapie': True, 'mc_dropout': False, 'deep_ensemble': False,
                          'quantile_regression': False, 'fusion_network': False}
        )
        
        # Test different regimes
        regimes = [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND, MarketRegime.HIGH_VOLATILITY]
        predictions_by_regime = {}
        
        for regime in regimes:
            mock_mapie = Mock()
            mock_mapie.predict_with_regime_adaptation.return_value = (
                np.array([0.05]), np.array([[0.02, 0.08]]),
                RegimeState(regime, 0.8, 5, {}, {}, datetime.now())
            )
            system.methods = {'regime_mapie': mock_mapie}
            
            pred = system.predict_with_uncertainty(X_single, sample_market_data, use_cache=False)
            predictions_by_regime[regime.value] = pred
        
        # Verify regime information is captured
        for regime_name, pred in predictions_by_regime.items():
            assert pred.market_regime == regime_name
            assert pred.regime_confidence > 0


class TestIndividualComponents:
    """Test individual uncertainty components."""
    
    @pytest.fixture
    def small_data(self):
        """Small dataset for component testing."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = X.sum(axis=1) + np.random.randn(100) * 0.1
        return X, y
    
    def test_monte_carlo_dropout(self, small_data):
        """Test Monte Carlo Dropout uncertainty."""
        X, y = small_data
        
        mc_dropout = MonteCarloDropoutUncertainty(
            input_dim=X.shape[1],
            hidden_dims=[32, 16],
            n_samples=10,  # Reduced for testing speed
            dropout_rate=0.2
        )
        
        # Quick training
        history = mc_dropout.fit(X, y, epochs=5, verbose=False)
        assert 'train_loss' in history
        assert 'val_loss' in history
        
        # Test prediction
        results = mc_dropout.predict_with_uncertainty(X[:5])
        
        assert 'predictions' in results
        assert 'epistemic_uncertainty' in results
        assert 'prediction_intervals' in results
        assert 'confidence' in results
        
        # Check shapes
        assert results['predictions'].shape[0] == 5
        assert results['epistemic_uncertainty'].shape[0] == 5
    
    def test_deep_ensemble(self, small_data):
        """Test Deep Ensemble uncertainty."""
        X, y = small_data
        
        ensemble = DeepEnsembleUncertainty(
            input_dim=X.shape[1],
            n_models=3,  # Reduced for testing
            hidden_dims=[32, 16],
            ensemble_method='mean_var'
        )
        
        # Quick training
        history = ensemble.fit(X, y, epochs=5, verbose=False)
        assert 'train_losses' in history
        assert len(history['train_losses']) == 3  # One per model
        
        # Test prediction
        results = ensemble.predict_with_uncertainty(X[:5])
        
        assert 'predictions' in results
        assert 'epistemic_uncertainty' in results
        assert 'aleatoric_uncertainty' in results
        assert 'total_uncertainty' in results
        
        # Verify uncertainty decomposition
        total_unc = results['total_uncertainty']
        epistemic_unc = results['epistemic_uncertainty']
        aleatoric_unc = results['aleatoric_uncertainty']
        
        # Total should be >= individual components
        combined_unc = np.sqrt(epistemic_unc**2 + aleatoric_unc**2)
        assert np.allclose(total_unc, combined_unc, rtol=1e-5)
    
    def test_quantile_regressor(self, small_data):
        """Test Quantile Regression uncertainty."""
        X, y = small_data
        
        quantile_reg = QuantileRegressor(
            input_dim=X.shape[1],
            hidden_dims=[32, 16],
            quantiles=[0.1, 0.5, 0.9]  # Simplified quantiles
        )
        
        # Quick training
        history = quantile_reg.fit(X, y, epochs=5, verbose=False)
        assert 'train_loss' in history
        
        # Test prediction
        results = quantile_reg.predict_with_intervals(X[:5], confidence_level=0.8)
        
        assert 'predictions' in results
        assert 'quantile_predictions' in results
        assert 'prediction_intervals' in results
        assert 'uncertainty' in results
        
        # Check quantile ordering
        quantile_preds = results['quantile_predictions']
        for i in range(quantile_preds.shape[0]):
            # Should be ordered: q0.1 <= q0.5 <= q0.9
            assert quantile_preds[i, 0] <= quantile_preds[i, 1] <= quantile_preds[i, 2]
    
    def test_fusion_network(self, small_data):
        """Test Uncertainty Fusion Network."""
        X, y = small_data
        
        # Create mock method predictions
        n_methods = 3
        method_predictions = []
        method_uncertainties = []
        
        for i in range(n_methods):
            pred = y + np.random.randn(len(y)) * 0.05  # Slightly different predictions
            unc = np.abs(np.random.randn(len(y)) * 0.02) + 0.01  # Small uncertainties
            method_predictions.append(pred)
            method_uncertainties.append(unc)
        
        fusion_net = UncertaintyFusionNetwork(
            input_dim=X.shape[1],
            n_uncertainty_methods=n_methods,
            hidden_dims=[16, 8],
            fusion_strategy='learned_weights'
        )
        
        # Train fusion network
        history = fusion_net.fit(
            X, y, method_predictions, method_uncertainties,
            epochs=5, verbose=False
        )
        assert 'train_loss' in history
        
        # Test fusion
        method_intervals = []
        for i in range(n_methods):
            lower = method_predictions[i] - 1.96 * method_uncertainties[i]
            upper = method_predictions[i] + 1.96 * method_uncertainties[i]
            method_intervals.append(np.column_stack([lower, upper]))
        
        results = fusion_net.fuse_predictions(
            X[:10], 
            [pred[:10] for pred in method_predictions],
            [unc[:10] for unc in method_uncertainties],
            [interval[:10] for interval in method_intervals]
        )
        
        assert 'fused_predictions' in results
        assert 'fused_uncertainty' in results
        assert 'fusion_weights' in results
        assert 'method_reliability' in results
        
        # Check fusion weights sum to 1
        fusion_weights = results['fusion_weights']
        weight_sums = np.sum(fusion_weights, axis=1)
        assert np.allclose(weight_sums, 1.0, rtol=1e-5)


if __name__ == "__main__":
    # Run specific test for development
    import sys
    sys.path.append('../..')
    
    # Example usage
    pytest.main([__file__ + "::TestEnhancedUncertaintySystem::test_system_initialization", "-v"])