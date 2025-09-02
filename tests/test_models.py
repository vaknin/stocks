"""Unit tests for AI models."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.timesfm_predictor import TimesFMPredictor
from models.tsmamba_predictor import TSMambaPredictor
from models.samba_predictor import SAMBAPredictor


class TestTimesFMPredictor(unittest.TestCase):
    """Test TimesFM model functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = TimesFMPredictor()
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        self.sample_data = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(100) * 0.02),
            'high': 100 + np.cumsum(np.random.randn(100) * 0.02) + np.abs(np.random.randn(100) * 0.5),
            'low': 100 + np.cumsum(np.random.randn(100) * 0.02) - np.abs(np.random.randn(100) * 0.5),
            'close': 100 + np.cumsum(np.random.randn(100) * 0.02),
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        self.sample_data['high'] = np.maximum(self.sample_data['high'], self.sample_data[['open', 'close']].max(axis=1))
        self.sample_data['low'] = np.minimum(self.sample_data['low'], self.sample_data[['open', 'close']].min(axis=1))
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.predictor, TimesFMPredictor)
        self.assertIn('horizon', self.predictor.horizon_len)
    
    def test_prediction_structure(self):
        """Test prediction output structure."""
        result = self.predictor.predict(self.sample_data, "TEST")
        
        self.assertIsInstance(result, dict)
        
        # Check that we have predictions for each horizon
        for horizon in self.predictor.horizon_len:
            horizon_key = f"horizon_{horizon}"
            self.assertIn(horizon_key, result)
            
            horizon_result = result[horizon_key]
            self.assertIn('prediction', horizon_result)
            self.assertIn('confidence', horizon_result)
            self.assertIn('prediction_interval', horizon_result)
            
            # Check value ranges
            self.assertIsInstance(horizon_result['prediction'], float)
            self.assertGreaterEqual(horizon_result['confidence'], 0.0)
            self.assertLessEqual(horizon_result['confidence'], 1.0)
            self.assertIsInstance(horizon_result['prediction_interval'], list)
            self.assertEqual(len(horizon_result['prediction_interval']), 2)
    
    def test_prediction_bounds(self):
        """Test that predictions are within reasonable bounds."""
        result = self.predictor.predict(self.sample_data, "TEST")
        
        for horizon_key, horizon_result in result.items():
            prediction = horizon_result['prediction']
            # Stock returns should typically be within ±50% for short-term predictions
            self.assertGreaterEqual(prediction, -0.5)
            self.assertLessEqual(prediction, 0.5)
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        short_data = self.sample_data.head(10)  # Very short dataset
        result = self.predictor.predict(short_data, "TEST")
        
        # Should still return a result (mock prediction)
        self.assertIsInstance(result, dict)
        for horizon_key in result:
            self.assertIn('prediction', result[horizon_key])
    
    def test_model_info(self):
        """Test model information retrieval."""
        info = self.predictor.get_model_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('model_name', info)
        self.assertIn('model_type', info)
        self.assertIn('is_loaded', info)
        self.assertEqual(info['model_name'], 'TimesFM')


class TestTSMambaPredictor(unittest.TestCase):
    """Test TSMamba model functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = TSMambaPredictor()
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        self.sample_data = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(100) * 0.02),
            'high': 100 + np.cumsum(np.random.randn(100) * 0.02) + np.abs(np.random.randn(100) * 0.5),
            'low': 100 + np.cumsum(np.random.randn(100) * 0.02) - np.abs(np.random.randn(100) * 0.5),
            'close': 100 + np.cumsum(np.random.randn(100) * 0.02),
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        self.sample_data['high'] = np.maximum(self.sample_data['high'], self.sample_data[['open', 'close']].max(axis=1))
        self.sample_data['low'] = np.minimum(self.sample_data['low'], self.sample_data[['open', 'close']].min(axis=1))
    
    def test_initialization(self):
        """Test TSMamba initialization."""
        self.assertIsInstance(self.predictor, TSMambaPredictor)
        self.assertEqual(self.predictor.input_dim, 5)  # OHLCV
        self.assertEqual(self.predictor.seq_len, 60)
    
    def test_feature_preparation(self):
        """Test feature preparation pipeline."""
        features = self.predictor._prepare_features(self.sample_data)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 5)  # Should have more than just OHLCV
        
        # Check for technical indicators
        expected_features = ['sma_10', 'sma_30', 'rsi', 'volatility', 'returns']
        for feature in expected_features:
            self.assertIn(feature, features.columns)
    
    def test_prediction_output(self):
        """Test TSMamba prediction output."""
        result = self.predictor.predict(self.sample_data, "TEST")
        
        self.assertIsInstance(result, dict)
        for horizon in self.predictor.horizon_len:
            horizon_key = f"horizon_{horizon}"
            self.assertIn(horizon_key, result)
            
            prediction = result[horizon_key]
            self.assertIn('prediction', prediction)
            self.assertIn('confidence', prediction)
            self.assertIn('prediction_interval', prediction)


class TestSAMBAPredictor(unittest.TestCase):
    """Test SAMBA model functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = SAMBAPredictor()
        
        # Create sample data for multiple tickers
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        np.random.seed(42)
        
        self.sample_data_dict = {}
        for ticker in ['AAPL', 'MSFT', 'GOOGL']:
            self.sample_data_dict[ticker] = pd.DataFrame({
                'open': 100 + np.cumsum(np.random.randn(50) * 0.02),
                'high': 100 + np.cumsum(np.random.randn(50) * 0.02) + np.abs(np.random.randn(50) * 0.5),
                'low': 100 + np.cumsum(np.random.randn(50) * 0.02) - np.abs(np.random.randn(50) * 0.5),
                'close': 100 + np.cumsum(np.random.randn(50) * 0.02),
                'volume': np.random.randint(1000000, 10000000, 50)
            }, index=dates)
            
            # Ensure high >= max(open, close) and low <= min(open, close)
            data = self.sample_data_dict[ticker]
            data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
            data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    def test_initialization(self):
        """Test SAMBA initialization."""
        self.assertIsInstance(self.predictor, SAMBAPredictor)
        self.assertEqual(self.predictor.correlation_threshold, 0.3)
        self.assertIsNotNone(self.predictor.graph_processor)
    
    def test_correlation_calculation(self):
        """Test correlation matrix calculation."""
        correlation_matrix = self.predictor.graph_processor.calculate_correlation_matrix(
            self.sample_data_dict
        )
        
        self.assertIsInstance(correlation_matrix, np.ndarray)
        self.assertEqual(correlation_matrix.shape[0], correlation_matrix.shape[1])
        self.assertEqual(correlation_matrix.shape[0], len(self.sample_data_dict))
        
        # Diagonal should be 1 (self-correlation)
        np.testing.assert_array_almost_equal(np.diag(correlation_matrix), np.ones(len(self.sample_data_dict)))
    
    def test_graph_features(self):
        """Test graph feature creation."""
        correlation_matrix = self.predictor.graph_processor.calculate_correlation_matrix(
            self.sample_data_dict
        )
        graph_features = self.predictor.graph_processor.create_graph_features(correlation_matrix)
        
        expected_keys = ['adjacency', 'correlation', 'degree', 'clustering', 'centrality']
        for key in expected_keys:
            self.assertIn(key, graph_features)
            self.assertIsInstance(graph_features[key], np.ndarray)
    
    def test_samba_prediction(self):
        """Test SAMBA prediction with multiple stocks."""
        result = self.predictor.predict(self.sample_data_dict, "AAPL")
        
        self.assertIsInstance(result, dict)
        for horizon in self.predictor.horizon_len:
            horizon_key = f"horizon_{horizon}"
            self.assertIn(horizon_key, result)
            
            prediction = result[horizon_key]
            self.assertIn('prediction', prediction)
            self.assertIn('confidence', prediction)
            self.assertIn('prediction_interval', prediction)
            self.assertIn('graph_features', prediction)
            
            # Test graph features structure
            graph_feat = prediction['graph_features']
            self.assertIn('centrality', graph_feat)
            self.assertIn('clustering', graph_feat)
            self.assertIn('degree', graph_feat)
    
    def test_model_info(self):
        """Test SAMBA model info."""
        info = self.predictor.get_model_info()
        
        self.assertIsInstance(info, dict)
        self.assertEqual(info['model_name'], 'SAMBA (Graph-Mamba)')
        self.assertIn('correlation_threshold', info)
        self.assertIn('is_loaded', info)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestTimesFMPredictor))
    test_suite.addTest(unittest.makeSuite(TestTSMambaPredictor))
    test_suite.addTest(unittest.makeSuite(TestSAMBAPredictor))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"\n✅ All tests passed! ({result.testsRun} tests)")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests")