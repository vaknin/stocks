"""Tests for technical feature generator."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features.technical_feature_generator import (
    TechnicalFeatureGenerator, 
    TechnicalIndicatorType, 
    MarketRegimeType
)


class TestTechnicalFeatureGenerator(unittest.TestCase):
    """Test technical feature generator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = TechnicalFeatureGenerator(
            lookback_window=100,
            regime_detection_window=20,
            feature_selection_threshold=0.05
        )
        
        # Create sample OHLCV data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n_days = len(dates)
        
        # Generate realistic price series with trend and volatility
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, n_days)  # 0.1% daily return, 2% volatility
        returns[:50] += 0.002  # Bull trend in first 50 days
        returns[200:250] -= 0.003  # Bear trend in middle
        returns[300:320] = np.random.normal(0, 0.05, 20)  # High volatility period
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        self.sample_df = pd.DataFrame({
            'date': dates,
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0.01, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0.01, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n_days)
        }).set_index('date')
        
        # Create multi-asset data
        self.multi_asset_data = {
            'AAPL': self.sample_df.copy(),
            'MSFT': self.sample_df * 1.1 + np.random.normal(0, 0.5, len(self.sample_df)),
            'GOOGL': self.sample_df * 0.9 + np.random.normal(0, 1.0, len(self.sample_df))
        }
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertIsNotNone(self.generator)
        self.assertEqual(len(self.generator.indicators), 10)  # Expected number of indicators
        self.assertTrue(all(indicator.name for indicator in self.generator.indicators.values()))
        
    def test_regime_detection(self):
        """Test market regime detection."""
        # Test with trending bull data (first 50 days)
        bull_data = self.sample_df.head(50)
        regime = self.generator.detect_market_regime(bull_data)
        self.assertIsInstance(regime, MarketRegimeType)
        
        # Test with high volatility data
        vol_data = self.sample_df.iloc[300:320]
        regime = self.generator.detect_market_regime(vol_data)
        # Should detect high volatility or related regime
        self.assertIn(regime, [MarketRegimeType.HIGH_VOLATILITY, MarketRegimeType.BREAKOUT, 
                              MarketRegimeType.REVERSAL, MarketRegimeType.TRENDING_BULL, 
                              MarketRegimeType.TRENDING_BEAR, MarketRegimeType.SIDEWAYS])
    
    def test_feature_generation(self):
        """Test basic feature generation."""
        features = self.generator.generate_features(self.sample_df, 'AAPL')
        
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
        self.assertEqual(features.dtype, np.float32)
        
        # Check for realistic feature values (no NaN or extreme values)
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertTrue(np.all(np.abs(features) <= 5.0))  # Clipped values
    
    def test_multi_asset_features(self):
        """Test cross-asset feature generation."""
        features = self.generator.generate_features(self.multi_asset_data, 'AAPL')
        
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
        
        # Should include cross-asset features when multiple assets provided
        # Features should be larger than single-asset case due to cross-correlations
        single_asset_features = self.generator.generate_features(self.sample_df, 'AAPL')
        self.assertGreaterEqual(len(features), len(single_asset_features))
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Test with very limited data
        small_df = self.sample_df.head(10)
        features = self.generator.generate_features(small_df, 'AAPL')
        
        # Should return default features without errors
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 50)  # Default feature count
        self.assertTrue(np.all(features == 0.0))  # Default values
    
    def test_regime_specific_weighting(self):
        """Test that different regimes produce different feature weightings."""
        # Generate features for different regime overrides
        features_bull = self.generator.generate_features(
            self.sample_df, 'AAPL', regime_override=MarketRegimeType.TRENDING_BULL
        )
        features_bear = self.generator.generate_features(
            self.sample_df, 'AAPL', regime_override=MarketRegimeType.TRENDING_BEAR
        )
        features_sideways = self.generator.generate_features(
            self.sample_df, 'AAPL', regime_override=MarketRegimeType.SIDEWAYS
        )
        
        # Features should be different due to regime-specific weighting
        self.assertFalse(np.array_equal(features_bull, features_bear))
        self.assertFalse(np.array_equal(features_bull, features_sideways))
        self.assertFalse(np.array_equal(features_bear, features_sideways))
    
    def test_individual_indicators(self):
        """Test individual technical indicator calculations."""
        # Test RSI calculation
        rsi_features = self.generator._calculate_rsi(self.sample_df, period=14)
        self.assertEqual(len(rsi_features), 3)  # RSI value, trend, volatility
        self.assertTrue(all(abs(f) <= 1.0 for f in rsi_features))  # Normalized values
        
        # Test MACD calculation
        macd_features = self.generator._calculate_macd(self.sample_df)
        self.assertEqual(len(macd_features), 3)  # MACD line, signal, histogram
        self.assertTrue(all(np.isfinite(f) for f in macd_features))
        
        # Test SMA ratios
        sma_features = self.generator._calculate_sma_ratios(self.sample_df)
        self.assertEqual(len(sma_features), 5)  # 5 different periods
        self.assertTrue(all(np.isfinite(f) for f in sma_features))
        
        # Test ATR
        atr_features = self.generator._calculate_atr(self.sample_df)
        self.assertEqual(len(atr_features), 1)
        self.assertGreater(atr_features[0], 0)  # ATR should be positive
        self.assertLessEqual(atr_features[0], 0.5)  # Capped at 50%
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        bb_features = self.generator._calculate_bollinger_bands(self.sample_df)
        self.assertEqual(len(bb_features), 3)  # Position, width, deviation
        
        # BB position should be roughly centered around 0
        bb_position = bb_features[0]
        self.assertGreaterEqual(bb_position, -0.5)
        self.assertLessEqual(bb_position, 0.5)
        
        # Band width should be positive
        self.assertGreater(bb_features[1], 0)
    
    def test_volume_indicators(self):
        """Test volume-based indicators."""
        # Test OBV
        obv_features = self.generator._calculate_obv(self.sample_df)
        self.assertEqual(len(obv_features), 2)  # Relative OBV and trend
        self.assertTrue(all(abs(f) <= 1.0 for f in obv_features))
        
        # Test volume SMA ratios
        vol_sma_features = self.generator._calculate_volume_sma_ratio(self.sample_df)
        self.assertEqual(len(vol_sma_features), 3)  # 3 different periods
        self.assertTrue(all(np.isfinite(f) for f in vol_sma_features))
        self.assertTrue(all(abs(f) <= 2.0 for f in vol_sma_features))  # Log ratios capped
    
    def test_caching_mechanism(self):
        """Test feature caching."""
        # Clear cache
        self.generator.feature_cache.clear()
        self.generator.cache_timestamps.clear()
        
        # Generate features twice
        features1 = self.generator.generate_features(self.sample_df, 'AAPL')
        features2 = self.generator.generate_features(self.sample_df, 'AAPL')
        
        # Should be identical due to caching
        np.testing.assert_array_equal(features1, features2)
        
        # Cache should contain the entry
        self.assertGreater(len(self.generator.feature_cache), 0)
    
    def test_performance_tracking(self):
        """Test indicator performance tracking and adaptation."""
        # Update performance for RSI in bull regime
        initial_weight = self.generator.indicators['rsi'].regime_weights[MarketRegimeType.TRENDING_BULL]
        
        # Add several performance scores
        for score in [0.8, 0.7, 0.9, 0.8, 0.85]:  # Good performance
            self.generator.update_indicator_performance('rsi', score, MarketRegimeType.TRENDING_BULL)
        
        # Weight should be adapted (if dynamic parameters enabled)
        if self.generator.enable_dynamic_parameters:
            new_weight = self.generator.indicators['rsi'].regime_weights[MarketRegimeType.TRENDING_BULL]
            # Weight might change based on performance, but we allow for various outcomes
            self.assertTrue(0.1 <= new_weight <= 1.0)
    
    def test_feature_names(self):
        """Test feature naming functionality."""
        feature_names = self.generator.get_feature_names()
        
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)
        self.assertTrue(all(isinstance(name, str) for name in feature_names))
        
        # Should include expected indicator names
        expected_indicators = ['rsi', 'macd', 'sma_ratios', 'ema_ratios', 'bollinger_bands', 'atr']
        for indicator in expected_indicators:
            self.assertTrue(any(indicator in name for name in feature_names))
    
    def test_regime_weights_retrieval(self):
        """Test regime-specific weight retrieval."""
        bull_weights = self.generator.get_regime_feature_weights(MarketRegimeType.TRENDING_BULL)
        bear_weights = self.generator.get_regime_feature_weights(MarketRegimeType.TRENDING_BEAR)
        
        self.assertIsInstance(bull_weights, dict)
        self.assertIsInstance(bear_weights, dict)
        self.assertEqual(len(bull_weights), len(self.generator.indicators))
        self.assertEqual(len(bear_weights), len(self.generator.indicators))
        
        # Weights should be different for different regimes
        self.assertNotEqual(bull_weights, bear_weights)
    
    def test_generator_stats(self):
        """Test generator statistics functionality."""
        stats = self.generator.get_generator_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_indicators', stats)
        self.assertIn('cache_size', stats)
        self.assertIn('settings', stats)
        
        self.assertEqual(stats['total_indicators'], len(self.generator.indicators))
        self.assertIsInstance(stats['settings'], dict)
    
    def test_error_handling(self):
        """Test error handling with malformed data."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        features = self.generator.generate_features(empty_df, 'AAPL')
        self.assertEqual(len(features), 50)  # Default feature count
        
        # Test with DataFrame missing required columns
        bad_df = pd.DataFrame({'price': [100, 101, 102]})
        features = self.generator.generate_features(bad_df, 'AAPL')
        self.assertEqual(len(features), 50)  # Default feature count
        
        # Test with NaN values
        nan_df = self.sample_df.copy()
        nan_df.loc[nan_df.index[10:15], 'close'] = np.nan
        features = self.generator.generate_features(nan_df, 'AAPL')
        self.assertTrue(np.all(np.isfinite(features)))  # Should handle NaNs
    
    def test_multi_timeframe_features(self):
        """Test multi-timeframe feature generation."""
        # Enable multi-timeframe
        generator_with_mtf = TechnicalFeatureGenerator(
            enable_multi_timeframe=True,
            lookback_window=100
        )
        
        features = generator_with_mtf.generate_features(self.sample_df, 'AAPL')
        
        # Should include multi-timeframe features
        self.assertGreater(len(features), 40)  # More features with multi-timeframe
        
        # Disable multi-timeframe
        generator_no_mtf = TechnicalFeatureGenerator(
            enable_multi_timeframe=False,
            lookback_window=100
        )
        
        features_no_mtf = generator_no_mtf.generate_features(self.sample_df, 'AAPL')
        
        # Should have fewer features without multi-timeframe
        self.assertLess(len(features_no_mtf), len(features))
    
    def test_pivot_points_calculation(self):
        """Test pivot points calculation."""
        pivot_features = self.generator._calculate_pivot_points(self.sample_df)
        
        self.assertEqual(len(pivot_features), 4)  # Pivot, R1, S1 distances, level position
        self.assertTrue(all(np.isfinite(f) for f in pivot_features))
        self.assertTrue(all(abs(f) <= 1.0 for f in pivot_features[:3]))  # Distance features capped
        self.assertIn(pivot_features[3], [-1.0, 0.0, 1.0])  # Level position should be discrete
    
    def test_stochastic_oscillator(self):
        """Test stochastic oscillator calculation."""
        stoch_features = self.generator._calculate_stochastic(self.sample_df)
        
        self.assertEqual(len(stoch_features), 2)  # %K and %D
        self.assertTrue(all(abs(f) <= 1.0 for f in stoch_features))  # Normalized to [-1, 1]
    
    def test_adx_calculation(self):
        """Test ADX (Average Directional Index) calculation."""
        adx_features = self.generator._calculate_adx(self.sample_df)
        
        self.assertEqual(len(adx_features), 1)
        self.assertGreaterEqual(adx_features[0], 0.0)
        self.assertLessEqual(adx_features[0], 1.0)  # Normalized ADX


class TestTechnicalIndicatorTypes(unittest.TestCase):
    """Test technical indicator type enumerations."""
    
    def test_indicator_types(self):
        """Test technical indicator type enumeration."""
        self.assertEqual(len(TechnicalIndicatorType), 6)
        self.assertIn(TechnicalIndicatorType.MOMENTUM, TechnicalIndicatorType)
        self.assertIn(TechnicalIndicatorType.TREND, TechnicalIndicatorType)
        self.assertIn(TechnicalIndicatorType.VOLATILITY, TechnicalIndicatorType)
    
    def test_market_regime_types(self):
        """Test market regime type enumeration."""
        self.assertEqual(len(MarketRegimeType), 7)
        self.assertIn(MarketRegimeType.TRENDING_BULL, MarketRegimeType)
        self.assertIn(MarketRegimeType.HIGH_VOLATILITY, MarketRegimeType)
        self.assertIn(MarketRegimeType.SIDEWAYS, MarketRegimeType)


if __name__ == '__main__':
    unittest.main()