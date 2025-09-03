"""Tests for microstructure features extractor."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features.microstructure_features import (
    MicrostructureFeatureExtractor,
    VolumeProfileType,
    MicrostructureRegime,
    VolumeNode,
    MarketMicrostructure
)


class TestMicrostructureFeatureExtractor(unittest.TestCase):
    """Test microstructure feature extractor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = MicrostructureFeatureExtractor(
            volume_profile_bins=20,
            lookback_periods=[10, 20, 50],
            liquidity_window=10
        )
        
        # Create realistic market data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
        n_days = len(dates)
        
        # Generate realistic price and volume data
        base_price = 100.0
        base_volume = 1000000
        
        # Create different market regimes in the data
        prices = [base_price]
        volumes = []
        
        for i in range(n_days):
            # Different microstructure regimes at different periods
            if i < 50:  # High liquidity period
                vol_multiplier = 1.5 + np.random.normal(0, 0.2)
                price_volatility = 0.01
            elif i < 100:  # Low liquidity period
                vol_multiplier = 0.5 + np.random.normal(0, 0.1)
                price_volatility = 0.03
            elif i < 130:  # Price discovery period
                vol_multiplier = 2.0 + np.random.normal(0, 0.3)
                price_volatility = 0.025
            else:  # Consolidation period
                vol_multiplier = 1.0 + np.random.normal(0, 0.15)
                price_volatility = 0.015
            
            # Generate volume
            volume = max(100000, int(base_volume * vol_multiplier))
            volumes.append(volume)
            
            # Generate price with regime-specific characteristics
            price_change = np.random.normal(0.001, price_volatility)
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(1.0, new_price))  # Ensure positive prices
        
        prices = prices[:-1]  # Remove extra element
        
        # Create OHLCV data with realistic spreads
        self.sample_df = pd.DataFrame({
            'date': dates,
            'open': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0.005, 0.003))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0.005, 0.003))) for p in prices],
            'close': prices,
            'volume': volumes
        }).set_index('date')
        
        # Ensure high >= low
        for i in range(len(self.sample_df)):
            if self.sample_df['high'].iloc[i] < self.sample_df['low'].iloc[i]:
                temp = self.sample_df['high'].iloc[i]
                self.sample_df.iloc[i, self.sample_df.columns.get_loc('high')] = self.sample_df['low'].iloc[i]
                self.sample_df.iloc[i, self.sample_df.columns.get_loc('low')] = temp
    
    def test_initialization(self):
        """Test extractor initialization."""
        self.assertIsNotNone(self.extractor)
        self.assertEqual(self.extractor.volume_profile_bins, 20)
        self.assertEqual(len(self.extractor.lookback_periods), 3)
        self.assertTrue(self.extractor.enable_regime_detection)
    
    def test_feature_extraction_basic(self):
        """Test basic feature extraction."""
        features = self.extractor.extract_features(self.sample_df, 'AAPL')
        
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
        self.assertEqual(features.dtype, np.float32)
        
        # Check for valid feature values
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertTrue(np.all(np.abs(features) <= 10.0))  # Within expected bounds
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        small_df = self.sample_df.head(5)
        features = self.extractor.extract_features(small_df, 'AAPL')
        
        # Should return default features
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 40)  # Default feature count
    
    def test_volume_profile_creation(self):
        """Test volume profile creation."""
        recent_data = self.sample_df.tail(20)
        volume_profile = self.extractor._create_volume_profile(recent_data)
        
        self.assertIsInstance(volume_profile, list)
        if volume_profile:  # If profile was created
            self.assertTrue(all(isinstance(node, VolumeNode) for node in volume_profile))
            self.assertTrue(all(node.total_volume >= 0 for node in volume_profile))
            
            # Check that buy + sell volume equals total volume (approximately)
            for node in volume_profile:
                if node.total_volume > 0:
                    volume_sum = node.buy_volume + node.sell_volume
                    self.assertAlmostEqual(volume_sum, node.total_volume, delta=0.1 * node.total_volume)
    
    def test_volume_profile_levels(self):
        """Test POC, VAH, VAL calculations."""
        recent_data = self.sample_df.tail(20)
        volume_profile = self.extractor._create_volume_profile(recent_data)
        
        if volume_profile:
            poc, vah, val = self.extractor._calculate_volume_profile_levels(volume_profile)
            
            # POC, VAH, VAL should be valid prices
            self.assertGreater(poc, 0)
            self.assertGreater(vah, 0) 
            self.assertGreater(val, 0)
            
            # VAH should be >= VAL (Value Area High >= Value Area Low)
            self.assertGreaterEqual(vah, val)
            
            # POC should be within the value area
            self.assertGreaterEqual(poc, val)
            self.assertLessEqual(poc, vah)
    
    def test_volume_profile_features(self):
        """Test volume profile feature extraction."""
        vp_features = self.extractor._extract_volume_profile_features(self.sample_df)
        
        # Should have features for each lookback period
        expected_length = len(self.extractor.lookback_periods) * 6  # 6 features per period
        self.assertEqual(len(vp_features), expected_length)
        
        # All features should be finite and reasonable
        self.assertTrue(all(np.isfinite(f) for f in vp_features))
        self.assertTrue(all(abs(f) <= 5.0 for f in vp_features))
    
    def test_tick_approximation_features(self):
        """Test tick-level approximation features."""
        tick_features = self.extractor._extract_tick_approximation_features(self.sample_df)
        
        self.assertEqual(len(tick_features), 15)  # Expected number of tick features
        self.assertTrue(all(np.isfinite(f) for f in tick_features))
        
        # Specific feature tests
        tick_count_avg = tick_features[0]
        self.assertGreaterEqual(tick_count_avg, 0)  # Should be non-negative
        
        # Price impact should be reasonable
        price_impact = tick_features[2]
        self.assertGreaterEqual(price_impact, 0)
        self.assertLessEqual(price_impact, 1.0)
    
    def test_liquidity_features(self):
        """Test liquidity feature extraction."""
        liquidity_features = self.extractor._extract_liquidity_features(self.sample_df)
        
        self.assertEqual(len(liquidity_features), 8)  # Expected number of liquidity features
        self.assertTrue(all(np.isfinite(f) for f in liquidity_features))
        
        # Illiquidity measure should be reasonable
        illiquidity = liquidity_features[0]
        self.assertGreater(illiquidity, 0)
        self.assertLessEqual(illiquidity, 10.0)
        
        # Liquidity ratio should be positive
        liquidity_ratio = liquidity_features[2]
        self.assertGreaterEqual(liquidity_ratio, 0)
    
    def test_order_flow_features(self):
        """Test order flow feature extraction."""
        order_flow_features = self.extractor._extract_order_flow_features(self.sample_df)
        
        self.assertEqual(len(order_flow_features), 6)  # Expected number of order flow features
        self.assertTrue(all(np.isfinite(f) for f in order_flow_features))
        
        # Order flow imbalance should be in [-1, 1]
        imbalance = order_flow_features[0]
        self.assertGreaterEqual(imbalance, -1.0)
        self.assertLessEqual(imbalance, 1.0)
        
        # PIN approximation should be in [0, 1]
        pin = order_flow_features[-1]
        self.assertGreaterEqual(pin, 0.0)
        self.assertLessEqual(pin, 1.0)
    
    def test_price_discovery_features(self):
        """Test price discovery feature extraction."""
        discovery_features = self.extractor._extract_price_discovery_features(self.sample_df)
        
        self.assertEqual(len(discovery_features), 5)  # Expected number of discovery features
        self.assertTrue(all(np.isfinite(f) for f in discovery_features))
        
        # Efficiency score should be in [0, 1]
        efficiency = discovery_features[0]
        self.assertGreaterEqual(efficiency, 0.0)
        self.assertLessEqual(efficiency, 1.0)
        
        # All features should be in reasonable bounds
        self.assertTrue(all(0.0 <= f <= 1.0 for f in discovery_features))
    
    def test_regime_detection(self):
        """Test microstructure regime detection."""
        # Test different parts of the sample data
        high_liquidity_data = self.sample_df.head(50)  # High liquidity period
        low_liquidity_data = self.sample_df.iloc[50:100]  # Low liquidity period
        discovery_data = self.sample_df.iloc[100:130]  # Price discovery period
        
        regime1 = self.extractor._detect_microstructure_regime(high_liquidity_data)
        regime2 = self.extractor._detect_microstructure_regime(low_liquidity_data)
        regime3 = self.extractor._detect_microstructure_regime(discovery_data)
        
        # All should be valid regime types
        self.assertIsInstance(regime1, MicrostructureRegime)
        self.assertIsInstance(regime2, MicrostructureRegime)
        self.assertIsInstance(regime3, MicrostructureRegime)
        
        # Regimes could be different (though not guaranteed due to complexity)
        self.assertIn(regime1, list(MicrostructureRegime))
        self.assertIn(regime2, list(MicrostructureRegime))
        self.assertIn(regime3, list(MicrostructureRegime))
    
    def test_regime_features(self):
        """Test regime feature extraction."""
        regime_features = self.extractor._extract_regime_features(self.sample_df)
        
        self.assertEqual(len(regime_features), 4)  # 3 regime indicators + stability
        self.assertTrue(all(np.isfinite(f) for f in regime_features))
        
        # First three should be regime indicators (0 or 1)
        for i in range(3):
            self.assertIn(regime_features[i], [0.0, 1.0])
        
        # Stability should be in [0, 1]
        stability = regime_features[3]
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
    
    def test_feature_caching(self):
        """Test feature caching mechanism."""
        # Clear cache
        self.extractor.feature_cache.clear()
        self.extractor.cache_timestamps.clear()
        
        # Extract features twice
        features1 = self.extractor.extract_features(self.sample_df, 'AAPL')
        features2 = self.extractor.extract_features(self.sample_df, 'AAPL')
        
        # Should be identical due to caching
        np.testing.assert_array_equal(features1, features2)
        
        # Cache should contain entry
        self.assertGreater(len(self.extractor.feature_cache), 0)
    
    def test_all_feature_types(self):
        """Test extraction with all feature types enabled."""
        features = self.extractor.extract_features(
            self.sample_df, 
            'AAPL',
            include_volume_profile=True,
            include_tick_features=True,
            include_liquidity_features=True
        )
        
        # Should have more features than partial extraction
        features_partial = self.extractor.extract_features(
            self.sample_df,
            'AAPL', 
            include_volume_profile=False,
            include_tick_features=False,
            include_liquidity_features=False
        )
        
        self.assertGreater(len(features), len(features_partial))
    
    def test_feature_names(self):
        """Test feature naming functionality."""
        feature_names = self.extractor.get_feature_names()
        
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)
        self.assertTrue(all(isinstance(name, str) for name in feature_names))
        
        # Should include expected feature categories
        name_string = ' '.join(feature_names)
        self.assertIn('vp_', name_string)  # Volume profile features
        self.assertIn('tick_', name_string)  # Tick features
        self.assertIn('liquidity', name_string)  # Liquidity features
        self.assertIn('regime_', name_string)  # Regime features
    
    def test_extractor_stats(self):
        """Test extractor statistics functionality."""
        # Run feature extraction to populate regime history
        self.extractor.extract_features(self.sample_df, 'AAPL')
        
        stats = self.extractor.get_extractor_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('volume_profile_bins', stats)
        self.assertIn('lookback_periods', stats)
        self.assertIn('cache_size', stats)
        self.assertIn('current_regime', stats)
        self.assertIn('settings', stats)
        
        self.assertEqual(stats['volume_profile_bins'], 20)
        self.assertEqual(stats['lookback_periods'], [10, 20, 50])
    
    def test_impact_calculations(self):
        """Test price impact related calculations."""
        # Test impact decay
        recent_data = self.sample_df.tail(20)
        decay = self.extractor._calculate_impact_decay(recent_data)
        
        self.assertGreaterEqual(decay, 0.0)
        self.assertLessEqual(decay, 1.0)
        
        # Test impact permanence
        permanence = self.extractor._calculate_impact_permanence(recent_data)
        
        self.assertGreaterEqual(permanence, 0.0)
        self.assertLessEqual(permanence, 1.0)
    
    def test_pin_approximation(self):
        """Test PIN (Probability of Informed Trading) approximation."""
        recent_data = self.sample_df.tail(15)
        pin = self.extractor._calculate_pin_approximation(recent_data)
        
        self.assertGreaterEqual(pin, 0.0)
        self.assertLessEqual(pin, 1.0)
        self.assertIsInstance(pin, float)
    
    def test_depth_imbalance(self):
        """Test market depth imbalance calculation."""
        recent_data = self.sample_df.tail(10)
        imbalance = self.extractor._calculate_depth_imbalance(recent_data)
        
        self.assertGreaterEqual(imbalance, -1.0)
        self.assertLessEqual(imbalance, 1.0)
        self.assertIsInstance(imbalance, float)
    
    def test_error_handling(self):
        """Test error handling with malformed data."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        features = self.extractor.extract_features(empty_df, 'AAPL')
        self.assertEqual(len(features), 40)  # Default feature count
        
        # DataFrame with missing columns
        bad_df = pd.DataFrame({'price': [100, 101, 102]})
        features = self.extractor.extract_features(bad_df, 'AAPL')
        self.assertEqual(len(features), 40)
        
        # DataFrame with NaN values
        nan_df = self.sample_df.copy()
        nan_df.iloc[10:15, 1] = np.nan  # Insert NaNs in 'high' column
        features = self.extractor.extract_features(nan_df, 'AAPL')
        self.assertTrue(np.all(np.isfinite(features)))
    
    def test_volume_profile_metrics(self):
        """Test volume profile specific metrics."""
        volume_profile = [
            VolumeNode(100.0, 1000, 600, 400, 200, 10, 1.0),
            VolumeNode(101.0, 1500, 800, 700, 100, 15, 1.5),
            VolumeNode(102.0, 800, 300, 500, -200, 8, 0.8)
        ]
        
        # Test volume concentration
        concentration = self.extractor._calculate_volume_concentration(volume_profile)
        self.assertGreaterEqual(concentration, 0.0)
        self.assertLessEqual(concentration, 1.0)
        
        # Test volume balance
        balance = self.extractor._calculate_volume_balance(volume_profile)
        self.assertGreaterEqual(balance, -1.0)
        self.assertLessEqual(balance, 1.0)
        
        # Test profile shape
        shape = self.extractor._calculate_profile_shape(volume_profile)
        self.assertGreaterEqual(shape, -2.0)
        self.assertLessEqual(shape, 2.0)
    
    def test_different_regimes_produce_different_features(self):
        """Test that different market microstructure regimes produce different features."""
        # Use different parts of data that should have different regimes
        high_liq_data = self.sample_df.head(40)
        low_liq_data = self.sample_df.iloc[50:90]
        
        features1 = self.extractor.extract_features(high_liq_data, 'AAPL')
        features2 = self.extractor.extract_features(low_liq_data, 'AAPL')
        
        # Features should be different due to different market conditions
        self.assertFalse(np.array_equal(features1, features2))
        
        # But both should be valid feature arrays
        self.assertTrue(np.all(np.isfinite(features1)))
        self.assertTrue(np.all(np.isfinite(features2)))


class TestVolumeProfileComponents(unittest.TestCase):
    """Test volume profile related components."""
    
    def test_volume_node_creation(self):
        """Test VolumeNode dataclass."""
        node = VolumeNode(
            price_level=100.0,
            total_volume=1000.0,
            buy_volume=600.0,
            sell_volume=400.0,
            delta=200.0,
            transactions=10,
            time_spent=1.0
        )
        
        self.assertEqual(node.price_level, 100.0)
        self.assertEqual(node.total_volume, 1000.0)
        self.assertEqual(node.delta, 200.0)
        self.assertEqual(node.transactions, 10)
    
    def test_market_microstructure_dataclass(self):
        """Test MarketMicrostructure dataclass."""
        microstructure = MarketMicrostructure(
            bid_ask_spread=0.02,
            effective_spread=0.018,
            price_impact=0.001,
            volume_imbalance=0.1,
            liquidity_ratio=0.5,
            market_depth=1000000,
            regime=MicrostructureRegime.HIGH_LIQUIDITY
        )
        
        self.assertEqual(microstructure.bid_ask_spread, 0.02)
        self.assertEqual(microstructure.regime, MicrostructureRegime.HIGH_LIQUIDITY)
        self.assertIsInstance(microstructure.regime, MicrostructureRegime)


class TestEnumerations(unittest.TestCase):
    """Test enumeration classes."""
    
    def test_volume_profile_type(self):
        """Test VolumeProfileType enumeration."""
        self.assertEqual(len(VolumeProfileType), 3)
        self.assertIn(VolumeProfileType.PRICE_VOLUME, VolumeProfileType)
        self.assertIn(VolumeProfileType.TIME_VOLUME, VolumeProfileType)
        self.assertIn(VolumeProfileType.DELTA_VOLUME, VolumeProfileType)
    
    def test_microstructure_regime(self):
        """Test MicrostructureRegime enumeration."""
        self.assertEqual(len(MicrostructureRegime), 6)
        self.assertIn(MicrostructureRegime.HIGH_LIQUIDITY, MicrostructureRegime)
        self.assertIn(MicrostructureRegime.PRICE_DISCOVERY, MicrostructureRegime)
        self.assertIn(MicrostructureRegime.INFORMATION_DRIVEN, MicrostructureRegime)


if __name__ == '__main__':
    unittest.main()