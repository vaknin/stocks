"""Tests for volatility regime features extractor."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features.volatility_regime_features import (
    VolatilityRegimeFeatureExtractor,
    VolatilityRegime,
    GapType,
    VolatilityModel,
    VolatilityMetrics,
    GapAnalysis
)


class TestVolatilityRegimeFeatureExtractor(unittest.TestCase):
    """Test volatility regime feature extractor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = VolatilityRegimeFeatureExtractor(
            volatility_windows=[5, 10, 20],
            regime_lookback=100,
            gap_threshold=0.02,  # 2% gap threshold for testing
        )
        
        # Create realistic market data with different volatility regimes
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n_days = len(dates)
        
        # Generate returns with regime changes
        returns = []
        prices = [100.0]
        
        for i in range(n_days):
            # Different volatility regimes
            if i < 60:  # Low volatility period
                vol = 0.008
                mean_return = 0.0005
            elif i < 120:  # Normal volatility
                vol = 0.015
                mean_return = 0.0003
            elif i < 180:  # High volatility period
                vol = 0.035
                mean_return = -0.0002
            elif i < 210:  # Extreme volatility (crisis-like)
                vol = 0.055
                mean_return = -0.001
            else:  # Return to normal
                vol = 0.018
                mean_return = 0.0004
            
            # Add some ARCH effects (volatility clustering)
            if i > 0:
                prev_return_impact = abs(returns[-1]) * 0.3
                vol *= (1 + prev_return_impact)
            
            daily_return = np.random.normal(mean_return, vol)
            returns.append(daily_return)
            
            # Convert to price
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(1.0, new_price))
        
        prices = prices[:-1]  # Remove extra element
        
        # Create OHLCV data with gaps
        opens = []
        highs = []
        lows = []
        volumes = []
        
        for i, (price, ret) in enumerate(zip(prices, returns)):
            # Create gaps occasionally
            if i > 0 and np.random.random() < 0.05:  # 5% chance of gap
                gap_size = np.random.normal(0, 0.03)  # Average 3% gap
                open_price = prices[i-1] * (1 + gap_size)
            else:
                open_price = price * (1 + np.random.normal(0, 0.002))
            
            # High/Low based on volatility
            daily_range = abs(ret) + np.random.exponential(0.01)
            high_price = max(open_price, price) * (1 + daily_range/2)
            low_price = min(open_price, price) * (1 - daily_range/2)
            
            # Volume with heteroskedasticity
            base_volume = 1000000
            vol_multiplier = 1 + abs(ret) * 10  # Higher volume on volatile days
            volume = int(base_volume * vol_multiplier * (1 + np.random.normal(0, 0.3)))
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(max(low_price, 0.1))  # Ensure positive
            volumes.append(max(volume, 10000))  # Minimum volume
        
        # Create DataFrame
        self.sample_df = pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        }).set_index('date')
        
        # Ensure high >= low and other basic constraints
        for i in range(len(self.sample_df)):
            if self.sample_df['high'].iloc[i] < self.sample_df['low'].iloc[i]:
                temp = self.sample_df['high'].iloc[i]
                self.sample_df.iloc[i, self.sample_df.columns.get_loc('high')] = self.sample_df['low'].iloc[i]
                self.sample_df.iloc[i, self.sample_df.columns.get_loc('low')] = temp
            
            # Ensure open and close are within high-low range
            open_price = self.sample_df['open'].iloc[i]
            close_price = self.sample_df['close'].iloc[i]
            high_price = self.sample_df['high'].iloc[i]
            low_price = self.sample_df['low'].iloc[i]
            
            self.sample_df.iloc[i, self.sample_df.columns.get_loc('high')] = max(high_price, open_price, close_price)
            self.sample_df.iloc[i, self.sample_df.columns.get_loc('low')] = min(low_price, open_price, close_price)
    
    def test_initialization(self):
        """Test extractor initialization."""
        self.assertIsNotNone(self.extractor)
        self.assertEqual(len(self.extractor.volatility_windows), 3)
        self.assertEqual(self.extractor.regime_lookback, 100)
        self.assertEqual(self.extractor.gap_threshold, 0.02)
        self.assertTrue(self.extractor.enable_garch_modeling)
        self.assertTrue(self.extractor.enable_regime_prediction)
    
    def test_basic_feature_extraction(self):
        """Test basic volatility regime feature extraction."""
        features = self.extractor.extract_features(self.sample_df, 'AAPL')
        
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
        self.assertEqual(features.dtype, np.float32)
        
        # Check for valid feature values
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertTrue(np.all(np.abs(features) <= 10.0))  # Within expected bounds
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Very small dataset
        small_df = self.sample_df.head(5)
        features = self.extractor.extract_features(small_df, 'AAPL')
        self.assertEqual(len(features), 40)  # Default feature count
        
        # Empty dataset
        empty_df = pd.DataFrame()
        features = self.extractor.extract_features(empty_df, 'AAPL')
        self.assertEqual(len(features), 40)
    
    def test_volatility_regime_classification(self):
        """Test volatility regime classification."""
        returns = self.sample_df['close'].pct_change().dropna()
        
        # Test with sufficient data
        regime_features = self.extractor._classify_volatility_regime(returns)
        
        self.assertEqual(len(regime_features), 4)  # 4 regime probabilities
        self.assertTrue(all(0.0 <= f <= 1.0 for f in regime_features))  # Valid probabilities
        self.assertAlmostEqual(sum(regime_features), 1.0, places=2)  # Should sum to ~1
        
        # Test with insufficient data
        short_returns = returns.head(10)
        short_regime_features = self.extractor._classify_volatility_regime(short_returns)
        self.assertEqual(len(short_regime_features), 4)
    
    def test_simple_regime_classification(self):
        """Test fallback regime classification."""
        returns = self.sample_df['close'].pct_change().dropna()
        rolling_vol = returns.rolling(window=20).std().dropna()
        
        regime_probs = self.extractor._simple_regime_classification(rolling_vol)
        
        self.assertEqual(len(regime_probs), 4)
        self.assertTrue(all(f in [0.0, 1.0] for f in regime_probs))  # Hard classification
        self.assertEqual(sum(regime_probs), 1.0)  # Exactly one regime active
    
    def test_volatility_clustering_calculation(self):
        """Test volatility clustering measures."""
        returns = self.sample_df['close'].pct_change().dropna()
        
        clustering_features = self.extractor._calculate_volatility_clustering(returns)
        
        self.assertEqual(len(clustering_features), 3)  # persistence, vol_of_vol, ARCH effect
        self.assertTrue(all(np.isfinite(f) for f in clustering_features))
        
        # Persistence should be in [-1, 1]
        persistence = clustering_features[0]
        self.assertGreaterEqual(persistence, -1.0)
        self.assertLessEqual(persistence, 1.0)
        
        # Vol of vol should be non-negative
        vol_of_vol = clustering_features[1]
        self.assertGreaterEqual(vol_of_vol, 0.0)
        self.assertLessEqual(vol_of_vol, 1.0)
        
        # ARCH effect should be in [0, 1]
        arch_effect = clustering_features[2]
        self.assertGreaterEqual(arch_effect, 0.0)
        self.assertLessEqual(arch_effect, 1.0)
    
    def test_arch_effect_calculation(self):
        """Test ARCH effect calculation."""
        returns = self.sample_df['close'].pct_change().dropna()
        
        arch_effect = self.extractor._calculate_arch_effect(returns)
        
        self.assertIsInstance(arch_effect, float)
        self.assertGreaterEqual(arch_effect, 0.0)
        self.assertLessEqual(arch_effect, 1.0)
        
        # Test with insufficient data
        short_returns = returns.head(5)
        short_arch = self.extractor._calculate_arch_effect(short_returns)
        self.assertEqual(short_arch, 0.5)  # Default value
    
    def test_gap_detection(self):
        """Test price gap detection."""
        gaps = self.extractor._detect_price_gaps(self.sample_df)
        
        # Should detect some gaps given our data generation
        self.assertIsInstance(gaps, list)
        
        if gaps:  # If gaps were detected
            for gap in gaps:
                self.assertIsInstance(gap, GapAnalysis)
                self.assertIsInstance(gap.gap_type, GapType)
                self.assertIn(gap.gap_direction, [-1, 1])
                self.assertGreaterEqual(abs(gap.gap_size), self.extractor.gap_threshold)
                self.assertGreaterEqual(gap.fill_probability, 0.1)
                self.assertLessEqual(gap.fill_probability, 0.95)
                self.assertGreater(gap.volume_confirmation, 0.0)
                self.assertGreaterEqual(gap.momentum_strength, 0.0)
    
    def test_gap_classification(self):
        """Test gap type classification."""
        # Test with different timestamps
        morning_time = pd.Timestamp('2023-06-15 09:45:00')
        afternoon_time = pd.Timestamp('2023-06-15 14:30:00')
        after_hours_time = pd.Timestamp('2023-06-15 18:00:00')
        monday_time = pd.Timestamp('2023-06-19 09:30:00')  # Monday
        
        morning_gap = self.extractor._classify_gap_type(morning_time, 0.03)
        afternoon_gap = self.extractor._classify_gap_type(afternoon_time, 0.02)
        after_hours_gap = self.extractor._classify_gap_type(after_hours_time, 0.025)
        monday_gap = self.extractor._classify_gap_type(monday_time, 0.02)
        
        # All should be valid gap types
        valid_gap_types = list(GapType)
        self.assertIn(morning_gap, valid_gap_types)
        self.assertIn(afternoon_gap, valid_gap_types)
        self.assertIn(after_hours_gap, valid_gap_types)
        self.assertIn(monday_gap, valid_gap_types)
    
    def test_gap_fill_probability_estimation(self):
        """Test gap fill probability estimation."""
        # Test different scenarios
        small_overnight_gap = self.extractor._estimate_gap_fill_probability(
            0.01, 1000000, GapType.OVERNIGHT_GAP
        )
        large_earnings_gap = self.extractor._estimate_gap_fill_probability(
            0.08, 5000000, GapType.EARNINGS_GAP
        )
        weekend_gap = self.extractor._estimate_gap_fill_probability(
            0.02, 2000000, GapType.WEEKEND_GAP
        )
        
        # All should be valid probabilities
        self.assertGreaterEqual(small_overnight_gap, 0.1)
        self.assertLessEqual(small_overnight_gap, 0.95)
        self.assertGreaterEqual(large_earnings_gap, 0.1)
        self.assertLessEqual(large_earnings_gap, 0.95)
        self.assertGreaterEqual(weekend_gap, 0.1)
        self.assertLessEqual(weekend_gap, 0.95)
        
        # Earnings gaps should generally have lower fill probability
        self.assertLess(large_earnings_gap, small_overnight_gap)
    
    def test_gap_analysis_features(self):
        """Test gap analysis feature extraction."""
        gap_features = self.extractor._extract_gap_analysis_features(self.sample_df, 'AAPL')
        
        self.assertEqual(len(gap_features), 12)  # Expected number of gap features
        self.assertTrue(all(np.isfinite(f) for f in gap_features))
        
        # Gap frequency should be reasonable
        gap_frequency = gap_features[0]
        self.assertGreaterEqual(gap_frequency, 0.0)
        self.assertLessEqual(gap_frequency, 1.0)
        
        # Up gap ratio should be a valid ratio
        up_gap_ratio = gap_features[1]
        self.assertGreaterEqual(up_gap_ratio, 0.0)
        self.assertLessEqual(up_gap_ratio, 1.0)
    
    def test_gap_pattern_analysis(self):
        """Test gap pattern analysis."""
        # Create some test gaps
        test_gaps = [
            GapAnalysis(GapType.OVERNIGHT_GAP, 0.02, 1, 0.7, None, 1.2, 0.015),
            GapAnalysis(GapType.OVERNIGHT_GAP, 0.025, 1, 0.6, None, 1.5, 0.018),
            GapAnalysis(GapType.INTRADAY_GAP, -0.015, -1, 0.8, None, 0.9, 0.012),
            GapAnalysis(GapType.OVERNIGHT_GAP, 0.03, 1, 0.5, None, 2.0, 0.022)
        ]
        
        pattern_features = self.extractor._analyze_gap_patterns(test_gaps)
        
        self.assertEqual(len(pattern_features), 6)  # Expected pattern features
        self.assertTrue(all(np.isfinite(f) for f in pattern_features))
        
        # Size trend should be reasonable
        size_trend = pattern_features[0]
        self.assertGreaterEqual(size_trend, -1.0)
        self.assertLessEqual(size_trend, 1.0)
        
        # Direction consistency should be in [0, 1]
        direction_consistency = pattern_features[1]
        self.assertGreaterEqual(direction_consistency, 0.0)
        self.assertLessEqual(direction_consistency, 1.0)
    
    def test_garch_like_features(self):
        """Test GARCH-like feature calculation."""
        returns = self.sample_df['close'].pct_change().dropna()
        
        garch_features = self.extractor._calculate_garch_like_features(returns)
        
        self.assertEqual(len(garch_features), 4)  # 4 GARCH-like features
        self.assertTrue(all(np.isfinite(f) for f in garch_features))
        
        # Current volatility should be positive and reasonable
        current_vol = garch_features[0]
        self.assertGreater(current_vol, 0.0)
        self.assertLessEqual(current_vol, 2.0)  # Capped at 200%
        
        # Mean reversion should be in [0, 1]
        mean_reversion = garch_features[1]
        self.assertGreaterEqual(mean_reversion, 0.0)
        self.assertLessEqual(mean_reversion, 1.0)
        
        # ARCH strength should be in [0, 1]
        arch_strength = garch_features[2]
        self.assertGreaterEqual(arch_strength, 0.0)
        self.assertLessEqual(arch_strength, 1.0)
    
    def test_range_based_volatility(self):
        """Test range-based volatility measures."""
        range_vol_features = self.extractor._calculate_range_based_volatility(self.sample_df)
        
        self.assertEqual(len(range_vol_features), 3)  # Parkinson, GK, efficiency
        self.assertTrue(all(np.isfinite(f) for f in range_vol_features))
        
        # All volatility measures should be positive
        parkinson_vol = range_vol_features[0]
        gk_vol = range_vol_features[1]
        efficiency = range_vol_features[2]
        
        self.assertGreater(parkinson_vol, 0.0)
        self.assertLessEqual(parkinson_vol, 2.0)
        self.assertGreater(gk_vol, 0.0)
        self.assertLessEqual(gk_vol, 2.0)
        self.assertGreater(efficiency, 0.1)
        self.assertLessEqual(efficiency, 2.0)
    
    def test_volatility_surface_features(self):
        """Test volatility surface feature calculation."""
        returns = self.sample_df['close'].pct_change().dropna()
        
        surface_features = self.extractor._calculate_volatility_surface_features(returns)
        
        self.assertEqual(len(surface_features), 3)  # Term structure, skew, curvature
        self.assertTrue(all(np.isfinite(f) for f in surface_features))
        
        # Term structure slope should be reasonable
        term_structure = surface_features[0]
        self.assertGreaterEqual(term_structure, -1.0)
        self.assertLessEqual(term_structure, 1.0)
        
        # Skew should be in reasonable range
        vol_skew = surface_features[1]
        self.assertGreaterEqual(vol_skew, -2.0)
        self.assertLessEqual(vol_skew, 2.0)
        
        # Curvature should be non-negative
        curvature = surface_features[2]
        self.assertGreaterEqual(curvature, 0.0)
        self.assertLessEqual(curvature, 1.0)
    
    def test_regime_transition_features(self):
        """Test regime transition feature extraction."""
        transition_features = self.extractor._extract_regime_transition_features(
            self.sample_df, 'AAPL'
        )
        
        self.assertEqual(len(transition_features), 8)  # Expected transition features
        self.assertTrue(all(np.isfinite(f) for f in transition_features))
        
        # Transition probability should be in reasonable range
        transition_prob = transition_features[0]
        self.assertGreaterEqual(transition_prob, 0.05)  # Minimum 5%
        self.assertLessEqual(transition_prob, 0.9)      # Maximum 90%
    
    def test_regime_transition_probability(self):
        """Test regime transition probability calculation."""
        returns = self.sample_df['close'].pct_change().dropna()
        
        transition_prob = self.extractor._calculate_regime_transition_probability(returns)
        
        self.assertIsInstance(transition_prob, float)
        self.assertGreaterEqual(transition_prob, 0.05)
        self.assertLessEqual(transition_prob, 0.9)
    
    def test_regime_stability(self):
        """Test regime stability measures."""
        returns = self.sample_df['close'].pct_change().dropna()
        
        stability_features = self.extractor._calculate_regime_stability(returns)
        
        self.assertEqual(len(stability_features), 3)
        self.assertTrue(all(np.isfinite(f) for f in stability_features))
        
        # All stability measures should be in [0, 1]
        for stability in stability_features:
            self.assertGreaterEqual(stability, 0.0)
            self.assertLessEqual(stability, 1.0)
    
    def test_regime_warning_indicators(self):
        """Test regime warning indicators."""
        returns = self.sample_df['close'].pct_change().dropna()
        
        warning_features = self.extractor._calculate_regime_warning_indicators(returns)
        
        self.assertEqual(len(warning_features), 4)
        self.assertTrue(all(np.isfinite(f) for f in warning_features))
        
        # All warning indicators should be non-negative and bounded
        for warning in warning_features:
            self.assertGreaterEqual(warning, 0.0)
            self.assertLessEqual(warning, 1.0)
    
    def test_volatility_risk_features(self):
        """Test volatility risk feature extraction."""
        risk_features = self.extractor._extract_volatility_risk_features(self.sample_df, 'AAPL')
        
        self.assertEqual(len(risk_features), 6)  # VaR, ES, premium, tail risk, downside
        self.assertTrue(all(np.isfinite(f) for f in risk_features))
        
        # VaR measures should be negative (losses)
        var_95 = risk_features[0]
        var_99 = risk_features[1]
        expected_shortfall = risk_features[2]
        
        self.assertLessEqual(var_95, 0.0)
        self.assertLessEqual(var_99, 0.0)
        self.assertLessEqual(expected_shortfall, 0.0)
        
        # VaR 99% should be more negative than VaR 95%
        self.assertLessEqual(var_99, var_95)
        
        # Extreme frequency should be reasonable
        extreme_freq = risk_features[4]
        self.assertGreaterEqual(extreme_freq, 0.0)
        self.assertLessEqual(extreme_freq, 0.2)  # Capped at 20%
        
        # Downside deviation should be positive
        downside_dev = risk_features[5]
        self.assertGreaterEqual(downside_dev, 0.0)
        self.assertLessEqual(downside_dev, 1.0)
    
    def test_simple_garch_prediction(self):
        """Test simple GARCH volatility prediction."""
        returns = self.sample_df['close'].pct_change().dropna()
        
        garch_vol = self.extractor._simple_garch_prediction(returns)
        
        self.assertIsInstance(garch_vol, float)
        self.assertGreater(garch_vol, 0.0)
        self.assertLessEqual(garch_vol, 2.0)  # Capped at 200%
        
        # Test with insufficient data
        short_returns = returns.head(5)
        short_garch = self.extractor._simple_garch_prediction(short_returns)
        self.assertEqual(short_garch, 0.15)  # Default value
    
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
    
    def test_different_regime_periods(self):
        """Test features on different volatility regime periods."""
        # Low volatility period (first 60 days)
        low_vol_data = self.sample_df.head(60)
        low_vol_features = self.extractor.extract_features(low_vol_data, 'AAPL')
        
        # High volatility period (days 120-180)
        high_vol_data = self.sample_df.iloc[120:180]
        high_vol_features = self.extractor.extract_features(high_vol_data, 'AAPL')
        
        # Features should be different
        self.assertFalse(np.array_equal(low_vol_features, high_vol_features))
        
        # Both should be valid
        self.assertTrue(np.all(np.isfinite(low_vol_features)))
        self.assertTrue(np.all(np.isfinite(high_vol_features)))
    
    def test_feature_names(self):
        """Test feature naming functionality."""
        feature_names = self.extractor.get_feature_names()
        
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)
        self.assertTrue(all(isinstance(name, str) for name in feature_names))
        
        # Should include expected feature categories
        name_string = ' '.join(feature_names)
        self.assertIn('vol_', name_string)  # Volatility features
        self.assertIn('regime_', name_string)  # Regime features
        self.assertIn('gap_', name_string)  # Gap features
        self.assertIn('garch_', name_string)  # GARCH features
        self.assertIn('var_', name_string)  # VaR features
    
    def test_extractor_stats(self):
        """Test extractor statistics functionality."""
        # Run feature extraction to populate history
        self.extractor.extract_features(self.sample_df, 'AAPL')
        
        stats = self.extractor.get_extractor_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('volatility_windows', stats)
        self.assertIn('regime_lookback', stats)
        self.assertIn('gap_threshold', stats)
        self.assertIn('cache_size', stats)
        self.assertIn('settings', stats)
        
        self.assertEqual(stats['volatility_windows'], [5, 10, 20])
        self.assertEqual(stats['regime_lookback'], 100)
        self.assertEqual(stats['gap_threshold'], 0.02)
    
    def test_all_feature_types(self):
        """Test extraction with all feature types enabled."""
        all_features = self.extractor.extract_features(
            self.sample_df,
            'AAPL',
            include_regime_features=True,
            include_gap_features=True,
            include_volatility_modeling=True
        )
        
        # Test with some features disabled
        partial_features = self.extractor.extract_features(
            self.sample_df,
            'AAPL',
            include_regime_features=True,
            include_gap_features=False,
            include_volatility_modeling=False
        )
        
        # All features should have more elements
        self.assertGreater(len(all_features), len(partial_features))
        
        # Both should be valid
        self.assertTrue(np.all(np.isfinite(all_features)))
        self.assertTrue(np.all(np.isfinite(partial_features)))
    
    def test_error_handling(self):
        """Test error handling with malformed data."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        features = self.extractor.extract_features(empty_df, 'AAPL')
        self.assertEqual(len(features), 40)
        
        # DataFrame with missing columns
        bad_df = pd.DataFrame({'price': [100, 101, 102]})
        features = self.extractor.extract_features(bad_df, 'AAPL')
        self.assertEqual(len(features), 40)
        
        # DataFrame with NaN values
        nan_df = self.sample_df.copy()
        nan_df.iloc[10:15, 1] = np.nan
        features = self.extractor.extract_features(nan_df, 'AAPL')
        self.assertTrue(np.all(np.isfinite(features)))


class TestDataStructures(unittest.TestCase):
    """Test data structures and enumerations."""
    
    def test_volatility_metrics_dataclass(self):
        """Test VolatilityMetrics dataclass."""
        metrics = VolatilityMetrics(
            current_regime=VolatilityRegime.HIGH_VOLATILITY,
            regime_probability=0.8,
            regime_duration=15,
            transition_probability=0.3,
            volatility_percentile=0.85,
            persistence=0.6,
            clustering=0.7,
            asymmetry=-0.2
        )
        
        self.assertEqual(metrics.current_regime, VolatilityRegime.HIGH_VOLATILITY)
        self.assertEqual(metrics.regime_probability, 0.8)
        self.assertEqual(metrics.regime_duration, 15)
        self.assertEqual(metrics.asymmetry, -0.2)
    
    def test_gap_analysis_dataclass(self):
        """Test GapAnalysis dataclass."""
        gap = GapAnalysis(
            gap_type=GapType.EARNINGS_GAP,
            gap_size=0.05,
            gap_direction=1,
            fill_probability=0.4,
            time_to_fill=3,
            volume_confirmation=2.5,
            momentum_strength=0.08
        )
        
        self.assertEqual(gap.gap_type, GapType.EARNINGS_GAP)
        self.assertEqual(gap.gap_size, 0.05)
        self.assertEqual(gap.gap_direction, 1)
        self.assertEqual(gap.time_to_fill, 3)
    
    def test_volatility_regime_enum(self):
        """Test VolatilityRegime enumeration."""
        self.assertEqual(len(VolatilityRegime), 6)
        self.assertIn(VolatilityRegime.LOW_VOLATILITY, VolatilityRegime)
        self.assertIn(VolatilityRegime.HIGH_VOLATILITY, VolatilityRegime)
        self.assertIn(VolatilityRegime.EXTREME_VOLATILITY, VolatilityRegime)
        self.assertIn(VolatilityRegime.VOLATILITY_SPIKE, VolatilityRegime)
    
    def test_gap_type_enum(self):
        """Test GapType enumeration."""
        self.assertEqual(len(GapType), 6)
        self.assertIn(GapType.OVERNIGHT_GAP, GapType)
        self.assertIn(GapType.EARNINGS_GAP, GapType)
        self.assertIn(GapType.WEEKEND_GAP, GapType)
        self.assertIn(GapType.OPENING_GAP, GapType)
    
    def test_volatility_model_enum(self):
        """Test VolatilityModel enumeration."""
        self.assertEqual(len(VolatilityModel), 4)
        self.assertIn(VolatilityModel.REALIZED_VOLATILITY, VolatilityModel)
        self.assertIn(VolatilityModel.GARCH_LIKE, VolatilityModel)
        self.assertIn(VolatilityModel.RANGE_BASED, VolatilityModel)


if __name__ == '__main__':
    unittest.main()