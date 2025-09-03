"""Tests for cross-asset features extractor."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features.cross_asset_features import (
    CrossAssetFeatureExtractor,
    CrossAssetRegime,
    AssetClass,
    CrossAssetMetrics,
    SectorRotationSignal
)


class TestCrossAssetFeatureExtractor(unittest.TestCase):
    """Test cross-asset feature extractor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = CrossAssetFeatureExtractor(
            correlation_windows=[10, 20, 30],
            max_assets=10,
            min_overlap_periods=5
        )
        
        # Create multi-asset market data with realistic relationships
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
        n_days = len(dates)
        
        # Create correlated market data
        base_market_return = np.random.normal(0.0005, 0.015, n_days)  # Market factor
        
        # Create different asset types with varying correlations to market
        asset_configs = {
            'SPY': {'beta': 1.0, 'alpha': 0.0, 'idio_vol': 0.01},      # Market proxy
            'AAPL': {'beta': 1.2, 'alpha': 0.0002, 'idio_vol': 0.02},   # Growth stock
            'MSFT': {'beta': 1.1, 'alpha': 0.0001, 'idio_vol': 0.018},  # Tech stock
            'TLT': {'beta': -0.3, 'alpha': 0.0, 'idio_vol': 0.012},     # Bonds (negative correlation)
            'GLD': {'beta': -0.1, 'alpha': 0.0, 'idio_vol': 0.015},     # Gold
            'VIX': {'beta': -2.0, 'alpha': 0.0, 'idio_vol': 0.1},       # Volatility (high negative correlation)
            'QQQ': {'beta': 1.15, 'alpha': 0.0001, 'idio_vol': 0.016},  # Tech-heavy
            'IWM': {'beta': 1.3, 'alpha': -0.0001, 'idio_vol': 0.025},  # Small caps
        }
        
        self.multi_asset_data = {}
        
        for ticker, config in asset_configs.items():
            # Generate returns using factor model: r = alpha + beta * market + idiosyncratic
            idiosyncratic_returns = np.random.normal(0, config['idio_vol'], n_days)
            asset_returns = (config['alpha'] + 
                           config['beta'] * base_market_return + 
                           idiosyncratic_returns)
            
            # Add some regime changes for more realistic data
            if ticker == 'AAPL':
                asset_returns[50:80] += np.random.normal(0.002, 0.01, 30)  # Growth period
            elif ticker == 'TLT':
                asset_returns[100:130] += np.random.normal(0.001, 0.005, 30)  # Flight to safety
            
            # Convert returns to prices
            prices = [100.0]  # Starting price
            for ret in asset_returns:
                prices.append(prices[-1] * (1 + ret))
            
            prices = prices[:-1]  # Remove extra element
            
            # Create OHLCV data
            self.multi_asset_data[ticker] = pd.DataFrame({
                'date': dates,
                'open': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
                'high': [p * (1 + abs(np.random.normal(0.003, 0.002))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0.003, 0.002))) for p in prices],
                'close': prices,
                'volume': np.random.randint(1000000, 5000000, n_days)
            }).set_index('date')
            
            # Ensure high >= low
            for i in range(len(self.multi_asset_data[ticker])):
                if self.multi_asset_data[ticker]['high'].iloc[i] < self.multi_asset_data[ticker]['low'].iloc[i]:
                    temp = self.multi_asset_data[ticker]['high'].iloc[i]
                    self.multi_asset_data[ticker].iloc[i, self.multi_asset_data[ticker].columns.get_loc('high')] = self.multi_asset_data[ticker]['low'].iloc[i]
                    self.multi_asset_data[ticker].iloc[i, self.multi_asset_data[ticker].columns.get_loc('low')] = temp
    
    def test_initialization(self):
        """Test extractor initialization."""
        self.assertIsNotNone(self.extractor)
        self.assertEqual(len(self.extractor.correlation_windows), 3)
        self.assertEqual(self.extractor.max_assets, 10)
        self.assertTrue(self.extractor.enable_sector_rotation)
        self.assertTrue(self.extractor.enable_factor_analysis)
    
    def test_basic_feature_extraction(self):
        """Test basic cross-asset feature extraction."""
        features = self.extractor.extract_features(
            self.multi_asset_data, 'AAPL', ['SPY', 'MSFT', 'TLT']
        )
        
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
        self.assertEqual(features.dtype, np.float32)
        
        # Check for valid feature values
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertTrue(np.all(np.abs(features) <= 5.0))  # Within expected bounds
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Empty multi-asset data
        empty_data = {}
        features = self.extractor.extract_features(empty_data, 'AAPL', ['SPY'])
        self.assertEqual(len(features), 50)  # Default feature count
        
        # Primary ticker not in data
        features = self.extractor.extract_features(self.multi_asset_data, 'NONEXISTENT')
        self.assertEqual(len(features), 50)
        
        # No reference tickers available
        single_asset_data = {'AAPL': self.multi_asset_data['AAPL']}
        features = self.extractor.extract_features(single_asset_data, 'AAPL')
        self.assertEqual(len(features), 50)
    
    def test_returns_alignment(self):
        """Test returns data alignment across multiple assets."""
        aligned_returns = self.extractor._align_returns_data(
            self.multi_asset_data, ['AAPL', 'SPY', 'MSFT'], 30
        )
        
        self.assertIsInstance(aligned_returns, pd.DataFrame)
        if not aligned_returns.empty:
            self.assertLessEqual(len(aligned_returns.columns), 3)
            self.assertGreater(len(aligned_returns), 0)
            
            # Check for no NaN values (should be dropped during alignment)
            self.assertFalse(aligned_returns.isna().any().any())
    
    def test_correlation_features(self):
        """Test correlation feature extraction."""
        corr_features = self.extractor._extract_correlation_features(
            self.multi_asset_data, 'AAPL', ['SPY', 'MSFT', 'TLT']
        )
        
        expected_length = len(self.extractor.correlation_windows) * 8  # 8 features per window
        self.assertEqual(len(corr_features), expected_length)
        
        # All correlation features should be finite
        self.assertTrue(all(np.isfinite(f) for f in corr_features))
        
        # Correlation values should be in reasonable ranges
        for i in range(0, len(corr_features), 8):
            mean_corr = corr_features[i]
            std_corr = corr_features[i + 1]
            self.assertGreaterEqual(mean_corr, -1.0)
            self.assertLessEqual(mean_corr, 1.0)
            self.assertGreaterEqual(std_corr, 0.0)
    
    def test_relative_strength_features(self):
        """Test relative strength feature extraction."""
        rel_str_features = self.extractor._extract_relative_strength_features(
            self.multi_asset_data, 'AAPL', ['SPY', 'MSFT', 'TLT']
        )
        
        expected_length = 3 * 6  # 3 horizons * 6 features each
        self.assertEqual(len(rel_str_features), expected_length)
        
        # Check feature ranges
        for i in range(0, len(rel_str_features), 6):
            relative_strength = rel_str_features[i]
            rank_percentile = rel_str_features[i + 1]
            beat_rate = rel_str_features[i + 2]
            
            self.assertGreaterEqual(relative_strength, -1.0)
            self.assertLessEqual(relative_strength, 1.0)
            self.assertGreaterEqual(rank_percentile, 0.0)
            self.assertLessEqual(rank_percentile, 1.0)
            self.assertGreaterEqual(beat_rate, 0.0)
            self.assertLessEqual(beat_rate, 1.0)
    
    def test_comovement_features(self):
        """Test co-movement and lead-lag features."""
        comovement_features = self.extractor._extract_comovement_features(
            self.multi_asset_data, 'AAPL', ['SPY', 'MSFT', 'TLT']
        )
        
        self.assertEqual(len(comovement_features), 12)  # Expected number of co-movement features
        self.assertTrue(all(np.isfinite(f) for f in comovement_features))
        
        # Beta features should be reasonable
        avg_beta = comovement_features[0]
        self.assertGreaterEqual(avg_beta, -3.0)
        self.assertLessEqual(avg_beta, 3.0)
        
        # Synchronization should be between 0 and 1
        if len(comovement_features) > 3:
            synchronization = comovement_features[3]
            self.assertGreaterEqual(synchronization, 0.0)
            self.assertLessEqual(synchronization, 1.0)
    
    def test_lead_lag_calculation(self):
        """Test lead-lag relationship calculation."""
        # Create two series with known lead-lag relationship
        base_series = pd.Series(np.random.normal(0, 0.01, 50))
        lagged_series = base_series.shift(2).fillna(0)  # 2-period lag
        
        lead_lag_score = self.extractor._calculate_lead_lag(base_series, lagged_series)
        
        self.assertIsInstance(lead_lag_score, float)
        self.assertGreaterEqual(lead_lag_score, -1.0)
        self.assertLessEqual(lead_lag_score, 1.0)
    
    def test_volatility_spillover_features(self):
        """Test volatility spillover feature extraction."""
        spillover_features = self.extractor._extract_volatility_spillover_features(
            self.multi_asset_data, 'AAPL', ['SPY', 'MSFT', 'TLT']
        )
        
        self.assertEqual(len(spillover_features), 8)  # Expected number of spillover features
        self.assertTrue(all(np.isfinite(f) for f in spillover_features))
        
        # Volatility correlations should be in [-1, 1]
        if len(spillover_features) > 0:
            vol_corr = spillover_features[0]
            self.assertGreaterEqual(vol_corr, -1.0)
            self.assertLessEqual(vol_corr, 1.0)
    
    def test_risk_regime_detection(self):
        """Test cross-asset regime detection."""
        regime = self.extractor._detect_cross_asset_regime(
            self.multi_asset_data, 'AAPL', ['SPY', 'MSFT', 'TLT']
        )
        
        self.assertIsInstance(regime, CrossAssetRegime)
        self.assertIn(regime, list(CrossAssetRegime))
    
    def test_risk_regime_features(self):
        """Test risk regime feature extraction."""
        regime_features = self.extractor._extract_risk_regime_features(
            self.multi_asset_data, 'AAPL', ['SPY', 'MSFT', 'TLT']
        )
        
        self.assertEqual(len(regime_features), 6)  # 4 regime indicators + stability + strength
        self.assertTrue(all(np.isfinite(f) for f in regime_features))
        
        # First 4 should be regime indicators (0 or 1)
        regime_indicators = regime_features[:4]
        for indicator in regime_indicators:
            self.assertIn(indicator, [0.0, 1.0])
        
        # Exactly one regime should be active
        self.assertEqual(sum(regime_indicators), 1.0)
        
        # Stability and strength should be in [0, 1]
        stability = regime_features[4]
        strength = regime_features[5]
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
        self.assertGreaterEqual(strength, 0.0)
        self.assertLessEqual(strength, 1.0)
    
    def test_sector_rotation_features(self):
        """Test sector rotation feature extraction."""
        rotation_features = self.extractor._extract_sector_rotation_features(
            self.multi_asset_data, 'AAPL', ['SPY', 'MSFT', 'TLT', 'GLD']
        )
        
        self.assertEqual(len(rotation_features), 6)  # Expected number of rotation features
        self.assertTrue(all(np.isfinite(f) for f in rotation_features))
        
        # Rotation strength should be non-negative
        rotation_strength = rotation_features[0]
        self.assertGreaterEqual(rotation_strength, 0.0)
        self.assertLessEqual(rotation_strength, 1.0)
        
        # Rank percentile should be in [0, 1]
        rank_percentile = rotation_features[1]
        self.assertGreaterEqual(rank_percentile, 0.0)
        self.assertLessEqual(rank_percentile, 1.0)
    
    def test_factor_exposure_features(self):
        """Test factor exposure feature extraction."""
        factor_features = self.extractor._extract_factor_exposure_features(
            self.multi_asset_data, 'AAPL', ['SPY', 'MSFT', 'TLT']
        )
        
        self.assertEqual(len(factor_features), 7)  # 7 factor exposures
        self.assertTrue(all(np.isfinite(f) for f in factor_features))
        
        # Factor exposures should be correlations, so in [-1, 1]
        for exposure in factor_features:
            self.assertGreaterEqual(exposure, -1.0)
            self.assertLessEqual(exposure, 1.0)
    
    def test_factor_exposure_calculation(self):
        """Test individual factor exposure calculation."""
        # Test with market factor (should have positive exposure for AAPL)
        market_exposure = self.extractor._calculate_factor_exposure(
            self.multi_asset_data, 'AAPL', ['SPY', 'QQQ']
        )
        
        self.assertIsInstance(market_exposure, float)
        self.assertGreaterEqual(market_exposure, -1.0)
        self.assertLessEqual(market_exposure, 1.0)
        
        # For a stock like AAPL, market exposure should typically be positive
        # (though we don't enforce this in the test due to randomness)
    
    def test_asset_class_performance(self):
        """Test asset class performance calculation."""
        # Get aligned returns
        aligned_returns = self.extractor._align_returns_data(
            self.multi_asset_data, ['AAPL', 'SPY', 'TLT'], 20
        )
        
        if not aligned_returns.empty:
            equity_performance = self.extractor._calculate_asset_class_performance(
                aligned_returns, AssetClass.EQUITY
            )
            bond_performance = self.extractor._calculate_asset_class_performance(
                aligned_returns, AssetClass.BOND
            )
            
            self.assertIsInstance(equity_performance, float)
            self.assertIsInstance(bond_performance, float)
            self.assertTrue(np.isfinite(equity_performance))
            self.assertTrue(np.isfinite(bond_performance))
    
    def test_correlation_stability(self):
        """Test correlation stability calculation."""
        aligned_returns = self.extractor._align_returns_data(
            self.multi_asset_data, ['AAPL', 'SPY', 'MSFT'], 40
        )
        
        if not aligned_returns.empty and len(aligned_returns) >= 20:
            stability = self.extractor._calculate_correlation_stability(
                aligned_returns, 'AAPL', ['SPY', 'MSFT']
            )
            
            self.assertIsInstance(stability, float)
            self.assertGreaterEqual(stability, 0.0)
            self.assertLessEqual(stability, 1.0)
    
    def test_momentum_consistency(self):
        """Test momentum consistency calculation."""
        consistency = self.extractor._calculate_momentum_consistency(
            self.multi_asset_data, 'AAPL', ['SPY', 'MSFT'], 30
        )
        
        self.assertIsInstance(consistency, float)
        self.assertGreaterEqual(consistency, 0.0)
        self.assertLessEqual(consistency, 1.0)
    
    def test_volatility_spillover_calculation(self):
        """Test volatility spillover calculation."""
        # Create volatility series
        vol1 = pd.Series(np.random.exponential(0.02, 30), 
                        index=pd.date_range('2023-01-01', periods=30))
        vol2 = pd.Series(np.random.exponential(0.025, 30),
                        index=pd.date_range('2023-01-01', periods=30))
        
        spillover = self.extractor._calculate_volatility_spillover(vol1, vol2)
        
        self.assertIsInstance(spillover, float)
        self.assertGreaterEqual(spillover, -1.0)
        self.assertLessEqual(spillover, 1.0)
    
    def test_feature_caching(self):
        """Test feature caching mechanism."""
        # Clear cache
        self.extractor.feature_cache.clear()
        self.extractor.cache_timestamps.clear()
        
        # Extract features twice
        features1 = self.extractor.extract_features(
            self.multi_asset_data, 'AAPL', ['SPY', 'MSFT']
        )
        features2 = self.extractor.extract_features(
            self.multi_asset_data, 'AAPL', ['SPY', 'MSFT']
        )
        
        # Should be identical due to caching
        np.testing.assert_array_equal(features1, features2)
        
        # Cache should contain entry
        self.assertGreater(len(self.extractor.feature_cache), 0)
    
    def test_different_asset_combinations(self):
        """Test with different combinations of reference assets."""
        # Test with bonds and commodities
        features1 = self.extractor.extract_features(
            self.multi_asset_data, 'AAPL', ['TLT', 'GLD']
        )
        
        # Test with other equities
        features2 = self.extractor.extract_features(
            self.multi_asset_data, 'AAPL', ['SPY', 'MSFT', 'QQQ']
        )
        
        # Features should be different due to different reference assets
        self.assertFalse(np.array_equal(features1, features2))
        
        # But both should be valid
        self.assertTrue(np.all(np.isfinite(features1)))
        self.assertTrue(np.all(np.isfinite(features2)))
    
    def test_regime_dependent_correlation(self):
        """Test regime-dependent correlation calculation."""
        aligned_returns = self.extractor._align_returns_data(
            self.multi_asset_data, ['AAPL', 'SPY', 'MSFT'], 50
        )
        
        if not aligned_returns.empty:
            regime_corr_features = self.extractor._calculate_regime_dependent_correlation(
                aligned_returns, 'AAPL'
            )
            
            self.assertEqual(len(regime_corr_features), 4)
            self.assertTrue(all(np.isfinite(f) for f in regime_corr_features))
            
            # High and low vol correlations should be in [-1, 1]
            high_vol_corr = regime_corr_features[0]
            low_vol_corr = regime_corr_features[1]
            self.assertGreaterEqual(high_vol_corr, -1.0)
            self.assertLessEqual(high_vol_corr, 1.0)
            self.assertGreaterEqual(low_vol_corr, -1.0)
            self.assertLessEqual(low_vol_corr, 1.0)
    
    def test_cross_asset_momentum(self):
        """Test cross-asset momentum calculation."""
        aligned_returns = self.extractor._align_returns_data(
            self.multi_asset_data, ['AAPL', 'SPY', 'MSFT'], 30
        )
        
        if not aligned_returns.empty:
            momentum_features = self.extractor._calculate_cross_asset_momentum(
                aligned_returns, 'AAPL'
            )
            
            self.assertEqual(len(momentum_features), 4)
            self.assertTrue(all(np.isfinite(f) for f in momentum_features))
            
            # Features should be in reasonable ranges
            avg_momentum = momentum_features[0]
            consistency = momentum_features[1]
            self.assertGreaterEqual(avg_momentum, -1.0)
            self.assertLessEqual(avg_momentum, 1.0)
            self.assertGreaterEqual(consistency, 0.0)
            self.assertLessEqual(consistency, 1.0)
    
    def test_feature_names(self):
        """Test feature naming functionality."""
        feature_names = self.extractor.get_feature_names()
        
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)
        self.assertTrue(all(isinstance(name, str) for name in feature_names))
        
        # Should include expected feature categories
        name_string = ' '.join(feature_names)
        self.assertIn('corr_', name_string)  # Correlation features
        self.assertIn('rel_str_', name_string)  # Relative strength features
        self.assertIn('comovement', name_string)  # Co-movement features
        self.assertIn('vol_spillover', name_string)  # Volatility spillover features
        self.assertIn('risk_regime', name_string)  # Risk regime features
        self.assertIn('sector_rotation', name_string)  # Sector rotation features
        self.assertIn('factor_', name_string)  # Factor features
    
    def test_extractor_stats(self):
        """Test extractor statistics functionality."""
        # Run feature extraction to populate regime history
        self.extractor.extract_features(self.multi_asset_data, 'AAPL', ['SPY', 'MSFT'])
        
        stats = self.extractor.get_extractor_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('correlation_windows', stats)
        self.assertIn('max_assets', stats)
        self.assertIn('cache_size', stats)
        self.assertIn('current_regime', stats)
        self.assertIn('asset_class_count', stats)
        self.assertIn('factor_count', stats)
        self.assertIn('settings', stats)
        
        self.assertEqual(stats['correlation_windows'], [10, 20, 30])
        self.assertEqual(stats['max_assets'], 10)
        self.assertGreater(stats['factor_count'], 0)
    
    def test_regime_strength_calculation(self):
        """Test regime strength calculation for different regimes."""
        # Test different regime types
        regimes_to_test = [
            CrossAssetRegime.RISK_ON,
            CrossAssetRegime.RISK_OFF,
            CrossAssetRegime.ROTATION
        ]
        
        for regime in regimes_to_test:
            strength = self.extractor._calculate_regime_strength(
                self.multi_asset_data, 'AAPL', ['SPY', 'MSFT', 'TLT'], regime
            )
            
            self.assertIsInstance(strength, float)
            self.assertGreaterEqual(strength, 0.0)
            self.assertLessEqual(strength, 1.0)
    
    def test_rotation_calculations(self):
        """Test rotation-specific calculations."""
        # Test rotation persistence
        persistence = self.extractor._calculate_rotation_persistence(
            self.multi_asset_data, 'AAPL', ['SPY', 'MSFT', 'TLT']
        )
        self.assertIsInstance(persistence, float)
        self.assertGreaterEqual(persistence, 0.0)
        self.assertLessEqual(persistence, 1.0)
        
        # Test rotation acceleration
        acceleration = self.extractor._calculate_rotation_acceleration(
            self.multi_asset_data, 'AAPL', ['SPY', 'MSFT', 'TLT']
        )
        self.assertIsInstance(acceleration, float)
        self.assertGreaterEqual(acceleration, -1.0)
        self.assertLessEqual(acceleration, 1.0)
        
        # Test rotation beta
        beta = self.extractor._calculate_rotation_beta(
            self.multi_asset_data, 'AAPL', ['SPY', 'MSFT', 'TLT']
        )
        self.assertIsInstance(beta, float)
        self.assertGreaterEqual(beta, -3.0)
        self.assertLessEqual(beta, 3.0)
    
    def test_volatility_persistence(self):
        """Test volatility persistence calculation."""
        # Create volatility data
        volatilities = {}
        for ticker in ['AAPL', 'SPY', 'MSFT']:
            if ticker in self.multi_asset_data:
                returns = self.multi_asset_data[ticker]['close'].pct_change().dropna()
                vol = returns.rolling(window=10).std().dropna()
                volatilities[ticker] = vol.tail(30)
        
        persistence_features = self.extractor._calculate_volatility_persistence(
            volatilities, 'AAPL'
        )
        
        self.assertEqual(len(persistence_features), 2)
        self.assertTrue(all(np.isfinite(f) for f in persistence_features))
        
        # Primary persistence should be in [-1, 1]
        primary_persistence = persistence_features[0]
        self.assertGreaterEqual(primary_persistence, -1.0)
        self.assertLessEqual(primary_persistence, 1.0)
    
    def test_contagion_indicator(self):
        """Test financial contagion indicator calculation."""
        # Create volatility data
        volatilities = {}
        for ticker in ['AAPL', 'SPY', 'MSFT']:
            if ticker in self.multi_asset_data:
                returns = self.multi_asset_data[ticker]['close'].pct_change().dropna()
                vol = returns.rolling(window=10).std().dropna()
                volatilities[ticker] = vol.tail(30)
        
        contagion = self.extractor._calculate_contagion_indicator(volatilities, 'AAPL')
        
        self.assertIsInstance(contagion, float)
        self.assertGreaterEqual(contagion, 0.0)
        self.assertLessEqual(contagion, 1.0)
    
    def test_error_handling(self):
        """Test error handling with malformed data."""
        # Test with empty data
        empty_data = {}
        features = self.extractor.extract_features(empty_data, 'AAPL')
        self.assertEqual(len(features), 50)
        
        # Test with mismatched data lengths
        short_data = {
            'AAPL': self.multi_asset_data['AAPL'].head(5),
            'SPY': self.multi_asset_data['SPY'].head(3)
        }
        features = self.extractor.extract_features(short_data, 'AAPL', ['SPY'])
        self.assertEqual(len(features), 50)
        
        # Test with NaN values
        nan_data = self.multi_asset_data.copy()
        nan_data['AAPL']['close'].iloc[10:15] = np.nan
        features = self.extractor.extract_features(nan_data, 'AAPL', ['SPY'])
        self.assertTrue(np.all(np.isfinite(features)))


class TestDataStructures(unittest.TestCase):
    """Test data structures and enumerations."""
    
    def test_cross_asset_metrics_dataclass(self):
        """Test CrossAssetMetrics dataclass."""
        metrics = CrossAssetMetrics(
            correlation=0.7,
            beta=1.2,
            relative_strength=0.05,
            co_movement=0.8,
            lead_lag=1,
            volatility_spillover=0.3
        )
        
        self.assertEqual(metrics.correlation, 0.7)
        self.assertEqual(metrics.beta, 1.2)
        self.assertEqual(metrics.lead_lag, 1)
    
    def test_sector_rotation_signal_dataclass(self):
        """Test SectorRotationSignal dataclass."""
        signal = SectorRotationSignal(
            rotation_strength=0.6,
            leading_sectors=['tech', 'healthcare'],
            lagging_sectors=['utilities', 'staples'],
            rotation_direction='growth_to_value'
        )
        
        self.assertEqual(signal.rotation_strength, 0.6)
        self.assertEqual(len(signal.leading_sectors), 2)
        self.assertEqual(signal.rotation_direction, 'growth_to_value')
    
    def test_cross_asset_regime_enum(self):
        """Test CrossAssetRegime enumeration."""
        self.assertEqual(len(CrossAssetRegime), 6)
        self.assertIn(CrossAssetRegime.RISK_ON, CrossAssetRegime)
        self.assertIn(CrossAssetRegime.RISK_OFF, CrossAssetRegime)
        self.assertIn(CrossAssetRegime.ROTATION, CrossAssetRegime)
        self.assertIn(CrossAssetRegime.CORRELATION_BREAKDOWN, CrossAssetRegime)
    
    def test_asset_class_enum(self):
        """Test AssetClass enumeration."""
        self.assertEqual(len(AssetClass), 7)
        self.assertIn(AssetClass.EQUITY, AssetClass)
        self.assertIn(AssetClass.BOND, AssetClass)
        self.assertIn(AssetClass.COMMODITY, AssetClass)
        self.assertIn(AssetClass.CURRENCY, AssetClass)
        self.assertIn(AssetClass.CRYPTO, AssetClass)


if __name__ == '__main__':
    unittest.main()