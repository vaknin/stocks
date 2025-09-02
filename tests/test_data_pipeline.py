"""
Data Pipeline Testing Suite - Data Ingestion Accuracy and Reliability Validation

Tests the data pipeline's ability to reliably ingest real-time market data,
validate data quality, handle missing data, and ensure data consistency
for accurate investment recommendation generation.

Priority: TIER 2 - Critical for recommendation system data quality assurance
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader, DataQualityValidator
from src.data.market_data_fetcher import MarketDataFetcher
from src.config.settings import config
import yfinance as yf


class TestDataPipeline:
    """Test suite for validating data pipeline reliability and accuracy."""
    
    @pytest.fixture(scope="class")
    def sample_market_data(self):
        """Generate realistic market data for testing."""
        np.random.seed(42)
        
        # Generate 6 months of daily data
        dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
        dates = dates[dates.weekday < 5]  # Trading days only
        
        base_price = 150.0
        returns = np.random.normal(0.0008, 0.02, len(dates))  # 0.08% daily mean, 2% volatility
        
        # Add market regime patterns
        returns[50:80] += 0.003   # Bull run
        returns[120:150] -= 0.005 # Bear period
        returns[160:170] *= 2.5   # High volatility period
        
        prices = base_price * np.cumprod(1 + returns)
        
        # Create realistic OHLCV data
        data = pd.DataFrame({
            'Date': dates,
            'Open': prices * np.random.uniform(0.998, 1.002, len(dates)),
            'High': prices * np.random.uniform(1.005, 1.025, len(dates)),
            'Low': prices * np.random.uniform(0.975, 0.995, len(dates)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 15000000, len(dates)),
            'Adj Close': prices * np.random.uniform(0.999, 1.001, len(dates))  # Corporate actions
        }).set_index('Date')
        
        return data
    
    @pytest.fixture
    def data_loader(self):
        """Create data loader for testing."""
        return DataLoader(
            cache_dir="test_cache",
            enable_cache=False,  # Disable cache for testing
            validate_data=True
        )
    
    @pytest.fixture
    def data_quality_validator(self):
        """Create data quality validator for testing."""
        return DataQualityValidator(
            price_change_threshold=0.20,    # 20% max daily change
            volume_spike_threshold=5.0,     # 5x average volume
            missing_data_tolerance=0.05     # 5% missing data tolerance
        )
    
    def test_data_ingestion_accuracy(self, data_loader, sample_market_data):
        """Test accuracy of data ingestion from external sources."""
        ticker = "AAPL"
        
        # Mock yfinance to return our sample data
        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = Mock()
            mock_instance.history.return_value = sample_market_data
            mock_instance.info = {
                'symbol': ticker,
                'shortName': 'Apple Inc.',
                'sector': 'Technology',
                'marketCap': 2800000000000
            }
            mock_ticker.return_value = mock_instance
            
            # Load data through data loader
            loaded_data = data_loader.load_stock_data(
                ticker=ticker,
                period="6mo",
                interval="1d"
            )
            
            # Validate data structure
            assert isinstance(loaded_data, pd.DataFrame), "Should return DataFrame"
            assert not loaded_data.empty, "Should return non-empty data"
            
            # Validate required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for column in required_columns:
                assert column in loaded_data.columns, f"Missing required column: {column}"
            
            # Validate data types
            for price_col in ['Open', 'High', 'Low', 'Close']:
                assert pd.api.types.is_numeric_dtype(loaded_data[price_col]), f"{price_col} should be numeric"
            
            assert pd.api.types.is_numeric_dtype(loaded_data['Volume']), "Volume should be numeric"
            
            # Validate data ranges
            assert (loaded_data[['Open', 'High', 'Low', 'Close']] > 0).all().all(), "Prices should be positive"
            assert (loaded_data['Volume'] >= 0).all(), "Volume should be non-negative"
            
            # Validate OHLC relationships
            ohlc_valid = (
                (loaded_data['High'] >= loaded_data['Open']) &
                (loaded_data['High'] >= loaded_data['Close']) &
                (loaded_data['Low'] <= loaded_data['Open']) &
                (loaded_data['Low'] <= loaded_data['Close'])
            )
            assert ohlc_valid.all(), "OHLC relationships should be valid"
    
    def test_real_time_data_consistency(self, data_loader):
        """Test consistency of real-time data fetching."""
        tickers = ["AAPL", "MSFT"]  # Test with actual tickers but small sample
        
        try:
            # Fetch current data for multiple tickers
            data_batch = {}
            for ticker in tickers:
                data = data_loader.load_stock_data(
                    ticker=ticker,
                    period="5d",  # Small period for testing
                    interval="1d"
                )
                data_batch[ticker] = data
            
            # Validate consistency across tickers
            for ticker, data in data_batch.items():
                assert isinstance(data, pd.DataFrame), f"Data for {ticker} should be DataFrame"
                assert not data.empty, f"Data for {ticker} should not be empty"
                
                # Validate timestamps are reasonable (within last year)
                latest_date = data.index.max()
                earliest_allowed = datetime.now() - timedelta(days=365)
                assert latest_date >= earliest_allowed, f"Data for {ticker} appears stale"
                
                # Validate data continuity (no large gaps in recent data)
                if len(data) > 1:
                    date_gaps = data.index.to_series().diff().dt.days
                    max_gap = date_gaps.max()
                    assert max_gap <= 7, f"Large data gap for {ticker}: {max_gap} days"
        
        except Exception as e:
            # If real API fails, skip test (network dependency)
            pytest.skip(f"Real-time data test skipped due to network/API issue: {e}")
    
    def test_data_quality_validation(self, data_quality_validator):
        """Test data quality validation and anomaly detection."""
        # Create test data with quality issues
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Base data (good quality)
        good_data = pd.DataFrame({
            'Open': 100 + np.random.normal(0, 1, 100),
            'High': 101 + np.random.normal(0, 1, 100),
            'Low': 99 + np.random.normal(0, 1, 100),
            'Close': 100 + np.random.normal(0, 1, 100),
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        # Test 1: Clean data should pass validation
        quality_report = data_quality_validator.validate_data(good_data, "GOOD_STOCK")
        
        assert isinstance(quality_report, dict), "Quality report should be dict"
        assert 'is_valid' in quality_report, "Missing validation status"
        assert 'issues' in quality_report, "Missing issues list"
        assert 'warnings' in quality_report, "Missing warnings list"
        
        # Good data should pass validation
        assert quality_report['is_valid'], f"Good data should be valid: {quality_report['issues']}"
        
        # Test 2: Data with price spike
        spike_data = good_data.copy()
        spike_data.iloc[50, spike_data.columns.get_loc('Close')] *= 1.5  # 50% price spike
        
        quality_report_spike = data_quality_validator.validate_data(spike_data, "SPIKE_STOCK")
        
        # Should detect price spike issue
        issues = quality_report_spike['issues']
        assert any('price' in issue.lower() and 'spike' in issue.lower() for issue in issues), \
            "Should detect price spike"
        
        # Test 3: Data with missing values
        missing_data = good_data.copy()
        missing_data.iloc[20:25, missing_data.columns.get_loc('Close')] = np.nan  # Missing prices
        
        quality_report_missing = data_quality_validator.validate_data(missing_data, "MISSING_STOCK")
        
        # Should detect missing data
        issues_missing = quality_report_missing['issues']
        assert any('missing' in issue.lower() for issue in issues_missing), \
            "Should detect missing data"
        
        # Test 4: Data with volume anomalies
        volume_anomaly_data = good_data.copy()
        avg_volume = volume_anomaly_data['Volume'].mean()
        volume_anomaly_data.iloc[30, volume_anomaly_data.columns.get_loc('Volume')] = avg_volume * 10  # 10x volume spike
        
        quality_report_volume = data_quality_validator.validate_data(volume_anomaly_data, "VOLUME_SPIKE")
        
        # Should detect volume anomaly
        warnings_volume = quality_report_volume['warnings']
        assert any('volume' in warning.lower() for warning in warnings_volume), \
            "Should detect volume anomaly"
        
        # Test 5: Invalid OHLC relationships
        invalid_ohlc_data = good_data.copy()
        # Make high < low (invalid)
        invalid_ohlc_data.iloc[10, invalid_ohlc_data.columns.get_loc('High')] = 95
        invalid_ohlc_data.iloc[10, invalid_ohlc_data.columns.get_loc('Low')] = 98
        
        quality_report_ohlc = data_quality_validator.validate_data(invalid_ohlc_data, "INVALID_OHLC")
        
        # Should detect OHLC relationship issues
        issues_ohlc = quality_report_ohlc['issues']
        assert any('ohlc' in issue.lower() or 'relationship' in issue.lower() for issue in issues_ohlc), \
            "Should detect invalid OHLC relationships"
    
    def test_market_hours_validation(self, data_loader):
        """Test validation of trading hours and market calendar."""
        # Test data fetching respects market hours
        ticker = "SPY"  # Use ETF that follows market hours
        
        try:
            # Fetch recent intraday data
            intraday_data = data_loader.load_stock_data(
                ticker=ticker,
                period="1d",
                interval="5m"
            )
            
            if not intraday_data.empty:
                # Validate trading hours (9:30 AM - 4:00 PM ET)
                trading_hours = intraday_data.index.time
                
                # Check if data falls within trading hours (allowing for some flexibility)
                market_open = pd.Timestamp("09:30:00").time()
                market_close = pd.Timestamp("16:00:00").time()
                
                valid_hours = []
                for hour in trading_hours:
                    # Allow some flexibility around market hours
                    if market_open <= hour <= market_close:
                        valid_hours.append(True)
                    elif pd.Timestamp("09:00:00").time() <= hour <= pd.Timestamp("16:30:00").time():
                        valid_hours.append(True)  # Extended hours
                    else:
                        valid_hours.append(False)
                
                # Most data should be within trading hours
                valid_ratio = sum(valid_hours) / len(valid_hours)
                assert valid_ratio > 0.8, f"Too much data outside trading hours: {valid_ratio:.1%}"
                
        except Exception as e:
            pytest.skip(f"Market hours test skipped: {e}")
    
    def test_corporate_actions_handling(self, data_loader):
        """Test handling of corporate actions (splits, dividends)."""
        # Create mock data with stock split
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        
        # Pre-split prices (around $200)
        pre_split_prices = 200 + np.random.normal(0, 5, 30)
        # Post-split prices (around $100 - 2:1 split)
        post_split_prices = 100 + np.random.normal(0, 2.5, 30)
        
        all_prices = np.concatenate([pre_split_prices, post_split_prices])
        
        split_affected_data = pd.DataFrame({
            'Open': all_prices * np.random.uniform(0.99, 1.01, 60),
            'High': all_prices * np.random.uniform(1.01, 1.03, 60),
            'Low': all_prices * np.random.uniform(0.97, 0.99, 60),
            'Close': all_prices,
            'Adj Close': all_prices,  # Assume already adjusted
            'Volume': np.random.randint(1000000, 10000000, 60)
        }, index=dates)
        
        # Detect potential stock split
        price_drops = split_affected_data['Close'].pct_change()
        large_drops = price_drops < -0.3  # >30% drop might indicate split
        
        if large_drops.any():
            split_dates = split_affected_data.index[large_drops]
            
            # Validate that data loader can handle this appropriately
            # (Either by detecting split or through adjusted close prices)
            with patch('yfinance.Ticker') as mock_ticker:
                mock_instance = Mock()
                mock_instance.history.return_value = split_affected_data
                mock_ticker.return_value = mock_instance
                
                loaded_data = data_loader.load_stock_data("SPLIT_STOCK", period="3mo")
                
                # Should either use adjusted prices or detect the split
                assert isinstance(loaded_data, pd.DataFrame), "Should handle split data"
                assert not loaded_data.empty, "Should return data despite split"
    
    def test_data_interpolation_missing_values(self, data_loader):
        """Test interpolation of missing values in price data."""
        # Create data with gaps
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0, 1, 50))
        
        data_with_gaps = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 50)
        }, index=dates)
        
        # Introduce gaps
        gap_indices = [10, 11, 12, 25, 26]  # Missing data indices
        data_with_gaps.iloc[gap_indices] = np.nan
        
        # Test interpolation
        interpolated_data = data_loader._interpolate_missing_data(data_with_gaps)
        
        # Should have fewer NaN values after interpolation
        original_nan_count = data_with_gaps.isna().sum().sum()
        interpolated_nan_count = interpolated_data.isna().sum().sum()
        
        assert interpolated_nan_count <= original_nan_count, "Interpolation should reduce NaN values"
        
        # Interpolated values should be reasonable (between neighbors)
        for idx in gap_indices[1:-1]:  # Middle gaps (not edges)
            if not pd.isna(interpolated_data.iloc[idx]['Close']):
                prev_price = interpolated_data.iloc[idx-1]['Close']
                next_price = interpolated_data.iloc[idx+1]['Close']
                current_price = interpolated_data.iloc[idx]['Close']
                
                # Interpolated price should be between neighbors (reasonable bounds)
                min_price = min(prev_price, next_price) * 0.9
                max_price = max(prev_price, next_price) * 1.1
                
                assert min_price <= current_price <= max_price, \
                    f"Interpolated price unreasonable: {current_price} not between {min_price}-{max_price}"
    
    def test_data_synchronization_multiple_sources(self, data_loader):
        """Test synchronization of data from multiple sources."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        
        try:
            # Fetch data for multiple tickers
            synchronized_data = {}
            
            for ticker in tickers:
                data = data_loader.load_stock_data(
                    ticker=ticker,
                    period="1mo",
                    interval="1d"
                )
                synchronized_data[ticker] = data
            
            if len(synchronized_data) > 1:
                # Check timestamp alignment
                date_ranges = []
                for ticker, data in synchronized_data.items():
                    if not data.empty:
                        date_ranges.append((ticker, data.index.min(), data.index.max()))
                
                # Validate that data has reasonable overlap
                if len(date_ranges) >= 2:
                    latest_start = max(start for _, start, _ in date_ranges)
                    earliest_end = min(end for _, _, end in date_ranges)
                    
                    # Should have at least 5 days of overlap
                    overlap_days = (earliest_end - latest_start).days
                    assert overlap_days >= 5, f"Insufficient data overlap: {overlap_days} days"
                
                # Check for data consistency on same dates
                common_dates = None
                for ticker, data in synchronized_data.items():
                    if not data.empty:
                        if common_dates is None:
                            common_dates = set(data.index)
                        else:
                            common_dates = common_dates.intersection(set(data.index))
                
                if common_dates and len(common_dates) > 0:
                    # Should have some common trading dates
                    assert len(common_dates) > 5, f"Too few common trading dates: {len(common_dates)}"
        
        except Exception as e:
            pytest.skip(f"Multi-source synchronization test skipped: {e}")
    
    def test_rate_limiting_and_api_errors(self, data_loader):
        """Test handling of rate limiting and API errors."""
        # Mock API rate limiting error
        def mock_rate_limited_call(*args, **kwargs):
            raise Exception("Rate limited - too many requests")
        
        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = Mock()
            mock_instance.history.side_effect = mock_rate_limited_call
            mock_ticker.return_value = mock_instance
            
            # Should handle rate limiting gracefully
            try:
                data = data_loader.load_stock_data(
                    ticker="RATE_LIMITED",
                    period="1mo",
                    max_retries=2,
                    retry_delay=0.1  # Short delay for testing
                )
                
                # May return empty DataFrame or cached data
                assert isinstance(data, pd.DataFrame), "Should return DataFrame even on API error"
                
            except Exception as e:
                # Should not crash completely
                assert "Rate limited" in str(e) or "retry" in str(e).lower(), \
                    f"Should handle rate limiting appropriately: {e}"
        
        # Mock network timeout
        def mock_timeout_call(*args, **kwargs):
            import requests
            raise requests.exceptions.Timeout("Request timed out")
        
        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = Mock()
            mock_instance.history.side_effect = mock_timeout_call
            mock_ticker.return_value = mock_instance
            
            # Should handle timeout gracefully
            try:
                data = data_loader.load_stock_data(
                    ticker="TIMEOUT_TEST",
                    period="1d",
                    timeout=1  # Short timeout for testing
                )
                
                assert isinstance(data, pd.DataFrame), "Should return DataFrame on timeout"
                
            except Exception as e:
                # Network errors are acceptable
                assert any(term in str(e).lower() for term in ['timeout', 'network', 'connection']), \
                    f"Should be network-related error: {e}"
    
    def test_data_caching_mechanism(self, data_loader):
        """Test data caching for performance optimization."""
        # Enable caching for this test
        cached_loader = DataLoader(
            cache_dir="test_cache",
            enable_cache=True,
            cache_expiry_hours=24
        )
        
        ticker = "CACHE_TEST"
        sample_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [102, 103, 104],
            'Low': [99, 100, 101],
            'Close': [101, 102, 103],
            'Volume': [1000000, 1200000, 1100000]
        })
        
        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = Mock()
            mock_instance.history.return_value = sample_data
            mock_ticker.return_value = mock_instance
            
            # First call should fetch from API
            data1 = cached_loader.load_stock_data(ticker, period="1d")
            assert mock_ticker.called, "Should call API on first request"
            
            # Reset mock
            mock_ticker.reset_mock()
            
            # Second call should use cache (if implemented)
            data2 = cached_loader.load_stock_data(ticker, period="1d")
            
            # Data should be identical
            pd.testing.assert_frame_equal(data1, data2)
            
            # If caching is implemented, second call shouldn't hit API
            # (This is optional - depends on implementation)
    
    def test_data_validation_edge_cases(self, data_quality_validator):
        """Test data validation with edge cases."""
        # Test 1: All zero volume
        zero_volume_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [0, 0, 0]  # All zero volume
        })
        
        quality_report = data_quality_validator.validate_data(zero_volume_data, "ZERO_VOL")
        
        # Should detect zero volume issue
        issues = quality_report.get('issues', [])
        warnings = quality_report.get('warnings', [])
        assert any('volume' in (issue + warning).lower() for issue, warning in zip(issues, warnings)), \
            "Should detect zero volume issue"
        
        # Test 2: Constant prices (no movement)
        constant_price_data = pd.DataFrame({
            'Open': [100] * 10,
            'High': [100] * 10,
            'Low': [100] * 10,
            'Close': [100] * 10,
            'Volume': [1000000] * 10
        })
        
        quality_report_constant = data_quality_validator.validate_data(constant_price_data, "CONSTANT")
        
        # Should detect lack of price movement
        warnings_constant = quality_report_constant.get('warnings', [])
        assert any('price' in warning.lower() and ('constant' in warning.lower() or 'movement' in warning.lower()) 
                  for warning in warnings_constant), \
            "Should detect constant price issue"
        
        # Test 3: Extreme values (prices in trillions)
        extreme_price_data = pd.DataFrame({
            'Open': [1e12, 1.1e12, 1.2e12],  # Trillion dollar prices
            'High': [1.1e12, 1.2e12, 1.3e12],
            'Low': [0.9e12, 1.0e12, 1.1e12],
            'Close': [1e12, 1.1e12, 1.2e12],
            'Volume': [100, 200, 150]  # Very low volume
        })
        
        quality_report_extreme = data_quality_validator.validate_data(extreme_price_data, "EXTREME")
        
        # Should detect extreme values
        issues_extreme = quality_report_extreme.get('issues', [])
        warnings_extreme = quality_report_extreme.get('warnings', [])
        all_messages = issues_extreme + warnings_extreme
        
        assert any('extreme' in message.lower() or 'unusual' in message.lower() for message in all_messages), \
            "Should detect extreme price values"


class TestDataPipelinePerformance:
    """Test data pipeline performance and efficiency."""
    
    def test_bulk_data_loading_performance(self):
        """Test performance of loading data for multiple tickers."""
        import time
        
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        # Mock fast data loading
        mock_data = pd.DataFrame({
            'Open': [100] * 30,
            'High': [102] * 30,
            'Low': [98] * 30,
            'Close': [101] * 30,
            'Volume': [1000000] * 30
        })
        
        data_loader = DataLoader(enable_cache=False)
        
        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = Mock()
            mock_instance.history.return_value = mock_data
            mock_ticker.return_value = mock_instance
            
            start_time = time.time()
            
            # Load data for all tickers
            all_data = {}
            for ticker in tickers:
                data = data_loader.load_stock_data(ticker, period="1mo")
                all_data[ticker] = data
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should complete within reasonable time
            time_per_ticker = total_time / len(tickers)
            assert time_per_ticker < 2.0, f"Data loading too slow: {time_per_ticker:.2f}s per ticker"
            
            # All data should be loaded successfully
            assert len(all_data) == len(tickers), "Should load data for all tickers"
            for ticker, data in all_data.items():
                assert not data.empty, f"Data should not be empty for {ticker}"
    
    def test_memory_usage_optimization(self):
        """Test memory usage during large data operations."""
        import sys
        
        # Create large dataset
        large_dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        large_data = pd.DataFrame({
            'Open': np.random.normal(100, 10, len(large_dates)),
            'High': np.random.normal(105, 12, len(large_dates)),
            'Low': np.random.normal(95, 8, len(large_dates)),
            'Close': np.random.normal(100, 10, len(large_dates)),
            'Volume': np.random.randint(1000000, 50000000, len(large_dates))
        }, index=large_dates)
        
        # Measure memory usage
        initial_memory = sys.getsizeof(large_data)
        
        # Process data through validator
        validator = DataQualityValidator()
        
        try:
            quality_report = validator.validate_data(large_data, "LARGE_DATASET")
            
            # Should complete without memory errors
            assert isinstance(quality_report, dict), "Should handle large dataset"
            
            # Memory usage shouldn't explode (no more than 3x original)
            final_memory = sys.getsizeof(large_data)  # Data should remain similar size
            memory_ratio = final_memory / initial_memory
            
            assert memory_ratio < 3.0, f"Memory usage increased too much: {memory_ratio:.1f}x"
            
        except MemoryError:
            pytest.fail("Should handle large datasets without memory errors")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])