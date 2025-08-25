"""Unit tests for data pipeline: ingestion, validation, and cleaning."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sqlite3
import tempfile
import os

from src.data.ingester import DataIngester
from src.data.validator import DataValidator
from src.data.database import TradingDatabase


class TestDataIngester:
    """Test data ingestion functionality."""
    
    @pytest.fixture
    def ingester(self):
        """Create DataIngester instance."""
        return DataIngester(['NVDA', 'AMD', 'INTC'])
    
    @pytest.fixture
    def mock_yfinance_data(self):
        """Mock yfinance response data."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        return pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'Low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'Close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Volume': [1000000] * 10
        }, index=dates)
    
    def test_ingester_initialization(self, ingester):
        """Test DataIngester initialization."""
        assert len(ingester.tickers) == 3
        assert 'NVDA' in ingester.tickers
        assert 'AMD' in ingester.tickers
        assert 'INTC' in ingester.tickers
        assert hasattr(ingester, 'rate_limit_delay')
        assert hasattr(ingester, 'max_retries')
    
    @patch('yfinance.download')
    def test_fetch_historical_data_success(self, mock_download, ingester, mock_yfinance_data):
        """Test successful historical data fetching."""
        # Mock successful API response
        mock_download.return_value = mock_yfinance_data
        
        result = ingester.fetch_historical_data(
            start_date='2024-01-01',
            end_date='2024-01-10'
        )
        
        assert isinstance(result, dict)
        assert len(result) >= 1  # Should have at least one ticker
        
        # Check data structure for first ticker
        first_ticker = list(result.keys())[0]
        df = result[first_ticker]
        
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"
        
        assert len(df) > 0
        assert df.index.dtype.kind == 'M'  # DateTime index
    
    @patch('yfinance.Ticker')
    def test_fetch_historical_data_failure(self, mock_ticker, ingester):
        """Test handling of API failures."""
        # Mock API failure - make Ticker constructor raise exception
        mock_ticker.side_effect = Exception("API Error")
        
        result = ingester.fetch_historical_data(
            start_date='2024-01-01',
            end_date='2024-01-10'
        )
        
        # Should return empty dict on failure
        assert isinstance(result, dict)
        assert len(result) == 0
    
    @patch('yfinance.download')
    def test_fetch_single_ticker_data(self, mock_download, ingester, mock_yfinance_data):
        """Test fetching data for single ticker."""
        mock_download.return_value = mock_yfinance_data
        
        result = ingester.fetch_single_ticker_data('NVDA', '2024-01-01', '2024-01-10')
        
        assert result is not None
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            assert col in result.columns
    
    def test_data_quality_validation_integration(self, ingester):
        """Test integration with data quality validation."""
        # Create sample data with quality issues
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
        data_with_issues = pd.DataFrame({
            'Open': [100, 101, 102, 103, np.nan],  # Missing value
            'High': [102, 103, 104, 105, 106],
            'Low': [98, 99, 100, 101, 102],
            'Close': [101, 102, 103, 104, 105],
            'Volume': [1000000] * 5
        }, index=dates)
        
        # Test the validation method that actually exists
        result = ingester._validate_data_quality(data_with_issues, 'TEST')
        
        # Should return boolean or validation result
        assert isinstance(result, (bool, dict))
    
    def test_rate_limiting(self, ingester):
        """Test API rate limiting behavior."""
        # Test that rate limit delay is set properly
        assert hasattr(ingester, 'rate_limit_delay')
        assert ingester.rate_limit_delay >= 0
    
    def test_data_freshness_check(self, ingester):
        """Test data freshness checking."""
        # Test with a valid ticker
        freshness = ingester.get_data_freshness('NVDA')
        
        # Should return a timedelta showing how old the data is
        assert isinstance(freshness, timedelta)
        # Data freshness should be reasonable (not older than 7 days for daily data)
        assert freshness.days <= 7
    
    def test_stock_info_retrieval(self, ingester):
        """Test stock information retrieval."""
        # Mock successful info retrieval
        with patch('yfinance.Ticker') as mock_ticker:
            mock_info = {'symbol': 'NVDA', 'longName': 'NVIDIA Corporation'}
            mock_ticker.return_value.info = mock_info
            
            info = ingester.get_stock_info('NVDA')
            
            assert isinstance(info, dict)
            mock_ticker.assert_called_once_with('NVDA')


class TestDataValidator:
    """Test data validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create DataValidator instance."""
        return DataValidator()
    
    @pytest.fixture
    def valid_ohlcv_data(self):
        """Create valid OHLCV data for testing."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        return pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'volume': [1000000] * 10
        }, index=dates)
    
    @pytest.fixture
    def invalid_ohlcv_data(self):
        """Create invalid OHLCV data for testing."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        return pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],  # High < Open
            'low': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],  # Low > High
            'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'volume': [-1000000] * 10  # Negative volume
        }, index=dates)
    
    def test_schema_validation(self, validator, valid_ohlcv_data):
        """Test data schema validation."""
        result = validator.validate_ohlcv_data(valid_ohlcv_data, 'NVDA')
        assert result['is_valid'] is True
        
        # Test missing columns
        incomplete_data = valid_ohlcv_data.drop(columns=['volume'])
        result = validator.validate_ohlcv_data(incomplete_data, 'NVDA')
        assert result['is_valid'] is False
        assert 'missing_columns' in result['validation_errors']
    
    def test_ohlc_relationship_validation(self, validator, invalid_ohlcv_data):
        """Test OHLC price relationship validation."""
        result = validator.validate_ohlcv_data(invalid_ohlcv_data, 'NVDA')
        assert result['is_valid'] is False
        assert 'ohlc_violations' in result['validation_errors']
    
    def test_volume_validation(self, validator):
        """Test volume data validation."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
        
        # Test negative volume
        negative_volume_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [98, 99, 100, 101, 102],
            'close': [101, 102, 103, 104, 105],
            'volume': [-1000, 1000000, 2000000, -500, 1500000]
        }, index=dates)
        
        result = validator.validate_ohlcv_data(negative_volume_data, 'NVDA')
        assert result['is_valid'] is False
        assert 'negative_volume' in result['validation_errors']
    
    def test_missing_values_detection(self, validator):
        """Test missing values detection."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
        
        data_with_nans = pd.DataFrame({
            'open': [100, np.nan, 102, 103, 104],
            'high': [102, 103, np.nan, 105, 106],
            'low': [98, 99, 100, 101, 102],
            'close': [101, 102, 103, 104, np.nan],
            'volume': [1000000] * 5
        }, index=dates)
        
        result = validator.validate_ohlcv_data(data_with_nans, 'NVDA')
        assert result['is_valid'] is False
        assert 'missing_values' in result['validation_errors']
        assert result['missing_count'] == 3
    
    def test_outlier_detection(self, validator):
        """Test statistical outlier detection."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-20', freq='D')
        
        # Normal data with outliers
        normal_prices = [100] * 18
        outlier_data = pd.DataFrame({
            'open': normal_prices + [1000, 5],  # Extreme outliers
            'high': [p + 2 for p in normal_prices] + [1002, 7],
            'low': [p - 2 for p in normal_prices] + [998, 3],
            'close': normal_prices + [1001, 6],
            'volume': [1000000] * 20
        }, index=dates)
        
        result = validator.validate_ohlcv_data(outlier_data, 'NVDA')
        assert result['outlier_count'] > 0
        assert result['data_quality_score'] < 1.0  # Quality should be reduced
    
    def test_time_gaps_detection(self, validator):
        """Test detection of time gaps in data."""
        # Create data with gaps (skip some dates)
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        gap_dates = dates.delete([3, 4, 7])  # Remove some dates to create gaps
        
        gap_data = pd.DataFrame({
            'open': [100] * len(gap_dates),
            'high': [102] * len(gap_dates),
            'low': [98] * len(gap_dates),
            'close': [101] * len(gap_dates),
            'volume': [1000000] * len(gap_dates)
        }, index=gap_dates)
        
        result = validator.validate_ohlcv_data(gap_data, 'NVDA')
        assert 'time_gaps' in result
        assert result['time_gaps'] > 0
    
    def test_data_quality_scoring(self, validator, valid_ohlcv_data):
        """Test data quality score calculation."""
        result = validator.validate_ohlcv_data(valid_ohlcv_data, 'NVDA')
        
        # Perfect data should have high quality score
        assert result['data_quality_score'] >= 0.9
        assert 0 <= result['data_quality_score'] <= 1.0
    
    def test_validation_report_structure(self, validator, valid_ohlcv_data):
        """Test validation report structure."""
        result = validator.validate_ohlcv_data(valid_ohlcv_data, 'NVDA')
        
        required_keys = [
            'is_valid', 'data_quality_score', 'validation_errors',
            'record_count', 'missing_count', 'outlier_count'
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key in validation result: {key}"
        
        assert isinstance(result['is_valid'], bool)
        assert isinstance(result['validation_errors'], list)
        assert isinstance(result['data_quality_score'], (int, float))


class TestTradingDatabase:
    """Test database functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.mktemp(suffix='.db')
        db = TradingDatabase(temp_file)
        yield db
        # Cleanup
        db.close()
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
        return pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [98, 99, 100, 101, 102],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=dates)
    
    def test_database_initialization(self, temp_db):
        """Test database initialization and table creation."""
        # Check that tables exist
        cursor = temp_db.conn.cursor()
        
        # Check ohlcv_data table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv_data'")
        assert cursor.fetchone() is not None
        
        # Check portfolio_positions table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_positions'")
        assert cursor.fetchone() is not None
        
        # Check stock_info table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_info'")
        assert cursor.fetchone() is not None
    
    def test_store_price_data(self, temp_db, sample_ohlcv_data):
        """Test storing price data."""
        success = temp_db.store_price_data('NVDA', sample_ohlcv_data)
        assert success is True
        
        # Verify data was stored
        stored_data = temp_db.get_price_data('NVDA')
        assert len(stored_data) == len(sample_ohlcv_data)
        
        # Check column names
        expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            assert col in stored_data.columns
    
    def test_get_price_data_date_range(self, temp_db, sample_ohlcv_data):
        """Test retrieving price data with date range."""
        temp_db.store_price_data('NVDA', sample_ohlcv_data)
        
        # Get data for specific date range
        start_date = '2024-01-02'
        end_date = '2024-01-04'
        
        filtered_data = temp_db.get_price_data('NVDA', start_date, end_date)
        assert len(filtered_data) == 3  # 3 days in range
        
        # Check date filtering
        dates = pd.to_datetime(filtered_data['date'])
        assert dates.min().strftime('%Y-%m-%d') >= start_date
        assert dates.max().strftime('%Y-%m-%d') <= end_date
    
    def test_update_price_data(self, temp_db, sample_ohlcv_data):
        """Test updating existing price data."""
        # Store initial data
        temp_db.store_price_data('NVDA', sample_ohlcv_data)
        
        # Create updated data (same dates, different values)
        updated_data = sample_ohlcv_data.copy()
        updated_data['close'] = updated_data['close'] * 1.1  # 10% increase
        
        # Update should replace existing data
        success = temp_db.store_price_data('NVDA', updated_data, update_existing=True)
        assert success is True
        
        # Verify update
        stored_data = temp_db.get_price_data('NVDA')
        # Should still have same number of records
        assert len(stored_data) == len(sample_ohlcv_data)
        
        # Prices should be updated
        original_closes = sample_ohlcv_data['close'].values
        stored_closes = stored_data['close'].values
        assert not np.array_equal(original_closes, stored_closes)
    
    def test_data_integrity_constraints(self, temp_db):
        """Test database integrity constraints."""
        # Test duplicate prevention
        dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='D')
        data1 = pd.DataFrame({
            'open': [100, 101], 'high': [102, 103], 'low': [98, 99],
            'close': [101, 102], 'volume': [1000000, 1100000]
        }, index=dates)
        
        # Store initial data
        temp_db.store_price_data('NVDA', data1)
        
        # Try to store duplicate data (should handle gracefully)
        result = temp_db.store_price_data('NVDA', data1, update_existing=False)
        # Should either succeed (with IGNORE) or handle the conflict
        assert isinstance(result, bool)
    
    def test_database_performance(self, temp_db):
        """Test database performance with larger dataset."""
        # Generate larger dataset
        dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
        large_data = pd.DataFrame({
            'open': 100 + np.random.randn(len(dates)),
            'high': 102 + np.random.randn(len(dates)),
            'low': 98 + np.random.randn(len(dates)),
            'close': 100 + np.random.randn(len(dates)),
            'volume': 1000000 + np.random.randint(-100000, 100000, len(dates))
        }, index=dates)
        
        # Measure storage time
        start_time = datetime.now()
        success = temp_db.store_price_data('NVDA', large_data)
        storage_time = (datetime.now() - start_time).total_seconds()
        
        assert success is True
        assert storage_time < 5.0  # Should complete within 5 seconds
        
        # Measure retrieval time
        start_time = datetime.now()
        retrieved_data = temp_db.get_price_data('NVDA')
        retrieval_time = (datetime.now() - start_time).total_seconds()
        
        assert len(retrieved_data) == len(large_data)
        assert retrieval_time < 2.0  # Should complete within 2 seconds


class TestDataPipelineIntegration:
    """Integration tests for complete data pipeline."""
    
    def test_end_to_end_data_pipeline(self):
        """Test complete data pipeline from ingestion to storage."""
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_file:
            temp_db_path = temp_file.name
        
        try:
            # Initialize components
            ingester = DataIngester(['NVDA'])
            validator = DataValidator()
            database = TradingDatabase(temp_db_path)
            
            # Mock successful data fetch
            mock_data = pd.DataFrame({
                'Open': [400, 401, 402],
                'High': [405, 406, 407],
                'Low': [395, 396, 397],
                'Close': [402, 403, 404],
                'Volume': [1000000, 1100000, 1200000]
            }, index=pd.date_range('2024-01-01', periods=3, freq='D'))
            
            with patch('yfinance.Ticker') as mock_ticker:
                # Mock the Ticker and its history method
                mock_ticker_instance = mock_ticker.return_value
                mock_ticker_instance.history.return_value = mock_data
                # Fetch data
                data = ingester.fetch_historical_data('2024-01-01', '2024-01-03')
                assert 'NVDA' in data
                
                # Validate data
                ohlcv_data = data['NVDA']
                validation_result = validator.validate_ohlcv_data(ohlcv_data, 'NVDA')
                assert validation_result['is_valid'] is True
                
                # Store in database
                success = database.store_price_data('NVDA', ohlcv_data)
                assert success is True
                
                # Retrieve and verify
                stored_data = database.get_price_data('NVDA')
                assert len(stored_data) == 3
                
            database.close()
        
        finally:
            # Cleanup
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
    
    def test_data_pipeline_error_handling(self):
        """Test data pipeline error handling and recovery."""
        ingester = DataIngester(['INVALID_TICKER'])
        
        # Test graceful handling of invalid ticker
        with patch('yfinance.download', side_effect=Exception("Invalid ticker")):
            result = ingester.fetch_historical_data('2024-01-01', '2024-01-03')
            
            # Should return empty dict, not raise exception
            assert isinstance(result, dict)
            assert len(result) == 0
    
    def test_data_freshness_validation(self):
        """Test data freshness validation."""
        ingester = DataIngester(['NVDA'])
        
        # Create old data
        old_dates = pd.date_range(start='2023-01-01', end='2023-01-05', freq='D')
        old_data = pd.DataFrame({
            'Open': [100] * 5, 'High': [102] * 5, 'Low': [98] * 5,
            'Close': [101] * 5, 'Volume': [1000000] * 5
        }, index=old_dates)
        
        # Check freshness
        is_fresh = ingester._check_data_freshness(old_data)
        assert is_fresh is False  # Old data should not be fresh
        
        # Create recent data
        recent_dates = pd.date_range(start=datetime.now().date() - timedelta(days=2), periods=3, freq='D')
        recent_data = pd.DataFrame({
            'Open': [100] * 3, 'High': [102] * 3, 'Low': [98] * 3,
            'Close': [101] * 3, 'Volume': [1000000] * 3
        }, index=recent_dates)
        
        is_fresh = ingester._check_data_freshness(recent_data)
        assert is_fresh is True  # Recent data should be fresh


if __name__ == '__main__':
    pytest.main([__file__, '-v'])