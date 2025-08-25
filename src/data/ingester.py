"""Data ingestion module for market data collection using yfinance."""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from loguru import logger
import time
import sqlite3
from pathlib import Path

from ..config.settings import config

class DataIngester:
    """Real-time and historical data ingestion using yfinance API."""
    
    def __init__(self, tickers: Optional[List[str]] = None):
        """Initialize DataIngester with stock tickers.
        
        Args:
            tickers: List of stock symbols. Defaults to config tickers.
        """
        self.tickers = tickers or config.STOCK_TICKERS
        self.rate_limit_delay = 0.1  # 100ms between requests to respect rate limits
        self.max_retries = 3
        
        logger.info(f"Initialized DataIngester with {len(self.tickers)} tickers: {self.tickers}")
    
    def fetch_historical_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical OHLCV data for all tickers.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to today)
            interval: Data interval (1d, 1h, 5m, etc.)
            
        Returns:
            Dictionary mapping ticker symbols to DataFrames with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Fetching historical data from {start_date} to {end_date}, interval: {interval}")
        
        data_dict = {}
        failed_tickers = []
        
        for ticker in self.tickers:
            try:
                logger.debug(f"Fetching data for {ticker}")
                
                # Add retry logic for API failures
                for attempt in range(self.max_retries):
                    try:
                        stock = yf.Ticker(ticker)
                        data = stock.history(
                            start=start_date,
                            end=end_date,
                            interval=interval,
                            auto_adjust=True,  # Adjust for splits and dividends
                            prepost=False      # Regular market hours only
                        )
                        
                        if data.empty:
                            logger.warning(f"No data returned for {ticker}")
                            failed_tickers.append(ticker)
                            break
                            
                        # Clean column names and add metadata
                        data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                        data['ticker'] = ticker
                        data['interval'] = interval
                        data.index.name = 'timestamp'
                        
                        # Validate data quality
                        if self._validate_data_quality(data, ticker):
                            data_dict[ticker] = data
                            logger.debug(f"Successfully fetched {len(data)} records for {ticker}")
                        else:
                            failed_tickers.append(ticker)
                            
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                        if attempt == self.max_retries - 1:
                            failed_tickers.append(ticker)
                        time.sleep(self.rate_limit_delay * (attempt + 1))
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {ticker}: {e}")
                failed_tickers.append(ticker)
        
        if failed_tickers:
            logger.warning(f"Failed to fetch data for: {failed_tickers}")
            
        logger.info(f"Successfully fetched data for {len(data_dict)} tickers")
        return data_dict
    
    def fetch_real_time_data(
        self,
        interval: str = "5m",
        period: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch recent real-time data for all tickers.
        
        Args:
            interval: Data interval (5m, 1m, etc.)
            period: Time period (1d, 1h, etc.)
            
        Returns:
            Dictionary mapping ticker symbols to recent DataFrames
        """
        logger.info(f"Fetching real-time data, interval: {interval}, period: {period}")
        
        data_dict = {}
        
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    prepost=False
                )
                
                if not data.empty:
                    # Clean column names and add metadata
                    data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                    data['ticker'] = ticker
                    data['interval'] = interval
                    data.index.name = 'timestamp'
                    
                    data_dict[ticker] = data
                    
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Failed to fetch real-time data for {ticker}: {e}")
        
        logger.info(f"Fetched real-time data for {len(data_dict)} tickers")
        return data_dict
    
    def get_stock_info(self, ticker: str) -> Dict:
        """Get stock information and metadata.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key information
            stock_info = {
                'ticker': ticker,
                'company_name': info.get('longName', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'beta': info.get('beta', 1.0),
                'avg_volume': info.get('averageVolume', 0),
                'pe_ratio': info.get('forwardPE', None),
                'dividend_yield': info.get('dividendYield', 0),
                'last_updated': datetime.now()
            }
            
            return stock_info
            
        except Exception as e:
            logger.error(f"Failed to fetch info for {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}
    
    def _validate_data_quality(self, data: pd.DataFrame, ticker: str) -> bool:
        """Validate data quality and completeness.
        
        Args:
            data: DataFrame with OHLCV data
            ticker: Stock symbol for logging
            
        Returns:
            True if data quality is acceptable
        """
        if data.empty:
            logger.warning(f"Empty dataset for {ticker}")
            return False
            
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.warning(f"Missing columns for {ticker}: {missing_columns}")
            return False
        
        # Check for excessive missing values
        missing_pct = data[required_columns].isnull().sum() / len(data)
        if (missing_pct > 0.1).any():  # More than 10% missing values
            logger.warning(f"Excessive missing values for {ticker}: {missing_pct}")
            return False
        
        # Basic data integrity checks
        if (data['high'] < data['low']).any():
            logger.warning(f"Invalid high/low values for {ticker}")
            return False
            
        if (data['high'] < data['open']).any() or (data['high'] < data['close']).any():
            logger.warning(f"Invalid high values for {ticker}")
            return False
            
        if (data['low'] > data['open']).any() or (data['low'] > data['close']).any():
            logger.warning(f"Invalid low values for {ticker}")
            return False
        
        # Check for non-positive prices
        price_columns = ['open', 'high', 'low', 'close']
        if (data[price_columns] <= 0).any().any():
            logger.warning(f"Non-positive prices found for {ticker}")
            return False
        
        logger.debug(f"Data quality validation passed for {ticker}")
        return True
    
    def fetch_single_ticker_data(self, ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Fetch data for a single ticker (compatibility method for tests).
        
        Args:
            ticker: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data for the ticker
        """
        try:
            result = self.fetch_historical_data(start_date, end_date, interval)
            return result.get(ticker, pd.DataFrame())
        except Exception as e:
            logger.error(f"Error fetching single ticker data for {ticker}: {e}")
            return pd.DataFrame()
    
    def _check_data_freshness(self, data: pd.DataFrame) -> bool:
        """Check if data is fresh (private method for test compatibility).
        
        Args:
            data: DataFrame with time-indexed data
            
        Returns:
            True if data is considered fresh (less than 24 hours old)
        """
        if data.empty:
            return False
            
        try:
            # Get the latest timestamp from the data
            latest_date = data.index.max()
            
            # Handle timezone-aware vs naive datetime
            if hasattr(latest_date, 'tz') and latest_date.tz is not None:
                # Convert to naive datetime for comparison
                latest_date = latest_date.tz_localize(None) if latest_date.tz else latest_date
            
            # Calculate age
            now = datetime.now()
            age = now - latest_date
            
            # Consider data fresh if less than 24 hours old
            return age < timedelta(hours=24)
            
        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
            return False

    def get_data_freshness(self, ticker: str, interval: str = "5m") -> timedelta:
        """Check how fresh the latest data is for a ticker.
        
        Args:
            ticker: Stock symbol
            interval: Data interval to check
            
        Returns:
            Time difference between now and latest data point
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d", interval=interval)
            
            if data.empty:
                return timedelta(hours=24)  # Very stale
                
            latest_timestamp = data.index[-1].tz_localize(None)
            current_time = datetime.now()
            
            return current_time - latest_timestamp
            
        except Exception as e:
            logger.error(f"Failed to check data freshness for {ticker}: {e}")
            return timedelta(hours=24)  # Assume very stale on error