#!/usr/bin/env python3
"""
Backup Data Sources for Market Data
Multi-source market data system with automatic failover and redundancy.
"""

import sys
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config

warnings.filterwarnings('ignore')


class DataSourceStatus(Enum):
    """Data source status levels."""
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    DISABLED = "disabled"


class DataQuality(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class DataSourceMetrics:
    """Metrics for data source monitoring."""
    source_name: str
    status: DataSourceStatus
    quality: DataQuality
    last_successful_call: Optional[datetime]
    success_rate: float  # 0.0 to 1.0
    avg_response_time_ms: float
    data_freshness_minutes: float
    error_count: int
    total_requests: int


class BaseDataSource:
    """Base class for market data sources."""
    
    def __init__(self, name: str, priority: int = 100):
        """Initialize data source.
        
        Args:
            name: Data source name
            priority: Priority (lower = higher priority)
        """
        self.name = name
        self.priority = priority
        self.status = DataSourceStatus.ACTIVE
        
        # Performance tracking
        self.success_count = 0
        self.error_count = 0
        self.response_times = []
        self.last_successful_call = None
        self.last_error = None
        
    def fetch_data(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch market data for symbols.
        
        Args:
            symbols: List of stock symbols
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        raise NotImplementedError("Subclasses must implement fetch_data")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price or None if unavailable
        """
        try:
            data = self.fetch_data([symbol], period="2d", interval="1d")
            if symbol in data and not data[symbol].empty:
                return float(data[symbol]['close'].iloc[-1])
        except Exception as e:
            logger.debug(f"Error getting current price for {symbol} from {self.name}: {e}")
        
        return None
    
    def test_connection(self) -> bool:
        """Test data source connection.
        
        Returns:
            True if connection successful
        """
        try:
            # Test with a reliable stock
            result = self.get_current_price('AAPL')
            success = result is not None and result > 0
            
            if success:
                self.success_count += 1
                self.last_successful_call = datetime.now()
            else:
                self.error_count += 1
                
            return success
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            return False
    
    def get_metrics(self) -> DataSourceMetrics:
        """Get performance metrics for this data source."""
        total_requests = self.success_count + self.error_count
        success_rate = self.success_count / max(1, total_requests)
        
        # Determine status based on success rate
        if success_rate >= 0.95:
            status = DataSourceStatus.ACTIVE
        elif success_rate >= 0.7:
            status = DataSourceStatus.DEGRADED
        else:
            status = DataSourceStatus.FAILED
        
        # Determine quality
        if success_rate >= 0.98 and len(self.response_times) > 0 and np.mean(self.response_times) < 2000:
            quality = DataQuality.EXCELLENT
        elif success_rate >= 0.95:
            quality = DataQuality.GOOD
        elif success_rate >= 0.8:
            quality = DataQuality.FAIR
        elif success_rate >= 0.5:
            quality = DataQuality.POOR
        else:
            quality = DataQuality.INVALID
        
        # Calculate freshness
        freshness_minutes = 0.0
        if self.last_successful_call:
            freshness_minutes = (datetime.now() - self.last_successful_call).total_seconds() / 60
        
        return DataSourceMetrics(
            source_name=self.name,
            status=status,
            quality=quality,
            last_successful_call=self.last_successful_call,
            success_rate=success_rate,
            avg_response_time_ms=np.mean(self.response_times) if self.response_times else 0.0,
            data_freshness_minutes=freshness_minutes,
            error_count=self.error_count,
            total_requests=total_requests
        )


class YahooFinanceSource(BaseDataSource):
    """Yahoo Finance data source."""
    
    def __init__(self):
        super().__init__("Yahoo Finance", priority=10)  # High priority
    
    def fetch_data(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data from Yahoo Finance."""
        start_time = time.time()
        
        try:
            import yfinance as yf
            
            data = {}
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period=period, interval=interval)
                    
                    if not df.empty:
                        # Standardize column names
                        df.columns = [col.lower() for col in df.columns]
                        data[symbol] = df
                    
                except Exception as e:
                    logger.debug(f"Error fetching {symbol} from Yahoo Finance: {e}")
                    continue
            
            # Record performance
            response_time = (time.time() - start_time) * 1000
            self.response_times.append(response_time)
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]  # Keep last 100
            
            if data:
                self.success_count += 1
                self.last_successful_call = datetime.now()
            else:
                self.error_count += 1
            
            return data
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Yahoo Finance fetch error: {e}")
            return {}


class AlphaVantageSource(BaseDataSource):
    """Alpha Vantage data source."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Alpha Vantage", priority=20)
        self.api_key = api_key or "demo"  # Use demo key if none provided
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 1.0  # 1 second between calls for free tier
        self.last_call_time = 0
    
    def fetch_data(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data from Alpha Vantage."""
        if self.api_key == "demo":
            logger.warning("Using Alpha Vantage demo key - limited functionality")
        
        data = {}
        
        for symbol in symbols:
            try:
                # Rate limiting
                time_since_last = time.time() - self.last_call_time
                if time_since_last < self.rate_limit_delay:
                    time.sleep(self.rate_limit_delay - time_since_last)
                
                start_time = time.time()
                
                # Determine function based on interval
                if interval in ['1d', '5d', '1wk', '1mo']:
                    function = "TIME_SERIES_DAILY"
                elif interval in ['1m', '5m', '15m', '30m', '60m']:
                    function = "TIME_SERIES_INTRADAY"
                else:
                    function = "TIME_SERIES_DAILY"
                
                params = {
                    'function': function,
                    'symbol': symbol,
                    'apikey': self.api_key,
                    'outputsize': 'full' if period in ['1y', '2y', '5y', 'max'] else 'compact'
                }
                
                if function == "TIME_SERIES_INTRADAY":
                    params['interval'] = interval
                
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                json_data = response.json()
                
                # Check for API errors
                if 'Error Message' in json_data:
                    logger.error(f"Alpha Vantage error for {symbol}: {json_data['Error Message']}")
                    continue
                
                if 'Note' in json_data:
                    logger.warning(f"Alpha Vantage rate limited: {json_data['Note']}")
                    break  # Stop processing to avoid further rate limiting
                
                # Parse time series data
                time_series_key = None
                for key in json_data.keys():
                    if 'Time Series' in key:
                        time_series_key = key
                        break
                
                if not time_series_key or time_series_key not in json_data:
                    logger.warning(f"No time series data for {symbol} from Alpha Vantage")
                    continue
                
                time_series = json_data[time_series_key]
                
                # Convert to DataFrame
                df_data = []
                for date_str, values in time_series.items():
                    row = {
                        'open': float(values.get('1. open', 0)),
                        'high': float(values.get('2. high', 0)),
                        'low': float(values.get('3. low', 0)),
                        'close': float(values.get('4. close', 0)),
                        'volume': int(values.get('5. volume', 0))
                    }
                    row['date'] = pd.to_datetime(date_str)
                    df_data.append(row)
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    df.set_index('date', inplace=True)
                    df.sort_index(inplace=True)
                    
                    # Filter by period if needed
                    if period != 'max':
                        cutoff_date = self._get_period_cutoff(period)
                        df = df[df.index >= cutoff_date]
                    
                    data[symbol] = df
                
                self.last_call_time = time.time()
                
                # Record performance
                response_time = (self.last_call_time - start_time) * 1000
                self.response_times.append(response_time)
                
            except Exception as e:
                logger.error(f"Error fetching {symbol} from Alpha Vantage: {e}")
                continue
        
        if data:
            self.success_count += len(data)
            self.last_successful_call = datetime.now()
        else:
            self.error_count += 1
        
        return data
    
    def _get_period_cutoff(self, period: str) -> datetime:
        """Get cutoff date for period."""
        now = datetime.now()
        
        if period == '1d':
            return now - timedelta(days=1)
        elif period == '5d':
            return now - timedelta(days=5)
        elif period == '1mo':
            return now - timedelta(days=30)
        elif period == '3mo':
            return now - timedelta(days=90)
        elif period == '6mo':
            return now - timedelta(days=180)
        elif period == '1y':
            return now - timedelta(days=365)
        elif period == '2y':
            return now - timedelta(days=730)
        elif period == '5y':
            return now - timedelta(days=1825)
        else:
            return now - timedelta(days=365)  # Default to 1 year


class IEXCloudSource(BaseDataSource):
    """IEX Cloud data source."""
    
    def __init__(self, api_token: Optional[str] = None):
        super().__init__("IEX Cloud", priority=30)
        self.api_token = api_token or "pk_test"  # Use test token if none provided
        self.base_url = "https://cloud.iexapis.com/stable"
    
    def fetch_data(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data from IEX Cloud."""
        if self.api_token.startswith("pk_test"):
            logger.warning("Using IEX Cloud test token - limited functionality")
        
        data = {}
        
        # Map period to IEX range
        range_mapping = {
            '1d': '1d',
            '5d': '5d',
            '1mo': '1m',
            '3mo': '3m',
            '6mo': '6m',
            '1y': '1y',
            '2y': '2y',
            '5y': '5y'
        }
        
        iex_range = range_mapping.get(period, '1y')
        
        for symbol in symbols:
            try:
                start_time = time.time()
                
                # Use chart endpoint for historical data
                url = f"{self.base_url}/stock/{symbol}/chart/{iex_range}"
                params = {'token': self.api_token}
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                json_data = response.json()
                
                if not json_data or not isinstance(json_data, list):
                    continue
                
                # Convert to DataFrame
                df_data = []
                for item in json_data:
                    if not item:
                        continue
                    
                    row = {
                        'open': float(item.get('open', 0)) if item.get('open') else None,
                        'high': float(item.get('high', 0)) if item.get('high') else None,
                        'low': float(item.get('low', 0)) if item.get('low') else None,
                        'close': float(item.get('close', 0)) if item.get('close') else None,
                        'volume': int(item.get('volume', 0)) if item.get('volume') else 0,
                        'date': pd.to_datetime(item.get('date'))
                    }
                    
                    # Skip rows with no price data
                    if all(v is None for v in [row['open'], row['high'], row['low'], row['close']]):
                        continue
                    
                    df_data.append(row)
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    df.set_index('date', inplace=True)
                    df.sort_index(inplace=True)
                    data[symbol] = df
                
                # Record performance
                response_time = (time.time() - start_time) * 1000
                self.response_times.append(response_time)
                
            except Exception as e:
                logger.error(f"Error fetching {symbol} from IEX Cloud: {e}")
                continue
        
        if data:
            self.success_count += len(data)
            self.last_successful_call = datetime.now()
        else:
            self.error_count += 1
        
        return data


class BackupDataManager:
    """Manager for multiple data sources with automatic failover."""
    
    def __init__(
        self,
        sources: Optional[List[BaseDataSource]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """Initialize backup data manager.
        
        Args:
            sources: List of data sources (will create defaults if None)
            max_retries: Maximum retries per source
            retry_delay: Delay between retries in seconds
        """
        if sources is None:
            self.sources = self._create_default_sources()
        else:
            self.sources = sources
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Sort by priority (lower number = higher priority)
        self.sources.sort(key=lambda x: x.priority)
        
        logger.info(f"Backup data manager initialized with {len(self.sources)} sources")
    
    def _create_default_sources(self) -> List[BaseDataSource]:
        """Create default data sources."""
        sources = [
            YahooFinanceSource(),
            # AlphaVantageSource(),  # Requires API key
            # IEXCloudSource(),      # Requires API token
        ]
        
        # Add other sources if API keys are available
        # These would typically be loaded from environment variables
        
        return sources
    
    def fetch_data(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        fallback_to_cache: bool = True
    ) -> Tuple[Dict[str, pd.DataFrame], DataSourceMetrics]:
        """Fetch market data with automatic failover.
        
        Args:
            symbols: List of stock symbols
            period: Time period
            interval: Data interval
            fallback_to_cache: Use cached data if all sources fail
            
        Returns:
            Tuple of (data_dict, source_metrics)
        """
        data = {}
        last_error = None
        successful_source = None
        
        for source in self.sources:
            if source.status == DataSourceStatus.DISABLED:
                continue
            
            logger.debug(f"Trying data source: {source.name}")
            
            for attempt in range(self.max_retries):
                try:
                    source_data = source.fetch_data(symbols, period, interval)
                    
                    if source_data:
                        # Validate data quality
                        valid_data = self._validate_data_quality(source_data)
                        if valid_data:
                            data.update(valid_data)
                            successful_source = source
                            logger.info(f"Successfully fetched {len(valid_data)} symbols from {source.name}")
                            break
                    
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Attempt {attempt + 1} failed for {source.name}: {e}")
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    continue
            
            # If we got data from this source, we're done
            if data:
                break
        
        # If no data was fetched, try fallback options
        if not data and fallback_to_cache:
            logger.warning("All data sources failed, attempting cache fallback")
            data = self._try_cache_fallback(symbols, period, interval)
        
        # Get metrics from successful source or best available source
        if successful_source:
            metrics = successful_source.get_metrics()
        else:
            # Use metrics from the first active source
            active_sources = [s for s in self.sources if s.status != DataSourceStatus.DISABLED]
            metrics = active_sources[0].get_metrics() if active_sources else None
        
        if not data:
            logger.error(f"Failed to fetch data from all sources. Last error: {last_error}")
        
        return data, metrics
    
    def get_current_price(self, symbol: str) -> Tuple[Optional[float], Optional[str]]:
        """Get current price with source information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuple of (price, source_name)
        """
        for source in self.sources:
            if source.status == DataSourceStatus.DISABLED:
                continue
            
            try:
                price = source.get_current_price(symbol)
                if price is not None and price > 0:
                    return price, source.name
            except Exception as e:
                logger.debug(f"Error getting {symbol} price from {source.name}: {e}")
                continue
        
        return None, None
    
    def _validate_data_quality(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Validate and clean data quality.
        
        Args:
            data: Raw data from source
            
        Returns:
            Validated and cleaned data
        """
        validated_data = {}
        
        for symbol, df in data.items():
            if df.empty:
                continue
            
            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing columns for {symbol}")
                continue
            
            # Check for reasonable price ranges (basic sanity check)
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if df[col].min() <= 0 or df[col].max() > 10000:  # $0 to $10k range
                    logger.warning(f"Suspicious price data for {symbol} in column {col}")
                    # Don't skip, but log the warning
            
            # Check for data freshness
            if not df.index.empty:
                latest_date = df.index.max()
                days_old = (datetime.now() - latest_date.to_pydatetime()).days
                
                if days_old > 7:  # More than 1 week old
                    logger.warning(f"Stale data for {symbol}: {days_old} days old")
                    # Continue anyway, stale data is better than no data
            
            # Remove rows with all NaN price data
            df = df.dropna(subset=price_cols, how='all')
            
            if not df.empty:
                validated_data[symbol] = df
        
        return validated_data
    
    def _try_cache_fallback(
        self,
        symbols: List[str],
        period: str,
        interval: str
    ) -> Dict[str, pd.DataFrame]:
        """Try to get data from local cache as fallback."""
        # This would implement a local cache system
        # For now, return empty dict
        logger.info("Cache fallback not implemented")
        return {}
    
    def test_all_sources(self) -> Dict[str, bool]:
        """Test all data sources and return their status.
        
        Returns:
            Dictionary mapping source names to success status
        """
        results = {}
        
        for source in self.sources:
            logger.info(f"Testing {source.name}...")
            success = source.test_connection()
            results[source.name] = success
            
            if success:
                logger.info(f"✅ {source.name}: Connection successful")
            else:
                logger.warning(f"❌ {source.name}: Connection failed")
        
        return results
    
    def get_source_metrics(self) -> List[DataSourceMetrics]:
        """Get metrics for all data sources.
        
        Returns:
            List of DataSourceMetrics for each source
        """
        return [source.get_metrics() for source in self.sources]
    
    def disable_source(self, source_name: str):
        """Disable a data source."""
        for source in self.sources:
            if source.name == source_name:
                source.status = DataSourceStatus.DISABLED
                logger.info(f"Disabled data source: {source_name}")
                return
        
        logger.warning(f"Data source not found: {source_name}")
    
    def enable_source(self, source_name: str):
        """Enable a data source."""
        for source in self.sources:
            if source.name == source_name:
                source.status = DataSourceStatus.ACTIVE
                logger.info(f"Enabled data source: {source_name}")
                return
        
        logger.warning(f"Data source not found: {source_name}")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report for all sources."""
        metrics = self.get_source_metrics()
        
        total_sources = len(metrics)
        active_sources = len([m for m in metrics if m.status == DataSourceStatus.ACTIVE])
        failed_sources = len([m for m in metrics if m.status == DataSourceStatus.FAILED])
        
        # Calculate overall system health
        if active_sources == 0:
            system_health = "CRITICAL"
        elif failed_sources > active_sources:
            system_health = "DEGRADED"
        elif active_sources >= 2:
            system_health = "EXCELLENT"
        else:
            system_health = "GOOD"
        
        return {
            'timestamp': datetime.now(),
            'system_health': system_health,
            'total_sources': total_sources,
            'active_sources': active_sources,
            'degraded_sources': len([m for m in metrics if m.status == DataSourceStatus.DEGRADED]),
            'failed_sources': failed_sources,
            'disabled_sources': len([m for m in metrics if m.status == DataSourceStatus.DISABLED]),
            'source_details': [
                {
                    'name': m.source_name,
                    'status': m.status.value,
                    'quality': m.quality.value,
                    'success_rate': m.success_rate,
                    'avg_response_time_ms': m.avg_response_time_ms,
                    'data_freshness_minutes': m.data_freshness_minutes,
                    'last_successful_call': m.last_successful_call.isoformat() if m.last_successful_call else None
                }
                for m in metrics
            ],
            'recommendations': self._generate_recommendations(metrics)
        }
    
    def _generate_recommendations(self, metrics: List[DataSourceMetrics]) -> List[str]:
        """Generate recommendations based on source performance."""
        recommendations = []
        
        failed_sources = [m for m in metrics if m.status == DataSourceStatus.FAILED]
        if failed_sources:
            recommendations.append(f"CRITICAL: {len(failed_sources)} data source(s) have failed")
        
        degraded_sources = [m for m in metrics if m.status == DataSourceStatus.DEGRADED]
        if degraded_sources:
            recommendations.append(f"WARNING: {len(degraded_sources)} data source(s) are degraded")
        
        stale_sources = [m for m in metrics if m.data_freshness_minutes > 60]
        if stale_sources:
            recommendations.append(f"Data freshness issue: {len(stale_sources)} source(s) have stale data")
        
        slow_sources = [m for m in metrics if m.avg_response_time_ms > 5000]
        if slow_sources:
            recommendations.append(f"Performance issue: {len(slow_sources)} source(s) are slow")
        
        active_sources = [m for m in metrics if m.status == DataSourceStatus.ACTIVE]
        if len(active_sources) == 1:
            recommendations.append("Single point of failure: Only one active data source")
        
        if not recommendations:
            recommendations.append("All data sources operating normally")
        
        return recommendations


def main():
    """Demo and testing of backup data sources."""
    # Create data manager
    manager = BackupDataManager()
    
    print("Backup Data Sources Demo")
    print("=" * 30)
    
    # Test all sources
    print("\n🔍 Testing Data Sources:")
    test_results = manager.test_all_sources()
    
    # Fetch sample data
    print("\n📊 Fetching Sample Data:")
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data, metrics = manager.fetch_data(symbols, period='5d', interval='1d')
    
    print(f"✅ Fetched data for {len(data)} symbols")
    if metrics:
        print(f"📡 Source: {metrics.source_name} (Quality: {metrics.quality.value})")
    
    # Show data samples
    for symbol, df in data.items():
        if not df.empty:
            latest_price = df['close'].iloc[-1]
            print(f"  {symbol}: ${latest_price:.2f} ({len(df)} data points)")
    
    # Get current prices
    print("\n💰 Current Prices:")
    for symbol in symbols:
        price, source = manager.get_current_price(symbol)
        if price:
            print(f"  {symbol}: ${price:.2f} (from {source})")
        else:
            print(f"  {symbol}: No price available")
    
    # Status report
    print("\n📋 System Status Report:")
    report = manager.get_status_report()
    print(f"System Health: {report['system_health']}")
    print(f"Active Sources: {report['active_sources']}/{report['total_sources']}")
    
    print("\nSource Details:")
    for source in report['source_details']:
        status_emoji = {"active": "🟢", "degraded": "🟡", "failed": "🔴", "disabled": "⚫"}
        emoji = status_emoji.get(source['status'], "❓")
        print(f"  {emoji} {source['name']}: {source['status']} "
              f"({source['success_rate']:.1%} success rate)")
    
    print("\nRecommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")


if __name__ == "__main__":
    main()