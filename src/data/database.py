"""Database management for OHLCV data storage using SQLite."""

import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from datetime import datetime
import numpy as np
from loguru import logger

from ..config.settings import config, DATA_DIR

class TradingDatabase:
    """SQLite database manager for trading data storage."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection and create tables.
        
        Args:
            db_path: Path to SQLite database file. Defaults to config setting.
        """
        if db_path is None:
            db_path = DATA_DIR / "trading_data.db"
        else:
            db_path = Path(db_path)
            
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database and create tables
        self._create_tables()
        logger.info(f"Database initialized at {self.db_path}")
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper configuration."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,  # 30 second timeout
            check_same_thread=False  # Allow multi-threading
        )
        
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=memory")
        
        return conn
    
    def _create_tables(self) -> None:
        """Create database tables with proper indexing."""
        
        with self.get_connection() as conn:
            # OHLCV data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    interval TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    dividends REAL DEFAULT 0,
                    stock_splits REAL DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, timestamp, interval)
                )
            """)
            
            # Stock information table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_info (
                    ticker TEXT PRIMARY KEY,
                    company_name TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap INTEGER,
                    beta REAL,
                    avg_volume INTEGER,
                    pe_ratio REAL,
                    dividend_yield REAL,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Trading signals table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    signal_type TEXT NOT NULL, -- 'BUY', 'SELL', 'HOLD'
                    confidence REAL NOT NULL,
                    prediction REAL,
                    prediction_interval_lower REAL,
                    prediction_interval_upper REAL,
                    model_name TEXT,
                    timeframe TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Portfolio positions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    position_size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_timestamp DATETIME NOT NULL,
                    exit_price REAL,
                    exit_timestamp DATETIME,
                    position_type TEXT NOT NULL, -- 'LONG', 'SHORT'
                    stop_loss REAL,
                    take_profit REAL,
                    status TEXT DEFAULT 'OPEN', -- 'OPEN', 'CLOSED'
                    pnl REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Performance metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    portfolio_value REAL NOT NULL,
                    daily_return REAL,
                    cumulative_return REAL,
                    drawdown REAL,
                    sharpe_ratio REAL,
                    win_rate REAL,
                    num_trades INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            """)
            
            # Create indexes for better query performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_timestamp ON ohlcv_data(ticker, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv_data(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_interval ON ohlcv_data(ticker, interval)",
                "CREATE INDEX IF NOT EXISTS idx_signals_ticker_timestamp ON trading_signals(ticker, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON trading_signals(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_positions_ticker ON portfolio_positions(ticker)",
                "CREATE INDEX IF NOT EXISTS idx_positions_status ON portfolio_positions(status)",
                "CREATE INDEX IF NOT EXISTS idx_performance_date ON performance_metrics(date)",
            ]
            
            for index_sql in indexes:
                conn.execute(index_sql)
            
            conn.commit()
            logger.info("Database tables and indexes created successfully")
    
    def insert_ohlcv_data(self, data_dict: Dict[str, pd.DataFrame]) -> int:
        """Insert OHLCV data into database.
        
        Args:
            data_dict: Dictionary mapping tickers to DataFrames with OHLCV data
            
        Returns:
            Number of records inserted
        """
        total_inserted = 0
        
        with self.get_connection() as conn:
            for ticker, df in data_dict.items():
                if df.empty:
                    continue
                    
                # Prepare data for insertion
                df_copy = df.copy()
                df_copy['ticker'] = ticker
                df_copy.reset_index(inplace=True)
                
                # Rename columns to match database schema
                column_mapping = {
                    'timestamp': 'timestamp',
                    'open': 'open',
                    'high': 'high', 
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                    'dividends': 'dividends',
                    'stock_splits': 'stock_splits'
                }
                
                # Select and rename only existing columns
                available_columns = {k: v for k, v in column_mapping.items() if k in df_copy.columns}
                df_insert = df_copy[list(available_columns.keys()) + ['ticker', 'interval']].rename(columns=available_columns)
                
                # Handle missing optional columns
                for col in ['dividends', 'stock_splits']:
                    if col not in df_insert.columns:
                        df_insert[col] = 0.0
                
                try:
                    # Use INSERT OR REPLACE to handle duplicates
                    records_before = conn.execute("SELECT COUNT(*) FROM ohlcv_data").fetchone()[0]
                    
                    df_insert.to_sql('ohlcv_data', conn, if_exists='append', index=False)
                    
                    records_after = conn.execute("SELECT COUNT(*) FROM ohlcv_data").fetchone()[0]
                    inserted = records_after - records_before
                    total_inserted += inserted
                    
                    logger.debug(f"Inserted {inserted} records for {ticker}")
                    
                except sqlite3.IntegrityError as e:
                    if "UNIQUE constraint failed" in str(e):
                        logger.debug(f"Skipping duplicate records for {ticker}")
                    else:
                        logger.error(f"Error inserting data for {ticker}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error inserting data for {ticker}: {e}")
        
        logger.info(f"Total records inserted: {total_inserted}")
        return total_inserted
    
    def get_ohlcv_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Retrieve OHLCV data from database.
        
        Args:
            ticker: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with OHLCV data
        """
        query = """
            SELECT timestamp, open, high, low, close, volume, dividends, stock_splits
            FROM ohlcv_data 
            WHERE ticker = ? AND interval = ?
        """
        params = [ticker, interval]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
            
        query += " ORDER BY timestamp ASC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        logger.debug(f"Retrieved {len(df)} records for {ticker}")
        return df
    
    def get_latest_timestamp(self, ticker: str, interval: str = "1d") -> Optional[datetime]:
        """Get the latest timestamp for a ticker and interval.
        
        Args:
            ticker: Stock symbol
            interval: Data interval
            
        Returns:
            Latest timestamp or None if no data exists
        """
        query = """
            SELECT MAX(timestamp) as latest_timestamp
            FROM ohlcv_data 
            WHERE ticker = ? AND interval = ?
        """
        
        with self.get_connection() as conn:
            result = conn.execute(query, [ticker, interval]).fetchone()
            
        if result and result[0]:
            return datetime.fromisoformat(result[0])
        return None
    
    def insert_stock_info(self, stock_info: Dict[str, Any]) -> None:
        """Insert or update stock information.
        
        Args:
            stock_info: Dictionary with stock information
        """
        with self.get_connection() as conn:
            # Use INSERT OR REPLACE for upsert behavior
            conn.execute("""
                INSERT OR REPLACE INTO stock_info 
                (ticker, company_name, sector, industry, market_cap, beta, 
                 avg_volume, pe_ratio, dividend_yield, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                stock_info.get('ticker'),
                stock_info.get('company_name'),
                stock_info.get('sector'),
                stock_info.get('industry'),
                stock_info.get('market_cap'),
                stock_info.get('beta'),
                stock_info.get('avg_volume'),
                stock_info.get('pe_ratio'),
                stock_info.get('dividend_yield'),
                stock_info.get('last_updated', datetime.now())
            ])
            
            conn.commit()
        
        logger.debug(f"Updated stock info for {stock_info.get('ticker')}")
    
    def get_data_coverage_report(self) -> pd.DataFrame:
        """Generate a report of data coverage by ticker and interval.
        
        Returns:
            DataFrame with data coverage statistics
        """
        query = """
            SELECT 
                ticker,
                interval,
                COUNT(*) as record_count,
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date,
                MAX(created_at) as last_updated
            FROM ohlcv_data 
            GROUP BY ticker, interval
            ORDER BY ticker, interval
        """
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, parse_dates=['start_date', 'end_date', 'last_updated'])
        
        return df
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> int:
        """Remove old data beyond specified retention period.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Number of records deleted
        """
        cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
        
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM ohlcv_data WHERE timestamp < ?",
                [cutoff_date]
            )
            deleted_count = cursor.rowcount
            conn.commit()
        
        logger.info(f"Deleted {deleted_count} old records (older than {days_to_keep} days)")
        return deleted_count
    
    def vacuum_database(self) -> None:
        """Optimize database by running VACUUM command."""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
            
        logger.info("Database vacuum completed")