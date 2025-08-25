"""Portfolio management for position tracking and P&L calculation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger
import sqlite3
from pathlib import Path

from ..data.database import TradingDatabase
from .signal_generator import TradingSignal, SignalType
from .position_sizer import PositionSizeResult
from ..config.settings import config


class PositionStatus(Enum):
    """Position status types."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING = "PENDING"


class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


@dataclass
class Position:
    """Individual position in the portfolio."""
    ticker: str
    size: float  # Number of shares (positive = long, negative = short)
    entry_price: float
    entry_timestamp: datetime
    current_price: float = 0.0
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    pnl_realized: float = 0.0
    pnl_unrealized: float = 0.0
    commission: float = 0.0
    signal_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.size > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.size < 0
    
    @property
    def market_value(self) -> float:
        """Calculate current market value of position."""
        return abs(self.size) * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Calculate cost basis of position."""
        return abs(self.size) * self.entry_price


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_value: float
    cash_balance: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    daily_return: float
    cumulative_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    largest_win: float
    largest_loss: float
    avg_holding_period: float  # days
    current_drawdown: float
    exposure: float  # % of portfolio invested


class PortfolioManager:
    """Comprehensive portfolio management system."""
    
    def __init__(
        self,
        initial_capital: float,
        commission_rate: float = 0.001,  # 0.1% commission
        database: Optional[TradingDatabase] = None
    ):
        """Initialize portfolio manager.
        
        Args:
            initial_capital: Starting capital
            commission_rate: Commission rate per trade
            database: Database connection for persistence
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.database = database or TradingDatabase()
        
        # Portfolio state
        self.cash_balance = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Performance tracking
        self.daily_values: List[Tuple[datetime, float]] = [(datetime.now(), initial_capital)]
        self.high_water_mark = initial_capital
        self.trade_history: List[Dict[str, Any]] = []
        
        # Risk tracking
        self.max_position_value = initial_capital * 0.1  # 10% max position
        self.max_total_exposure = initial_capital * 0.8  # 80% max exposure
        
        logger.info(f"Portfolio initialized with ${initial_capital:,.2f} capital")
    
    def open_position(
        self,
        ticker: str,
        size: float,
        entry_price: float,
        signal: Optional[TradingSignal] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """Open a new position.
        
        Args:
            ticker: Stock ticker
            size: Position size (shares, positive = long, negative = short)
            entry_price: Entry price per share
            signal: Trading signal that generated this position
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            True if position opened successfully
        """
        try:
            # Calculate position value and commission
            position_value = abs(size) * entry_price
            commission = position_value * self.commission_rate
            total_cost = position_value + commission
            
            # Check available capital
            if total_cost > self.cash_balance:
                logger.warning(f"Insufficient capital for {ticker}: ${total_cost:,.2f} > ${self.cash_balance:,.2f}")
                return False
            
            # Check position size limits
            if position_value > self.max_position_value:
                logger.warning(f"Position too large for {ticker}: ${position_value:,.2f}")
                return False
            
            # Check if position already exists
            if ticker in self.positions:
                logger.warning(f"Position already exists for {ticker}")
                return False
            
            # Create position
            position = Position(
                ticker=ticker,
                size=size,
                entry_price=entry_price,
                entry_timestamp=datetime.now(),
                current_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                commission=commission,
                signal_data=signal.__dict__ if signal else None
            )
            
            # Update portfolio
            self.positions[ticker] = position
            self.cash_balance -= total_cost
            
            # Log trade
            trade_record = {
                'ticker': ticker,
                'action': 'OPEN',
                'size': size,
                'price': entry_price,
                'timestamp': datetime.now(),
                'commission': commission,
                'signal_type': signal.signal_type.value if signal else None
            }
            self.trade_history.append(trade_record)
            
            # Persist to database
            self._save_position_to_db(position)
            
            logger.info(f"Opened {size:+.0f} shares of {ticker} at ${entry_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to open position for {ticker}: {e}")
            return False
    
    def close_position(
        self,
        ticker: str,
        exit_price: Optional[float] = None,
        partial_size: Optional[float] = None,
        reason: str = "MANUAL"
    ) -> bool:
        """Close an existing position.
        
        Args:
            ticker: Stock ticker
            exit_price: Exit price (uses current_price if None)
            partial_size: Close partial position (closes all if None)
            reason: Reason for closing
            
        Returns:
            True if position closed successfully
        """
        if ticker not in self.positions:
            logger.warning(f"No position exists for {ticker}")
            return False
        
        try:
            position = self.positions[ticker]
            
            # Determine exit price
            if exit_price is None:
                exit_price = position.current_price
            
            # Determine size to close
            if partial_size is None:
                close_size = position.size
            else:
                close_size = min(abs(partial_size), abs(position.size))
                if position.size < 0:  # Short position
                    close_size = -close_size
            
            # Calculate P&L
            if position.is_long:
                pnl = close_size * (exit_price - position.entry_price)
            else:
                pnl = -close_size * (exit_price - position.entry_price)
            
            # Calculate commission
            close_value = abs(close_size) * exit_price
            commission = close_value * self.commission_rate
            net_pnl = pnl - commission - position.commission
            
            # Update cash balance
            proceeds = close_value - commission
            self.cash_balance += proceeds
            
            # Update position
            position.exit_price = exit_price
            position.exit_timestamp = datetime.now()
            position.pnl_realized = net_pnl
            
            # Handle partial vs full close
            closed_position = None
            if abs(close_size) >= abs(position.size):
                # Full close
                position.status = PositionStatus.CLOSED
                closed_position = position
                self.closed_positions.append(position)
                del self.positions[ticker]
            else:
                # Partial close - create new position for remainder
                remaining_size = position.size - close_size
                position.size = remaining_size
                
                # Create closed position record
                closed_position = Position(
                    ticker=ticker,
                    size=close_size,
                    entry_price=position.entry_price,
                    entry_timestamp=position.entry_timestamp,
                    current_price=exit_price,
                    exit_price=exit_price,
                    exit_timestamp=datetime.now(),
                    status=PositionStatus.CLOSED,
                    pnl_realized=net_pnl,
                    commission=commission + position.commission
                )
                self.closed_positions.append(closed_position)
            
            # Log trade
            trade_record = {
                'ticker': ticker,
                'action': 'CLOSE',
                'size': close_size,
                'price': exit_price,
                'timestamp': datetime.now(),
                'pnl': net_pnl,
                'commission': commission,
                'reason': reason
            }
            self.trade_history.append(trade_record)
            
            # Update database
            self._update_position_in_db(position if ticker in self.positions else closed_position)
            
            logger.info(f"Closed {close_size:+.0f} shares of {ticker} at ${exit_price:.2f}, P&L: ${net_pnl:+.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close position for {ticker}: {e}")
            return False
    
    def update_positions(self, current_prices: Dict[str, float]) -> None:
        """Update all positions with current market prices.
        
        Args:
            current_prices: Current market prices by ticker
        """
        for ticker, position in self.positions.items():
            if ticker in current_prices:
                position.current_price = current_prices[ticker]
                
                # Calculate unrealized P&L
                if position.is_long:
                    position.pnl_unrealized = position.size * (position.current_price - position.entry_price)
                else:
                    position.pnl_unrealized = -position.size * (position.current_price - position.entry_price)
    
    def check_stop_loss_take_profit(self, current_prices: Dict[str, float]) -> List[str]:
        """Check for stop loss and take profit triggers.
        
        Args:
            current_prices: Current market prices
            
        Returns:
            List of tickers that should be closed
        """
        triggers = []
        
        for ticker, position in self.positions.items():
            if ticker not in current_prices:
                continue
                
            current_price = current_prices[ticker]
            
            # Check stop loss
            if position.stop_loss:
                if (position.is_long and current_price <= position.stop_loss) or \
                   (position.is_short and current_price >= position.stop_loss):
                    triggers.append(ticker)
                    logger.info(f"Stop loss triggered for {ticker}: ${current_price:.2f}")
                    continue
            
            # Check take profit
            if position.take_profit:
                if (position.is_long and current_price >= position.take_profit) or \
                   (position.is_short and current_price <= position.take_profit):
                    triggers.append(ticker)
                    logger.info(f"Take profit triggered for {ticker}: ${current_price:.2f}")
        
        return triggers
    
    def calculate_portfolio_metrics(self, current_prices: Dict[str, float]) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics.
        
        Args:
            current_prices: Current market prices
            
        Returns:
            PortfolioMetrics object
        """
        # Update positions with current prices
        self.update_positions(current_prices)
        
        # Calculate position values
        positions_value = sum(pos.market_value for pos in self.positions.values())
        unrealized_pnl = sum(pos.pnl_unrealized for pos in self.positions.values())
        realized_pnl = sum(pos.pnl_realized for pos in self.closed_positions)
        
        # Total portfolio value
        total_value = self.cash_balance + positions_value
        total_pnl = realized_pnl + unrealized_pnl
        
        # Daily return
        if len(self.daily_values) > 1:
            prev_value = self.daily_values[-1][1]
            daily_return = (total_value - prev_value) / prev_value if prev_value > 0 else 0
        else:
            daily_return = 0
        
        # Cumulative return
        cumulative_return = (total_value - self.initial_capital) / self.initial_capital
        
        # Drawdown calculations
        if total_value > self.high_water_mark:
            self.high_water_mark = total_value
        
        current_drawdown = (self.high_water_mark - total_value) / self.high_water_mark
        max_drawdown = self._calculate_max_drawdown()
        
        # Trade statistics
        closed_trades = len(self.closed_positions)
        winning_trades = len([p for p in self.closed_positions if p.pnl_realized > 0])
        losing_trades = closed_trades - winning_trades
        
        win_rate = winning_trades / closed_trades if closed_trades > 0 else 0
        
        # Average win/loss
        wins = [p.pnl_realized for p in self.closed_positions if p.pnl_realized > 0]
        losses = [p.pnl_realized for p in self.closed_positions if p.pnl_realized < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        # Average holding period
        holding_periods = [
            (p.exit_timestamp - p.entry_timestamp).days
            for p in self.closed_positions
            if p.exit_timestamp
        ]
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        # Sharpe ratio (simplified)
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Portfolio exposure
        exposure = positions_value / total_value if total_value > 0 else 0
        
        # Update daily values
        today = datetime.now().date()
        if not self.daily_values or self.daily_values[-1][0].date() != today:
            self.daily_values.append((datetime.now(), total_value))
        else:
            self.daily_values[-1] = (datetime.now(), total_value)
        
        return PortfolioMetrics(
            total_value=total_value,
            cash_balance=self.cash_balance,
            positions_value=positions_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            total_pnl=total_pnl,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=closed_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_holding_period=avg_holding_period,
            exposure=exposure
        )
    
    def get_positions_summary(self) -> pd.DataFrame:
        """Get summary of current positions.
        
        Returns:
            DataFrame with position details
        """
        if not self.positions:
            return pd.DataFrame()
        
        position_data = []
        for ticker, pos in self.positions.items():
            position_data.append({
                'ticker': ticker,
                'size': pos.size,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.pnl_unrealized,
                'unrealized_pnl_pct': pos.pnl_unrealized / pos.cost_basis if pos.cost_basis > 0 else 0,
                'entry_date': pos.entry_timestamp.date(),
                'days_held': (datetime.now() - pos.entry_timestamp).days,
                'stop_loss': pos.stop_loss,
                'take_profit': pos.take_profit
            })
        
        df = pd.DataFrame(position_data)
        return df.sort_values('unrealized_pnl_pct', ascending=False)
    
    def get_performance_summary(self, periods: List[int] = [1, 7, 30, 90]) -> pd.DataFrame:
        """Get performance summary for different periods.
        
        Args:
            periods: List of periods in days
            
        Returns:
            DataFrame with performance metrics
        """
        performance_data = []
        
        for period in periods:
            cutoff_date = datetime.now() - timedelta(days=period)
            
            # Get portfolio value at start of period
            start_values = [
                (date, value) for date, value in self.daily_values
                if date >= cutoff_date
            ]
            
            if len(start_values) >= 2:
                start_value = start_values[0][1]
                end_value = start_values[-1][1]
                period_return = (end_value - start_value) / start_value
            else:
                period_return = 0
            
            # Calculate period trades
            period_trades = [
                pos for pos in self.closed_positions
                if pos.exit_timestamp and pos.exit_timestamp >= cutoff_date
            ]
            
            period_pnl = sum(pos.pnl_realized for pos in period_trades)
            
            performance_data.append({
                'period_days': period,
                'return_pct': period_return * 100,
                'total_pnl': period_pnl,
                'num_trades': len(period_trades),
                'win_rate': len([p for p in period_trades if p.pnl_realized > 0]) / len(period_trades) if period_trades else 0
            })
        
        return pd.DataFrame(performance_data)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum historical drawdown."""
        if len(self.daily_values) < 2:
            return 0.0
        
        values = [v[1] for v in self.daily_values]
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(self.daily_values) < 30:
            return 0.0
        
        # Calculate daily returns
        values = [v[1] for v in self.daily_values]
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        
        if not returns:
            return 0.0
        
        avg_return = np.mean(returns) * 252  # Annualize
        volatility = np.std(returns) * np.sqrt(252)
        
        if volatility == 0:
            return 0.0
        
        return (avg_return - risk_free_rate) / volatility
    
    def _save_position_to_db(self, position: Position) -> None:
        """Save position to database."""
        try:
            with self.database.get_connection() as conn:
                conn.execute("""
                    INSERT INTO portfolio_positions 
                    (ticker, position_size, entry_price, entry_timestamp, position_type, 
                     stop_loss, take_profit, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    position.ticker,
                    position.size,
                    position.entry_price,
                    position.entry_timestamp,
                    'LONG' if position.is_long else 'SHORT',
                    position.stop_loss,
                    position.take_profit,
                    position.status.value
                ])
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save position to database: {e}")
    
    def _update_position_in_db(self, position: Position) -> None:
        """Update position in database."""
        try:
            with self.database.get_connection() as conn:
                conn.execute("""
                    UPDATE portfolio_positions 
                    SET exit_price = ?, exit_timestamp = ?, status = ?, pnl = ?
                    WHERE ticker = ? AND entry_timestamp = ?
                """, [
                    position.exit_price,
                    position.exit_timestamp,
                    position.status.value,
                    position.pnl_realized,
                    position.ticker,
                    position.entry_timestamp
                ])
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update position in database: {e}")
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status summary.
        
        Returns:
            Dictionary with portfolio status
        """
        return {
            'total_positions': len(self.positions),
            'cash_balance': self.cash_balance,
            'initial_capital': self.initial_capital,
            'total_trades': len(self.closed_positions),
            'positions': list(self.positions.keys()),
            'last_updated': datetime.now()
        }