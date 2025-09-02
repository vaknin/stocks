#!/usr/bin/env python3
"""
Paper Trading Engine
Simulated trading environment for strategy validation and testing.
"""

import sys
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config
from trading.portfolio_tracker import PortfolioTracker
from trading.enhanced_risk_manager import EnhancedRiskManager
from analysis.market_analyzer import DailyMarketAnalyzer

warnings.filterwarnings('ignore')


class OrderType(Enum):
    """Order types for paper trading."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status states."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class PaperOrder:
    """Paper trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    created_at: datetime = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    filled_at: Optional[datetime] = None
    commission: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class PaperTrade:
    """Executed paper trade."""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    
    @property
    def value(self) -> float:
        """Total trade value including commission."""
        return abs(self.quantity * self.price) + self.commission


@dataclass
class MarketSimulation:
    """Market simulation parameters."""
    bid_ask_spread: float = 0.001  # 0.1% spread
    slippage_factor: float = 0.0005  # 0.05% slippage
    commission_per_share: float = 0.005  # $0.005 per share
    min_commission: float = 1.0  # $1 minimum commission
    market_impact_threshold: float = 10000.0  # $10k impact threshold
    volatility_multiplier: float = 1.0  # Volatility scaling
    liquidity_constraints: bool = True  # Enable liquidity simulation


class PaperTradingEngine:
    """Paper trading engine for strategy validation."""
    
    def __init__(
        self,
        starting_capital: float = 10000.0,
        simulation_params: Optional[MarketSimulation] = None,
        enable_realistic_fills: bool = True,
        save_to_file: bool = True
    ):
        """Initialize paper trading engine.
        
        Args:
            starting_capital: Starting portfolio value
            simulation_params: Market simulation parameters
            enable_realistic_fills: Enable realistic order fills
            save_to_file: Save trades to file
        """
        self.starting_capital = starting_capital
        self.simulation_params = simulation_params or MarketSimulation()
        self.enable_realistic_fills = enable_realistic_fills
        self.save_to_file = save_to_file
        
        # Trading state
        self.cash_balance = starting_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.pending_orders: Dict[str, PaperOrder] = {}
        self.order_history: List[PaperOrder] = []
        self.trade_history: List[PaperTrade] = []
        
        # Components
        self.market_analyzer = DailyMarketAnalyzer()
        self.risk_manager = EnhancedRiskManager()
        
        # Performance tracking
        self.portfolio_history: List[Dict[str, Any]] = []
        self.daily_pnl_history: List[float] = []
        
        # File storage
        self.data_file = Path("paper_trading_data.json")
        
        # Load existing data if available
        if self.save_to_file and self.data_file.exists():
            self._load_data()
        
        logger.info(f"Paper trading engine initialized with ${starting_capital:,.2f}")
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        force_order: bool = False
    ) -> Tuple[str, bool, List[str]]:
        """Place a paper trading order.
        
        Args:
            symbol: Stock symbol
            side: Buy or sell
            quantity: Number of shares
            order_type: Type of order
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            force_order: Skip risk checks
            
        Returns:
            Tuple of (order_id, success, error_messages)
        """
        order_id = str(uuid.uuid4())[:8]
        errors = []
        
        # Validate inputs
        if quantity <= 0:
            errors.append("Quantity must be positive")
        
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
            errors.append("Limit orders require a price")
        
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None:
            errors.append("Stop orders require a stop price")
        
        if errors:
            return order_id, False, errors
        
        # Get current market price
        current_price = self._get_current_price(symbol)
        if current_price is None:
            errors.append(f"Unable to get market price for {symbol}")
            return order_id, False, errors
        
        # Risk management checks
        if not force_order:
            portfolio_value = self.get_portfolio_value()
            
            # For buy orders, check if we have enough cash
            if side == OrderSide.BUY:
                estimated_cost = quantity * current_price
                estimated_commission = self._calculate_commission(quantity, current_price)
                total_cost = estimated_cost + estimated_commission
                
                if total_cost > self.cash_balance:
                    errors.append(f"Insufficient cash: need ${total_cost:,.2f}, have ${self.cash_balance:,.2f}")
                    return order_id, False, errors
                
                # Risk manager pre-trade check
                approved, violations, adjusted_size = self.risk_manager.check_enhanced_pre_trade_risk(
                    symbol, quantity, current_price, portfolio_value, self.positions
                )
                
                if not approved:
                    errors.extend(violations)
                    return order_id, False, errors
                
                # Use adjusted size if provided
                if adjusted_size is not None:
                    quantity = adjusted_size
                    logger.info(f"Position size adjusted by risk manager: {adjusted_size:.2f} shares")
            
            # For sell orders, check if we have enough shares
            elif side == OrderSide.SELL:
                current_shares = self.positions.get(symbol, {}).get('shares', 0)
                if quantity > current_shares:
                    errors.append(f"Insufficient shares: trying to sell {quantity}, have {current_shares}")
                    return order_id, False, errors
        
        # Create order
        order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        # Add to pending orders
        self.pending_orders[order_id] = order
        
        # Try immediate execution for market orders
        if order_type == OrderType.MARKET:
            self._try_fill_order(order_id)
        
        # Save data
        if self.save_to_file:
            self._save_data()
        
        logger.info(f"Order placed: {side.value} {quantity} {symbol} @ {order_type.value}")
        
        return order_id, True, []
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successfully cancelled
        """
        if order_id not in self.pending_orders:
            return False
        
        order = self.pending_orders[order_id]
        order.status = OrderStatus.CANCELLED
        
        # Move to history
        self.order_history.append(order)
        del self.pending_orders[order_id]
        
        # Save data
        if self.save_to_file:
            self._save_data()
        
        logger.info(f"Order cancelled: {order_id}")
        return True
    
    def update_market_prices(self) -> int:
        """Update market prices and process pending orders with real-time validation.
        
        Returns:
            Number of orders processed
        """
        processed = 0
        orders_to_remove = []
        
        for order_id, order in self.pending_orders.items():
            if self._try_fill_order(order_id):
                orders_to_remove.append(order_id)
                processed += 1
        
        # Remove filled orders
        for order_id in orders_to_remove:
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
        
        # Update portfolio history
        self._update_portfolio_history()
        
        # Perform real-time strategy validation
        self._validate_strategy_performance()
        
        # Save data
        if self.save_to_file:
            self._save_data()
        
        return processed
    
    def _validate_strategy_performance(self):
        """Real-time strategy validation for revenue optimization."""
        try:
            if len(self.portfolio_history) < 5:  # Need minimum history
                return
            
            current_value = self.get_portfolio_value()
            recent_values = [h['portfolio_value'] for h in self.portfolio_history[-5:]]
            
            # Calculate short-term performance metrics
            returns = [(recent_values[i] - recent_values[i-1]) / recent_values[i-1] 
                      for i in range(1, len(recent_values))]
            
            avg_return = np.mean(returns)
            volatility = np.std(returns)
            
            # Real-time risk assessment
            if volatility > 0.05:  # >5% daily volatility
                logger.warning(f"High strategy volatility detected: {volatility:.1%}")
            
            # Performance tracking against benchmarks
            total_return = (current_value - self.starting_capital) / self.starting_capital
            
            # Log performance milestones
            if total_return >= 0.025:  # 2.5% return milestone
                logger.info(f"🎯 Strategy milestone achieved: {total_return:.1%} return")
            
            # Validate position sizes against Kelly criterion
            self._validate_kelly_sizing()
            
        except Exception as e:
            logger.error(f"Strategy validation error: {e}")
    
    def _validate_kelly_sizing(self):
        """Validate current position sizes against Kelly optimal sizing."""
        try:
            portfolio_value = self.get_portfolio_value()
            
            for symbol, position in self.positions.items():
                current_weight = (position['shares'] * position['current_price']) / portfolio_value
                
                # Get recent price data for Kelly calculation
                market_data = self.market_analyzer.fetch_market_data([symbol], lookback_days=30)
                
                if symbol in market_data and len(market_data[symbol]) > 10:
                    prices = market_data[symbol]['close'].values
                    returns = np.diff(prices) / prices[:-1]
                    
                    if len(returns) > 5:
                        mean_return = np.mean(returns)
                        variance = np.var(returns)
                        
                        if variance > 0 and mean_return > 0:
                            # Kelly fraction calculation
                            kelly_fraction = mean_return / variance
                            kelly_weight = min(kelly_fraction * 0.4, 0.15)  # Safety factor + max weight
                            
                            weight_deviation = abs(current_weight - kelly_weight)
                            
                            if weight_deviation > 0.03:  # 3% deviation threshold
                                logger.info(f"Kelly rebalancing opportunity for {symbol}: "
                                          f"Current {current_weight:.1%}, Optimal {kelly_weight:.1%}")
                        
        except Exception as e:
            logger.warning(f"Kelly sizing validation error: {e}")
    
    def run_forward_test(
        self,
        strategy_func: Callable,
        test_days: int = 30,
        update_interval_minutes: int = 60
    ) -> Dict[str, Any]:
        """Run forward testing of a strategy function.
        
        Args:
            strategy_func: Strategy function that takes (engine, market_data) and returns trade decisions
            test_days: Number of days to test
            update_interval_minutes: How often to update (in minutes)
            
        Returns:
            Forward test results
        """
        logger.info(f"Starting {test_days}-day forward test")
        
        start_time = datetime.now()
        start_value = self.get_portfolio_value()
        
        # Test results tracking
        decisions_made = 0
        successful_trades = 0
        failed_trades = 0
        
        # Simulate trading over test period
        for day in range(test_days):
            try:
                # Get fresh market data
                symbols = list(config.STOCK_TICKERS[:10])  # Test with subset
                market_data = self.market_analyzer.fetch_market_data(symbols, lookback_days=5)
                
                # Execute strategy
                decisions = strategy_func(self, market_data)
                decisions_made += len(decisions)
                
                # Process strategy decisions
                for decision in decisions:
                    symbol = decision.get('symbol')
                    action = decision.get('action')  # 'buy', 'sell', 'hold'
                    quantity = decision.get('quantity', 0)
                    
                    if action == 'buy' and quantity > 0:
                        order_id, success, errors = self.place_order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            order_type=OrderType.MARKET
                        )
                        if success:
                            successful_trades += 1
                        else:
                            failed_trades += 1
                            logger.warning(f"Forward test buy failed for {symbol}: {errors}")
                    
                    elif action == 'sell' and quantity > 0:
                        order_id, success, errors = self.place_order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            quantity=quantity,
                            order_type=OrderType.MARKET
                        )
                        if success:
                            successful_trades += 1
                        else:
                            failed_trades += 1
                            logger.warning(f"Forward test sell failed for {symbol}: {errors}")
                
                # Process market updates
                self.update_market_prices()
                
                # Log daily progress
                current_value = self.get_portfolio_value()
                daily_return = (current_value - start_value) / start_value
                logger.info(f"Forward test day {day + 1}: Portfolio ${current_value:,.2f} ({daily_return:+.1%})")
                
            except Exception as e:
                logger.error(f"Forward test error on day {day + 1}: {e}")
                continue
        
        # Calculate final results
        end_time = datetime.now()
        end_value = self.get_portfolio_value()
        total_return = (end_value - start_value) / start_value
        
        # Generate comprehensive test report
        performance_report = self.get_performance_report()
        
        forward_test_results = {
            'test_period_days': test_days,
            'start_time': start_time,
            'end_time': end_time,
            'start_value': start_value,
            'end_value': end_value,
            'total_return_pct': total_return * 100,
            'decisions_made': decisions_made,
            'successful_trades': successful_trades,
            'failed_trades': failed_trades,
            'success_rate_pct': (successful_trades / max(1, successful_trades + failed_trades)) * 100,
            'performance_metrics': performance_report,
            'risk_adjusted_return': total_return / max(0.01, performance_report.get('annualized_volatility_pct', 1) / 100),
            'forward_test_sharpe': performance_report.get('sharpe_ratio', 0),
            'max_drawdown_pct': performance_report.get('max_drawdown_pct', 0)
        }
        
        logger.info(f"Forward test complete: {total_return:+.1%} return, "
                   f"{forward_test_results['success_rate_pct']:.1f}% success rate")
        
        return forward_test_results
    
    def _try_fill_order(self, order_id: str) -> bool:
        """Attempt to fill an order.
        
        Args:
            order_id: Order ID to fill
            
        Returns:
            True if order was filled
        """
        if order_id not in self.pending_orders:
            return False
        
        order = self.pending_orders[order_id]
        current_price = self._get_current_price(order.symbol)
        
        if current_price is None:
            return False
        
        # Determine if order should be filled
        should_fill = False
        fill_price = current_price
        
        if order.order_type == OrderType.MARKET:
            should_fill = True
            fill_price = self._apply_market_effects(order.symbol, current_price, order.quantity, order.side)
        
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and current_price <= order.price:
                should_fill = True
                fill_price = min(order.price, current_price)
            elif order.side == OrderSide.SELL and current_price >= order.price:
                should_fill = True
                fill_price = max(order.price, current_price)
        
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY and current_price >= order.stop_price:
                should_fill = True
                fill_price = self._apply_market_effects(order.symbol, current_price, order.quantity, order.side)
            elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                should_fill = True
                fill_price = self._apply_market_effects(order.symbol, current_price, order.quantity, order.side)
        
        elif order.order_type == OrderType.STOP_LIMIT:
            stop_triggered = False
            if order.side == OrderSide.BUY and current_price >= order.stop_price:
                stop_triggered = True
            elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                stop_triggered = True
            
            if stop_triggered:
                # Convert to limit order behavior
                if order.side == OrderSide.BUY and current_price <= order.price:
                    should_fill = True
                    fill_price = min(order.price, current_price)
                elif order.side == OrderSide.SELL and current_price >= order.price:
                    should_fill = True
                    fill_price = max(order.price, current_price)
        
        if should_fill:
            return self._execute_order(order, fill_price)
        
        return False
    
    def _execute_order(self, order: PaperOrder, fill_price: float) -> bool:
        """Execute an order at the specified price.
        
        Args:
            order: Order to execute
            fill_price: Execution price
            
        Returns:
            True if successfully executed
        """
        commission = self._calculate_commission(order.quantity, fill_price)
        
        # Check if we still have sufficient cash for buy orders
        if order.side == OrderSide.BUY:
            total_cost = (order.quantity * fill_price) + commission
            if total_cost > self.cash_balance:
                # Reject order
                order.status = OrderStatus.REJECTED
                self.order_history.append(order)
                logger.warning(f"Order rejected: insufficient cash for {order.symbol}")
                return False
        
        # Execute the trade
        trade_id = str(uuid.uuid4())[:8]
        trade = PaperTrade(
            trade_id=trade_id,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=datetime.now(),
            commission=commission
        )
        
        # Update positions and cash
        if order.side == OrderSide.BUY:
            self._add_position(order.symbol, order.quantity, fill_price, commission)
            self.cash_balance -= (order.quantity * fill_price) + commission
        else:
            self._reduce_position(order.symbol, order.quantity, fill_price, commission)
            self.cash_balance += (order.quantity * fill_price) - commission
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.filled_at = datetime.now()
        order.commission = commission
        
        # Record trade and order
        self.trade_history.append(trade)
        self.order_history.append(order)
        
        logger.info(f"Trade executed: {order.side.value} {order.quantity} {order.symbol} @ ${fill_price:.2f}")
        
        return True
    
    def _add_position(self, symbol: str, quantity: float, price: float, commission: float):
        """Add to a position."""
        if symbol in self.positions:
            # Update existing position
            current = self.positions[symbol]
            total_shares = current['shares'] + quantity
            total_cost = (current['shares'] * current['avg_cost']) + (quantity * price) + commission
            avg_cost = total_cost / total_shares
            
            self.positions[symbol] = {
                'shares': total_shares,
                'avg_cost': avg_cost,
                'current_price': price,
                'last_updated': datetime.now()
            }
        else:
            # New position
            self.positions[symbol] = {
                'shares': quantity,
                'avg_cost': (quantity * price + commission) / quantity,
                'current_price': price,
                'last_updated': datetime.now()
            }
    
    def _reduce_position(self, symbol: str, quantity: float, price: float, commission: float):
        """Reduce a position."""
        if symbol not in self.positions:
            return
        
        current = self.positions[symbol]
        new_shares = current['shares'] - quantity
        
        if new_shares <= 0:
            # Close position
            del self.positions[symbol]
        else:
            # Reduce position
            self.positions[symbol] = {
                'shares': new_shares,
                'avg_cost': current['avg_cost'],  # Keep original avg cost
                'current_price': price,
                'last_updated': datetime.now()
            }
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol."""
        try:
            # Try to get real market data
            market_data = self.market_analyzer.fetch_market_data([symbol], lookback_days=2)
            if symbol in market_data and not market_data[symbol].empty:
                return float(market_data[symbol]['close'].iloc[-1])
            
            # Fallback to synthetic price if no real data
            return self._generate_synthetic_price(symbol)
        
        except Exception as e:
            logger.warning(f"Error getting price for {symbol}: {e}")
            return self._generate_synthetic_price(symbol)
    
    def _generate_synthetic_price(self, symbol: str) -> float:
        """Generate synthetic price for testing."""
        # Use symbol hash for consistent but random-looking prices
        symbol_hash = hash(symbol) % 10000
        base_price = 50 + (symbol_hash / 100)
        
        # Add time-based variation
        time_factor = datetime.now().timestamp() % 86400  # Seconds in day
        variation = np.sin(time_factor / 3600) * 0.05  # 5% daily variation
        
        return base_price * (1 + variation)
    
    def _apply_market_effects(
        self,
        symbol: str,
        base_price: float,
        quantity: float,
        side: OrderSide
    ) -> float:
        """Apply realistic market effects to execution price."""
        if not self.enable_realistic_fills:
            return base_price
        
        price = base_price
        
        # Apply bid-ask spread
        if side == OrderSide.BUY:
            price = base_price * (1 + self.simulation_params.bid_ask_spread / 2)
        else:
            price = base_price * (1 - self.simulation_params.bid_ask_spread / 2)
        
        # Apply slippage for larger orders
        order_value = quantity * base_price
        if order_value > self.simulation_params.market_impact_threshold:
            impact_factor = min(0.01, order_value / 100000 * 0.001)  # Max 1% impact
            if side == OrderSide.BUY:
                price += base_price * impact_factor
            else:
                price -= base_price * impact_factor
        
        # Apply random slippage
        random_slippage = np.random.normal(0, self.simulation_params.slippage_factor)
        if side == OrderSide.BUY:
            price += base_price * abs(random_slippage)
        else:
            price -= base_price * abs(random_slippage)
        
        return max(0.01, price)  # Ensure positive price
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate trading commission."""
        commission = quantity * self.simulation_params.commission_per_share
        return max(commission, self.simulation_params.min_commission)
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        total_value = self.cash_balance
        
        for symbol, position in self.positions.items():
            current_price = self._get_current_price(symbol)
            if current_price is not None:
                position_value = position['shares'] * current_price
                total_value += position_value
        
        return total_value
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        portfolio_value = self.get_portfolio_value()
        total_pnl = portfolio_value - self.starting_capital
        total_return_pct = (total_pnl / self.starting_capital) * 100
        
        # Calculate position details
        position_details = []
        total_position_value = 0
        
        for symbol, position in self.positions.items():
            current_price = self._get_current_price(symbol)
            if current_price is None:
                continue
            
            shares = position['shares']
            avg_cost = position['avg_cost']
            position_value = shares * current_price
            cost_basis = shares * avg_cost
            unrealized_pnl = position_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0
            
            total_position_value += position_value
            
            position_details.append({
                'symbol': symbol,
                'shares': shares,
                'avg_cost': avg_cost,
                'current_price': current_price,
                'position_value': position_value,
                'cost_basis': cost_basis,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct,
                'weight_pct': (position_value / portfolio_value) * 100 if portfolio_value > 0 else 0
            })
        
        # Calculate trading statistics
        total_trades = len(self.trade_history)
        total_commission = sum(trade.commission for trade in self.trade_history)
        
        return {
            'timestamp': datetime.now(),
            'starting_capital': self.starting_capital,
            'cash_balance': self.cash_balance,
            'total_position_value': total_position_value,
            'portfolio_value': portfolio_value,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'cash_pct': (self.cash_balance / portfolio_value) * 100 if portfolio_value > 0 else 100,
            'positions_count': len(self.positions),
            'positions': position_details,
            'pending_orders': len(self.pending_orders),
            'total_trades': total_trades,
            'total_commission': total_commission
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance analysis report."""
        if len(self.portfolio_history) < 2:
            return {'error': 'Insufficient historical data'}
        
        # Calculate daily returns
        portfolio_values = [h['portfolio_value'] for h in self.portfolio_history]
        daily_returns = [
            (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            for i in range(1, len(portfolio_values))
        ]
        
        # Performance metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = (np.mean(daily_returns) * 252 - 0.02) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak_value = portfolio_values[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak_value:
                peak_value = value
            else:
                drawdown = (peak_value - value) / peak_value
                max_drawdown = max(max_drawdown, drawdown)
        
        # Win rate
        winning_days = sum(1 for r in daily_returns if r > 0)
        win_rate = (winning_days / len(daily_returns)) * 100 if daily_returns else 0
        
        # Best and worst days
        best_day = max(daily_returns) * 100 if daily_returns else 0
        worst_day = min(daily_returns) * 100 if daily_returns else 0
        
        return {
            'period_days': len(self.portfolio_history) - 1,
            'total_return_pct': total_return * 100,
            'annualized_volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate_pct': win_rate,
            'best_day_pct': best_day,
            'worst_day_pct': worst_day,
            'average_daily_return_pct': np.mean(daily_returns) * 100 if daily_returns else 0,
            'total_trades': len(self.trade_history),
            'trades_per_day': len(self.trade_history) / max(1, len(self.portfolio_history) - 1)
        }
    
    def _update_portfolio_history(self):
        """Update portfolio value history."""
        summary = self.get_portfolio_summary()
        history_entry = {
            'timestamp': summary['timestamp'],
            'portfolio_value': summary['portfolio_value'],
            'cash_balance': summary['cash_balance'],
            'position_value': summary['total_position_value'],
            'positions_count': summary['positions_count']
        }
        
        self.portfolio_history.append(history_entry)
        
        # Keep only last 252 days (1 trading year)
        if len(self.portfolio_history) > 252:
            self.portfolio_history = self.portfolio_history[-252:]
    
    def _save_data(self):
        """Save trading data to file."""
        try:
            data = {
                'starting_capital': self.starting_capital,
                'cash_balance': self.cash_balance,
                'positions': {
                    symbol: {**pos, 'last_updated': pos['last_updated'].isoformat()}
                    for symbol, pos in self.positions.items()
                },
                'pending_orders': [
                    {**asdict(order), 'created_at': order.created_at.isoformat(),
                     'side': order.side.value, 'order_type': order.order_type.value, 'status': order.status.value}
                    for order in self.pending_orders.values()
                ],
                'order_history': [
                    {**asdict(order), 'created_at': order.created_at.isoformat(),
                     'side': order.side.value, 'order_type': order.order_type.value, 'status': order.status.value,
                     'filled_at': order.filled_at.isoformat() if order.filled_at else None}
                    for order in self.order_history
                ],
                'trade_history': [
                    {**asdict(trade), 'side': trade.side.value, 'timestamp': trade.timestamp.isoformat()}
                    for trade in self.trade_history
                ],
                'portfolio_history': [
                    {**entry, 'timestamp': entry['timestamp'].isoformat()}
                    for entry in self.portfolio_history
                ]
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving paper trading data: {e}")
    
    def _load_data(self):
        """Load trading data from file."""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            
            self.starting_capital = data.get('starting_capital', self.starting_capital)
            self.cash_balance = data.get('cash_balance', self.cash_balance)
            
            # Load positions
            for symbol, pos_data in data.get('positions', {}).items():
                pos_data['last_updated'] = datetime.fromisoformat(pos_data['last_updated'])
                self.positions[symbol] = pos_data
            
            # Load order history
            for order_data in data.get('order_history', []):
                order = PaperOrder(
                    order_id=order_data['order_id'],
                    symbol=order_data['symbol'],
                    side=OrderSide(order_data['side']),
                    order_type=OrderType(order_data['order_type']),
                    quantity=order_data['quantity'],
                    price=order_data.get('price'),
                    stop_price=order_data.get('stop_price'),
                    created_at=datetime.fromisoformat(order_data['created_at']),
                    status=OrderStatus(order_data['status']),
                    filled_quantity=order_data.get('filled_quantity', 0),
                    filled_price=order_data.get('filled_price'),
                    filled_at=datetime.fromisoformat(order_data['filled_at']) if order_data.get('filled_at') else None,
                    commission=order_data.get('commission', 0)
                )
                self.order_history.append(order)
            
            # Load trade history
            for trade_data in data.get('trade_history', []):
                trade = PaperTrade(
                    trade_id=trade_data['trade_id'],
                    order_id=trade_data['order_id'],
                    symbol=trade_data['symbol'],
                    side=OrderSide(trade_data['side']),
                    quantity=trade_data['quantity'],
                    price=trade_data['price'],
                    timestamp=datetime.fromisoformat(trade_data['timestamp']),
                    commission=trade_data['commission']
                )
                self.trade_history.append(trade)
            
            # Load portfolio history
            for entry in data.get('portfolio_history', []):
                entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
                self.portfolio_history.append(entry)
            
            logger.info("Paper trading data loaded successfully")
            
        except Exception as e:
            logger.warning(f"Error loading paper trading data: {e}")
    
    def reset(self):
        """Reset paper trading engine to initial state."""
        self.cash_balance = self.starting_capital
        self.positions.clear()
        self.pending_orders.clear()
        self.order_history.clear()
        self.trade_history.clear()
        self.portfolio_history.clear()
        self.daily_pnl_history.clear()
        
        if self.data_file.exists():
            self.data_file.unlink()
        
        logger.info("Paper trading engine reset to initial state")


def main():
    """Demo and testing of paper trading engine."""
    engine = PaperTradingEngine(starting_capital=10000.0)
    
    print("Paper Trading Engine Demo")
    print("=" * 30)
    
    # Place some test orders
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in symbols:
        order_id, success, errors = engine.place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        
        if success:
            print(f"✅ Order placed for {symbol}: {order_id}")
        else:
            print(f"❌ Order failed for {symbol}: {errors}")
    
    # Update market and process orders
    processed = engine.update_market_prices()
    print(f"\n📊 Processed {processed} orders")
    
    # Show portfolio summary
    summary = engine.get_portfolio_summary()
    print(f"\n💰 Portfolio Value: ${summary['portfolio_value']:,.2f}")
    print(f"💵 Cash Balance: ${summary['cash_balance']:,.2f}")
    print(f"📈 Total Return: {summary['total_return_pct']:+.2f}%")
    print(f"📍 Positions: {summary['positions_count']}")
    
    # Show positions
    if summary['positions']:
        print("\n📋 Current Positions:")
        for pos in summary['positions']:
            print(f"  {pos['symbol']}: {pos['shares']} shares @ ${pos['current_price']:.2f} "
                  f"({pos['unrealized_pnl_pct']:+.1f}%)")
    
    # Performance report
    if len(engine.portfolio_history) > 1:
        perf = engine.get_performance_report()
        print(f"\n📈 Performance Report:")
        print(f"  Total Return: {perf['total_return_pct']:+.2f}%")
        print(f"  Volatility: {perf['annualized_volatility_pct']:.1f}%")
        print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {perf['max_drawdown_pct']:.1f}%")
        print(f"  Win Rate: {perf['win_rate_pct']:.1f}%")


if __name__ == "__main__":
    main()