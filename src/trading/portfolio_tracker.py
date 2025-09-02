"""
TOML-based Portfolio Tracking System for AI Trading Advisor
Manages portfolio state, positions, performance metrics, and settings.
"""

import toml
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class Position:
    """Individual stock position with entry details and targets."""
    symbol: str
    shares: float
    avg_cost: float
    current_price: float
    current_value: float
    entry_date: str
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L for this position."""
        return self.current_value - (self.shares * self.avg_cost)
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        cost_basis = self.shares * self.avg_cost
        if cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / cost_basis) * 100


@dataclass
class Performance:
    """Portfolio performance metrics."""
    total_return_pct: float = 0.0
    monthly_return_pct: float = 0.0
    win_rate_pct: float = 0.0
    trades_won: int = 0
    trades_total: int = 0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0


@dataclass
class Settings:
    """Portfolio management settings from README.md."""
    max_positions: int = 5
    max_position_size_pct: int = 15  # Max 15% per stock (optimized)
    kelly_multiplier: float = 0.40   # 0.40x Kelly criterion (2024 research)
    confidence_threshold: int = 78   # >78% confidence requirement (ML-optimized)
    target_stocks: List[str] = None
    
    def __post_init__(self):
        if self.target_stocks is None:
            # 10 semiconductor stocks from README.md
            self.target_stocks = [
                "NVDA", "AMD", "ASML", "TSM", "INTC", 
                "QCOM", "AVGO", "MU", "SMCI", "ARM"
            ]


class PortfolioTracker:
    """TOML-based portfolio state management."""
    
    def __init__(self, portfolio_file: str = "portfolio.toml"):
        self.portfolio_file = Path(portfolio_file)
        self.positions: Dict[str, Position] = {}
        self.cash_reserve: float = 0.0
        self.total_value: float = 0.0
        self.last_updated: str = ""
        self.performance = Performance()
        self.settings = Settings()
        
        logger.info(f"Initialized PortfolioTracker with file: {self.portfolio_file}")
        
        # Load existing portfolio if file exists
        if self.portfolio_file.exists():
            self.load_portfolio()
        else:
            # Create initial portfolio with default settings
            self._create_initial_portfolio()
    
    def _create_initial_portfolio(self):
        """Create initial portfolio structure."""
        logger.info("Creating initial portfolio structure")
        self.cash_reserve = 1000.0  # Start with $1k as per README
        self.total_value = self.cash_reserve
        self.last_updated = datetime.now().strftime("%Y-%m-%d")
        self.save_portfolio()
    
    def load_portfolio(self) -> bool:
        """Load portfolio state from TOML file."""
        try:
            logger.info(f"Loading portfolio from {self.portfolio_file}")
            data = toml.load(self.portfolio_file)
            
            # Load basic portfolio info
            portfolio_info = data.get("portfolio", {})
            self.total_value = portfolio_info.get("total_value", 0.0)
            self.cash_reserve = portfolio_info.get("cash_reserve", 0.0)
            self.last_updated = portfolio_info.get("last_updated", "")
            
            # Load positions
            positions_data = data.get("positions", {})
            self.positions = {}
            for symbol, pos_data in positions_data.items():
                self.positions[symbol] = Position(
                    symbol=symbol,
                    shares=pos_data["shares"],
                    avg_cost=pos_data["avg_cost"],
                    current_price=pos_data.get("current_price", pos_data["avg_cost"]),
                    current_value=pos_data.get("current_value", pos_data["shares"] * pos_data["avg_cost"]),
                    entry_date=pos_data["entry_date"],
                    stop_loss=pos_data.get("stop_loss"),
                    target_price=pos_data.get("target_price")
                )
            
            # Load performance metrics
            perf_data = data.get("performance", {})
            self.performance = Performance(
                total_return_pct=perf_data.get("total_return_pct", 0.0),
                monthly_return_pct=perf_data.get("monthly_return_pct", 0.0),
                win_rate_pct=perf_data.get("win_rate_pct", 0.0),
                trades_won=perf_data.get("trades_won", 0),
                trades_total=perf_data.get("trades_total", 0),
                sharpe_ratio=perf_data.get("sharpe_ratio", 0.0),
                max_drawdown_pct=perf_data.get("max_drawdown_pct", 0.0)
            )
            
            # Load settings
            settings_data = data.get("settings", {})
            self.settings = Settings(
                max_positions=settings_data.get("max_positions", 5),
                max_position_size_pct=settings_data.get("max_position_size_pct", 20),
                kelly_multiplier=settings_data.get("kelly_multiplier", 0.40),
                confidence_threshold=settings_data.get("confidence_threshold", 75),
                target_stocks=settings_data.get("target_stocks", self.settings.target_stocks)
            )
            
            logger.info(f"Loaded portfolio: {len(self.positions)} positions, ${self.total_value:.2f} total value")
            return True
            
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            return False
    
    def save_portfolio(self) -> bool:
        """Save current portfolio state to TOML file."""
        try:
            logger.info(f"Saving portfolio to {self.portfolio_file}")
            
            # Prepare data structure for TOML
            data = {
                "portfolio": {
                    "total_value": round(self.total_value, 2),
                    "last_updated": datetime.now().strftime("%Y-%m-%d"),
                    "cash_reserve": round(self.cash_reserve, 2)
                },
                "positions": {},
                "performance": asdict(self.performance),
                "settings": asdict(self.settings)
            }
            
            # Add positions data
            for symbol, position in self.positions.items():
                pos_dict = asdict(position)
                # Remove symbol from dict as it's the key
                pos_dict.pop("symbol")
                # Round financial values
                for key in ["avg_cost", "current_price", "current_value", "stop_loss", "target_price"]:
                    if pos_dict.get(key) is not None:
                        pos_dict[key] = round(pos_dict[key], 2)
                pos_dict["shares"] = round(pos_dict["shares"], 4)
                data["positions"][symbol] = pos_dict
            
            # Write to TOML file
            with open(self.portfolio_file, 'w') as f:
                toml.dump(data, f)
            
            logger.info("Portfolio saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
            return False
    
    def add_position(self, symbol: str, shares: float, avg_cost: float, 
                    stop_loss: Optional[float] = None, target_price: Optional[float] = None) -> bool:
        """Add a new position to the portfolio."""
        try:
            if symbol in self.positions:
                logger.warning(f"Position {symbol} already exists, updating instead")
                return self.update_position(symbol, shares, avg_cost, stop_loss, target_price)
            
            # Check position limit
            if len(self.positions) >= self.settings.max_positions:
                logger.error(f"Cannot add position: max positions ({self.settings.max_positions}) reached")
                return False
            
            # Calculate position value
            position_value = shares * avg_cost
            
            # Check if we have enough cash
            if position_value > self.cash_reserve:
                logger.error(f"Insufficient cash: need ${position_value:.2f}, have ${self.cash_reserve:.2f}")
                return False
            
            # Check position size limit (20% max per README)
            position_pct = (position_value / self.total_value) * 100
            if position_pct > self.settings.max_position_size_pct:
                logger.error(f"Position too large: {position_pct:.1f}% > {self.settings.max_position_size_pct}%")
                return False
            
            # Create position
            position = Position(
                symbol=symbol,
                shares=shares,
                avg_cost=avg_cost,
                current_price=avg_cost,  # Will be updated with market data
                current_value=position_value,
                entry_date=datetime.now().strftime("%Y-%m-%d"),
                stop_loss=stop_loss,
                target_price=target_price
            )
            
            # Update portfolio
            self.positions[symbol] = position
            self.cash_reserve -= position_value
            
            logger.info(f"Added position: {symbol} {shares} shares @ ${avg_cost:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding position {symbol}: {e}")
            return False
    
    def remove_position(self, symbol: str, shares: Optional[float] = None) -> bool:
        """Remove position (full or partial)."""
        try:
            if symbol not in self.positions:
                logger.error(f"Position {symbol} not found")
                return False
            
            position = self.positions[symbol]
            
            if shares is None or shares >= position.shares:
                # Full position close
                self.cash_reserve += position.current_value
                self.performance.trades_total += 1
                
                # Check if it's a winning trade
                if position.unrealized_pnl > 0:
                    self.performance.trades_won += 1
                
                del self.positions[symbol]
                logger.info(f"Closed full position: {symbol}")
            else:
                # Partial position close
                if shares <= 0 or shares > position.shares:
                    logger.error(f"Invalid shares amount: {shares}")
                    return False
                
                sale_value = shares * position.current_price
                self.cash_reserve += sale_value
                
                position.shares -= shares
                position.current_value = position.shares * position.current_price
                
                logger.info(f"Partial close: {symbol} sold {shares} shares")
            
            # Update win rate
            if self.performance.trades_total > 0:
                self.performance.win_rate_pct = (self.performance.trades_won / self.performance.trades_total) * 100
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing position {symbol}: {e}")
            return False
    
    def update_position(self, symbol: str, shares: Optional[float] = None, 
                       avg_cost: Optional[float] = None, stop_loss: Optional[float] = None,
                       target_price: Optional[float] = None) -> bool:
        """Update existing position parameters."""
        try:
            if symbol not in self.positions:
                logger.error(f"Position {symbol} not found")
                return False
            
            position = self.positions[symbol]
            
            if shares is not None:
                position.shares = shares
            if avg_cost is not None:
                position.avg_cost = avg_cost
            if stop_loss is not None:
                position.stop_loss = stop_loss
            if target_price is not None:
                position.target_price = target_price
            
            # Recalculate current value
            position.current_value = position.shares * position.current_price
            
            logger.info(f"Updated position: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating position {symbol}: {e}")
            return False
    
    def update_market_prices(self, prices: Dict[str, float]) -> bool:
        """Update current market prices for all positions."""
        try:
            updated_count = 0
            total_value = self.cash_reserve
            
            for symbol, position in self.positions.items():
                if symbol in prices:
                    position.current_price = prices[symbol]
                    position.current_value = position.shares * position.current_price
                    updated_count += 1
                
                total_value += position.current_value
            
            self.total_value = total_value
            
            logger.info(f"Updated prices for {updated_count} positions, total value: ${self.total_value:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating market prices: {e}")
            return False
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary."""
        try:
            # Calculate portfolio metrics
            total_cost_basis = sum(pos.shares * pos.avg_cost for pos in self.positions.values())
            total_market_value = sum(pos.current_value for pos in self.positions.values())
            total_unrealized_pnl = total_market_value - total_cost_basis
            
            if total_cost_basis > 0:
                total_return_pct = (total_unrealized_pnl / total_cost_basis) * 100
            else:
                total_return_pct = 0.0
            
            # Position allocation percentages
            position_allocations = {}
            for symbol, position in self.positions.items():
                allocation_pct = (position.current_value / self.total_value) * 100
                position_allocations[symbol] = allocation_pct
            
            summary = {
                "total_value": self.total_value,
                "cash_reserve": self.cash_reserve,
                "invested_value": total_market_value,
                "total_cost_basis": total_cost_basis,
                "unrealized_pnl": total_unrealized_pnl,
                "total_return_pct": total_return_pct,
                "position_count": len(self.positions),
                "cash_allocation_pct": (self.cash_reserve / self.total_value) * 100,
                "position_allocations": position_allocations,
                "last_updated": self.last_updated,
                "performance": asdict(self.performance),
                "positions": [asdict(pos) for pos in self.positions.values()]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return {}
    
    def print_portfolio_status(self):
        """Print human-readable portfolio status."""
        summary = self.get_portfolio_summary()
        if not summary:
            print("Error: Unable to generate portfolio summary")
            return
        
        print("=" * 50)
        print("PORTFOLIO STATUS")
        print("=" * 50)
        print(f"Date: {summary['last_updated']}")
        print(f"Total Value: ${summary['total_value']:,.2f}")
        print(f"Cash Reserve: ${summary['cash_reserve']:,.2f} ({summary['cash_allocation_pct']:.1f}%)")
        print(f"Invested: ${summary['invested_value']:,.2f}")
        print(f"Unrealized P&L: ${summary['unrealized_pnl']:,.2f} ({summary['total_return_pct']:+.2f}%)")
        print()
        
        if self.positions:
            print("POSITIONS:")
            print("-" * 50)
            for symbol, position in self.positions.items():
                allocation_pct = summary['position_allocations'][symbol]
                pnl_pct = position.unrealized_pnl_pct
                print(f"{symbol:>6}: {position.shares:>8.2f} shares @ ${position.avg_cost:>7.2f} = ${position.current_value:>8,.2f} "
                      f"({allocation_pct:>5.1f}%) P&L: {pnl_pct:+6.2f}%")
        
        print()
        print("PERFORMANCE:")
        print("-" * 50)
        print(f"Win Rate: {summary['performance']['trades_won']}/{summary['performance']['trades_total']} "
              f"({summary['performance']['win_rate_pct']:.1f}%)")
        print(f"Monthly Return: {summary['performance']['monthly_return_pct']:+.2f}%")
        print(f"Sharpe Ratio: {summary['performance']['sharpe_ratio']:.2f}")
        print("=" * 50)


if __name__ == "__main__":
    # Test the portfolio tracker
    tracker = PortfolioTracker("test_portfolio.toml")
    
    # Test adding positions
    tracker.add_position("NVDA", 4.5, 888.89, stop_loss=850.0, target_price=950.0)
    tracker.add_position("AMD", 15.0, 133.33, stop_loss=125.0, target_price=160.0)
    
    # Update with current market prices
    tracker.update_market_prices({"NVDA": 892.50, "AMD": 142.50})
    
    # Save portfolio
    tracker.save_portfolio()
    
    # Print status
    tracker.print_portfolio_status()