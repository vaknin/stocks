"""Kelly Criterion position sizing for optimal capital allocation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from ..config.settings import config


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    ticker: str
    recommended_size: float
    kelly_fraction: float
    win_probability: float
    avg_win: float
    avg_loss: float
    edge: float
    odds: float
    confidence: float
    risk_adjusted_size: float
    max_position_limit: float


class KellyPositionSizer:
    """Kelly Criterion position sizer with safety factors and risk controls."""
    
    def __init__(
        self,
        safety_factor: float = 0.25,
        max_position_pct: float = 0.10,
        max_total_exposure: float = 0.50,
        min_confidence: float = 0.60,
        lookback_days: int = 252
    ):
        """Initialize Kelly position sizer.
        
        Args:
            safety_factor: Safety multiplier for Kelly fraction (0.25 = quarter Kelly)
            max_position_pct: Maximum position size as % of portfolio (0.10 = 10%)
            max_total_exposure: Maximum total exposure across all positions
            min_confidence: Minimum confidence threshold for position sizing
            lookback_days: Historical lookback period for win rate calculation
        """
        self.safety_factor = safety_factor
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure
        self.min_confidence = min_confidence
        self.lookback_days = lookback_days
        
        # Historical performance tracking
        self.trade_history: Dict[str, List[Dict]] = {}
        
        logger.info(f"Kelly position sizer initialized with {safety_factor}x safety factor")
    
    def calculate_position_size(
        self,
        ticker: str,
        prediction: float,
        confidence: float,
        current_price: float,
        portfolio_value: float,
        current_positions: Optional[Dict[str, float]] = None
    ) -> PositionSizeResult:
        """Calculate optimal position size using Kelly Criterion.
        
        Args:
            ticker: Stock ticker symbol
            prediction: Predicted return (e.g., 0.05 for 5%)
            confidence: Model confidence (0.0 to 1.0)
            current_price: Current stock price
            portfolio_value: Total portfolio value
            current_positions: Current position sizes as dict {ticker: position_value}
            
        Returns:
            PositionSizeResult with recommended sizing
        """
        current_positions = current_positions or {}
        
        # Calculate historical performance metrics
        win_prob, avg_win, avg_loss = self._calculate_historical_metrics(ticker)
        
        # Calculate edge and odds
        edge = win_prob - (1 - win_prob)
        
        if avg_loss == 0:
            # Avoid division by zero, use predicted values
            odds = abs(prediction) / 0.025  # Assume 2.5% stop loss
        else:
            odds = avg_win / abs(avg_loss)
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = win probability, q = lose probability
        if odds > 0:
            kelly_fraction = (odds * win_prob - (1 - win_prob)) / odds
        else:
            kelly_fraction = 0
        
        # Apply safety factor
        safe_kelly = kelly_fraction * self.safety_factor
        
        # Confidence adjustment
        confidence_adjusted = safe_kelly * confidence
        
        # Apply maximum position limits
        max_position_value = portfolio_value * self.max_position_pct
        kelly_position_value = portfolio_value * confidence_adjusted
        
        # Check total exposure constraint
        current_exposure = sum(current_positions.values()) / portfolio_value
        remaining_exposure = self.max_total_exposure - current_exposure
        
        # Final position size (as fraction of portfolio)
        recommended_fraction = min(
            confidence_adjusted,
            self.max_position_pct,
            remaining_exposure if remaining_exposure > 0 else 0
        )
        
        # Convert to actual position size
        recommended_value = portfolio_value * recommended_fraction
        recommended_shares = recommended_value / current_price if current_price > 0 else 0
        
        result = PositionSizeResult(
            ticker=ticker,
            recommended_size=recommended_shares,
            kelly_fraction=kelly_fraction,
            win_probability=win_prob,
            avg_win=avg_win,
            avg_loss=avg_loss,
            edge=edge,
            odds=odds,
            confidence=confidence,
            risk_adjusted_size=recommended_fraction,
            max_position_limit=self.max_position_pct
        )
        
        logger.debug(f"Position sizing for {ticker}: {recommended_shares:.0f} shares "
                    f"({recommended_fraction:.1%} of portfolio)")
        
        return result
    
    def batch_calculate_positions(
        self,
        predictions: Dict[str, Dict[str, Any]],
        current_prices: Dict[str, float],
        portfolio_value: float,
        current_positions: Optional[Dict[str, float]] = None
    ) -> Dict[str, PositionSizeResult]:
        """Calculate position sizes for multiple tickers.
        
        Args:
            predictions: Dictionary of predictions by ticker
            current_prices: Current prices by ticker
            portfolio_value: Total portfolio value
            current_positions: Current position values by ticker
            
        Returns:
            Dictionary mapping tickers to position size results
        """
        results = {}
        current_positions = current_positions or {}
        
        for ticker, pred_data in predictions.items():
            if ticker not in current_prices:
                logger.warning(f"No current price available for {ticker}")
                continue
            
            # Extract prediction and confidence from prediction data
            if isinstance(pred_data, dict):
                # Handle TimesFM prediction format
                best_horizon = None
                best_confidence = 0
                
                for horizon_key, horizon_data in pred_data.items():
                    if horizon_key.startswith('horizon_') and isinstance(horizon_data, dict):
                        conf = horizon_data.get('confidence', 0)
                        if conf > best_confidence:
                            best_confidence = conf
                            best_horizon = horizon_data
                
                if best_horizon:
                    prediction = best_horizon.get('prediction', 0)
                    confidence = best_horizon.get('confidence', 0)
                else:
                    prediction = pred_data.get('prediction', 0)
                    confidence = pred_data.get('confidence', 0)
            else:
                # Handle simple prediction format
                prediction = float(pred_data)
                confidence = 0.7  # Default confidence
            
            # Only size positions above minimum confidence
            if confidence >= self.min_confidence:
                result = self.calculate_position_size(
                    ticker=ticker,
                    prediction=prediction,
                    confidence=confidence,
                    current_price=current_prices[ticker],
                    portfolio_value=portfolio_value,
                    current_positions=current_positions
                )
                results[ticker] = result
            else:
                logger.debug(f"Skipping {ticker} due to low confidence: {confidence:.1%}")
        
        logger.info(f"Calculated positions for {len(results)} tickers")
        return results
    
    def _calculate_historical_metrics(self, ticker: str) -> Tuple[float, float, float]:
        """Calculate historical win rate and average win/loss for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Tuple of (win_probability, avg_win, avg_loss)
        """
        if ticker not in self.trade_history or not self.trade_history[ticker]:
            # Use default values for new tickers
            return 0.55, 0.08, 0.04  # 55% win rate, 8% avg win, 4% avg loss
        
        trades = self.trade_history[ticker]
        
        # Filter recent trades within lookback period
        cutoff_date = datetime.now() - pd.Timedelta(days=self.lookback_days)
        recent_trades = [
            trade for trade in trades 
            if trade.get('exit_date', datetime.now()) > cutoff_date
        ]
        
        if not recent_trades:
            return 0.55, 0.08, 0.04
        
        returns = [trade['return'] for trade in recent_trades if 'return' in trade]
        
        if not returns:
            return 0.55, 0.08, 0.04
        
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        win_probability = len(wins) / len(returns) if returns else 0.55
        avg_win = np.mean(wins) if wins else 0.08
        avg_loss = abs(np.mean(losses)) if losses else 0.04
        
        return win_probability, avg_win, avg_loss
    
    def update_trade_history(
        self,
        ticker: str,
        entry_date: datetime,
        exit_date: datetime,
        entry_price: float,
        exit_price: float,
        position_size: float
    ) -> None:
        """Update trade history for improving position sizing accuracy.
        
        Args:
            ticker: Stock ticker symbol
            entry_date: Trade entry timestamp
            exit_date: Trade exit timestamp
            entry_price: Entry price
            exit_price: Exit price
            position_size: Position size (shares)
        """
        if ticker not in self.trade_history:
            self.trade_history[ticker] = []
        
        trade_return = (exit_price - entry_price) / entry_price
        trade_pnl = (exit_price - entry_price) * position_size
        
        trade_record = {
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'return': trade_return,
            'pnl': trade_pnl,
            'duration_days': (exit_date - entry_date).days
        }
        
        self.trade_history[ticker].append(trade_record)
        
        # Keep only recent history to prevent memory bloat
        max_history_size = 1000
        if len(self.trade_history[ticker]) > max_history_size:
            self.trade_history[ticker] = self.trade_history[ticker][-max_history_size:]
        
        logger.debug(f"Updated trade history for {ticker}: {trade_return:.2%} return")
    
    def get_portfolio_allocation_summary(
        self,
        position_results: Dict[str, PositionSizeResult],
        current_prices: Dict[str, float],
        portfolio_value: float
    ) -> pd.DataFrame:
        """Generate portfolio allocation summary.
        
        Args:
            position_results: Position sizing results
            current_prices: Current stock prices
            portfolio_value: Total portfolio value
            
        Returns:
            DataFrame with allocation summary
        """
        summary_data = []
        
        for ticker, result in position_results.items():
            if ticker in current_prices:
                position_value = result.recommended_size * current_prices[ticker]
                position_pct = position_value / portfolio_value
                
                summary_data.append({
                    'ticker': ticker,
                    'shares': result.recommended_size,
                    'position_value': position_value,
                    'position_pct': position_pct,
                    'kelly_fraction': result.kelly_fraction,
                    'win_probability': result.win_probability,
                    'confidence': result.confidence,
                    'edge': result.edge,
                    'risk_adjusted_size': result.risk_adjusted_size
                })
        
        df = pd.DataFrame(summary_data)
        
        if not df.empty:
            # Add totals
            total_allocation = df['position_pct'].sum()
            df = df.sort_values('position_pct', ascending=False)
            
            logger.info(f"Total portfolio allocation: {total_allocation:.1%}")
        
        return df
    
    def validate_position_constraints(
        self,
        position_results: Dict[str, PositionSizeResult],
        current_positions: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, str]:
        """Validate position sizing constraints.
        
        Args:
            position_results: Position sizing results
            current_positions: Current positions
            portfolio_value: Total portfolio value
            
        Returns:
            Dictionary of constraint violations
        """
        violations = {}
        
        # Check individual position limits
        for ticker, result in position_results.items():
            position_pct = result.risk_adjusted_size
            
            if position_pct > self.max_position_pct:
                violations[f"{ticker}_individual"] = (
                    f"Position {position_pct:.1%} exceeds individual limit "
                    f"{self.max_position_pct:.1%}"
                )
        
        # Check total exposure
        current_exposure = sum(current_positions.values()) / portfolio_value
        new_exposure = sum(r.risk_adjusted_size for r in position_results.values())
        total_exposure = current_exposure + new_exposure
        
        if total_exposure > self.max_total_exposure:
            violations['total_exposure'] = (
                f"Total exposure {total_exposure:.1%} exceeds limit "
                f"{self.max_total_exposure:.1%}"
            )
        
        # Check minimum confidence
        low_confidence = [
            ticker for ticker, result in position_results.items()
            if result.confidence < self.min_confidence
        ]
        
        if low_confidence:
            violations['low_confidence'] = (
                f"Low confidence positions: {low_confidence}"
            )
        
        return violations
    
    def get_sizing_metrics(self) -> Dict[str, Any]:
        """Get position sizing configuration and metrics.
        
        Returns:
            Dictionary with sizing parameters and statistics
        """
        total_trades = sum(len(trades) for trades in self.trade_history.values())
        
        # Calculate overall win rate across all tickers
        all_returns = []
        for trades in self.trade_history.values():
            all_returns.extend([t['return'] for t in trades if 'return' in t])
        
        overall_win_rate = (
            len([r for r in all_returns if r > 0]) / len(all_returns)
            if all_returns else 0
        )
        
        return {
            'safety_factor': self.safety_factor,
            'max_position_pct': self.max_position_pct,
            'max_total_exposure': self.max_total_exposure,
            'min_confidence': self.min_confidence,
            'lookback_days': self.lookback_days,
            'total_trades_tracked': total_trades,
            'overall_win_rate': overall_win_rate,
            'tickers_tracked': len(self.trade_history)
        }