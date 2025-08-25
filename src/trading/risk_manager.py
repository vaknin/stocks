"""Risk management system for trading operations."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

from ..config.settings import config


class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio or individual positions."""
    current_drawdown: float
    max_drawdown: float
    var_95: float  # Value at Risk (95% confidence)
    var_99: float  # Value at Risk (99% confidence)
    sharpe_ratio: float
    beta: float
    correlation_to_market: float
    concentration_risk: float  # Largest position as % of portfolio
    risk_level: RiskLevel
    risk_score: float  # 0-100 scale


@dataclass
class RiskAlert:
    """Risk alert for violations or warnings."""
    timestamp: datetime
    alert_type: str
    severity: RiskLevel
    ticker: Optional[str]
    message: str
    current_value: float
    threshold: float
    action_required: str


class RiskManager:
    """Comprehensive risk management system."""
    
    def __init__(
        self,
        max_drawdown: float = 0.10,
        stop_loss_pct: float = 0.025,
        max_position_pct: float = 0.10,
        max_sector_exposure: float = 0.30,
        max_correlation: float = 0.70,
        var_confidence: float = 0.95,
        lookback_days: int = 252
    ):
        """Initialize risk manager.
        
        Args:
            max_drawdown: Maximum portfolio drawdown allowed (0.10 = 10%)
            stop_loss_pct: Individual position stop loss (0.025 = 2.5%)
            max_position_pct: Maximum single position size (0.10 = 10%)
            max_sector_exposure: Maximum sector concentration (0.30 = 30%)
            max_correlation: Maximum correlation between positions
            var_confidence: VaR confidence level (0.95 = 95%)
            lookback_days: Historical lookback for risk calculations
        """
        self.max_drawdown = max_drawdown
        self.stop_loss_pct = stop_loss_pct
        self.max_position_pct = max_position_pct
        self.max_sector_exposure = max_sector_exposure
        self.max_correlation = max_correlation
        self.var_confidence = var_confidence
        self.lookback_days = lookback_days
        
        # Risk tracking
        self.risk_alerts: List[RiskAlert] = []
        self.portfolio_high_water_mark = 0.0
        self.daily_returns: List[float] = []
        self.position_history: Dict[str, List[Dict]] = {}
        
        logger.info(f"Risk manager initialized with {max_drawdown:.1%} max drawdown")
    
    def check_pre_trade_risk(
        self,
        ticker: str,
        proposed_size: float,
        current_price: float,
        portfolio_value: float,
        current_positions: Dict[str, Dict[str, Any]],
        sector_map: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, List[str]]:
        """Check if proposed trade meets risk requirements.
        
        Args:
            ticker: Stock ticker symbol
            proposed_size: Proposed position size (shares)
            current_price: Current stock price
            portfolio_value: Total portfolio value
            current_positions: Current positions with metadata
            sector_map: Mapping of tickers to sectors
            
        Returns:
            Tuple of (approved, list_of_violations)
        """
        violations = []
        
        # Position size check
        position_value = proposed_size * current_price
        position_pct = position_value / portfolio_value
        
        if position_pct > self.max_position_pct:
            violations.append(
                f"Position size {position_pct:.1%} exceeds limit {self.max_position_pct:.1%}"
            )
        
        # Sector concentration check
        if sector_map and ticker in sector_map:
            sector = sector_map[ticker]
            current_sector_exposure = self._calculate_sector_exposure(
                current_positions, sector_map, sector
            )
            new_sector_exposure = (current_sector_exposure + position_value) / portfolio_value
            
            if new_sector_exposure > self.max_sector_exposure:
                violations.append(
                    f"Sector exposure {new_sector_exposure:.1%} exceeds limit "
                    f"{self.max_sector_exposure:.1%} for {sector}"
                )
        
        # Correlation check (simplified - would need full implementation)
        if len(current_positions) > 0:
            estimated_correlation = self._estimate_position_correlation(ticker, current_positions)
            if estimated_correlation > self.max_correlation:
                violations.append(
                    f"Estimated correlation {estimated_correlation:.2f} exceeds limit "
                    f"{self.max_correlation:.2f}"
                )
        
        # Portfolio drawdown check
        current_drawdown = self.calculate_current_drawdown(portfolio_value)
        if current_drawdown > self.max_drawdown * 0.8:  # Warning at 80% of limit
            violations.append(
                f"Portfolio drawdown {current_drawdown:.1%} approaching limit "
                f"{self.max_drawdown:.1%}"
            )
        
        approved = len(violations) == 0
        
        if not approved:
            logger.warning(f"Trade rejected for {ticker}: {violations}")
        
        return approved, violations
    
    def check_stop_loss_triggers(
        self,
        current_positions: Dict[str, Dict[str, Any]],
        current_prices: Dict[str, float]
    ) -> List[str]:
        """Check for stop-loss triggers in current positions.
        
        Args:
            current_positions: Current positions with entry data
            current_prices: Current market prices
            
        Returns:
            List of tickers that should be closed due to stop-loss
        """
        stop_loss_triggers = []
        
        for ticker, position_data in current_positions.items():
            if ticker not in current_prices:
                continue
            
            current_price = current_prices[ticker]
            entry_price = position_data.get('entry_price', 0)
            position_size = position_data.get('size', 0)
            
            if entry_price <= 0 or position_size == 0:
                continue
            
            # Calculate unrealized P&L percentage
            if position_size > 0:  # Long position
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # Short position
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Check for stop loss trigger
            stop_loss_price = position_data.get('stop_loss', None)
            triggered = False
            
            if stop_loss_price is not None:
                # Price-based stop loss
                if position_size > 0:  # Long position
                    triggered = current_price <= stop_loss_price
                else:  # Short position
                    triggered = current_price >= stop_loss_price
            else:
                # Percentage-based stop loss (default)
                triggered = pnl_pct <= -self.stop_loss_pct
            
            if triggered:
                stop_loss_triggers.append(ticker)
                
                alert = RiskAlert(
                    timestamp=datetime.now(),
                    alert_type="STOP_LOSS_TRIGGER",
                    severity=RiskLevel.HIGH,
                    ticker=ticker,
                    message=f"Stop loss triggered for {ticker}",
                    current_value=current_price,
                    threshold=stop_loss_price if stop_loss_price is not None else -self.stop_loss_pct,
                    action_required="CLOSE_POSITION"
                )
                
                self.risk_alerts.append(alert)
                logger.warning(f"Stop loss triggered for {ticker}: {pnl_pct:.2%}")
        
        return stop_loss_triggers
    
    def calculate_portfolio_risk_metrics(
        self,
        current_positions: Dict[str, Dict[str, Any]],
        current_prices: Dict[str, float],
        portfolio_value: float,
        historical_returns: Optional[pd.DataFrame] = None
    ) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics.
        
        Args:
            current_positions: Current portfolio positions
            current_prices: Current market prices
            portfolio_value: Total portfolio value
            historical_returns: Historical return data for VaR calculation
            
        Returns:
            RiskMetrics object with portfolio risk assessment
        """
        # Current drawdown
        current_drawdown = self.calculate_current_drawdown(portfolio_value)
        max_drawdown = self._calculate_max_historical_drawdown()
        
        # Concentration risk (largest position)
        concentration_risk = 0.0
        if current_positions:
            position_values = []
            for ticker, pos_data in current_positions.items():
                if ticker in current_prices:
                    pos_value = abs(pos_data.get('size', 0)) * current_prices[ticker]
                    position_values.append(pos_value)
            
            if position_values:
                concentration_risk = max(position_values) / portfolio_value
        
        # VaR calculations
        var_95, var_99 = self._calculate_var(historical_returns)
        
        # Sharpe ratio (simplified)
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Beta and correlation (simplified)
        beta = 1.0  # Default market beta
        correlation_to_market = 0.5  # Default correlation
        
        # Overall risk score (0-100)
        risk_score = self._calculate_risk_score(
            current_drawdown, concentration_risk, var_95
        )
        
        # Risk level classification
        risk_level = self._classify_risk_level(risk_score)
        
        return RiskMetrics(
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            var_95=var_95,
            var_99=var_99,
            sharpe_ratio=sharpe_ratio,
            beta=beta,
            correlation_to_market=correlation_to_market,
            concentration_risk=concentration_risk,
            risk_level=risk_level,
            risk_score=risk_score
        )
    
    def calculate_current_drawdown(self, current_portfolio_value: float) -> float:
        """Calculate current portfolio drawdown from high water mark.
        
        Args:
            current_portfolio_value: Current portfolio value
            
        Returns:
            Current drawdown as decimal (0.1 = 10% drawdown)
        """
        if current_portfolio_value > self.portfolio_high_water_mark:
            self.portfolio_high_water_mark = current_portfolio_value
        
        if self.portfolio_high_water_mark == 0:
            return 0.0
        
        drawdown = (self.portfolio_high_water_mark - current_portfolio_value) / self.portfolio_high_water_mark
        return max(0.0, drawdown)
    
    def update_daily_returns(self, daily_return: float) -> None:
        """Update daily return history for risk calculations.
        
        Args:
            daily_return: Daily portfolio return as decimal
        """
        self.daily_returns.append(daily_return)
        
        # Keep only lookback period
        if len(self.daily_returns) > self.lookback_days:
            self.daily_returns = self.daily_returns[-self.lookback_days:]
    
    def get_active_risk_alerts(
        self,
        max_age_hours: int = 24
    ) -> List[RiskAlert]:
        """Get recent risk alerts.
        
        Args:
            max_age_hours: Maximum age of alerts to return
            
        Returns:
            List of recent risk alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        return [
            alert for alert in self.risk_alerts
            if alert.timestamp > cutoff_time
        ]
    
    def _calculate_sector_exposure(
        self,
        positions: Dict[str, Dict[str, Any]],
        sector_map: Dict[str, str],
        target_sector: str
    ) -> float:
        """Calculate current exposure to a specific sector.
        
        Args:
            positions: Current positions
            sector_map: Ticker to sector mapping
            target_sector: Sector to calculate exposure for
            
        Returns:
            Total value in target sector
        """
        sector_value = 0.0
        
        for ticker, pos_data in positions.items():
            if sector_map.get(ticker) == target_sector:
                # Would need current prices to calculate exact value
                # This is a simplified implementation
                sector_value += abs(pos_data.get('market_value', 0))
        
        return sector_value
    
    def _estimate_position_correlation(
        self,
        new_ticker: str,
        current_positions: Dict[str, Dict[str, Any]]
    ) -> float:
        """Estimate correlation of new position with existing portfolio.
        
        Args:
            new_ticker: New ticker to add
            current_positions: Existing positions
            
        Returns:
            Estimated correlation (simplified implementation)
        """
        # Simplified correlation estimation
        # In production, this would use historical price correlations
        
        # Tech stocks typically have higher correlation
        tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'INTC', 'QCOM']
        
        if new_ticker in tech_tickers:
            tech_positions = [
                ticker for ticker in current_positions.keys()
                if ticker in tech_tickers
            ]
            
            if tech_positions:
                return 0.6  # High tech correlation
        
        return 0.3  # Default moderate correlation
    
    def _calculate_var(
        self,
        historical_returns: Optional[pd.DataFrame] = None
    ) -> Tuple[float, float]:
        """Calculate Value at Risk at 95% and 99% confidence levels.
        
        Args:
            historical_returns: Historical return data
            
        Returns:
            Tuple of (VaR_95%, VaR_99%)
        """
        if historical_returns is None or self.daily_returns:
            # Use portfolio daily returns
            returns = np.array(self.daily_returns) if self.daily_returns else np.array([0])
        else:
            # Use provided historical returns
            returns = historical_returns.values.flatten()
        
        if len(returns) < 30:  # Insufficient data
            return 0.05, 0.08  # Default VaR estimates
        
        var_95 = np.percentile(returns, 5)  # 5th percentile for 95% VaR
        var_99 = np.percentile(returns, 1)  # 1st percentile for 99% VaR
        
        return abs(var_95), abs(var_99)
    
    def _calculate_max_historical_drawdown(self) -> float:
        """Calculate maximum historical drawdown."""
        if len(self.daily_returns) < 2:
            return 0.0
        
        # Convert daily returns to cumulative returns
        cumulative = np.cumprod(1 + np.array(self.daily_returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        
        return np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio of portfolio.
        
        Args:
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if len(self.daily_returns) < 30:
            return 0.0
        
        returns = np.array(self.daily_returns)
        annual_return = np.mean(returns) * 252  # Annualize
        annual_volatility = np.std(returns) * np.sqrt(252)
        
        if annual_volatility == 0:
            return 0.0
        
        return (annual_return - risk_free_rate) / annual_volatility
    
    def _calculate_risk_score(
        self,
        drawdown: float,
        concentration: float,
        var_95: float
    ) -> float:
        """Calculate overall risk score (0-100).
        
        Args:
            drawdown: Current drawdown
            concentration: Concentration risk
            var_95: 95% VaR
            
        Returns:
            Risk score from 0 (low risk) to 100 (high risk)
        """
        # Weighted risk score
        drawdown_score = min(drawdown / self.max_drawdown * 40, 40)  # Max 40 points
        concentration_score = min(concentration / 0.5 * 30, 30)      # Max 30 points
        var_score = min(var_95 / 0.1 * 30, 30)                      # Max 30 points
        
        return drawdown_score + concentration_score + var_score
    
    def _classify_risk_level(self, risk_score: float) -> RiskLevel:
        """Classify risk level based on risk score.
        
        Args:
            risk_score: Risk score (0-100)
            
        Returns:
            Risk level classification
        """
        if risk_score < 25:
            return RiskLevel.LOW
        elif risk_score < 50:
            return RiskLevel.MEDIUM
        elif risk_score < 75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def generate_risk_report(
        self,
        current_positions: Dict[str, Dict[str, Any]],
        current_prices: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, Any]:
        """Generate comprehensive risk report.
        
        Args:
            current_positions: Current portfolio positions
            current_prices: Current market prices
            portfolio_value: Total portfolio value
            
        Returns:
            Dictionary with risk report data
        """
        risk_metrics = self.calculate_portfolio_risk_metrics(
            current_positions, current_prices, portfolio_value
        )
        
        active_alerts = self.get_active_risk_alerts()
        stop_loss_triggers = self.check_stop_loss_triggers(current_positions, current_prices)
        
        return {
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_value,
            'risk_metrics': risk_metrics,
            'active_alerts': len(active_alerts),
            'stop_loss_triggers': len(stop_loss_triggers),
            'risk_parameters': {
                'max_drawdown': self.max_drawdown,
                'stop_loss_pct': self.stop_loss_pct,
                'max_position_pct': self.max_position_pct,
                'max_sector_exposure': self.max_sector_exposure
            },
            'recommendations': self._generate_risk_recommendations(risk_metrics)
        }
    
    def _generate_risk_recommendations(self, risk_metrics: RiskMetrics) -> List[str]:
        """Generate risk management recommendations.
        
        Args:
            risk_metrics: Current risk metrics
            
        Returns:
            List of risk management recommendations
        """
        recommendations = []
        
        if risk_metrics.risk_level == RiskLevel.CRITICAL:
            recommendations.append("CRITICAL: Reduce position sizes immediately")
            recommendations.append("Consider closing highest-risk positions")
        
        if risk_metrics.current_drawdown > self.max_drawdown * 0.8:
            recommendations.append("Approaching maximum drawdown - reduce exposure")
        
        if risk_metrics.concentration_risk > 0.15:
            recommendations.append("High concentration risk - diversify positions")
        
        if risk_metrics.var_95 > 0.08:
            recommendations.append("High VaR - consider reducing portfolio leverage")
        
        if not recommendations:
            recommendations.append("Risk levels within acceptable parameters")
        
        return recommendations