#!/usr/bin/env python3
"""
Enhanced Risk Management System
Advanced risk controls with sector limits, circuit breakers, and volatility-based sizing.
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config
from trading.risk_manager import RiskManager, RiskLevel, RiskMetrics, RiskAlert


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    COOLING_DOWN = "cooling_down"
    DISABLED = "disabled"


@dataclass
class SectorLimits:
    """Sector-specific risk limits."""
    name: str
    max_exposure_pct: float
    max_correlation: float
    volatility_multiplier: float
    beta_limit: float = 2.0


@dataclass
class CircuitBreakerEvent:
    """Circuit breaker trigger event."""
    timestamp: datetime
    trigger_type: str
    threshold_value: float
    current_value: float
    action_taken: str
    recovery_time: datetime


class EnhancedRiskManager(RiskManager):
    """Enhanced risk management with advanced controls."""
    
    def __init__(
        self,
        max_drawdown: float = 0.18,
        stop_loss_pct: float = 0.025,
        max_position_pct: float = 0.15,
        max_sector_exposure: float = 0.25,  # Enhanced: Reduced from 30% to 25%
        max_correlation: float = 0.65,      # Enhanced: Reduced from 70% to 65%
        var_confidence: float = 0.95,
        lookback_days: int = 252,
        daily_loss_limit: float = 0.05,     # Enhanced: Daily loss circuit breaker
        intraday_loss_limit: float = 0.03,  # Enhanced: Intraday loss limit
        volatility_threshold: float = 0.30   # Enhanced: Volatility-based position sizing
    ):
        """Initialize enhanced risk manager."""
        super().__init__(
            max_drawdown=max_drawdown,
            stop_loss_pct=stop_loss_pct,
            max_position_pct=max_position_pct,
            max_sector_exposure=max_sector_exposure,
            max_correlation=max_correlation,
            var_confidence=var_confidence,
            lookback_days=lookback_days
        )
        
        # Enhanced risk controls
        self.daily_loss_limit = daily_loss_limit
        self.intraday_loss_limit = intraday_loss_limit
        self.volatility_threshold = volatility_threshold
        
        # Circuit breaker state
        self.circuit_breaker_state = CircuitBreakerState.ACTIVE
        self.trading_halted = False
        self.halt_reason = ""
        self.halt_timestamp = None
        self.recovery_time = None
        
        # Circuit breaker events history
        self.circuit_breaker_events: List[CircuitBreakerEvent] = []
        
        # Enhanced sector limits
        self.sector_limits = self._initialize_sector_limits()
        
        # Volatility tracking for dynamic sizing
        self.intraday_returns: List[float] = []
        self.portfolio_start_value = 0.0
        self.session_start_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        
        logger.info("Enhanced risk manager initialized with advanced controls")
    
    def _initialize_sector_limits(self) -> Dict[str, SectorLimits]:
        """Initialize sector-specific risk limits."""
        return {
            'Technology': SectorLimits(
                name='Technology',
                max_exposure_pct=0.35,  # Allow higher tech exposure
                max_correlation=0.75,   # Tech stocks can be more correlated
                volatility_multiplier=1.2,  # Higher volatility tolerance
                beta_limit=2.5
            ),
            'Healthcare': SectorLimits(
                name='Healthcare',
                max_exposure_pct=0.25,
                max_correlation=0.60,
                volatility_multiplier=1.0,
                beta_limit=1.5
            ),
            'Financials': SectorLimits(
                name='Financials',
                max_exposure_pct=0.25,
                max_correlation=0.70,
                volatility_multiplier=1.1,
                beta_limit=2.0
            ),
            'Consumer Discretionary': SectorLimits(
                name='Consumer Discretionary',
                max_exposure_pct=0.20,
                max_correlation=0.65,
                volatility_multiplier=1.0,
                beta_limit=1.8
            ),
            'Energy': SectorLimits(
                name='Energy',
                max_exposure_pct=0.15,  # Lower energy exposure
                max_correlation=0.80,   # Energy stocks highly correlated
                volatility_multiplier=1.3,  # Higher volatility
                beta_limit=2.0
            ),
            'Consumer Staples': SectorLimits(
                name='Consumer Staples',
                max_exposure_pct=0.15,
                max_correlation=0.50,
                volatility_multiplier=0.8,  # Lower volatility
                beta_limit=1.2
            ),
            'Utilities': SectorLimits(
                name='Utilities',
                max_exposure_pct=0.10,
                max_correlation=0.45,
                volatility_multiplier=0.7,  # Lowest volatility
                beta_limit=1.0
            ),
            'Industrial': SectorLimits(
                name='Industrial',
                max_exposure_pct=0.15,
                max_correlation=0.60,
                volatility_multiplier=0.9,
                beta_limit=1.5
            )
        }
    
    def check_circuit_breakers(
        self,
        current_portfolio_value: float,
        session_start_value: Optional[float] = None
    ) -> bool:
        """Check and trigger circuit breakers if necessary.
        
        Args:
            current_portfolio_value: Current portfolio value
            session_start_value: Portfolio value at session start
            
        Returns:
            True if trading should continue, False if halted
        """
        if self.circuit_breaker_state == CircuitBreakerState.DISABLED:
            return True
        
        now = datetime.now()
        
        # Check if cooling down period has passed
        if (self.circuit_breaker_state == CircuitBreakerState.COOLING_DOWN and 
            self.recovery_time and now >= self.recovery_time):
            self._reset_circuit_breaker()
        
        # If already triggered, don't allow trading
        if self.circuit_breaker_state == CircuitBreakerState.TRIGGERED:
            return False
        
        # Initialize session start value if not provided
        if session_start_value is None:
            if self.portfolio_start_value == 0:
                self.portfolio_start_value = current_portfolio_value
            session_start_value = self.portfolio_start_value
        
        # Check daily loss limit
        if session_start_value > 0:
            daily_loss_pct = (session_start_value - current_portfolio_value) / session_start_value
            
            if daily_loss_pct > self.daily_loss_limit:
                self._trigger_circuit_breaker(
                    "DAILY_LOSS_LIMIT",
                    self.daily_loss_limit,
                    daily_loss_pct,
                    "Trading halted due to daily loss limit",
                    recovery_minutes=60  # 1 hour recovery
                )
                return False
        
        # Check intraday loss limit (more frequent checks)
        if len(self.intraday_returns) >= 3:  # Need some history
            recent_returns = self.intraday_returns[-6:]  # Last 6 updates
            cumulative_intraday_loss = sum(r for r in recent_returns if r < 0)
            
            if abs(cumulative_intraday_loss) > self.intraday_loss_limit:
                self._trigger_circuit_breaker(
                    "INTRADAY_LOSS_LIMIT",
                    self.intraday_loss_limit,
                    abs(cumulative_intraday_loss),
                    "Trading halted due to intraday loss accumulation",
                    recovery_minutes=30  # 30 minute recovery
                )
                return False
        
        # Check maximum drawdown circuit breaker
        current_drawdown = self.calculate_current_drawdown(current_portfolio_value)
        if current_drawdown > self.max_drawdown:
            self._trigger_circuit_breaker(
                "MAX_DRAWDOWN",
                self.max_drawdown,
                current_drawdown,
                "Trading halted due to maximum drawdown breach",
                recovery_minutes=120  # 2 hour recovery
            )
            return False
        
        return True
    
    def _trigger_circuit_breaker(
        self,
        trigger_type: str,
        threshold: float,
        current_value: float,
        action_description: str,
        recovery_minutes: int = 60
    ):
        """Trigger circuit breaker and halt trading."""
        now = datetime.now()
        self.circuit_breaker_state = CircuitBreakerState.TRIGGERED
        self.trading_halted = True
        self.halt_reason = action_description
        self.halt_timestamp = now
        self.recovery_time = now + timedelta(minutes=recovery_minutes)
        
        # Record event
        event = CircuitBreakerEvent(
            timestamp=now,
            trigger_type=trigger_type,
            threshold_value=threshold,
            current_value=current_value,
            action_taken=action_description,
            recovery_time=self.recovery_time
        )
        self.circuit_breaker_events.append(event)
        
        # Create risk alert
        alert = RiskAlert(
            timestamp=now,
            alert_type="CIRCUIT_BREAKER",
            severity=RiskLevel.CRITICAL,
            ticker=None,
            message=action_description,
            current_value=current_value,
            threshold=threshold,
            action_required="HALT_TRADING"
        )
        self.risk_alerts.append(alert)
        
        logger.critical(f"Circuit breaker triggered: {trigger_type} - {action_description}")
        logger.info(f"Trading will resume at {self.recovery_time}")
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker to active state."""
        self.circuit_breaker_state = CircuitBreakerState.ACTIVE
        self.trading_halted = False
        self.halt_reason = ""
        self.halt_timestamp = None
        self.recovery_time = None
        
        logger.info("Circuit breaker reset - trading resumed")
    
    def check_enhanced_pre_trade_risk(
        self,
        ticker: str,
        proposed_size: float,
        current_price: float,
        portfolio_value: float,
        current_positions: Dict[str, Dict[str, Any]],
        sector_map: Optional[Dict[str, str]] = None,
        volatility_data: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, List[str], Optional[float]]:
        """Enhanced pre-trade risk check with sector limits and volatility adjustment.
        
        Returns:
            Tuple of (approved, violations, adjusted_position_size)
        """
        # First run base risk checks
        base_approved, base_violations = self.check_pre_trade_risk(
            ticker, proposed_size, current_price, portfolio_value, current_positions, sector_map
        )
        
        violations = base_violations.copy()
        
        # Check circuit breaker status
        if self.trading_halted:
            violations.append(f"Trading halted: {self.halt_reason}")
            return False, violations, None
        
        # Enhanced sector-specific checks
        if sector_map and ticker in sector_map:
            sector = sector_map[ticker]
            sector_limits = self.sector_limits.get(sector)
            
            if sector_limits:
                # Check sector-specific exposure limit
                current_sector_exposure = self._calculate_sector_exposure(
                    current_positions, sector_map, sector
                )
                position_value = proposed_size * current_price
                new_sector_exposure = (current_sector_exposure + position_value) / portfolio_value
                
                if new_sector_exposure > sector_limits.max_exposure_pct:
                    violations.append(
                        f"Sector {sector} exposure {new_sector_exposure:.1%} exceeds limit "
                        f"{sector_limits.max_exposure_pct:.1%}"
                    )
                
                # Check sector-specific correlation limit
                estimated_correlation = self._estimate_enhanced_correlation(
                    ticker, current_positions, sector
                )
                if estimated_correlation > sector_limits.max_correlation:
                    violations.append(
                        f"Sector {sector} correlation {estimated_correlation:.2f} exceeds limit "
                        f"{sector_limits.max_correlation:.2f}"
                    )
        
        # Volatility-based position size adjustment
        adjusted_size = proposed_size
        if volatility_data and ticker in volatility_data:
            stock_volatility = volatility_data[ticker]
            adjusted_size = self._adjust_position_for_volatility(
                proposed_size, stock_volatility, portfolio_value
            )
            
            if adjusted_size < proposed_size * 0.5:  # More than 50% reduction
                violations.append(
                    f"High volatility ({stock_volatility:.1%}) requires significant position reduction"
                )
        
        # Portfolio volatility check
        portfolio_volatility = self._estimate_portfolio_volatility(current_positions, volatility_data)
        if portfolio_volatility > self.volatility_threshold:
            violations.append(
                f"Portfolio volatility {portfolio_volatility:.1%} exceeds threshold "
                f"{self.volatility_threshold:.1%}"
            )
        
        approved = len(violations) == 0
        
        return approved, violations, adjusted_size if adjusted_size != proposed_size else None
    
    def _adjust_position_for_volatility(
        self,
        proposed_size: float,
        stock_volatility: float,
        portfolio_value: float
    ) -> float:
        """Adjust position size based on stock volatility."""
        # Base volatility assumption (20% annualized)
        base_volatility = 0.20
        
        # Calculate volatility adjustment factor
        if stock_volatility <= base_volatility:
            volatility_multiplier = 1.0
        else:
            # Reduce position size for higher volatility
            volatility_multiplier = base_volatility / stock_volatility
            volatility_multiplier = max(0.3, volatility_multiplier)  # Minimum 30% of original
        
        adjusted_size = proposed_size * volatility_multiplier
        
        logger.debug(f"Volatility adjustment: {stock_volatility:.1%} -> {volatility_multiplier:.2f}x")
        
        return adjusted_size
    
    def _estimate_enhanced_correlation(
        self,
        new_ticker: str,
        current_positions: Dict[str, Dict[str, Any]],
        sector: str
    ) -> float:
        """Enhanced correlation estimation with sector-specific logic."""
        sector_correlations = {
            'Technology': 0.75,
            'Healthcare': 0.50,
            'Financials': 0.65,
            'Consumer Discretionary': 0.55,
            'Energy': 0.80,
            'Consumer Staples': 0.40,
            'Utilities': 0.35,
            'Industrial': 0.50
        }
        
        base_correlation = sector_correlations.get(sector, 0.5)
        
        # Increase correlation if multiple positions in same sector
        same_sector_positions = sum(
            1 for ticker in current_positions.keys()
            if self._get_ticker_sector(ticker) == sector
        )
        
        if same_sector_positions > 0:
            base_correlation += min(0.2, same_sector_positions * 0.05)
        
        return min(1.0, base_correlation)
    
    def _estimate_portfolio_volatility(
        self,
        current_positions: Dict[str, Dict[str, Any]],
        volatility_data: Optional[Dict[str, float]] = None
    ) -> float:
        """Estimate overall portfolio volatility."""
        if not current_positions or not volatility_data:
            return 0.20  # Default 20% volatility
        
        total_value = sum(
            abs(pos.get('market_value', 0)) for pos in current_positions.values()
        )
        
        if total_value == 0:
            return 0.20
        
        weighted_volatility = 0.0
        for ticker, position in current_positions.items():
            if ticker in volatility_data:
                weight = abs(position.get('market_value', 0)) / total_value
                stock_vol = volatility_data[ticker]
                weighted_volatility += weight * stock_vol
        
        return weighted_volatility
    
    def _get_ticker_sector(self, ticker: str) -> str:
        """Get sector for ticker (simplified mapping)."""
        sector_mapping = {
            # Technology
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'META': 'Technology',
            'NVDA': 'Technology', 'AMD': 'Technology', 'INTC': 'Technology', 'QCOM': 'Technology',
            'ASML': 'Technology', 'TSM': 'Technology', 'AVGO': 'Technology', 'MU': 'Technology',
            
            # Healthcare
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
            
            # Financials
            'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
            
            # Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy'
        }
        
        return sector_mapping.get(ticker, 'Other')
    
    def update_intraday_return(self, return_pct: float):
        """Update intraday return for circuit breaker monitoring."""
        self.intraday_returns.append(return_pct)
        
        # Keep only last 2 hours of data (assuming 5-minute updates = 24 points)
        if len(self.intraday_returns) > 24:
            self.intraday_returns = self.intraday_returns[-24:]
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            'state': self.circuit_breaker_state.value,
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason,
            'halt_timestamp': self.halt_timestamp.isoformat() if self.halt_timestamp else None,
            'recovery_time': self.recovery_time.isoformat() if self.recovery_time else None,
            'events_count': len(self.circuit_breaker_events),
            'daily_loss_limit': self.daily_loss_limit,
            'intraday_loss_limit': self.intraday_loss_limit
        }
    
    def get_sector_utilization(
        self,
        current_positions: Dict[str, Dict[str, Any]],
        portfolio_value: float
    ) -> Dict[str, Dict[str, float]]:
        """Get current sector utilization vs limits."""
        sector_utilization = {}
        
        for sector_name, limits in self.sector_limits.items():
            current_exposure = 0.0
            
            # Calculate current exposure
            for ticker, position in current_positions.items():
                if self._get_ticker_sector(ticker) == sector_name:
                    current_exposure += abs(position.get('market_value', 0))
            
            exposure_pct = (current_exposure / portfolio_value) if portfolio_value > 0 else 0
            
            sector_utilization[sector_name] = {
                'current_exposure_pct': exposure_pct,
                'max_exposure_pct': limits.max_exposure_pct,
                'utilization_pct': (exposure_pct / limits.max_exposure_pct) * 100,
                'available_capacity': max(0, limits.max_exposure_pct - exposure_pct),
                'max_correlation': limits.max_correlation,
                'volatility_multiplier': limits.volatility_multiplier
            }
        
        return sector_utilization
    
    def generate_enhanced_risk_report(
        self,
        current_positions: Dict[str, Dict[str, Any]],
        current_prices: Dict[str, float],
        portfolio_value: float,
        volatility_data: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Generate enhanced risk report with circuit breaker and sector analysis."""
        # Get base risk report
        base_report = self.generate_risk_report(current_positions, current_prices, portfolio_value)
        
        # Add enhanced features
        enhanced_report = base_report.copy()
        enhanced_report.update({
            'circuit_breaker_status': self.get_circuit_breaker_status(),
            'sector_utilization': self.get_sector_utilization(current_positions, portfolio_value),
            'enhanced_risk_parameters': {
                'daily_loss_limit': self.daily_loss_limit,
                'intraday_loss_limit': self.intraday_loss_limit,
                'volatility_threshold': self.volatility_threshold,
                'enhanced_sector_limits': len(self.sector_limits)
            },
            'volatility_analysis': self._analyze_portfolio_volatility(
                current_positions, volatility_data
            ) if volatility_data else None,
            'enhanced_recommendations': self._generate_enhanced_recommendations(
                base_report['risk_metrics'], current_positions, portfolio_value
            )
        })
        
        return enhanced_report
    
    def _analyze_portfolio_volatility(
        self,
        current_positions: Dict[str, Dict[str, Any]],
        volatility_data: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze portfolio volatility characteristics."""
        if not current_positions:
            return {'portfolio_volatility': 0, 'volatility_contributors': []}
        
        total_value = sum(abs(pos.get('market_value', 0)) for pos in current_positions.values())
        
        volatility_contributors = []
        weighted_vol = 0.0
        
        for ticker, position in current_positions.items():
            if ticker in volatility_data:
                position_value = abs(position.get('market_value', 0))
                weight = position_value / total_value if total_value > 0 else 0
                stock_vol = volatility_data[ticker]
                contribution = weight * stock_vol
                weighted_vol += contribution
                
                volatility_contributors.append({
                    'ticker': ticker,
                    'weight': weight,
                    'volatility': stock_vol,
                    'contribution': contribution,
                    'risk_level': 'HIGH' if stock_vol > 0.3 else 'MEDIUM' if stock_vol > 0.2 else 'LOW'
                })
        
        # Sort by contribution
        volatility_contributors.sort(key=lambda x: x['contribution'], reverse=True)
        
        return {
            'portfolio_volatility': weighted_vol,
            'volatility_vs_threshold': weighted_vol / self.volatility_threshold,
            'high_volatility_positions': len([c for c in volatility_contributors if c['risk_level'] == 'HIGH']),
            'volatility_contributors': volatility_contributors[:10]  # Top 10
        }
    
    def _generate_enhanced_recommendations(
        self,
        risk_metrics: RiskMetrics,
        current_positions: Dict[str, Dict[str, Any]],
        portfolio_value: float
    ) -> List[str]:
        """Generate enhanced risk management recommendations."""
        recommendations = self._generate_risk_recommendations(risk_metrics)
        
        # Add circuit breaker recommendations
        if self.trading_halted:
            recommendations.insert(0, f"URGENT: Trading halted - {self.halt_reason}")
            recommendations.insert(1, f"Review positions and wait for recovery at {self.recovery_time}")
        
        # Add sector-specific recommendations
        sector_util = self.get_sector_utilization(current_positions, portfolio_value)
        for sector, data in sector_util.items():
            if data['utilization_pct'] > 90:
                recommendations.append(f"Sector {sector} near limit ({data['utilization_pct']:.0f}% utilized)")
            elif data['utilization_pct'] > 75:
                recommendations.append(f"Sector {sector} approaching limit ({data['utilization_pct']:.0f}% utilized)")
        
        # Add volatility recommendations
        if len(self.intraday_returns) > 0:
            recent_vol = np.std(self.intraday_returns[-12:]) if len(self.intraday_returns) >= 12 else 0
            if recent_vol > 0.02:  # 2% intraday volatility
                recommendations.append("High intraday volatility detected - consider reducing position sizes")
        
        return recommendations[:8]  # Limit to top 8 recommendations


def main():
    """Demo and testing of enhanced risk manager."""
    enhanced_rm = EnhancedRiskManager()
    
    # Sample portfolio for testing
    sample_positions = {
        'NVDA': {'market_value': 15000, 'entry_price': 800, 'size': 18.75},
        'AAPL': {'market_value': 12000, 'entry_price': 200, 'size': 60},
        'GOOGL': {'market_value': 10000, 'entry_price': 2500, 'size': 4},
        'JNJ': {'market_value': 8000, 'entry_price': 160, 'size': 50}
    }
    
    sample_prices = {'NVDA': 850, 'AAPL': 205, 'GOOGL': 2600, 'JNJ': 165}
    portfolio_value = 50000
    
    # Generate enhanced risk report
    report = enhanced_rm.generate_enhanced_risk_report(
        sample_positions, sample_prices, portfolio_value
    )
    
    print("Enhanced Risk Manager Report")
    print("=" * 40)
    print(f"Portfolio Value: ${portfolio_value:,.2f}")
    print(f"Circuit Breaker State: {report['circuit_breaker_status']['state']}")
    print(f"Risk Level: {report['risk_metrics'].risk_level.value}")
    print(f"Risk Score: {report['risk_metrics'].risk_score:.1f}/100")
    
    print("\nSector Utilization:")
    for sector, data in report['sector_utilization'].items():
        if data['current_exposure_pct'] > 0:
            print(f"  {sector}: {data['current_exposure_pct']:.1%} / {data['max_exposure_pct']:.1%}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(report['enhanced_recommendations'][:5], 1):
        print(f"  {i}. {rec}")


if __name__ == "__main__":
    main()