#!/usr/bin/env python3
"""
Real-time Portfolio Monitoring Dashboard
Advanced portfolio monitoring with key performance metrics and alerts.
"""

import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.portfolio_tracker import PortfolioTracker
from trading.risk_manager import RiskManager
from analysis.market_analyzer import DailyMarketAnalyzer
from config.settings import config


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Portfolio alert structure."""
    timestamp: datetime
    level: AlertLevel
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    action_required: bool = False


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics structure."""
    total_value: float
    total_return_pct: float
    daily_return_pct: float
    weekly_return_pct: float
    monthly_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    avg_trade_return: float
    volatility_annualized: float
    var_95_daily: float
    positions_count: int
    cash_balance: float
    market_exposure_pct: float
    largest_position_pct: float
    sector_concentration: Dict[str, float]
    correlation_risk_score: float


class PortfolioDashboard:
    """Real-time portfolio monitoring dashboard."""
    
    def __init__(self, portfolio_file: str = "portfolio.toml"):
        self.portfolio_file = Path(portfolio_file)
        self.tracker = PortfolioTracker(str(portfolio_file))
        self.risk_manager = RiskManager()
        self.analyzer = DailyMarketAnalyzer()
        
        # Alert thresholds
        self.alert_thresholds = {
            'max_drawdown_pct': 15.0,  # Critical at 15%
            'daily_loss_pct': 5.0,     # Warning at 5% daily loss
            'position_size_pct': 18.0, # Warning at 18% position
            'correlation_risk': 0.8,   # Warning at 80% correlation
            'sharpe_ratio': 1.0,       # Warning below 1.0
            'volatility_annual': 25.0, # Warning above 25%
            'var_breach_count': 3      # Warning after 3 VaR breaches
        }
        
        # Performance history for trend analysis
        self.performance_history: List[Dict] = []
        self.alerts_history: List[Alert] = []
        
        # Load historical performance data
        self._load_performance_history()
    
    def generate_dashboard_report(self, update_prices: bool = True) -> Dict[str, Any]:
        """Generate comprehensive dashboard report."""
        logger.info("Generating portfolio dashboard report")
        
        try:
            # Update market prices if requested
            if update_prices:
                self._update_market_prices()
            
            # Calculate current performance metrics
            metrics = self._calculate_performance_metrics()
            
            # Check for alerts
            alerts = self._check_alerts(metrics)
            
            # Generate summary statistics
            summary = self._generate_summary(metrics, alerts)
            
            # Update performance history
            self._update_performance_history(metrics)
            
            # Create dashboard report
            dashboard_report = {
                'timestamp': datetime.now(),
                'portfolio_file': str(self.portfolio_file),
                'summary': summary,
                'metrics': metrics,
                'alerts': [self._alert_to_dict(alert) for alert in alerts],
                'positions': self._get_position_details(),
                'sector_allocation': metrics.sector_concentration,
                'performance_trend': self._get_performance_trend(),
                'risk_metrics': self._get_risk_metrics(metrics),
                'recommendations': self._generate_recommendations(metrics, alerts)
            }
            
            return dashboard_report
            
        except Exception as e:
            logger.error(f"Error generating dashboard report: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}
    
    def display_dashboard(self, update_prices: bool = True):
        """Display formatted dashboard in terminal."""
        report = self.generate_dashboard_report(update_prices)
        
        if 'error' in report:
            print(f"❌ Dashboard Error: {report['error']}")
            return
        
        # Dashboard header
        print("\n" + "="*80)
        print("🎯 AI TRADING SYSTEM - PORTFOLIO DASHBOARD")
        print("="*80)
        print(f"📅 {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | Portfolio: {report['portfolio_file']}")
        print()
        
        # Summary section
        summary = report['summary']
        metrics = report['metrics']
        
        print("📊 PORTFOLIO SUMMARY")
        print("-" * 30)
        print(f"💰 Total Value:     ${metrics.total_value:,.2f}")
        print(f"📈 Total Return:    {metrics.total_return_pct:+.2f}%")
        print(f"📅 Daily Return:    {metrics.daily_return_pct:+.2f}%")
        print(f"🗓️ Weekly Return:   {metrics.weekly_return_pct:+.2f}%")
        print(f"📆 Monthly Return:  {metrics.monthly_return_pct:+.2f}%")
        print(f"⚡ Sharpe Ratio:    {metrics.sharpe_ratio:.2f}")
        print(f"📉 Max Drawdown:    {metrics.max_drawdown_pct:.2f}%")
        print(f"🎯 Win Rate:        {metrics.win_rate:.1f}%")
        print()
        
        # Risk metrics
        print("⚠️ RISK METRICS")
        print("-" * 20)
        print(f"💹 Volatility:      {metrics.volatility_annualized:.1f}%")
        print(f"📊 VaR (95%):       ${metrics.var_95_daily:,.0f}")
        print(f"🔗 Correlation Risk: {metrics.correlation_risk_score:.2f}")
        print(f"🎪 Market Exposure: {metrics.market_exposure_pct:.1f}%")
        print(f"🏆 Largest Position: {metrics.largest_position_pct:.1f}%")
        print()
        
        # Alerts section
        alerts = report['alerts']
        if alerts:
            print("🚨 ACTIVE ALERTS")
            print("-" * 20)
            for alert in alerts:
                level_emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🔴"}
                emoji = level_emoji.get(alert['level'], "⚠️")
                print(f"{emoji} {alert['title']}")
                print(f"   {alert['message']}")
                if alert['action_required']:
                    print("   🎯 ACTION REQUIRED")
                print()
        else:
            print("✅ NO ALERTS - System operating normally")
            print()
        
        # Position details
        positions = report['positions']
        if positions:
            print("📋 ACTIVE POSITIONS")
            print("-" * 25)
            print("Symbol    Shares    Value      P&L     Weight")
            print("-" * 45)
            for pos in positions[:10]:  # Show top 10
                pnl_emoji = "📈" if pos['unrealized_pnl'] >= 0 else "📉"
                print(f"{pos['symbol']:<8} {pos['shares']:>6.1f} ${pos['value']:>8,.0f} {pnl_emoji}{pos['unrealized_pnl_pct']:>+6.1f}% {pos['weight_pct']:>6.1f}%")
            
            if len(positions) > 10:
                print(f"... and {len(positions)-10} more positions")
            print()
        
        # Sector allocation
        sectors = report['sector_allocation']
        if sectors:
            print("🏭 SECTOR ALLOCATION")
            print("-" * 25)
            for sector, weight in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
                if weight > 1.0:  # Show sectors with >1% allocation
                    print(f"{sector:<20} {weight:>6.1f}%")
            print()
        
        # Recommendations
        recommendations = report['recommendations']
        if recommendations:
            print("💡 RECOMMENDATIONS")
            print("-" * 25)
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"{i}. {rec}")
            print()
        
        # Performance trend
        trend = report['performance_trend']
        if trend and len(trend) > 1:
            print("📈 PERFORMANCE TREND (Last 7 Days)")
            print("-" * 40)
            recent_returns = [t['daily_return_pct'] for t in trend[-7:]]
            avg_return = np.mean(recent_returns)
            trend_emoji = "📈" if avg_return > 0 else "📉" if avg_return < 0 else "➡️"
            print(f"Average Daily Return: {trend_emoji} {avg_return:+.2f}%")
            print(f"Best Day: {max(recent_returns):+.2f}%")
            print(f"Worst Day: {min(recent_returns):+.2f}%")
            print()
        
        print("="*80)
    
    def _update_market_prices(self):
        """Update portfolio with latest market prices."""
        try:
            market_data = self.analyzer.fetch_market_data(lookback_days=2)
            if market_data:
                latest_prices = {}
                for symbol, data in market_data.items():
                    if not data.empty:
                        latest_prices[symbol] = data['close'].iloc[-1]
                
                if latest_prices:
                    self.tracker.update_market_prices(latest_prices)
                    logger.info(f"Updated prices for {len(latest_prices)} stocks")
                    
        except Exception as e:
            logger.warning(f"Failed to update market prices: {e}")
    
    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        portfolio = self.tracker.load_portfolio()
        
        # Basic portfolio values
        total_value = self.tracker.get_total_portfolio_value()
        cash_balance = portfolio.get('cash_balance', 0)
        starting_capital = portfolio.get('starting_capital', config.STARTING_CAPITAL)
        
        # Calculate returns
        total_return_pct = ((total_value - starting_capital) / starting_capital) * 100
        
        # Get position details
        positions = []
        total_position_value = 0
        sector_values = {}
        
        for symbol, position in portfolio.get('positions', {}).items():
            shares = position['shares']
            current_price = position.get('current_price', position['avg_price'])
            value = shares * current_price
            
            total_position_value += value
            
            # Sector allocation (simplified)
            sector = self._get_stock_sector(symbol)
            sector_values[sector] = sector_values.get(sector, 0) + value
            
            positions.append({
                'symbol': symbol,
                'shares': shares,
                'value': value,
                'weight_pct': (value / total_value) * 100
            })
        
        # Market exposure
        market_exposure_pct = (total_position_value / total_value) * 100
        
        # Largest position
        largest_position_pct = max([p['weight_pct'] for p in positions], default=0)
        
        # Sector concentration
        sector_concentration = {
            sector: (value / total_value) * 100 
            for sector, value in sector_values.items()
        }
        
        # Calculate time-based returns
        performance_history = self._get_performance_history()
        daily_return_pct = self._calculate_period_return(performance_history, days=1)
        weekly_return_pct = self._calculate_period_return(performance_history, days=7)
        monthly_return_pct = self._calculate_period_return(performance_history, days=30)
        
        # Risk metrics
        returns_series = [p.get('daily_return_pct', 0) for p in performance_history[-252:]]  # 1 year
        volatility_annualized = np.std(returns_series) * np.sqrt(252) if returns_series else 0
        
        # Sharpe ratio (assuming 3% risk-free rate)
        risk_free_rate = 0.03
        excess_returns = np.mean(returns_series) - (risk_free_rate / 252)
        sharpe_ratio = (excess_returns / (volatility_annualized / np.sqrt(252))) if volatility_annualized > 0 else 0
        
        # Maximum drawdown
        max_drawdown_pct = self._calculate_max_drawdown(performance_history)
        
        # VaR (95% confidence)
        var_95_daily = np.percentile(returns_series, 5) * total_value / 100 if returns_series else 0
        
        # Win rate and average trade return
        trades = portfolio.get('trade_history', [])
        win_rate = self._calculate_win_rate(trades)
        avg_trade_return = self._calculate_avg_trade_return(trades)
        
        # Correlation risk score
        correlation_risk_score = self._calculate_correlation_risk(positions)
        
        return PerformanceMetrics(
            total_value=total_value,
            total_return_pct=total_return_pct,
            daily_return_pct=daily_return_pct,
            weekly_return_pct=weekly_return_pct,
            monthly_return_pct=monthly_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            avg_trade_return=avg_trade_return,
            volatility_annualized=volatility_annualized,
            var_95_daily=var_95_daily,
            positions_count=len(positions),
            cash_balance=cash_balance,
            market_exposure_pct=market_exposure_pct,
            largest_position_pct=largest_position_pct,
            sector_concentration=sector_concentration,
            correlation_risk_score=correlation_risk_score
        )
    
    def _check_alerts(self, metrics: PerformanceMetrics) -> List[Alert]:
        """Check for portfolio alerts based on metrics."""
        alerts = []
        now = datetime.now()
        
        # Critical drawdown alert
        if metrics.max_drawdown_pct > self.alert_thresholds['max_drawdown_pct']:
            alerts.append(Alert(
                timestamp=now,
                level=AlertLevel.CRITICAL,
                title="Critical Drawdown Alert",
                message=f"Portfolio drawdown ({metrics.max_drawdown_pct:.1f}%) exceeds {self.alert_thresholds['max_drawdown_pct']:.1f}% limit",
                metric_name="max_drawdown_pct",
                current_value=metrics.max_drawdown_pct,
                threshold_value=self.alert_thresholds['max_drawdown_pct'],
                action_required=True
            ))
        
        # Daily loss alert
        if metrics.daily_return_pct < -self.alert_thresholds['daily_loss_pct']:
            alerts.append(Alert(
                timestamp=now,
                level=AlertLevel.WARNING,
                title="Daily Loss Alert",
                message=f"Daily return ({metrics.daily_return_pct:.2f}%) exceeds {self.alert_thresholds['daily_loss_pct']:.1f}% loss limit",
                metric_name="daily_return_pct",
                current_value=metrics.daily_return_pct,
                threshold_value=-self.alert_thresholds['daily_loss_pct']
            ))
        
        # Position size alert
        if metrics.largest_position_pct > self.alert_thresholds['position_size_pct']:
            alerts.append(Alert(
                timestamp=now,
                level=AlertLevel.WARNING,
                title="Position Size Alert",
                message=f"Largest position ({metrics.largest_position_pct:.1f}%) exceeds {self.alert_thresholds['position_size_pct']:.1f}% limit",
                metric_name="largest_position_pct",
                current_value=metrics.largest_position_pct,
                threshold_value=self.alert_thresholds['position_size_pct']
            ))
        
        # Correlation risk alert
        if metrics.correlation_risk_score > self.alert_thresholds['correlation_risk']:
            alerts.append(Alert(
                timestamp=now,
                level=AlertLevel.WARNING,
                title="Correlation Risk Alert",
                message=f"Portfolio correlation risk ({metrics.correlation_risk_score:.2f}) exceeds {self.alert_thresholds['correlation_risk']:.2f} threshold",
                metric_name="correlation_risk_score",
                current_value=metrics.correlation_risk_score,
                threshold_value=self.alert_thresholds['correlation_risk']
            ))
        
        # Sharpe ratio alert
        if metrics.sharpe_ratio < self.alert_thresholds['sharpe_ratio']:
            alerts.append(Alert(
                timestamp=now,
                level=AlertLevel.INFO,
                title="Sharpe Ratio Alert",
                message=f"Sharpe ratio ({metrics.sharpe_ratio:.2f}) below target {self.alert_thresholds['sharpe_ratio']:.1f}",
                metric_name="sharpe_ratio",
                current_value=metrics.sharpe_ratio,
                threshold_value=self.alert_thresholds['sharpe_ratio']
            ))
        
        # High volatility alert
        if metrics.volatility_annualized > self.alert_thresholds['volatility_annual']:
            alerts.append(Alert(
                timestamp=now,
                level=AlertLevel.WARNING,
                title="High Volatility Alert",
                message=f"Portfolio volatility ({metrics.volatility_annualized:.1f}%) exceeds {self.alert_thresholds['volatility_annual']:.1f}% target",
                metric_name="volatility_annualized",
                current_value=metrics.volatility_annualized,
                threshold_value=self.alert_thresholds['volatility_annual']
            ))
        
        # Store alerts in history
        self.alerts_history.extend(alerts)
        
        return alerts
    
    def _generate_summary(self, metrics: PerformanceMetrics, alerts: List[Alert]) -> Dict[str, Any]:
        """Generate portfolio summary."""
        # Determine overall health
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        warning_alerts = [a for a in alerts if a.level == AlertLevel.WARNING]
        
        if critical_alerts:
            health_status = "CRITICAL"
            health_emoji = "🔴"
        elif warning_alerts:
            health_status = "WARNING"
            health_emoji = "🟡"
        else:
            health_status = "HEALTHY"
            health_emoji = "🟢"
        
        # Performance assessment
        performance_score = self._calculate_performance_score(metrics)
        
        return {
            'health_status': health_status,
            'health_emoji': health_emoji,
            'performance_score': performance_score,
            'total_alerts': len(alerts),
            'critical_alerts': len(critical_alerts),
            'warning_alerts': len(warning_alerts),
            'last_updated': datetime.now()
        }
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall performance score (0-100)."""
        score = 50  # Base score
        
        # Return contribution (0-25 points)
        if metrics.monthly_return_pct >= 3.0:
            score += 25
        elif metrics.monthly_return_pct >= 1.0:
            score += 15
        elif metrics.monthly_return_pct >= 0:
            score += 5
        else:
            score -= 10
        
        # Sharpe ratio contribution (0-15 points)
        if metrics.sharpe_ratio >= 2.0:
            score += 15
        elif metrics.sharpe_ratio >= 1.5:
            score += 10
        elif metrics.sharpe_ratio >= 1.0:
            score += 5
        
        # Drawdown penalty (0 to -20 points)
        if metrics.max_drawdown_pct <= 5:
            score += 5
        elif metrics.max_drawdown_pct <= 10:
            pass  # No change
        elif metrics.max_drawdown_pct <= 15:
            score -= 10
        else:
            score -= 20
        
        # Win rate contribution (0-10 points)
        if metrics.win_rate >= 60:
            score += 10
        elif metrics.win_rate >= 55:
            score += 5
        
        return max(0, min(100, score))
    
    def _get_position_details(self) -> List[Dict]:
        """Get detailed position information."""
        portfolio = self.tracker.load_portfolio()
        positions = []
        
        for symbol, position in portfolio.get('positions', {}).items():
            shares = position['shares']
            avg_price = position['avg_price']
            current_price = position.get('current_price', avg_price)
            
            value = shares * current_price
            cost = shares * avg_price
            unrealized_pnl = value - cost
            unrealized_pnl_pct = (unrealized_pnl / cost) * 100 if cost > 0 else 0
            
            total_value = self.tracker.get_total_portfolio_value()
            weight_pct = (value / total_value) * 100 if total_value > 0 else 0
            
            positions.append({
                'symbol': symbol,
                'shares': shares,
                'avg_price': avg_price,
                'current_price': current_price,
                'value': value,
                'cost': cost,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct,
                'weight_pct': weight_pct
            })
        
        # Sort by value (largest first)
        return sorted(positions, key=lambda x: x['value'], reverse=True)
    
    def _get_stock_sector(self, symbol: str) -> str:
        """Get stock sector (simplified mapping)."""
        sector_mapping = {
            # Technology
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'META': 'Technology',
            'NVDA': 'Technology', 'AMD': 'Technology', 'INTC': 'Technology', 'QCOM': 'Technology',
            'ASML': 'Technology', 'TSM': 'Technology', 'AVGO': 'Technology', 'MU': 'Technology',
            'SMCI': 'Technology', 'ARM': 'Technology', 'CRM': 'Technology',
            
            # Healthcare
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
            'TMO': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare', 'MRNA': 'Healthcare',
            'BMY': 'Healthcare', 'LLY': 'Healthcare',
            
            # Financials
            'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
            'MS': 'Financials', 'V': 'Financials', 'MA': 'Financials', 'BRK-B': 'Financials',
            'AXP': 'Financials', 'C': 'Financials',
            
            # Consumer Discretionary
            'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary', 'HD': 'Consumer Discretionary',
            'MCD': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary',
            'TGT': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary', 'DIS': 'Consumer Discretionary',
            'BKNG': 'Consumer Discretionary',
            
            # Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
            'EOG': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy', 'MPC': 'Energy'
        }
        
        return sector_mapping.get(symbol, 'Other')
    
    def _get_performance_history(self) -> List[Dict]:
        """Get performance history for trend analysis."""
        return self.performance_history
    
    def _update_performance_history(self, metrics: PerformanceMetrics):
        """Update performance history with current metrics."""
        history_entry = {
            'timestamp': datetime.now(),
            'total_value': metrics.total_value,
            'total_return_pct': metrics.total_return_pct,
            'daily_return_pct': metrics.daily_return_pct,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown_pct': metrics.max_drawdown_pct,
            'volatility_annualized': metrics.volatility_annualized
        }
        
        self.performance_history.append(history_entry)
        
        # Keep only last 365 days
        cutoff_date = datetime.now() - timedelta(days=365)
        self.performance_history = [
            h for h in self.performance_history 
            if h['timestamp'] > cutoff_date
        ]
    
    def _load_performance_history(self):
        """Load historical performance data."""
        history_file = Path("portfolio_performance_history.json")
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.performance_history = [
                        {**h, 'timestamp': datetime.fromisoformat(h['timestamp'])}
                        for h in data
                    ]
            except Exception as e:
                logger.warning(f"Could not load performance history: {e}")
    
    def save_performance_history(self):
        """Save performance history to file."""
        history_file = Path("portfolio_performance_history.json")
        try:
            data = [
                {**h, 'timestamp': h['timestamp'].isoformat()}
                for h in self.performance_history
            ]
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save performance history: {e}")
    
    def _calculate_period_return(self, history: List[Dict], days: int) -> float:
        """Calculate return for specified period."""
        if len(history) < 2:
            return 0.0
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_history = [h for h in history if h['timestamp'] > cutoff_date]
        
        if len(recent_history) < 2:
            return 0.0
        
        start_value = recent_history[0]['total_value']
        end_value = recent_history[-1]['total_value']
        
        return ((end_value - start_value) / start_value) * 100 if start_value > 0 else 0.0
    
    def _calculate_max_drawdown(self, history: List[Dict]) -> float:
        """Calculate maximum drawdown from history."""
        if len(history) < 2:
            return 0.0
        
        values = [h['total_value'] for h in history]
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak * 100
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trade history."""
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        return (winning_trades / len(trades)) * 100
    
    def _calculate_avg_trade_return(self, trades: List[Dict]) -> float:
        """Calculate average trade return."""
        if not trades:
            return 0.0
        
        returns = [trade.get('return_pct', 0) for trade in trades]
        return np.mean(returns)
    
    def _calculate_correlation_risk(self, positions: List[Dict]) -> float:
        """Calculate portfolio correlation risk score."""
        if len(positions) < 2:
            return 0.0
        
        # Simplified correlation risk based on sector concentration
        sectors = {}
        total_value = sum(pos['value'] for pos in positions)
        
        for pos in positions:
            sector = self._get_stock_sector(pos['symbol'])
            sectors[sector] = sectors.get(sector, 0) + pos['value']
        
        # Calculate Herfindahl index for sector concentration
        sector_weights = [v/total_value for v in sectors.values()]
        herfindahl = sum(w**2 for w in sector_weights)
        
        # Convert to risk score (0-1, where 1 is maximum risk)
        return min(1.0, herfindahl * 2)
    
    def _get_performance_trend(self) -> List[Dict]:
        """Get recent performance trend."""
        return self.performance_history[-30:] if len(self.performance_history) > 30 else self.performance_history
    
    def _get_risk_metrics(self, metrics: PerformanceMetrics) -> Dict[str, float]:
        """Get detailed risk metrics."""
        return {
            'volatility_annualized': metrics.volatility_annualized,
            'var_95_daily': metrics.var_95_daily,
            'max_drawdown_pct': metrics.max_drawdown_pct,
            'correlation_risk_score': metrics.correlation_risk_score,
            'market_exposure_pct': metrics.market_exposure_pct,
            'largest_position_pct': metrics.largest_position_pct,
            'sharpe_ratio': metrics.sharpe_ratio
        }
    
    def _generate_recommendations(self, metrics: PerformanceMetrics, alerts: List[Alert]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Critical alerts first
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        if critical_alerts:
            recommendations.append("URGENT: Address critical portfolio alerts immediately")
        
        # Performance-based recommendations
        if metrics.sharpe_ratio < 1.0:
            recommendations.append("Consider reducing position sizes or improving stock selection")
        
        if metrics.max_drawdown_pct > 10:
            recommendations.append("Review stop-loss levels and risk management rules")
        
        if metrics.largest_position_pct > 15:
            recommendations.append("Reduce concentration in largest position")
        
        if metrics.correlation_risk_score > 0.7:
            recommendations.append("Diversify across more sectors to reduce correlation risk")
        
        if metrics.market_exposure_pct > 85:
            recommendations.append("Consider increasing cash position for opportunities")
        
        if metrics.volatility_annualized > 25:
            recommendations.append("Reduce portfolio volatility through position sizing")
        
        if metrics.monthly_return_pct < 1.0 and metrics.monthly_return_pct > -1.0:
            recommendations.append("Portfolio performance is flat - review strategy effectiveness")
        
        # Positive recommendations
        if not recommendations and metrics.sharpe_ratio > 1.5:
            recommendations.append("Portfolio performing well - maintain current strategy")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _alert_to_dict(self, alert: Alert) -> Dict[str, Any]:
        """Convert alert to dictionary for JSON serialization."""
        return {
            'timestamp': alert.timestamp.isoformat(),
            'level': alert.level.value,
            'title': alert.title,
            'message': alert.message,
            'metric_name': alert.metric_name,
            'current_value': alert.current_value,
            'threshold_value': alert.threshold_value,
            'action_required': alert.action_required
        }


def main():
    """Main entry point for dashboard CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Portfolio Dashboard")
    parser.add_argument("--portfolio", "-p", default="portfolio.toml", help="Portfolio file")
    parser.add_argument("--no-update", action="store_true", help="Skip price updates")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    dashboard = PortfolioDashboard(args.portfolio)
    
    try:
        if args.json:
            report = dashboard.generate_dashboard_report(not args.no_update)
            print(json.dumps(report, indent=2, default=str))
        else:
            dashboard.display_dashboard(not args.no_update)
            
    except KeyboardInterrupt:
        print("\n⚠️ Dashboard interrupted by user")
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        print(f"❌ Dashboard Error: {e}")


if __name__ == "__main__":
    main()