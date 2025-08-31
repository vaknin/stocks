"""
Trading Recommendation Engine
Generates actionable buy/sell/hold recommendations based on market analysis and current portfolio.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

from src.trading.portfolio_tracker import PortfolioTracker, Position
from src.analysis.market_analyzer import DailyMarketAnalyzer, MarketAnalysis, MarketSignal


@dataclass
class TradingRecommendation:
    """Individual trading recommendation with specific actions."""
    action: str  # 'BUY', 'SELL', 'HOLD', 'REDUCE', 'ADD'
    symbol: str
    current_position: Optional[float]  # Current shares held
    recommended_shares: float  # Target shares to hold
    dollar_amount: float  # Dollar amount to buy/sell
    entry_price: float  # Recommended entry price
    stop_loss: float
    target_price: float
    confidence: float
    reasoning: str
    priority: int  # 1=highest, 5=lowest priority
    timeframe: str


@dataclass
class PortfolioRecommendation:
    """Complete portfolio recommendations."""
    analysis_date: str
    current_portfolio_value: float
    recommended_allocation: float  # Total % to invest based on regime
    cash_reserve: float
    recommendations: List[TradingRecommendation]
    portfolio_summary: str
    market_outlook: str
    risk_assessment: str


class TradingRecommendationEngine:
    """Generate actionable trading recommendations."""
    
    def __init__(self, portfolio_file: str = "portfolio.toml"):
        logger.info("Initializing TradingRecommendationEngine")
        
        self.portfolio_tracker = PortfolioTracker(portfolio_file)
        self.market_analyzer = DailyMarketAnalyzer()
        
        # Kelly sizing parameters from README.md
        self.kelly_multiplier = 0.25  # Conservative 0.25x Kelly
        self.max_position_pct = 0.20  # Max 20% per stock
        self.max_positions = 5        # Max 5 positions
        
        # Transaction cost estimate (0.1% per trade)
        self.transaction_cost_pct = 0.001
        
        logger.info("TradingRecommendationEngine initialized")
    
    def calculate_position_size(self, signal: MarketSignal, portfolio_value: float, 
                              current_allocation: float) -> float:
        """Calculate optimal position size using Kelly criterion."""
        try:
            # Kelly formula: f = (bp - q) / b
            # Where: b = odds, p = win probability, q = loss probability
            
            # Estimate win probability from confidence
            win_prob = signal.confidence
            loss_prob = 1 - win_prob
            
            # Estimate odds from predicted return and risk
            expected_return = abs(signal.predicted_return)
            estimated_risk = max(0.02, expected_return * 0.5)  # Assume 50% of return as risk
            
            if estimated_risk > 0:
                odds = expected_return / estimated_risk
                kelly_fraction = (odds * win_prob - loss_prob) / odds
            else:
                kelly_fraction = 0.1  # Default conservative sizing
            
            # Apply safety multiplier and constraints
            kelly_fraction = max(0, min(kelly_fraction, 0.5))  # Cap at 50%
            safe_kelly = kelly_fraction * self.kelly_multiplier
            
            # Respect position size limits
            max_position_size = portfolio_value * self.max_position_pct
            kelly_position_size = portfolio_value * safe_kelly
            
            position_size = min(max_position_size, kelly_position_size)
            
            # Account for current allocation limits
            available_cash = portfolio_value * (1.0 - current_allocation)
            position_size = min(position_size, available_cash * 0.8)  # Leave some cash buffer
            
            logger.debug(f"Position size calculation for {signal.symbol}: "
                        f"Kelly={safe_kelly:.2%}, Size=${position_size:.0f}")
            
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size for {signal.symbol}: {e}")
            return 0
    
    def prioritize_signals(self, signals: List[MarketSignal], current_positions: Dict[str, Position]) -> List[MarketSignal]:
        """Prioritize trading signals based on confidence, opportunity, and portfolio balance."""
        try:
            # Score each signal
            scored_signals = []
            
            for signal in signals:
                score = 0
                
                # Base score from confidence and expected return
                score += signal.confidence * 40  # Up to 40 points for confidence
                score += abs(signal.predicted_return) * 100 * 20  # Up to ~20 points for return
                
                # Bonus for diversification (new positions)
                if signal.symbol not in current_positions:
                    score += 15
                
                # Penalty for over-concentration
                if signal.symbol in current_positions:
                    current_weight = current_positions[signal.symbol].current_value / sum(
                        p.current_value for p in current_positions.values()
                    )
                    if current_weight > 0.15:  # Already >15% of portfolio
                        score -= 10
                
                # Bonus for regime alignment
                if signal.market_regime in ['bull_trend', 'sideways']:
                    if signal.signal_type == 'BUY':
                        score += 10
                elif signal.market_regime in ['bear_trend', 'high_volatility']:
                    if signal.signal_type == 'SELL':
                        score += 10
                
                scored_signals.append((signal, score))
            
            # Sort by score (highest first)
            scored_signals.sort(key=lambda x: x[1], reverse=True)
            
            # Return prioritized signals
            prioritized = [signal for signal, score in scored_signals]
            
            logger.info(f"Prioritized {len(prioritized)} signals")
            return prioritized
            
        except Exception as e:
            logger.error(f"Error prioritizing signals: {e}")
            return signals
    
    def generate_recommendations(self, timeframes: List[str] = ['daily']) -> PortfolioRecommendation:
        """Generate complete portfolio recommendations."""
        try:
            logger.info("Generating trading recommendations")
            analysis_date = datetime.now().strftime("%Y-%m-%d")
            
            # Get current portfolio state
            portfolio_summary = self.portfolio_tracker.get_portfolio_summary()
            current_portfolio_value = portfolio_summary['total_value']
            current_positions = self.portfolio_tracker.positions
            current_allocation = (portfolio_summary['invested_value'] / current_portfolio_value)
            
            # Perform market analysis
            market_analysis = self.market_analyzer.analyze_market(timeframes)
            
            # Determine target allocation based on market regime
            target_allocation = market_analysis.recommended_allocation
            
            # Prioritize signals
            prioritized_signals = self.prioritize_signals(market_analysis.signals, current_positions)
            
            # Generate specific recommendations
            recommendations = []
            total_recommended_investment = 0
            processed_symbols = set()
            
            # 1. Process existing positions first
            for symbol, position in current_positions.items():
                if symbol in processed_symbols:
                    continue
                    
                processed_symbols.add(symbol)
                
                # Find signal for this position
                signal = next((s for s in prioritized_signals if s.symbol == symbol), None)
                
                if signal and signal.signal_type == 'SELL':
                    # Recommend selling position
                    recommendation = TradingRecommendation(
                        action='SELL',
                        symbol=symbol,
                        current_position=position.shares,
                        recommended_shares=0,
                        dollar_amount=position.current_value,
                        entry_price=position.current_price,
                        stop_loss=signal.stop_loss,
                        target_price=signal.target_price,
                        confidence=signal.confidence,
                        reasoning=f"SELL signal: {signal.reasoning}. Current P&L: {position.unrealized_pnl_pct:+.1f}%",
                        priority=1 if signal.confidence > 0.8 else 2,
                        timeframe=signal.timeframe
                    )
                    recommendations.append(recommendation)
                    
                elif signal and signal.signal_type == 'BUY' and signal.predicted_return > 0.02:
                    # Consider adding to position
                    additional_size = self.calculate_position_size(signal, current_portfolio_value, current_allocation)
                    if additional_size > 100:  # Only if meaningful size
                        shares_to_add = additional_size / position.current_price
                        recommendation = TradingRecommendation(
                            action='ADD',
                            symbol=symbol,
                            current_position=position.shares,
                            recommended_shares=position.shares + shares_to_add,
                            dollar_amount=additional_size,
                            entry_price=position.current_price,
                            stop_loss=signal.stop_loss,
                            target_price=signal.target_price,
                            confidence=signal.confidence,
                            reasoning=f"ADD to position: {signal.reasoning}",
                            priority=3,
                            timeframe=signal.timeframe
                        )
                        recommendations.append(recommendation)
                        total_recommended_investment += additional_size
                
                else:
                    # HOLD current position
                    recommendation = TradingRecommendation(
                        action='HOLD',
                        symbol=symbol,
                        current_position=position.shares,
                        recommended_shares=position.shares,
                        dollar_amount=0,
                        entry_price=position.current_price,
                        stop_loss=position.stop_loss or position.current_price * 0.95,
                        target_price=position.target_price or position.current_price * 1.1,
                        confidence=signal.confidence if signal else 0.6,
                        reasoning=f"HOLD: {'Maintain position per market analysis' if not signal else signal.reasoning}",
                        priority=4,
                        timeframe=signal.timeframe if signal else 'daily'
                    )
                    recommendations.append(recommendation)
            
            # 2. Process new BUY signals
            available_cash = current_portfolio_value * target_allocation - total_recommended_investment
            position_count = len([r for r in recommendations if r.action in ['HOLD', 'ADD']])
            
            for signal in prioritized_signals:
                if signal.symbol in processed_symbols:
                    continue
                if signal.signal_type != 'BUY':
                    continue
                if position_count >= self.max_positions:
                    break
                if available_cash < 200:  # Minimum position size
                    break
                
                # Calculate position size
                position_size = self.calculate_position_size(signal, current_portfolio_value, 0)
                position_size = min(position_size, available_cash * 0.8)  # Don't use all cash
                
                if position_size < 100:  # Skip tiny positions
                    continue
                
                shares = position_size / signal.target_price * 0.99  # Slight buffer for price movement
                
                recommendation = TradingRecommendation(
                    action='BUY',
                    symbol=signal.symbol,
                    current_position=0,
                    recommended_shares=shares,
                    dollar_amount=position_size,
                    entry_price=signal.target_price,
                    stop_loss=signal.stop_loss,
                    target_price=signal.target_price,
                    confidence=signal.confidence,
                    reasoning=f"NEW position: {signal.reasoning}",
                    priority=1 if signal.confidence > 0.8 else 2,
                    timeframe=signal.timeframe
                )
                
                recommendations.append(recommendation)
                available_cash -= position_size
                position_count += 1
                processed_symbols.add(signal.symbol)
            
            # Sort recommendations by priority
            recommendations.sort(key=lambda x: (x.priority, -x.confidence))
            
            # Generate portfolio summary and outlook
            portfolio_summary_text = self._generate_portfolio_summary(
                portfolio_summary, market_analysis, recommendations
            )
            
            market_outlook = self._generate_market_outlook(market_analysis)
            risk_assessment = self._generate_risk_assessment(market_analysis, recommendations)
            
            recommendation = PortfolioRecommendation(
                analysis_date=analysis_date,
                current_portfolio_value=current_portfolio_value,
                recommended_allocation=target_allocation,
                cash_reserve=portfolio_summary['cash_reserve'],
                recommendations=recommendations,
                portfolio_summary=portfolio_summary_text,
                market_outlook=market_outlook,
                risk_assessment=risk_assessment
            )
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return PortfolioRecommendation(
                analysis_date=datetime.now().strftime("%Y-%m-%d"),
                current_portfolio_value=0,
                recommended_allocation=0.5,
                cash_reserve=0,
                recommendations=[],
                portfolio_summary="Error generating recommendations",
                market_outlook="Unable to analyze market",
                risk_assessment="High risk due to system error"
            )
    
    def _generate_portfolio_summary(self, portfolio_summary: Dict, 
                                  market_analysis: MarketAnalysis,
                                  recommendations: List[TradingRecommendation]) -> str:
        """Generate human-readable portfolio summary."""
        try:
            lines = []
            lines.append(f"Portfolio Value: ${portfolio_summary['total_value']:,.2f}")
            lines.append(f"Cash Reserve: ${portfolio_summary['cash_reserve']:,.2f} "
                        f"({portfolio_summary['cash_allocation_pct']:.1f}%)")
            lines.append(f"Invested: ${portfolio_summary['invested_value']:,.2f}")
            lines.append(f"Unrealized P&L: ${portfolio_summary['unrealized_pnl']:,.2f} "
                        f"({portfolio_summary['total_return_pct']:+.2f}%)")
            lines.append("")
            
            buy_count = len([r for r in recommendations if r.action == 'BUY'])
            sell_count = len([r for r in recommendations if r.action == 'SELL'])
            hold_count = len([r for r in recommendations if r.action == 'HOLD'])
            
            lines.append(f"Today's Actions: {buy_count} BUY, {sell_count} SELL, {hold_count} HOLD")
            lines.append(f"Market Regime: {market_analysis.market_regime.replace('_', ' ').title()} "
                        f"({market_analysis.regime_confidence:.0%} confidence)")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return "Error generating summary"
    
    def _generate_market_outlook(self, market_analysis: MarketAnalysis) -> str:
        """Generate market outlook description."""
        regime_descriptions = {
            'bull_trend': "Strong upward momentum expected to continue. Favorable for growth stocks.",
            'bear_trend': "Downward pressure likely to persist. Focus on defensive positions and cash.",
            'high_volatility': "Expect sharp price swings. Quick profit-taking recommended.",
            'sideways': "Range-bound market. Look for mean reversion opportunities."
        }
        
        base_outlook = regime_descriptions.get(market_analysis.market_regime, 
                                             "Market direction unclear")
        
        return f"{base_outlook} Market sentiment is {market_analysis.market_sentiment}. " \
               f"Generated {len(market_analysis.signals)} actionable signals."
    
    def _generate_risk_assessment(self, market_analysis: MarketAnalysis, 
                                 recommendations: List[TradingRecommendation]) -> str:
        """Generate risk assessment."""
        risk_factors = []
        
        if market_analysis.risk_level == "high":
            risk_factors.append("High market volatility detected")
        
        if market_analysis.regime_confidence < 0.6:
            risk_factors.append("Uncertain market regime")
        
        high_confidence_trades = len([r for r in recommendations if r.confidence > 0.8])
        if high_confidence_trades < len(recommendations) * 0.5:
            risk_factors.append("Lower confidence in trading signals")
        
        if not risk_factors:
            return f"Risk Level: {market_analysis.risk_level.upper()}. Favorable conditions for trading."
        else:
            return f"Risk Level: {market_analysis.risk_level.upper()}. Factors: {'; '.join(risk_factors)}."
    
    def format_recommendations_report(self, portfolio_rec: PortfolioRecommendation) -> str:
        """Format recommendations as human-readable report."""
        try:
            lines = []
            lines.append("=" * 60)
            lines.append("AI SEMICONDUCTOR TRADING ADVISOR")
            lines.append("=" * 60)
            lines.append(f"Date: {portfolio_rec.analysis_date}")
            lines.append(f"Recommended Allocation: {portfolio_rec.recommended_allocation:.0%}")
            lines.append("")
            
            lines.append("PORTFOLIO STATUS:")
            lines.append("-" * 30)
            lines.append(portfolio_rec.portfolio_summary)
            lines.append("")
            
            lines.append("MARKET OUTLOOK:")
            lines.append("-" * 30)
            lines.append(portfolio_rec.market_outlook)
            lines.append("")
            
            lines.append("RISK ASSESSMENT:")
            lines.append("-" * 30)  
            lines.append(portfolio_rec.risk_assessment)
            lines.append("")
            
            if portfolio_rec.recommendations:
                lines.append("TODAY'S ACTIONS:")
                lines.append("-" * 30)
                
                for i, rec in enumerate(portfolio_rec.recommendations, 1):
                    action_symbol = {
                        'BUY': '🔵 BUY',
                        'SELL': '🔴 SELL', 
                        'HOLD': '⚪ HOLD',
                        'ADD': '🟡 ADD',
                        'REDUCE': '🟠 REDUCE'
                    }.get(rec.action, rec.action)
                    
                    lines.append(f"{i}. {action_symbol} {rec.symbol}")
                    
                    if rec.action in ['BUY', 'ADD']:
                        lines.append(f"   Amount: ${rec.dollar_amount:,.0f} "
                                   f"({rec.recommended_shares:.1f} shares)")
                        lines.append(f"   Entry: ${rec.entry_price:.2f} | "
                                   f"Target: ${rec.target_price:.2f} | "
                                   f"Stop: ${rec.stop_loss:.2f}")
                    elif rec.action == 'SELL':
                        lines.append(f"   Sell: {rec.current_position:.1f} shares "
                                   f"(${rec.dollar_amount:,.0f})")
                    elif rec.action == 'HOLD':
                        lines.append(f"   Hold: {rec.current_position:.1f} shares | "
                                   f"Stop: ${rec.stop_loss:.2f}")
                    
                    lines.append(f"   Confidence: {rec.confidence:.0%} | "
                               f"Timeframe: {rec.timeframe}")
                    lines.append(f"   Reason: {rec.reasoning[:100]}{'...' if len(rec.reasoning) > 100 else ''}")
                    lines.append("")
                
            lines.append("=" * 60)
            lines.append("Execute trades manually and update portfolio when complete.")
            lines.append("=" * 60)
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error formatting report: {e}")
            return f"Error formatting recommendations report: {e}"


if __name__ == "__main__":
    # Test the recommendation engine
    engine = TradingRecommendationEngine("test_portfolio.toml")
    recommendations = engine.generate_recommendations()
    
    print(engine.format_recommendations_report(recommendations))