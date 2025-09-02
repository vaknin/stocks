"""
Advanced Sector Rotation Analysis for AI Trading System
Implements institutional-grade sector momentum and rotation strategies for revenue generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
import yfinance as yf
from scipy.stats import percentileofscore


@dataclass
class SectorMomentum:
    """Sector momentum analysis results."""
    sector_name: str
    momentum_score: float  # 0-100 percentile
    relative_strength: float  # vs SPY
    volatility_adjusted_return: float
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    allocation_weight: float  # 0.0-2.0 multiplier
    top_performers: List[str]  # Best stocks in sector
    recommendation: str  # 'overweight', 'neutral', 'underweight'


@dataclass 
class RotationSignal:
    """Sector rotation trading signal."""
    from_sector: str
    to_sector: str
    strength: float  # Signal strength 0-1
    expected_duration: int  # Expected holding period in days
    confidence: float
    rebalance_pct: float  # Percentage of portfolio to rotate


class SectorRotationAnalyzer:
    """Advanced sector rotation analysis using institutional techniques."""
    
    def __init__(self):
        logger.info("Initializing SectorRotationAnalyzer")
        
        # Sector ETF proxies for rotation analysis
        self.sector_etfs = {
            'technology': 'XLK',
            'healthcare': 'XLV', 
            'financials': 'XLF',
            'consumer_discretionary': 'XLY',
            'consumer_staples': 'XLP',
            'energy': 'XLE',
            'utilities': 'XLU',
            'industrials': 'XLI',
            'materials': 'XLB',
            'real_estate': 'XLRE',
            'telecommunications': 'XLC'
        }
        
        # Individual stocks by sector (from market analyzer)
        self.sector_stocks = {
            'technology': ["NVDA", "AMD", "ASML", "TSM", "INTC", "QCOM", "AVGO", "MU", "SMCI", "ARM",
                         "AAPL", "MSFT", "GOOGL", "META", "CRM", "ADBE", "NFLX", "ORCL", "PYPL"],
            'healthcare': ["JNJ", "PFE", "UNH", "ABBV", "TMO", "AMGN", "GILD", "MRNA", "BMY", "LLY"],
            'financials': ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "BRK-B", "AXP", "C"],
            'consumer_discretionary': ["AMZN", "TSLA", "HD", "MCD", "SBUX", "NKE", "TGT", "LOW", "DIS", "BKNG"],
            'consumer_staples': ["KO", "PG", "WMT", "PEP", "COST", "CL", "KMB", "GIS"],
            'energy': ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO", "MPC"],
            'utilities': ["NEE", "DUK", "SO", "D", "EXC", "SRE"],
            'industrials': ["CAT", "BA", "GE", "UNP", "HON", "MMM"],
            'international': ["SHOP", "SAP"]
        }
        
        # Risk-adjusted allocation bounds by market regime
        self.regime_bounds = {
            'bull_trend': {'min_weight': 0.5, 'max_weight': 2.0, 'volatility_target': 0.15},
            'bear_trend': {'min_weight': 0.3, 'max_weight': 1.2, 'volatility_target': 0.10}, 
            'high_volatility': {'min_weight': 0.2, 'max_weight': 1.5, 'volatility_target': 0.12},
            'sideways': {'min_weight': 0.6, 'max_weight': 1.4, 'volatility_target': 0.13}
        }
        
        logger.info(f"Loaded {len(self.sector_etfs)} sector ETFs for rotation analysis")
        
    def fetch_sector_data(self, lookback_days: int = 252) -> Dict[str, pd.DataFrame]:
        """Fetch sector ETF and stock data for analysis."""
        try:
            logger.info("Fetching sector rotation data")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            sector_data = {}
            
            # Fetch sector ETF data
            for sector, etf in self.sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf)
                    data = ticker.history(
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        interval='1d'
                    )
                    
                    if not data.empty and len(data) > 50:
                        # Standardize column names
                        data = data.rename(columns={
                            'Open': 'open', 'High': 'high', 'Low': 'low', 
                            'Close': 'close', 'Volume': 'volume'
                        })
                        sector_data[f"{sector}_etf"] = data[['open', 'high', 'low', 'close', 'volume']]
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch {sector} ETF data: {e}")
            
            # Fetch benchmark data (SPY)
            try:
                spy = yf.Ticker("SPY")
                spy_data = spy.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d'
                )
                
                if not spy_data.empty:
                    spy_data = spy_data.rename(columns={
                        'Open': 'open', 'High': 'high', 'Low': 'low',
                        'Close': 'close', 'Volume': 'volume'
                    })
                    sector_data['benchmark'] = spy_data[['open', 'high', 'low', 'close', 'volume']]
                    
            except Exception as e:
                logger.warning(f"Failed to fetch SPY benchmark data: {e}")
            
            logger.info(f"Fetched data for {len(sector_data)} sectors/benchmarks")
            return sector_data
            
        except Exception as e:
            logger.error(f"Error fetching sector data: {e}")
            return {}
    
    def calculate_sector_momentum(self, sector_data: Dict[str, pd.DataFrame]) -> List[SectorMomentum]:
        """Calculate momentum scores for all sectors using institutional techniques."""
        try:
            logger.info("Calculating sector momentum scores")
            
            if 'benchmark' not in sector_data:
                logger.error("Benchmark data not available")
                return []
            
            benchmark_returns = sector_data['benchmark']['close'].pct_change().dropna()
            momentum_results = []
            
            # Calculate momentum for each sector
            for sector_key in sector_data.keys():
                if sector_key == 'benchmark' or not sector_key.endswith('_etf'):
                    continue
                
                sector_name = sector_key.replace('_etf', '')
                sector_data_df = sector_data[sector_key]
                
                if len(sector_data_df) < 60:  # Need at least 3 months of data
                    continue
                
                try:
                    # Calculate returns
                    sector_returns = sector_data_df['close'].pct_change().dropna()
                    
                    # Align with benchmark for comparison
                    aligned_data = pd.concat([sector_returns, benchmark_returns], axis=1, join='inner')
                    aligned_data.columns = ['sector', 'benchmark']
                    
                    if len(aligned_data) < 30:
                        continue
                    
                    # Multiple timeframe momentum (institutional approach)
                    momentum_1m = aligned_data['sector'].tail(21).mean() * 21  # 1 month annualized
                    momentum_3m = aligned_data['sector'].tail(63).mean() * 252  # 3 month annualized  
                    momentum_6m = aligned_data['sector'].tail(126).mean() * 252  # 6 month annualized
                    
                    # Weighted momentum score (emphasis on recent performance)
                    composite_momentum = (momentum_1m * 0.5 + momentum_3m * 0.3 + momentum_6m * 0.2)
                    
                    # Relative strength vs benchmark
                    relative_strength = (aligned_data['sector'] - aligned_data['benchmark']).tail(63).mean() * 252
                    
                    # Risk-adjusted momentum (Sharpe-like)
                    volatility = aligned_data['sector'].tail(63).std() * np.sqrt(252)
                    risk_adjusted_momentum = composite_momentum / max(volatility, 0.05)  # Floor volatility at 5%
                    
                    # Trend direction analysis
                    recent_trend = sector_data_df['close'].tail(21)
                    trend_slope = np.polyfit(range(len(recent_trend)), recent_trend.values, 1)[0]
                    
                    if trend_slope > recent_trend.mean() * 0.001:  # 0.1% of price per day
                        trend_direction = 'bullish'
                    elif trend_slope < -recent_trend.mean() * 0.001:
                        trend_direction = 'bearish'
                    else:
                        trend_direction = 'neutral'
                    
                    # Get top performing stocks in this sector
                    top_performers = self._get_sector_top_performers(sector_name)
                    
                    # Generate allocation weight (0.5 to 2.0x)
                    momentum_percentile = percentileofscore([m.get('risk_adjusted_momentum', 0) for m in 
                                                           [{'risk_adjusted_momentum': risk_adjusted_momentum}] + 
                                                           [r.__dict__ if hasattr(r, '__dict__') else {'risk_adjusted_momentum': 0} 
                                                            for r in momentum_results]], 
                                                           risk_adjusted_momentum)
                    
                    allocation_weight = 0.5 + (momentum_percentile / 100) * 1.5  # Scale 0.5-2.0
                    
                    # Determine recommendation
                    if momentum_percentile > 75 and relative_strength > 0.02:  # Top quartile + 2% outperformance
                        recommendation = 'overweight'
                    elif momentum_percentile < 25 or relative_strength < -0.02:  # Bottom quartile or underperforming
                        recommendation = 'underweight'
                    else:
                        recommendation = 'neutral'
                    
                    momentum_result = SectorMomentum(
                        sector_name=sector_name,
                        momentum_score=momentum_percentile,
                        relative_strength=relative_strength,
                        volatility_adjusted_return=risk_adjusted_momentum,
                        trend_direction=trend_direction,
                        allocation_weight=allocation_weight,
                        top_performers=top_performers,
                        recommendation=recommendation
                    )
                    
                    momentum_results.append(momentum_result)
                    
                    logger.debug(f"{sector_name}: momentum={momentum_percentile:.1f}, relative_strength={relative_strength:.2%}, "
                               f"allocation={allocation_weight:.2f}x, recommendation={recommendation}")
                    
                except Exception as e:
                    logger.warning(f"Error calculating momentum for {sector_name}: {e}")
                    continue
            
            # Sort by momentum score
            momentum_results.sort(key=lambda x: x.momentum_score, reverse=True)
            
            logger.info(f"Calculated momentum for {len(momentum_results)} sectors")
            return momentum_results
            
        except Exception as e:
            logger.error(f"Error calculating sector momentum: {e}")
            return []
    
    def _get_sector_top_performers(self, sector_name: str, top_n: int = 3) -> List[str]:
        """Get top performing stocks in a sector based on recent momentum."""
        try:
            if sector_name not in self.sector_stocks:
                return []
            
            stocks = self.sector_stocks[sector_name][:top_n]  # For now, return first N stocks
            return stocks
            
        except Exception as e:
            logger.warning(f"Error getting top performers for {sector_name}: {e}")
            return []
    
    def generate_rotation_signals(self, momentum_analysis: List[SectorMomentum], 
                                current_regime: str = 'bull_trend') -> List[RotationSignal]:
        """Generate sector rotation signals based on momentum analysis."""
        try:
            logger.info("Generating sector rotation signals")
            
            if len(momentum_analysis) < 2:
                logger.warning("Insufficient sector data for rotation analysis")
                return []
            
            rotation_signals = []
            regime_bounds = self.regime_bounds.get(current_regime, self.regime_bounds['bull_trend'])
            
            # Sort sectors by momentum
            sorted_sectors = sorted(momentum_analysis, key=lambda x: x.momentum_score, reverse=True)
            
            # Identify rotation opportunities
            strong_sectors = [s for s in sorted_sectors[:3] if s.momentum_score > 70]  # Top 3, >70th percentile
            weak_sectors = [s for s in sorted_sectors[-3:] if s.momentum_score < 30]  # Bottom 3, <30th percentile
            
            # Generate rotation pairs
            for weak_sector in weak_sectors:
                for strong_sector in strong_sectors:
                    if weak_sector.sector_name == strong_sector.sector_name:
                        continue
                    
                    # Calculate rotation strength
                    momentum_diff = strong_sector.momentum_score - weak_sector.momentum_score
                    relative_strength_diff = strong_sector.relative_strength - weak_sector.relative_strength
                    
                    # Signal strength based on momentum gap and relative performance
                    signal_strength = min(1.0, (momentum_diff / 100) * 2 + abs(relative_strength_diff) * 5)
                    
                    # Confidence based on trend consistency
                    trend_alignment = 1.0 if strong_sector.trend_direction == 'bullish' and weak_sector.trend_direction == 'bearish' else 0.7
                    confidence = signal_strength * trend_alignment
                    
                    # Expected duration based on momentum persistence (academic research: 3-6 months)
                    expected_duration = int(90 + momentum_diff)  # 90-190 days based on momentum gap
                    
                    # Rebalance percentage (conservative approach)
                    max_rebalance = regime_bounds['max_weight'] * 10  # Convert to percentage
                    rebalance_pct = min(max_rebalance, signal_strength * 15)  # Max 15% rebalance
                    
                    # Only generate signals above minimum thresholds
                    if confidence > 0.6 and rebalance_pct > 2:  # 60% confidence, 2% minimum rebalance
                        rotation_signal = RotationSignal(
                            from_sector=weak_sector.sector_name,
                            to_sector=strong_sector.sector_name,
                            strength=signal_strength,
                            expected_duration=expected_duration,
                            confidence=confidence,
                            rebalance_pct=rebalance_pct
                        )
                        rotation_signals.append(rotation_signal)
            
            # Sort by confidence and strength
            rotation_signals.sort(key=lambda x: x.confidence * x.strength, reverse=True)
            
            # Limit to top 3 signals to avoid over-trading
            rotation_signals = rotation_signals[:3]
            
            logger.info(f"Generated {len(rotation_signals)} rotation signals")
            for signal in rotation_signals:
                logger.info(f"Rotation: {signal.from_sector} → {signal.to_sector} "
                          f"(strength: {signal.strength:.2f}, confidence: {signal.confidence:.1%}, "
                          f"rebalance: {signal.rebalance_pct:.1f}%)")
            
            return rotation_signals
            
        except Exception as e:
            logger.error(f"Error generating rotation signals: {e}")
            return []
    
    def analyze_sector_rotation(self, current_regime: str = 'bull_trend') -> Tuple[List[SectorMomentum], List[RotationSignal]]:
        """Perform comprehensive sector rotation analysis."""
        try:
            logger.info(f"Starting sector rotation analysis for regime: {current_regime}")
            
            # Fetch sector data
            sector_data = self.fetch_sector_data()
            if not sector_data:
                logger.error("No sector data available")
                return [], []
            
            # Calculate momentum
            momentum_analysis = self.calculate_sector_momentum(sector_data)
            if not momentum_analysis:
                logger.error("No momentum analysis available")
                return [], []
            
            # Generate rotation signals
            rotation_signals = self.generate_rotation_signals(momentum_analysis, current_regime)
            
            logger.info("Sector rotation analysis complete")
            return momentum_analysis, rotation_signals
            
        except Exception as e:
            logger.error(f"Error in sector rotation analysis: {e}")
            return [], []
    
    def print_rotation_summary(self, momentum_analysis: List[SectorMomentum], 
                             rotation_signals: List[RotationSignal]):
        """Print human-readable sector rotation summary."""
        print("=" * 70)
        print("SECTOR ROTATION ANALYSIS")
        print("=" * 70)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
        print()
        
        if momentum_analysis:
            print("SECTOR MOMENTUM RANKINGS:")
            print("-" * 70)
            print(f"{'Sector':<20} {'Momentum':<10} {'Rel Strength':<12} {'Weight':<8} {'Recommendation':<12}")
            print("-" * 70)
            
            for sector in momentum_analysis:
                print(f"{sector.sector_name:<20} "
                      f"{sector.momentum_score:>7.1f} "
                      f"{sector.relative_strength:>9.2%} "
                      f"{sector.allocation_weight:>6.2f}x "
                      f"{sector.recommendation:<12}")
            print()
        
        if rotation_signals:
            print("ROTATION SIGNALS:")
            print("-" * 70)
            print(f"{'From → To':<25} {'Strength':<10} {'Confidence':<12} {'Rebalance %':<12}")
            print("-" * 70)
            
            for signal in rotation_signals:
                rotation_pair = f"{signal.from_sector} → {signal.to_sector}"
                print(f"{rotation_pair:<25} "
                      f"{signal.strength:>7.2f} "
                      f"{signal.confidence:>9.1%} "
                      f"{signal.rebalance_pct:>9.1f}%")
            print()
        
        print("=" * 70)


if __name__ == "__main__":
    # Test sector rotation analyzer
    analyzer = SectorRotationAnalyzer()
    momentum, signals = analyzer.analyze_sector_rotation('bull_trend')
    analyzer.print_rotation_summary(momentum, signals)