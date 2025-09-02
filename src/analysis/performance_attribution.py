"""Performance attribution system for separating alpha from beta."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ..data.ingester import DataIngester
from ..config.settings import config


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    
    # Risk-adjusted metrics
    sortino_ratio: float
    information_ratio: float
    treynor_ratio: float
    
    # Attribution metrics
    alpha: float
    beta: float
    r_squared: float
    tracking_error: float
    
    # Factor exposures
    market_return: float
    factor_returns: Dict[str, float]
    attribution_breakdown: Dict[str, float]


@dataclass
class FactorExposure:
    """Factor exposure analysis."""
    factor_name: str
    exposure: float  # Beta to factor
    t_statistic: float
    p_value: float
    contribution: float  # Contribution to total return
    is_significant: bool


class PerformanceAttributor:
    """Performance attribution system using factor models."""
    
    def __init__(self, benchmark_ticker: str = "SPY"):
        """Initialize performance attributor.
        
        Args:
            benchmark_ticker: Ticker for market benchmark
        """
        self.benchmark_ticker = benchmark_ticker
        self.data_ingester = DataIngester(tickers=[benchmark_ticker])
        
        # Common factor tickers for multi-factor model
        self.factor_tickers = {
            'market': 'SPY',      # S&P 500 (market factor)
            'tech': 'QQQ',        # Technology factor  
            'value': 'VTV',       # Value factor
            'growth': 'VUG',      # Growth factor
            'small_cap': 'VB',    # Small cap factor
            'momentum': 'MTUM',   # Momentum factor
            'quality': 'QUAL',    # Quality factor
            'volatility': 'VIX'   # Volatility factor (inverted)
        }
        
        self.factor_data = {}
        self.benchmark_data = None
        
        logger.info(f"Performance attributor initialized with benchmark: {benchmark_ticker}")
    
    def load_factor_data(self, start_date: str, end_date: str) -> None:
        """Load factor and benchmark data for attribution analysis.
        
        Args:
            start_date: Start date for data loading
            end_date: End date for data loading
        """
        logger.info("Loading factor data for performance attribution...")
        
        # Load benchmark data
        try:
            benchmark_ingester = DataIngester(tickers=[self.benchmark_ticker])
            benchmark_data = benchmark_ingester.fetch_historical_data(
                start_date=start_date,
                end_date=end_date,
                interval='1d'
            )
            self.benchmark_data = benchmark_data.get(self.benchmark_ticker)
            logger.info(f"Loaded benchmark data: {len(self.benchmark_data)} records")
        except Exception as e:
            logger.error(f"Failed to load benchmark data: {e}")
            return
        
        # Load factor data
        all_factor_tickers = list(self.factor_tickers.values())
        factor_ingester = DataIngester(tickers=all_factor_tickers)
        
        try:
            factor_data = factor_ingester.fetch_historical_data(
                start_date=start_date,
                end_date=end_date,
                interval='1d'
            )
            
            # Convert to factor returns
            for factor_name, ticker in self.factor_tickers.items():
                if ticker in factor_data:
                    factor_df = factor_data[ticker]
                    
                    # Handle VIX differently (volatility index)
                    if factor_name == 'volatility':
                        # VIX is already in percentage points, convert to returns
                        vix_returns = factor_df['close'].pct_change()
                        # Invert VIX returns (negative volatility exposure is good)
                        self.factor_data[factor_name] = -vix_returns
                    else:
                        # Regular ETF returns
                        self.factor_data[factor_name] = factor_df['close'].pct_change()
                    
                    logger.debug(f"Loaded factor data for {factor_name}: {len(self.factor_data[factor_name])} returns")
                else:
                    logger.warning(f"Could not load factor data for {factor_name} ({ticker})")
        
        except Exception as e:
            logger.error(f"Failed to load factor data: {e}")
    
    def calculate_performance_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics.
        
        Args:
            portfolio_returns: Daily returns of the portfolio
            benchmark_returns: Daily returns of benchmark (optional)
            
        Returns:
            PerformanceMetrics object with complete analysis
        """
        if benchmark_returns is None and self.benchmark_data is not None:
            benchmark_returns = self.benchmark_data['close'].pct_change().dropna()
        
        # Align returns if benchmark is provided
        if benchmark_returns is not None:
            # Align dates
            aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
            aligned_returns.columns = ['portfolio', 'benchmark']
            portfolio_returns = aligned_returns['portfolio']
            benchmark_returns = aligned_returns['benchmark']
        
        # Basic return metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        risk_free_rate = 0.02  # 2% annual risk-free rate
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Downside deviation for Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Drawdown metrics
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trading metrics (simplified)
        positive_days = (portfolio_returns > 0).sum()
        total_days = len(portfolio_returns)
        win_rate = positive_days / total_days if total_days > 0 else 0
        
        # Profit factor (simplified)
        positive_returns = portfolio_returns[portfolio_returns > 0].sum()
        negative_returns = abs(portfolio_returns[portfolio_returns < 0].sum())
        profit_factor = positive_returns / negative_returns if negative_returns > 0 else 0
        
        # Benchmark-relative metrics
        alpha = beta = r_squared = information_ratio = treynor_ratio = tracking_error = 0
        market_return = 0
        
        if benchmark_returns is not None and len(benchmark_returns) > 10:
            try:
                # Calculate beta and alpha using linear regression
                covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
                
                benchmark_annualized = (1 + benchmark_returns).prod() ** (252 / len(benchmark_returns)) - 1
                alpha = annualized_return - (risk_free_rate + beta * (benchmark_annualized - risk_free_rate))
                
                # R-squared from regression
                correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
                r_squared = correlation ** 2 if not np.isnan(correlation) else 0
                
                # Tracking error
                excess_returns = portfolio_returns - benchmark_returns
                tracking_error = excess_returns.std() * np.sqrt(252)
                
                # Information ratio
                information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
                
                # Treynor ratio
                treynor_ratio = (annualized_return - risk_free_rate) / beta if beta != 0 else 0
                
                market_return = benchmark_annualized
                
            except Exception as e:
                logger.warning(f"Error calculating benchmark-relative metrics: {e}")
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sortino_ratio=sortino_ratio,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            alpha=alpha,
            beta=beta,
            r_squared=r_squared,
            tracking_error=tracking_error,
            market_return=market_return,
            factor_returns={},
            attribution_breakdown={}
        )
    
    def run_factor_attribution(
        self,
        portfolio_returns: pd.Series,
        factors_to_include: Optional[List[str]] = None
    ) -> Tuple[PerformanceMetrics, Dict[str, FactorExposure]]:
        """Run multi-factor attribution analysis.
        
        Args:
            portfolio_returns: Daily returns of the portfolio
            factors_to_include: List of factors to include in analysis
            
        Returns:
            Tuple of (PerformanceMetrics, factor_exposures_dict)
        """
        if factors_to_include is None:
            factors_to_include = ['market', 'tech', 'value', 'growth', 'small_cap']
        
        # Get available factors
        available_factors = {name: returns for name, returns in self.factor_data.items()
                           if name in factors_to_include and len(returns) > 10}
        
        if not available_factors:
            logger.warning("No factor data available for attribution analysis")
            metrics = self.calculate_performance_metrics(portfolio_returns)
            return metrics, {}
        
        # Align all data
        factor_df = pd.DataFrame(available_factors)
        combined_data = pd.concat([portfolio_returns, factor_df], axis=1, join='inner')
        combined_data.columns = ['portfolio'] + list(available_factors.keys())
        
        if len(combined_data) < 30:
            logger.warning("Insufficient aligned data for factor analysis")
            metrics = self.calculate_performance_metrics(portfolio_returns)
            return metrics, {}
        
        portfolio_aligned = combined_data['portfolio']
        factor_aligned = combined_data[list(available_factors.keys())]
        
        # Run multi-factor regression
        factor_exposures = {}
        
        try:
            # Prepare regression data
            X = factor_aligned.values
            y = portfolio_aligned.values
            
            # Add constant for alpha
            X_with_const = np.column_stack([np.ones(len(X)), X])
            
            # Run regression
            coefficients = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            
            alpha_estimate = coefficients[0]
            factor_betas = coefficients[1:]
            
            # Calculate residuals and statistics
            predicted = X_with_const @ coefficients
            residuals = y - predicted
            
            # Standard errors
            residual_variance = np.var(residuals, ddof=len(coefficients))
            X_cov = np.linalg.inv(X_with_const.T @ X_with_const) * residual_variance
            standard_errors = np.sqrt(np.diag(X_cov))
            
            # T-statistics and p-values
            t_stats = coefficients / standard_errors
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(y) - len(coefficients)))
            
            # Create factor exposure objects
            for i, factor_name in enumerate(available_factors.keys()):
                beta = factor_betas[i]
                t_stat = t_stats[i + 1]  # +1 because first is alpha
                p_val = p_values[i + 1]
                
                # Calculate contribution to returns
                factor_return = factor_aligned[factor_name].mean() * 252  # Annualized
                contribution = beta * factor_return
                
                factor_exposures[factor_name] = FactorExposure(
                    factor_name=factor_name,
                    exposure=beta,
                    t_statistic=t_stat,
                    p_value=p_val,
                    contribution=contribution,
                    is_significant=p_val < 0.05
                )
                
                logger.debug(f"Factor {factor_name}: Beta={beta:.3f}, t-stat={t_stat:.2f}, p-val={p_val:.3f}")
            
            # Calculate performance metrics with attribution
            metrics = self.calculate_performance_metrics(portfolio_aligned)
            
            # Add factor-specific information
            factor_returns = {name: factor_aligned[name].mean() * 252 
                            for name in available_factors.keys()}
            attribution_breakdown = {name: exp.contribution 
                                   for name, exp in factor_exposures.items()}
            
            # Add alpha from multi-factor model
            attribution_breakdown['alpha'] = alpha_estimate * 252  # Annualized alpha
            
            # Update metrics
            metrics.factor_returns = factor_returns
            metrics.attribution_breakdown = attribution_breakdown
            
            # Use multi-factor alpha instead of single-factor alpha
            metrics.alpha = alpha_estimate * 252
            
            logger.info(f"Factor attribution completed with {len(factor_exposures)} factors")
            
        except Exception as e:
            logger.error(f"Factor attribution failed: {e}")
            # Fall back to basic metrics
            metrics = self.calculate_performance_metrics(portfolio_aligned)
        
        return metrics, factor_exposures
    
    def generate_attribution_report(
        self,
        portfolio_returns: pd.Series,
        portfolio_name: str = "Portfolio",
        include_factors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive performance attribution report.
        
        Args:
            portfolio_returns: Daily returns of the portfolio
            portfolio_name: Name of the portfolio for reporting
            include_factors: List of factors to include in analysis
            
        Returns:
            Dictionary containing complete attribution analysis
        """
        logger.info(f"Generating attribution report for {portfolio_name}...")
        
        # Run factor attribution
        metrics, factor_exposures = self.run_factor_attribution(
            portfolio_returns, include_factors
        )
        
        # Calculate additional risk metrics
        portfolio_value_series = (1 + portfolio_returns).cumprod()
        
        # Risk metrics
        var_95 = portfolio_returns.quantile(0.05)
        var_99 = portfolio_returns.quantile(0.01)
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
        
        # Maximum consecutive losses
        losses = (portfolio_returns < 0).astype(int)
        max_consecutive_losses = losses.groupby((losses == 0).cumsum()).sum().max()
        
        # Performance periods analysis
        monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        positive_months = (monthly_returns > 0).sum()
        total_months = len(monthly_returns)
        monthly_win_rate = positive_months / total_months if total_months > 0 else 0
        
        report = {
            'portfolio_name': portfolio_name,
            'analysis_period': {
                'start_date': str(portfolio_returns.index.min().date()),
                'end_date': str(portfolio_returns.index.max().date()),
                'total_days': len(portfolio_returns),
                'trading_days': len(portfolio_returns[portfolio_returns != 0])
            },
            'performance_metrics': {
                'total_return': metrics.total_return,
                'annualized_return': metrics.annualized_return,
                'volatility': metrics.volatility,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'max_drawdown': metrics.max_drawdown,
                'calmar_ratio': metrics.calmar_ratio,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor
            },
            'risk_metrics': {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'tracking_error': metrics.tracking_error,
                'max_consecutive_losses': int(max_consecutive_losses),
                'monthly_win_rate': monthly_win_rate
            },
            'attribution_metrics': {
                'alpha': metrics.alpha,
                'beta': metrics.beta,
                'r_squared': metrics.r_squared,
                'information_ratio': metrics.information_ratio,
                'treynor_ratio': metrics.treynor_ratio,
                'market_return': metrics.market_return
            },
            'factor_exposures': {
                name: {
                    'exposure': exp.exposure,
                    't_statistic': exp.t_statistic,
                    'p_value': exp.p_value,
                    'contribution': exp.contribution,
                    'is_significant': exp.is_significant
                } for name, exp in factor_exposures.items()
            },
            'attribution_breakdown': metrics.attribution_breakdown,
            'factor_returns': metrics.factor_returns,
            'summary': {
                'outperformed_benchmark': metrics.alpha > 0,
                'significant_alpha': abs(metrics.alpha) > 0.02,  # 2% threshold
                'market_neutral': abs(metrics.beta) < 0.3,
                'diversified': metrics.r_squared < 0.7,
                'factor_exposures_count': len([exp for exp in factor_exposures.values() if exp.is_significant])
            }
        }
        
        # Log key findings
        self._log_attribution_summary(report)
        
        return report
    
    def _log_attribution_summary(self, report: Dict[str, Any]) -> None:
        """Log key findings from attribution analysis."""
        name = report['portfolio_name']
        perf = report['performance_metrics']
        attr = report['attribution_metrics']
        
        logger.info(f"=== PERFORMANCE ATTRIBUTION: {name} ===")
        logger.info(f"Total Return: {perf['total_return']:.2%}")
        logger.info(f"Annualized Return: {perf['annualized_return']:.2%}")
        logger.info(f"Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
        logger.info(f"Max Drawdown: {perf['max_drawdown']:.2%}")
        logger.info("")
        logger.info("ATTRIBUTION ANALYSIS:")
        logger.info(f"Alpha: {attr['alpha']:.2%}")
        logger.info(f"Beta: {attr['beta']:.3f}")
        logger.info(f"R-squared: {attr['r_squared']:.3f}")
        logger.info(f"Information Ratio: {attr['information_ratio']:.3f}")
        
        if report['factor_exposures']:
            logger.info("")
            logger.info("SIGNIFICANT FACTOR EXPOSURES:")
            for name, exp in report['factor_exposures'].items():
                if exp['is_significant']:
                    logger.info(f"{name.title()}: {exp['exposure']:.3f} "
                              f"(contribution: {exp['contribution']:.2%})")
        
        summary = report['summary']
        logger.info("")
        logger.info("SUMMARY:")
        logger.info(f"Outperformed Benchmark: {summary['outperformed_benchmark']}")
        logger.info(f"Significant Alpha: {summary['significant_alpha']}")
        logger.info(f"Market Neutral: {summary['market_neutral']}")
        logger.info("=" * 50)


def analyze_portfolio_attribution(
    portfolio_returns: pd.Series,
    benchmark_ticker: str = "SPY",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    portfolio_name: str = "Strategy"
) -> Dict[str, Any]:
    """Convenience function for running performance attribution analysis.
    
    Args:
        portfolio_returns: Daily returns of the portfolio
        benchmark_ticker: Benchmark ticker for comparison
        start_date: Start date for factor data loading
        end_date: End date for factor data loading  
        portfolio_name: Name for reporting
        
    Returns:
        Complete attribution analysis report
    """
    # Default date range if not provided
    if start_date is None:
        start_date = (portfolio_returns.index.min() - timedelta(days=30)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = portfolio_returns.index.max().strftime('%Y-%m-%d')
    
    # Initialize attributor
    attributor = PerformanceAttributor(benchmark_ticker=benchmark_ticker)
    
    # Load factor data
    attributor.load_factor_data(start_date, end_date)
    
    # Generate report
    report = attributor.generate_attribution_report(
        portfolio_returns=portfolio_returns,
        portfolio_name=portfolio_name
    )
    
    return report


if __name__ == "__main__":
    # Example usage
    logger.info("Running performance attribution example...")
    
    # Generate sample portfolio returns
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    
    # Simulate portfolio with some alpha and market exposure
    market_returns = np.random.normal(0.0008, 0.015, 252)  # ~20% annual return, 15% volatility
    alpha_component = np.random.normal(0.0002, 0.005, 252)  # ~5% annual alpha
    portfolio_returns = pd.Series(
        0.7 * market_returns + 0.3 * alpha_component,  # 70% market exposure + alpha
        index=dates
    )
    
    # Run attribution analysis
    report = analyze_portfolio_attribution(
        portfolio_returns=portfolio_returns,
        portfolio_name="Sample Strategy"
    )
    
    logger.info("Attribution analysis completed successfully!")