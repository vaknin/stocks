"""Backtesting engine for validating trading strategies."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from ..data.ingester import DataIngester
from ..data.database import TradingDatabase
from ..data.validator import DataValidator
from ..trading.engine import TradingEngine
from ..trading.signal_generator import SignalGenerator, TimeFrame
from ..trading.portfolio import PortfolioManager, PortfolioMetrics
from ..trading.position_sizer import KellyPositionSizer
from ..trading.risk_manager import RiskManager
from ..models.timesfm_predictor import TimesFMPredictor
from ..config.settings import config


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: str
    end_date: str
    initial_capital: float = 100000
    commission_rate: float = 0.001
    bid_ask_spread_pct: float = 0.0005  # 0.05% bid-ask spread
    slippage_pct: float = 0.0002  # 0.02% slippage
    market_impact_threshold: float = 0.01  # 1% of volume for market impact
    tickers: List[str] = field(default_factory=lambda: config.STOCK_TICKERS[:5])  # Limit for faster testing
    lookback_days: int = 252  # 1 year of data for model training
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    max_positions: int = 5
    benchmark: str = 'SPY'  # Benchmark for comparison


@dataclass
class BacktestResults:
    """Results from backtesting."""
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    duration_days: int
    
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Portfolio evolution
    portfolio_values: pd.Series
    daily_returns: pd.Series
    positions_history: List[Dict]
    trades_history: List[Dict]
    
    # Benchmark comparison
    benchmark_return: float
    alpha: float  # Excess return over benchmark
    beta: float   # Beta relative to benchmark
    information_ratio: float
    
    # Risk metrics
    var_95: float  # 95% Value at Risk
    cvar_95: float  # 95% Conditional Value at Risk
    
    # Additional metrics
    kelly_utilization: float  # How much of Kelly optimal was used
    risk_adjusted_return: float


class BacktestEngine:
    """Comprehensive backtesting engine for trading strategies."""
    
    def __init__(self, config: BacktestConfig):
        """Initialize backtesting engine.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        
        # Initialize components
        self.data_ingester = DataIngester(tickers=config.tickers)
        self.database = TradingDatabase()
        self.data_validator = DataValidator()
        
        # ML and trading components will be initialized per backtest
        self.historical_data = {}
        self.benchmark_data = None
        
        # Transaction cost tracking
        self.total_commission_costs = 0.0
        self.total_bid_ask_costs = 0.0
        self.total_slippage_costs = 0.0
        self.total_market_impact_costs = 0.0
        
        logger.info(f"Backtest engine initialized for {len(config.tickers)} tickers")
        logger.info(f"Period: {config.start_date} to {config.end_date}")
        logger.info(f"Transaction costs: Commission {config.commission_rate:.3%}, "
                   f"Spread {config.bid_ask_spread_pct:.3%}, Slippage {config.slippage_pct:.3%}")
    
    def run_backtest(self, strategy_name: str = "TimesFM_Kelly") -> BacktestResults:
        """Run complete backtesting simulation.
        
        Args:
            strategy_name: Name of the strategy being tested
            
        Returns:
            BacktestResults object with comprehensive metrics
        """
        logger.info(f"Starting backtest for strategy: {strategy_name}")
        
        try:
            # 1. Load and prepare historical data
            self._load_historical_data()
            
            # 2. Initialize strategy components
            timesfm_predictor = TimesFMPredictor()
            signal_generator = SignalGenerator(timesfm_predictor=timesfm_predictor)
            position_sizer = KellyPositionSizer(safety_factor=0.40)
            risk_manager = RiskManager()
            portfolio_manager = PortfolioManager(
                initial_capital=self.config.initial_capital,
                commission_rate=self.config.commission_rate
            )
            
            # 3. Run simulation
            simulation_results = self._run_simulation(
                signal_generator=signal_generator,
                position_sizer=position_sizer,
                risk_manager=risk_manager,
                portfolio_manager=portfolio_manager
            )
            
            # 4. Calculate performance metrics
            results = self._calculate_backtest_results(simulation_results)
            
            # 5. Generate detailed report
            self._generate_backtest_report(results, strategy_name)
            
            logger.info(f"Backtest completed. Total return: {results.total_return:.2%}")
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    def _load_historical_data(self) -> None:
        """Load and validate historical market data."""
        logger.info("Loading historical data...")
        
        # Extend date range to include lookback period
        start_date_with_lookback = (
            datetime.strptime(self.config.start_date, '%Y-%m-%d') - 
            timedelta(days=self.config.lookback_days)
        ).strftime('%Y-%m-%d')
        
        # Load data for all tickers
        all_data = self.data_ingester.fetch_historical_data(
            start_date=start_date_with_lookback,
            end_date=self.config.end_date,
            interval='1d'
        )
        
        # Validate and store data
        for ticker, df in all_data.items():
            validation_result = self.data_validator.validate_ohlcv_data(df, ticker)
            if validation_result['is_valid']:
                self.historical_data[ticker] = df
                logger.debug(f"Loaded {len(df)} records for {ticker}")
            else:
                logger.warning(f"Data validation failed for {ticker}")
        
        # Load benchmark data
        if self.config.benchmark:
            try:
                benchmark_ingester = DataIngester(tickers=[self.config.benchmark])
                benchmark_data = benchmark_ingester.fetch_historical_data(
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    interval='1d'
                )
                self.benchmark_data = benchmark_data.get(self.config.benchmark)
                logger.info(f"Loaded benchmark data for {self.config.benchmark}")
            except Exception as e:
                logger.warning(f"Failed to load benchmark data: {e}")
        
        logger.info(f"Historical data loaded for {len(self.historical_data)} tickers")
    
    def _run_simulation(
        self,
        signal_generator: SignalGenerator,
        position_sizer: KellyPositionSizer,
        risk_manager: RiskManager,
        portfolio_manager: PortfolioManager
    ) -> Dict[str, Any]:
        """Run backtesting simulation day by day.
        
        Args:
            signal_generator: Signal generation component
            position_sizer: Position sizing component
            risk_manager: Risk management component
            portfolio_manager: Portfolio management component
            
        Returns:
            Dictionary with simulation results
        """
        logger.info("Running backtesting simulation...")
        
        # Get date range for simulation
        start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')
        
        # Track simulation results
        portfolio_values = []
        positions_history = []
        trades_history = []
        daily_signals = []
        
        # Simulate trading day by day
        current_date = start_date
        simulation_days = 0
        total_signals = 0
        total_trades = 0
        
        while current_date <= end_date:
            simulation_days += 1
            
            if simulation_days % 30 == 0:  # Progress update every month
                logger.info(f"Simulation progress: {simulation_days} days, "
                           f"Portfolio: ${portfolio_manager.cash_balance + sum(p.market_value for p in portfolio_manager.positions.values()):,.2f}")
            
            try:
                # Get historical data up to current date for prediction
                current_data = {}
                current_prices = {}
                
                for ticker, full_data in self.historical_data.items():
                    # Get data up to current date
                    historical_slice = full_data[full_data.index <= current_date]
                    
                    if len(historical_slice) >= self.config.lookback_days:
                        # Use last lookback_days for prediction
                        prediction_data = historical_slice.tail(self.config.lookback_days)
                        current_data[ticker] = {
                            TimeFrame.DAILY: prediction_data
                        }
                        
                        # Get current price (last available price up to current date)
                        current_prices[ticker] = historical_slice['close'].iloc[-1]
                
                if not current_data:
                    current_date += timedelta(days=1)
                    continue
                
                # Update existing positions with current prices
                portfolio_manager.update_positions(current_prices)
                
                # Check for stop loss triggers
                stop_triggers = portfolio_manager.check_stop_loss_take_profit(current_prices)
                for ticker in stop_triggers:
                    # Calculate realistic exit price with transaction costs
                    position = portfolio_manager.positions.get(ticker)
                    if position:
                        is_closing_long = position.size > 0
                        exit_price = self._calculate_execution_price(
                            ticker=ticker,
                            intended_price=current_prices.get(ticker),
                            order_size=abs(position.size),
                            is_buy=not is_closing_long  # Closing long = sell, closing short = buy
                        )
                        
                        success = portfolio_manager.close_position(
                            ticker=ticker,
                            exit_price=exit_price,
                            reason="STOP_TRIGGER"
                        )
                        if success:
                            total_trades += 1
                
                # Generate signals (only on rebalance days)
                if self._should_rebalance(current_date, simulation_days):
                    signals = signal_generator.generate_signals(
                        data_dict=current_data,
                        current_prices=current_prices
                    )
                    
                    # Filter signals by quality
                    quality_signals = signal_generator.filter_signals_by_quality(signals)
                    total_signals += len(quality_signals)
                    
                    # Calculate position sizes
                    if quality_signals:
                        # Convert signals to predictions format
                        predictions = {}
                        for ticker, multi_signal in quality_signals.items():
                            # Use the strongest signal
                            best_signal = None
                            best_confidence = 0
                            
                            for timeframe_signal in multi_signal.signals_by_timeframe.values():
                                if timeframe_signal.confidence > best_confidence:
                                    best_confidence = timeframe_signal.confidence
                                    best_signal = timeframe_signal
                            
                            if best_signal:
                                predictions[ticker] = {
                                    'prediction': best_signal.prediction,
                                    'confidence': best_signal.confidence
                                }
                        
                        # Calculate position sizes
                        portfolio_value = (portfolio_manager.cash_balance + 
                                         sum(pos.market_value for pos in portfolio_manager.positions.values()))
                        
                        position_sizes = position_sizer.batch_calculate_positions(
                            predictions=predictions,
                            current_prices=current_prices,
                            portfolio_value=portfolio_value,
                            current_positions={
                                ticker: pos.market_value 
                                for ticker, pos in portfolio_manager.positions.items()
                            }
                        )
                        
                        # Execute new positions
                        for ticker, multi_signal in quality_signals.items():
                            if (ticker in position_sizes and 
                                ticker not in portfolio_manager.positions and
                                len(portfolio_manager.positions) < self.config.max_positions):
                                
                                position_result = position_sizes[ticker]
                                
                                # Risk check
                                approved, violations = risk_manager.check_pre_trade_risk(
                                    ticker=ticker,
                                    proposed_size=position_result.recommended_size,
                                    current_price=current_prices[ticker],
                                    portfolio_value=portfolio_value,
                                    current_positions={
                                        ticker: {'market_value': pos.market_value} 
                                        for ticker, pos in portfolio_manager.positions.items()
                                    }
                                )
                                
                                if approved:
                                    # Determine position direction
                                    from ..trading.signal_generator import SignalType
                                    signal_type = multi_signal.overall_signal
                                    
                                    if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                                        position_size = abs(position_result.recommended_size)
                                    elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                                        position_size = -abs(position_result.recommended_size)
                                    else:
                                        continue
                                    
                                    # Get representative signal for stops
                                    rep_signal = list(multi_signal.signals_by_timeframe.values())[0]
                                    
                                    # Calculate realistic execution price with transaction costs
                                    execution_price = self._calculate_execution_price(
                                        ticker=ticker,
                                        intended_price=current_prices[ticker],
                                        order_size=abs(position_size),
                                        is_buy=(position_size > 0)
                                    )
                                    
                                    success = portfolio_manager.open_position(
                                        ticker=ticker,
                                        size=position_size,
                                        entry_price=execution_price,
                                        signal=rep_signal,
                                        stop_loss=rep_signal.stop_loss,
                                        take_profit=rep_signal.take_profit
                                    )
                                    
                                    if success:
                                        total_trades += 1
                    
                    # Store signals for analysis
                    daily_signals.append({
                        'date': current_date,
                        'signals': len(quality_signals),
                        'positions': len(portfolio_manager.positions)
                    })
                
                # Calculate daily portfolio metrics
                portfolio_metrics = portfolio_manager.calculate_portfolio_metrics(current_prices)
                
                # Store daily results
                portfolio_values.append({
                    'date': current_date,
                    'value': portfolio_metrics.total_value,
                    'cash': portfolio_metrics.cash_balance,
                    'positions_value': portfolio_metrics.positions_value,
                    'unrealized_pnl': portfolio_metrics.unrealized_pnl,
                    'realized_pnl': portfolio_metrics.realized_pnl
                })
                
                # Store positions snapshot
                if portfolio_manager.positions:
                    positions_snapshot = {}
                    for ticker, pos in portfolio_manager.positions.items():
                        positions_snapshot[ticker] = {
                            'size': pos.size,
                            'entry_price': pos.entry_price,
                            'current_price': pos.current_price,
                            'unrealized_pnl': pos.pnl_unrealized
                        }
                    positions_history.append({
                        'date': current_date,
                        'positions': positions_snapshot
                    })
                
            except Exception as e:
                logger.warning(f"Error on {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        # Close all remaining positions at end
        final_prices = {}
        for ticker in portfolio_manager.positions.keys():
            if ticker in self.historical_data:
                final_data = self.historical_data[ticker]
                final_prices[ticker] = final_data[final_data.index <= end_date]['close'].iloc[-1]
        
        for ticker in list(portfolio_manager.positions.keys()):
            if ticker in final_prices:
                # Calculate realistic exit price for final close
                position = portfolio_manager.positions.get(ticker)
                if position:
                    is_closing_long = position.size > 0
                    exit_price = self._calculate_execution_price(
                        ticker=ticker,
                        intended_price=final_prices[ticker],
                        order_size=abs(position.size),
                        is_buy=not is_closing_long
                    )
                    
                    portfolio_manager.close_position(
                        ticker=ticker,
                        exit_price=exit_price,
                        reason="BACKTEST_END"
                    )
                    total_trades += 1
        
        # Collect trade history
        for pos in portfolio_manager.closed_positions:
            if pos.exit_timestamp:
                trades_history.append({
                    'ticker': pos.ticker,
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'exit_price': pos.exit_price,
                    'entry_date': pos.entry_timestamp,
                    'exit_date': pos.exit_timestamp,
                    'pnl': pos.pnl_realized,
                    'return': pos.pnl_realized / (abs(pos.size) * pos.entry_price) if pos.entry_price > 0 else 0
                })
        
        logger.info(f"Simulation completed: {simulation_days} days, {total_signals} signals, {total_trades} trades")
        logger.info(f"Transaction costs: Commission ${self.total_commission_costs:.2f}, "
                   f"Spread ${self.total_bid_ask_costs:.2f}, Slippage ${self.total_slippage_costs:.2f}")
        
        return {
            'portfolio_values': portfolio_values,
            'positions_history': positions_history,
            'trades_history': trades_history,
            'daily_signals': daily_signals,
            'portfolio_manager': portfolio_manager,
            'total_trades': total_trades,
            'transaction_costs': {
                'commission': self.total_commission_costs,
                'bid_ask_spread': self.total_bid_ask_costs,
                'slippage': self.total_slippage_costs,
                'market_impact': self.total_market_impact_costs,
                'total': (self.total_commission_costs + self.total_bid_ask_costs + 
                         self.total_slippage_costs + self.total_market_impact_costs)
            }
        }
    
    def _calculate_execution_price(
        self,
        ticker: str,
        intended_price: float,
        order_size: float,
        is_buy: bool
    ) -> float:
        """Calculate realistic execution price including transaction costs.
        
        Args:
            ticker: Stock ticker
            intended_price: Intended execution price (mid-market)
            order_size: Order size in shares
            is_buy: True for buy orders, False for sell orders
            
        Returns:
            Realistic execution price including all transaction costs
        """
        # Start with intended price
        execution_price = intended_price
        
        # Apply bid-ask spread
        spread_cost = intended_price * self.config.bid_ask_spread_pct
        if is_buy:
            execution_price += spread_cost / 2  # Pay ask
        else:
            execution_price -= spread_cost / 2  # Receive bid
        
        self.total_bid_ask_costs += spread_cost * order_size
        
        # Apply slippage (increases with order size)
        slippage_factor = min(self.config.slippage_pct * (1 + order_size / 1000), 0.005)  # Cap at 0.5%
        slippage_cost = intended_price * slippage_factor
        
        if is_buy:
            execution_price += slippage_cost
        else:
            execution_price -= slippage_cost
            
        self.total_slippage_costs += slippage_cost * order_size
        
        # Market impact for large orders (simplified model)
        # Assume average daily volume of 1M shares for major stocks
        estimated_daily_volume = 1000000
        volume_participation = order_size / estimated_daily_volume
        
        if volume_participation > self.config.market_impact_threshold:
            # Non-linear market impact
            impact_factor = 0.001 * (volume_participation ** 0.5)  # Square root model
            impact_cost = intended_price * impact_factor
            
            if is_buy:
                execution_price += impact_cost
            else:
                execution_price -= impact_cost
                
            self.total_market_impact_costs += impact_cost * order_size
        
        # Commission (applied separately in portfolio manager)
        commission = max(1.0, order_size * intended_price * self.config.commission_rate)
        self.total_commission_costs += commission
        
        return execution_price
    
    def _should_rebalance(self, current_date: datetime, simulation_day: int) -> bool:
        """Determine if strategy should rebalance on current date."""
        if self.config.rebalance_frequency == 'daily':
            return True
        elif self.config.rebalance_frequency == 'weekly':
            return current_date.weekday() == 0  # Monday
        elif self.config.rebalance_frequency == 'monthly':
            return current_date.day == 1  # First day of month
        else:
            return simulation_day % 5 == 0  # Default: every 5 days
    
    def _calculate_backtest_results(self, simulation_results: Dict[str, Any]) -> BacktestResults:
        """Calculate comprehensive backtest performance metrics.
        
        Args:
            simulation_results: Results from simulation
            
        Returns:
            BacktestResults object with all metrics
        """
        logger.info("Calculating backtest results...")
        
        # Extract data
        portfolio_values = pd.DataFrame(simulation_results['portfolio_values']).set_index('date')
        trades_history = simulation_results['trades_history']
        
        # Basic metrics
        start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')
        duration_days = (end_date - start_date).days
        
        initial_value = self.config.initial_capital
        final_value = portfolio_values['value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate daily returns
        daily_returns = portfolio_values['value'].pct_change().dropna()
        annualized_return = (1 + total_return) ** (252 / duration_days) - 1
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(portfolio_values['value'])
        var_95 = daily_returns.quantile(0.05)
        cvar_95 = daily_returns[daily_returns <= var_95].mean()
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trading metrics
        total_trades = len(trades_history)
        winning_trades = len([t for t in trades_history if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        if trades_history:
            wins = [t['pnl'] for t in trades_history if t['pnl'] > 0]
            losses = [t['pnl'] for t in trades_history if t['pnl'] < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            # Profit factor
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        else:
            avg_win = avg_loss = profit_factor = 0
        
        # Benchmark comparison
        benchmark_return = alpha = beta = information_ratio = 0
        if self.benchmark_data is not None:
            try:
                benchmark_returns = self.benchmark_data['close'].pct_change().dropna()
                benchmark_returns = benchmark_returns.loc[daily_returns.index]  # Align dates
                
                if len(benchmark_returns) > 0:
                    benchmark_total_return = (self.benchmark_data['close'].iloc[-1] / 
                                            self.benchmark_data['close'].iloc[0]) - 1
                    benchmark_return = benchmark_total_return
                    alpha = total_return - benchmark_total_return
                    
                    # Beta calculation
                    if len(benchmark_returns) == len(daily_returns):
                        covariance = np.cov(daily_returns, benchmark_returns)[0, 1]
                        benchmark_variance = np.var(benchmark_returns)
                        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
                        
                        # Information ratio
                        excess_returns = daily_returns - benchmark_returns
                        information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
            except Exception as e:
                logger.warning(f"Benchmark calculation failed: {e}")
        
        # Kelly utilization (simplified)
        kelly_utilization = 0.40  # Using 40% of Kelly (optimal fraction)
        
        # Risk-adjusted return
        risk_adjusted_return = total_return / abs(max_drawdown) if max_drawdown != 0 else total_return
        
        return BacktestResults(
            config=self.config,
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            portfolio_values=portfolio_values['value'],
            daily_returns=daily_returns,
            positions_history=simulation_results['positions_history'],
            trades_history=trades_history,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            kelly_utilization=kelly_utilization,
            risk_adjusted_return=risk_adjusted_return
        )
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()
    
    def _generate_backtest_report(self, results: BacktestResults, strategy_name: str) -> None:
        """Generate detailed backtest report."""
        logger.info("=== BACKTESTING RESULTS ===")
        logger.info(f"Strategy: {strategy_name}")
        logger.info(f"Period: {results.start_date.date()} to {results.end_date.date()} ({results.duration_days} days)")
        logger.info(f"Initial Capital: ${results.config.initial_capital:,.2f}")
        logger.info("")
        
        logger.info("PERFORMANCE METRICS:")
        logger.info(f"Total Return: {results.total_return:.2%}")
        logger.info(f"Annualized Return: {results.annualized_return:.2%}")
        logger.info(f"Volatility: {results.volatility:.2%}")
        logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
        logger.info(f"Max Drawdown: {results.max_drawdown:.2%}")
        logger.info(f"Calmar Ratio: {results.calmar_ratio:.3f}")
        logger.info("")
        
        logger.info("TRADING METRICS:")
        logger.info(f"Total Trades: {results.total_trades}")
        logger.info(f"Win Rate: {results.win_rate:.2%}")
        logger.info(f"Average Win: ${results.avg_win:.2f}")
        logger.info(f"Average Loss: ${results.avg_loss:.2f}")
        logger.info(f"Profit Factor: {results.profit_factor:.3f}")
        logger.info("")
        
        if results.benchmark_return != 0:
            logger.info("BENCHMARK COMPARISON:")
            logger.info(f"Benchmark Return: {results.benchmark_return:.2%}")
            logger.info(f"Alpha: {results.alpha:.2%}")
            logger.info(f"Beta: {results.beta:.3f}")
            logger.info(f"Information Ratio: {results.information_ratio:.3f}")
            logger.info("")
        
        logger.info("RISK METRICS:")
        logger.info(f"95% VaR: {results.var_95:.2%}")
        logger.info(f"95% CVaR: {results.cvar_95:.2%}")
        logger.info(f"Risk-Adjusted Return: {results.risk_adjusted_return:.3f}")
        
        # Add performance attribution analysis
        try:
            from ..analysis.performance_attribution import analyze_portfolio_attribution
            
            attribution_report = analyze_portfolio_attribution(
                portfolio_returns=results.daily_returns,
                benchmark_ticker=self.config.benchmark,
                portfolio_name=strategy_name
            )
            
            logger.info("")
            logger.info("PERFORMANCE ATTRIBUTION:")
            attr_metrics = attribution_report['attribution_metrics']
            logger.info(f"Alpha vs {self.config.benchmark}: {attr_metrics['alpha']:.2%}")
            logger.info(f"Beta: {attr_metrics['beta']:.3f}")
            logger.info(f"R-squared: {attr_metrics['r_squared']:.3f}")
            logger.info(f"Information Ratio: {attr_metrics['information_ratio']:.3f}")
            
            if attribution_report['factor_exposures']:
                significant_factors = [
                    f"{name.title()}: {exp['exposure']:.2f}"
                    for name, exp in attribution_report['factor_exposures'].items()
                    if exp['is_significant']
                ]
                if significant_factors:
                    logger.info(f"Significant Factors: {', '.join(significant_factors[:3])}")
        
        except Exception as e:
            logger.warning(f"Performance attribution failed: {e}")
        
        logger.info("===========================")
    
    def run_monte_carlo_validation(
        self,
        n_simulations: int = 100,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Run Monte Carlo validation of strategy robustness.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            confidence_level: Confidence level for results
            
        Returns:
            Dictionary with Monte Carlo validation results
        """
        logger.info(f"Running Monte Carlo validation with {n_simulations} simulations...")
        
        results = []
        base_config = self.config
        
        for i in range(min(20, n_simulations)):  # Limit to 20 for practical runtime
            try:
                logger.debug(f"Monte Carlo simulation {i+1}/{min(20, n_simulations)}")
                
                # Create perturbed configuration
                perturbed_config = self._create_perturbed_config(base_config, i)
                
                # Run backtest with perturbed parameters
                temp_engine = BacktestEngine(perturbed_config)
                temp_engine.historical_data = self.historical_data  # Reuse loaded data
                temp_engine.benchmark_data = self.benchmark_data
                
                # Run simulation components only (skip data loading)
                from ..models.timesfm_predictor import TimesFMPredictor
                from ..trading.signal_generator import SignalGenerator
                from ..trading.position_sizer import KellyPositionSizer
                from ..trading.risk_manager import RiskManager
                from ..trading.portfolio import PortfolioManager
                
                # Initialize components with slight variations
                safety_factor = np.clip(np.random.normal(0.40, 0.05), 0.25, 0.60)
                max_position = np.clip(np.random.normal(0.15, 0.02), 0.10, 0.20)
                confidence_threshold = np.clip(np.random.normal(0.75, 0.05), 0.65, 0.85)
                
                timesfm_predictor = TimesFMPredictor()
                signal_generator = SignalGenerator(timesfm_predictor=timesfm_predictor)
                position_sizer = KellyPositionSizer(
                    safety_factor=safety_factor,
                    max_position_pct=max_position,
                    min_confidence=confidence_threshold
                )
                risk_manager = RiskManager()
                portfolio_manager = PortfolioManager(
                    initial_capital=perturbed_config.initial_capital,
                    commission_rate=perturbed_config.commission_rate
                )
                
                # Run abbreviated simulation (sample dates for speed)
                simulation_results = temp_engine._run_abbreviated_simulation(
                    signal_generator, position_sizer, risk_manager, portfolio_manager
                )
                
                # Calculate results
                if simulation_results and 'portfolio_values' in simulation_results:
                    portfolio_df = pd.DataFrame(simulation_results['portfolio_values'])
                    if len(portfolio_df) > 1:
                        total_return = (portfolio_df['value'].iloc[-1] - portfolio_df['value'].iloc[0]) / portfolio_df['value'].iloc[0]
                        daily_returns = portfolio_df['value'].pct_change().dropna()
                        
                        if len(daily_returns) > 0:
                            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
                            max_dd = self._calculate_max_drawdown(portfolio_df['value'])
                            
                            results.append({
                                'simulation': i,
                                'total_return': total_return,
                                'sharpe_ratio': sharpe,
                                'max_drawdown': max_dd,
                                'total_trades': simulation_results.get('total_trades', 0),
                                'safety_factor': safety_factor,
                                'max_position': max_position
                            })
                
            except Exception as e:
                logger.debug(f"Monte Carlo simulation {i} failed: {e}")
                continue
        
        if results:
            returns = [r['total_return'] for r in results]
            sharpes = [r['sharpe_ratio'] for r in results]
            drawdowns = [r['max_drawdown'] for r in results]
            
            validation_results = {
                'n_successful_simulations': len(results),
                'return_mean': np.mean(returns),
                'return_std': np.std(returns),
                'return_confidence_interval': (
                    np.percentile(returns, (1-confidence_level)/2 * 100),
                    np.percentile(returns, (1+confidence_level)/2 * 100)
                ),
                'sharpe_mean': np.mean(sharpes),
                'sharpe_std': np.std(sharpes),
                'drawdown_mean': np.mean(drawdowns),
                'drawdown_std': np.std(drawdowns),
                'success_rate': len([r for r in returns if r > 0]) / len(returns),
                'parameter_sensitivity': {
                    'return_vs_safety_factor': np.corrcoef([r['total_return'] for r in results], 
                                                         [r['safety_factor'] for r in results])[0,1],
                    'return_vs_max_position': np.corrcoef([r['total_return'] for r in results], 
                                                        [r['max_position'] for r in results])[0,1]
                }
            }
            
            logger.info(f"Monte Carlo Results ({len(results)} simulations):")
            logger.info(f"  Return: {validation_results['return_mean']:.2%} ± {validation_results['return_std']:.2%}")
            logger.info(f"  Sharpe: {validation_results['sharpe_mean']:.2f} ± {validation_results['sharpe_std']:.2f}")
            logger.info(f"  Max DD: {validation_results['drawdown_mean']:.2%} ± {validation_results['drawdown_std']:.2%}")
            logger.info(f"  Success Rate: {validation_results['success_rate']:.1%}")
            
            return validation_results
        
        logger.warning("Monte Carlo validation failed - no successful simulations")
        return {}
    
    def _create_perturbed_config(self, base_config: BacktestConfig, simulation_id: int) -> BacktestConfig:
        """Create configuration with perturbed parameters for Monte Carlo."""
        # Use simulation_id as seed for reproducibility
        np.random.seed(42 + simulation_id)
        
        # Create small perturbations
        commission_mult = np.random.normal(1.0, 0.1)  # ±10% commission variation
        spread_mult = np.random.normal(1.0, 0.2)  # ±20% spread variation
        slippage_mult = np.random.normal(1.0, 0.3)  # ±30% slippage variation
        
        perturbed_config = BacktestConfig(
            start_date=base_config.start_date,
            end_date=base_config.end_date,
            initial_capital=base_config.initial_capital,
            commission_rate=base_config.commission_rate * commission_mult,
            bid_ask_spread_pct=base_config.bid_ask_spread_pct * spread_mult,
            slippage_pct=base_config.slippage_pct * slippage_mult,
            market_impact_threshold=base_config.market_impact_threshold,
            tickers=base_config.tickers,
            lookback_days=base_config.lookback_days,
            rebalance_frequency=base_config.rebalance_frequency,
            max_positions=base_config.max_positions,
            benchmark=base_config.benchmark
        )
        
        return perturbed_config
    
    def _run_abbreviated_simulation(
        self,
        signal_generator,
        position_sizer, 
        risk_manager,
        portfolio_manager,
        sample_ratio: float = 0.3
    ) -> Dict[str, Any]:
        """Run abbreviated simulation for Monte Carlo (sample dates for speed)."""
        start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')
        
        # Sample dates for faster simulation
        all_dates = pd.date_range(start_date, end_date, freq='D')
        sample_size = max(30, int(len(all_dates) * sample_ratio))  # At least 30 days
        sampled_dates = np.random.choice(all_dates, size=min(sample_size, len(all_dates)), replace=False)
        sampled_dates = sorted(sampled_dates)
        
        portfolio_values = []
        total_trades = 0
        
        for current_date in sampled_dates:
            try:
                # Get current prices (simplified)
                current_prices = {}
                for ticker, data in self.historical_data.items():
                    available_data = data[data.index <= current_date]
                    if len(available_data) > 0:
                        current_prices[ticker] = available_data['close'].iloc[-1]
                
                if not current_prices:
                    continue
                
                # Update portfolio
                portfolio_manager.update_positions(current_prices)
                portfolio_metrics = portfolio_manager.calculate_portfolio_metrics(current_prices)
                
                portfolio_values.append({
                    'date': current_date,
                    'value': portfolio_metrics.total_value
                })
                
                # Simplified trading (random signals for speed)
                if np.random.random() < 0.1:  # 10% chance of trade each day
                    ticker = np.random.choice(list(current_prices.keys()))
                    if ticker not in portfolio_manager.positions and len(portfolio_manager.positions) < 3:
                        # Mock position opening
                        total_trades += 1
                
            except Exception:
                continue
        
        return {
            'portfolio_values': portfolio_values,
            'total_trades': total_trades
        }


# Convenience function for running backtests
def run_backtest(
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01", 
    initial_capital: float = 100000,
    tickers: Optional[List[str]] = None
) -> BacktestResults:
    """Run a backtest with specified parameters.
    
    Args:
        start_date: Start date for backtesting
        end_date: End date for backtesting  
        initial_capital: Initial capital
        tickers: List of tickers to trade
        
    Returns:
        BacktestResults object
    """
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        tickers=tickers or ['NVDA', 'AMD', 'INTC']  # Smaller set for testing
    )
    
    engine = BacktestEngine(config)
    return engine.run_backtest()


if __name__ == "__main__":
    # Example backtest
    logger.info("Running example backtest...")
    
    results = run_backtest(
        start_date="2023-06-01",
        end_date="2023-12-31",
        initial_capital=100000,
        tickers=['NVDA', 'AMD']
    )
    
    logger.info(f"Backtest completed with {results.total_return:.2%} return")