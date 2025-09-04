"""
Comprehensive Backtesting Framework for Phase 7 Integration.
Provides end-to-end backtesting capabilities for the integrated ensemble system
with Phase 6 features and Phase 7 optimizations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pickle
import json
import time
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from ..models.efficient_prediction_pipeline import EfficientPredictionPipeline, PredictionRequest
from ..models.memory_optimization import MemoryOptimizationManager
from ..trading.recommendation_engine import TradingRecommendationEngine
from ..trading.portfolio_tracker import PortfolioTracker
from ..analysis.performance_attribution import PerformanceAttributor
from ..config.settings import config


@dataclass
class BacktestConfig:
    """Configuration for backtesting framework."""
    
    # Time period
    start_date: str
    end_date: str
    
    # Universe and data
    tickers: List[str]
    benchmark_ticker: str = 'SPY'
    lookback_window: int = 252  # Days of history needed for features
    
    # Prediction settings
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 5, 20])
    rebalance_frequency: str = 'daily'  # daily, weekly, monthly
    confidence_threshold: float = 0.6
    
    # Portfolio settings
    starting_capital: float = 100000.0
    max_position_size: float = 0.15
    max_positions: int = 10
    transaction_costs: float = 0.001  # 0.1%
    
    # Risk management
    stop_loss_pct: float = 0.08
    max_drawdown_pct: float = 0.20
    position_sizing_method: str = 'kelly'  # kelly, equal_weight, risk_parity
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    cache_predictions: bool = True
    
    # Output settings
    save_detailed_results: bool = True
    save_predictions: bool = False
    results_directory: str = "backtest_results"


@dataclass
class BacktestResult:
    """Results from backtesting run."""
    
    # Configuration
    config: BacktestConfig
    
    # Performance metrics
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Benchmark comparison
    benchmark_return: float
    benchmark_volatility: float
    benchmark_sharpe: float
    alpha: float
    beta: float
    information_ratio: float
    
    # Portfolio analytics
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Detailed results
    portfolio_values: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    predictions: Optional[pd.DataFrame] = None
    
    # Performance attribution
    sector_attribution: Optional[Dict[str, float]] = None
    factor_attribution: Optional[Dict[str, float]] = None
    
    # Metadata
    backtest_duration: float
    start_timestamp: datetime
    end_timestamp: datetime


class Phase7BacktestFramework:
    """Comprehensive backtesting framework with Phase 7 enhancements."""
    
    def __init__(
        self,
        ensemble_model,
        data_loader,
        config: BacktestConfig
    ):
        self.ensemble_model = ensemble_model
        self.data_loader = data_loader
        self.config = config
        
        # Initialize components
        self.prediction_pipeline = EfficientPredictionPipeline(
            ensemble_model=ensemble_model,
            cache_config={
                'max_size': 5000,
                'ttl_seconds': 3600,  # 1 hour for backtesting
                'memory_limit_mb': 1024
            },
            parallel_config={
                'max_workers': config.max_workers,
                'use_process_pool': False  # Use threads for better shared state
            }
        )
        
        self.memory_optimizer = MemoryOptimizationManager(enable_monitoring=False)
        
        # Portfolio and trading components
        self.portfolio_tracker = None
        self.recommendation_engine = None
        
        # Results storage
        self.daily_results = []
        self.position_history = []
        self.prediction_history = []
        self.trade_history = []
        
        # Performance tracking
        self.performance_stats = {
            'predictions_made': 0,
            'trades_executed': 0,
            'cache_hit_rate': 0.0,
            'avg_prediction_time': 0.0
        }
        
        logger.info(f"Phase7BacktestFramework initialized for {len(config.tickers)} tickers")
        logger.info(f"Backtest period: {config.start_date} to {config.end_date}")
    
    def run_backtest(self, save_results: bool = True) -> BacktestResult:
        """
        Run comprehensive backtest with Phase 7 enhancements.
        
        Args:
            save_results: Whether to save detailed results to disk
            
        Returns:
            BacktestResult with comprehensive metrics
        """
        start_time = time.time()
        start_timestamp = datetime.now()
        
        logger.info("Starting Phase 7 backtest...")
        
        try:
            # Initialize portfolio and trading systems
            self._initialize_trading_systems()
            
            # Load and prepare data
            market_data = self._load_and_prepare_data()
            if not market_data:
                raise ValueError("Failed to load market data")
            
            # Get trading dates
            trading_dates = self._get_trading_dates(market_data)
            total_days = len(trading_dates)
            
            logger.info(f"Backtesting over {total_days} trading days")
            
            # Initialize progress tracking
            progress_interval = max(1, total_days // 20)  # 20 progress updates
            
            # Main backtesting loop
            for i, current_date in enumerate(trading_dates):
                try:
                    # Progress logging
                    if i % progress_interval == 0 or i == total_days - 1:
                        progress = (i + 1) / total_days * 100
                        logger.info(f"Backtest progress: {progress:.1f}% ({i+1}/{total_days})")
                    
                    # Run single day backtest
                    self._run_single_day(current_date, market_data)
                    
                    # Memory management
                    if i % 50 == 0:  # Every 50 days
                        self.memory_optimizer.perform_light_cleanup()
                
                except Exception as e:
                    logger.error(f"Error on {current_date}: {e}")
                    continue
            
            # Calculate final results
            end_time = time.time()
            end_timestamp = datetime.now()
            
            backtest_result = self._calculate_final_results(
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                duration=end_time - start_time
            )
            
            # Save results if requested
            if save_results:
                self._save_results(backtest_result)
            
            logger.info(f"Backtest completed in {end_time - start_time:.1f} seconds")
            logger.info(f"Total Return: {backtest_result.total_return:.2%}")
            logger.info(f"Annual Return: {backtest_result.annual_return:.2%}")
            logger.info(f"Sharpe Ratio: {backtest_result.sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {backtest_result.max_drawdown:.2%}")
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
        
        finally:
            # Cleanup
            self._cleanup()
    
    def _initialize_trading_systems(self):
        """Initialize portfolio tracking and recommendation engine."""
        # Create temporary portfolio file for backtesting
        temp_portfolio_config = {
            'starting_capital': self.config.starting_capital,
            'max_position_size': self.config.max_position_size,
            'max_positions': self.config.max_positions,
            'transaction_costs': self.config.transaction_costs
        }
        
        # Initialize portfolio tracker with backtest settings
        self.portfolio_tracker = PortfolioTracker("backtest_portfolio.toml")
        self.portfolio_tracker.starting_capital = self.config.starting_capital
        
        # Initialize recommendation engine
        self.recommendation_engine = TradingRecommendationEngine("backtest_portfolio.toml")
        self.recommendation_engine.max_position_pct = self.config.max_position_size
        self.recommendation_engine.max_positions = self.config.max_positions
    
    def _load_and_prepare_data(self) -> Dict[str, pd.DataFrame]:
        """Load and prepare market data for backtesting."""
        logger.info("Loading market data...")
        
        market_data = {}
        failed_tickers = []
        
        for ticker in self.config.tickers + [self.config.benchmark_ticker]:
            try:
                data = self.data_loader.load_data(
                    ticker=ticker,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    include_volume=True
                )
                
                if data is not None and len(data) > self.config.lookback_window:
                    # Validate data quality
                    if self._validate_data_quality(data, ticker):
                        market_data[ticker] = data
                    else:
                        failed_tickers.append(ticker)
                else:
                    failed_tickers.append(ticker)
                    
            except Exception as e:
                logger.warning(f"Failed to load data for {ticker}: {e}")
                failed_tickers.append(ticker)
        
        if failed_tickers:
            logger.warning(f"Failed to load data for {len(failed_tickers)} tickers: {failed_tickers}")
        
        logger.info(f"Loaded data for {len(market_data)} tickers")
        return market_data
    
    def _validate_data_quality(self, data: pd.DataFrame, ticker: str) -> bool:
        """Validate data quality for backtesting."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in data.columns for col in required_columns):
            logger.warning(f"Missing required columns for {ticker}")
            return False
        
        # Check for excessive missing values
        missing_pct = data[required_columns].isnull().sum().sum() / (len(data) * len(required_columns))
        if missing_pct > 0.05:  # More than 5% missing
            logger.warning(f"Excessive missing data for {ticker}: {missing_pct:.1%}")
            return False
        
        # Check for price anomalies
        returns = data['close'].pct_change().dropna()
        if (abs(returns) > 0.5).any():  # 50%+ single day moves
            logger.warning(f"Extreme returns detected for {ticker}")
        
        return True
    
    def _get_trading_dates(self, market_data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """Get list of trading dates for backtesting."""
        # Use benchmark ticker as reference for trading calendar
        benchmark_data = market_data.get(self.config.benchmark_ticker)
        if benchmark_data is None:
            # Fallback to first available ticker
            benchmark_data = next(iter(market_data.values()))
        
        # Ensure we have enough lookback data
        all_dates = benchmark_data.index.tolist()
        start_idx = self.config.lookback_window
        
        trading_dates = all_dates[start_idx:]
        
        # Filter by rebalancing frequency
        if self.config.rebalance_frequency == 'weekly':
            # Keep only Monday or first day of week
            trading_dates = [d for d in trading_dates if d.weekday() == 0 or 
                           (d - timedelta(days=1)) not in trading_dates]
        elif self.config.rebalance_frequency == 'monthly':
            # Keep only first day of month or first trading day
            trading_dates = [d for d in trading_dates if d.day <= 3 or
                           (d - timedelta(days=1)).month != d.month]
        
        return trading_dates
    
    def _run_single_day(self, current_date: datetime, market_data: Dict[str, pd.DataFrame]):
        """Run backtest for a single trading day."""
        # Get available data up to current date for all tickers
        current_data = {}
        for ticker, data in market_data.items():
            if ticker in self.config.tickers:
                # Get data up to current date
                available_data = data[data.index <= current_date]
                if len(available_data) >= self.config.lookback_window:
                    current_data[ticker] = available_data
        
        if not current_data:
            return
        
        # Generate predictions for all tickers
        predictions = self._generate_predictions(current_date, current_data)
        
        # Update portfolio values with current market prices
        self._update_portfolio_values(current_date, market_data)
        
        # Generate trading recommendations
        recommendations = self._generate_recommendations(current_date, predictions, current_data)
        
        # Execute trades
        trades_executed = self._execute_trades(current_date, recommendations, market_data)
        
        # Record daily results
        self._record_daily_results(current_date, predictions, trades_executed)
        
        # Update performance statistics
        self._update_performance_stats(predictions, trades_executed)
    
    def _generate_predictions(
        self, 
        current_date: datetime, 
        current_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """Generate predictions for all tickers."""
        predictions = {}
        prediction_requests = []
        
        # Create prediction requests
        for ticker, data in current_data.items():
            request = PredictionRequest(
                ticker=ticker,
                price_data=data,
                multi_asset_data=current_data,
                horizons=self.config.prediction_horizons,
                return_confidence=True,
                priority=1
            )
            prediction_requests.append(request)
        
        # Process predictions in parallel if enabled
        if self.config.enable_parallel_processing and len(prediction_requests) > 1:
            prediction_results = self.prediction_pipeline.predict_parallel(
                requests=prediction_requests,
                max_parallel=self.config.max_workers
            )
        else:
            # Sequential processing
            prediction_results = []
            for request in prediction_requests:
                result = self.prediction_pipeline.predict_single(
                    ticker=request.ticker,
                    price_data=request.price_data,
                    multi_asset_data=request.multi_asset_data,
                    horizons=request.horizons,
                    return_confidence=request.return_confidence,
                    use_cache=self.config.cache_predictions
                )
                prediction_results.append(result)
        
        # Process prediction results
        for result in prediction_results:
            if 'error' not in result.predictions:
                predictions[result.ticker] = {
                    'predictions': result.predictions,
                    'processing_time': result.processing_time_ms,
                    'cache_hit': result.cache_hit,
                    'date': current_date
                }
        
        return predictions
    
    def _update_portfolio_values(self, current_date: datetime, market_data: Dict[str, pd.DataFrame]):
        """Update portfolio values with current market prices."""
        for ticker, position in self.portfolio_tracker.positions.items():
            if ticker in market_data:
                ticker_data = market_data[ticker]
                current_prices = ticker_data[ticker_data.index <= current_date]
                
                if len(current_prices) > 0:
                    current_price = current_prices['close'].iloc[-1]
                    position.update_current_price(current_price)
    
    def _generate_recommendations(
        self,
        current_date: datetime,
        predictions: Dict[str, Dict],
        current_data: Dict[str, pd.DataFrame]
    ) -> List:
        """Generate trading recommendations based on predictions."""
        recommendations = []
        
        for ticker, pred_data in predictions.items():
            try:
                pred_dict = pred_data['predictions']
                
                # Focus on primary horizon (usually 1-day)
                primary_horizon = f"horizon_{self.config.prediction_horizons[0]}"
                
                if primary_horizon in pred_dict:
                    horizon_pred = pred_dict[primary_horizon]
                    
                    predicted_return = horizon_pred.get('prediction', 0)
                    confidence = horizon_pred.get('confidence', 0)
                    
                    # Apply confidence threshold
                    if confidence >= self.config.confidence_threshold:
                        # Determine action based on prediction
                        if predicted_return > 0.005:  # 0.5% threshold for buy
                            action = 'BUY'
                        elif predicted_return < -0.005:  # -0.5% threshold for sell
                            action = 'SELL'
                        else:
                            action = 'HOLD'
                        
                        if action != 'HOLD':
                            # Get current price
                            current_price = current_data[ticker]['close'].iloc[-1]
                            
                            # Calculate position size
                            portfolio_value = self.portfolio_tracker.get_portfolio_summary()['total_value']
                            position_size = self._calculate_position_size(
                                predicted_return, confidence, portfolio_value
                            )
                            
                            recommendation = {
                                'ticker': ticker,
                                'action': action,
                                'predicted_return': predicted_return,
                                'confidence': confidence,
                                'position_size': position_size,
                                'current_price': current_price,
                                'date': current_date
                            }
                            
                            recommendations.append(recommendation)
            
            except Exception as e:
                logger.debug(f"Error generating recommendation for {ticker}: {e}")
        
        # Sort by confidence (highest first)
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return recommendations
    
    def _calculate_position_size(
        self, 
        predicted_return: float, 
        confidence: float, 
        portfolio_value: float
    ) -> float:
        """Calculate optimal position size based on prediction and confidence."""
        if self.config.position_sizing_method == 'kelly':
            # Simplified Kelly criterion
            win_prob = confidence
            loss_prob = 1 - confidence
            expected_return = abs(predicted_return)
            estimated_risk = max(0.02, expected_return * 0.5)  # Assume 50% of return as risk
            
            if estimated_risk > 0:
                odds = expected_return / estimated_risk
                kelly_fraction = (odds * win_prob - loss_prob) / odds
                kelly_fraction = max(0, min(kelly_fraction, 0.5))  # Cap at 50%
                kelly_fraction *= 0.4  # Safety factor
            else:
                kelly_fraction = 0.05  # Default 5%
            
            position_value = portfolio_value * kelly_fraction
            
        elif self.config.position_sizing_method == 'equal_weight':
            position_value = portfolio_value * (1 / self.config.max_positions)
        
        else:  # risk_parity or default
            # Simple confidence-weighted sizing
            base_size = portfolio_value * 0.1  # 10% base
            confidence_multiplier = confidence * 2  # 0-2x multiplier
            position_value = base_size * confidence_multiplier
        
        # Apply position size limits
        max_position_value = portfolio_value * self.config.max_position_size
        position_value = min(position_value, max_position_value)
        
        return max(0, position_value)
    
    def _execute_trades(
        self,
        current_date: datetime,
        recommendations: List[Dict],
        market_data: Dict[str, pd.DataFrame]
    ) -> List[Dict]:
        """Execute trades based on recommendations."""
        executed_trades = []
        
        for rec in recommendations:
            try:
                ticker = rec['ticker']
                action = rec['action']
                position_size = rec['position_size']
                current_price = rec['current_price']
                
                # Check if we can execute this trade
                if self._can_execute_trade(ticker, action, position_size):
                    # Calculate shares to trade
                    shares_to_trade = int(position_size / current_price)
                    
                    if shares_to_trade > 0:
                        # Execute the trade
                        trade_result = self._execute_single_trade(
                            ticker, action, shares_to_trade, current_price, current_date
                        )
                        
                        if trade_result:
                            executed_trades.append({
                                'ticker': ticker,
                                'action': action,
                                'shares': shares_to_trade,
                                'price': current_price,
                                'value': shares_to_trade * current_price,
                                'date': current_date,
                                'predicted_return': rec['predicted_return'],
                                'confidence': rec['confidence']
                            })
            
            except Exception as e:
                logger.debug(f"Error executing trade for {rec['ticker']}: {e}")
        
        return executed_trades
    
    def _can_execute_trade(self, ticker: str, action: str, position_size: float) -> bool:
        """Check if trade can be executed given current portfolio state."""
        portfolio_summary = self.portfolio_tracker.get_portfolio_summary()
        
        if action == 'BUY':
            # Check available cash
            available_cash = portfolio_summary.get('cash_balance', 0)
            if position_size > available_cash * 0.95:  # 5% cash buffer
                return False
            
            # Check position limits
            current_positions = len(self.portfolio_tracker.positions)
            if ticker not in self.portfolio_tracker.positions:
                if current_positions >= self.config.max_positions:
                    return False
        
        elif action == 'SELL':
            # Check if we own the position
            if ticker not in self.portfolio_tracker.positions:
                return False
        
        return True
    
    def _execute_single_trade(
        self,
        ticker: str,
        action: str,
        shares: int,
        price: float,
        date: datetime
    ) -> bool:
        """Execute a single trade."""
        try:
            if action == 'BUY':
                success = self.portfolio_tracker.buy_stock(
                    symbol=ticker,
                    shares=shares,
                    price=price
                )
            elif action == 'SELL':
                success = self.portfolio_tracker.sell_stock(
                    symbol=ticker,
                    shares=shares,
                    price=price
                )
            else:
                success = False
            
            return success
            
        except Exception as e:
            logger.debug(f"Trade execution failed for {ticker}: {e}")
            return False
    
    def _record_daily_results(
        self,
        current_date: datetime,
        predictions: Dict[str, Dict],
        trades: List[Dict]
    ):
        """Record daily results for analysis."""
        portfolio_summary = self.portfolio_tracker.get_portfolio_summary()
        
        daily_result = {
            'date': current_date,
            'portfolio_value': portfolio_summary['total_value'],
            'cash_balance': portfolio_summary.get('cash_balance', 0),
            'invested_value': portfolio_summary.get('invested_value', 0),
            'num_positions': len(self.portfolio_tracker.positions),
            'num_predictions': len(predictions),
            'num_trades': len(trades),
            'predictions': predictions if self.config.save_predictions else None,
            'trades': trades
        }
        
        self.daily_results.append(daily_result)
        
        # Record positions
        for ticker, position in self.portfolio_tracker.positions.items():
            self.position_history.append({
                'date': current_date,
                'ticker': ticker,
                'shares': position.shares,
                'current_price': position.current_price,
                'current_value': position.current_value,
                'unrealized_pnl': position.unrealized_pnl
            })
        
        # Record trades
        self.trade_history.extend(trades)
    
    def _update_performance_stats(self, predictions: Dict, trades: List):
        """Update performance tracking statistics."""
        self.performance_stats['predictions_made'] += len(predictions)
        self.performance_stats['trades_executed'] += len(trades)
        
        # Update cache statistics from prediction pipeline
        cache_stats = self.prediction_pipeline.get_performance_stats()
        if 'cache_stats' in cache_stats:
            self.performance_stats['cache_hit_rate'] = cache_stats['cache_stats'].get('hit_rate', 0)
    
    def _calculate_final_results(
        self,
        start_timestamp: datetime,
        end_timestamp: datetime,
        duration: float
    ) -> BacktestResult:
        """Calculate final backtest results and metrics."""
        logger.info("Calculating final backtest results...")
        
        # Extract portfolio values
        portfolio_values = pd.Series(
            data=[r['portfolio_value'] for r in self.daily_results],
            index=[r['date'] for r in self.daily_results],
            name='portfolio_value'
        )
        
        # Calculate returns
        returns = portfolio_values.pct_change().dropna()
        
        # Basic performance metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Drawdown calculation
        running_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Benchmark comparison (if available)
        benchmark_return, benchmark_volatility, benchmark_sharpe = 0, 0, 0
        alpha, beta, information_ratio = 0, 1, 0
        
        # Trading statistics
        num_trades = len(self.trade_history)
        
        win_trades = [t for t in self.trade_history if t.get('predicted_return', 0) > 0]
        win_rate = len(win_trades) / num_trades if num_trades > 0 else 0
        
        avg_win = np.mean([t['predicted_return'] for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([t['predicted_return'] for t in self.trade_history if t.get('predicted_return', 0) <= 0]) if self.trade_history else 0
        
        profit_factor = abs(avg_win) / abs(avg_loss) if avg_loss != 0 else float('inf')
        
        # Create DataFrames for detailed analysis
        positions_df = pd.DataFrame(self.position_history) if self.position_history else pd.DataFrame()
        trades_df = pd.DataFrame(self.trade_history) if self.trade_history else pd.DataFrame()
        predictions_df = None
        
        if self.config.save_predictions:
            pred_records = []
            for daily_result in self.daily_results:
                if daily_result['predictions']:
                    for ticker, pred_data in daily_result['predictions'].items():
                        pred_records.append({
                            'date': daily_result['date'],
                            'ticker': ticker,
                            **pred_data['predictions'].get('horizon_1', {}),
                            'processing_time': pred_data.get('processing_time', 0),
                            'cache_hit': pred_data.get('cache_hit', False)
                        })
            
            predictions_df = pd.DataFrame(pred_records) if pred_records else pd.DataFrame()
        
        # Performance attribution (simplified)
        sector_attribution = self._calculate_sector_attribution(trades_df)
        factor_attribution = self._calculate_factor_attribution(returns)
        
        return BacktestResult(
            config=self.config,
            total_return=total_return,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            benchmark_return=benchmark_return,
            benchmark_volatility=benchmark_volatility,
            benchmark_sharpe=benchmark_sharpe,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            num_trades=num_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            portfolio_values=portfolio_values,
            positions=positions_df,
            trades=trades_df,
            predictions=predictions_df,
            sector_attribution=sector_attribution,
            factor_attribution=factor_attribution,
            backtest_duration=duration,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp
        )
    
    def _calculate_sector_attribution(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate simplified sector attribution."""
        # This is a simplified version - would need sector mapping in practice
        if trades_df.empty:
            return {}
        
        sector_mapping = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'TSLA': 'Consumer Discretionary', 'NVDA': 'Technology',
            'JPM': 'Financials', 'JNJ': 'Healthcare', 'PG': 'Consumer Staples'
        }
        
        attribution = {}
        for _, trade in trades_df.iterrows():
            ticker = trade['ticker']
            sector = sector_mapping.get(ticker, 'Other')
            
            if sector not in attribution:
                attribution[sector] = 0
            
            # Simplified attribution based on trade value
            attribution[sector] += trade.get('value', 0)
        
        return attribution
    
    def _calculate_factor_attribution(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate simplified factor attribution."""
        # This is a placeholder - would implement proper factor analysis
        return {
            'market_beta': 1.0,
            'size_factor': 0.0,
            'value_factor': 0.0,
            'momentum_factor': 0.0,
            'quality_factor': 0.0,
            'alpha': returns.mean() * 252  # Annualized
        }
    
    def _save_results(self, result: BacktestResult):
        """Save backtest results to disk."""
        results_dir = Path(self.config.results_directory)
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        with open(results_dir / f"backtest_result_{timestamp}.pkl", 'wb') as f:
            pickle.dump(result, f)
        
        # Save summary as JSON
        summary = {
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'num_trades': result.num_trades,
            'win_rate': result.win_rate,
            'backtest_duration': result.backtest_duration,
            'config': {
                'start_date': result.config.start_date,
                'end_date': result.config.end_date,
                'tickers': result.config.tickers,
                'starting_capital': result.config.starting_capital
            }
        }
        
        with open(results_dir / f"backtest_summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed CSVs
        if result.portfolio_values is not None:
            result.portfolio_values.to_csv(results_dir / f"portfolio_values_{timestamp}.csv")
        
        if result.trades is not None and not result.trades.empty:
            result.trades.to_csv(results_dir / f"trades_{timestamp}.csv", index=False)
        
        if result.positions is not None and not result.positions.empty:
            result.positions.to_csv(results_dir / f"positions_{timestamp}.csv", index=False)
        
        logger.info(f"Results saved to {results_dir}")
    
    def _cleanup(self):
        """Clean up resources after backtesting."""
        self.prediction_pipeline.shutdown()
        self.memory_optimizer.perform_light_cleanup()
        
        logger.info("Backtest cleanup completed")


# Convenience functions
def run_phase7_backtest(
    ensemble_model,
    data_loader,
    config: BacktestConfig,
    save_results: bool = True
) -> BacktestResult:
    """
    Convenience function to run Phase 7 backtest.
    
    Args:
        ensemble_model: Trained ensemble model
        data_loader: Data loader for market data
        config: Backtest configuration
        save_results: Whether to save results
        
    Returns:
        BacktestResult with comprehensive metrics
    """
    framework = Phase7BacktestFramework(ensemble_model, data_loader, config)
    return framework.run_backtest(save_results)


def create_backtest_config(
    start_date: str,
    end_date: str,
    tickers: List[str],
    starting_capital: float = 100000.0
) -> BacktestConfig:
    """
    Create a default backtest configuration.
    
    Args:
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        tickers: List of tickers to trade
        starting_capital: Starting capital amount
        
    Returns:
        BacktestConfig with sensible defaults
    """
    return BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        tickers=tickers,
        starting_capital=starting_capital,
        prediction_horizons=[1, 5, 20],
        rebalance_frequency='daily',
        confidence_threshold=0.6,
        max_position_size=0.15,
        max_positions=10,
        transaction_costs=0.001,
        enable_parallel_processing=True,
        max_workers=4,
        save_detailed_results=True
    )