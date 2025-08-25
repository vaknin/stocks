"""Main trading engine orchestrator for automated trading operations."""

import asyncio
import time
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import pandas as pd
import numpy as np

from ..data.ingester import DataIngester
from ..data.database import TradingDatabase
from ..data.validator import DataValidator
from ..models.timesfm_predictor import TimesFMPredictor
from ..models.uncertainty import TradingUncertaintyFilter
from .signal_generator import SignalGenerator, MultiTimeFrameSignal, SignalType, TimeFrame
from .position_sizer import KellyPositionSizer
from .risk_manager import RiskManager, RiskLevel
from .portfolio import PortfolioManager
from ..config.settings import config


class EngineState(Enum):
    """Trading engine states."""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


@dataclass
class TradingCycle:
    """Results from a trading cycle."""
    timestamp: datetime
    signals_generated: int
    positions_opened: int
    positions_closed: int
    risk_alerts: int
    portfolio_value: float
    execution_time_ms: float
    errors: List[str]


class TradingEngine:
    """Main automated trading engine."""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        tickers: Optional[List[str]] = None,
        trading_mode: str = "paper",  # "paper" or "live"
        cycle_interval_minutes: int = 5,
        max_positions: int = 10
    ):
        """Initialize trading engine.
        
        Args:
            initial_capital: Starting capital
            tickers: List of tickers to trade (defaults to config)
            trading_mode: Trading mode (paper/live)
            cycle_interval_minutes: Minutes between trading cycles
            max_positions: Maximum number of concurrent positions
        """
        self.initial_capital = initial_capital
        self.tickers = tickers or config.STOCK_TICKERS
        self.trading_mode = trading_mode
        self.cycle_interval = timedelta(minutes=cycle_interval_minutes)
        self.max_positions = max_positions
        
        # Engine state
        self.state = EngineState.STOPPED
        self.last_cycle_time = None
        self.cycle_count = 0
        self.total_errors = 0
        
        # Initialize components
        logger.info("Initializing trading engine components...")
        
        self.data_ingester = DataIngester(tickers=self.tickers)
        self.database = TradingDatabase()
        self.data_validator = DataValidator()
        
        # ML Models - Using our state-of-the-art ensemble!
        from ..models.ensemble import MetaLearningEnsemble
        self.ensemble_predictor = MetaLearningEnsemble()
        self.uncertainty_filter = TradingUncertaintyFilter()
        
        # Trading components
        self.signal_generator = SignalGenerator(
            predictor=self.ensemble_predictor,  # Use ensemble instead of single model
            uncertainty_filter=self.uncertainty_filter
        )
        self.position_sizer = KellyPositionSizer()
        self.risk_manager = RiskManager()
        self.portfolio_manager = PortfolioManager(
            initial_capital=initial_capital,
            database=self.database
        )
        
        # Performance tracking
        self.cycle_history: List[TradingCycle] = []
        self.performance_metrics = {}
        
        # Sector mapping for risk management
        self.sector_map = self._initialize_sector_map()
        
        logger.info(f"Trading engine initialized in {trading_mode} mode with ${initial_capital:,.2f}")
        logger.info(f"Tracking {len(self.tickers)} tickers: {self.tickers}")
    
    async def start(self) -> None:
        """Start the trading engine."""
        if self.state != EngineState.STOPPED:
            logger.warning(f"Engine already running in state: {self.state}")
            return
        
        logger.info("Starting trading engine...")
        self.state = EngineState.STARTING
        
        try:
            # Perform startup checks
            await self._startup_checks()
            
            # Start main trading loop
            self.state = EngineState.RUNNING
            logger.info("Trading engine started successfully")
            
            await self._main_trading_loop()
            
        except Exception as e:
            logger.error(f"Failed to start trading engine: {e}")
            self.state = EngineState.ERROR
            raise
    
    async def stop(self) -> None:
        """Stop the trading engine."""
        logger.info("Stopping trading engine...")
        self.state = EngineState.STOPPED
        
        # Close any pending orders (in live trading)
        if self.trading_mode == "live":
            await self._close_pending_orders()
        
        # Generate final performance report
        self._generate_performance_report()
        
        logger.info("Trading engine stopped")
    
    async def pause(self) -> None:
        """Pause the trading engine."""
        if self.state == EngineState.RUNNING:
            self.state = EngineState.PAUSED
            logger.info("Trading engine paused")
    
    async def resume(self) -> None:
        """Resume the trading engine."""
        if self.state == EngineState.PAUSED:
            self.state = EngineState.RUNNING
            logger.info("Trading engine resumed")
    
    async def _main_trading_loop(self) -> None:
        """Main trading loop."""
        logger.info("Starting main trading loop")
        
        while self.state in [EngineState.RUNNING, EngineState.PAUSED]:
            try:
                cycle_start = datetime.now()
                
                if self.state == EngineState.RUNNING:
                    # Execute trading cycle
                    cycle_result = await self._execute_trading_cycle()
                    self.cycle_history.append(cycle_result)
                    self.cycle_count += 1
                    
                    # Log cycle summary
                    logger.info(
                        f"Cycle {self.cycle_count} completed in {cycle_result.execution_time_ms:.0f}ms: "
                        f"Signals={cycle_result.signals_generated}, "
                        f"Opened={cycle_result.positions_opened}, "
                        f"Closed={cycle_result.positions_closed}, "
                        f"Value=${cycle_result.portfolio_value:,.2f}"
                    )
                    
                    self.last_cycle_time = cycle_start
                
                # Wait for next cycle
                await asyncio.sleep(60)  # Check every minute if it's time for next cycle
                
                # Check if it's time for next trading cycle
                if (self.last_cycle_time is None or 
                    datetime.now() - self.last_cycle_time >= self.cycle_interval):
                    continue
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping engine")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                self.total_errors += 1
                
                # Stop engine if too many errors
                if self.total_errors > 10:
                    logger.error("Too many errors, stopping engine")
                    self.state = EngineState.ERROR
                    break
                
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _execute_trading_cycle(self) -> TradingCycle:
        """Execute a single trading cycle."""
        cycle_start = time.time()
        errors = []
        signals_generated = 0
        positions_opened = 0
        positions_closed = 0
        risk_alerts = 0
        
        try:
            # 1. Fetch latest market data
            logger.debug("Fetching market data")
            market_data = await self._fetch_market_data()
            
            # 2. Validate data quality
            logger.debug("Validating data quality")
            validated_data = self._validate_market_data(market_data)
            
            # 3. Update current positions with latest prices
            current_prices = self._extract_current_prices(validated_data)
            self.portfolio_manager.update_positions(current_prices)
            
            # 4. Check for stop loss / take profit triggers
            stop_triggers = self.portfolio_manager.check_stop_loss_take_profit(current_prices)
            for ticker in stop_triggers:
                success = self.portfolio_manager.close_position(
                    ticker=ticker,
                    exit_price=current_prices.get(ticker),
                    reason="STOP_LOSS_TRIGGER"
                )
                if success:
                    positions_closed += 1
            
            # 5. Risk management checks
            risk_metrics = self.risk_manager.calculate_portfolio_risk_metrics(
                current_positions=self._get_current_positions_dict(),
                current_prices=current_prices,
                portfolio_value=self.portfolio_manager.cash_balance + sum(
                    pos.market_value for pos in self.portfolio_manager.positions.values()
                )
            )
            
            # Count risk alerts
            active_alerts = self.risk_manager.get_active_risk_alerts(max_age_hours=1)
            risk_alerts = len(active_alerts)
            
            # 6. Generate trading signals
            logger.debug("Generating trading signals")
            signals = self.signal_generator.generate_signals(
                data_dict=self._format_data_for_signals(validated_data),
                current_prices=current_prices
            )
            signals_generated = len(signals)
            
            # 7. Filter signals by quality and risk
            quality_signals = self.signal_generator.filter_signals_by_quality(signals)
            
            # 8. Position sizing
            logger.debug("Calculating position sizes")
            position_sizes = self.position_sizer.batch_calculate_positions(
                predictions=self._convert_signals_to_predictions(quality_signals),
                current_prices=current_prices,
                portfolio_value=self.portfolio_manager.cash_balance + sum(
                    pos.market_value for pos in self.portfolio_manager.positions.values()
                ),
                current_positions=self._get_current_positions_values()
            )
            
            # 9. Execute new positions
            positions_opened += await self._execute_new_positions(
                signals=quality_signals,
                position_sizes=position_sizes,
                current_prices=current_prices
            )
            
            # 10. Update performance metrics
            portfolio_metrics = self.portfolio_manager.calculate_portfolio_metrics(current_prices)
            self.performance_metrics = portfolio_metrics
            
            # 11. Log to database
            await self._log_cycle_to_database(portfolio_metrics)
            
        except Exception as e:
            error_msg = f"Error in trading cycle: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        execution_time = (time.time() - cycle_start) * 1000  # Convert to milliseconds
        
        portfolio_value = getattr(self.performance_metrics, 'total_value', self.initial_capital)
        
        return TradingCycle(
            timestamp=datetime.now(),
            signals_generated=signals_generated,
            positions_opened=positions_opened,
            positions_closed=positions_closed,
            risk_alerts=risk_alerts,
            portfolio_value=portfolio_value,
            execution_time_ms=execution_time,
            errors=errors
        )
    
    async def _fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch latest market data for all tickers."""
        try:
            # Fetch real-time data (5-minute intervals for intraday)
            real_time_data = self.data_ingester.fetch_real_time_data(
                interval="5m",
                period="1d"
            )
            
            # Also fetch daily data for longer-term signals
            daily_data = self.data_ingester.fetch_historical_data(
                start_date=(datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d'),
                interval="1d"
            )
            
            # Combine data
            combined_data = {}
            for ticker in self.tickers:
                if ticker in real_time_data and ticker in daily_data:
                    combined_data[ticker] = {
                        TimeFrame.INTRADAY: real_time_data[ticker],
                        TimeFrame.DAILY: daily_data[ticker]
                    }
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            return {}
    
    def _validate_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate market data quality."""
        validated_data = {}
        
        for ticker, timeframe_data in market_data.items():
            validated_timeframes = {}
            
            for timeframe, df in timeframe_data.items():
                if self.data_validator.validate_ohlcv_data(df):
                    validated_timeframes[timeframe] = df
                else:
                    logger.warning(f"Data validation failed for {ticker} {timeframe.value}")
            
            if validated_timeframes:
                validated_data[ticker] = validated_timeframes
        
        return validated_data
    
    def _extract_current_prices(self, validated_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract current prices from validated data."""
        current_prices = {}
        
        for ticker, timeframe_data in validated_data.items():
            # Use intraday data for current price if available, otherwise daily
            if TimeFrame.INTRADAY in timeframe_data:
                df = timeframe_data[TimeFrame.INTRADAY]
            elif TimeFrame.DAILY in timeframe_data:
                df = timeframe_data[TimeFrame.DAILY]
            else:
                continue
            
            if not df.empty:
                current_prices[ticker] = df['close'].iloc[-1]
        
        return current_prices
    
    def _format_data_for_signals(self, validated_data: Dict[str, Any]) -> Dict[str, Dict[TimeFrame, pd.DataFrame]]:
        """Format data for signal generation."""
        return validated_data
    
    def _convert_signals_to_predictions(self, signals: Dict[str, MultiTimeFrameSignal]) -> Dict[str, Dict[str, Any]]:
        """Convert signals to format expected by position sizer."""
        predictions = {}
        
        for ticker, multi_signal in signals.items():
            # Use the strongest individual signal
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
        
        return predictions
    
    def _get_current_positions_dict(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions in format expected by risk manager."""
        positions_dict = {}
        
        for ticker, position in self.portfolio_manager.positions.items():
            positions_dict[ticker] = {
                'size': position.size,
                'entry_price': position.entry_price,
                'market_value': position.market_value,
                'stop_loss': position.stop_loss
            }
        
        return positions_dict
    
    def _get_current_positions_values(self) -> Dict[str, float]:
        """Get current position values."""
        return {
            ticker: position.market_value
            for ticker, position in self.portfolio_manager.positions.items()
        }
    
    async def _execute_new_positions(
        self,
        signals: Dict[str, MultiTimeFrameSignal],
        position_sizes: Dict[str, Any],
        current_prices: Dict[str, float]
    ) -> int:
        """Execute new positions based on signals and sizing."""
        positions_opened = 0
        
        # Skip if we already have max positions
        if len(self.portfolio_manager.positions) >= self.max_positions:
            logger.debug(f"Max positions ({self.max_positions}) reached, skipping new positions")
            return 0
        
        for ticker, multi_signal in signals.items():
            if ticker not in position_sizes or ticker not in current_prices:
                continue
            
            # Skip if we already have a position in this ticker
            if ticker in self.portfolio_manager.positions:
                continue
            
            position_result = position_sizes[ticker]
            current_price = current_prices[ticker]
            
            # Pre-trade risk check
            portfolio_value = (self.portfolio_manager.cash_balance + 
                             sum(pos.market_value for pos in self.portfolio_manager.positions.values()))
            
            approved, violations = self.risk_manager.check_pre_trade_risk(
                ticker=ticker,
                proposed_size=position_result.recommended_size,
                current_price=current_price,
                portfolio_value=portfolio_value,
                current_positions=self._get_current_positions_dict(),
                sector_map=self.sector_map
            )
            
            if not approved:
                logger.debug(f"Trade rejected for {ticker}: {violations}")
                continue
            
            # Determine position direction
            signal_type = multi_signal.overall_signal
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                position_size = abs(position_result.recommended_size)
            elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                position_size = -abs(position_result.recommended_size)  # Short position
            else:
                continue  # Skip HOLD signals
            
            # Get representative signal for stop loss/take profit
            representative_signal = list(multi_signal.signals_by_timeframe.values())[0]
            
            # Open position
            success = self.portfolio_manager.open_position(
                ticker=ticker,
                size=position_size,
                entry_price=current_price,
                signal=representative_signal,
                stop_loss=representative_signal.stop_loss,
                take_profit=representative_signal.take_profit
            )
            
            if success:
                positions_opened += 1
                logger.info(f"Opened position: {position_size:+.0f} shares of {ticker} at ${current_price:.2f}")
        
        return positions_opened
    
    async def _startup_checks(self) -> None:
        """Perform startup health checks."""
        logger.info("Performing startup checks...")
        
        # Check database connectivity
        try:
            coverage = self.database.get_data_coverage_report()
            logger.info(f"Database connected, {len(coverage)} ticker-interval pairs available")
        except Exception as e:
            raise RuntimeError(f"Database check failed: {e}")
        
        # Check data ingester
        try:
            freshness = self.data_ingester.get_data_freshness(self.tickers[0])
            logger.info(f"Data freshness check: {freshness}")
        except Exception as e:
            logger.warning(f"Data freshness check failed: {e}")
        
        # Check ML models
        model_info = self.timesfm_predictor.get_model_info()
        logger.info(f"TimesFM model loaded: {model_info['is_loaded']}")
        
        logger.info("Startup checks completed")
    
    async def _close_pending_orders(self) -> None:
        """Close any pending orders (for live trading)."""
        # Placeholder for live trading integration
        pass
    
    async def _log_cycle_to_database(self, portfolio_metrics) -> None:
        """Log cycle results to database."""
        try:
            # Insert performance metrics
            with self.database.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO performance_metrics
                    (date, portfolio_value, daily_return, cumulative_return, 
                     drawdown, sharpe_ratio, win_rate, num_trades)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    datetime.now().date(),
                    portfolio_metrics.total_value,
                    portfolio_metrics.daily_return,
                    portfolio_metrics.cumulative_return,
                    portfolio_metrics.current_drawdown,
                    portfolio_metrics.sharpe_ratio,
                    portfolio_metrics.win_rate,
                    portfolio_metrics.total_trades
                ])
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log cycle to database: {e}")
    
    def _generate_performance_report(self) -> None:
        """Generate final performance report."""
        if not self.cycle_history:
            logger.info("No trading cycles to report")
            return
        
        total_cycles = len(self.cycle_history)
        avg_execution_time = np.mean([c.execution_time_ms for c in self.cycle_history])
        total_signals = sum(c.signals_generated for c in self.cycle_history)
        total_opened = sum(c.positions_opened for c in self.cycle_history)
        total_closed = sum(c.positions_closed for c in self.cycle_history)
        
        final_value = self.cycle_history[-1].portfolio_value
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        logger.info("=== TRADING ENGINE PERFORMANCE REPORT ===")
        logger.info(f"Total Cycles: {total_cycles}")
        logger.info(f"Average Execution Time: {avg_execution_time:.1f}ms")
        logger.info(f"Total Signals Generated: {total_signals}")
        logger.info(f"Total Positions Opened: {total_opened}")
        logger.info(f"Total Positions Closed: {total_closed}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Final Value: ${final_value:,.2f}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info("==========================================")
    
    def _initialize_sector_map(self) -> Dict[str, str]:
        """Initialize sector mapping for risk management."""
        # Simplified sector mapping for AI/semiconductor stocks
        return {
            'NVDA': 'Semiconductors',
            'AMD': 'Semiconductors', 
            'INTC': 'Semiconductors',
            'QCOM': 'Semiconductors',
            'AVGO': 'Semiconductors',
            'MU': 'Semiconductors',
            'TSM': 'Semiconductors',
            'ASML': 'Semiconductors',
            'ARM': 'Semiconductors',
            'SMCI': 'Technology Hardware'
        }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        current_positions = len(self.portfolio_manager.positions)
        portfolio_value = getattr(self.performance_metrics, 'total_value', self.initial_capital)
        
        return {
            'state': self.state.value,
            'cycle_count': self.cycle_count,
            'last_cycle': self.last_cycle_time,
            'current_positions': current_positions,
            'max_positions': self.max_positions,
            'portfolio_value': portfolio_value,
            'total_return': ((portfolio_value - self.initial_capital) / self.initial_capital) if self.initial_capital > 0 else 0,
            'total_errors': self.total_errors,
            'trading_mode': self.trading_mode,
            'tickers_count': len(self.tickers)
        }


# Convenience function to run the trading engine
async def run_trading_engine(
    initial_capital: float = 100000,
    tickers: Optional[List[str]] = None,
    trading_mode: str = "paper"
) -> None:
    """Run the trading engine with specified parameters.
    
    Args:
        initial_capital: Starting capital
        tickers: List of tickers to trade
        trading_mode: Trading mode (paper/live)
    """
    engine = TradingEngine(
        initial_capital=initial_capital,
        tickers=tickers,
        trading_mode=trading_mode
    )
    
    try:
        await engine.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await engine.stop()


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    logger.info("Starting trading engine example")
    asyncio.run(run_trading_engine(
        initial_capital=100000,
        tickers=['NVDA', 'AMD', 'INTC', 'QCOM'],
        trading_mode='paper'
    ))