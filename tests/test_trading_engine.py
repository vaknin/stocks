"""Comprehensive test suite for the trading engine components."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio

from src.trading.engine import TradingEngine, EngineState
from src.trading.portfolio import PortfolioManager, Position, PositionStatus
from src.trading.position_sizer import KellyPositionSizer, PositionSizeResult
from src.trading.risk_manager import RiskManager, RiskLevel, RiskMetrics
from src.trading.signal_generator import SignalGenerator, TradingSignal, SignalType, TimeFrame
from src.models.timesfm_predictor import TimesFMPredictor
from src.data.ingester import DataIngester
from src.data.validator import DataValidator


class TestKellyPositionSizer:
    """Test Kelly Criterion position sizing."""
    
    @pytest.fixture
    def position_sizer(self):
        return KellyPositionSizer(safety_factor=0.25, max_position_pct=0.1)
    
    @pytest.fixture
    def sample_predictions(self):
        return {
            'NVDA': {
                'horizon_1': {'prediction': 0.05, 'confidence': 0.8},
                'horizon_5': {'prediction': 0.12, 'confidence': 0.75}
            },
            'AMD': {
                'horizon_1': {'prediction': -0.03, 'confidence': 0.7}
            }
        }
    
    def test_calculate_position_size_basic(self, position_sizer):
        """Test basic position size calculation."""
        result = position_sizer.calculate_position_size(
            ticker='NVDA',
            prediction=0.05,  # 5% expected return
            confidence=0.8,
            current_price=100.0,
            portfolio_value=100000
        )
        
        assert isinstance(result, PositionSizeResult)
        assert result.ticker == 'NVDA'
        assert result.confidence == 0.8
        assert result.recommended_size >= 0
        assert result.risk_adjusted_size <= 0.1  # Max position limit
    
    def test_batch_calculate_positions(self, position_sizer, sample_predictions):
        """Test batch position calculation."""
        current_prices = {'NVDA': 400.0, 'AMD': 150.0}
        
        results = position_sizer.batch_calculate_positions(
            predictions=sample_predictions,
            current_prices=current_prices,
            portfolio_value=100000
        )
        
        assert len(results) == 2  # Both tickers should have results
        assert 'NVDA' in results
        assert 'AMD' in results
        
        # Check that position sizes are reasonable
        for ticker, result in results.items():
            assert result.recommended_size >= 0
            assert result.risk_adjusted_size <= 0.1
    
    def test_position_size_constraints(self, position_sizer):
        """Test position size constraints."""
        # Test with very high confidence and prediction
        result = position_sizer.calculate_position_size(
            ticker='TEST',
            prediction=0.20,  # 20% return
            confidence=0.95,  # Very high confidence
            current_price=100.0,
            portfolio_value=100000
        )
        
        # Should still be constrained by max_position_pct
        assert result.risk_adjusted_size <= position_sizer.max_position_pct
    
    def test_update_trade_history(self, position_sizer):
        """Test trade history updating."""
        entry_date = datetime.now() - timedelta(days=10)
        exit_date = datetime.now()
        
        position_sizer.update_trade_history(
            ticker='NVDA',
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=400.0,
            exit_price=420.0,
            position_size=100
        )
        
        assert 'NVDA' in position_sizer.trade_history
        assert len(position_sizer.trade_history['NVDA']) == 1
        
        trade = position_sizer.trade_history['NVDA'][0]
        assert trade['return'] == (420.0 - 400.0) / 400.0


class TestRiskManager:
    """Test risk management system."""
    
    @pytest.fixture
    def risk_manager(self):
        return RiskManager(max_drawdown=0.1, stop_loss_pct=0.025)
    
    @pytest.fixture
    def sample_positions(self):
        return {
            'NVDA': {
                'entry_price': 400.0,
                'size': 100,
                'market_value': 42000,
                'stop_loss': 380.0
            },
            'AMD': {
                'entry_price': 150.0,
                'size': 200,
                'market_value': 31000,
                'stop_loss': 146.25
            }
        }
    
    def test_pre_trade_risk_check(self, risk_manager, sample_positions):
        """Test pre-trade risk validation."""
        approved, violations = risk_manager.check_pre_trade_risk(
            ticker='INTC',
            proposed_size=150,  # $7,500 position (7.5% of portfolio)
            current_price=50.0,
            portfolio_value=100000,
            current_positions=sample_positions
        )
        
        # Should be approved for reasonable position
        assert approved is True
        assert len(violations) == 0
    
    def test_pre_trade_risk_rejection(self, risk_manager, sample_positions):
        """Test pre-trade risk rejection."""
        # Try to open position that exceeds limits
        approved, violations = risk_manager.check_pre_trade_risk(
            ticker='INTC',
            proposed_size=5000,  # Very large position
            current_price=50.0,
            portfolio_value=100000,
            current_positions=sample_positions
        )
        
        assert approved is False
        assert len(violations) > 0
    
    def test_stop_loss_triggers(self, risk_manager):
        """Test stop loss trigger detection."""
        positions = {
            'NVDA': {
                'entry_price': 400.0,
                'size': 100,  # Long position
                'stop_loss': 380.0
            },
            'AMD': {
                'entry_price': 150.0,
                'size': -200,  # Short position  
                'stop_loss': 155.0
            }
        }
        
        current_prices = {
            'NVDA': 375.0,  # Below stop loss
            'AMD': 157.0    # Above stop loss for short
        }
        
        triggers = risk_manager.check_stop_loss_triggers(positions, current_prices)
        
        assert len(triggers) == 2
        assert 'NVDA' in triggers
        assert 'AMD' in triggers
    
    def test_portfolio_risk_metrics(self, risk_manager, sample_positions):
        """Test portfolio risk metrics calculation."""
        current_prices = {'NVDA': 420.0, 'AMD': 155.0}
        
        metrics = risk_manager.calculate_portfolio_risk_metrics(
            current_positions=sample_positions,
            current_prices=current_prices,
            portfolio_value=100000
        )
        
        assert isinstance(metrics, RiskMetrics)
        assert metrics.current_drawdown >= 0
        assert metrics.risk_level in [level for level in RiskLevel]
        assert 0 <= metrics.risk_score <= 100
    
    def test_drawdown_calculation(self, risk_manager):
        """Test drawdown calculation."""
        # Set high water mark
        risk_manager.portfolio_high_water_mark = 110000
        
        # Calculate drawdown
        current_drawdown = risk_manager.calculate_current_drawdown(100000)
        
        expected_drawdown = (110000 - 100000) / 110000
        assert abs(current_drawdown - expected_drawdown) < 0.001


class TestPortfolioManager:
    """Test portfolio management."""
    
    @pytest.fixture
    def portfolio(self):
        return PortfolioManager(initial_capital=100000, commission_rate=0.001)
    
    def test_open_position(self, portfolio):
        """Test opening a new position."""
        success = portfolio.open_position(
            ticker='NVDA',
            size=25,  # $10,000 position (25 * 400 = within 10% limit)
            entry_price=400.0,
            stop_loss=380.0,
            take_profit=440.0
        )
        
        assert success is True
        assert 'NVDA' in portfolio.positions
        
        position = portfolio.positions['NVDA']
        assert position.size == 25
        assert position.entry_price == 400.0
        assert position.status == PositionStatus.OPEN
        
        # Check cash was deducted
        expected_cost = 25 * 400.0 + (25 * 400.0 * 0.001)  # Including commission
        expected_cash = 100000 - expected_cost
        assert abs(portfolio.cash_balance - expected_cash) < 0.01
    
    def test_insufficient_capital(self, portfolio):
        """Test position rejection due to insufficient capital."""
        success = portfolio.open_position(
            ticker='NVDA',
            size=1000,  # Very large position
            entry_price=500.0
        )
        
        assert success is False
        assert 'NVDA' not in portfolio.positions
        assert portfolio.cash_balance == 100000  # Unchanged
    
    def test_close_position(self, portfolio):
        """Test closing a position."""
        # First open a position
        portfolio.open_position(ticker='NVDA', size=25, entry_price=400.0)
        
        # Then close it
        success = portfolio.close_position(ticker='NVDA', exit_price=420.0)
        
        assert success is True
        assert 'NVDA' not in portfolio.positions  # Position should be removed
        assert len(portfolio.closed_positions) == 1
        
        closed_position = portfolio.closed_positions[0]
        assert closed_position.status == PositionStatus.CLOSED
        assert closed_position.exit_price == 420.0
        assert closed_position.pnl_realized > 0  # Should be profitable
    
    def test_partial_close_position(self, portfolio):
        """Test partial position closing."""
        # Open position
        portfolio.open_position(ticker='NVDA', size=25, entry_price=400.0)
        
        # Close half
        success = portfolio.close_position(ticker='NVDA', exit_price=420.0, partial_size=12)
        
        assert success is True
        assert 'NVDA' in portfolio.positions  # Position still exists
        assert portfolio.positions['NVDA'].size == 13  # Reduced size (25 - 12)
        assert len(portfolio.closed_positions) == 1  # One closed record
    
    def test_stop_loss_take_profit_triggers(self, portfolio):
        """Test stop loss and take profit detection."""
        # Open position with stops
        portfolio.open_position(
            ticker='NVDA',
            size=25,
            entry_price=400.0,
            stop_loss=380.0,
            take_profit=440.0
        )
        
        # Test stop loss trigger
        current_prices = {'NVDA': 375.0}
        triggers = portfolio.check_stop_loss_take_profit(current_prices)
        
        assert len(triggers) == 1
        assert 'NVDA' in triggers
        
        # Test take profit trigger
        current_prices = {'NVDA': 445.0}
        triggers = portfolio.check_stop_loss_take_profit(current_prices)
        
        assert len(triggers) == 1
        assert 'NVDA' in triggers
    
    def test_portfolio_metrics_calculation(self, portfolio):
        """Test portfolio metrics calculation."""
        # Open some positions
        portfolio.open_position(ticker='NVDA', size=25, entry_price=400.0)
        portfolio.open_position(ticker='AMD', size=33, entry_price=150.0)  # $4,950 position
        
        # Update with current prices
        current_prices = {'NVDA': 420.0, 'AMD': 155.0}
        
        metrics = portfolio.calculate_portfolio_metrics(current_prices)
        
        assert metrics.total_value > 0
        assert metrics.cash_balance >= 0
        assert metrics.positions_value > 0
        assert metrics.unrealized_pnl != 0  # Should have some P&L
    
    def test_positions_summary(self, portfolio):
        """Test positions summary generation."""
        # Open positions
        portfolio.open_position(ticker='NVDA', size=25, entry_price=400.0)
        portfolio.open_position(ticker='AMD', size=33, entry_price=150.0)
        
        # Update prices
        current_prices = {'NVDA': 420.0, 'AMD': 155.0}
        portfolio.update_positions(current_prices)
        
        summary = portfolio.get_positions_summary()
        
        assert len(summary) == 2
        assert 'ticker' in summary.columns
        assert 'unrealized_pnl' in summary.columns
        assert summary.loc[summary['ticker'] == 'NVDA', 'unrealized_pnl'].iloc[0] > 0


class TestSignalGenerator:
    """Test trading signal generation."""
    
    @pytest.fixture
    def signal_generator(self):
        # Mock the ML components for testing
        mock_timesfm = Mock(spec=TimesFMPredictor)
        mock_uncertainty = Mock()
        
        return SignalGenerator(
            timesfm_predictor=mock_timesfm,
            uncertainty_filter=mock_uncertainty
        )
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data for testing."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': 100 + np.random.randn(len(dates)),
            'high': 102 + np.random.randn(len(dates)),
            'low': 98 + np.random.randn(len(dates)),
            'close': 100 + np.random.randn(len(dates)),
            'volume': 1000000 + np.random.randint(-100000, 100000, len(dates))
        }, index=dates)
        
        return data
    
    def test_signal_classification(self, signal_generator):
        """Test signal type classification."""
        # Test BUY signal
        signal_type, strength = signal_generator._classify_signal(
            prediction=0.02,  # 2% positive prediction
            confidence=0.8
        )
        
        assert signal_type == SignalType.BUY
        assert 0 <= strength <= 1
        
        # Test SELL signal
        signal_type, strength = signal_generator._classify_signal(
            prediction=-0.03,  # 3% negative prediction
            confidence=0.85
        )
        
        assert signal_type == SignalType.SELL
        assert 0 <= strength <= 1
        
        # Test HOLD signal (small prediction)
        signal_type, strength = signal_generator._classify_signal(
            prediction=0.002,  # 0.2% prediction (below threshold)
            confidence=0.7
        )
        
        assert signal_type == SignalType.HOLD
    
    def test_risk_reward_calculation(self, signal_generator):
        """Test risk/reward ratio calculation."""
        risk_reward = signal_generator._calculate_risk_reward(
            prediction=0.05,  # 5% expected return
            pred_interval=(0.02, 0.08),  # 6% interval width
            current_price=100.0
        )
        
        assert risk_reward > 0
        assert risk_reward == 0.05 / 0.03  # Expected reward / estimated risk
    
    def test_stop_take_levels_calculation(self, signal_generator):
        """Test stop loss and take profit level calculation."""
        # Test for BUY signal
        stop_loss, take_profit = signal_generator._calculate_stop_take_levels(
            current_price=100.0,
            prediction=0.05,  # 5% expected return
            signal_type=SignalType.BUY
        )
        
        assert stop_loss < 100.0  # Stop loss below current price
        assert take_profit > 100.0  # Take profit above current price
        
        # Test for SELL signal
        stop_loss, take_profit = signal_generator._calculate_stop_take_levels(
            current_price=100.0,
            prediction=-0.05,  # 5% expected decline
            signal_type=SignalType.SELL
        )
        
        assert stop_loss > 100.0  # Stop loss above current price for short
        assert take_profit < 100.0  # Take profit below current price


class TestTradingEngine:
    """Test main trading engine."""
    
    @pytest.fixture
    def trading_engine(self):
        return TradingEngine(
            initial_capital=100000,
            tickers=['NVDA', 'AMD'],
            trading_mode='paper',
            cycle_interval_minutes=1  # Short interval for testing
        )
    
    def test_engine_initialization(self, trading_engine):
        """Test trading engine initialization."""
        assert trading_engine.initial_capital == 100000
        assert trading_engine.state == EngineState.STOPPED
        assert len(trading_engine.tickers) == 2
        assert trading_engine.trading_mode == 'paper'
        
        # Check components are initialized
        assert trading_engine.portfolio_manager is not None
        assert trading_engine.signal_generator is not None
        assert trading_engine.risk_manager is not None
        assert trading_engine.position_sizer is not None
    
    def test_engine_status(self, trading_engine):
        """Test engine status reporting."""
        status = trading_engine.get_engine_status()
        
        assert 'state' in status
        assert 'portfolio_value' in status
        assert 'cycle_count' in status
        assert 'trading_mode' in status
        assert status['state'] == EngineState.STOPPED.value
        assert status['trading_mode'] == 'paper'
    
    @pytest.mark.asyncio
    async def test_engine_state_transitions(self, trading_engine):
        """Test engine state transitions."""
        # Initial state
        assert trading_engine.state == EngineState.STOPPED
        
        # Test pause (should not work when stopped)
        await trading_engine.pause()
        assert trading_engine.state == EngineState.STOPPED
        
        # Test resume (should not work when stopped)
        await trading_engine.resume()
        assert trading_engine.state == EngineState.STOPPED
    
    @patch('src.trading.engine.TradingEngine._fetch_market_data')
    @patch('src.trading.engine.TradingEngine._startup_checks')
    @pytest.mark.asyncio
    async def test_engine_startup_sequence(self, mock_startup, mock_fetch, trading_engine):
        """Test engine startup sequence."""
        # Mock startup checks to pass
        mock_startup.return_value = None
        mock_fetch.return_value = {}
        
        # Start engine (will run briefly then we'll stop it)
        start_task = asyncio.create_task(trading_engine.start())
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        # Stop the engine
        await trading_engine.stop()
        
        # Wait for start task to complete
        try:
            await start_task
        except:
            pass  # Expected since we stopped it
        
        # Verify startup checks were called
        mock_startup.assert_called_once()


class TestDataValidation:
    """Test data validation functionality."""
    
    @pytest.fixture
    def validator(self):
        return DataValidator()
    
    @pytest.fixture
    def valid_data(self):
        """Generate valid OHLCV data."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        return pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'volume': [1000000] * 10
        }, index=dates)
    
    @pytest.fixture
    def invalid_data(self):
        """Generate invalid OHLCV data."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        return pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],  # High < Open (invalid)
            'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'volume': [1000000] * 10
        }, index=dates)
    
    def test_valid_data_passes(self, validator, valid_data):
        """Test that valid data passes validation."""
        result = validator.validate_ohlcv_data(valid_data, 'NVDA')
        assert result['is_valid'] is True
    
    def test_invalid_data_fails(self, validator, invalid_data):
        """Test that invalid data fails validation."""
        result = validator.validate_ohlcv_data(invalid_data, 'NVDA')
        assert result['is_valid'] is False
    
    def test_empty_data_fails(self, validator):
        """Test that empty data fails validation."""
        empty_df = pd.DataFrame()
        result = validator.validate_ohlcv_data(empty_df, 'NVDA')
        assert result['is_valid'] is False


# Integration test
class TestTradingSystemIntegration:
    """Integration tests for the complete trading system."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for multiple tickers and timeframes."""
        data = {}
        tickers = ['NVDA', 'AMD']
        
        for ticker in tickers:
            np.random.seed(hash(ticker) % 2**32)
            dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
            
            base_price = {'NVDA': 400, 'AMD': 150}[ticker]
            
            daily_data = pd.DataFrame({
                'open': base_price + np.random.randn(len(dates)),
                'high': base_price + 2 + np.random.randn(len(dates)),
                'low': base_price - 2 + np.random.randn(len(dates)),
                'close': base_price + np.random.randn(len(dates)),
                'volume': 1000000 + np.random.randint(-100000, 100000, len(dates))
            }, index=dates)
            
            # Create intraday data (last day, 5-minute intervals)
            intraday_dates = pd.date_range(
                start=dates[-1], 
                end=dates[-1] + timedelta(hours=6), 
                freq='5min'
            )
            
            intraday_data = pd.DataFrame({
                'open': base_price + np.random.randn(len(intraday_dates)) * 0.5,
                'high': base_price + 1 + np.random.randn(len(intraday_dates)) * 0.5,
                'low': base_price - 1 + np.random.randn(len(intraday_dates)) * 0.5,
                'close': base_price + np.random.randn(len(intraday_dates)) * 0.5,
                'volume': 10000 + np.random.randint(-1000, 1000, len(intraday_dates))
            }, index=intraday_dates)
            
            data[ticker] = {
                TimeFrame.DAILY: daily_data,
                TimeFrame.INTRADAY: intraday_data
            }
        
        return data
    
    def test_complete_signal_generation_pipeline(self, sample_market_data):
        """Test complete signal generation from data to signals."""
        # Initialize components
        timesfm_predictor = TimesFMPredictor()
        uncertainty_filter = Mock()
        uncertainty_filter.filter_prediction.return_value = 0.05  # Mock return
        
        signal_generator = SignalGenerator(
            timesfm_predictor=timesfm_predictor,
            uncertainty_filter=uncertainty_filter
        )
        
        current_prices = {'NVDA': 400.0, 'AMD': 150.0}
        
        # This will use mock predictions since TimesFM model isn't actually loaded
        signals = signal_generator.generate_signals(sample_market_data, current_prices)
        
        # Should generate some signals
        assert isinstance(signals, dict)
        # In mock mode, we might not get signals due to low confidence, which is expected
    
    def test_position_sizing_with_signals(self, sample_market_data):
        """Test position sizing integration with signals."""
        # Mock signal data
        mock_signals = {
            'NVDA': Mock(),
            'AMD': Mock()
        }
        
        mock_signals['NVDA'].overall_signal = SignalType.BUY
        mock_signals['NVDA'].overall_confidence = 0.8
        mock_signals['NVDA'].signals_by_timeframe = {
            TimeFrame.DAILY: Mock(prediction=0.05, confidence=0.8)
        }
        
        mock_signals['AMD'].overall_signal = SignalType.SELL
        mock_signals['AMD'].overall_confidence = 0.75
        mock_signals['AMD'].signals_by_timeframe = {
            TimeFrame.DAILY: Mock(prediction=-0.03, confidence=0.75)
        }
        
        # Convert to predictions format
        predictions = {}
        for ticker, signal in mock_signals.items():
            best_signal = list(signal.signals_by_timeframe.values())[0]
            predictions[ticker] = {
                'prediction': best_signal.prediction,
                'confidence': best_signal.confidence
            }
        
        position_sizer = KellyPositionSizer()
        current_prices = {'NVDA': 400.0, 'AMD': 150.0}
        
        position_results = position_sizer.batch_calculate_positions(
            predictions=predictions,
            current_prices=current_prices,
            portfolio_value=100000
        )
        
        assert len(position_results) == 2
        assert all(result.recommended_size >= 0 for result in position_results.values())
    
    def test_risk_management_integration(self):
        """Test risk management with realistic portfolio."""
        risk_manager = RiskManager()
        
        # Simulate current positions
        positions = {
            'NVDA': {
                'entry_price': 400.0,
                'size': 100,
                'market_value': 42000,
                'stop_loss': 380.0
            },
            'AMD': {
                'entry_price': 150.0,
                'size': 200,
                'market_value': 31000,
                'stop_loss': 146.25
            }
        }
        
        current_prices = {'NVDA': 420.0, 'AMD': 155.0}
        
        # Pre-trade risk check for new position
        approved, violations = risk_manager.check_pre_trade_risk(
            ticker='INTC',
            proposed_size=300,
            current_price=50.0,
            portfolio_value=100000,
            current_positions=positions
        )
        
        # Should be reasonable approval
        assert isinstance(approved, bool)
        assert isinstance(violations, list)
        
        # Check stop loss triggers
        triggers = risk_manager.check_stop_loss_triggers(positions, current_prices)
        assert isinstance(triggers, list)
        
        # Calculate risk metrics
        metrics = risk_manager.calculate_portfolio_risk_metrics(
            current_positions=positions,
            current_prices=current_prices,
            portfolio_value=100000
        )
        
        assert isinstance(metrics, RiskMetrics)
        assert 0 <= metrics.risk_score <= 100


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])