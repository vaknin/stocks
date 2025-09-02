"""Unit tests for trading components."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trading.position_sizer import KellyPositionSizer, PositionSizeResult
from trading.risk_manager import RiskManager, RiskLevel, RiskMetrics
from trading.portfolio_tracker import PortfolioTracker, Position


class TestKellyPositionSizer(unittest.TestCase):
    """Test Kelly Criterion position sizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.position_sizer = KellyPositionSizer(
            safety_factor=0.40,
            max_position_pct=0.15,
            min_confidence=0.60
        )
    
    def test_initialization(self):
        """Test position sizer initialization."""
        self.assertEqual(self.position_sizer.safety_factor, 0.40)
        self.assertEqual(self.position_sizer.max_position_pct, 0.15)
        self.assertEqual(self.position_sizer.min_confidence, 0.60)
    
    def test_position_size_calculation(self):
        """Test basic position size calculation."""
        result = self.position_sizer.calculate_position_size(
            ticker="TEST",
            prediction=0.05,  # 5% expected return
            confidence=0.80,  # 80% confidence
            current_price=100.0,
            portfolio_value=10000.0
        )
        
        self.assertIsInstance(result, PositionSizeResult)
        self.assertEqual(result.ticker, "TEST")
        self.assertGreaterEqual(result.recommended_size, 0)
        self.assertLessEqual(result.risk_adjusted_size, self.position_sizer.max_position_pct)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
    
    def test_transaction_cost_adjustment(self):
        """Test that transaction costs reduce position sizes."""
        # Test with and without transaction costs
        result_no_costs = self.position_sizer.calculate_position_size(
            ticker="TEST",
            prediction=0.05,
            confidence=0.80,
            current_price=100.0,
            portfolio_value=10000.0,
            transaction_cost_pct=0.0
        )
        
        result_with_costs = self.position_sizer.calculate_position_size(
            ticker="TEST",
            prediction=0.05,
            confidence=0.80,
            current_price=100.0,
            portfolio_value=10000.0,
            transaction_cost_pct=0.002
        )
        
        # Position size should be smaller with transaction costs
        self.assertLessEqual(result_with_costs.recommended_size, result_no_costs.recommended_size)
    
    def test_confidence_threshold(self):
        """Test that low confidence predictions are filtered out."""
        predictions = {
            'HIGH_CONF': {'prediction': 0.03, 'confidence': 0.80},
            'LOW_CONF': {'prediction': 0.03, 'confidence': 0.50}
        }
        
        current_prices = {'HIGH_CONF': 100.0, 'LOW_CONF': 100.0}
        
        results = self.position_sizer.batch_calculate_positions(
            predictions=predictions,
            current_prices=current_prices,
            portfolio_value=10000.0
        )
        
        # Should only have the high confidence position
        self.assertIn('HIGH_CONF', results)
        self.assertNotIn('LOW_CONF', results)
    
    def test_max_position_limit(self):
        """Test maximum position size enforcement."""
        result = self.position_sizer.calculate_position_size(
            ticker="TEST",
            prediction=0.20,  # Very high prediction
            confidence=0.95,  # Very high confidence
            current_price=100.0,
            portfolio_value=10000.0
        )
        
        # Should be capped at max_position_pct
        self.assertLessEqual(result.risk_adjusted_size, self.position_sizer.max_position_pct)
    
    def test_trade_history_update(self):
        """Test trade history tracking."""
        entry_date = datetime.now() - timedelta(days=10)
        exit_date = datetime.now()
        
        self.position_sizer.update_trade_history(
            ticker="TEST",
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=100.0,
            exit_price=105.0,
            position_size=100
        )
        
        self.assertIn("TEST", self.position_sizer.trade_history)
        self.assertEqual(len(self.position_sizer.trade_history["TEST"]), 1)
        
        trade = self.position_sizer.trade_history["TEST"][0]
        self.assertEqual(trade['entry_price'], 100.0)
        self.assertEqual(trade['exit_price'], 105.0)
        self.assertEqual(trade['return'], 0.05)  # 5% return


class TestRiskManager(unittest.TestCase):
    """Test risk management system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.risk_manager = RiskManager(
            max_drawdown=0.15,
            stop_loss_pct=0.025,
            max_position_pct=0.15,
            max_correlation=0.70
        )
        
        # Mock current positions
        self.current_positions = {
            'AAPL': {
                'size': 100,
                'entry_price': 150.0,
                'market_value': 15000.0,
                'stop_loss': 145.0
            },
            'MSFT': {
                'size': 50,
                'entry_price': 200.0,
                'market_value': 10000.0,
                'stop_loss': 190.0
            }
        }
        
        self.current_prices = {'AAPL': 155.0, 'MSFT': 205.0}
    
    def test_initialization(self):
        """Test risk manager initialization."""
        self.assertEqual(self.risk_manager.max_drawdown, 0.15)
        self.assertEqual(self.risk_manager.stop_loss_pct, 0.025)
    
    def test_pre_trade_risk_check(self):
        """Test pre-trade risk assessment."""
        approved, violations = self.risk_manager.check_pre_trade_risk(
            ticker="GOOGL",
            proposed_size=30,  # $3000 position
            current_price=100.0,
            portfolio_value=100000.0,
            current_positions=self.current_positions
        )
        
        self.assertIsInstance(approved, bool)
        self.assertIsInstance(violations, list)
        
        # Small position should be approved
        self.assertTrue(approved)
    
    def test_position_size_violation(self):
        """Test position size limit violation."""
        approved, violations = self.risk_manager.check_pre_trade_risk(
            ticker="GOOGL",
            proposed_size=200,  # $20,000 position (20% of $100k portfolio)
            current_price=100.0,
            portfolio_value=100000.0,
            current_positions={}
        )
        
        self.assertFalse(approved)  # Should be rejected
        self.assertGreater(len(violations), 0)  # Should have violations
    
    def test_stop_loss_triggers(self):
        """Test stop-loss trigger detection."""
        # Simulate price drops
        falling_prices = {'AAPL': 144.0, 'MSFT': 189.0}  # Both below stop loss
        
        triggers = self.risk_manager.check_stop_loss_triggers(
            current_positions=self.current_positions,
            current_prices=falling_prices
        )
        
        self.assertEqual(len(triggers), 2)  # Both positions should trigger
        self.assertIn('AAPL', triggers)
        self.assertIn('MSFT', triggers)
    
    def test_portfolio_risk_metrics(self):
        """Test portfolio risk metric calculation."""
        # Set up a high water mark
        self.risk_manager.portfolio_high_water_mark = 110000.0
        
        risk_metrics = self.risk_manager.calculate_portfolio_risk_metrics(
            current_positions=self.current_positions,
            current_prices=self.current_prices,
            portfolio_value=100000.0
        )
        
        self.assertIsInstance(risk_metrics, RiskMetrics)
        self.assertGreaterEqual(risk_metrics.current_drawdown, 0.0)
        self.assertIsInstance(risk_metrics.risk_level, RiskLevel)
        self.assertGreaterEqual(risk_metrics.risk_score, 0.0)
        self.assertLessEqual(risk_metrics.risk_score, 100.0)
    
    def test_daily_returns_tracking(self):
        """Test daily returns tracking."""
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        
        for ret in returns:
            self.risk_manager.update_daily_returns(ret)
        
        self.assertEqual(len(self.risk_manager.daily_returns), len(returns))
        self.assertEqual(self.risk_manager.daily_returns[-1], 0.015)
    
    def test_risk_level_classification(self):
        """Test risk level classification."""
        # Test different risk scores
        low_risk = self.risk_manager._classify_risk_level(20)
        medium_risk = self.risk_manager._classify_risk_level(40)
        high_risk = self.risk_manager._classify_risk_level(65)
        critical_risk = self.risk_manager._classify_risk_level(85)
        
        self.assertEqual(low_risk, RiskLevel.LOW)
        self.assertEqual(medium_risk, RiskLevel.MEDIUM)
        self.assertEqual(high_risk, RiskLevel.HIGH)
        self.assertEqual(critical_risk, RiskLevel.CRITICAL)


class TestPortfolioTracker(unittest.TestCase):
    """Test portfolio tracking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.portfolio_tracker = PortfolioTracker(initial_capital=100000.0)
    
    def test_initialization(self):
        """Test portfolio tracker initialization."""
        self.assertEqual(self.portfolio_tracker.initial_capital, 100000.0)
        self.assertEqual(self.portfolio_tracker.cash_balance, 100000.0)
        self.assertEqual(len(self.portfolio_tracker.positions), 0)
    
    def test_position_opening(self):
        """Test opening a new position."""
        success = self.portfolio_tracker.add_position(
            ticker="AAPL",
            size=100,
            entry_price=150.0,
            current_price=150.0
        )
        
        self.assertTrue(success)
        self.assertIn("AAPL", self.portfolio_tracker.positions)
        
        position = self.portfolio_tracker.positions["AAPL"]
        self.assertEqual(position.size, 100)
        self.assertEqual(position.entry_price, 150.0)
        self.assertEqual(position.market_value, 15000.0)
    
    def test_position_closing(self):
        """Test closing a position."""
        # First open a position
        self.portfolio_tracker.add_position(
            ticker="AAPL",
            size=100,
            entry_price=150.0,
            current_price=150.0
        )
        
        # Then close it
        pnl = self.portfolio_tracker.close_position(
            ticker="AAPL",
            exit_price=155.0
        )
        
        self.assertIsNotNone(pnl)
        self.assertEqual(pnl, 500.0)  # (155-150) * 100
        self.assertNotIn("AAPL", self.portfolio_tracker.positions)
    
    def test_portfolio_value_calculation(self):
        """Test total portfolio value calculation."""
        # Add positions
        self.portfolio_tracker.add_position("AAPL", 100, 150.0, 155.0)
        self.portfolio_tracker.add_position("MSFT", 50, 200.0, 210.0)
        
        total_value = self.portfolio_tracker.get_total_portfolio_value()
        expected_value = self.portfolio_tracker.cash_balance + 15500 + 10500  # Position values
        
        self.assertEqual(total_value, expected_value)
    
    def test_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        self.portfolio_tracker.add_position("AAPL", 100, 150.0, 155.0)
        
        unrealized_pnl = self.portfolio_tracker.get_unrealized_pnl()
        self.assertEqual(unrealized_pnl, 500.0)  # (155-150) * 100
    
    def test_position_updates(self):
        """Test position price updates."""
        self.portfolio_tracker.add_position("AAPL", 100, 150.0, 150.0)
        
        # Update price
        self.portfolio_tracker.update_position_prices({"AAPL": 160.0})
        
        position = self.portfolio_tracker.positions["AAPL"]
        self.assertEqual(position.current_price, 160.0)
        self.assertEqual(position.market_value, 16000.0)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestKellyPositionSizer))
    test_suite.addTest(unittest.makeSuite(TestRiskManager))
    test_suite.addTest(unittest.makeSuite(TestPortfolioTracker))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"\n✅ All trading tests passed! ({result.testsRun} tests)")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests")