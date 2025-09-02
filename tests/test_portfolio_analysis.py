"""
Portfolio Analysis Testing Suite - Risk Assessment and Position Sizing Validation

Tests the portfolio analysis system's ability to provide accurate risk assessment,
position sizing recommendations (Kelly criterion), diversification analysis,
and portfolio rebalancing suggestions for reliable investment advice.

Priority: TIER 1 - Critical for investment recommendation system risk management
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.trading.portfolio_analyzer import PortfolioAnalyzer, PortfolioPosition
from src.trading.risk_manager import RiskManager
from src.config.settings import config


class TestPortfolioAnalysis:
    """Test suite for validating portfolio analysis and risk assessment."""
    
    @pytest.fixture(scope="class")
    def sample_portfolio_data(self):
        """Generate realistic portfolio data for testing."""
        np.random.seed(42)
        
        # Sample portfolio positions
        positions = [
            PortfolioPosition(
                ticker='AAPL',
                shares=100,
                entry_price=150.0,
                current_price=165.0,
                entry_date=datetime(2023, 1, 15),
                sector='Technology',
                position_value=16500.0
            ),
            PortfolioPosition(
                ticker='MSFT',
                shares=75,
                entry_price=280.0,
                current_price=290.0,
                entry_date=datetime(2023, 2, 1),
                sector='Technology',
                position_value=21750.0
            ),
            PortfolioPosition(
                ticker='JPM',
                shares=50,
                entry_price=140.0,
                current_price=145.0,
                entry_date=datetime(2023, 1, 20),
                sector='Financials',
                position_value=7250.0
            ),
            PortfolioPosition(
                ticker='JNJ',
                shares=80,
                entry_price=160.0,
                current_price=155.0,
                entry_date=datetime(2023, 2, 10),
                sector='Healthcare',
                position_value=12400.0
            )
        ]
        
        return positions
    
    @pytest.fixture(scope="class")
    def price_history_data(self):
        """Generate price history for portfolio stocks."""
        np.random.seed(42)
        
        tickers = ['AAPL', 'MSFT', 'JPM', 'JNJ']
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        
        price_data = {}
        base_prices = {'AAPL': 120, 'MSFT': 250, 'JPM': 130, 'JNJ': 155}
        
        for ticker in tickers:
            # Generate correlated returns with different volatilities
            base_vol = {'AAPL': 0.25, 'MSFT': 0.22, 'JPM': 0.28, 'JNJ': 0.15}[ticker]
            returns = np.random.normal(0.0008, base_vol/np.sqrt(252), len(dates))
            
            # Add some correlation structure
            if ticker == 'MSFT':  # Correlate MSFT with AAPL
                market_factor = returns * 0.6  # 60% common market factor
                idiosyncratic = np.random.normal(0, base_vol/np.sqrt(252)*0.8, len(dates))
                returns = market_factor + idiosyncratic
            
            prices = base_prices[ticker] * np.cumprod(1 + returns)
            
            price_data[ticker] = pd.DataFrame({
                'date': dates,
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, len(dates))
            }).set_index('date')
        
        return price_data
    
    @pytest.fixture
    def portfolio_analyzer(self, sample_portfolio_data, price_history_data):
        """Create portfolio analyzer with sample data."""
        analyzer = PortfolioAnalyzer(
            starting_capital=100000.0,
            max_position_size=0.15,
            target_volatility=0.18
        )
        
        # Set portfolio positions
        analyzer.positions = {pos.ticker: pos for pos in sample_portfolio_data}
        
        # Set price history
        analyzer.price_history = price_history_data
        
        return analyzer
    
    def test_portfolio_composition_analysis(self, portfolio_analyzer):
        """Test portfolio composition and diversification analysis."""
        composition = portfolio_analyzer.get_portfolio_composition()
        
        # Validate composition structure
        assert isinstance(composition, dict), "Composition should be a dictionary"
        assert 'by_sector' in composition, "Missing sector breakdown"
        assert 'by_position' in composition, "Missing position breakdown"
        assert 'total_value' in composition, "Missing total portfolio value"
        assert 'cash_allocation' in composition, "Missing cash allocation"
        
        # Validate sector diversification
        sector_breakdown = composition['by_sector']
        assert isinstance(sector_breakdown, dict), "Sector breakdown should be dict"
        
        # Check that Technology sector is properly weighted (has 2 stocks)
        if 'Technology' in sector_breakdown:
            tech_weight = sector_breakdown['Technology']['weight']
            assert 0.0 <= tech_weight <= 1.0, f"Invalid tech sector weight: {tech_weight}"
            
            # Should be significant portion due to 2 large positions
            assert tech_weight > 0.3, f"Technology sector should be substantial: {tech_weight:.1%}"
        
        # Validate position-level analysis
        position_breakdown = composition['by_position']
        assert isinstance(position_breakdown, dict), "Position breakdown should be dict"
        
        for ticker, position_info in position_breakdown.items():
            assert 'weight' in position_info, f"Missing weight for {ticker}"
            assert 'value' in position_info, f"Missing value for {ticker}"
            assert 'pnl' in position_info, f"Missing P&L for {ticker}"
            
            weight = position_info['weight']
            assert 0.0 <= weight <= 1.0, f"Invalid weight for {ticker}: {weight}"
        
        # Validate total portfolio value matches sum of positions
        total_value = composition['total_value']
        sum_positions = sum(pos.position_value for pos in portfolio_analyzer.positions.values())
        assert abs(total_value - sum_positions) < 1.0, f"Total value mismatch: {total_value} vs {sum_positions}"
    
    def test_kelly_criterion_position_sizing(self, portfolio_analyzer):
        """Test Kelly criterion position sizing recommendations."""
        # Mock prediction data for Kelly calculation
        mock_prediction = {
            'expected_return': 0.12,      # 12% expected annual return
            'win_probability': 0.65,      # 65% win rate
            'avg_win': 0.08,              # 8% average win
            'avg_loss': 0.05,             # 5% average loss
            'volatility': 0.25,           # 25% volatility
            'confidence_interval': (0.05, 0.20)  # 5-20% return range
        }
        
        kelly_recommendation = portfolio_analyzer.calculate_kelly_position_size(
            ticker='NVDA',
            prediction_data=mock_prediction,
            current_price=400.0
        )
        
        # Validate Kelly recommendation structure
        assert isinstance(kelly_recommendation, dict), "Kelly result should be dict"
        assert 'optimal_fraction' in kelly_recommendation, "Missing optimal fraction"
        assert 'recommended_shares' in kelly_recommendation, "Missing recommended shares"
        assert 'position_value' in kelly_recommendation, "Missing position value"
        assert 'kelly_score' in kelly_recommendation, "Missing Kelly score"
        
        # Validate Kelly fraction bounds
        kelly_fraction = kelly_recommendation['optimal_fraction']
        assert 0.0 <= kelly_fraction <= 1.0, f"Kelly fraction out of bounds: {kelly_fraction}"
        
        # With reasonable prediction data, Kelly should recommend meaningful position
        if mock_prediction['expected_return'] > 0 and mock_prediction['win_probability'] > 0.5:
            assert kelly_fraction > 0.02, f"Kelly fraction too small for good prediction: {kelly_fraction:.1%}"
        
        # Should respect maximum position size constraints
        max_position_fraction = portfolio_analyzer.max_position_size
        assert kelly_fraction <= max_position_fraction * 1.1, f"Kelly exceeds max position size: {kelly_fraction:.1%}"
        
        # Validate recommended shares calculation
        recommended_shares = kelly_recommendation['recommended_shares']
        portfolio_value = portfolio_analyzer.get_total_portfolio_value()
        expected_position_value = portfolio_value * kelly_fraction
        calculated_shares = int(expected_position_value / mock_prediction.get('current_price', 400.0))
        
        assert abs(recommended_shares - calculated_shares) <= 1, "Share calculation error"
    
    def test_risk_assessment_calculations(self, portfolio_analyzer):
        """Test comprehensive portfolio risk assessment."""
        risk_assessment = portfolio_analyzer.calculate_portfolio_risk()
        
        # Validate risk assessment structure
        assert isinstance(risk_assessment, dict), "Risk assessment should be dict"
        assert 'portfolio_volatility' in risk_assessment, "Missing portfolio volatility"
        assert 'var_95' in risk_assessment, "Missing VaR calculation"
        assert 'correlation_matrix' in risk_assessment, "Missing correlation matrix"
        assert 'concentration_risk' in risk_assessment, "Missing concentration risk"
        assert 'sector_diversification' in risk_assessment, "Missing sector diversification"
        
        # Validate portfolio volatility
        portfolio_vol = risk_assessment['portfolio_volatility']
        assert 0.0 < portfolio_vol < 1.0, f"Invalid portfolio volatility: {portfolio_vol}"
        assert portfolio_vol < 0.5, f"Portfolio volatility too high: {portfolio_vol:.1%}"
        
        # Validate Value at Risk (VaR)
        var_95 = risk_assessment['var_95']
        assert isinstance(var_95, dict), "VaR should be detailed breakdown"
        assert 'daily_var' in var_95, "Missing daily VaR"
        assert 'monthly_var' in var_95, "Missing monthly VaR"
        
        daily_var = var_95['daily_var']
        assert daily_var < 0, "Daily VaR should be negative (loss)"
        assert daily_var > -0.15, f"Daily VaR too extreme: {daily_var:.1%}"
        
        # Validate correlation matrix
        corr_matrix = risk_assessment['correlation_matrix']
        assert isinstance(corr_matrix, (pd.DataFrame, np.ndarray, dict)), "Correlation matrix invalid type"
        
        if isinstance(corr_matrix, pd.DataFrame):
            # All correlations should be between -1 and 1
            corr_values = corr_matrix.values.flatten()
            assert all(-1.0 <= corr <= 1.0 for corr in corr_values), "Invalid correlation values"
            
            # Diagonal should be 1.0 (self-correlation)
            diagonal_values = np.diag(corr_matrix.values)
            assert all(abs(val - 1.0) < 0.01 for val in diagonal_values), "Diagonal should be 1.0"
        
        # Validate concentration risk
        concentration_risk = risk_assessment['concentration_risk']
        assert isinstance(concentration_risk, dict), "Concentration risk should be dict"
        assert 'max_position_weight' in concentration_risk, "Missing max position weight"
        assert 'top3_concentration' in concentration_risk, "Missing top 3 concentration"
        
        max_weight = concentration_risk['max_position_weight']
        assert 0.0 <= max_weight <= 1.0, f"Invalid max position weight: {max_weight}"
        
        # Validate sector diversification
        sector_div = risk_assessment['sector_diversification']
        assert isinstance(sector_div, dict), "Sector diversification should be dict"
        assert 'num_sectors' in sector_div, "Missing number of sectors"
        assert 'sector_weights' in sector_div, "Missing sector weights"
        
        num_sectors = sector_div['num_sectors']
        assert num_sectors > 0, "Should have at least one sector"
        assert num_sectors <= 20, f"Too many sectors: {num_sectors}"  # Sanity check
    
    def test_correlation_analysis_validation(self, portfolio_analyzer):
        """Test correlation analysis between portfolio positions."""
        # Ensure we have price history for correlation calculation
        if len(portfolio_analyzer.price_history) < 2:
            pytest.skip("Need at least 2 stocks for correlation analysis")
        
        correlation_analysis = portfolio_analyzer.analyze_correlations()
        
        # Validate correlation analysis structure
        assert isinstance(correlation_analysis, dict), "Correlation analysis should be dict"
        assert 'correlation_matrix' in correlation_analysis, "Missing correlation matrix"
        assert 'high_correlations' in correlation_analysis, "Missing high correlation pairs"
        assert 'diversification_score' in correlation_analysis, "Missing diversification score"
        
        # Validate correlation matrix properties
        corr_matrix = correlation_analysis['correlation_matrix']
        if isinstance(corr_matrix, pd.DataFrame):
            # Matrix should be square
            assert corr_matrix.shape[0] == corr_matrix.shape[1], "Correlation matrix should be square"
            
            # Check for reasonable correlations (no perfect correlations except diagonal)
            off_diagonal_mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            off_diagonal_corrs = corr_matrix.values[off_diagonal_mask]
            
            # Should not have perfect correlations (except self)
            perfect_corrs = np.abs(off_diagonal_corrs) > 0.99
            assert np.sum(perfect_corrs) == 0, f"Found {np.sum(perfect_corrs)} perfect correlations"
        
        # Validate high correlation identification
        high_corrs = correlation_analysis['high_correlations']
        assert isinstance(high_corrs, list), "High correlations should be list"
        
        for corr_pair in high_corrs:
            assert 'ticker1' in corr_pair, "Missing ticker1 in correlation pair"
            assert 'ticker2' in corr_pair, "Missing ticker2 in correlation pair"
            assert 'correlation' in corr_pair, "Missing correlation value"
            
            correlation = corr_pair['correlation']
            assert -1.0 <= correlation <= 1.0, f"Invalid correlation: {correlation}"
            assert abs(correlation) >= 0.7, f"High correlation should be >= 0.7: {abs(correlation)}"
        
        # Validate diversification score
        div_score = correlation_analysis['diversification_score']
        assert 0.0 <= div_score <= 1.0, f"Invalid diversification score: {div_score}"
        
        # Higher diversification score = better diversified
        # If we have positions in different sectors, score should be reasonable
        if len(set(pos.sector for pos in portfolio_analyzer.positions.values())) > 1:
            assert div_score > 0.2, f"Diversification score too low for multi-sector portfolio: {div_score}"
    
    def test_rebalancing_recommendations(self, portfolio_analyzer):
        """Test portfolio rebalancing recommendations."""
        # Define target allocation
        target_allocation = {
            'Technology': 0.40,
            'Financials': 0.25,
            'Healthcare': 0.20,
            'Consumer': 0.15
        }
        
        rebalancing_recs = portfolio_analyzer.get_rebalancing_recommendations(
            target_allocation=target_allocation,
            rebalancing_threshold=0.05  # 5% deviation threshold
        )
        
        # Validate rebalancing recommendation structure
        assert isinstance(rebalancing_recs, dict), "Rebalancing recommendations should be dict"
        assert 'current_allocation' in rebalancing_recs, "Missing current allocation"
        assert 'target_allocation' in rebalancing_recs, "Missing target allocation"
        assert 'rebalancing_actions' in rebalancing_recs, "Missing rebalancing actions"
        assert 'deviation_score' in rebalancing_recs, "Missing deviation score"
        
        # Validate current allocation
        current_alloc = rebalancing_recs['current_allocation']
        assert isinstance(current_alloc, dict), "Current allocation should be dict"
        
        # Allocation weights should sum to approximately 1.0
        if current_alloc:
            total_weight = sum(current_alloc.values())
            assert 0.95 <= total_weight <= 1.05, f"Allocation weights should sum to ~1.0: {total_weight}"
        
        # Validate rebalancing actions
        actions = rebalancing_recs['rebalancing_actions']
        assert isinstance(actions, list), "Rebalancing actions should be list"
        
        for action in actions:
            assert 'action' in action, "Missing action type"
            assert 'sector' in action, "Missing sector"
            assert 'amount' in action, "Missing amount"
            assert 'reason' in action, "Missing reason"
            
            assert action['action'] in ['BUY', 'SELL', 'HOLD'], f"Invalid action: {action['action']}"
            
            amount = action['amount']
            assert amount >= 0, f"Amount should be non-negative: {amount}"
        
        # Validate deviation score
        deviation_score = rebalancing_recs['deviation_score']
        assert 0.0 <= deviation_score <= 1.0, f"Invalid deviation score: {deviation_score}"
    
    def test_position_sizing_constraints(self, portfolio_analyzer):
        """Test position sizing adheres to risk management constraints."""
        # Test maximum position size constraint
        large_prediction = {
            'expected_return': 0.50,      # 50% expected return (very optimistic)
            'win_probability': 0.90,      # 90% win rate (very optimistic)
            'avg_win': 0.20,
            'avg_loss': 0.05,
            'volatility': 0.15,           # Low volatility
            'confidence_interval': (0.30, 0.70)
        }
        
        kelly_result = portfolio_analyzer.calculate_kelly_position_size(
            ticker='GROWTH_STOCK',
            prediction_data=large_prediction,
            current_price=100.0
        )
        
        # Even with very optimistic predictions, should respect max position size
        recommended_fraction = kelly_result['optimal_fraction']
        max_allowed = portfolio_analyzer.max_position_size
        
        assert recommended_fraction <= max_allowed, (
            f"Position size {recommended_fraction:.1%} exceeds max allowed {max_allowed:.1%}"
        )
        
        # Test minimum position size (should not recommend tiny positions)
        small_prediction = {
            'expected_return': 0.02,      # 2% expected return
            'win_probability': 0.51,      # Barely above random
            'avg_win': 0.03,
            'avg_loss': 0.03,
            'volatility': 0.30,           # High volatility
            'confidence_interval': (-0.01, 0.05)
        }
        
        kelly_result_small = portfolio_analyzer.calculate_kelly_position_size(
            ticker='RISKY_STOCK',
            prediction_data=small_prediction,
            current_price=50.0
        )
        
        small_fraction = kelly_result_small['optimal_fraction']
        # Should either recommend 0 or meaningful position (not tiny amounts)
        assert small_fraction == 0.0 or small_fraction >= 0.01, (
            f"Position size should be 0 or >= 1%: {small_fraction:.1%}"
        )
    
    def test_portfolio_value_tracking(self, portfolio_analyzer):
        """Test accurate portfolio value and P&L tracking."""
        total_value = portfolio_analyzer.get_total_portfolio_value()
        
        # Should match sum of individual position values
        expected_total = sum(pos.position_value for pos in portfolio_analyzer.positions.values())
        assert abs(total_value - expected_total) < 1.0, f"Total value mismatch: {total_value} vs {expected_total}"
        
        # Test P&L calculations
        pnl_summary = portfolio_analyzer.calculate_pnl_summary()
        
        assert isinstance(pnl_summary, dict), "P&L summary should be dict"
        assert 'total_pnl' in pnl_summary, "Missing total P&L"
        assert 'total_pnl_pct' in pnl_summary, "Missing total P&L percentage"
        assert 'realized_pnl' in pnl_summary, "Missing realized P&L"
        assert 'unrealized_pnl' in pnl_summary, "Missing unrealized P&L"
        assert 'by_position' in pnl_summary, "Missing position-level P&L"
        
        # Validate individual position P&L
        position_pnl = pnl_summary['by_position']
        for ticker, pos_data in position_pnl.items():
            assert 'pnl' in pos_data, f"Missing P&L for {ticker}"
            assert 'pnl_pct' in pos_data, f"Missing P&L percentage for {ticker}"
            assert 'current_value' in pos_data, f"Missing current value for {ticker}"
            assert 'cost_basis' in pos_data, f"Missing cost basis for {ticker}"
        
        # Total P&L should match sum of position P&Ls
        total_pnl_check = sum(pos_data['pnl'] for pos_data in position_pnl.values())
        assert abs(pnl_summary['total_pnl'] - total_pnl_check) < 1.0, "P&L calculation mismatch"
    
    def test_sector_exposure_analysis(self, portfolio_analyzer):
        """Test sector exposure analysis and limits."""
        sector_analysis = portfolio_analyzer.analyze_sector_exposure()
        
        assert isinstance(sector_analysis, dict), "Sector analysis should be dict"
        assert 'sector_weights' in sector_analysis, "Missing sector weights"
        assert 'sector_risk_metrics' in sector_analysis, "Missing sector risk metrics"
        assert 'concentration_warnings' in sector_analysis, "Missing concentration warnings"
        
        # Validate sector weights
        sector_weights = sector_analysis['sector_weights']
        assert isinstance(sector_weights, dict), "Sector weights should be dict"
        
        total_weight = sum(sector_weights.values())
        assert 0.95 <= total_weight <= 1.05, f"Sector weights should sum to ~1.0: {total_weight}"
        
        for sector, weight in sector_weights.items():
            assert 0.0 <= weight <= 1.0, f"Invalid sector weight for {sector}: {weight}"
        
        # Check for concentration warnings
        warnings = sector_analysis['concentration_warnings']
        assert isinstance(warnings, list), "Concentration warnings should be list"
        
        # If any single sector > 50%, should warn
        max_sector_weight = max(sector_weights.values()) if sector_weights else 0
        if max_sector_weight > 0.5:
            assert any('concentration' in warning.lower() for warning in warnings), \
                "Should warn about sector concentration"
        
        # Validate sector risk metrics
        sector_risks = sector_analysis['sector_risk_metrics']
        assert isinstance(sector_risks, dict), "Sector risk metrics should be dict"
        
        for sector, metrics in sector_risks.items():
            if isinstance(metrics, dict):
                # Should have risk-related metrics
                expected_metrics = ['volatility', 'correlation', 'contribution_to_risk']
                for metric in expected_metrics:
                    if metric in metrics:
                        value = metrics[metric]
                        assert isinstance(value, (int, float)), f"Invalid {metric} for {sector}: {value}"
    
    def test_drawdown_analysis(self, portfolio_analyzer):
        """Test portfolio drawdown analysis and risk assessment."""
        # Mock historical portfolio values for drawdown calculation
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Simulate portfolio value history with drawdowns
        base_value = 100000
        returns = np.random.normal(0.0005, 0.015, 100)
        # Add a significant drawdown period
        returns[30:50] = np.random.normal(-0.005, 0.025, 20)  # Bad period
        
        portfolio_values = base_value * np.cumprod(1 + returns)
        
        # Mock the portfolio history data
        portfolio_analyzer.portfolio_history = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_values,
            'daily_return': returns
        }).set_index('date')
        
        drawdown_analysis = portfolio_analyzer.analyze_drawdowns()
        
        assert isinstance(drawdown_analysis, dict), "Drawdown analysis should be dict"
        assert 'max_drawdown' in drawdown_analysis, "Missing max drawdown"
        assert 'current_drawdown' in drawdown_analysis, "Missing current drawdown"
        assert 'drawdown_periods' in drawdown_analysis, "Missing drawdown periods"
        assert 'recovery_time' in drawdown_analysis, "Missing recovery time"
        
        # Validate max drawdown
        max_dd = drawdown_analysis['max_drawdown']
        assert max_dd <= 0, "Max drawdown should be negative or zero"
        assert max_dd >= -0.5, f"Max drawdown too extreme: {max_dd:.1%}"
        
        # Validate current drawdown
        current_dd = drawdown_analysis['current_drawdown']
        assert current_dd <= 0, "Current drawdown should be non-positive"
        
        # Validate drawdown periods
        dd_periods = drawdown_analysis['drawdown_periods']
        assert isinstance(dd_periods, list), "Drawdown periods should be list"
        
        for period in dd_periods:
            assert 'start_date' in period, "Missing drawdown start date"
            assert 'end_date' in period, "Missing drawdown end date"
            assert 'peak_value' in period, "Missing drawdown peak value"
            assert 'trough_value' in period, "Missing drawdown trough value"
            assert 'drawdown_pct' in period, "Missing drawdown percentage"
            
            # Trough should be less than peak
            assert period['trough_value'] <= period['peak_value'], "Trough should be <= peak"
            
            # Drawdown percentage should be negative
            dd_pct = period['drawdown_pct']
            assert dd_pct <= 0, f"Drawdown percentage should be negative: {dd_pct}"


class TestPortfolioAnalysisEdgeCases:
    """Test edge cases and error handling in portfolio analysis."""
    
    @pytest.fixture
    def empty_portfolio_analyzer(self):
        """Create portfolio analyzer with no positions."""
        return PortfolioAnalyzer(
            starting_capital=100000.0,
            max_position_size=0.15,
            target_volatility=0.18
        )
    
    def test_empty_portfolio_handling(self, empty_portfolio_analyzer):
        """Test handling of empty portfolio."""
        # Should handle empty portfolio gracefully
        composition = empty_portfolio_analyzer.get_portfolio_composition()
        
        assert isinstance(composition, dict), "Should return composition dict for empty portfolio"
        assert composition.get('total_value', 0) == 0, "Empty portfolio should have zero value"
        assert composition.get('by_position', {}) == {}, "Should have no positions"
        
        # Risk assessment on empty portfolio
        risk_assessment = empty_portfolio_analyzer.calculate_portfolio_risk()
        assert isinstance(risk_assessment, dict), "Should return risk assessment for empty portfolio"
        
        # P&L summary for empty portfolio
        pnl_summary = empty_portfolio_analyzer.calculate_pnl_summary()
        assert isinstance(pnl_summary, dict), "Should return P&L summary for empty portfolio"
        assert pnl_summary.get('total_pnl', 0) == 0, "Empty portfolio should have zero P&L"
    
    def test_single_position_portfolio(self, empty_portfolio_analyzer):
        """Test portfolio analysis with only one position."""
        # Add single position
        single_position = PortfolioPosition(
            ticker='AAPL',
            shares=100,
            entry_price=150.0,
            current_price=160.0,
            entry_date=datetime(2023, 1, 15),
            sector='Technology',
            position_value=16000.0
        )
        
        empty_portfolio_analyzer.positions = {'AAPL': single_position}
        
        # Should handle single position portfolio
        composition = empty_portfolio_analyzer.get_portfolio_composition()
        
        assert len(composition['by_position']) == 1, "Should have one position"
        assert 'AAPL' in composition['by_position'], "Should contain AAPL position"
        
        # Single position should have 100% weight
        aapl_weight = composition['by_position']['AAPL']['weight']
        assert abs(aapl_weight - 1.0) < 0.01, f"Single position should have ~100% weight: {aapl_weight}"
        
        # Sector analysis should show 100% concentration
        sector_analysis = empty_portfolio_analyzer.analyze_sector_exposure()
        tech_weight = sector_analysis['sector_weights'].get('Technology', 0)
        assert abs(tech_weight - 1.0) < 0.01, f"Single sector should have ~100% weight: {tech_weight}"
        
        # Should warn about concentration
        warnings = sector_analysis['concentration_warnings']
        assert len(warnings) > 0, "Should warn about single position concentration"
    
    def test_extreme_position_values(self, empty_portfolio_analyzer):
        """Test handling of extreme position values."""
        # Very small position
        tiny_position = PortfolioPosition(
            ticker='PENNY',
            shares=1000000,
            entry_price=0.001,
            current_price=0.0015,
            entry_date=datetime(2023, 1, 15),
            sector='Speculative',
            position_value=1500.0  # $1,500 position
        )
        
        # Very large position
        large_position = PortfolioPosition(
            ticker='BRK-A',
            shares=1,
            entry_price=400000.0,
            current_price=450000.0,
            entry_date=datetime(2023, 1, 15),
            sector='Financials',
            position_value=450000.0  # $450k position
        )
        
        empty_portfolio_analyzer.positions = {
            'PENNY': tiny_position,
            'BRK-A': large_position
        }
        
        # Should handle extreme values without errors
        composition = empty_portfolio_analyzer.get_portfolio_composition()
        assert len(composition['by_position']) == 2, "Should handle both positions"
        
        # Large position should dominate weights
        brk_weight = composition['by_position']['BRK-A']['weight']
        penny_weight = composition['by_position']['PENNY']['weight']
        
        assert brk_weight > 0.9, f"Large position should dominate: {brk_weight:.1%}"
        assert penny_weight < 0.1, f"Tiny position should be small fraction: {penny_weight:.1%}"
        
        # Risk assessment should handle extreme values
        risk_assessment = empty_portfolio_analyzer.calculate_portfolio_risk()
        assert isinstance(risk_assessment, dict), "Should handle extreme position values"
    
    def test_missing_price_data_handling(self):
        """Test handling when price history is missing or incomplete."""
        analyzer = PortfolioAnalyzer(starting_capital=100000.0)
        
        # Add position but no price history
        position = PortfolioPosition(
            ticker='MISSING_DATA',
            shares=100,
            entry_price=100.0,
            current_price=105.0,
            entry_date=datetime(2023, 1, 15),
            sector='Unknown',
            position_value=10500.0
        )
        
        analyzer.positions = {'MISSING_DATA': position}
        # analyzer.price_history = {}  # No price history
        
        # Should handle gracefully
        try:
            risk_assessment = analyzer.calculate_portfolio_risk()
            assert isinstance(risk_assessment, dict), "Should handle missing price data"
            
            correlation_analysis = analyzer.analyze_correlations()
            assert isinstance(correlation_analysis, dict), "Should handle missing correlation data"
            
        except Exception as e:
            # Should not crash, but may return limited results
            assert False, f"Should handle missing data gracefully: {e}"
    
    def test_invalid_kelly_inputs(self):
        """Test Kelly criterion with invalid inputs."""
        analyzer = PortfolioAnalyzer(starting_capital=100000.0)
        
        # Negative expected return
        negative_prediction = {
            'expected_return': -0.10,     # -10% expected return
            'win_probability': 0.40,      # Low win rate
            'avg_win': 0.05,
            'avg_loss': 0.08,
            'volatility': 0.25
        }
        
        kelly_result = analyzer.calculate_kelly_position_size(
            ticker='LOSING_STOCK',
            prediction_data=negative_prediction,
            current_price=50.0
        )
        
        # Should recommend no position for negative expected return
        assert kelly_result['optimal_fraction'] == 0.0, "Should recommend 0% for negative expected return"
        
        # Zero volatility (edge case)
        zero_vol_prediction = {
            'expected_return': 0.10,
            'win_probability': 0.70,
            'avg_win': 0.10,
            'avg_loss': 0.05,
            'volatility': 0.0  # Zero volatility
        }
        
        try:
            kelly_result_zero_vol = analyzer.calculate_kelly_position_size(
                ticker='ZERO_VOL',
                prediction_data=zero_vol_prediction,
                current_price=100.0
            )
            
            # Should handle zero volatility gracefully (may recommend max position or use fallback)
            assert isinstance(kelly_result_zero_vol, dict), "Should handle zero volatility"
            
        except Exception as e:
            # Division by zero should be handled
            assert False, f"Should handle zero volatility gracefully: {e}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])