"""
Signal Generation Testing Suite - Investment Recommendation Accuracy Validation

Tests the signal generation system's ability to produce accurate buy/sell/hold recommendations
with proper timing, confidence scores, and multi-timeframe consistency. Validates the core
recommendation logic for reliable investment advice generation.

Priority: TIER 1 - Critical for investment recommendation quality assurance
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

from src.trading.signal_generator import (
    SignalGenerator, SignalType, TimeFrame, TradingSignal, 
    MultiTimeFrameSignal
)
from src.models.ensemble import MetaLearningEnsemble
from src.models.uncertainty import TradingUncertaintyFilter
from src.config.settings import config


class TestSignalGeneration:
    """Test suite for validating signal generation and recommendation accuracy."""
    
    @pytest.fixture(scope="class")
    def sample_price_data(self):
        """Generate realistic multi-timeframe price data for testing."""
        np.random.seed(42)
        
        # Generate base daily data
        daily_dates = pd.date_range('2023-01-01', periods=100, freq='D')
        daily_returns = np.random.normal(0.001, 0.02, 100)  # 0.1% mean, 2% volatility
        
        # Add market regime patterns
        daily_returns[20:40] += 0.005  # Bull run
        daily_returns[60:80] -= 0.008  # Bear period
        daily_returns[85:95] *= 2.5    # High volatility
        
        daily_prices = 100 * np.cumprod(1 + daily_returns)
        
        # Create OHLCV for daily timeframe
        daily_data = pd.DataFrame({
            'open': daily_prices * np.random.uniform(0.995, 1.005, 100),
            'high': daily_prices * np.random.uniform(1.005, 1.025, 100),
            'low': daily_prices * np.random.uniform(0.975, 0.995, 100),
            'close': daily_prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=daily_dates)
        
        # Generate intraday data (simplified - last 5 days with 5-min bars)
        intraday_periods = 5 * 78  # 5 days * 78 bars per day (6.5 hour trading day)
        intraday_dates = pd.date_range(
            daily_dates[-5], 
            periods=intraday_periods, 
            freq='5min'
        )
        
        # Use last daily price as base for intraday
        base_intraday_price = daily_prices[-5]
        intraday_returns = np.random.normal(0.0001, 0.005, intraday_periods)
        intraday_prices = base_intraday_price * np.cumprod(1 + intraday_returns)
        
        intraday_data = pd.DataFrame({
            'open': intraday_prices * np.random.uniform(0.999, 1.001, intraday_periods),
            'high': intraday_prices * np.random.uniform(1.001, 1.008, intraday_periods), 
            'low': intraday_prices * np.random.uniform(0.992, 0.999, intraday_periods),
            'close': intraday_prices,
            'volume': np.random.randint(100000, 1000000, intraday_periods)
        }, index=intraday_dates)
        
        # Generate weekly data
        weekly_dates = pd.date_range('2023-01-01', periods=15, freq='W')
        weekly_prices = daily_prices[::7][:15]  # Sample from daily prices
        
        weekly_data = pd.DataFrame({
            'open': weekly_prices * np.random.uniform(0.99, 1.01, 15),
            'high': weekly_prices * np.random.uniform(1.01, 1.05, 15),
            'low': weekly_prices * np.random.uniform(0.95, 0.99, 15),
            'close': weekly_prices,
            'volume': np.random.randint(5000000, 50000000, 15)
        }, index=weekly_dates)
        
        return {
            TimeFrame.DAILY: daily_data,
            TimeFrame.INTRADAY: intraday_data,
            TimeFrame.WEEKLY: weekly_data
        }
    
    @pytest.fixture
    def mock_ensemble_predictor(self):
        """Create mock ensemble predictor for controlled testing."""
        mock_predictor = Mock(spec=MetaLearningEnsemble)
        
        # Define realistic prediction responses
        def mock_predict(df, ticker, return_confidence=True):
            return {
                'horizon_1': {
                    'prediction': 0.025,  # 2.5% expected return
                    'confidence': 0.78,   # High confidence
                    'prediction_interval': [0.01, 0.04],
                    'model_disagreement': 0.015,
                    'regime_info': {
                        'regime_confidence': 0.82
                    }
                },
                'horizon_5': {
                    'prediction': 0.045,  # 4.5% expected return over 5 days
                    'confidence': 0.72,
                    'prediction_interval': [0.02, 0.07],
                    'model_disagreement': 0.025,
                    'regime_info': {
                        'regime_confidence': 0.82
                    }
                }
            }
        
        mock_predictor.predict.side_effect = mock_predict
        return mock_predictor
    
    @pytest.fixture
    def signal_generator(self, mock_ensemble_predictor):
        """Create signal generator with mocked predictor."""
        uncertainty_filter = TradingUncertaintyFilter()
        
        return SignalGenerator(
            predictor=mock_ensemble_predictor,
            uncertainty_filter=uncertainty_filter,
            confidence_thresholds={
                'intraday': 0.65,
                'daily': 0.70,
                'weekly': 0.75
            },
            timeframes=[TimeFrame.INTRADAY, TimeFrame.DAILY, TimeFrame.WEEKLY]
        )
    
    def test_signal_type_classification(self, signal_generator):
        """Test correct classification of predictions into signal types."""
        test_cases = [
            # (prediction, confidence, expected_signal_type)
            (0.035, 0.90, SignalType.STRONG_BUY),    # Strong positive with high confidence
            (0.015, 0.75, SignalType.BUY),           # Moderate positive 
            (0.002, 0.80, SignalType.HOLD),          # Small positive (below threshold)
            (-0.002, 0.80, SignalType.HOLD),         # Small negative (below threshold)
            (-0.018, 0.75, SignalType.SELL),         # Moderate negative
            (-0.030, 0.85, SignalType.STRONG_SELL),  # Strong negative with high confidence
        ]
        
        for prediction, confidence, expected_signal in test_cases:
            signal_type, strength = signal_generator._classify_signal(prediction, confidence)
            assert signal_type == expected_signal, (
                f"Prediction {prediction:.1%} with confidence {confidence:.1%} "
                f"should be {expected_signal.value}, got {signal_type.value}"
            )
            
            # Validate strength calculation
            assert 0.0 <= strength <= 1.0, f"Invalid strength: {strength}"
            if abs(prediction) > signal_generator.signal_threshold:
                assert strength > 0.1, f"Non-hold signal should have meaningful strength: {strength}"
    
    def test_risk_reward_calculation(self, signal_generator):
        """Test risk/reward ratio calculations for investment decisions."""
        test_cases = [
            # (prediction, prediction_interval, current_price, expected_min_ratio)
            (0.05, (0.02, 0.08), 100.0, 1.0),   # Good risk/reward
            (0.02, (0.015, 0.025), 100.0, 1.5), # Excellent risk/reward (tight interval)
            (0.01, (-0.05, 0.07), 100.0, 0.2),  # Poor risk/reward (wide interval)
        ]
        
        for prediction, interval, current_price, min_expected_ratio in test_cases:
            ratio = signal_generator._calculate_risk_reward(prediction, interval, current_price)
            
            assert ratio >= 0.0, f"Risk/reward ratio should be non-negative: {ratio}"
            if min_expected_ratio > 1.0:
                assert ratio >= min_expected_ratio, (
                    f"Expected ratio >= {min_expected_ratio}, got {ratio:.2f}"
                )
    
    def test_stop_take_level_calculation(self, signal_generator):
        """Test stop loss and take profit level calculations."""
        current_price = 100.0
        
        test_cases = [
            # (signal_type, prediction, expected_stop_direction, expected_take_direction)
            (SignalType.BUY, 0.03, 'below', 'above'),         # Long position
            (SignalType.STRONG_BUY, 0.05, 'below', 'above'),  # Strong long
            (SignalType.SELL, -0.03, 'above', 'below'),       # Short position
            (SignalType.STRONG_SELL, -0.05, 'above', 'below'), # Strong short
            (SignalType.HOLD, 0.001, 'none', 'none'),         # No position
        ]
        
        for signal_type, prediction, stop_direction, take_direction in test_cases:
            stop_loss, take_profit = signal_generator._calculate_stop_take_levels(
                current_price, prediction, signal_type
            )
            
            if signal_type == SignalType.HOLD:
                assert stop_loss is None and take_profit is None, "HOLD should have no stop/take levels"
            else:
                assert stop_loss is not None and take_profit is not None, (
                    f"Signal {signal_type.value} should have stop/take levels"
                )
                
                if stop_direction == 'below':
                    assert stop_loss < current_price, f"Long stop should be below current: {stop_loss}"
                    assert take_profit > current_price, f"Long take should be above current: {take_profit}"
                elif stop_direction == 'above':
                    assert stop_loss > current_price, f"Short stop should be above current: {stop_loss}"
                    assert take_profit < current_price, f"Short take should be below current: {take_profit}"
    
    def test_multi_timeframe_signal_generation(self, signal_generator, sample_price_data):
        """Test signal generation across multiple timeframes."""
        data_dict = {'AAPL': sample_price_data}
        current_prices = {'AAPL': sample_price_data[TimeFrame.DAILY]['close'].iloc[-1]}
        
        # Generate multi-timeframe signals
        signals = signal_generator.generate_signals(data_dict, current_prices)
        
        assert isinstance(signals, dict), "Should return signals dictionary"
        
        if 'AAPL' in signals:
            signal = signals['AAPL']
            assert isinstance(signal, MultiTimeFrameSignal), "Should return MultiTimeFrameSignal"
            
            # Validate signal attributes
            assert hasattr(signal, 'overall_signal'), "Missing overall signal"
            assert hasattr(signal, 'overall_confidence'), "Missing overall confidence"
            assert hasattr(signal, 'consensus_score'), "Missing consensus score"
            assert hasattr(signal, 'signals_by_timeframe'), "Missing timeframe signals"
            
            # Validate ranges
            assert 0.0 <= signal.overall_confidence <= 1.0, "Invalid overall confidence"
            assert 0.0 <= signal.consensus_score <= 1.0, "Invalid consensus score"
            assert 0.0 <= signal.overall_strength <= 1.0, "Invalid overall strength"
            
            # Validate individual timeframe signals
            for timeframe, tf_signal in signal.signals_by_timeframe.items():
                assert isinstance(tf_signal, TradingSignal), f"Invalid signal for {timeframe}"
                assert tf_signal.timeframe == timeframe, "Timeframe mismatch"
                assert tf_signal.ticker == 'AAPL', "Ticker mismatch"
                assert 0.0 <= tf_signal.confidence <= 1.0, f"Invalid confidence for {timeframe}"
    
    def test_timeframe_consensus_calculation(self, signal_generator):
        """Test consensus calculation across timeframes."""
        # Create mock timeframe signals with different signal types
        from datetime import datetime
        
        mock_signals = {
            TimeFrame.INTRADAY: TradingSignal(
                ticker='TEST', signal_type=SignalType.BUY, timeframe=TimeFrame.INTRADAY,
                confidence=0.75, prediction=0.02, prediction_interval=(0.01, 0.03),
                timestamp=datetime.now(), model_name='test', horizon=1, strength=0.6,
                risk_reward_ratio=2.0
            ),
            TimeFrame.DAILY: TradingSignal(
                ticker='TEST', signal_type=SignalType.BUY, timeframe=TimeFrame.DAILY,
                confidence=0.80, prediction=0.025, prediction_interval=(0.015, 0.035),
                timestamp=datetime.now(), model_name='test', horizon=1, strength=0.7,
                risk_reward_ratio=2.2
            ),
            TimeFrame.WEEKLY: TradingSignal(
                ticker='TEST', signal_type=SignalType.STRONG_BUY, timeframe=TimeFrame.WEEKLY,
                confidence=0.85, prediction=0.05, prediction_interval=(0.03, 0.07),
                timestamp=datetime.now(), model_name='test', horizon=5, strength=0.8,
                risk_reward_ratio=2.5
            )
        }
        
        # Test consensus calculation
        consensus_signal = signal_generator._aggregate_timeframe_signals('TEST', mock_signals)
        
        # Should aggregate to BUY or STRONG_BUY (high consensus)
        assert consensus_signal.overall_signal in [SignalType.BUY, SignalType.STRONG_BUY]
        
        # High consensus should result in high consensus score
        assert consensus_signal.consensus_score > 0.8, f"Expected high consensus: {consensus_signal.consensus_score}"
        
        # Overall confidence should be weighted average (weekly has highest weight)
        expected_min_confidence = 0.75  # Should be at least the minimum
        assert consensus_signal.overall_confidence >= expected_min_confidence
        
    def test_conflicting_timeframe_signals(self, signal_generator):
        """Test handling of conflicting signals across timeframes."""
        from datetime import datetime
        
        # Create conflicting signals
        conflicting_signals = {
            TimeFrame.INTRADAY: TradingSignal(
                ticker='TEST', signal_type=SignalType.SELL, timeframe=TimeFrame.INTRADAY,
                confidence=0.70, prediction=-0.015, prediction_interval=(-0.025, -0.005),
                timestamp=datetime.now(), model_name='test', horizon=1, strength=0.5,
                risk_reward_ratio=1.8
            ),
            TimeFrame.DAILY: TradingSignal(
                ticker='TEST', signal_type=SignalType.BUY, timeframe=TimeFrame.DAILY,
                confidence=0.75, prediction=0.02, prediction_interval=(0.01, 0.03),
                timestamp=datetime.now(), model_name='test', horizon=1, strength=0.6,
                risk_reward_ratio=2.0
            ),
            TimeFrame.WEEKLY: TradingSignal(
                ticker='TEST', signal_type=SignalType.STRONG_BUY, timeframe=TimeFrame.WEEKLY,
                confidence=0.85, prediction=0.04, prediction_interval=(0.025, 0.055),
                timestamp=datetime.now(), model_name='test', horizon=5, strength=0.8,
                risk_reward_ratio=2.5
            )
        }
        
        consensus_signal = signal_generator._aggregate_timeframe_signals('TEST', conflicting_signals)
        
        # With conflicts, consensus score should be lower
        assert consensus_signal.consensus_score < 0.8, f"Expected lower consensus due to conflicts: {consensus_signal.consensus_score}"
        
        # Overall confidence should be reduced due to disagreement
        max_individual_confidence = max(s.confidence for s in conflicting_signals.values())
        assert consensus_signal.overall_confidence <= max_individual_confidence
        
        # Longer timeframes should have more influence (WEEKLY weight = 3.0)
        # So result should lean towards BUY despite INTRADAY SELL
        assert consensus_signal.overall_signal in [SignalType.BUY, SignalType.STRONG_BUY]
    
    def test_confidence_threshold_filtering(self, signal_generator):
        """Test filtering of signals below confidence thresholds."""
        # Test with different confidence levels
        with patch.object(signal_generator.predictor, 'predict') as mock_predict:
            
            # Low confidence prediction (below threshold)
            mock_predict.return_value = {
                'horizon_1': {
                    'prediction': 0.03,
                    'confidence': 0.60,  # Below daily threshold (0.70)
                    'prediction_interval': [0.01, 0.05],
                    'model_disagreement': 0.02,
                    'regime_info': {'regime_confidence': 0.60}
                }
            }
            
            # Create sample data
            df = pd.DataFrame({
                'open': [100, 101, 102],
                'high': [102, 103, 104], 
                'low': [99, 100, 101],
                'close': [101, 102, 103],
                'volume': [1000000, 1200000, 1100000]
            })
            
            # Should return None due to low confidence
            signal = signal_generator._generate_single_signal(
                'TEST', df, TimeFrame.DAILY, 103.0
            )
            
            assert signal is None, "Low confidence signal should be filtered out"
            
            # High confidence prediction (above threshold)
            mock_predict.return_value = {
                'horizon_1': {
                    'prediction': 0.03,
                    'confidence': 0.85,  # Above daily threshold
                    'prediction_interval': [0.02, 0.04],
                    'model_disagreement': 0.01,
                    'regime_info': {'regime_confidence': 0.85}
                }
            }
            
            signal = signal_generator._generate_single_signal(
                'TEST', df, TimeFrame.DAILY, 103.0
            )
            
            assert signal is not None, "High confidence signal should pass filter"
            assert signal.confidence == 0.85, "Should preserve confidence level"
    
    def test_signal_history_tracking(self, signal_generator):
        """Test signal history tracking for performance analysis."""
        from datetime import datetime
        
        # Create mock signals
        mock_signals = {
            TimeFrame.DAILY: TradingSignal(
                ticker='AAPL', signal_type=SignalType.BUY, timeframe=TimeFrame.DAILY,
                confidence=0.80, prediction=0.025, prediction_interval=(0.015, 0.035),
                timestamp=datetime.now(), model_name='test', horizon=1, strength=0.7,
                risk_reward_ratio=2.0
            )
        }
        
        # Update signal history
        signal_generator._update_signal_history('AAPL', mock_signals)
        
        # Check history was recorded
        assert 'AAPL' in signal_generator.signal_history
        assert len(signal_generator.signal_history['AAPL']) > 0
        
        # Test signal performance retrieval
        performance = signal_generator.get_signal_performance('AAPL', lookback_days=30)
        
        assert 'ticker' in performance, "Missing ticker in performance"
        assert 'total_signals' in performance, "Missing signal count"
        assert 'avg_confidence' in performance, "Missing average confidence"
        assert 'signal_distribution' in performance, "Missing signal distribution"
        
        # Should show BUY signal in distribution
        signal_dist = performance['signal_distribution']
        assert 'BUY' in signal_dist, "BUY signal should be in distribution"
        assert signal_dist['BUY'] > 0, "Should count BUY signals"
    
    def test_signal_quality_filtering(self, signal_generator):
        """Test quality-based filtering of generated signals."""
        from datetime import datetime
        
        # Create high and low quality mock signals
        high_quality = MultiTimeFrameSignal(
            ticker='HIGH_QUAL', overall_signal=SignalType.BUY,
            overall_confidence=0.85, overall_strength=0.75,
            signals_by_timeframe={}, consensus_score=0.90,
            timestamp=datetime.now()
        )
        
        medium_quality = MultiTimeFrameSignal(
            ticker='MED_QUAL', overall_signal=SignalType.SELL,
            overall_confidence=0.72, overall_strength=0.55,
            signals_by_timeframe={}, consensus_score=0.65,
            timestamp=datetime.now()
        )
        
        low_quality = MultiTimeFrameSignal(
            ticker='LOW_QUAL', overall_signal=SignalType.HOLD,
            overall_confidence=0.45, overall_strength=0.25,
            signals_by_timeframe={}, consensus_score=0.35,
            timestamp=datetime.now()
        )
        
        signals = {
            'HIGH_QUAL': high_quality,
            'MED_QUAL': medium_quality,
            'LOW_QUAL': low_quality
        }
        
        # Apply quality filters
        filtered = signal_generator.filter_signals_by_quality(
            signals,
            min_consensus=0.6,
            min_confidence=0.7,
            min_strength=0.3
        )
        
        # Only high quality should pass all filters
        assert 'HIGH_QUAL' in filtered, "High quality signal should pass"
        assert 'LOW_QUAL' not in filtered, "Low quality signal should be filtered"
        
        # Medium quality depends on exact thresholds
        if medium_quality.consensus_score >= 0.6 and medium_quality.overall_confidence >= 0.7:
            assert 'MED_QUAL' in filtered, "Medium quality signal meeting thresholds should pass"
        else:
            assert 'MED_QUAL' not in filtered, "Medium quality signal below thresholds should be filtered"
    
    def test_signal_summary_generation(self, signal_generator):
        """Test signal summary DataFrame generation."""
        from datetime import datetime
        
        # Create sample signals
        signals = {
            'AAPL': MultiTimeFrameSignal(
                ticker='AAPL', overall_signal=SignalType.BUY,
                overall_confidence=0.85, overall_strength=0.75,
                signals_by_timeframe={}, consensus_score=0.90,
                timestamp=datetime.now()
            ),
            'MSFT': MultiTimeFrameSignal(
                ticker='MSFT', overall_signal=SignalType.STRONG_SELL,
                overall_confidence=0.78, overall_strength=0.65,
                signals_by_timeframe={}, consensus_score=0.82,
                timestamp=datetime.now()
            )
        }
        
        summary_df = signal_generator.get_signal_summary(signals)
        
        # Validate DataFrame structure
        assert isinstance(summary_df, pd.DataFrame), "Should return DataFrame"
        assert len(summary_df) == len(signals), "Should have row for each signal"
        
        required_columns = ['ticker', 'signal', 'confidence', 'strength', 'consensus', 'timeframes']
        for col in required_columns:
            assert col in summary_df.columns, f"Missing column: {col}"
        
        # Validate sorting (should be by confidence and strength descending)
        if len(summary_df) > 1:
            confidences = summary_df['confidence'].values
            # Should be sorted in descending order
            assert all(confidences[i] >= confidences[i+1] for i in range(len(confidences)-1)), \
                "Summary should be sorted by confidence descending"
        
        # Validate content
        assert 'AAPL' in summary_df['ticker'].values, "AAPL should be in summary"
        assert 'MSFT' in summary_df['ticker'].values, "MSFT should be in summary"
        
        aapl_row = summary_df[summary_df['ticker'] == 'AAPL'].iloc[0]
        assert aapl_row['signal'] == 'BUY', "AAPL signal should be BUY"
        assert aapl_row['confidence'] == 0.85, "AAPL confidence should match"


class TestSignalGenerationEdgeCases:
    """Test edge cases and error handling in signal generation."""
    
    @pytest.fixture
    def minimal_signal_generator(self):
        """Create signal generator with minimal configuration for edge case testing."""
        mock_predictor = Mock()
        mock_uncertainty_filter = Mock()
        
        # Default mock behavior
        mock_predictor.predict.return_value = {}
        mock_uncertainty_filter.filter_prediction.return_value = 0.02
        
        return SignalGenerator(
            predictor=mock_predictor,
            uncertainty_filter=mock_uncertainty_filter
        )
    
    def test_empty_data_handling(self, minimal_signal_generator):
        """Test handling of empty or insufficient data."""
        empty_data_dict = {}
        current_prices = {}
        
        # Should handle empty data gracefully
        signals = minimal_signal_generator.generate_signals(empty_data_dict, current_prices)
        assert isinstance(signals, dict), "Should return dict even with empty data"
        assert len(signals) == 0, "Should return empty signals for empty data"
        
        # Test insufficient data
        insufficient_data = pd.DataFrame({
            'open': [100], 'high': [101], 'low': [99], 'close': [100.5], 'volume': [1000000]
        })
        
        data_dict = {'TEST': {TimeFrame.DAILY: insufficient_data}}
        current_prices = {'TEST': 100.5}
        
        signals = minimal_signal_generator.generate_signals(data_dict, current_prices)
        # Should handle gracefully (might return empty or fallback signals)
        assert isinstance(signals, dict), "Should handle insufficient data gracefully"
    
    def test_prediction_failure_handling(self, minimal_signal_generator):
        """Test handling when prediction models fail."""
        # Mock prediction failure
        minimal_signal_generator.predictor.predict.side_effect = Exception("Model failed")
        
        # Create valid data
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101], 
            'close': [101, 102, 103],
            'volume': [1000000, 1200000, 1100000]
        })
        
        data_dict = {'TEST': {TimeFrame.DAILY: test_data}}
        current_prices = {'TEST': 103.0}
        
        # Should handle prediction failures gracefully
        signals = minimal_signal_generator.generate_signals(data_dict, current_prices)
        assert isinstance(signals, dict), "Should handle prediction failures"
        # May return empty signals or fallback signals
    
    def test_invalid_price_data_handling(self, minimal_signal_generator):
        """Test handling of invalid price data (NaN, negative values, etc.)."""
        # Create data with NaN values
        invalid_data = pd.DataFrame({
            'open': [100, np.nan, 102],
            'high': [102, 103, np.nan],
            'low': [99, np.nan, 101],
            'close': [101, np.nan, 103],
            'volume': [1000000, np.nan, 1100000]
        })
        
        data_dict = {'TEST': {TimeFrame.DAILY: invalid_data}}
        current_prices = {'TEST': 103.0}
        
        # Should handle NaN values gracefully
        try:
            signals = minimal_signal_generator.generate_signals(data_dict, current_prices)
            assert isinstance(signals, dict), "Should handle NaN data"
        except Exception as e:
            pytest.fail(f"Should handle invalid data gracefully: {e}")
        
        # Test negative prices
        negative_price_data = pd.DataFrame({
            'open': [100, -101, 102],  # Negative price
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, -103],  # Negative price
            'volume': [1000000, 1200000, 1100000]
        })
        
        data_dict = {'TEST': {TimeFrame.DAILY: negative_price_data}}
        current_prices = {'TEST': 103.0}
        
        # Should handle negative prices gracefully 
        try:
            signals = minimal_signal_generator.generate_signals(data_dict, current_prices)
            assert isinstance(signals, dict), "Should handle negative price data"
        except Exception as e:
            pytest.fail(f"Should handle negative prices gracefully: {e}")
    
    def test_extreme_prediction_values(self, minimal_signal_generator):
        """Test handling of extreme prediction values."""
        # Mock extreme predictions
        extreme_predictions = {
            'horizon_1': {
                'prediction': 5.0,  # 500% return (unrealistic)
                'confidence': 0.95,
                'prediction_interval': [3.0, 7.0],
                'model_disagreement': 0.1,
                'regime_info': {'regime_confidence': 0.9}
            }
        }
        
        minimal_signal_generator.predictor.predict.return_value = extreme_predictions
        
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103], 
            'volume': [1000000, 1200000, 1100000]
        })
        
        # Should apply sanity checks to extreme predictions
        signal = minimal_signal_generator._generate_single_signal(
            'TEST', test_data, TimeFrame.DAILY, 103.0
        )
        
        if signal is not None:
            # Extreme prediction should be clipped to reasonable bounds
            assert -0.5 <= signal.prediction <= 0.5, (
                f"Extreme prediction should be clipped: {signal.prediction}"
            )
    
    def test_timezone_handling(self, minimal_signal_generator):
        """Test handling of different timezones in data."""
        import pytz
        
        # Create data with timezone-aware timestamps
        dates_utc = pd.date_range('2023-01-01', periods=10, freq='D', tz='UTC')
        dates_est = dates_utc.tz_convert('US/Eastern')
        
        utc_data = pd.DataFrame({
            'open': [100 + i for i in range(10)],
            'high': [102 + i for i in range(10)],
            'low': [99 + i for i in range(10)],
            'close': [101 + i for i in range(10)],
            'volume': [1000000] * 10
        }, index=dates_utc)
        
        est_data = pd.DataFrame({
            'open': [100 + i for i in range(10)],
            'high': [102 + i for i in range(10)],
            'low': [99 + i for i in range(10)],
            'close': [101 + i for i in range(10)],
            'volume': [1000000] * 10
        }, index=dates_est)
        
        # Should handle different timezones gracefully
        data_dict_utc = {'TEST': {TimeFrame.DAILY: utc_data}}
        data_dict_est = {'TEST': {TimeFrame.DAILY: est_data}}
        current_prices = {'TEST': 110.0}
        
        try:
            signals_utc = minimal_signal_generator.generate_signals(data_dict_utc, current_prices)
            signals_est = minimal_signal_generator.generate_signals(data_dict_est, current_prices)
            
            assert isinstance(signals_utc, dict), "Should handle UTC timezone"
            assert isinstance(signals_est, dict), "Should handle EST timezone"
            
        except Exception as e:
            pytest.fail(f"Should handle timezones gracefully: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])