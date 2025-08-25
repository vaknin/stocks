"""Trading signal generation with ML predictions and uncertainty filtering."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

from ..models.ensemble import MetaLearningEnsemble
from ..models.uncertainty import ConformalPredictor, TradingUncertaintyFilter
from ..config.settings import config


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class TimeFrame(Enum):
    """Trading timeframes."""
    INTRADAY = "5m"
    DAILY = "1d"
    WEEKLY = "1wk"


@dataclass
class TradingSignal:
    """Individual trading signal."""
    ticker: str
    signal_type: SignalType
    timeframe: TimeFrame
    confidence: float
    prediction: float
    prediction_interval: Tuple[float, float]
    timestamp: datetime
    model_name: str
    horizon: int
    strength: float  # 0-1 scale
    risk_reward_ratio: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MultiTimeFrameSignal:
    """Aggregated signal across multiple timeframes."""
    ticker: str
    overall_signal: SignalType
    overall_confidence: float
    overall_strength: float
    signals_by_timeframe: Dict[TimeFrame, TradingSignal]
    consensus_score: float  # Agreement between timeframes
    timestamp: datetime


class SignalGenerator:
    """Advanced trading signal generator with ML predictions and uncertainty filtering."""
    
    def __init__(
        self,
        predictor: Optional[MetaLearningEnsemble] = None,
        uncertainty_filter: Optional[TradingUncertaintyFilter] = None,
        confidence_thresholds: Optional[Dict[str, float]] = None,
        timeframes: Optional[List[TimeFrame]] = None
    ):
        """Initialize signal generator.
        
        Args:
            predictor: MetaLearningEnsemble model for predictions
            uncertainty_filter: Uncertainty quantification filter
            confidence_thresholds: Minimum confidence by timeframe
            timeframes: Trading timeframes to analyze
        """
        self.predictor = predictor or MetaLearningEnsemble()
        self.uncertainty_filter = uncertainty_filter or TradingUncertaintyFilter()
        
        # Default confidence thresholds
        self.confidence_thresholds = confidence_thresholds or {
            'intraday': 0.70,   # 70% for 5-minute signals
            'daily': 0.75,      # 75% for daily signals
            'weekly': 0.80      # 80% for weekly signals
        }
        
        self.timeframes = timeframes or [TimeFrame.INTRADAY, TimeFrame.DAILY, TimeFrame.WEEKLY]
        
        # Signal strength parameters
        self.strong_signal_threshold = 0.85  # Above this = STRONG_BUY/SELL
        self.signal_threshold = 0.005        # Minimum prediction magnitude (0.5%)
        self.risk_reward_min = 1.5          # Minimum risk/reward ratio
        
        # Historical signal tracking
        self.signal_history: Dict[str, List[TradingSignal]] = {}
        
        logger.info(f"Signal generator initialized with {len(self.timeframes)} timeframes")
    
    def generate_signals(
        self,
        data_dict: Dict[str, Dict[TimeFrame, pd.DataFrame]],
        current_prices: Dict[str, float]
    ) -> Dict[str, MultiTimeFrameSignal]:
        """Generate trading signals for multiple tickers and timeframes.
        
        Args:
            data_dict: Historical data by ticker and timeframe
            current_prices: Current market prices
            
        Returns:
            Dictionary mapping tickers to multi-timeframe signals
        """
        multi_signals = {}
        
        for ticker in data_dict.keys():
            try:
                # Generate signals for each timeframe
                timeframe_signals = {}
                
                for timeframe in self.timeframes:
                    if timeframe in data_dict[ticker]:
                        df = data_dict[ticker][timeframe]
                        current_price = current_prices.get(ticker, 0)
                        
                        signal = self._generate_single_signal(
                            ticker, df, timeframe, current_price
                        )
                        
                        if signal:
                            timeframe_signals[timeframe] = signal
                
                # Aggregate signals across timeframes
                if timeframe_signals:
                    multi_signal = self._aggregate_timeframe_signals(
                        ticker, timeframe_signals
                    )
                    multi_signals[ticker] = multi_signal
                    
                    # Update signal history
                    self._update_signal_history(ticker, timeframe_signals)
                    
            except Exception as e:
                logger.error(f"Failed to generate signals for {ticker}: {e}")
        
        logger.info(f"Generated signals for {len(multi_signals)} tickers")
        return multi_signals
    
    def _generate_single_signal(
        self,
        ticker: str,
        df: pd.DataFrame,
        timeframe: TimeFrame,
        current_price: float
    ) -> Optional[TradingSignal]:
        """Generate signal for single ticker and timeframe.
        
        Args:
            ticker: Stock ticker
            df: Historical OHLCV data
            timeframe: Trading timeframe
            current_price: Current market price
            
        Returns:
            TradingSignal or None if insufficient confidence
        """
        try:
            # Get ML predictions from ensemble
            predictions = self.predictor.predict(df, ticker, return_confidence=True)
            
            # Select best horizon based on timeframe
            horizon_map = {
                TimeFrame.INTRADAY: 1,   # 1-period ahead
                TimeFrame.DAILY: 1,      # 1-day ahead  
                TimeFrame.WEEKLY: 5      # 1-week ahead
            }
            
            target_horizon = horizon_map.get(timeframe, 1)
            horizon_key = f'horizon_{target_horizon}'
            
            if horizon_key not in predictions:
                logger.debug(f"No prediction for {ticker} horizon {target_horizon}")
                return None
            
            pred_data = predictions[horizon_key]
            prediction = pred_data['prediction']
            confidence = pred_data['confidence']
            pred_interval = pred_data['prediction_interval']
            
            # Apply uncertainty filtering
            timeframe_name = timeframe.value
            confidence_threshold = self.confidence_thresholds.get(
                timeframe_name.replace('m', 'minute').replace('d', 'daily').replace('wk', 'weekly'),
                0.7
            )
            
            if confidence < confidence_threshold:
                logger.debug(f"Low confidence for {ticker} {timeframe.value}: {confidence:.1%}")
                return None
            
            # Apply uncertainty filter with regime-aware filtering
            regime_info = pred_data.get('regime_info', {})
            filtered_prediction = self.uncertainty_filter.filter_prediction(
                prediction, confidence, pred_interval, regime_info
            )
            
            if filtered_prediction is None:
                logger.debug(f"Prediction filtered out for {ticker} {timeframe.value}")
                return None
            
            # Determine signal type and strength
            signal_type, strength = self._classify_signal(prediction, confidence)
            
            # Calculate risk/reward ratio
            risk_reward = self._calculate_risk_reward(
                prediction, pred_interval, current_price
            )
            
            # Skip signals with poor risk/reward
            if risk_reward < self.risk_reward_min:
                logger.debug(f"Poor risk/reward for {ticker}: {risk_reward:.2f}")
                return None
            
            # Set stop loss and take profit levels
            stop_loss, take_profit = self._calculate_stop_take_levels(
                current_price, prediction, signal_type
            )
            
            return TradingSignal(
                ticker=ticker,
                signal_type=signal_type,
                timeframe=timeframe,
                confidence=confidence,
                prediction=prediction,
                prediction_interval=pred_interval,
                timestamp=datetime.now(),
                model_name="MetaLearningEnsemble",
                horizon=target_horizon,
                strength=strength,
                risk_reward_ratio=risk_reward,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'current_price': current_price,
                    'data_points': len(df)
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating signal for {ticker} {timeframe.value}: {e}")
            return None
    
    def _classify_signal(
        self,
        prediction: float,
        confidence: float
    ) -> Tuple[SignalType, float]:
        """Classify prediction into signal type and strength.
        
        Args:
            prediction: Model prediction (return %)
            confidence: Model confidence
            
        Returns:
            Tuple of (signal_type, strength)
        """
        abs_prediction = abs(prediction)
        
        # Calculate signal strength (0-1)
        strength = min(abs_prediction * 10, 1.0) * confidence
        
        # Determine signal type
        if abs_prediction < self.signal_threshold:
            return SignalType.HOLD, strength
        
        if prediction > 0:  # Positive prediction
            if confidence > self.strong_signal_threshold and abs_prediction > 0.02:
                return SignalType.STRONG_BUY, strength
            else:
                return SignalType.BUY, strength
        else:  # Negative prediction
            if confidence > self.strong_signal_threshold and abs_prediction > 0.02:
                return SignalType.STRONG_SELL, strength
            else:
                return SignalType.SELL, strength
    
    def _calculate_risk_reward(
        self,
        prediction: float,
        pred_interval: Tuple[float, float],
        current_price: float
    ) -> float:
        """Calculate risk/reward ratio for the signal.
        
        Args:
            prediction: Expected return
            pred_interval: Prediction confidence interval
            current_price: Current stock price
            
        Returns:
            Risk/reward ratio
        """
        if current_price <= 0:
            return 0.0
        
        # Expected reward (prediction magnitude)
        expected_reward = abs(prediction)
        
        # Estimated risk (wider intervals = higher risk)
        interval_width = pred_interval[1] - pred_interval[0]
        estimated_risk = max(interval_width / 2, 0.01)  # Minimum 1% risk
        
        return expected_reward / estimated_risk if estimated_risk > 0 else 0.0
    
    def _calculate_stop_take_levels(
        self,
        current_price: float,
        prediction: float,
        signal_type: SignalType
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels.
        
        Args:
            current_price: Current stock price
            prediction: Expected return
            signal_type: Signal type
            
        Returns:
            Tuple of (stop_loss, take_profit) prices
        """
        if current_price <= 0 or signal_type == SignalType.HOLD:
            return None, None
        
        # Default stop loss at 2.5%
        stop_loss_pct = 0.025
        
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + abs(prediction))
        else:  # SELL signals
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - abs(prediction))
        
        return stop_loss, take_profit
    
    def _aggregate_timeframe_signals(
        self,
        ticker: str,
        timeframe_signals: Dict[TimeFrame, TradingSignal]
    ) -> MultiTimeFrameSignal:
        """Aggregate signals across multiple timeframes.
        
        Args:
            ticker: Stock ticker
            timeframe_signals: Signals by timeframe
            
        Returns:
            Aggregated multi-timeframe signal
        """
        if not timeframe_signals:
            raise ValueError("No timeframe signals to aggregate")
        
        # Weighted aggregation (longer timeframes have higher weight)
        timeframe_weights = {
            TimeFrame.INTRADAY: 1.0,
            TimeFrame.DAILY: 2.0,
            TimeFrame.WEEKLY: 3.0
        }
        
        # Collect signal votes and weights
        signal_votes = {}
        total_weight = 0
        weighted_confidence = 0
        weighted_strength = 0
        
        for timeframe, signal in timeframe_signals.items():
            weight = timeframe_weights.get(timeframe, 1.0)
            signal_type = signal.signal_type
            
            if signal_type not in signal_votes:
                signal_votes[signal_type] = 0
            
            signal_votes[signal_type] += weight
            total_weight += weight
            weighted_confidence += signal.confidence * weight
            weighted_strength += signal.strength * weight
        
        # Determine consensus signal
        overall_signal = max(signal_votes.items(), key=lambda x: x[1])[0]
        
        # Calculate consensus score (agreement level)
        max_votes = max(signal_votes.values())
        consensus_score = max_votes / total_weight if total_weight > 0 else 0
        
        # Weighted averages
        overall_confidence = weighted_confidence / total_weight if total_weight > 0 else 0
        overall_strength = weighted_strength / total_weight if total_weight > 0 else 0
        
        # Reduce confidence if low consensus
        if consensus_score < 0.6:  # Less than 60% agreement
            overall_confidence *= consensus_score
        
        return MultiTimeFrameSignal(
            ticker=ticker,
            overall_signal=overall_signal,
            overall_confidence=overall_confidence,
            overall_strength=overall_strength,
            signals_by_timeframe=timeframe_signals,
            consensus_score=consensus_score,
            timestamp=datetime.now()
        )
    
    def _update_signal_history(
        self,
        ticker: str,
        timeframe_signals: Dict[TimeFrame, TradingSignal]
    ) -> None:
        """Update signal history for performance tracking.
        
        Args:
            ticker: Stock ticker
            timeframe_signals: New signals by timeframe
        """
        if ticker not in self.signal_history:
            self.signal_history[ticker] = []
        
        # Add all timeframe signals to history
        for signal in timeframe_signals.values():
            self.signal_history[ticker].append(signal)
        
        # Keep limited history
        max_history = 1000
        if len(self.signal_history[ticker]) > max_history:
            self.signal_history[ticker] = self.signal_history[ticker][-max_history:]
    
    def get_signal_performance(
        self,
        ticker: str,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """Get signal performance metrics for a ticker.
        
        Args:
            ticker: Stock ticker
            lookback_days: Lookback period for analysis
            
        Returns:
            Dictionary with performance metrics
        """
        if ticker not in self.signal_history:
            return {'error': 'No signal history available'}
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_signals = [
            s for s in self.signal_history[ticker]
            if s.timestamp > cutoff_date
        ]
        
        if not recent_signals:
            return {'error': 'No recent signals'}
        
        # Count signals by type
        signal_counts = {}
        total_confidence = 0
        total_strength = 0
        
        for signal in recent_signals:
            signal_type = signal.signal_type.value
            if signal_type not in signal_counts:
                signal_counts[signal_type] = 0
            signal_counts[signal_type] += 1
            total_confidence += signal.confidence
            total_strength += signal.strength
        
        num_signals = len(recent_signals)
        
        return {
            'ticker': ticker,
            'lookback_days': lookback_days,
            'total_signals': num_signals,
            'signal_distribution': signal_counts,
            'avg_confidence': total_confidence / num_signals,
            'avg_strength': total_strength / num_signals,
            'timeframes': list(set(s.timeframe.value for s in recent_signals))
        }
    
    def filter_signals_by_quality(
        self,
        signals: Dict[str, MultiTimeFrameSignal],
        min_consensus: float = 0.6,
        min_confidence: float = 0.7,
        min_strength: float = 0.3
    ) -> Dict[str, MultiTimeFrameSignal]:
        """Filter signals by quality criteria.
        
        Args:
            signals: Multi-timeframe signals
            min_consensus: Minimum consensus score
            min_confidence: Minimum confidence
            min_strength: Minimum signal strength
            
        Returns:
            Filtered signals dictionary
        """
        filtered_signals = {}
        
        for ticker, signal in signals.items():
            # Quality checks
            if (signal.consensus_score >= min_consensus and
                signal.overall_confidence >= min_confidence and
                signal.overall_strength >= min_strength and
                signal.overall_signal != SignalType.HOLD):
                
                filtered_signals[ticker] = signal
            else:
                logger.debug(f"Filtered out {ticker} signal due to quality criteria")
        
        logger.info(f"Quality filter: {len(filtered_signals)}/{len(signals)} signals passed")
        return filtered_signals
    
    def get_signal_summary(
        self,
        signals: Dict[str, MultiTimeFrameSignal]
    ) -> pd.DataFrame:
        """Create summary DataFrame of signals.
        
        Args:
            signals: Multi-timeframe signals
            
        Returns:
            Summary DataFrame
        """
        summary_data = []
        
        for ticker, signal in signals.items():
            summary_data.append({
                'ticker': ticker,
                'signal': signal.overall_signal.value,
                'confidence': signal.overall_confidence,
                'strength': signal.overall_strength,
                'consensus': signal.consensus_score,
                'timeframes': len(signal.signals_by_timeframe),
                'timestamp': signal.timestamp
            })
        
        df = pd.DataFrame(summary_data)
        
        if not df.empty:
            df = df.sort_values(['confidence', 'strength'], ascending=False)
        
        return df