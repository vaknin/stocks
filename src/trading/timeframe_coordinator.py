"""Multi-timeframe trading coordinator for scalping, swing, and position trading."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from loguru import logger
import warnings
from pathlib import Path

from ..models.ensemble import MetaLearningEnsemble
from .signal_generator import SignalGenerator, SignalType, TradingSignal
from .risk_manager import RiskManager
from .position_sizer import KellyPositionSizer
from ..config.settings import config

class TimeframeType(Enum):
    """Different trading timeframes."""
    SCALP_5MIN = "5min"
    SWING_DAILY = "1d" 
    POSITION_WEEKLY = "1wk"

@dataclass
class TimeframeConfig:
    """Configuration for each timeframe strategy."""
    timeframe: TimeframeType
    interval: str
    horizon_days: int
    max_positions: int
    profit_target: float
    stop_loss: float
    min_confidence: float
    weight: float  # Portfolio allocation weight

class MultiTimeframeCoordinator:
    """Coordinates trading across multiple timeframes for optimal performance."""
    
    def __init__(
        self,
        ensemble_predictor: Optional[MetaLearningEnsemble] = None,
        risk_manager: Optional[RiskManager] = None,
        position_sizer: Optional[KellyPositionSizer] = None,
        enable_scalping: bool = True,
        enable_swing: bool = True,
        enable_position: bool = True
    ):
        """
        Initialize multi-timeframe coordinator.
        
        Args:
            ensemble_predictor: ML ensemble for predictions
            risk_manager: Risk management system
            position_sizer: Position sizing system
            enable_scalping: Whether to enable 5-min scalping
            enable_swing: Whether to enable daily swing trading
            enable_position: Whether to enable weekly position trading
        """
        
        # Initialize core components
        self.ensemble_predictor = ensemble_predictor or MetaLearningEnsemble()
        self.risk_manager = risk_manager or RiskManager()
        self.position_sizer = position_sizer or KellyPositionSizer()
        
        # Define timeframe configurations
        self.timeframe_configs = {
            TimeframeType.SCALP_5MIN: TimeframeConfig(
                timeframe=TimeframeType.SCALP_5MIN,
                interval="5m",
                horizon_days=1,  # Intraday predictions
                max_positions=3,  # Quick in/out
                profit_target=0.015,  # 1.5% target
                stop_loss=0.008,  # 0.8% stop
                min_confidence=0.75,  # High confidence only
                weight=0.20  # 20% of capital
            ),
            TimeframeType.SWING_DAILY: TimeframeConfig(
                timeframe=TimeframeType.SWING_DAILY,
                interval="1d",
                horizon_days=5,  # 5-day swing
                max_positions=5,  # Medium term
                profit_target=0.04,  # 4% target
                stop_loss=0.02,  # 2% stop
                min_confidence=0.65,  # Medium confidence
                weight=0.50  # 50% of capital
            ),
            TimeframeType.POSITION_WEEKLY: TimeframeConfig(
                timeframe=TimeframeType.POSITION_WEEKLY,
                interval="1d",  # Daily data for weekly analysis
                horizon_days=20,  # 20-day position
                max_positions=8,  # Longer term
                profit_target=0.08,  # 8% target
                stop_loss=0.04,  # 4% stop
                min_confidence=0.60,  # Lower confidence OK
                weight=0.30  # 30% of capital
            )
        }
        
        # Filter enabled timeframes
        if not enable_scalping:
            del self.timeframe_configs[TimeframeType.SCALP_5MIN]
        if not enable_swing:
            del self.timeframe_configs[TimeframeType.SWING_DAILY]
        if not enable_position:
            del self.timeframe_configs[TimeframeType.POSITION_WEEKLY]
        
        # Renormalize weights
        total_weight = sum(tf_config.weight for tf_config in self.timeframe_configs.values())
        if total_weight > 0:
            for tf_config in self.timeframe_configs.values():
                tf_config.weight = tf_config.weight / total_weight
        
        # Initialize signal generators for each timeframe
        self.signal_generators = {}
        for tf_type, tf_config in self.timeframe_configs.items():
            self.signal_generators[tf_type] = SignalGenerator(
                predictor=self.ensemble_predictor,
                confidence_thresholds={
                    'buy': tf_config.min_confidence,
                    'sell': tf_config.min_confidence,
                    'hold': 0.5
                }
            )
        
        # Active positions by timeframe
        self.active_positions = {tf_type: {} for tf_type in self.timeframe_configs.keys()}
        
        # Performance tracking
        self.performance_metrics = {tf_type: {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        } for tf_type in self.timeframe_configs.keys()}
        
        logger.info(f"Multi-timeframe coordinator initialized with {len(self.timeframe_configs)} timeframes")
        
    def generate_coordinated_signals(
        self,
        data_dict: Dict[str, pd.DataFrame],
        current_prices: Dict[str, float],
        tickers: List[str]
    ) -> Dict[TimeframeType, List[TradingSignal]]:
        """
        Generate trading signals across all timeframes with coordination.
        
        Args:
            data_dict: Historical OHLCV data for each ticker
            current_prices: Current market prices
            tickers: List of tickers to analyze
            
        Returns:
            Dictionary of signals by timeframe
        """
        all_signals = {}
        
        try:
            for tf_type, tf_config in self.timeframe_configs.items():
                logger.debug(f"Generating signals for {tf_type.value}")
                
                # Generate signals for this timeframe
                signals = self._generate_timeframe_signals(
                    data_dict, current_prices, tickers, tf_type, tf_config
                )
                
                # Apply inter-timeframe coordination
                coordinated_signals = self._coordinate_signals(signals, tf_type, all_signals)
                
                all_signals[tf_type] = coordinated_signals
                
                logger.debug(f"Generated {len(coordinated_signals)} signals for {tf_type.value}")
            
            return all_signals
            
        except Exception as e:
            logger.error(f"Error generating coordinated signals: {e}")
            return {tf_type: [] for tf_type in self.timeframe_configs.keys()}
    
    def _generate_timeframe_signals(
        self,
        data_dict: Dict[str, pd.DataFrame],
        current_prices: Dict[str, float],
        tickers: List[str],
        tf_type: TimeframeType,
        tf_config: TimeframeConfig
    ) -> List[TradingSignal]:
        """Generate signals for a specific timeframe."""
        signals = []
        
        try:
            signal_generator = self.signal_generators[tf_type]
            
            for ticker in tickers:
                if ticker not in data_dict or ticker not in current_prices:
                    continue
                
                df = data_dict[ticker]
                if len(df) < 20:  # Minimum data requirement
                    continue
                
                # Get ensemble predictions for this timeframe's horizon
                predictions = self.ensemble_predictor.predict(data_dict, ticker)
                horizon_key = f"horizon_{tf_config.horizon_days}"
                
                if horizon_key not in predictions:
                    continue
                
                pred_data = predictions[horizon_key]
                prediction = pred_data['prediction']
                confidence = pred_data['confidence']
                
                # Check if confidence meets threshold
                if confidence < tf_config.min_confidence:
                    continue
                
                # Generate signal based on prediction and timeframe rules
                signal = self._create_timeframe_signal(
                    ticker, prediction, confidence, current_prices[ticker],
                    tf_type, tf_config, pred_data
                )
                
                if signal:
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating {tf_type.value} signals: {e}")
            return []
    
    def _create_timeframe_signal(
        self,
        ticker: str,
        prediction: float,
        confidence: float,
        current_price: float,
        tf_type: TimeframeType,
        tf_config: TimeframeConfig,
        pred_data: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """Create a trading signal for specific timeframe."""
        try:
            # Determine signal direction and strength
            if prediction > 0.01:  # Positive prediction threshold
                signal_type = SignalType.BUY
                strength = min(confidence * abs(prediction) * 10, 1.0)
            elif prediction < -0.01:  # Negative prediction threshold  
                signal_type = SignalType.SELL
                strength = min(confidence * abs(prediction) * 10, 1.0)
            else:
                return None  # No clear signal
            
            # Calculate position size based on timeframe
            result = self.position_sizer.calculate_position_size(
                ticker=ticker,
                prediction=prediction,
                confidence=confidence,
                current_price=current_price,
                portfolio_value=100000.0 * tf_config.weight  # Scale by timeframe allocation
            )
            position_size = result.recommended_size
            
            # Create signal with timeframe-specific parameters
            signal = TradingSignal(
                ticker=ticker,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                price=current_price,
                quantity=position_size,
                timestamp=datetime.now(),
                reasoning=f"{tf_type.value}_ensemble",
                metadata={
                    'timeframe': tf_type.value,
                    'horizon_days': tf_config.horizon_days,
                    'profit_target': tf_config.profit_target,
                    'stop_loss': tf_config.stop_loss,
                    'prediction': prediction,
                    'model_disagreement': pred_data.get('model_disagreement', 0.0),
                    'individual_predictions': pred_data.get('individual_predictions', {}),
                    'model_weights': pred_data.get('model_weights', {})
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error creating signal for {ticker} on {tf_type.value}: {e}")
            return None
    
    def _coordinate_signals(
        self,
        new_signals: List[TradingSignal],
        current_tf: TimeframeType,
        existing_signals: Dict[TimeframeType, List[TradingSignal]]
    ) -> List[TradingSignal]:
        """Coordinate signals across timeframes to avoid conflicts."""
        coordinated_signals = []
        
        for signal in new_signals:
            # Check for conflicts with other timeframes
            if self._has_signal_conflicts(signal, current_tf, existing_signals):
                logger.debug(f"Signal conflict detected for {signal.ticker} on {current_tf.value}")
                continue
            
            # Check position limits for this timeframe
            active_positions = len(self.active_positions[current_tf])
            max_positions = self.timeframe_configs[current_tf].max_positions
            
            if active_positions >= max_positions:
                logger.debug(f"Position limit reached for {current_tf.value}: {active_positions}/{max_positions}")
                continue
            
            # Apply risk management
            if not self.risk_manager.validate_signal(signal):
                logger.debug(f"Risk management rejected signal for {signal.ticker}")
                continue
            
            coordinated_signals.append(signal)
        
        return coordinated_signals
    
    def _has_signal_conflicts(
        self,
        signal: TradingSignal,
        current_tf: TimeframeType,
        existing_signals: Dict[TimeframeType, List[TradingSignal]]
    ) -> bool:
        """Check if signal conflicts with signals from other timeframes."""
        for tf_type, signals in existing_signals.items():
            for existing_signal in signals:
                if existing_signal.ticker == signal.ticker:
                    # Check for opposite signals (conflict)
                    if (signal.signal_type == SignalType.BUY and existing_signal.signal_type == SignalType.SELL) or \
                       (signal.signal_type == SignalType.SELL and existing_signal.signal_type == SignalType.BUY):
                        # Allow if current timeframe has higher priority (shorter timeframe wins)
                        if self._get_timeframe_priority(current_tf) > self._get_timeframe_priority(tf_type):
                            return False
                        else:
                            return True
        
        return False
    
    def _get_timeframe_priority(self, tf_type: TimeframeType) -> int:
        """Get priority level for timeframe (higher = more priority)."""
        priority_map = {
            TimeframeType.SCALP_5MIN: 3,    # Highest priority
            TimeframeType.SWING_DAILY: 2,   # Medium priority
            TimeframeType.POSITION_WEEKLY: 1 # Lowest priority
        }
        return priority_map.get(tf_type, 0)
    
    def update_positions(
        self,
        executed_trades: Dict[str, Dict],
        current_prices: Dict[str, float]
    ) -> None:
        """Update active positions based on executed trades."""
        for tf_type in self.active_positions.keys():
            # Update positions for this timeframe
            self._update_timeframe_positions(tf_type, executed_trades, current_prices)
    
    def _update_timeframe_positions(
        self,
        tf_type: TimeframeType,
        executed_trades: Dict[str, Dict],
        current_prices: Dict[str, float]
    ) -> None:
        """Update positions for a specific timeframe."""
        tf_config = self.timeframe_configs[tf_type]
        active_positions = self.active_positions[tf_type]
        
        # Check for profit targets and stop losses
        positions_to_close = []
        
        for ticker, position in active_positions.items():
            if ticker not in current_prices:
                continue
            
            current_price = current_prices[ticker]
            entry_price = position['entry_price']
            quantity = position['quantity']
            
            # Calculate current P&L
            if position['side'] == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # short
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Check exit conditions
            should_close = False
            close_reason = ""
            
            if pnl_pct >= tf_config.profit_target:
                should_close = True
                close_reason = "profit_target"
            elif pnl_pct <= -tf_config.stop_loss:
                should_close = True
                close_reason = "stop_loss"
            
            if should_close:
                positions_to_close.append((ticker, close_reason))
                
                # Update performance metrics
                self._update_performance_metrics(tf_type, pnl_pct, close_reason == "profit_target")
        
        # Close positions that hit targets
        for ticker, reason in positions_to_close:
            logger.info(f"Closing {tf_type.value} position for {ticker}: {reason}")
            del active_positions[ticker]
    
    def _update_performance_metrics(
        self,
        tf_type: TimeframeType,
        pnl_pct: float,
        is_winner: bool
    ) -> None:
        """Update performance metrics for timeframe."""
        metrics = self.performance_metrics[tf_type]
        
        metrics['total_trades'] += 1
        if is_winner:
            metrics['winning_trades'] += 1
        
        metrics['total_pnl'] += pnl_pct
        
        # Update max drawdown if needed
        if pnl_pct < 0:
            metrics['max_drawdown'] = min(metrics['max_drawdown'], pnl_pct)
    
    def get_timeframe_summary(self) -> Dict[str, Any]:
        """Get summary of all timeframe activities."""
        summary = {}
        
        for tf_type, tf_config in self.timeframe_configs.items():
            active_count = len(self.active_positions[tf_type])
            metrics = self.performance_metrics[tf_type]
            
            win_rate = (metrics['winning_trades'] / metrics['total_trades'] 
                       if metrics['total_trades'] > 0 else 0.0)
            
            summary[tf_type.value] = {
                'config': {
                    'interval': tf_config.interval,
                    'horizon_days': tf_config.horizon_days,
                    'weight': tf_config.weight,
                    'max_positions': tf_config.max_positions
                },
                'active_positions': active_count,
                'performance': {
                    'total_trades': metrics['total_trades'],
                    'win_rate': win_rate,
                    'total_pnl': metrics['total_pnl'],
                    'max_drawdown': metrics['max_drawdown']
                }
            }
        
        return summary
    
    def get_enabled_timeframes(self) -> List[TimeframeType]:
        """Get list of enabled timeframes."""
        return list(self.timeframe_configs.keys())
    
    def get_timeframe_allocation(self) -> Dict[str, float]:
        """Get capital allocation by timeframe."""
        return {tf_type.value: tf_config.weight 
                for tf_type, tf_config in self.timeframe_configs.items()}
