"""
Daily Market Analysis Pipeline
Integrates with existing ensemble models to generate trading signals and market insights.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
import yfinance as yf

# Import our existing models
from src.models.ensemble import MetaLearningEnsemble
from src.models.regime_detector import HiddenMarkovRegimeDetector
from src.models.uncertainty import TradingUncertaintyFilter
from src.data.ingester import DataIngester


@dataclass
class MarketSignal:
    """Individual stock trading signal."""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    predicted_return: float
    prediction_interval: Tuple[float, float]
    target_price: float
    stop_loss: float
    timeframe: str  # 'intraday', 'daily', 'weekly'
    reasoning: str
    market_regime: str


@dataclass
class MarketAnalysis:
    """Complete market analysis results."""
    analysis_date: str
    market_regime: str
    regime_confidence: float
    recommended_allocation: float  # Portfolio allocation percentage
    signals: List[MarketSignal]
    market_sentiment: str
    risk_level: str


class DailyMarketAnalyzer:
    """Daily market analysis using our complete AI ensemble."""
    
    def __init__(self):
        logger.info("Initializing DailyMarketAnalyzer with AI ensemble")
        
        # Initialize components
        self.data_ingester = DataIngester()
        self.ensemble = MetaLearningEnsemble()
        self.regime_detector = HiddenMarkovRegimeDetector()
        self.uncertainty_filter = TradingUncertaintyFilter()
        
        # Semiconductor stocks from README.md
        self.target_stocks = [
            "NVDA", "AMD", "ASML", "TSM", "INTC", 
            "QCOM", "AVGO", "MU", "SMCI", "ARM"
        ]
        
        # Trading parameters from README.md
        self.confidence_thresholds = {
            'intraday': 0.70,  # 70% for 5-min scalping
            'daily': 0.75,     # 75% for daily swings  
            'weekly': 0.75     # 75% for weekly positions
        }
        
        self.return_targets = {
            'intraday': (0.5, 1.0),   # 0.5-1% per trade
            'daily': (2.0, 5.0),      # 2-5% targets
            'weekly': (5.0, 10.0)     # 5-10% targets
        }
        
        # Regime-based allocation from README.md
        self.regime_allocations = {
            'bull_trend': 0.75,      # 75% allocation 
            'bear_trend': 0.40,      # 40% allocation
            'high_volatility': 0.25, # 25% allocation
            'sideways': 0.65         # 65% allocation
        }
        
        logger.info("DailyMarketAnalyzer initialized successfully")
    
    def fetch_market_data(self, lookback_days: int = 252) -> Dict[str, pd.DataFrame]:
        """Fetch recent market data for all target stocks."""
        try:
            logger.info(f"Fetching market data for {len(self.target_stocks)} stocks")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            market_data = {}
            failed_stocks = []
            
            for symbol in self.target_stocks:
                try:
                    # Use yfinance for quick data fetch
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        interval='1d'
                    )
                    
                    if not data.empty and len(data) > 50:  # Ensure sufficient data
                        # Standardize column names
                        data.columns = ['open', 'high', 'low', 'close', 'volume']
                        market_data[symbol] = data
                        logger.debug(f"Fetched {len(data)} days of data for {symbol}")
                    else:
                        failed_stocks.append(symbol)
                        logger.warning(f"Insufficient data for {symbol}")
                
                except Exception as e:
                    failed_stocks.append(symbol)
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
            
            if failed_stocks:
                logger.warning(f"Failed to fetch data for stocks: {failed_stocks}")
            
            logger.info(f"Successfully fetched data for {len(market_data)} stocks")
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}
    
    def detect_market_regime(self, market_data: Dict[str, pd.DataFrame]) -> Tuple[str, float]:
        """Detect current market regime using HMM."""
        try:
            logger.info("Detecting market regime")
            
            if not market_data:
                return "unknown", 0.0
            
            # Use SPY or a representative stock for regime detection
            # For now, use NVDA as it's highly representative of semiconductor sector
            primary_stock = "NVDA" if "NVDA" in market_data else list(market_data.keys())[0]
            data = market_data[primary_stock]
            
            # Calculate returns for regime detection
            returns = data['close'].pct_change().dropna()
            
            # Use our HMM regime detector
            if len(returns) >= 100:  # Need sufficient data
                regime_probs = self.regime_detector.predict_regime(returns.values.reshape(-1, 1))
                current_regime_idx = regime_probs[-1].argmax()
                regime_confidence = regime_probs[-1].max()
                
                # Map regime index to regime name
                regime_names = ['bull_trend', 'bear_trend', 'high_volatility', 'sideways']
                current_regime = regime_names[current_regime_idx] if current_regime_idx < len(regime_names) else 'unknown'
                
                logger.info(f"Detected regime: {current_regime} (confidence: {regime_confidence:.2f})")
                return current_regime, float(regime_confidence)
            else:
                logger.warning("Insufficient data for regime detection")
                return "unknown", 0.0
                
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return "unknown", 0.0
    
    def generate_stock_signal(self, symbol: str, data: pd.DataFrame, 
                            market_regime: str, timeframe: str = 'daily') -> Optional[MarketSignal]:
        """Generate trading signal for individual stock."""
        try:
            if len(data) < 50:  # Need sufficient data
                return None
            
            logger.debug(f"Generating signal for {symbol} ({timeframe})")
            
            # Prepare features for ensemble prediction
            features = self._prepare_features(data)
            if features is None or len(features) == 0:
                return None
            
            # Get ensemble prediction with uncertainty
            try:
                prediction_result = self.ensemble.predict(
                    features.values.reshape(1, -1),
                    return_uncertainty=True
                )
                
                if isinstance(prediction_result, tuple):
                    prediction, confidence, pred_interval = prediction_result
                    prediction = prediction[0] if hasattr(prediction, '__len__') else prediction
                else:
                    prediction = prediction_result
                    confidence = 0.6  # Default moderate confidence
                    pred_interval = (prediction * 0.95, prediction * 1.05)  # 5% interval
                
            except Exception as e:
                logger.warning(f"Ensemble prediction failed for {symbol}, using fallback: {e}")
                # Fallback to simple technical analysis
                recent_returns = data['close'].pct_change().tail(20).mean()
                prediction = recent_returns * 5  # Scale up for target prediction
                confidence = 0.5  # Low confidence for fallback
                pred_interval = (prediction * 0.9, prediction * 1.1)
            
            # Apply confidence threshold
            confidence_threshold = self.confidence_thresholds[timeframe]
            if confidence < confidence_threshold:
                logger.debug(f"Signal filtered: {symbol} confidence {confidence:.2f} < {confidence_threshold}")
                return None
            
            # Filter with MAPIE uncertainty
            try:
                filtered_prediction = self.uncertainty_filter.filter_prediction(
                    prediction, confidence, pred_interval
                )
                if filtered_prediction is None:
                    logger.debug(f"Signal filtered by MAPIE uncertainty: {symbol}")
                    return None
            except Exception as e:
                logger.warning(f"MAPIE filtering failed for {symbol}: {e}")
                # Continue without MAPIE filtering
            
            # Determine signal type
            min_return, max_return = self.return_targets[timeframe]
            min_return_threshold = min_return / 100.0  # Convert percentage to decimal
            
            if prediction > min_return_threshold:
                signal_type = "BUY"
            elif prediction < -min_return_threshold:
                signal_type = "SELL"
            else:
                signal_type = "HOLD"
            
            if signal_type == "HOLD":
                return None  # Don't generate HOLD signals
            
            # Calculate price targets
            current_price = data['close'].iloc[-1]
            target_price = current_price * (1 + prediction)
            
            # Set stop loss (2% max loss per README risk management)
            if signal_type == "BUY":
                stop_loss = current_price * 0.98  # 2% stop loss
            else:
                stop_loss = current_price * 1.02  # 2% stop loss for short
            
            # Generate reasoning
            reasoning = self._generate_reasoning(symbol, prediction, confidence, market_regime, timeframe)
            
            signal = MarketSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                predicted_return=prediction,
                prediction_interval=pred_interval,
                target_price=target_price,
                stop_loss=stop_loss,
                timeframe=timeframe,
                reasoning=reasoning,
                market_regime=market_regime
            )
            
            logger.debug(f"Generated {signal_type} signal for {symbol}: {prediction:.2%} return (confidence: {confidence:.2f})")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _prepare_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare features for ensemble prediction."""
        try:
            if len(data) < 20:
                return None
            
            # Basic technical features
            features = pd.DataFrame(index=[data.index[-1]])
            
            # Price features
            features['current_price'] = data['close'].iloc[-1]
            features['price_change_1d'] = data['close'].pct_change().iloc[-1]
            features['price_change_5d'] = data['close'].pct_change(5).iloc[-1]
            features['price_change_20d'] = data['close'].pct_change(20).iloc[-1]
            
            # Volume features
            features['volume_ratio'] = data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]
            
            # Moving averages
            features['sma_5'] = data['close'].rolling(5).mean().iloc[-1]
            features['sma_20'] = data['close'].rolling(20).mean().iloc[-1]
            features['price_vs_sma5'] = (data['close'].iloc[-1] / features['sma_5']) - 1
            features['price_vs_sma20'] = (data['close'].iloc[-1] / features['sma_20']) - 1
            
            # Volatility
            features['volatility_20d'] = data['close'].pct_change().rolling(20).std().iloc[-1]
            
            # RSI-like momentum
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['momentum'] = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Fill any NaN values
            features = features.fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def _generate_reasoning(self, symbol: str, prediction: float, confidence: float, 
                          market_regime: str, timeframe: str) -> str:
        """Generate human-readable reasoning for the trading signal."""
        signal_type = "BUY" if prediction > 0 else "SELL"
        
        reasoning_parts = [
            f"AI ensemble predicts {prediction:.1%} return with {confidence:.0%} confidence",
            f"Current market regime: {market_regime.replace('_', ' ').title()}",
            f"{timeframe.capitalize()} timeframe analysis"
        ]
        
        # Add regime-specific reasoning
        if market_regime == "bull_trend":
            reasoning_parts.append("Bull market supports momentum strategy")
        elif market_regime == "bear_trend":
            reasoning_parts.append("Bear market - defensive positioning")
        elif market_regime == "high_volatility":
            reasoning_parts.append("High volatility - quick profit taking recommended")
        elif market_regime == "sideways":
            reasoning_parts.append("Range-bound market - mean reversion play")
        
        return ". ".join(reasoning_parts)
    
    def analyze_market(self, timeframes: List[str] = ['daily']) -> MarketAnalysis:
        """Perform comprehensive daily market analysis."""
        try:
            logger.info("Starting daily market analysis")
            analysis_date = datetime.now().strftime("%Y-%m-%d")
            
            # Fetch market data
            market_data = self.fetch_market_data()
            if not market_data:
                logger.error("No market data available")
                return MarketAnalysis(
                    analysis_date=analysis_date,
                    market_regime="unknown",
                    regime_confidence=0.0,
                    recommended_allocation=0.0,
                    signals=[],
                    market_sentiment="unknown",
                    risk_level="high"
                )
            
            # Detect market regime
            market_regime, regime_confidence = self.detect_market_regime(market_data)
            
            # Get recommended allocation based on regime
            recommended_allocation = self.regime_allocations.get(market_regime, 0.5)
            
            # Generate signals for all stocks and timeframes
            all_signals = []
            for timeframe in timeframes:
                for symbol in market_data.keys():
                    signal = self.generate_stock_signal(
                        symbol, market_data[symbol], market_regime, timeframe
                    )
                    if signal:
                        all_signals.append(signal)
            
            # Determine market sentiment and risk level
            buy_signals = [s for s in all_signals if s.signal_type == "BUY"]
            sell_signals = [s for s in all_signals if s.signal_type == "SELL"]
            
            if len(buy_signals) > len(sell_signals) * 1.5:
                market_sentiment = "bullish"
            elif len(sell_signals) > len(buy_signals) * 1.5:
                market_sentiment = "bearish"
            else:
                market_sentiment = "neutral"
            
            # Risk level based on regime and signal confidence
            avg_confidence = np.mean([s.confidence for s in all_signals]) if all_signals else 0.0
            if market_regime == "high_volatility" or avg_confidence < 0.6:
                risk_level = "high"
            elif regime_confidence > 0.8 and avg_confidence > 0.8:
                risk_level = "low"
            else:
                risk_level = "medium"
            
            analysis = MarketAnalysis(
                analysis_date=analysis_date,
                market_regime=market_regime,
                regime_confidence=regime_confidence,
                recommended_allocation=recommended_allocation,
                signals=all_signals,
                market_sentiment=market_sentiment,
                risk_level=risk_level
            )
            
            logger.info(f"Market analysis complete: {len(all_signals)} signals generated, "
                       f"regime: {market_regime}, sentiment: {market_sentiment}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return MarketAnalysis(
                analysis_date=datetime.now().strftime("%Y-%m-%d"),
                market_regime="error",
                regime_confidence=0.0,
                recommended_allocation=0.0,
                signals=[],
                market_sentiment="unknown",
                risk_level="high"
            )


if __name__ == "__main__":
    # Test the market analyzer
    analyzer = DailyMarketAnalyzer()
    analysis = analyzer.analyze_market(['daily'])
    
    print(f"Market Analysis for {analysis.analysis_date}")
    print(f"Regime: {analysis.market_regime} (confidence: {analysis.regime_confidence:.2f})")
    print(f"Recommended Allocation: {analysis.recommended_allocation:.0%}")
    print(f"Market Sentiment: {analysis.market_sentiment}")
    print(f"Risk Level: {analysis.risk_level}")
    print(f"Generated {len(analysis.signals)} trading signals")
    
    for signal in analysis.signals[:5]:  # Show first 5 signals
        print(f"  {signal.signal_type} {signal.symbol}: {signal.predicted_return:.1%} "
              f"(confidence: {signal.confidence:.0%})")