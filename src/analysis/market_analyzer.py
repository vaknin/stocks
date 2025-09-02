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
try:
    from src.models.ensemble import MetaLearningEnsemble
except ImportError as e:
    # Re-raise with more detailed error info
    raise ImportError(f"Failed to import MetaLearningEnsemble: {e}. Check if all dependencies are installed and paths are correct.") from e
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
        
        # Import expanded stock universe from config
        from ..config.settings import config
        self.target_stocks = config.STOCK_TICKERS
        
        # Optimized confidence thresholds based on 2024-2025 ML research
        self.confidence_thresholds = {
            'intraday': 0.72,  # 72% for 5-min scalping (ML-optimized)
            'daily': 0.78,     # 78% for daily swings (ML-optimized)
            'weekly': 0.68     # 68% for weekly positions (ML-optimized)
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
        
        # Sector classifications for bear market defensive weighting
        self.sector_classifications = {
            # Defensive sectors (perform better in bear markets)
            'defensive': {
                'utilities': ["NEE", "DUK", "SO", "D", "EXC", "SRE"],
                'consumer_staples': ["KO", "PG", "WMT", "PEP", "COST", "CL", "KMB", "GIS"],
                'healthcare': ["JNJ", "PFE", "UNH", "ABBV", "TMO", "AMGN", "GILD", "MRNA", "BMY", "LLY"]
            },
            # Cyclical sectors (more volatile in different market conditions)
            'cyclical': {
                'technology': ["NVDA", "AMD", "ASML", "TSM", "INTC", "QCOM", "AVGO", "MU", "SMCI", "ARM",
                             "AAPL", "MSFT", "GOOGL", "META", "CRM", "ADBE", "NFLX", "ORCL", "PYPL"],
                'financials': ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "BRK-B", "AXP", "C"],
                'consumer_discretionary': ["AMZN", "TSLA", "HD", "MCD", "SBUX", "NKE", "TGT", "LOW", "DIS", "BKNG"],
                'energy': ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO", "MPC"],
                'industrial': ["CAT", "BA", "GE", "UNP", "HON", "MMM"],
                'international': ["SHOP", "SAP"]
            }
        }
        
        # Market regime weightings for defensive positioning
        self.regime_sector_weights = {
            'bear_trend': {
                'defensive': 1.1,    # 10% boost for defensive stocks (reduced from excessive 50%)
                'cyclical': 0.7     # 30% reduction for cyclical stocks  
            },
            'high_volatility': {
                'defensive': 1.3,    # 30% boost for defensive stocks
                'cyclical': 0.8     # 20% reduction for cyclical stocks
            },
            'bull_trend': {
                'defensive': 0.9,    # 10% reduction for defensive stocks
                'cyclical': 1.2     # 20% boost for cyclical stocks
            },
            'sideways': {
                'defensive': 1.0,    # Neutral weighting
                'cyclical': 1.0     # Neutral weighting
            }
        }
        
        # Import optimized parameters from config
        from ..config.settings import config
        self.atr_multiplier = config.STOP_LOSS_ATR_MULTIPLIER
        
        logger.info("DailyMarketAnalyzer initialized successfully")
        logger.info(f"Loaded {len(self.target_stocks)} stocks across {len(self.sector_classifications['defensive']) + len(self.sector_classifications['cyclical'])} sectors")
    
    def fetch_market_data(self, lookback_days: int = 750) -> Dict[str, pd.DataFrame]:
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
                        # Handle yfinance column names (they are capitalized)
                        column_mapping = {
                            'Open': 'open',
                            'High': 'high', 
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume'
                        }
                        
                        # Check if we have the required columns
                        missing_columns = []
                        for expected_cap, expected_lower in column_mapping.items():
                            if expected_cap not in data.columns:
                                missing_columns.append(expected_cap)
                        
                        if not missing_columns:
                            # Rename columns to lowercase and select only what we need
                            data = data.rename(columns=column_mapping)
                            data = data[['open', 'high', 'low', 'close', 'volume']]
                            market_data[symbol] = data
                            logger.debug(f"Fetched {len(data)} days of data for {symbol}")
                        else:
                            failed_stocks.append(symbol)
                            logger.warning(f"Missing required columns for {symbol}. Missing: {missing_columns}, Available: {list(data.columns)}")
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
                # Create data dict for the regime detector
                regime_data = {primary_stock: market_data[primary_stock]}
                regime_state = self.regime_detector.detect_regime(regime_data)
                
                current_regime = regime_state.regime.value
                regime_confidence = regime_state.confidence
                
                logger.info(f"Detected regime: {current_regime} (confidence: {regime_confidence:.2f})")
                return current_regime, float(regime_confidence)
            else:
                logger.warning("Insufficient data for regime detection")
                return "unknown", 0.0
                
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return "unknown", 0.0
    
    def generate_stock_signal(self, symbol: str, data: pd.DataFrame, 
                            market_regime: str, timeframe: str = 'daily',
                            debug_info: Dict = None) -> Optional[MarketSignal]:
        """Generate trading signal for individual stock with detailed debugging."""
        
        # Initialize debug tracking for this stock
        stock_debug = {
            'symbol': symbol,
            'timeframe': timeframe,
            'data_length': len(data),
            'regime': market_regime,
            'filters_passed': [],
            'filters_failed': [],
            'final_result': None,
            'rejection_reason': None
        }
        
        try:
            if len(data) < 50:  # Need sufficient data
                stock_debug['rejection_reason'] = f"Insufficient data: {len(data)} < 50"
                stock_debug['final_result'] = 'REJECTED'
                if debug_info is not None:
                    debug_info.append(stock_debug)
                logger.info(f"🔍 {symbol}: REJECTED - {stock_debug['rejection_reason']}")
                return None
            
            logger.info(f"🔍 {symbol}: Starting signal generation ({timeframe})")
            
            # Prepare features for ensemble prediction
            features = self._prepare_features(data)
            if features is None or len(features) == 0:
                stock_debug['rejection_reason'] = "Feature preparation failed"
                stock_debug['final_result'] = 'REJECTED'
                if debug_info is not None:
                    debug_info.append(stock_debug)
                logger.info(f"🔍 {symbol}: REJECTED - {stock_debug['rejection_reason']}")
                return None
            
            stock_debug['filters_passed'].append("Data & Features OK")
            
            # Get ensemble prediction with uncertainty
            try:
                prediction_result = self.ensemble.predict(
                    data,
                    symbol,
                    return_confidence=True
                )
                
                # Extract horizon_1 prediction (1-day ahead)
                horizon_key = 'horizon_1'
                if horizon_key in prediction_result:
                    horizon_result = prediction_result[horizon_key]
                    prediction = horizon_result['prediction']
                    confidence = horizon_result.get('confidence', 0.6)
                    pred_interval = horizon_result.get('prediction_interval', (prediction * 0.95, prediction * 1.05))
                    stock_debug['prediction_source'] = 'ensemble'
                else:
                    # Fallback if no horizon_1 available
                    prediction = 0.0
                    confidence = 0.5
                    pred_interval = (prediction * 0.95, prediction * 1.05)
                    stock_debug['prediction_source'] = 'fallback_no_horizon'
                
            except Exception as e:
                logger.warning(f"Ensemble prediction failed for {symbol}, using fallback: {e}")
                # Fallback to simple technical analysis
                recent_returns = data['close'].pct_change().tail(20).mean()
                prediction = recent_returns * 5  # Scale up for target prediction
                confidence = 0.5  # Low confidence for fallback
                pred_interval = (prediction * 0.9, prediction * 1.1)
                stock_debug['prediction_source'] = 'fallback_technical'
                stock_debug['prediction_error'] = str(e)
            
            # Store prediction details for debugging
            stock_debug.update({
                'prediction': prediction,
                'confidence': confidence,
                'prediction_interval': pred_interval,
                'interval_width': pred_interval[1] - pred_interval[0],
                'interval_width_pct': ((pred_interval[1] - pred_interval[0]) / abs(prediction) * 100) if abs(prediction) > 0.001 else 0
            })
            
            logger.info(f"🔍 {symbol}: Prediction={prediction:.3%}, Confidence={confidence:.1%}, Interval=[{pred_interval[0]:.3%}, {pred_interval[1]:.3%}]")
            
            # Apply regime-based sector weighting for defensive positioning
            original_prediction, original_confidence = prediction, confidence
            prediction, confidence = self._apply_regime_weighting(prediction, confidence, symbol, market_regime)
            
            # Update debug info with adjusted values
            if original_prediction != prediction or original_confidence != confidence:
                stock_debug.update({
                    'original_prediction': original_prediction,
                    'original_confidence': original_confidence,
                    'sector_type': self._get_stock_sector_type(symbol),
                    'regime_weighting_applied': True
                })
                stock_debug['prediction'] = prediction  # Update with adjusted value
                stock_debug['confidence'] = confidence  # Update with adjusted value
                logger.info(f"🎯 {symbol}: Regime-adjusted Prediction={prediction:.3%}, Confidence={confidence:.1%}")
            
            # Apply confidence threshold
            confidence_threshold = self.confidence_thresholds[timeframe]
            if confidence < confidence_threshold:
                stock_debug['filters_failed'].append(f"Confidence {confidence:.1%} < {confidence_threshold:.1%}")
                stock_debug['rejection_reason'] = f"Low confidence: {confidence:.1%} < {confidence_threshold:.1%}"
                stock_debug['final_result'] = 'REJECTED'
                if debug_info is not None:
                    debug_info.append(stock_debug)
                logger.info(f"🔍 {symbol}: REJECTED - {stock_debug['rejection_reason']}")
                return None
            
            stock_debug['filters_passed'].append(f"Confidence {confidence:.1%} >= {confidence_threshold:.1%}")
            
            # Filter with MAPIE uncertainty
            try:
                filtered_prediction = self.uncertainty_filter.filter_prediction(
                    prediction, confidence, pred_interval
                )
                if filtered_prediction is None:
                    stock_debug['filters_failed'].append("MAPIE uncertainty filter")
                    stock_debug['rejection_reason'] = "Failed MAPIE uncertainty filtering"
                    stock_debug['final_result'] = 'REJECTED'
                    if debug_info is not None:
                        debug_info.append(stock_debug)
                    logger.info(f"🔍 {symbol}: REJECTED - {stock_debug['rejection_reason']}")
                    return None
                else:
                    stock_debug['filters_passed'].append("MAPIE uncertainty OK")
            except Exception as e:
                logger.warning(f"MAPIE filtering failed for {symbol}: {e}")
                stock_debug['mapie_error'] = str(e)
                stock_debug['filters_passed'].append("MAPIE skipped (error)")
            
            # Determine signal type
            min_return, max_return = self.return_targets[timeframe]
            min_return_threshold = min_return / 100.0  # Convert percentage to decimal
            
            stock_debug['min_return_threshold'] = min_return_threshold
            stock_debug['prediction_vs_threshold'] = prediction / min_return_threshold if min_return_threshold > 0 else 0
            
            if prediction > min_return_threshold:
                signal_type = "BUY"
                stock_debug['filters_passed'].append(f"Return {prediction:.3%} > {min_return_threshold:.3%} (BUY)")
            elif prediction < -min_return_threshold:
                signal_type = "SELL"  
                stock_debug['filters_passed'].append(f"Return {prediction:.3%} < {-min_return_threshold:.3%} (SELL)")
            else:
                signal_type = "HOLD"
                stock_debug['filters_failed'].append(f"Return {prediction:.3%} too small (abs < {min_return_threshold:.3%})")
            
            stock_debug['signal_type'] = signal_type
            
            # Note: Allow HOLD signals for portfolio transparency in bear markets
            # This was previously filtered out but now we show them for transparency
            
            # Calculate price targets
            current_price = data['close'].iloc[-1]
            target_price = current_price * (1 + prediction)
            
            # Calculate ATR-based dynamic stop loss (2024-2025 research optimized)
            atr_value = self._calculate_atr(data, period=14)
            atr_stop_distance = atr_value * self.atr_multiplier
            
            if signal_type == "BUY":
                stop_loss = current_price - atr_stop_distance
            else:
                stop_loss = current_price + atr_stop_distance
            
            # Store final prices for debugging
            stock_debug.update({
                'current_price': current_price,
                'target_price': target_price, 
                'stop_loss': stop_loss,
                'price_target_pct': (target_price / current_price - 1) * 100
            })
            
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
            
            # Final debugging info
            stock_debug['final_result'] = 'ACCEPTED'
            stock_debug['reasoning'] = reasoning
            if debug_info is not None:
                debug_info.append(stock_debug)
            
            logger.info(f"🔍 {symbol}: ACCEPTED - {signal_type} signal: {prediction:.3%} return (confidence: {confidence:.1%})")
            return signal
            
        except Exception as e:
            # Error handling with debugging
            stock_debug['final_result'] = 'ERROR'
            stock_debug['rejection_reason'] = f"Exception: {str(e)}"
            stock_debug['error_details'] = str(e)
            if debug_info is not None:
                debug_info.append(stock_debug)
            logger.error(f"🔍 {symbol}: ERROR - {e}")
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
    
    def _get_stock_sector_type(self, symbol: str) -> str:
        """Determine if a stock is defensive or cyclical based on sector."""
        # Check defensive sectors
        for sector_group, sectors in self.sector_classifications['defensive'].items():
            if symbol in sectors:
                return 'defensive'
        
        # Check cyclical sectors  
        for sector_group, sectors in self.sector_classifications['cyclical'].items():
            if symbol in sectors:
                return 'cyclical'
        
        # Default to cyclical if not found
        return 'cyclical'
    
    def _apply_regime_weighting(self, prediction: float, confidence: float, symbol: str, market_regime: str) -> Tuple[float, float]:
        """Apply regime-based sector weighting to predictions."""
        sector_type = self._get_stock_sector_type(symbol)
        weights = self.regime_sector_weights.get(market_regime, {'defensive': 1.0, 'cyclical': 1.0})
        weight = weights.get(sector_type, 1.0)
        
        # Apply weighting to both prediction and confidence
        # For defensive stocks in bear markets: boost both prediction and confidence
        # For cyclical stocks in bear markets: reduce both
        adjusted_prediction = prediction * weight
        
        # Confidence adjustment is more conservative (smaller effect)
        confidence_weight = 1.0 + (weight - 1.0) * 0.3  # 30% of the prediction weight effect
        adjusted_confidence = min(0.95, confidence * confidence_weight)  # Cap at 95%
        
        if weight != 1.0:
            logger.info(f"🎯 Regime Weighting Decision: {symbol} ({sector_type}) in {market_regime} regime: {weight:.1f}x applied - "
                        f"Pred: {prediction:.2%} → {adjusted_prediction:.2%}, Conf: {confidence:.1%} → {adjusted_confidence:.1%} "
                        f"(reduced from 1.5x to 1.1x for defensive stocks)")
        
        return adjusted_prediction, adjusted_confidence
    
    def _print_debugging_summary(self, debug_info: List[Dict], market_regime: str, timeframes: List[str]) -> None:
        """Print comprehensive debugging summary for all stocks."""
        if not debug_info:
            logger.info("🔍 DEBUGGING: No debug information collected")
            return
        
        logger.info("🔍" + "=" * 80)
        logger.info(f"🔍 SIGNAL GENERATION DEBUG SUMMARY")
        logger.info(f"🔍 Market Regime: {market_regime} | Timeframes: {timeframes}")
        logger.info("🔍" + "=" * 80)
        
        # Group by result type
        accepted = [d for d in debug_info if d['final_result'] == 'ACCEPTED']
        rejected = [d for d in debug_info if d['final_result'] == 'REJECTED']
        errors = [d for d in debug_info if d['final_result'] == 'ERROR']
        
        logger.info(f"🔍 RESULTS: {len(accepted)} ACCEPTED | {len(rejected)} REJECTED | {len(errors)} ERRORS")
        logger.info("🔍" + "-" * 80)
        
        # Show accepted signals
        if accepted:
            logger.info("🔍 ✅ ACCEPTED SIGNALS:")
            for stock in accepted:
                logger.info(f"🔍   {stock['symbol']:>6} | {stock['signal_type']:>4} | "
                          f"Pred: {stock['prediction']:>7.2%} | Conf: {stock['confidence']:>6.1%} | "
                          f"Source: {stock.get('prediction_source', 'unknown')}")
        
        # Show rejected signals with reasons
        if rejected:
            logger.info("🔍 ❌ REJECTED SIGNALS:")
            rejection_reasons = {}
            for stock in rejected:
                reason = stock['rejection_reason']
                if reason not in rejection_reasons:
                    rejection_reasons[reason] = []
                rejection_reasons[reason].append(stock)
            
            for reason, stocks in rejection_reasons.items():
                logger.info(f"🔍   {reason}: {len(stocks)} stocks")
                for stock in stocks:
                    pred = stock.get('prediction', 'N/A')
                    conf = stock.get('confidence', 'N/A')
                    pred_str = f"{pred:.2%}" if isinstance(pred, (int, float)) else str(pred)
                    conf_str = f"{conf:.1%}" if isinstance(conf, (int, float)) else str(conf)
                    logger.info(f"🔍     {stock['symbol']:>6} | Pred: {pred_str:>7} | Conf: {conf_str:>6}")
        
        # Show prediction statistics
        predictions = [d['prediction'] for d in debug_info if 'prediction' in d and isinstance(d['prediction'], (int, float))]
        confidences = [d['confidence'] for d in debug_info if 'confidence' in d and isinstance(d['confidence'], (int, float))]
        
        if predictions and confidences:
            logger.info("🔍" + "-" * 80)
            logger.info("🔍 📊 PREDICTION STATISTICS:")
            logger.info(f"🔍   Predictions - Mean: {np.mean(predictions):>7.2%} | "
                       f"Min: {np.min(predictions):>7.2%} | Max: {np.max(predictions):>7.2%}")
            logger.info(f"🔍   Confidences - Mean: {np.mean(confidences):>7.1%} | "
                       f"Min: {np.min(confidences):>7.1%} | Max: {np.max(confidences):>7.1%}")
            
            # Show how many would pass with different thresholds
            threshold_2pct = sum(1 for p in predictions if abs(p) >= 0.02)
            threshold_1pct = sum(1 for p in predictions if abs(p) >= 0.01)
            threshold_0_5pct = sum(1 for p in predictions if abs(p) >= 0.005)
            
            logger.info(f"🔍   Return Thresholds - ≥2%: {threshold_2pct} stocks | "
                       f"≥1%: {threshold_1pct} stocks | ≥0.5%: {threshold_0_5pct} stocks")
            
            # Show confidence distribution
            conf_75 = sum(1 for c in confidences if c >= 0.75)
            conf_65 = sum(1 for c in confidences if c >= 0.65)
            conf_50 = sum(1 for c in confidences if c >= 0.50)
            
            logger.info(f"🔍   Confidence Levels - ≥75%: {conf_75} stocks | "
                       f"≥65%: {conf_65} stocks | ≥50%: {conf_50} stocks")
        
        logger.info("🔍" + "=" * 80)
    
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
            
            # Generate signals for all stocks and timeframes with debugging
            all_signals = []
            debug_info = []  # Collect debugging information for all stocks
            
            logger.info(f"🔍 DEBUGGING: Analyzing {len(market_data)} stocks for {timeframes} timeframes")
            
            for timeframe in timeframes:
                for symbol in market_data.keys():
                    signal = self.generate_stock_signal(
                        symbol, market_data[symbol], market_regime, timeframe, debug_info
                    )
                    if signal:
                        all_signals.append(signal)
            
            # Print comprehensive debugging summary
            self._print_debugging_summary(debug_info, market_regime, timeframes)
            
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
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range for dynamic stop losses."""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range components
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            # True Range is the maximum of the three
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR as the rolling mean of True Range
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 0.02 * close.iloc[-1]  # Fallback to 2% of price
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.02 * data['close'].iloc[-1]  # Fallback


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