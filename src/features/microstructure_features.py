"""Microstructure features with volume profile and tick-level approximations."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import warnings
from loguru import logger
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum

from ..config.settings import config


class VolumeProfileType(Enum):
    """Types of volume profile analysis."""
    PRICE_VOLUME = "price_volume"
    TIME_VOLUME = "time_volume"
    DELTA_VOLUME = "delta_volume"


class MicrostructureRegime(Enum):
    """Microstructure regime types."""
    HIGH_LIQUIDITY = "high_liquidity"
    LOW_LIQUIDITY = "low_liquidity"
    PRICE_DISCOVERY = "price_discovery"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    INFORMATION_DRIVEN = "information_driven"


@dataclass
class VolumeNode:
    """Volume profile node representing price level activity."""
    price_level: float
    total_volume: float
    buy_volume: float
    sell_volume: float
    delta: float
    transactions: int
    time_spent: float


@dataclass
class MarketMicrostructure:
    """Market microstructure analysis results."""
    bid_ask_spread: float
    effective_spread: float
    price_impact: float
    volume_imbalance: float
    liquidity_ratio: float
    market_depth: float
    regime: MicrostructureRegime


class MicrostructureFeatureExtractor:
    """
    Advanced microstructure feature extractor.
    
    Features:
    - Volume profile analysis (POC, VAH, VAL)
    - Tick-level approximations from OHLCV data
    - Order flow imbalance estimation
    - Market depth and liquidity metrics
    - Price discovery and information flow
    - Regime-specific microstructure analysis
    """
    
    def __init__(
        self,
        volume_profile_bins: int = 50,
        lookback_periods: List[int] = [20, 50, 100],
        tick_approximation_method: str = "high_low_volume",
        enable_regime_detection: bool = True,
        liquidity_window: int = 20
    ):
        """
        Initialize microstructure feature extractor.
        
        Args:
            volume_profile_bins: Number of bins for volume profile analysis
            lookback_periods: Different lookback periods for analysis
            tick_approximation_method: Method for tick-level approximation
            enable_regime_detection: Enable microstructure regime detection
            liquidity_window: Window for liquidity calculations
        """
        self.volume_profile_bins = volume_profile_bins
        self.lookback_periods = lookback_periods
        self.tick_approximation_method = tick_approximation_method
        self.enable_regime_detection = enable_regime_detection
        self.liquidity_window = liquidity_window
        
        # Feature caching
        self.feature_cache = {}
        self.cache_timestamps = {}
        self.cache_ttl_minutes = 10
        
        # Volume profile cache
        self.volume_profile_cache = {}
        
        # Microstructure regime tracking
        self.regime_history = deque(maxlen=100)
        
        logger.info("MicrostructureFeatureExtractor initialized")
    
    def extract_features(
        self,
        df: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        ticker: str,
        include_volume_profile: bool = True,
        include_tick_features: bool = True,
        include_liquidity_features: bool = True
    ) -> np.ndarray:
        """
        Extract comprehensive microstructure features.
        
        Args:
            df: OHLCV price data
            ticker: Stock ticker symbol
            include_volume_profile: Include volume profile features
            include_tick_features: Include tick-level approximation features
            include_liquidity_features: Include liquidity and market depth features
            
        Returns:
            Array of microstructure features
        """
        try:
            # Handle input data format
            if isinstance(df, dict):
                if ticker not in df:
                    logger.warning(f"Ticker {ticker} not found in multi-asset data")
                    return self._get_default_features()
                primary_df = df[ticker]
            else:
                primary_df = df
            
            # Minimum data requirement
            if len(primary_df) < max(self.lookback_periods):
                logger.warning(f"Insufficient data for {ticker}: {len(primary_df)} rows")
                return self._get_default_features()
            
            # Check cache
            cache_key = f"{ticker}_{len(primary_df)}_{primary_df.index[-1]}"
            if self._is_cache_valid(cache_key):
                logger.debug(f"Using cached microstructure features for {ticker}")
                return self.feature_cache[cache_key]
            
            # Extract different feature groups
            all_features = []
            
            # Volume profile features
            if include_volume_profile:
                vp_features = self._extract_volume_profile_features(primary_df)
                all_features.extend(vp_features)
            
            # Tick-level approximation features
            if include_tick_features:
                tick_features = self._extract_tick_approximation_features(primary_df)
                all_features.extend(tick_features)
            
            # Liquidity and market depth features
            if include_liquidity_features:
                liquidity_features = self._extract_liquidity_features(primary_df)
                all_features.extend(liquidity_features)
            
            # Order flow features
            order_flow_features = self._extract_order_flow_features(primary_df)
            all_features.extend(order_flow_features)
            
            # Price discovery features
            price_discovery_features = self._extract_price_discovery_features(primary_df)
            all_features.extend(price_discovery_features)
            
            # Regime-specific features
            if self.enable_regime_detection:
                regime_features = self._extract_regime_features(primary_df)
                all_features.extend(regime_features)
            
            # Convert to numpy array and clean
            features_array = np.array(all_features, dtype=np.float32)
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
            features_array = np.clip(features_array, -10.0, 10.0)
            
            # Ensure minimum feature count
            min_features = 40
            if len(features_array) < min_features:
                features_array = np.pad(features_array, (0, min_features - len(features_array)))
            
            # Cache result
            self._cache_features(cache_key, features_array)
            
            logger.debug(f"Generated {len(features_array)} microstructure features for {ticker}")
            
            return features_array
            
        except Exception as e:
            logger.error(f"Error extracting microstructure features for {ticker}: {e}")
            return self._get_default_features()
    
    def _extract_volume_profile_features(self, df: pd.DataFrame) -> List[float]:
        """Extract volume profile features (POC, VAH, VAL, etc.)."""
        try:
            features = []
            
            for lookback in self.lookback_periods:
                if len(df) < lookback:
                    features.extend([0.0] * 6)  # Default features per lookback
                    continue
                
                # Get data for this lookback period
                period_df = df.tail(lookback)
                
                # Create volume profile
                volume_profile = self._create_volume_profile(period_df)
                
                if not volume_profile:
                    features.extend([0.0] * 6)
                    continue
                
                # Calculate key volume profile levels
                poc, vah, val = self._calculate_volume_profile_levels(volume_profile)
                current_price = period_df['close'].iloc[-1]
                
                # Volume profile features
                features.extend([
                    (current_price - poc) / current_price if poc > 0 else 0.0,  # Distance from POC
                    (current_price - vah) / current_price if vah > 0 else 0.0,  # Distance from VAH
                    (current_price - val) / current_price if val > 0 else 0.0,  # Distance from VAL
                    self._calculate_volume_concentration(volume_profile),        # Volume concentration
                    self._calculate_volume_balance(volume_profile),             # Buy/sell balance
                    self._calculate_profile_shape(volume_profile)               # Profile shape factor
                ])
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting volume profile features: {e}")
            return [0.0] * (len(self.lookback_periods) * 6)
    
    def _create_volume_profile(self, df: pd.DataFrame) -> List[VolumeNode]:
        """Create volume profile from OHLCV data."""
        try:
            if len(df) < 2:
                return []
            
            # Determine price range
            price_min = df['low'].min() if 'low' in df.columns else df['close'].min()
            price_max = df['high'].max() if 'high' in df.columns else df['close'].max()
            
            if price_max <= price_min:
                return []
            
            # Create price bins
            bin_size = (price_max - price_min) / self.volume_profile_bins
            bins = [price_min + i * bin_size for i in range(self.volume_profile_bins + 1)]
            
            # Initialize volume nodes
            volume_nodes = []
            for i in range(self.volume_profile_bins):
                volume_nodes.append(VolumeNode(
                    price_level=(bins[i] + bins[i + 1]) / 2,
                    total_volume=0.0,
                    buy_volume=0.0,
                    sell_volume=0.0,
                    delta=0.0,
                    transactions=0,
                    time_spent=0.0
                ))
            
            # Aggregate volume into bins
            for _, row in df.iterrows():
                # Approximate volume distribution within OHLC range
                high = row.get('high', row['close'])
                low = row.get('low', row['close'])
                close = row['close']
                volume = row.get('volume', 0)
                
                if volume <= 0 or high <= low:
                    continue
                
                # Simple approximation: distribute volume proportionally to price range
                price_range = high - low
                if price_range > 0:
                    # More sophisticated volume distribution
                    # Assume more volume near close and weighted by candle body
                    open_price = row.get('open', close)
                    body_size = abs(close - open_price)
                    wick_weight = 0.3  # 30% of volume in wicks, 70% in body
                    
                    # Distribute volume to appropriate bins
                    for node in volume_nodes:
                        price_level = node.price_level
                        
                        if low <= price_level <= high:
                            # Calculate volume weight for this price level
                            if body_size > 0:
                                body_min = min(open_price, close)
                                body_max = max(open_price, close)
                                
                                if body_min <= price_level <= body_max:
                                    # Within body - higher weight
                                    weight = (1 - wick_weight) / max(1, body_max - body_min)
                                else:
                                    # In wicks - lower weight
                                    weight = wick_weight / max(1, price_range - body_size)
                            else:
                                # Doji or very small body
                                weight = 1.0 / max(1, price_range)
                            
                            level_volume = volume * weight
                            node.total_volume += level_volume
                            node.transactions += 1
                            
                            # Approximate buy/sell split based on close relative to OHLC
                            if close > open_price:
                                # Bullish candle - more buying pressure
                                if price_level >= (open_price + close) / 2:
                                    node.buy_volume += level_volume * 0.7
                                    node.sell_volume += level_volume * 0.3
                                else:
                                    node.buy_volume += level_volume * 0.6
                                    node.sell_volume += level_volume * 0.4
                            else:
                                # Bearish candle - more selling pressure
                                if price_level <= (open_price + close) / 2:
                                    node.buy_volume += level_volume * 0.3
                                    node.sell_volume += level_volume * 0.7
                                else:
                                    node.buy_volume += level_volume * 0.4
                                    node.sell_volume += level_volume * 0.6
                            
                            node.delta = node.buy_volume - node.sell_volume
            
            return [node for node in volume_nodes if node.total_volume > 0]
            
        except Exception as e:
            logger.debug(f"Error creating volume profile: {e}")
            return []
    
    def _calculate_volume_profile_levels(self, volume_profile: List[VolumeNode]) -> Tuple[float, float, float]:
        """Calculate POC (Point of Control), VAH (Value Area High), VAL (Value Area Low)."""
        try:
            if not volume_profile:
                return 0.0, 0.0, 0.0
            
            # Sort by volume to find POC
            sorted_nodes = sorted(volume_profile, key=lambda x: x.total_volume, reverse=True)
            poc_price = sorted_nodes[0].price_level
            
            # Calculate total volume
            total_volume = sum(node.total_volume for node in volume_profile)
            
            # Find value area (70% of volume)
            value_area_volume = total_volume * 0.7
            
            # Sort by price level
            price_sorted = sorted(volume_profile, key=lambda x: x.price_level)
            
            # Find optimal value area around POC
            poc_index = next((i for i, node in enumerate(price_sorted) if node.price_level == poc_price), 0)
            
            # Expand around POC to capture 70% of volume
            volume_captured = price_sorted[poc_index].total_volume
            low_idx = poc_index
            high_idx = poc_index
            
            while volume_captured < value_area_volume and (low_idx > 0 or high_idx < len(price_sorted) - 1):
                # Expand to the side with higher volume
                low_volume = price_sorted[low_idx - 1].total_volume if low_idx > 0 else 0
                high_volume = price_sorted[high_idx + 1].total_volume if high_idx < len(price_sorted) - 1 else 0
                
                if low_volume >= high_volume and low_idx > 0:
                    low_idx -= 1
                    volume_captured += low_volume
                elif high_idx < len(price_sorted) - 1:
                    high_idx += 1
                    volume_captured += high_volume
                else:
                    break
            
            val = price_sorted[low_idx].price_level
            vah = price_sorted[high_idx].price_level
            
            return poc_price, vah, val
            
        except Exception as e:
            logger.debug(f"Error calculating volume profile levels: {e}")
            return 0.0, 0.0, 0.0
    
    def _calculate_volume_concentration(self, volume_profile: List[VolumeNode]) -> float:
        """Calculate volume concentration (how concentrated volume is)."""
        try:
            if not volume_profile:
                return 0.5
            
            volumes = [node.total_volume for node in volume_profile]
            total_volume = sum(volumes)
            
            if total_volume <= 0:
                return 0.5
            
            # Calculate Gini coefficient for volume distribution
            sorted_volumes = sorted(volumes)
            n = len(sorted_volumes)
            
            gini_sum = sum((2 * i - n - 1) * vol for i, vol in enumerate(sorted_volumes, 1))
            gini = gini_sum / (n * total_volume)
            
            return max(0.0, min(1.0, gini))
            
        except Exception as e:
            logger.debug(f"Error calculating volume concentration: {e}")
            return 0.5
    
    def _calculate_volume_balance(self, volume_profile: List[VolumeNode]) -> float:
        """Calculate overall buy/sell volume balance."""
        try:
            if not volume_profile:
                return 0.0
            
            total_buy = sum(node.buy_volume for node in volume_profile)
            total_sell = sum(node.sell_volume for node in volume_profile)
            total_volume = total_buy + total_sell
            
            if total_volume <= 0:
                return 0.0
            
            # Return buy dominance ratio (-1 to 1)
            return (total_buy - total_sell) / total_volume
            
        except Exception as e:
            logger.debug(f"Error calculating volume balance: {e}")
            return 0.0
    
    def _calculate_profile_shape(self, volume_profile: List[VolumeNode]) -> float:
        """Calculate volume profile shape factor."""
        try:
            if not volume_profile or len(volume_profile) < 3:
                return 0.0
            
            # Sort by price level
            sorted_profile = sorted(volume_profile, key=lambda x: x.price_level)
            
            # Calculate shape metrics
            volumes = [node.total_volume for node in sorted_profile]
            
            # Skewness of volume distribution
            mean_vol = np.mean(volumes)
            if mean_vol > 0 and np.std(volumes) > 0:
                skewness = np.mean([(vol - mean_vol) ** 3 for vol in volumes]) / (np.std(volumes) ** 3)
                return max(-2.0, min(2.0, skewness))
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating profile shape: {e}")
            return 0.0
    
    def _extract_tick_approximation_features(self, df: pd.DataFrame) -> List[float]:
        """Extract tick-level approximation features from OHLCV data."""
        try:
            features = []
            
            if len(df) < self.liquidity_window:
                return [0.0] * 15  # Default tick features
            
            # Approximate tick features using high-low-volume method
            recent_df = df.tail(self.liquidity_window)
            
            # 1. Approximate tick count (using high-low range)
            tick_counts = []
            for _, row in recent_df.iterrows():
                high = row.get('high', row['close'])
                low = row.get('low', row['close'])
                close = row['close']
                
                # Approximate tick count based on price range
                # Assume average tick size is 0.01 for most stocks
                tick_size = 0.01
                estimated_ticks = max(1, int((high - low) / tick_size))
                tick_counts.append(estimated_ticks)
            
            avg_ticks = np.mean(tick_counts)
            tick_volatility = np.std(tick_counts)
            
            features.extend([
                np.log(avg_ticks + 1) / 10,          # Log-normalized average ticks
                min(tick_volatility / avg_ticks, 1.0) if avg_ticks > 0 else 0.0  # Tick volatility ratio
            ])
            
            # 2. Price impact approximation
            price_impacts = []
            volumes = recent_df['volume'].values
            returns = recent_df['close'].pct_change().abs().values[1:]
            
            for i in range(len(returns)):
                if volumes[i + 1] > 0:
                    # Approximate price impact per unit volume
                    impact = returns[i] / np.log(volumes[i + 1] + 1)
                    price_impacts.append(impact)
            
            if price_impacts:
                avg_impact = np.mean(price_impacts)
                features.append(min(avg_impact * 1000, 1.0))  # Scaled price impact
            else:
                features.append(0.0)
            
            # 3. Bid-ask spread approximation
            spread_approximations = []
            for _, row in recent_df.iterrows():
                high = row.get('high', row['close'])
                low = row.get('low', row['close'])
                close = row['close']
                
                # Approximate spread using high-low range
                if close > 0:
                    spread_approx = (high - low) / close
                    spread_approximations.append(spread_approx)
            
            if spread_approximations:
                avg_spread = np.mean(spread_approximations)
                spread_volatility = np.std(spread_approximations)
                features.extend([
                    min(avg_spread, 0.1),  # Average spread (capped at 10%)
                    min(spread_volatility, 0.05)  # Spread volatility (capped at 5%)
                ])
            else:
                features.extend([0.0, 0.0])
            
            # 4. Volume arrival rate approximation
            volume_rates = []
            for i in range(1, len(recent_df)):
                prev_volume = recent_df['volume'].iloc[i - 1]
                curr_volume = recent_df['volume'].iloc[i]
                
                if prev_volume > 0:
                    rate_change = (curr_volume - prev_volume) / prev_volume
                    volume_rates.append(rate_change)
            
            if volume_rates:
                avg_rate_change = np.mean(volume_rates)
                rate_volatility = np.std(volume_rates)
                features.extend([
                    max(-1.0, min(1.0, avg_rate_change)),  # Average volume rate change
                    min(rate_volatility, 2.0)              # Volume rate volatility
                ])
            else:
                features.extend([0.0, 0.0])
            
            # 5. Trade size distribution approximation
            # Use volume/tick_count as approximate trade size
            trade_sizes = []
            for i, tick_count in enumerate(tick_counts):
                volume = recent_df['volume'].iloc[i]
                if tick_count > 0:
                    avg_trade_size = volume / tick_count
                    trade_sizes.append(avg_trade_size)
            
            if trade_sizes:
                # Trade size characteristics
                avg_trade_size = np.mean(trade_sizes)
                trade_size_std = np.std(trade_sizes)
                
                # Large trade indicator (top 20%)
                large_trade_threshold = np.percentile(trade_sizes, 80)
                large_trade_ratio = len([ts for ts in trade_sizes if ts >= large_trade_threshold]) / len(trade_sizes)
                
                features.extend([
                    np.log(avg_trade_size + 1) / 20,       # Log-normalized average trade size
                    min(trade_size_std / avg_trade_size, 2.0) if avg_trade_size > 0 else 0.0,  # Trade size CV
                    large_trade_ratio                       # Large trade ratio
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # 6. Information flow approximation
            # Based on price-volume relationship
            price_changes = recent_df['close'].pct_change().abs().values[1:]
            volume_changes = recent_df['volume'].pct_change().abs().values[1:]
            
            info_flows = []
            for pc, vc in zip(price_changes, volume_changes):
                if not (np.isnan(pc) or np.isnan(vc)):
                    # Information flow proxy: price sensitivity to volume
                    if vc > 0:
                        info_flow = pc / (vc + 0.001)  # Add small constant to avoid division by zero
                        info_flows.append(info_flow)
            
            if info_flows:
                avg_info_flow = np.mean(info_flows)
                info_flow_volatility = np.std(info_flows)
                features.extend([
                    min(avg_info_flow, 5.0),              # Average information flow
                    min(info_flow_volatility, 2.0)        # Information flow volatility
                ])
            else:
                features.extend([0.0, 0.0])
            
            # 7. Market efficiency approximation
            # Based on return autocorrelation
            returns = recent_df['close'].pct_change().dropna()
            if len(returns) > 10:
                autocorr_1 = returns.autocorr(lag=1) if len(returns) > 1 else 0.0
                autocorr_5 = returns.autocorr(lag=5) if len(returns) > 5 else 0.0
                
                # Efficiency score (lower autocorrelation = more efficient)
                efficiency_score = 1 - abs(autocorr_1 if not np.isnan(autocorr_1) else 0)
                
                features.extend([
                    max(0.0, min(1.0, efficiency_score)),
                    max(-1.0, min(1.0, autocorr_5 if not np.isnan(autocorr_5) else 0))
                ])
            else:
                features.extend([0.5, 0.0])  # Neutral efficiency
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting tick approximation features: {e}")
            return [0.0] * 15
    
    def _extract_liquidity_features(self, df: pd.DataFrame) -> List[float]:
        """Extract liquidity and market depth features."""
        try:
            if len(df) < self.liquidity_window:
                return [0.0] * 8
            
            features = []
            recent_df = df.tail(self.liquidity_window)
            
            # 1. Volume-based liquidity proxy
            volumes = recent_df['volume'].values
            returns = recent_df['close'].pct_change().abs().values[1:]
            
            # Amihud illiquidity measure approximation
            illiquidity_measures = []
            for i in range(len(returns)):
                if volumes[i + 1] > 0:
                    illiquidity = returns[i] / volumes[i + 1]  # Return per unit volume
                    illiquidity_measures.append(illiquidity)
            
            if illiquidity_measures:
                avg_illiquidity = np.mean(illiquidity_measures)
                features.append(min(avg_illiquidity * 1e6, 10.0))  # Scaled and capped
            else:
                features.append(5.0)  # Neutral liquidity
            
            # 2. Market depth approximation using volume concentration
            volume_concentration = np.std(volumes) / (np.mean(volumes) + 1e-8)
            features.append(min(volume_concentration, 5.0))
            
            # 3. Liquidity ratio (volume / volatility)
            avg_volume = np.mean(volumes)
            price_volatility = recent_df['close'].pct_change().std()
            
            if price_volatility > 0:
                liquidity_ratio = avg_volume / (price_volatility * 1e6)  # Scale appropriately
                features.append(min(liquidity_ratio, 10.0))
            else:
                features.append(10.0)  # High liquidity when no volatility
            
            # 4. Volume persistence
            volume_changes = pd.Series(volumes).pct_change().dropna()
            if len(volume_changes) > 5:
                volume_autocorr = volume_changes.autocorr(lag=1)
                features.append(max(-1.0, min(1.0, volume_autocorr if not np.isnan(volume_autocorr) else 0)))
            else:
                features.append(0.0)
            
            # 5. Turnover approximation
            # Using volume relative to price changes as proxy
            price_changes = recent_df['close'].pct_change().abs().values[1:]
            turnover_proxies = []
            
            for i, pc in enumerate(price_changes):
                if pc > 0:
                    turnover_proxy = volumes[i + 1] * pc  # Volume * price change
                    turnover_proxies.append(turnover_proxy)
            
            if turnover_proxies:
                avg_turnover_proxy = np.mean(turnover_proxies)
                features.append(np.log(avg_turnover_proxy + 1) / 20)  # Log-normalized
            else:
                features.append(0.0)
            
            # 6. Market impact resilience
            # How quickly does price impact decay?
            impact_decay = self._calculate_impact_decay(recent_df)
            features.append(max(0.0, min(1.0, impact_decay)))
            
            # 7. Liquidity risk (volatility of liquidity)
            if len(illiquidity_measures) > 5:
                liquidity_volatility = np.std(illiquidity_measures)
                normalized_liq_vol = liquidity_volatility / (avg_illiquidity + 1e-8)
                features.append(min(normalized_liq_vol, 5.0))
            else:
                features.append(1.0)
            
            # 8. Depth imbalance approximation
            # Using volume patterns around close prices
            depth_imbalance = self._calculate_depth_imbalance(recent_df)
            features.append(max(-1.0, min(1.0, depth_imbalance)))
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting liquidity features: {e}")
            return [0.0] * 8
    
    def _calculate_impact_decay(self, df: pd.DataFrame) -> float:
        """Calculate price impact decay rate."""
        try:
            if len(df) < 10:
                return 0.5
            
            returns = df['close'].pct_change().abs().values[1:]
            volumes = df['volume'].values[1:]
            
            # Look for mean reversion patterns after volume spikes
            volume_threshold = np.percentile(volumes, 80)  # Top 20% volume
            
            decay_rates = []
            for i in range(len(volumes) - 5):
                if volumes[i] >= volume_threshold:
                    # Measure how price impact decays over next few periods
                    initial_impact = returns[i]
                    subsequent_returns = returns[i + 1:i + 4]  # Next 3 periods
                    
                    if initial_impact > 0:
                        # Calculate decay rate
                        avg_subsequent = np.mean(subsequent_returns)
                        decay_rate = 1 - (avg_subsequent / initial_impact) if initial_impact > 0 else 0
                        decay_rates.append(max(0, min(1, decay_rate)))
            
            return np.mean(decay_rates) if decay_rates else 0.5
            
        except Exception as e:
            logger.debug(f"Error calculating impact decay: {e}")
            return 0.5
    
    def _calculate_depth_imbalance(self, df: pd.DataFrame) -> float:
        """Calculate market depth imbalance approximation."""
        try:
            if len(df) < 5:
                return 0.0
            
            # Approximate depth imbalance using OHLC patterns
            imbalances = []
            
            for _, row in df.iterrows():
                open_price = row.get('open', row['close'])
                high = row.get('high', row['close'])
                low = row.get('low', row['close'])
                close = row['close']
                volume = row.get('volume', 0)
                
                if high > low and volume > 0:
                    # Calculate where most volume likely occurred
                    upper_wick = high - max(open_price, close)
                    lower_wick = min(open_price, close) - low
                    body_size = abs(close - open_price)
                    
                    # Approximate imbalance based on wick sizes and volume
                    if upper_wick + lower_wick > 0:
                        imbalance = (upper_wick - lower_wick) / (upper_wick + lower_wick)
                        # Weight by volume
                        weighted_imbalance = imbalance * min(volume / 1e6, 1.0)
                        imbalances.append(weighted_imbalance)
            
            return np.mean(imbalances) if imbalances else 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating depth imbalance: {e}")
            return 0.0
    
    def _extract_order_flow_features(self, df: pd.DataFrame) -> List[float]:
        """Extract order flow and trade classification features."""
        try:
            if len(df) < 10:
                return [0.0] * 6
            
            features = []
            recent_df = df.tail(20)
            
            # 1. Buy/sell pressure using tick rule approximation
            buy_volume = 0
            sell_volume = 0
            
            for i in range(1, len(recent_df)):
                curr_close = recent_df['close'].iloc[i]
                prev_close = recent_df['close'].iloc[i - 1]
                volume = recent_df['volume'].iloc[i]
                
                if curr_close > prev_close:
                    buy_volume += volume
                elif curr_close < prev_close:
                    sell_volume += volume
                else:
                    # No change - split volume
                    buy_volume += volume * 0.5
                    sell_volume += volume * 0.5
            
            total_volume = buy_volume + sell_volume
            if total_volume > 0:
                order_flow_imbalance = (buy_volume - sell_volume) / total_volume
                features.append(max(-1.0, min(1.0, order_flow_imbalance)))
            else:
                features.append(0.0)
            
            # 2. Volume-weighted order flow
            vwap_deviations = []
            for i in range(5, len(recent_df)):
                # Simple VWAP calculation
                period_data = recent_df.iloc[i - 4:i + 1]
                vwap = (period_data['close'] * period_data['volume']).sum() / period_data['volume'].sum()
                current_price = recent_df['close'].iloc[i]
                deviation = (current_price - vwap) / vwap if vwap > 0 else 0
                vwap_deviations.append(deviation)
            
            if vwap_deviations:
                avg_vwap_deviation = np.mean(vwap_deviations)
                features.append(max(-0.1, min(0.1, avg_vwap_deviation)) * 10)  # Scale to [-1, 1]
            else:
                features.append(0.0)
            
            # 3. Trade size clustering
            # Approximate different trade sizes based on volume patterns
            volumes = recent_df['volume'].values
            volume_changes = np.diff(volumes)
            
            # Categorize volume changes
            large_trades = np.sum(volume_changes > np.percentile(volume_changes, 90))
            small_trades = np.sum(volume_changes < np.percentile(volume_changes, 10))
            total_trades = len(volume_changes)
            
            if total_trades > 0:
                large_trade_ratio = large_trades / total_trades
                small_trade_ratio = small_trades / total_trades
                features.extend([large_trade_ratio, small_trade_ratio])
            else:
                features.extend([0.0, 0.0])
            
            # 4. Order persistence
            # How long do order flow imbalances persist?
            price_changes = recent_df['close'].pct_change().values[1:]
            persistence_score = 0
            
            for i in range(len(price_changes) - 1):
                if price_changes[i] * price_changes[i + 1] > 0:  # Same direction
                    persistence_score += 1
            
            if len(price_changes) > 1:
                persistence_ratio = persistence_score / (len(price_changes) - 1)
                features.append(persistence_ratio)
            else:
                features.append(0.5)
            
            # 5. Informed trading approximation
            # PIN (Probability of Informed Trading) approximation
            pin_approximation = self._calculate_pin_approximation(recent_df)
            features.append(max(0.0, min(1.0, pin_approximation)))
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting order flow features: {e}")
            return [0.0] * 6
    
    def _calculate_pin_approximation(self, df: pd.DataFrame) -> float:
        """Approximate PIN (Probability of Informed Trading)."""
        try:
            if len(df) < 10:
                return 0.2  # Default PIN
            
            # Approximate PIN using volume and price movement patterns
            returns = df['close'].pct_change().values[1:]
            volumes = df['volume'].values[1:]
            
            # Count days with significant price movements and volume
            significant_moves = 0
            high_volume_moves = 0
            
            volume_threshold = np.percentile(volumes, 70)  # Top 30% volume
            return_threshold = np.percentile(np.abs(returns), 70)  # Top 30% returns
            
            for i in range(len(returns)):
                if abs(returns[i]) > return_threshold:
                    significant_moves += 1
                    if volumes[i] > volume_threshold:
                        high_volume_moves += 1
            
            if significant_moves > 0:
                # PIN approximation: ratio of high-volume significant moves
                pin_approx = high_volume_moves / significant_moves
                return min(pin_approx, 1.0)
            
            return 0.2  # Default
            
        except Exception as e:
            logger.debug(f"Error calculating PIN approximation: {e}")
            return 0.2
    
    def _extract_price_discovery_features(self, df: pd.DataFrame) -> List[float]:
        """Extract price discovery and information incorporation features."""
        try:
            if len(df) < 20:
                return [0.0] * 5
            
            features = []
            recent_df = df.tail(20)
            
            # 1. Price discovery efficiency
            # Based on how quickly prices incorporate information
            returns = recent_df['close'].pct_change().values[1:]
            
            # Variance ratio test approximation
            # If prices are efficient, variance should scale linearly with time
            if len(returns) >= 10:
                var_1day = np.var(returns)
                var_5day = np.var(pd.Series(returns).rolling(5).sum().dropna().values)
                
                if var_1day > 0:
                    variance_ratio = var_5day / (5 * var_1day)
                    # Efficiency score: closer to 1 is more efficient
                    efficiency_score = 1 - abs(variance_ratio - 1)
                    features.append(max(0.0, min(1.0, efficiency_score)))
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
            
            # 2. Information asymmetry proxy
            # Based on bid-ask spread approximation and volume
            spreads = []
            for _, row in recent_df.iterrows():
                high = row.get('high', row['close'])
                low = row.get('low', row['close'])
                close = row['close']
                if close > 0:
                    spread = (high - low) / close
                    spreads.append(spread)
            
            if spreads:
                avg_spread = np.mean(spreads)
                # Higher spreads suggest more information asymmetry
                asymmetry_proxy = min(avg_spread * 10, 1.0)  # Scale and cap
                features.append(asymmetry_proxy)
            else:
                features.append(0.5)
            
            # 3. News impact proxy
            # Based on volume spikes and price jumps
            volumes = recent_df['volume'].values
            price_jumps = np.abs(returns)
            
            # Identify potential news days (high volume + high returns)
            volume_threshold = np.percentile(volumes[1:], 80)
            return_threshold = np.percentile(price_jumps, 80)
            
            news_days = 0
            for i in range(len(price_jumps)):
                if volumes[i + 1] > volume_threshold and price_jumps[i] > return_threshold:
                    news_days += 1
            
            news_frequency = news_days / len(price_jumps) if len(price_jumps) > 0 else 0
            features.append(min(news_frequency * 5, 1.0))  # Scale and cap
            
            # 4. Market reaction speed
            # How quickly does price react to volume changes?
            reaction_speeds = []
            for i in range(1, len(volumes) - 1):
                volume_change = (volumes[i] - volumes[i - 1]) / (volumes[i - 1] + 1e-8)
                price_reaction = abs(returns[i])
                
                if abs(volume_change) > 0.1:  # Significant volume change
                    reaction_speed = price_reaction / (abs(volume_change) + 1e-8)
                    reaction_speeds.append(min(reaction_speed, 10.0))
            
            if reaction_speeds:
                avg_reaction_speed = np.mean(reaction_speeds)
                normalized_speed = min(avg_reaction_speed / 5.0, 1.0)
                features.append(normalized_speed)
            else:
                features.append(0.5)
            
            # 5. Price impact permanence
            # How much of price impact is permanent vs temporary?
            impact_permanence = self._calculate_impact_permanence(recent_df)
            features.append(max(0.0, min(1.0, impact_permanence)))
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting price discovery features: {e}")
            return [0.0] * 5
    
    def _calculate_impact_permanence(self, df: pd.DataFrame) -> float:
        """Calculate the permanence of price impact."""
        try:
            if len(df) < 15:
                return 0.5
            
            returns = df['close'].pct_change().values[1:]
            volumes = df['volume'].values[1:]
            
            # Find large volume events
            volume_threshold = np.percentile(volumes, 85)
            
            permanence_scores = []
            for i in range(len(volumes) - 10):
                if volumes[i] > volume_threshold:
                    immediate_impact = abs(returns[i])
                    
                    # Look at price 5 periods later
                    if i + 5 < len(returns):
                        future_position = abs(sum(returns[i + 1:i + 6]))
                        
                        # Permanence = how much of impact remains
                        if immediate_impact > 0:
                            permanence = future_position / immediate_impact
                            permanence_scores.append(min(permanence, 2.0))
            
            return np.mean(permanence_scores) if permanence_scores else 0.5
            
        except Exception as e:
            logger.debug(f"Error calculating impact permanence: {e}")
            return 0.5
    
    def _extract_regime_features(self, df: pd.DataFrame) -> List[float]:
        """Extract microstructure regime features."""
        try:
            if len(df) < 20:
                return [0.0] * 4
            
            # Detect current microstructure regime
            regime = self._detect_microstructure_regime(df)
            self.regime_history.append(regime)
            
            features = []
            
            # Regime indicator (one-hot style)
            regime_indicators = [0.0] * len(MicrostructureRegime)
            regime_index = list(MicrostructureRegime).index(regime)
            regime_indicators[regime_index] = 1.0
            
            # Take first 3 regime indicators to limit feature count
            features.extend(regime_indicators[:3])
            
            # Regime stability (how often regime changes)
            if len(self.regime_history) >= 10:
                regime_changes = sum(1 for i in range(1, len(self.regime_history)) 
                                   if self.regime_history[i] != self.regime_history[i - 1])
                stability = 1 - (regime_changes / len(self.regime_history))
                features.append(stability)
            else:
                features.append(0.5)
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting regime features: {e}")
            return [0.0] * 4
    
    def _detect_microstructure_regime(self, df: pd.DataFrame) -> MicrostructureRegime:
        """Detect current microstructure regime."""
        try:
            if len(df) < 10:
                return MicrostructureRegime.CONSOLIDATION
            
            recent_df = df.tail(10)
            
            # Calculate regime indicators
            volumes = recent_df['volume'].values
            returns = recent_df['close'].pct_change().abs().values[1:]
            
            avg_volume = np.mean(volumes)
            volume_std = np.std(volumes)
            avg_return = np.mean(returns)
            return_std = np.std(returns)
            
            # Regime classification logic
            volume_cv = volume_std / (avg_volume + 1e-8)
            return_cv = return_std / (avg_return + 1e-8)
            
            # High volume + low volatility = high liquidity
            if avg_volume > np.percentile(volumes, 70) and avg_return < np.percentile(returns, 30):
                return MicrostructureRegime.HIGH_LIQUIDITY
            
            # Low volume + high volatility = low liquidity
            elif avg_volume < np.percentile(volumes, 30) and avg_return > np.percentile(returns, 70):
                return MicrostructureRegime.LOW_LIQUIDITY
            
            # High volume + high volatility = price discovery
            elif avg_volume > np.percentile(volumes, 60) and avg_return > np.percentile(returns, 60):
                return MicrostructureRegime.PRICE_DISCOVERY
            
            # High return volatility = information driven
            elif return_cv > 2.0:
                return MicrostructureRegime.INFORMATION_DRIVEN
            
            # High volume coefficient of variation = breakout
            elif volume_cv > 1.5:
                return MicrostructureRegime.BREAKOUT
            
            else:
                return MicrostructureRegime.CONSOLIDATION
                
        except Exception as e:
            logger.debug(f"Error detecting microstructure regime: {e}")
            return MicrostructureRegime.CONSOLIDATION
    
    def _get_default_features(self) -> np.ndarray:
        """Get default features when extraction fails."""
        return np.zeros(40, dtype=np.float32)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached features are still valid."""
        if cache_key not in self.feature_cache:
            return False
        
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[cache_key]
        elapsed_minutes = (datetime.now() - cache_time).seconds / 60
        
        return elapsed_minutes < self.cache_ttl_minutes
    
    def _cache_features(self, cache_key: str, features: np.ndarray) -> None:
        """Cache calculated features."""
        self.feature_cache[cache_key] = features
        self.cache_timestamps[cache_key] = datetime.now()
        
        # Clean old cache entries
        self._clean_cache()
    
    def _clean_cache(self) -> None:
        """Remove expired cache entries."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            elapsed_minutes = (current_time - timestamp).seconds / 60
            if elapsed_minutes >= self.cache_ttl_minutes:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.feature_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all microstructure features."""
        names = []
        
        # Volume profile features
        for period in self.lookback_periods:
            names.extend([
                f"vp_{period}_poc_distance",
                f"vp_{period}_vah_distance", 
                f"vp_{period}_val_distance",
                f"vp_{period}_concentration",
                f"vp_{period}_balance",
                f"vp_{period}_shape"
            ])
        
        # Tick approximation features
        names.extend([
            "tick_count_avg", "tick_volatility", "price_impact",
            "spread_avg", "spread_volatility", "volume_rate_change",
            "rate_volatility", "trade_size_avg", "trade_size_cv",
            "large_trade_ratio", "info_flow_avg", "info_flow_volatility",
            "efficiency_score", "autocorr_lag5"
        ])
        
        # Liquidity features
        names.extend([
            "illiquidity", "volume_concentration", "liquidity_ratio",
            "volume_persistence", "turnover_proxy", "impact_decay",
            "liquidity_volatility", "depth_imbalance"
        ])
        
        # Order flow features
        names.extend([
            "order_flow_imbalance", "vwap_deviation", "large_trade_ratio_of",
            "small_trade_ratio_of", "order_persistence", "pin_approximation"
        ])
        
        # Price discovery features
        names.extend([
            "price_discovery_efficiency", "information_asymmetry",
            "news_impact_frequency", "market_reaction_speed", "impact_permanence"
        ])
        
        # Regime features
        names.extend([
            "regime_high_liquidity", "regime_low_liquidity", "regime_price_discovery",
            "regime_stability"
        ])
        
        return names
    
    def get_extractor_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the microstructure extractor."""
        return {
            'volume_profile_bins': self.volume_profile_bins,
            'lookback_periods': self.lookback_periods,
            'cache_size': len(self.feature_cache),
            'regime_history_length': len(self.regime_history),
            'current_regime': self.regime_history[-1].value if self.regime_history else 'unknown',
            'settings': {
                'tick_approximation_method': self.tick_approximation_method,
                'enable_regime_detection': self.enable_regime_detection,
                'liquidity_window': self.liquidity_window,
                'cache_ttl_minutes': self.cache_ttl_minutes
            }
        }