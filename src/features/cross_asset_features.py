"""Cross-asset features with correlation and relative strength indicators."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from datetime import datetime, timedelta
import warnings
from loguru import logger
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import itertools

from ..config.settings import config


class CrossAssetRegime(Enum):
    """Cross-asset regime types."""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    ROTATION = "rotation"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    CONVERGENCE = "convergence"
    DIVERGENCE = "divergence"


class AssetClass(Enum):
    """Asset class types for classification."""
    EQUITY = "equity"
    BOND = "bond"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTO = "crypto"
    REIT = "reit"
    UNKNOWN = "unknown"


@dataclass
class CrossAssetMetrics:
    """Cross-asset relationship metrics."""
    correlation: float
    beta: float
    relative_strength: float
    co_movement: float
    lead_lag: int
    volatility_spillover: float


@dataclass
class SectorRotationSignal:
    """Sector rotation analysis results."""
    rotation_strength: float
    leading_sectors: List[str]
    lagging_sectors: List[str]
    rotation_direction: str  # "growth_to_value", "value_to_growth", "defensive", "cyclical"


class CrossAssetFeatureExtractor:
    """
    Advanced cross-asset feature extractor.
    
    Features:
    - Cross-asset correlations and covariances
    - Relative strength analysis
    - Sector rotation detection
    - Risk-on/risk-off regime identification
    - Lead-lag relationships
    - Volatility spillovers
    - Factor exposures and style analysis
    """
    
    def __init__(
        self,
        correlation_windows: List[int] = [10, 20, 60],
        max_assets: int = 20,
        min_overlap_periods: int = 10,
        enable_sector_rotation: bool = True,
        enable_factor_analysis: bool = True,
        volatility_window: int = 20
    ):
        """
        Initialize cross-asset feature extractor.
        
        Args:
            correlation_windows: Different windows for correlation calculation
            max_assets: Maximum number of assets to analyze
            min_overlap_periods: Minimum overlapping periods required
            enable_sector_rotation: Enable sector rotation analysis
            enable_factor_analysis: Enable factor exposure analysis
            volatility_window: Window for volatility calculations
        """
        self.correlation_windows = correlation_windows
        self.max_assets = max_assets
        self.min_overlap_periods = min_overlap_periods
        self.enable_sector_rotation = enable_sector_rotation
        self.enable_factor_analysis = enable_factor_analysis
        self.volatility_window = volatility_window
        
        # Feature caching
        self.feature_cache = {}
        self.cache_timestamps = {}
        self.cache_ttl_minutes = 15
        
        # Cross-asset relationship tracking
        self.correlation_history = defaultdict(lambda: deque(maxlen=100))
        self.relative_strength_history = defaultdict(lambda: deque(maxlen=100))
        
        # Asset classification (simplified heuristics)
        self.asset_class_mapping = self._initialize_asset_classification()
        
        # Factor proxies (simplified)
        self.factor_proxies = {
            'market': ['SPY', 'VTI', 'AAPL', 'MSFT'],
            'value': ['VTV', 'IWD', 'MTUM'],
            'growth': ['VUG', 'IWF', 'ARKK'],
            'momentum': ['MTUM', 'PDP'],
            'quality': ['QUAL', 'USMV'],
            'size': ['IWM', 'VB', 'IJR'],
            'volatility': ['VIX', 'UVXY', 'SVXY']
        }
        
        # Regime detection
        self.regime_history = deque(maxlen=50)
        
        logger.info("CrossAssetFeatureExtractor initialized")
    
    def _initialize_asset_classification(self) -> Dict[str, AssetClass]:
        """Initialize basic asset class mapping."""
        return {
            # Major equities
            'SPY': AssetClass.EQUITY, 'QQQ': AssetClass.EQUITY, 'IWM': AssetClass.EQUITY,
            'AAPL': AssetClass.EQUITY, 'MSFT': AssetClass.EQUITY, 'GOOGL': AssetClass.EQUITY,
            'AMZN': AssetClass.EQUITY, 'TSLA': AssetClass.EQUITY, 'NVDA': AssetClass.EQUITY,
            
            # Bonds
            'TLT': AssetClass.BOND, 'IEF': AssetClass.BOND, 'SHY': AssetClass.BOND,
            'HYG': AssetClass.BOND, 'LQD': AssetClass.BOND, 'TIP': AssetClass.BOND,
            
            # Commodities
            'GLD': AssetClass.COMMODITY, 'SLV': AssetClass.COMMODITY, 'USO': AssetClass.COMMODITY,
            'UNG': AssetClass.COMMODITY, 'DBA': AssetClass.COMMODITY, 'JJC': AssetClass.COMMODITY,
            
            # Currency
            'UUP': AssetClass.CURRENCY, 'FXE': AssetClass.CURRENCY, 'FXY': AssetClass.CURRENCY,
            
            # REITs
            'VNQ': AssetClass.REIT, 'IYR': AssetClass.REIT, 'SCHH': AssetClass.REIT,
            
            # Crypto (if available)
            'BTC': AssetClass.CRYPTO, 'ETH': AssetClass.CRYPTO
        }
    
    def extract_features(
        self,
        multi_asset_data: Dict[str, pd.DataFrame],
        primary_ticker: str,
        reference_tickers: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract comprehensive cross-asset features.
        
        Args:
            multi_asset_data: Dictionary of asset data
            primary_ticker: Primary asset to analyze
            reference_tickers: Specific reference assets (if None, uses all available)
            
        Returns:
            Array of cross-asset features
        """
        try:
            if primary_ticker not in multi_asset_data:
                logger.warning(f"Primary ticker {primary_ticker} not found in data")
                return self._get_default_features()
            
            # Limit number of assets for performance
            available_tickers = list(multi_asset_data.keys())
            if reference_tickers:
                reference_tickers = [t for t in reference_tickers if t in available_tickers]
            else:
                reference_tickers = [t for t in available_tickers if t != primary_ticker][:self.max_assets - 1]
            
            if not reference_tickers:
                logger.warning("No reference tickers available for cross-asset analysis")
                return self._get_default_features()
            
            # Check cache
            cache_key = f"{primary_ticker}_{'_'.join(sorted(reference_tickers))}_{len(multi_asset_data[primary_ticker])}"
            if self._is_cache_valid(cache_key):
                logger.debug(f"Using cached cross-asset features for {primary_ticker}")
                return self.feature_cache[cache_key]
            
            # Extract different feature groups
            all_features = []
            
            # Correlation features
            correlation_features = self._extract_correlation_features(
                multi_asset_data, primary_ticker, reference_tickers
            )
            all_features.extend(correlation_features)
            
            # Relative strength features
            relative_strength_features = self._extract_relative_strength_features(
                multi_asset_data, primary_ticker, reference_tickers
            )
            all_features.extend(relative_strength_features)
            
            # Co-movement and lead-lag features
            comovement_features = self._extract_comovement_features(
                multi_asset_data, primary_ticker, reference_tickers
            )
            all_features.extend(comovement_features)
            
            # Volatility spillover features
            spillover_features = self._extract_volatility_spillover_features(
                multi_asset_data, primary_ticker, reference_tickers
            )
            all_features.extend(spillover_features)
            
            # Risk regime features
            risk_regime_features = self._extract_risk_regime_features(
                multi_asset_data, primary_ticker, reference_tickers
            )
            all_features.extend(risk_regime_features)
            
            # Sector rotation features (if enabled)
            if self.enable_sector_rotation:
                sector_features = self._extract_sector_rotation_features(
                    multi_asset_data, primary_ticker, reference_tickers
                )
                all_features.extend(sector_features)
            
            # Factor exposure features (if enabled)
            if self.enable_factor_analysis:
                factor_features = self._extract_factor_exposure_features(
                    multi_asset_data, primary_ticker, reference_tickers
                )
                all_features.extend(factor_features)
            
            # Convert to numpy array and clean
            features_array = np.array(all_features, dtype=np.float32)
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
            features_array = np.clip(features_array, -5.0, 5.0)
            
            # Ensure minimum feature count
            min_features = 50
            if len(features_array) < min_features:
                features_array = np.pad(features_array, (0, min_features - len(features_array)))
            
            # Cache result
            self._cache_features(cache_key, features_array)
            
            logger.debug(f"Generated {len(features_array)} cross-asset features for {primary_ticker}")
            
            return features_array
            
        except Exception as e:
            logger.error(f"Error extracting cross-asset features for {primary_ticker}: {e}")
            return self._get_default_features()
    
    def _extract_correlation_features(
        self, 
        multi_asset_data: Dict[str, pd.DataFrame],
        primary_ticker: str,
        reference_tickers: List[str]
    ) -> List[float]:
        """Extract correlation-based features."""
        try:
            features = []
            primary_data = multi_asset_data[primary_ticker]
            
            for window in self.correlation_windows:
                if len(primary_data) < window:
                    features.extend([0.0] * 8)  # Default features per window
                    continue
                
                # Get aligned returns for correlation calculation
                aligned_returns = self._align_returns_data(
                    multi_asset_data, [primary_ticker] + reference_tickers, window
                )
                
                if aligned_returns.empty or len(aligned_returns) < self.min_overlap_periods:
                    features.extend([0.0] * 8)
                    continue
                
                primary_returns = aligned_returns[primary_ticker]
                
                # Calculate correlations with all reference assets
                correlations = []
                for ref_ticker in reference_tickers:
                    if ref_ticker in aligned_returns.columns:
                        corr = aligned_returns[primary_ticker].corr(aligned_returns[ref_ticker])
                        if not np.isnan(corr):
                            correlations.append(corr)
                
                if correlations:
                    # Correlation statistics
                    mean_corr = np.mean(correlations)
                    std_corr = np.std(correlations)
                    max_corr = np.max(correlations)
                    min_corr = np.min(correlations)
                    
                    # Correlation distribution features
                    positive_corr_ratio = len([c for c in correlations if c > 0.1]) / len(correlations)
                    high_corr_ratio = len([c for c in correlations if abs(c) > 0.5]) / len(correlations)
                    
                    # Correlation stability (how much correlations changed)
                    corr_stability = self._calculate_correlation_stability(
                        aligned_returns, primary_ticker, reference_tickers
                    )
                    
                    # Average absolute correlation
                    avg_abs_corr = np.mean([abs(c) for c in correlations])
                    
                    features.extend([
                        mean_corr, std_corr, max_corr, min_corr,
                        positive_corr_ratio, high_corr_ratio, corr_stability, avg_abs_corr
                    ])
                else:
                    features.extend([0.0] * 8)
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting correlation features: {e}")
            return [0.0] * (len(self.correlation_windows) * 8)
    
    def _align_returns_data(
        self,
        multi_asset_data: Dict[str, pd.DataFrame],
        tickers: List[str],
        window: int
    ) -> pd.DataFrame:
        """Align returns data across multiple assets."""
        try:
            returns_data = {}
            
            for ticker in tickers:
                if ticker in multi_asset_data:
                    df = multi_asset_data[ticker]
                    if len(df) >= window:
                        returns = df['close'].pct_change().dropna().tail(window - 1)
                        if len(returns) > 0:
                            returns_data[ticker] = returns
            
            if not returns_data:
                return pd.DataFrame()
            
            # Align all series to common index
            aligned_df = pd.DataFrame(returns_data)
            aligned_df = aligned_df.dropna()  # Keep only periods where all assets have data
            
            return aligned_df
            
        except Exception as e:
            logger.debug(f"Error aligning returns data: {e}")
            return pd.DataFrame()
    
    def _calculate_correlation_stability(
        self,
        aligned_returns: pd.DataFrame,
        primary_ticker: str,
        reference_tickers: List[str]
    ) -> float:
        """Calculate correlation stability over time."""
        try:
            if len(aligned_returns) < 20:
                return 0.5  # Neutral stability
            
            # Split data into two halves
            mid_point = len(aligned_returns) // 2
            first_half = aligned_returns.iloc[:mid_point]
            second_half = aligned_returns.iloc[mid_point:]
            
            if len(first_half) < 5 or len(second_half) < 5:
                return 0.5
            
            # Calculate correlations for each half
            first_half_corrs = []
            second_half_corrs = []
            
            for ref_ticker in reference_tickers:
                if ref_ticker in aligned_returns.columns:
                    corr1 = first_half[primary_ticker].corr(first_half[ref_ticker])
                    corr2 = second_half[primary_ticker].corr(second_half[ref_ticker])
                    
                    if not (np.isnan(corr1) or np.isnan(corr2)):
                        first_half_corrs.append(corr1)
                        second_half_corrs.append(corr2)
            
            if first_half_corrs and second_half_corrs:
                # Stability = 1 - average absolute change in correlations
                corr_changes = [abs(c1 - c2) for c1, c2 in zip(first_half_corrs, second_half_corrs)]
                avg_change = np.mean(corr_changes)
                stability = max(0.0, 1.0 - avg_change)
                return stability
            
            return 0.5
            
        except Exception as e:
            logger.debug(f"Error calculating correlation stability: {e}")
            return 0.5
    
    def _extract_relative_strength_features(
        self,
        multi_asset_data: Dict[str, pd.DataFrame],
        primary_ticker: str,
        reference_tickers: List[str]
    ) -> List[float]:
        """Extract relative strength features."""
        try:
            features = []
            primary_data = multi_asset_data[primary_ticker]
            
            # Different time horizons for relative strength
            horizons = [5, 20, 60]
            
            for horizon in horizons:
                if len(primary_data) < horizon + 1:
                    features.extend([0.0] * 6)
                    continue
                
                # Calculate primary asset return
                primary_return = (
                    primary_data['close'].iloc[-1] - primary_data['close'].iloc[-horizon - 1]
                ) / primary_data['close'].iloc[-horizon - 1]
                
                # Calculate reference returns
                reference_returns = []
                for ref_ticker in reference_tickers:
                    if ref_ticker in multi_asset_data:
                        ref_data = multi_asset_data[ref_ticker]
                        if len(ref_data) >= horizon + 1:
                            ref_return = (
                                ref_data['close'].iloc[-1] - ref_data['close'].iloc[-horizon - 1]
                            ) / ref_data['close'].iloc[-horizon - 1]
                            reference_returns.append(ref_return)
                
                if reference_returns:
                    # Relative strength metrics
                    mean_ref_return = np.mean(reference_returns)
                    relative_strength = primary_return - mean_ref_return
                    
                    # Ranking (percentile of primary return vs references)
                    all_returns = reference_returns + [primary_return]
                    rank_percentile = (sorted(all_returns).index(primary_return) + 1) / len(all_returns)
                    
                    # Beat rate (percentage of reference assets outperformed)
                    beat_count = sum(1 for ret in reference_returns if primary_return > ret)
                    beat_rate = beat_count / len(reference_returns)
                    
                    # Relative volatility
                    primary_vol = primary_data['close'].pct_change().tail(horizon).std()
                    ref_vols = []
                    for ref_ticker in reference_tickers:
                        if ref_ticker in multi_asset_data:
                            ref_data = multi_asset_data[ref_ticker]
                            if len(ref_data) >= horizon:
                                ref_vol = ref_data['close'].pct_change().tail(horizon).std()
                                if not np.isnan(ref_vol):
                                    ref_vols.append(ref_vol)
                    
                    if ref_vols:
                        mean_ref_vol = np.mean(ref_vols)
                        relative_volatility = (primary_vol - mean_ref_vol) / (mean_ref_vol + 1e-8)
                    else:
                        relative_volatility = 0.0
                    
                    # Risk-adjusted relative strength (Sharpe-like)
                    if primary_vol > 0:
                        risk_adj_rel_strength = relative_strength / primary_vol
                    else:
                        risk_adj_rel_strength = 0.0
                    
                    # Momentum consistency (how often primary outperforms)
                    consistency = self._calculate_momentum_consistency(
                        multi_asset_data, primary_ticker, reference_tickers, horizon
                    )
                    
                    features.extend([
                        max(-1.0, min(1.0, relative_strength)),         # Bounded relative strength
                        rank_percentile,                                # Rank percentile
                        beat_rate,                                      # Beat rate
                        max(-2.0, min(2.0, relative_volatility)),       # Bounded relative volatility
                        max(-5.0, min(5.0, risk_adj_rel_strength)),     # Risk-adjusted strength
                        consistency                                      # Momentum consistency
                    ])
                else:
                    features.extend([0.0] * 6)
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting relative strength features: {e}")
            return [0.0] * (len([5, 20, 60]) * 6)
    
    def _calculate_momentum_consistency(
        self,
        multi_asset_data: Dict[str, pd.DataFrame],
        primary_ticker: str,
        reference_tickers: List[str],
        lookback: int
    ) -> float:
        """Calculate momentum consistency over time."""
        try:
            if lookback < 10:
                return 0.5
            
            primary_data = multi_asset_data[primary_ticker]
            if len(primary_data) < lookback:
                return 0.5
            
            # Check momentum consistency over sub-periods
            sub_period = max(5, lookback // 4)
            consistent_periods = 0
            total_periods = 0
            
            for i in range(sub_period, lookback, sub_period):
                end_idx = len(primary_data) - lookback + i
                start_idx = end_idx - sub_period
                
                if start_idx >= 0:
                    period_primary_return = (
                        primary_data['close'].iloc[end_idx] - primary_data['close'].iloc[start_idx]
                    ) / primary_data['close'].iloc[start_idx]
                    
                    period_ref_returns = []
                    for ref_ticker in reference_tickers[:5]:  # Limit for performance
                        if ref_ticker in multi_asset_data:
                            ref_data = multi_asset_data[ref_ticker]
                            if len(ref_data) > end_idx:
                                ref_return = (
                                    ref_data['close'].iloc[end_idx] - ref_data['close'].iloc[start_idx]
                                ) / ref_data['close'].iloc[start_idx]
                                period_ref_returns.append(ref_return)
                    
                    if period_ref_returns:
                        avg_ref_return = np.mean(period_ref_returns)
                        if period_primary_return > avg_ref_return:
                            consistent_periods += 1
                        total_periods += 1
            
            return consistent_periods / total_periods if total_periods > 0 else 0.5
            
        except Exception as e:
            logger.debug(f"Error calculating momentum consistency: {e}")
            return 0.5
    
    def _extract_comovement_features(
        self,
        multi_asset_data: Dict[str, pd.DataFrame],
        primary_ticker: str,
        reference_tickers: List[str]
    ) -> List[float]:
        """Extract co-movement and lead-lag features."""
        try:
            features = []
            
            # Get aligned returns
            aligned_returns = self._align_returns_data(
                multi_asset_data, [primary_ticker] + reference_tickers, 60
            )
            
            if aligned_returns.empty or len(aligned_returns) < 20:
                return [0.0] * 12
            
            primary_returns = aligned_returns[primary_ticker]
            
            # 1. Co-movement strength (average beta with other assets)
            betas = []
            for ref_ticker in reference_tickers:
                if ref_ticker in aligned_returns.columns:
                    ref_returns = aligned_returns[ref_ticker]
                    # Calculate beta (covariance / variance)
                    covariance = np.cov(primary_returns, ref_returns)[0, 1]
                    ref_variance = np.var(ref_returns)
                    if ref_variance > 0:
                        beta = covariance / ref_variance
                        betas.append(beta)
            
            if betas:
                avg_beta = np.mean(betas)
                beta_std = np.std(betas)
                features.extend([
                    max(-3.0, min(3.0, avg_beta)),    # Average beta (capped)
                    min(beta_std, 2.0)                # Beta dispersion (capped)
                ])
            else:
                features.extend([1.0, 0.0])
            
            # 2. Lead-lag relationships
            lead_lag_scores = []
            for ref_ticker in reference_tickers[:5]:  # Limit for performance
                if ref_ticker in aligned_returns.columns:
                    lead_lag = self._calculate_lead_lag(
                        primary_returns, aligned_returns[ref_ticker]
                    )
                    lead_lag_scores.append(lead_lag)
            
            if lead_lag_scores:
                avg_lead_lag = np.mean(lead_lag_scores)
                features.append(max(-3.0, min(3.0, avg_lead_lag)))
            else:
                features.append(0.0)
            
            # 3. Synchronization measure (how often assets move in same direction)
            sync_scores = []
            for ref_ticker in reference_tickers:
                if ref_ticker in aligned_returns.columns:
                    # Count periods where both assets moved in same direction
                    same_direction = (
                        (primary_returns > 0) & (aligned_returns[ref_ticker] > 0)
                    ) | (
                        (primary_returns < 0) & (aligned_returns[ref_ticker] < 0)
                    )
                    sync_score = same_direction.sum() / len(same_direction)
                    sync_scores.append(sync_score)
            
            if sync_scores:
                avg_synchronization = np.mean(sync_scores)
                features.append(avg_synchronization)
            else:
                features.append(0.5)
            
            # 4. Regime-dependent correlation
            regime_corr = self._calculate_regime_dependent_correlation(aligned_returns, primary_ticker)
            features.extend(regime_corr)  # Should return 4 values
            
            # 5. Cross-asset momentum (how primary asset moves after other assets move)
            cross_momentum = self._calculate_cross_asset_momentum(aligned_returns, primary_ticker)
            features.extend(cross_momentum)  # Should return 4 values
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting co-movement features: {e}")
            return [0.0] * 12
    
    def _calculate_lead_lag(self, primary_returns: pd.Series, ref_returns: pd.Series) -> float:
        """Calculate lead-lag relationship between two return series."""
        try:
            if len(primary_returns) < 10 or len(ref_returns) < 10:
                return 0.0
            
            # Calculate correlations at different lags
            max_lag = min(5, len(primary_returns) // 4)
            correlations = {}
            
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    corr = primary_returns.corr(ref_returns)
                elif lag > 0:
                    # Primary leads reference by lag periods
                    if len(primary_returns) > lag:
                        corr = primary_returns.iloc[:-lag].corr(ref_returns.iloc[lag:])
                else:
                    # Reference leads primary by |lag| periods
                    abs_lag = abs(lag)
                    if len(ref_returns) > abs_lag:
                        corr = primary_returns.iloc[abs_lag:].corr(ref_returns.iloc[:-abs_lag])
                
                if not np.isnan(corr):
                    correlations[lag] = abs(corr)  # Use absolute correlation
            
            if not correlations:
                return 0.0
            
            # Find lag with highest correlation
            best_lag = max(correlations.keys(), key=lambda k: correlations[k])
            
            # Normalize lead-lag score
            lead_lag_score = best_lag / max_lag if max_lag > 0 else 0
            
            return lead_lag_score
            
        except Exception as e:
            logger.debug(f"Error calculating lead-lag: {e}")
            return 0.0
    
    def _calculate_regime_dependent_correlation(
        self, 
        aligned_returns: pd.DataFrame, 
        primary_ticker: str
    ) -> List[float]:
        """Calculate correlations in different volatility regimes."""
        try:
            if len(aligned_returns) < 20:
                return [0.0, 0.0, 0.0, 0.0]
            
            primary_returns = aligned_returns[primary_ticker]
            
            # Define volatility regimes based on rolling volatility
            rolling_vol = primary_returns.rolling(window=10).std()
            vol_threshold_high = rolling_vol.quantile(0.7)
            vol_threshold_low = rolling_vol.quantile(0.3)
            
            high_vol_periods = rolling_vol > vol_threshold_high
            low_vol_periods = rolling_vol < vol_threshold_low
            
            # Calculate average correlations in each regime
            other_tickers = [col for col in aligned_returns.columns if col != primary_ticker]
            
            high_vol_corrs = []
            low_vol_corrs = []
            
            for ticker in other_tickers:
                if high_vol_periods.any():
                    high_vol_corr = primary_returns[high_vol_periods].corr(
                        aligned_returns[ticker][high_vol_periods]
                    )
                    if not np.isnan(high_vol_corr):
                        high_vol_corrs.append(high_vol_corr)
                
                if low_vol_periods.any():
                    low_vol_corr = primary_returns[low_vol_periods].corr(
                        aligned_returns[ticker][low_vol_periods]
                    )
                    if not np.isnan(low_vol_corr):
                        low_vol_corrs.append(low_vol_corr)
            
            # Features
            high_vol_avg_corr = np.mean(high_vol_corrs) if high_vol_corrs else 0.0
            low_vol_avg_corr = np.mean(low_vol_corrs) if low_vol_corrs else 0.0
            correlation_regime_diff = high_vol_avg_corr - low_vol_avg_corr
            correlation_regime_stability = 1 - abs(correlation_regime_diff)
            
            return [
                max(-1.0, min(1.0, high_vol_avg_corr)),
                max(-1.0, min(1.0, low_vol_avg_corr)),
                max(-2.0, min(2.0, correlation_regime_diff)),
                max(0.0, min(1.0, correlation_regime_stability))
            ]
            
        except Exception as e:
            logger.debug(f"Error calculating regime-dependent correlation: {e}")
            return [0.0, 0.0, 0.0, 0.0]
    
    def _calculate_cross_asset_momentum(
        self, 
        aligned_returns: pd.DataFrame, 
        primary_ticker: str
    ) -> List[float]:
        """Calculate cross-asset momentum effects."""
        try:
            if len(aligned_returns) < 15:
                return [0.0, 0.0, 0.0, 0.0]
            
            primary_returns = aligned_returns[primary_ticker]
            other_tickers = [col for col in aligned_returns.columns if col != primary_ticker]
            
            # Calculate momentum effects with different lags
            momentum_effects = []
            
            for lag in [1, 2, 3]:  # 1, 2, 3 day lags
                if len(aligned_returns) > lag:
                    # Average return of other assets lag periods ago
                    lagged_other_returns = aligned_returns[other_tickers].shift(lag).mean(axis=1)
                    
                    # Current primary returns
                    current_primary = primary_returns
                    
                    # Align data
                    valid_idx = (~lagged_other_returns.isna()) & (~current_primary.isna())
                    
                    if valid_idx.sum() > 10:
                        # Calculate correlation between lagged other returns and current primary
                        momentum_corr = lagged_other_returns[valid_idx].corr(
                            current_primary[valid_idx]
                        )
                        
                        if not np.isnan(momentum_corr):
                            momentum_effects.append(momentum_corr)
            
            # Momentum features
            if momentum_effects:
                avg_momentum_effect = np.mean(momentum_effects)
                momentum_consistency = len([m for m in momentum_effects if m > 0.1]) / len(momentum_effects)
                max_momentum_effect = max(momentum_effects)
                momentum_decay = momentum_effects[0] - momentum_effects[-1] if len(momentum_effects) > 1 else 0
            else:
                avg_momentum_effect = 0.0
                momentum_consistency = 0.5
                max_momentum_effect = 0.0
                momentum_decay = 0.0
            
            return [
                max(-1.0, min(1.0, avg_momentum_effect)),
                momentum_consistency,
                max(-1.0, min(1.0, max_momentum_effect)),
                max(-2.0, min(2.0, momentum_decay))
            ]
            
        except Exception as e:
            logger.debug(f"Error calculating cross-asset momentum: {e}")
            return [0.0, 0.0, 0.0, 0.0]
    
    def _extract_volatility_spillover_features(
        self,
        multi_asset_data: Dict[str, pd.DataFrame],
        primary_ticker: str,
        reference_tickers: List[str]
    ) -> List[float]:
        """Extract volatility spillover features."""
        try:
            features = []
            
            # Get volatility series for all assets
            volatilities = {}
            
            for ticker in [primary_ticker] + reference_tickers:
                if ticker in multi_asset_data:
                    df = multi_asset_data[ticker]
                    if len(df) >= self.volatility_window:
                        returns = df['close'].pct_change().dropna()
                        rolling_vol = returns.rolling(window=10).std()
                        volatilities[ticker] = rolling_vol.dropna().tail(self.volatility_window)
            
            if primary_ticker not in volatilities or len(volatilities) < 2:
                return [0.0] * 8
            
            primary_vol = volatilities[primary_ticker]
            
            # 1. Volatility correlation
            vol_correlations = []
            for ref_ticker in reference_tickers:
                if ref_ticker in volatilities:
                    # Align volatility series
                    common_idx = primary_vol.index.intersection(volatilities[ref_ticker].index)
                    if len(common_idx) >= 10:
                        vol_corr = primary_vol[common_idx].corr(volatilities[ref_ticker][common_idx])
                        if not np.isnan(vol_corr):
                            vol_correlations.append(vol_corr)
            
            if vol_correlations:
                avg_vol_corr = np.mean(vol_correlations)
                vol_corr_stability = 1 - np.std(vol_correlations)
                features.extend([
                    max(-1.0, min(1.0, avg_vol_corr)),
                    max(0.0, min(1.0, vol_corr_stability))
                ])
            else:
                features.extend([0.0, 0.5])
            
            # 2. Volatility spillover (how much other assets' volatility affects primary)
            spillover_effects = []
            for ref_ticker in reference_tickers[:3]:  # Limit for performance
                if ref_ticker in volatilities:
                    spillover = self._calculate_volatility_spillover(
                        primary_vol, volatilities[ref_ticker]
                    )
                    spillover_effects.append(spillover)
            
            if spillover_effects:
                avg_spillover = np.mean(spillover_effects)
                max_spillover = max(spillover_effects)
                features.extend([
                    max(-1.0, min(1.0, avg_spillover)),
                    max(-1.0, min(1.0, max_spillover))
                ])
            else:
                features.extend([0.0, 0.0])
            
            # 3. Volatility regime synchronization
            regime_sync = self._calculate_volatility_regime_sync(volatilities, primary_ticker)
            features.append(regime_sync)
            
            # 4. Relative volatility persistence
            vol_persistence = self._calculate_volatility_persistence(volatilities, primary_ticker)
            features.extend(vol_persistence)  # Should return 2 values
            
            # 5. Contagion indicator
            contagion = self._calculate_contagion_indicator(volatilities, primary_ticker)
            features.append(contagion)
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting volatility spillover features: {e}")
            return [0.0] * 8
    
    def _calculate_volatility_spillover(
        self, 
        primary_vol: pd.Series, 
        ref_vol: pd.Series
    ) -> float:
        """Calculate volatility spillover effect."""
        try:
            # Align series
            common_idx = primary_vol.index.intersection(ref_vol.index)
            if len(common_idx) < 10:
                return 0.0
            
            primary_aligned = primary_vol[common_idx]
            ref_aligned = ref_vol[common_idx]
            
            # Calculate lagged correlation (ref volatility -> primary volatility)
            if len(common_idx) > 5:
                lagged_ref = ref_aligned.shift(1).dropna()
                current_primary = primary_aligned.loc[lagged_ref.index]
                
                if len(lagged_ref) > 5:
                    spillover_corr = lagged_ref.corr(current_primary)
                    return spillover_corr if not np.isnan(spillover_corr) else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating volatility spillover: {e}")
            return 0.0
    
    def _calculate_volatility_regime_sync(
        self, 
        volatilities: Dict[str, pd.Series], 
        primary_ticker: str
    ) -> float:
        """Calculate volatility regime synchronization."""
        try:
            if primary_ticker not in volatilities:
                return 0.5
            
            primary_vol = volatilities[primary_ticker]
            
            # Define high/low volatility regimes
            primary_high_vol = primary_vol > primary_vol.quantile(0.7)
            
            sync_scores = []
            for ticker, vol_series in volatilities.items():
                if ticker != primary_ticker:
                    # Align indices
                    common_idx = primary_high_vol.index.intersection(vol_series.index)
                    if len(common_idx) > 10:
                        ref_high_vol = vol_series[common_idx] > vol_series[common_idx].quantile(0.7)
                        primary_high_aligned = primary_high_vol[common_idx]
                        
                        # Calculate synchronization (how often both are in same regime)
                        same_regime = (primary_high_aligned == ref_high_vol).sum()
                        sync_score = same_regime / len(common_idx)
                        sync_scores.append(sync_score)
            
            return np.mean(sync_scores) if sync_scores else 0.5
            
        except Exception as e:
            logger.debug(f"Error calculating volatility regime sync: {e}")
            return 0.5
    
    def _calculate_volatility_persistence(
        self, 
        volatilities: Dict[str, pd.Series], 
        primary_ticker: str
    ) -> List[float]:
        """Calculate volatility persistence measures."""
        try:
            if primary_ticker not in volatilities:
                return [0.5, 0.5]
            
            primary_vol = volatilities[primary_ticker]
            
            # Primary volatility persistence (autocorrelation)
            if len(primary_vol) > 10:
                primary_persistence = primary_vol.autocorr(lag=1)
                primary_persistence = primary_persistence if not np.isnan(primary_persistence) else 0.5
            else:
                primary_persistence = 0.5
            
            # Relative persistence (vs other assets)
            other_persistences = []
            for ticker, vol_series in volatilities.items():
                if ticker != primary_ticker and len(vol_series) > 10:
                    persistence = vol_series.autocorr(lag=1)
                    if not np.isnan(persistence):
                        other_persistences.append(persistence)
            
            if other_persistences:
                avg_other_persistence = np.mean(other_persistences)
                relative_persistence = primary_persistence - avg_other_persistence
            else:
                relative_persistence = 0.0
            
            return [
                max(-1.0, min(1.0, primary_persistence)),
                max(-2.0, min(2.0, relative_persistence))
            ]
            
        except Exception as e:
            logger.debug(f"Error calculating volatility persistence: {e}")
            return [0.5, 0.0]
    
    def _calculate_contagion_indicator(
        self, 
        volatilities: Dict[str, pd.Series], 
        primary_ticker: str
    ) -> float:
        """Calculate financial contagion indicator."""
        try:
            if primary_ticker not in volatilities or len(volatilities) < 3:
                return 0.0
            
            primary_vol = volatilities[primary_ticker]
            
            # Identify high volatility periods for primary asset
            high_vol_threshold = primary_vol.quantile(0.8)
            high_vol_periods = primary_vol > high_vol_threshold
            
            if not high_vol_periods.any():
                return 0.0
            
            # Check how often other assets also have high volatility during these periods
            contagion_scores = []
            
            for ticker, vol_series in volatilities.items():
                if ticker != primary_ticker:
                    # Align series
                    common_idx = high_vol_periods.index.intersection(vol_series.index)
                    if len(common_idx) > 5:
                        primary_high_aligned = high_vol_periods[common_idx]
                        other_vol_aligned = vol_series[common_idx]
                        
                        # During primary high vol periods, how often is other asset also high vol?
                        primary_high_periods = primary_high_aligned[primary_high_aligned]
                        if len(primary_high_periods) > 0:
                            other_high_threshold = other_vol_aligned.quantile(0.7)
                            other_high_during_primary = (
                                other_vol_aligned[primary_high_periods.index] > other_high_threshold
                            ).sum()
                            contagion_score = other_high_during_primary / len(primary_high_periods)
                            contagion_scores.append(contagion_score)
            
            return np.mean(contagion_scores) if contagion_scores else 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating contagion indicator: {e}")
            return 0.0
    
    def _extract_risk_regime_features(
        self,
        multi_asset_data: Dict[str, pd.DataFrame],
        primary_ticker: str,
        reference_tickers: List[str]
    ) -> List[float]:
        """Extract risk-on/risk-off regime features."""
        try:
            features = []
            
            # Detect current cross-asset regime
            regime = self._detect_cross_asset_regime(multi_asset_data, primary_ticker, reference_tickers)
            self.regime_history.append(regime)
            
            # Regime indicator features
            regime_indicators = [0.0] * len(CrossAssetRegime)
            regime_index = list(CrossAssetRegime).index(regime)
            regime_indicators[regime_index] = 1.0
            
            # Take first 4 regime indicators
            features.extend(regime_indicators[:4])
            
            # Regime stability
            if len(self.regime_history) >= 10:
                regime_changes = sum(1 for i in range(1, len(self.regime_history)) 
                                   if self.regime_history[i] != self.regime_history[i - 1])
                stability = 1 - (regime_changes / len(self.regime_history))
                features.append(stability)
            else:
                features.append(0.5)
            
            # Risk-on strength (how strong is the current regime)
            regime_strength = self._calculate_regime_strength(
                multi_asset_data, primary_ticker, reference_tickers, regime
            )
            features.append(regime_strength)
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting risk regime features: {e}")
            return [0.0] * 6
    
    def _detect_cross_asset_regime(
        self,
        multi_asset_data: Dict[str, pd.DataFrame],
        primary_ticker: str,
        reference_tickers: List[str]
    ) -> CrossAssetRegime:
        """Detect current cross-asset regime."""
        try:
            # Get aligned returns for regime classification
            aligned_returns = self._align_returns_data(
                multi_asset_data, [primary_ticker] + reference_tickers, 20
            )
            
            if aligned_returns.empty or len(aligned_returns) < 10:
                return CrossAssetRegime.RISK_OFF
            
            # Calculate regime indicators
            
            # 1. Average correlation (high = risk-off, low = regime breakdown)
            correlations = []
            for i, ticker1 in enumerate(aligned_returns.columns):
                for ticker2 in aligned_returns.columns[i+1:]:
                    corr = aligned_returns[ticker1].corr(aligned_returns[ticker2])
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            avg_correlation = np.mean(correlations) if correlations else 0.0
            
            # 2. Volatility level (recent vs historical)
            recent_vol = aligned_returns.std().mean()
            historical_vol = aligned_returns.rolling(window=15).std().mean().mean()
            vol_ratio = recent_vol / (historical_vol + 1e-8)
            
            # 3. Return dispersion (how much returns vary across assets)
            recent_returns = aligned_returns.iloc[-5:].mean()  # Last 5 days average
            return_dispersion = recent_returns.std()
            
            # 4. Asset class performance patterns
            equity_performance = self._calculate_asset_class_performance(
                aligned_returns, AssetClass.EQUITY
            )
            bond_performance = self._calculate_asset_class_performance(
                aligned_returns, AssetClass.BOND
            )
            
            # Regime classification logic
            if avg_correlation > 0.7 and vol_ratio > 1.5:
                return CrossAssetRegime.RISK_OFF
            elif avg_correlation < 0.3:
                return CrossAssetRegime.CORRELATION_BREAKDOWN
            elif return_dispersion > 0.02:  # High dispersion = rotation
                return CrossAssetRegime.ROTATION
            elif equity_performance > bond_performance + 0.01:
                return CrossAssetRegime.RISK_ON
            elif abs(equity_performance - bond_performance) < 0.005:
                return CrossAssetRegime.CONVERGENCE
            else:
                return CrossAssetRegime.DIVERGENCE
                
        except Exception as e:
            logger.debug(f"Error detecting cross-asset regime: {e}")
            return CrossAssetRegime.RISK_OFF
    
    def _calculate_asset_class_performance(
        self, 
        aligned_returns: pd.DataFrame, 
        asset_class: AssetClass
    ) -> float:
        """Calculate average performance for an asset class."""
        try:
            class_returns = []
            
            for ticker in aligned_returns.columns:
                if self.asset_class_mapping.get(ticker, AssetClass.UNKNOWN) == asset_class:
                    recent_return = aligned_returns[ticker].iloc[-5:].mean()  # Last 5 days
                    class_returns.append(recent_return)
            
            return np.mean(class_returns) if class_returns else 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating asset class performance: {e}")
            return 0.0
    
    def _calculate_regime_strength(
        self,
        multi_asset_data: Dict[str, pd.DataFrame],
        primary_ticker: str,
        reference_tickers: List[str],
        regime: CrossAssetRegime
    ) -> float:
        """Calculate strength of the current regime."""
        try:
            aligned_returns = self._align_returns_data(
                multi_asset_data, [primary_ticker] + reference_tickers, 15
            )
            
            if aligned_returns.empty:
                return 0.5
            
            # Calculate regime-specific strength metrics
            if regime == CrossAssetRegime.RISK_ON:
                # Strength = consistent positive correlations with risk assets
                risk_asset_corrs = []
                for ticker in aligned_returns.columns:
                    if self.asset_class_mapping.get(ticker, AssetClass.UNKNOWN) == AssetClass.EQUITY:
                        corr = aligned_returns[primary_ticker].corr(aligned_returns[ticker])
                        if not np.isnan(corr) and corr > 0:
                            risk_asset_corrs.append(corr)
                
                strength = np.mean(risk_asset_corrs) if risk_asset_corrs else 0.5
                
            elif regime == CrossAssetRegime.RISK_OFF:
                # Strength = high correlations across all assets
                all_corrs = []
                for i, ticker1 in enumerate(aligned_returns.columns):
                    for ticker2 in aligned_returns.columns[i+1:]:
                        corr = aligned_returns[ticker1].corr(aligned_returns[ticker2])
                        if not np.isnan(corr):
                            all_corrs.append(abs(corr))
                
                strength = np.mean(all_corrs) if all_corrs else 0.5
                
            elif regime == CrossAssetRegime.ROTATION:
                # Strength = high return dispersion
                recent_returns = aligned_returns.iloc[-3:].mean()
                strength = recent_returns.std() * 10  # Scale up
                strength = min(strength, 1.0)
                
            else:
                strength = 0.5  # Default strength
            
            return max(0.0, min(1.0, strength))
            
        except Exception as e:
            logger.debug(f"Error calculating regime strength: {e}")
            return 0.5
    
    def _extract_sector_rotation_features(
        self,
        multi_asset_data: Dict[str, pd.DataFrame],
        primary_ticker: str,
        reference_tickers: List[str]
    ) -> List[float]:
        """Extract sector rotation features (simplified)."""
        try:
            # Simplified sector rotation based on available data
            features = []
            
            # Get recent performance of different assets
            asset_performances = {}
            
            for ticker in [primary_ticker] + reference_tickers:
                if ticker in multi_asset_data:
                    df = multi_asset_data[ticker]
                    if len(df) >= 20:
                        # 20-day performance
                        performance = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
                        asset_performances[ticker] = performance
            
            if len(asset_performances) < 3:
                return [0.0] * 6
            
            # Calculate rotation metrics
            performances = list(asset_performances.values())
            
            # Performance dispersion (rotation strength indicator)
            rotation_strength = np.std(performances)
            features.append(min(rotation_strength * 10, 1.0))  # Scale and cap
            
            # Primary asset performance rank
            sorted_performances = sorted(performances, reverse=True)
            primary_performance = asset_performances.get(primary_ticker, 0)
            try:
                rank = sorted_performances.index(primary_performance) + 1
                rank_percentile = 1 - ((rank - 1) / len(sorted_performances))
            except ValueError:
                rank_percentile = 0.5
            
            features.append(rank_percentile)
            
            # Asset class rotation patterns (simplified)
            equity_perf = np.mean([perf for ticker, perf in asset_performances.items() 
                                  if self.asset_class_mapping.get(ticker, AssetClass.UNKNOWN) == AssetClass.EQUITY])
            bond_perf = np.mean([perf for ticker, perf in asset_performances.items() 
                                if self.asset_class_mapping.get(ticker, AssetClass.UNKNOWN) == AssetClass.BOND])
            
            equity_bond_rotation = equity_perf - bond_perf if not (np.isnan(equity_perf) or np.isnan(bond_perf)) else 0.0
            features.append(max(-0.5, min(0.5, equity_bond_rotation)) * 2)  # Scale to [-1, 1]
            
            # Momentum persistence (how long current rotation has lasted)
            momentum_persistence = self._calculate_rotation_persistence(
                multi_asset_data, primary_ticker, reference_tickers
            )
            features.append(momentum_persistence)
            
            # Rotation acceleration (is rotation speeding up?)
            rotation_acceleration = self._calculate_rotation_acceleration(
                multi_asset_data, primary_ticker, reference_tickers
            )
            features.append(rotation_acceleration)
            
            # Primary asset rotation beta (sensitivity to rotation)
            rotation_beta = self._calculate_rotation_beta(
                multi_asset_data, primary_ticker, reference_tickers
            )
            features.append(rotation_beta)
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting sector rotation features: {e}")
            return [0.0] * 6
    
    def _calculate_rotation_persistence(
        self,
        multi_asset_data: Dict[str, pd.DataFrame],
        primary_ticker: str,
        reference_tickers: List[str]
    ) -> float:
        """Calculate rotation persistence."""
        try:
            # Look at performance rankings over time
            lookback_periods = [5, 10, 15]
            consistent_ranking = 0
            total_comparisons = 0
            
            for period in lookback_periods:
                performances = {}
                
                for ticker in [primary_ticker] + reference_tickers[:5]:
                    if ticker in multi_asset_data:
                        df = multi_asset_data[ticker]
                        if len(df) > period:
                            perf = (df['close'].iloc[-1] - df['close'].iloc[-period]) / df['close'].iloc[-period]
                            performances[ticker] = perf
                
                if len(performances) >= 3:
                    sorted_tickers = sorted(performances.keys(), key=lambda t: performances[t], reverse=True)
                    primary_rank = sorted_tickers.index(primary_ticker) + 1 if primary_ticker in sorted_tickers else len(sorted_tickers) // 2
                    
                    # Compare with previous period ranking if available
                    if period < max(lookback_periods):
                        # Simplified persistence check
                        if primary_rank <= len(sorted_tickers) // 2:  # Top half
                            consistent_ranking += 1
                    
                    total_comparisons += 1
            
            return consistent_ranking / total_comparisons if total_comparisons > 0 else 0.5
            
        except Exception as e:
            logger.debug(f"Error calculating rotation persistence: {e}")
            return 0.5
    
    def _calculate_rotation_acceleration(
        self,
        multi_asset_data: Dict[str, pd.DataFrame],
        primary_ticker: str,
        reference_tickers: List[str]
    ) -> float:
        """Calculate rotation acceleration."""
        try:
            if primary_ticker not in multi_asset_data:
                return 0.0
            
            df = multi_asset_data[primary_ticker]
            if len(df) < 20:
                return 0.0
            
            # Compare recent vs earlier relative performance
            recent_perf = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            earlier_perf = (df['close'].iloc[-10] - df['close'].iloc[-15]) / df['close'].iloc[-15]
            
            # Calculate average reference performance for same periods
            recent_ref_perfs = []
            earlier_ref_perfs = []
            
            for ref_ticker in reference_tickers[:5]:
                if ref_ticker in multi_asset_data:
                    ref_df = multi_asset_data[ref_ticker]
                    if len(ref_df) >= 20:
                        recent_ref = (ref_df['close'].iloc[-1] - ref_df['close'].iloc[-5]) / ref_df['close'].iloc[-5]
                        earlier_ref = (ref_df['close'].iloc[-10] - ref_df['close'].iloc[-15]) / ref_df['close'].iloc[-15]
                        recent_ref_perfs.append(recent_ref)
                        earlier_ref_perfs.append(earlier_ref)
            
            if recent_ref_perfs and earlier_ref_perfs:
                recent_relative = recent_perf - np.mean(recent_ref_perfs)
                earlier_relative = earlier_perf - np.mean(earlier_ref_perfs)
                acceleration = recent_relative - earlier_relative
                return max(-1.0, min(1.0, acceleration * 5))  # Scale and bound
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating rotation acceleration: {e}")
            return 0.0
    
    def _calculate_rotation_beta(
        self,
        multi_asset_data: Dict[str, pd.DataFrame],
        primary_ticker: str,
        reference_tickers: List[str]
    ) -> float:
        """Calculate rotation beta (sensitivity to sector rotation)."""
        try:
            # Get returns data
            aligned_returns = self._align_returns_data(
                multi_asset_data, [primary_ticker] + reference_tickers, 30
            )
            
            if aligned_returns.empty or len(aligned_returns) < 15:
                return 1.0
            
            # Create rotation factor (dispersion of returns across assets each day)
            rotation_factor = aligned_returns.std(axis=1)  # Daily cross-sectional std
            primary_returns = aligned_returns[primary_ticker]
            
            # Calculate beta of primary vs rotation factor
            if len(rotation_factor) > 10 and rotation_factor.std() > 0:
                covariance = np.cov(primary_returns, rotation_factor)[0, 1]
                rotation_variance = np.var(rotation_factor)
                beta = covariance / rotation_variance if rotation_variance > 0 else 1.0
                return max(-3.0, min(3.0, beta))
            
            return 1.0
            
        except Exception as e:
            logger.debug(f"Error calculating rotation beta: {e}")
            return 1.0
    
    def _extract_factor_exposure_features(
        self,
        multi_asset_data: Dict[str, pd.DataFrame],
        primary_ticker: str,
        reference_tickers: List[str]
    ) -> List[float]:
        """Extract factor exposure features (simplified)."""
        try:
            features = []
            
            # Calculate exposures to different factors using available proxies
            for factor_name, factor_tickers in self.factor_proxies.items():
                exposure = self._calculate_factor_exposure(
                    multi_asset_data, primary_ticker, factor_tickers
                )
                features.append(exposure)
            
            # Ensure we have exactly 7 factor exposures
            while len(features) < 7:
                features.append(0.0)
            
            return features[:7]
            
        except Exception as e:
            logger.debug(f"Error extracting factor exposure features: {e}")
            return [0.0] * 7
    
    def _calculate_factor_exposure(
        self,
        multi_asset_data: Dict[str, pd.DataFrame],
        primary_ticker: str,
        factor_tickers: List[str]
    ) -> float:
        """Calculate exposure to a specific factor."""
        try:
            if primary_ticker not in multi_asset_data:
                return 0.0
            
            # Find available factor proxies in the data
            available_factor_tickers = [t for t in factor_tickers if t in multi_asset_data]
            
            if not available_factor_tickers:
                return 0.0
            
            # Get aligned returns
            all_tickers = [primary_ticker] + available_factor_tickers
            aligned_returns = self._align_returns_data(multi_asset_data, all_tickers, 30)
            
            if aligned_returns.empty or len(aligned_returns) < 10:
                return 0.0
            
            primary_returns = aligned_returns[primary_ticker]
            
            # Create factor return (average of available factor proxies)
            factor_returns = aligned_returns[available_factor_tickers].mean(axis=1)
            
            # Calculate correlation as proxy for factor exposure
            exposure = primary_returns.corr(factor_returns)
            
            return exposure if not np.isnan(exposure) else 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating factor exposure: {e}")
            return 0.0
    
    def _get_default_features(self) -> np.ndarray:
        """Get default features when extraction fails."""
        return np.zeros(50, dtype=np.float32)
    
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
        """Get names of all cross-asset features."""
        names = []
        
        # Correlation features
        for window in self.correlation_windows:
            names.extend([
                f"corr_{window}_mean", f"corr_{window}_std", f"corr_{window}_max", f"corr_{window}_min",
                f"corr_{window}_pos_ratio", f"corr_{window}_high_ratio", f"corr_{window}_stability", f"corr_{window}_avg_abs"
            ])
        
        # Relative strength features
        for horizon in [5, 20, 60]:
            names.extend([
                f"rel_str_{horizon}_strength", f"rel_str_{horizon}_rank", f"rel_str_{horizon}_beat_rate",
                f"rel_str_{horizon}_rel_vol", f"rel_str_{horizon}_risk_adj", f"rel_str_{horizon}_consistency"
            ])
        
        # Co-movement features
        names.extend([
            "comovement_avg_beta", "comovement_beta_std", "comovement_lead_lag", "comovement_synchronization",
            "regime_corr_high_vol", "regime_corr_low_vol", "regime_corr_diff", "regime_corr_stability",
            "cross_momentum_avg", "cross_momentum_consistency", "cross_momentum_max", "cross_momentum_decay"
        ])
        
        # Volatility spillover features
        names.extend([
            "vol_spillover_avg_corr", "vol_spillover_corr_stability", "vol_spillover_avg_effect", "vol_spillover_max_effect",
            "vol_spillover_regime_sync", "vol_spillover_primary_persistence", "vol_spillover_relative_persistence", "vol_spillover_contagion"
        ])
        
        # Risk regime features
        names.extend([
            "risk_regime_risk_on", "risk_regime_risk_off", "risk_regime_rotation", "risk_regime_breakdown",
            "risk_regime_stability", "risk_regime_strength"
        ])
        
        # Sector rotation features (if enabled)
        if self.enable_sector_rotation:
            names.extend([
                "sector_rotation_strength", "sector_rotation_rank", "sector_rotation_equity_bond",
                "sector_rotation_persistence", "sector_rotation_acceleration", "sector_rotation_beta"
            ])
        
        # Factor exposure features (if enabled)
        if self.enable_factor_analysis:
            factor_names = list(self.factor_proxies.keys())
            names.extend([f"factor_{name}" for name in factor_names])
        
        return names
    
    def get_extractor_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the cross-asset extractor."""
        return {
            'correlation_windows': self.correlation_windows,
            'max_assets': self.max_assets,
            'cache_size': len(self.feature_cache),
            'regime_history_length': len(self.regime_history),
            'current_regime': self.regime_history[-1].value if self.regime_history else 'unknown',
            'asset_class_count': len(set(self.asset_class_mapping.values())),
            'factor_count': len(self.factor_proxies),
            'settings': {
                'min_overlap_periods': self.min_overlap_periods,
                'enable_sector_rotation': self.enable_sector_rotation,
                'enable_factor_analysis': self.enable_factor_analysis,
                'volatility_window': self.volatility_window,
                'cache_ttl_minutes': self.cache_ttl_minutes
            }
        }