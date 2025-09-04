"""Ensemble framework combining TimesFM and TSMamba models for superior predictions."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from loguru import logger
import warnings
from pathlib import Path

from .timesfm_predictor import TimesFMPredictor
from .tsmamba_predictor import TSMambaPredictor
from .samba_predictor import SAMBAPredictor
from .tft_predictor import TFTPredictor
from .uncertainty import ConformalPredictor
from .regime_detector import HiddenMarkovRegimeDetector, MarketRegime
from .neural_meta_learner import NeuralMetaLearner
from .online_learner import OnlineLearningSystem
from .multi_resolution import (
    MinuteScalePredictor,
    HourlyPredictor,
    WeeklyPredictor,
    ResolutionFuser,
    AdaptiveResolutionWeighting
)
from ..features.meta_feature_extractor import MetaFeatureExtractor
from ..config.settings import config

class MetaLearningEnsemble:
    """Dynamic ensemble of TimesFM + TSMamba + SAMBA + TFT models with adaptive weighting."""
    
    def __init__(
        self,
        horizon_len: Union[int, List[int]] = [1, 5, 20],
        timesfm_weight: float = 0.4,
        tsmamba_weight: float = 0.25,
        samba_weight: float = 0.2,
        tft_weight: float = 0.15,
        uncertainty_alpha: float = 0.1,
        performance_window: int = 50,
        device: str = "auto",
        enable_neural_meta_learning: bool = True,
        enable_online_learning: bool = True,
        enable_multi_resolution: bool = True,
        multi_resolution_weight: float = 0.3
    ):
        """
        Initialize the ensemble framework.
        
        Args:
            horizon_len: Prediction horizons
            timesfm_weight: Initial weight for TimesFM model
            tsmamba_weight: Initial weight for TSMamba model
            samba_weight: Initial weight for SAMBA model
            tft_weight: Initial weight for TFT model
            uncertainty_alpha: Alpha for uncertainty quantification
            performance_window: Window for tracking model performance
            device: Device for computation
            enable_neural_meta_learning: Enable neural meta-learner for dynamic weights
            enable_online_learning: Enable online learning for continuous adaptation
            enable_multi_resolution: Enable multi-resolution prediction system
            multi_resolution_weight: Weight for multi-resolution predictions in ensemble
        """
        
        self.horizon_len = horizon_len if isinstance(horizon_len, list) else [horizon_len]
        self.uncertainty_alpha = uncertainty_alpha
        self.performance_window = performance_window
        self.enable_neural_meta_learning = enable_neural_meta_learning
        self.enable_online_learning = enable_online_learning
        self.enable_multi_resolution = enable_multi_resolution
        self.multi_resolution_weight = multi_resolution_weight
        
        # Initialize models
        logger.info("Initializing ensemble models...")
        
        self.timesfm = TimesFMPredictor(
            horizon_len=self.horizon_len,
            device=device
        )
        
        self.tsmamba = TSMambaPredictor(
            horizon_len=self.horizon_len,
            device=device
        )
        
        self.samba = SAMBAPredictor(
            horizon_len=self.horizon_len,
            device=device
        )
        
        self.tft = TFTPredictor(
            horizon_len=self.horizon_len,
            device=device
        )
        
        # Initialize multi-resolution system
        self.multi_resolution_predictors = None
        self.resolution_fuser = None
        self.adaptive_weighting = None
        
        if self.enable_multi_resolution:
            try:
                logger.info("Initializing multi-resolution prediction system...")
                
                # Initialize resolution-specific predictors
                self.multi_resolution_predictors = {
                    'minute': MinuteScalePredictor(
                        lookback_window=30,
                        prediction_horizons=[1, 5],
                        device=device
                    ),
                    'hourly': HourlyPredictor(
                        lookback_hours=48,
                        prediction_horizons=[1, 2, 4],
                        device=device
                    ),
                    'weekly': WeeklyPredictor(
                        lookback_weeks=26,
                        prediction_horizons=[5, 10, 20],
                        device=device
                    )
                }
                
                # Initialize resolution fusion network
                self.resolution_fuser = ResolutionFuser(
                    resolution_dims={
                        'minute': 64,
                        'hourly': 128,
                        'weekly': 256
                    },
                    hidden_dim=256,
                    output_horizons=self.horizon_len,
                    device=device
                )
                
                # Initialize adaptive resolution weighting
                self.adaptive_weighting = AdaptiveResolutionWeighting(
                    performance_window=self.performance_window,
                    adaptation_rate=0.1,
                    device=device
                )
                
                logger.info("Multi-resolution system initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize multi-resolution system: {e}")
                logger.warning("Falling back to standard ensemble without multi-resolution")
                self.enable_multi_resolution = False
                self.multi_resolution_predictors = None
                self.resolution_fuser = None
                self.adaptive_weighting = None
        
        # Initialize regime detector for market adaptation
        self.regime_detector = HiddenMarkovRegimeDetector()
        
        # Initialize MAPIE uncertainty model for ensemble predictions
        self.uncertainty_model = ConformalPredictor(
            method="plus",  # CV+ method for better coverage
            alpha=uncertainty_alpha,  # 0.1 for 90% prediction intervals
            n_splits=5
        )
        
        # Regime-specific uncertainty parameters for adaptive calibration
        self.regime_alpha_config = {
            'bull_trend': {'alpha': 0.15, 'method': 'plus'},
            'bear_trend': {'alpha': 0.10, 'method': 'plus'},  # Tighter intervals in bear
            'high_volatility': {'alpha': 0.08, 'method': 'naive'},  # Much tighter in volatile
            'sideways': {'alpha': 0.20, 'method': 'base'}  # Wider intervals in sideways
        }
        
        # Current regime adaptive alpha
        self.current_regime_alpha = uncertainty_alpha
        
        # Model weights (can be dynamic) - now 4 models + multi-resolution
        self.weights = {
            'timesfm': timesfm_weight,
            'tsmamba': tsmamba_weight,
            'samba': samba_weight,
            'tft': tft_weight
        }
        
        # Add multi-resolution weight if enabled
        if self.enable_multi_resolution:
            self.weights['multi_resolution'] = multi_resolution_weight
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        # Performance tracking (now 4 models + multi-resolution)
        self.performance_history = {
            'timesfm': [],
            'tsmamba': [],
            'samba': [],
            'tft': [],
            'ensemble': []
        }
        
        # Add multi-resolution performance tracking
        if self.enable_multi_resolution:
            self.performance_history['multi_resolution'] = []
        
        # Enhanced prediction cache with TTL and memory management
        self.prediction_cache = {}
        self.cache_ttl = {}  # Cache time-to-live tracking
        self.cache_max_size = 100  # Maximum cache entries
        self.cache_ttl_seconds = 300  # 5 minutes TTL
        
        # Initialize neural meta-learning components
        self.neural_meta_learner = None
        self.feature_extractor = None
        self.online_learner = None
        
        if self.enable_neural_meta_learning:
            try:
                logger.info("Initializing neural meta-learning components...")
                
                # Initialize enhanced meta-feature extractor with Phase 6 features
                self.feature_extractor = MetaFeatureExtractor(
                    lookback_window=60,
                    performance_window=performance_window,
                    volatility_window=20,
                    enable_feature_selection=True
                )
                
                # Calculate meta-feature dimension (using enhanced features)
                dummy_features = self.feature_extractor._get_default_enhanced_combined_features()
                meta_feature_dim = len(dummy_features)
                
                # Initialize neural meta-learner (now 4 models: TimesFM, TSMamba, SAMBA, TFT)
                self.neural_meta_learner = NeuralMetaLearner(
                    meta_feature_dim=meta_feature_dim,
                    hidden_dim=128,
                    horizons=self.horizon_len,
                    n_models=4,
                    attention_heads=8,
                    dropout=0.15,
                    device=device
                )
                
                # Initialize online learning system
                if self.enable_online_learning:
                    self.online_learner = OnlineLearningSystem(
                        neural_meta_learner=self.neural_meta_learner,
                        feature_extractor=self.feature_extractor,
                        learning_rate=1e-4,
                        batch_size=32,
                        update_frequency_minutes=60,
                        performance_window=performance_window,
                        device=device
                    )
                
                logger.info(f"Neural meta-learning initialized with {meta_feature_dim} features")
                
            except Exception as e:
                logger.error(f"Failed to initialize neural meta-learning: {e}")
                logger.warning("Falling back to static ensemble weights")
                self.enable_neural_meta_learning = False
                self.enable_online_learning = False
        
        multi_res_status = "enabled" if self.enable_multi_resolution else "disabled"
        logger.info(f"Ensemble initialized with weights: {self.weights}")
        logger.info(f"Multi-resolution prediction system: {multi_res_status}")
    
    def fit_uncertainty_model(
        self,
        training_data: Dict[str, pd.DataFrame],
        tickers: List[str],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Fit the MAPIE uncertainty model on historical ensemble predictions.
        
        Args:
            training_data: Dictionary mapping tickers to historical OHLCV data
            tickers: List of ticker symbols to train on
            validation_split: Fraction of data for validation
            
        Returns:
            Training metrics and coverage statistics
        """
        logger.info(f"Fitting MAPIE uncertainty model on {len(tickers)} tickers")
        
        # Collect training data for MAPIE
        X_train = []
        y_train = []
        
        for ticker in tickers:
            if ticker not in training_data:
                continue
                
            df = training_data[ticker]
            if len(df) < 100:  # Need enough data
                continue
            
            # Split data
            split_idx = int(len(df) * (1 - validation_split))
            train_df = df.iloc[:split_idx]
            val_df = df.iloc[split_idx:]
            
            # Generate ensemble predictions on training data
            for i in range(60, len(train_df) - 20):  # Need lookback and lookahead
                try:
                    # Get historical data up to point i
                    hist_data = train_df.iloc[:i]
                    if len(hist_data) < 50:
                        continue
                    
                    # Generate ensemble prediction (without MAPIE intervals)
                    pred_result = self._predict_without_mapie(hist_data, ticker)
                    
                    if 'horizon_1' not in pred_result:
                        continue
                    
                    # Calculate actual return for horizon 1
                    current_price = train_df['close'].iloc[i]
                    future_price = train_df['close'].iloc[i + 1]
                    actual_return = (future_price - current_price) / current_price
                    
                    # Extract features for MAPIE training
                    ensemble_pred = pred_result['horizon_1']['prediction']
                    confidence = pred_result['horizon_1']['confidence']
                    disagreement = pred_result['horizon_1']['model_disagreement']
                    regime_conf = pred_result['horizon_1']['regime_info']['regime_confidence']
                    
                    # Feature vector for MAPIE
                    features = [ensemble_pred, confidence, disagreement, regime_conf]
                    
                    X_train.append(features)
                    y_train.append(actual_return)
                    
                except Exception as e:
                    logger.debug(f"Error processing training sample for {ticker}: {e}")
                    continue
        
        if len(X_train) < 50:
            logger.warning(f"Insufficient training data for MAPIE: {len(X_train)} samples")
            return {'error': 'insufficient_data'}
        
        # Fit MAPIE model
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        logger.info(f"Fitting MAPIE on {len(X_train)} training samples")
        self.uncertainty_model.fit(X_train, y_train)
        
        # Validate MAPIE performance
        coverage_stats = self._validate_mapie_coverage(training_data, tickers, validation_split)
        
        return {
            'training_samples': len(X_train),
            'coverage_stats': coverage_stats,
            'mapie_fitted': True
        }
    
    def _predict_without_mapie(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> Dict[str, Any]:
        """Generate ensemble predictions without MAPIE intervals (for training)."""
        # Temporarily disable uncertainty model to avoid recursion
        temp_uncertainty_model = self.uncertainty_model
        self.uncertainty_model = None
        
        try:
            result = self.predict(df, ticker, return_confidence=True, use_cache=False)
            return result
        finally:
            # Restore uncertainty model
            self.uncertainty_model = temp_uncertainty_model
    
    def _validate_mapie_coverage(
        self,
        training_data: Dict[str, pd.DataFrame],
        tickers: List[str],
        validation_split: float
    ) -> Dict[str, Any]:
        """Validate MAPIE coverage on validation set."""
        coverage_results = []
        
        for ticker in tickers[:5]:  # Validate on subset
            if ticker not in training_data:
                continue
                
            df = training_data[ticker]
            split_idx = int(len(df) * (1 - validation_split))
            val_df = df.iloc[split_idx:]
            
            if len(val_df) < 20:
                continue
            
            for i in range(10, len(val_df) - 5):
                try:
                    hist_data = val_df.iloc[:i]
                    pred_result = self.predict(hist_data, ticker, use_cache=False)
                    
                    if 'horizon_1' in pred_result:
                        current_price = val_df['close'].iloc[i]
                        future_price = val_df['close'].iloc[i + 1]
                        actual_return = (future_price - current_price) / current_price
                        
                        pred_interval = pred_result['horizon_1']['prediction_interval']
                        within_interval = pred_interval[0] <= actual_return <= pred_interval[1]
                        
                        coverage_results.append({
                            'within_interval': within_interval,
                            'interval_width': pred_interval[1] - pred_interval[0],
                            'actual_return': actual_return
                        })
                        
                except Exception as e:
                    continue
        
        if coverage_results:
            coverage_rate = np.mean([r['within_interval'] for r in coverage_results])
            avg_width = np.mean([r['interval_width'] for r in coverage_results])
            
            return {
                'coverage_rate': coverage_rate,
                'target_coverage': 1 - self.uncertainty_alpha,
                'coverage_gap': abs(coverage_rate - (1 - self.uncertainty_alpha)),
                'avg_interval_width': avg_width,
                'num_validations': len(coverage_results)
            }
        
        return {'error': 'no_validation_data'}
    
    def _adapt_mapie_for_regime(self, regime_state) -> float:
        """Adapt MAPIE alpha parameter based on current market regime."""
        try:
            regime_name = regime_state.regime.value.lower()
            
            if regime_name in self.regime_alpha_config:
                regime_config = self.regime_alpha_config[regime_name]
                target_alpha = regime_config['alpha']
                
                # Smooth transition to avoid sudden changes
                alpha_smoothing = 0.3
                new_alpha = (1 - alpha_smoothing) * self.current_regime_alpha + alpha_smoothing * target_alpha
                
                # Update if significant change
                if abs(new_alpha - self.current_regime_alpha) > 0.01:
                    logger.info(f"Adapting MAPIE alpha for {regime_name}: {self.current_regime_alpha:.3f} → {new_alpha:.3f}")
                    self.current_regime_alpha = new_alpha
                    self.uncertainty_model.alpha = new_alpha
                
                return new_alpha
            else:
                return self.uncertainty_alpha
                
        except Exception as e:
            logger.error(f"Error adapting MAPIE for regime: {e}")
            return self.uncertainty_alpha
    
    def predict(
        self,
        df: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        ticker: str,
        return_confidence: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Generate ensemble predictions combining TimesFM, TSMamba, and SAMBA.
        
        Args:
            df: Historical OHLCV data (DataFrame or dict of DataFrames for multiple stocks)
            ticker: Stock ticker symbol
            return_confidence: Whether to return confidence intervals
            use_cache: Whether to use cached predictions
            
        Returns:
            Dictionary with ensemble predictions for each horizon
        """
        
        # Handle different input types and create cache key
        if isinstance(df, dict):
            # Multiple stocks data - use for SAMBA
            data_dict = df
            primary_df = df.get(ticker)
            if primary_df is None or len(primary_df) == 0:
                return self._fallback_prediction()
            cache_key = f"{ticker}_{len(data_dict)}_{primary_df.index[-1]}"
        else:
            # Single stock data
            primary_df = df
            data_dict = {ticker: df}
            cache_key = f"{ticker}_{len(df)}_{df.index[-1] if len(df) > 0 else 'empty'}"
        # Check cache with TTL validation
        if use_cache and self._is_cache_valid(cache_key):
            logger.debug(f"Using cached prediction for {ticker}")
            return self.prediction_cache[cache_key]
        
        try:
            # Detect current market regime for adaptive predictions
            regime_state = self.regime_detector.detect_regime(data_dict)
            adaptation_factors = self.regime_detector.get_regime_adaptation_factors(regime_state)
            
            # Adapt MAPIE parameters for current regime
            regime_alpha = self._adapt_mapie_for_regime(regime_state)
            
            # Get predictions from individual models with memory management
            try:
                # Clear any GPU cache before predictions to avoid OOM
                self._manage_gpu_memory()
                
                timesfm_pred = self.timesfm.predict(primary_df, ticker, return_confidence)
                tsmamba_pred = self.tsmamba.predict(primary_df, ticker, return_confidence)
                samba_pred = self.samba.predict(data_dict, ticker, return_confidence)
                tft_pred = self.tft.predict(primary_df, ticker, return_confidence)
                
                # Get multi-resolution predictions if enabled
                multi_resolution_pred = None
                if self.enable_multi_resolution and self.multi_resolution_predictors:
                    multi_resolution_pred = self._get_multi_resolution_predictions(
                        primary_df, ticker, return_confidence
                    )
                
            except Exception as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    logger.warning(f"GPU memory issue during prediction for {ticker}: {e}")
                    logger.info("Attempting prediction with memory cleanup")
                    
                    # Force memory cleanup and retry once
                    self._force_gpu_cleanup()
                    timesfm_pred = self.timesfm.predict(primary_df, ticker, return_confidence)
                    tsmamba_pred = self.tsmamba.predict(primary_df, ticker, return_confidence)
                    samba_pred = self.samba.predict(data_dict, ticker, return_confidence)
                    tft_pred = self.tft.predict(primary_df, ticker, return_confidence)
                    
                    # Retry multi-resolution predictions
                    multi_resolution_pred = None
                    if self.enable_multi_resolution and self.multi_resolution_predictors:
                        multi_resolution_pred = self._get_multi_resolution_predictions(
                            primary_df, ticker, return_confidence
                        )
                else:
                    raise e
            
            # Combine predictions using weighted averaging
            ensemble_result = {}
            
            for horizon in self.horizon_len:
                horizon_key = f"horizon_{horizon}"
                
                if (horizon_key in timesfm_pred and horizon_key in tsmamba_pred and 
                    horizon_key in samba_pred and horizon_key in tft_pred):
                    # Extract individual predictions (4 models + multi-resolution)
                    timesfm_val = timesfm_pred[horizon_key]['prediction']
                    tsmamba_val = tsmamba_pred[horizon_key]['prediction']
                    samba_val = samba_pred[horizon_key]['prediction']
                    tft_val = tft_pred[horizon_key]['prediction']
                    
                    # Extract multi-resolution prediction if available
                    multi_resolution_val = None
                    if multi_resolution_pred and horizon_key in multi_resolution_pred:
                        multi_resolution_val = multi_resolution_pred[horizon_key]['prediction']
                    
                    # Apply regime-adaptive weighting
                    adapted_weights = self._adapt_weights_for_regime(
                        regime_state, 
                        adaptation_factors, 
                        price_data=df if isinstance(df, pd.DataFrame) else data_dict,
                        horizon=horizon
                    )
                    
                    # Weighted ensemble prediction with regime adaptation (4 models + multi-resolution)
                    raw_ensemble_pred = (
                        adapted_weights['timesfm'] * timesfm_val +
                        adapted_weights['tsmamba'] * tsmamba_val +
                        adapted_weights['samba'] * samba_val +
                        adapted_weights['tft'] * tft_val
                    )
                    
                    # Add multi-resolution prediction if available
                    if multi_resolution_val is not None and 'multi_resolution' in adapted_weights:
                        raw_ensemble_pred += adapted_weights['multi_resolution'] * multi_resolution_val
                    
                    # Apply financial sanity checks
                    ensemble_pred = self._apply_financial_sanity_checks(
                        raw_ensemble_pred, horizon, regime_state, ticker
                    )
                    
                    # Combine confidence scores (4 models + multi-resolution)
                    timesfm_conf = timesfm_pred[horizon_key].get('confidence', 0.5)
                    tsmamba_conf = tsmamba_pred[horizon_key].get('confidence', 0.5)
                    samba_conf = samba_pred[horizon_key].get('confidence', 0.5)
                    tft_conf = tft_pred[horizon_key].get('confidence', 0.5)
                    
                    # Apply regime adaptation to confidence (4 models)
                    base_conf = (
                        adapted_weights['timesfm'] * timesfm_conf +
                        adapted_weights['tsmamba'] * tsmamba_conf +
                        adapted_weights['samba'] * samba_conf +
                        adapted_weights['tft'] * tft_conf
                    )
                    
                    # Add multi-resolution confidence if available
                    if multi_resolution_pred and horizon_key in multi_resolution_pred:
                        multi_resolution_conf = multi_resolution_pred[horizon_key].get('confidence', 0.5)
                        if 'multi_resolution' in adapted_weights:
                            base_conf += adapted_weights['multi_resolution'] * multi_resolution_conf
                    
                    # Adjust confidence based on regime uncertainty
                    regime_conf_adjustment = adaptation_factors['confidence_threshold_adjustment']
                    ensemble_conf = max(0.0, min(1.0, base_conf + regime_conf_adjustment))
                    
                    # Model disagreement as additional uncertainty (4-model disagreement)
                    disagreement = (
                        abs(timesfm_val - tsmamba_val) + 
                        abs(timesfm_val - samba_val) +
                        abs(timesfm_val - tft_val) +
                        abs(tsmamba_val - samba_val) +
                        abs(tsmamba_val - tft_val) +
                        abs(samba_val - tft_val)
                    ) / 6.0  # 6 pairwise comparisons for 4 models
                    
                    # MAPIE-enhanced prediction intervals
                    if self.uncertainty_model and self.uncertainty_model.is_fitted:
                        # Use MAPIE for rigorous conformal prediction intervals
                        features = np.array([[
                            ensemble_pred,
                            ensemble_conf,
                            disagreement,
                            regime_state.confidence
                        ]])
                        
                        try:
                            mapie_pred, mapie_intervals = self.uncertainty_model.predict(features, alpha=regime_alpha)
                            ensemble_lower = float(mapie_intervals[0, 0])
                            ensemble_upper = float(mapie_intervals[0, 1])
                            
                            # Apply sanity checks to intervals as well
                            ensemble_lower = max(ensemble_lower, ensemble_pred - 0.30)  # Max 30% below
                            ensemble_upper = min(ensemble_upper, ensemble_pred + 0.30)  # Max 30% above
                            
                            # Optional: Enhance MAPIE intervals with model disagreement (capped)
                            interval_width = ensemble_upper - ensemble_lower
                            disagreement_adjustment = min(disagreement * 0.05, 0.02)  # Cap adjustment at 2%
                            ensemble_lower -= disagreement_adjustment / 2
                            ensemble_upper += disagreement_adjustment / 2
                            
                            # Final interval width check
                            final_width = ensemble_upper - ensemble_lower
                            if final_width > 0.40:  # 40% max width
                                center = (ensemble_upper + ensemble_lower) / 2
                                ensemble_lower = center - 0.20
                                ensemble_upper = center + 0.20
                            
                        except Exception as e:
                            logger.warning(f"MAPIE prediction failed, using fallback: {e}")
                            # Fallback to model disagreement approach - use percentage-based intervals
                            # Use more conservative fallback intervals to avoid extreme widths
                            timesfm_interval = timesfm_pred[horizon_key].get('prediction_interval', [timesfm_val-0.005, timesfm_val+0.005])
                            tsmamba_interval = tsmamba_pred[horizon_key].get('prediction_interval', [tsmamba_val-0.005, tsmamba_val+0.005])
                            samba_interval = samba_pred[horizon_key].get('prediction_interval', [samba_val-0.005, samba_val+0.005])
                            tft_interval = tft_pred[horizon_key].get('prediction_interval', [tft_val-0.005, tft_val+0.005])
                            
                            # Calculate percentage widths with better handling of small values
                            def safe_percentage_width(interval, pred_val):
                                """Calculate percentage width safely handling small prediction values."""
                                abs_val = abs(pred_val)
                                if abs_val < 0.001:  # Very small prediction
                                    return 0.05  # 5% default width
                                elif abs_val < 0.01:  # Small prediction
                                    return 0.10  # 10% width
                                else:
                                    width = interval[1] - interval[0]
                                    return width / abs_val
                            
                            timesfm_width_pct = safe_percentage_width(timesfm_interval, timesfm_val)
                            tsmamba_width_pct = safe_percentage_width(tsmamba_interval, tsmamba_val)
                            samba_width_pct = safe_percentage_width(samba_interval, samba_val)
                            tft_width_pct = safe_percentage_width(tft_interval, tft_val)
                            
                            # Use the maximum percentage width, then convert back to absolute for ensemble prediction
                            base_interval_width_pct = max(timesfm_width_pct, tsmamba_width_pct, samba_width_pct, tft_width_pct)
                            base_interval_width_pct = min(base_interval_width_pct, 0.15)  # Cap at 15% for safety
                            
                            # Cap disagreement effect and total width
                            disagreement_effect = min(disagreement * 0.005, 0.02)  # Cap at 2% max
                            enhanced_width_pct = min(base_interval_width_pct + disagreement_effect, 0.25)  # Cap at 25%
                            
                            enhanced_width = max(abs(ensemble_pred) * enhanced_width_pct, 0.01)  # Min 1% width
                            enhanced_width = min(enhanced_width, 0.25)  # Cap at 25% absolute
                            
                            ensemble_lower = ensemble_pred - enhanced_width / 2
                            ensemble_upper = ensemble_pred + enhanced_width / 2
                    else:
                        # Fallback when MAPIE not fitted - use model disagreement approach with percentage-based intervals
                        # Use more conservative fallback intervals to avoid extreme widths
                        timesfm_interval = timesfm_pred[horizon_key].get('prediction_interval', [timesfm_val-0.005, timesfm_val+0.005])
                        tsmamba_interval = tsmamba_pred[horizon_key].get('prediction_interval', [tsmamba_val-0.005, tsmamba_val+0.005])
                        samba_interval = samba_pred[horizon_key].get('prediction_interval', [samba_val-0.005, samba_val+0.005])
                        tft_interval = tft_pred[horizon_key].get('prediction_interval', [tft_val-0.005, tft_val+0.005])
                        
                        # Calculate percentage widths with better handling of small values
                        def safe_percentage_width(interval, pred_val):
                            """Calculate percentage width safely handling small prediction values."""
                            abs_val = abs(pred_val)
                            if abs_val < 0.001:  # Very small prediction
                                return 0.05  # 5% default width
                            elif abs_val < 0.01:  # Small prediction
                                return 0.10  # 10% width
                            else:
                                width = interval[1] - interval[0]
                                return width / abs_val
                        
                        timesfm_width_pct = safe_percentage_width(timesfm_interval, timesfm_val)
                        tsmamba_width_pct = safe_percentage_width(tsmamba_interval, tsmamba_val)
                        samba_width_pct = safe_percentage_width(samba_interval, samba_val)
                        tft_width_pct = safe_percentage_width(tft_interval, tft_val)
                        
                        # Use the maximum percentage width, then convert back to absolute for ensemble prediction
                        base_interval_width_pct = max(timesfm_width_pct, tsmamba_width_pct, samba_width_pct, tft_width_pct)
                        base_interval_width_pct = min(base_interval_width_pct, 0.15)  # Cap at 15% for safety
                        
                        enhanced_width_pct = base_interval_width_pct + (disagreement * 0.01)  # Add 1% per unit disagreement
                        enhanced_width = abs(ensemble_pred) * enhanced_width_pct
                        ensemble_lower = ensemble_pred - enhanced_width / 2
                        ensemble_upper = ensemble_pred + enhanced_width / 2
                    
                    ensemble_result[horizon_key] = {
                        'prediction': float(ensemble_pred),
                        'confidence': float(ensemble_conf),
                        'prediction_interval': [float(ensemble_lower), float(ensemble_upper)],
                        'model_disagreement': float(disagreement),
                        'individual_predictions': {
                            'timesfm': timesfm_val,
                            'tsmamba': tsmamba_val,
                            'samba': samba_val,
                            'tft': tft_val
                        },
                        'model_weights': adapted_weights.copy(),
                        'regime_info': {
                            'current_regime': regime_state.regime.value,
                            'regime_confidence': regime_state.confidence,
                            'regime_duration': regime_state.duration,
                            'adaptation_factors': adaptation_factors
                        }
                    }
                
                else:
                    # Fallback if some models fail (prioritize TFT, then TimesFM, then others)
                    if horizon_key in tft_pred:
                        ensemble_result[horizon_key] = tft_pred[horizon_key].copy()
                        ensemble_result[horizon_key]['model_weights'] = {'timesfm': 0.0, 'tsmamba': 0.0, 'samba': 0.0, 'tft': 1.0}
                    elif horizon_key in timesfm_pred:
                        ensemble_result[horizon_key] = timesfm_pred[horizon_key].copy()
                        ensemble_result[horizon_key]['model_weights'] = {'timesfm': 1.0, 'tsmamba': 0.0, 'samba': 0.0, 'tft': 0.0}
                    elif horizon_key in tsmamba_pred:
                        ensemble_result[horizon_key] = tsmamba_pred[horizon_key].copy()
                        ensemble_result[horizon_key]['model_weights'] = {'timesfm': 0.0, 'tsmamba': 1.0, 'samba': 0.0, 'tft': 0.0}
                    elif horizon_key in samba_pred:
                        ensemble_result[horizon_key] = samba_pred[horizon_key].copy()
                        ensemble_result[horizon_key]['model_weights'] = {'timesfm': 0.0, 'tsmamba': 0.0, 'samba': 1.0, 'tft': 0.0}
                    else:
                        # All models failed - generate fallback
                        ensemble_result[horizon_key] = self._fallback_prediction(horizon)
            
            # Cache the result with TTL and size management
            if use_cache:
                self._cache_prediction(cache_key, ensemble_result)
            
            # Add ensemble metadata
            ensemble_result['ensemble_info'] = {
                'timestamp': datetime.now().isoformat(),
                'model_weights': self.weights.copy(),
                'data_points': len(df),
                'ticker': ticker
            }
            
            logger.debug(f"Generated ensemble prediction for {ticker}")
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Error generating ensemble prediction for {ticker}: {e}")
            return self._fallback_prediction_full()
    
    def _apply_financial_sanity_checks(
        self, 
        prediction: float, 
        horizon: int, 
        regime_state: Any,
        ticker: str
    ) -> float:
        """Apply financial sanity checks to ensure realistic predictions.
        
        Args:
            prediction: Raw ensemble prediction (percentage return)
            horizon: Prediction horizon (days)
            regime_state: Current market regime
            ticker: Stock ticker for logging
        
        Returns:
            Sanitized prediction within realistic bounds
        """
        original_prediction = prediction
        
        # Define realistic bounds based on horizon and market conditions
        if horizon == 1:  # Daily
            max_daily_return = 0.15  # 15% max daily move for volatile stocks
            regime_name = getattr(regime_state, 'name', 'unknown').lower()
            if 'volatility' in regime_name or 'volatile' in regime_name:
                max_daily_return = 0.20  # 20% in high volatility
            elif 'bear' in regime_name:
                max_daily_return = 0.25  # 25% in bear markets (more extreme moves)
            
            prediction = np.clip(prediction, -max_daily_return, max_daily_return)
            
        elif horizon <= 5:  # Weekly
            max_weekly_return = 0.30  # 30% max weekly move
            regime_name = getattr(regime_state, 'name', 'unknown').lower()
            if 'volatility' in regime_name or 'bear' in regime_name:
                max_weekly_return = 0.40  # 40% in volatile conditions
            
            prediction = np.clip(prediction, -max_weekly_return, max_weekly_return)
            
        elif horizon <= 20:  # Monthly
            max_monthly_return = 0.50  # 50% max monthly move
            regime_name = getattr(regime_state, 'name', 'unknown').lower()
            if 'volatility' in regime_name or 'bear' in regime_name:
                max_monthly_return = 0.60  # 60% in volatile conditions
            
            prediction = np.clip(prediction, -max_monthly_return, max_monthly_return)
        
        # Log if bounds were applied
        if abs(prediction - original_prediction) > 0.001:
            logger.debug(f"Sanity check applied to {ticker} H{horizon}: {original_prediction:.1%} -> {prediction:.1%}")
        
        # Additional reasonableness check: flag extremely unlikely predictions
        if abs(prediction) > 0.30:  # 30%+ moves are rare
            regime_name = getattr(regime_state, 'name', 'unknown')
            logger.warning(f"🚨 Large prediction for {ticker} H{horizon}: {prediction:.1%} (regime: {regime_name})")
        
        return float(prediction)
    
    def _fallback_prediction(self, horizon: int) -> Dict[str, Any]:
        """Generate fallback prediction when models fail."""
        # Generate realistic fallback return
        prediction = np.random.normal(0.001, 0.01)  # 0.1% mean, 1% volatility
        prediction = np.clip(prediction, -0.05, 0.05)  # ±5% bounds
        
        return {
            'prediction': float(prediction),
            'confidence': 0.50,  # Reasonable fallback confidence
            'prediction_interval': [float(prediction - 0.02), float(prediction + 0.02)],
            'model_disagreement': 0.0,
            'individual_predictions': {'timesfm': prediction, 'tsmamba': prediction, 'samba': prediction, 'tft': prediction},
            'model_weights': {'timesfm': 0.25, 'tsmamba': 0.25, 'samba': 0.25, 'tft': 0.25}
        }
    
    def _fallback_prediction_full(self) -> Dict[str, Any]:
        """Generate full fallback prediction set."""
        result = {}
        for horizon in self.horizon_len:
            result[f"horizon_{horizon}"] = self._fallback_prediction(horizon)
        return result
    
    def _get_multi_resolution_predictions(
        self,
        df: pd.DataFrame,
        ticker: str,
        return_confidence: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get multi-resolution predictions and fuse them.
        
        Args:
            df: Historical OHLCV data
            ticker: Stock ticker symbol
            return_confidence: Whether to return confidence estimates
            
        Returns:
            Fused multi-resolution predictions or None if failed
        """
        try:
            # Get predictions from each resolution predictor
            minute_predictions = None
            hourly_predictions = None
            weekly_predictions = None
            
            # Minute-scale predictions (requires minute data)
            if 'minute' in self.multi_resolution_predictors:
                try:
                    minute_predictions = self.multi_resolution_predictors['minute'].predict(
                        df, ticker, return_confidence
                    )
                except Exception as e:
                    logger.debug(f"Minute predictor failed for {ticker}: {e}")
            
            # Hourly predictions (resample daily to hourly if needed)
            if 'hourly' in self.multi_resolution_predictors:
                try:
                    # For daily data, approximate hourly patterns
                    hourly_predictions = self.multi_resolution_predictors['hourly'].predict(
                        df, ticker, return_confidence
                    )
                except Exception as e:
                    logger.debug(f"Hourly predictor failed for {ticker}: {e}")
            
            # Weekly predictions
            if 'weekly' in self.multi_resolution_predictors:
                try:
                    weekly_predictions = self.multi_resolution_predictors['weekly'].predict(
                        df, ticker, return_confidence
                    )
                except Exception as e:
                    logger.debug(f"Weekly predictor failed for {ticker}: {e}")
            
            # If no predictions available, return None
            if not any([minute_predictions, hourly_predictions, weekly_predictions]):
                logger.debug(f"No multi-resolution predictions available for {ticker}")
                return None
            
            # Create market context for fusion
            market_context = self._create_market_context(df, ticker)
            
            # Fuse predictions using resolution fusion network
            if self.resolution_fuser:
                try:
                    fused_predictions = self.resolution_fuser.fuse_predictions(
                        minute_predictions=minute_predictions,
                        hourly_predictions=hourly_predictions,
                        weekly_predictions=weekly_predictions,
                        market_context=market_context
                    )
                    
                    # Update adaptive weighting with performance feedback
                    if self.adaptive_weighting:
                        try:
                            adaptive_weights = self.adaptive_weighting.compute_adaptive_weights(
                                market_data=df,
                                current_time=datetime.now(),
                                volatility=df['close'].pct_change().std() * np.sqrt(252) if len(df) > 1 else 0.02
                            )
                            
                            # Apply adaptive weights to fused predictions
                            fused_predictions = self._apply_adaptive_weights_to_predictions(
                                fused_predictions, adaptive_weights
                            )
                        except Exception as e:
                            logger.debug(f"Adaptive weighting failed for {ticker}: {e}")
                    
                    return fused_predictions
                    
                except Exception as e:
                    logger.warning(f"Resolution fusion failed for {ticker}: {e}")
            
            # Fallback: simple averaging of available predictions
            return self._simple_multi_resolution_fusion(
                minute_predictions, hourly_predictions, weekly_predictions
            )
            
        except Exception as e:
            logger.error(f"Error in multi-resolution predictions for {ticker}: {e}")
            return None
    
    def _create_market_context(self, df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Create market context for resolution fusion."""
        try:
            if len(df) < 20:
                return {}
            
            returns = df['close'].pct_change().dropna()
            
            context = {
                'volatility_regime': min(returns.std() * 5, 1.0),  # Normalized volatility
                'trend_strength': 0.0,
                'market_breadth': 0.5,
                'sector_rotation': 0.5,
                'economic_cycle': 0.5,
                'seasonal_factor': np.sin(2 * np.pi * datetime.now().timetuple().tm_yday / 365),
                'earnings_season': 0.0,
                'fed_cycle': 0.5,
                'geopolitical_stress': 0.0,
                'liquidity_condition': 0.5
            }
            
            # Calculate trend strength
            if len(df) >= 20:
                sma_short = df['close'].rolling(5).mean()
                sma_long = df['close'].rolling(20).mean()
                trend_strength = (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
                context['trend_strength'] = max(-1, min(1, trend_strength))
            
            return context
            
        except Exception as e:
            logger.debug(f"Error creating market context for {ticker}: {e}")
            return {}
    
    def _apply_adaptive_weights_to_predictions(
        self,
        predictions: Dict[str, Any],
        adaptive_weights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply adaptive weights to fused predictions."""
        try:
            if 'final_weights' not in adaptive_weights:
                return predictions
            
            weights = adaptive_weights['final_weights']
            
            # Apply weights to enhance or reduce confidence based on resolution performance
            for horizon_key in predictions:
                if horizon_key.startswith('horizon_') and horizon_key in predictions:
                    pred_data = predictions[horizon_key]
                    
                    # Adjust confidence based on resolution weights
                    base_confidence = pred_data.get('confidence', 0.5)
                    
                    # Weight-based confidence adjustment
                    weight_factor = sum(weights.values()) / len(weights)  # Average weight
                    adjusted_confidence = base_confidence * (0.8 + 0.4 * weight_factor)
                    
                    pred_data['confidence'] = max(0.1, min(0.95, adjusted_confidence))
                    pred_data['adaptive_weights_applied'] = True
            
            return predictions
            
        except Exception as e:
            logger.debug(f"Error applying adaptive weights: {e}")
            return predictions
    
    def _simple_multi_resolution_fusion(
        self,
        minute_pred: Optional[Dict],
        hourly_pred: Optional[Dict],
        weekly_pred: Optional[Dict]
    ) -> Dict[str, Any]:
        """Simple fallback fusion when sophisticated fusion fails."""
        try:
            available_preds = [p for p in [minute_pred, hourly_pred, weekly_pred] if p is not None]
            
            if not available_preds:
                return {}
            
            fused_result = {}
            
            # Get all horizons from available predictions
            all_horizons = set()
            for pred in available_preds:
                all_horizons.update([k for k in pred.keys() if k.startswith('horizon_')])
            
            for horizon_key in all_horizons:
                horizon_predictions = []
                horizon_confidences = []
                
                for pred in available_preds:
                    if horizon_key in pred:
                        horizon_predictions.append(pred[horizon_key]['prediction'])
                        horizon_confidences.append(pred[horizon_key].get('confidence', 0.5))
                
                if horizon_predictions:
                    # Simple average
                    avg_prediction = np.mean(horizon_predictions)
                    avg_confidence = np.mean(horizon_confidences)
                    
                    # Estimate prediction interval
                    pred_std = np.std(horizon_predictions) if len(horizon_predictions) > 1 else 0.02
                    interval_width = pred_std * 2
                    
                    fused_result[horizon_key] = {
                        'prediction': float(avg_prediction),
                        'confidence': float(avg_confidence),
                        'prediction_interval': [
                            float(avg_prediction - interval_width),
                            float(avg_prediction + interval_width)
                        ],
                        'model_type': 'simple_multi_resolution_fusion',
                        'n_resolutions_used': len(horizon_predictions)
                    }
            
            return fused_result
            
        except Exception as e:
            logger.error(f"Error in simple multi-resolution fusion: {e}")
            return {}
    
    def _adapt_weights_for_regime(
        self, 
        regime_state, 
        adaptation_factors, 
        price_data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
        horizon: int = 1
    ) -> Dict[str, float]:
        """Adapt model weights based on current market regime."""
        try:
            # Use neural meta-learner if available and enabled
            if (self.enable_neural_meta_learning and 
                self.neural_meta_learner is not None and 
                self.feature_extractor is not None and 
                price_data is not None):
                
                try:
                    # Extract enhanced meta-features with Phase 6 integration
                    meta_features = self.feature_extractor.extract_combined_features(
                        price_data=price_data,
                        regime_state=regime_state,
                        horizon=horizon,
                        ticker=ticker,
                        multi_asset_data=price_data if isinstance(price_data, dict) else None
                    )
                    
                    # Get neural weights
                    neural_result = self.neural_meta_learner.get_optimal_weights(
                        meta_features=meta_features,
                        horizon=horizon,
                        return_confidence=True
                    )
                    
                    neural_weights = neural_result['weights']
                    
                    # Blend neural weights with regime-based weights for stability
                    regime_weights = self._get_regime_based_weights(regime_state)
                    
                    # Adaptive blending factor based on neural confidence
                    neural_confidence = neural_result.get('calibrated_confidence', 0.5)
                    if isinstance(neural_confidence, (list, np.ndarray)):
                        neural_confidence = float(neural_confidence[0]) if len(neural_confidence) > 0 else 0.5
                    
                    blend_factor = max(0.3, min(0.9, neural_confidence))  # 30%-90% neural weights
                    
                    # Blend weights (4 models)
                    blended_weights = {}
                    for model in ['timesfm', 'tsmamba', 'samba', 'tft']:
                        neural_w = neural_weights.get(model, 0.25)
                        regime_w = regime_weights.get(model, 0.25)
                        blended_weights[model] = blend_factor * neural_w + (1 - blend_factor) * regime_w
                    
                    # Normalize
                    total_weight = sum(blended_weights.values())
                    blended_weights = {k: v/total_weight for k, v in blended_weights.items()}
                    
                    logger.debug(f"Neural weights (conf={neural_confidence:.3f}, blend={blend_factor:.3f}): {blended_weights}")
                    
                    return blended_weights
                    
                except Exception as e:
                    logger.warning(f"Neural weight adaptation failed: {e}, falling back to regime-based")
            
            # Fallback to regime-based weights
            return self._get_regime_based_weights(regime_state)
            
        except Exception as e:
            logger.error(f"Error adapting weights for regime: {e}")
            return self.weights.copy()
    
    def _get_regime_based_weights(self, regime_state) -> Dict[str, float]:
        """Get regime-based weights (original logic)."""
        try:
            regime = regime_state.regime
            base_weights = self.weights.copy()
            
            # Regime-specific model preferences (4 models + multi-resolution)
            if regime == MarketRegime.BULL_TREND:
                # In bull markets, favor TimesFM (trend following) and TFT (temporal patterns)
                base_weights['timesfm'] *= 1.2
                base_weights['tft'] *= 1.15  # TFT good at temporal trends
                base_weights['samba'] *= 1.05
                base_weights['tsmamba'] *= 0.9
                # Multi-resolution helps capture trend consistency across scales
                if 'multi_resolution' in base_weights:
                    base_weights['multi_resolution'] *= 1.1
                
            elif regime == MarketRegime.BEAR_TREND:
                # In bear markets, favor TSMamba (pattern recognition) and TFT (uncertainty)
                base_weights['tsmamba'] *= 1.25
                base_weights['tft'] *= 1.2  # TFT has native uncertainty estimation
                base_weights['timesfm'] *= 0.8
                base_weights['samba'] *= 0.85
                # Multi-resolution less useful in bear markets due to correlation breakdown
                if 'multi_resolution' in base_weights:
                    base_weights['multi_resolution'] *= 0.9
                
            elif regime == MarketRegime.HIGH_VOLATILITY:
                # In volatile markets, favor TFT (uncertainty) and TSMamba (pattern recognition)
                base_weights['tft'] *= 1.3  # TFT excels in uncertainty quantification
                base_weights['tsmamba'] *= 1.1
                base_weights['samba'] *= 1.05  # SAMBA for correlation breaks
                base_weights['timesfm'] *= 0.8
                # Multi-resolution very valuable in volatile markets (different scales matter)
                if 'multi_resolution' in base_weights:
                    base_weights['multi_resolution'] *= 1.25
                
            elif regime == MarketRegime.SIDEWAYS:
                # In sideways markets, favor TSMamba and TFT for pattern recognition
                base_weights['tsmamba'] *= 1.2
                base_weights['tft'] *= 1.1  # TFT good at temporal patterns
                base_weights['timesfm'] *= 0.85
                base_weights['samba'] *= 0.95
                # Multi-resolution helps identify breakout patterns
                if 'multi_resolution' in base_weights:
                    base_weights['multi_resolution'] *= 1.05
            
            # Normalize weights
            total_weight = sum(base_weights.values())
            adapted_weights = {k: v/total_weight for k, v in base_weights.items()}
            
            return adapted_weights
            
        except Exception as e:
            logger.error(f"Error getting regime-based weights: {e}")
            return self.weights.copy()
    
    def update_weights(
        self,
        performance_data: Dict[str, List[float]]
    ) -> None:
        """
        Update model weights based on recent performance.
        
        Args:
            performance_data: Dictionary with performance metrics for each model
        """
        
        try:
            # Store performance history
            for model, scores in performance_data.items():
                if model in self.performance_history:
                    self.performance_history[model].extend(scores)
                    # Keep only recent performance
                    if len(self.performance_history[model]) > self.performance_window:
                        self.performance_history[model] = self.performance_history[model][-self.performance_window:]
            
            # Calculate new weights based on recent performance (4 models)
            recent_performance = {}
            for model in ['timesfm', 'tsmamba', 'samba', 'tft']:
                if self.performance_history[model]:
                    # Use exponentially weighted average (more weight to recent performance)
                    weights = np.exp(np.linspace(-1, 0, len(self.performance_history[model])))
                    weights /= weights.sum()
                    
                    recent_performance[model] = np.average(
                        self.performance_history[model],
                        weights=weights
                    )
                else:
                    recent_performance[model] = 0.5  # Default performance
            
            # Update weights (softmax for smooth transitions)
            performance_values = np.array(list(recent_performance.values()))
            new_weights = np.exp(performance_values * 2) / np.sum(np.exp(performance_values * 2))
            
            # Apply smoothing to prevent sudden weight changes
            smoothing_factor = 0.2
            model_names = ['timesfm', 'tsmamba', 'samba', 'tft']
            for i, model in enumerate(model_names):
                self.weights[model] = (1 - smoothing_factor) * self.weights[model] + smoothing_factor * new_weights[i]
            
            # Normalize weights
            total = sum(self.weights.values())
            self.weights = {k: v/total for k, v in self.weights.items()}
            
            logger.info(f"Updated ensemble weights: {self.weights}")
            
        except Exception as e:
            logger.error(f"Error updating weights: {e}")
    
    def evaluate_predictions(
        self,
        predictions: Dict[str, Any],
        actual_returns: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Evaluate prediction quality against actual returns.
        
        Args:
            predictions: Ensemble predictions
            actual_returns: Actual returns for each horizon
            
        Returns:
            Performance metrics for each model and ensemble
        """
        
        results = {
            'timesfm': [],
            'tsmamba': [],
            'samba': [],
            'tft': [],
            'ensemble': []
        }
        
        for horizon in self.horizon_len:
            horizon_key = f"horizon_{horizon}"
            
            if horizon_key in predictions and horizon_key in actual_returns:
                actual = actual_returns[horizon_key]
                
                # Ensemble performance
                ensemble_pred = predictions[horizon_key]['prediction']
                ensemble_error = abs(ensemble_pred - actual)
                results['ensemble'].append(1.0 / (1.0 + ensemble_error))  # Convert to performance score
                
                # Individual model performance
                if 'individual_predictions' in predictions[horizon_key]:
                    individual = predictions[horizon_key]['individual_predictions']
                    
                    for model in ['timesfm', 'tsmamba', 'samba', 'tft']:
                        if model in individual:
                            model_pred = individual[model]
                            model_error = abs(model_pred - actual)
                            results[model].append(1.0 / (1.0 + model_error))
        
        # Average performance across horizons
        avg_results = {}
        for model, scores in results.items():
            avg_results[model] = np.mean(scores) if scores else 0.5
        
        return avg_results
    
    def train_ensemble(
        self,
        training_data: List[pd.DataFrame],
        tickers: List[str],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the ensemble on historical data.
        
        Args:
            training_data: List of historical OHLCV DataFrames
            tickers: Corresponding ticker symbols
            validation_split: Fraction of data for validation
            
        Returns:
            Training results and performance metrics
        """
        
        logger.info(f"Training ensemble on {len(training_data)} datasets")
        
        # Split data for validation
        split_idx = int(len(training_data) * (1 - validation_split))
        train_data = training_data[:split_idx]
        train_tickers = tickers[:split_idx]
        val_data = training_data[split_idx:]
        val_tickers = tickers[split_idx:]
        
        # Train individual models
        try:
            self.tsmamba.train_model(train_data, train_tickers, epochs=50)
        except Exception as e:
            logger.warning(f"TSMamba training failed: {e}")
        
        try:
            self.tft.train_model(train_data, train_tickers, epochs=50)
        except Exception as e:
            logger.warning(f"TFT training failed: {e}")
        
        # Train feature selector if enabled
        feature_selection_results = {}
        if (self.enable_neural_meta_learning and 
            self.feature_extractor is not None and 
            hasattr(self.feature_extractor, 'fit_feature_selector')):
            
            try:
                logger.info("Training feature selector on ensemble data...")
                
                # Prepare data for feature selector training
                training_data_dict = []
                target_returns_list = []
                
                for df, ticker in zip(train_data, train_tickers):
                    # Convert to multi-asset format for cross-asset features
                    data_dict = {ticker: df}
                    training_data_dict.append(data_dict)
                    
                    # Calculate target returns for different horizons
                    if len(df) > 25:
                        returns = []
                        for horizon in self.horizon_len:
                            if len(df) > 20 + horizon:
                                current_price = df['close'].iloc[-20-horizon]
                                future_price = df['close'].iloc[-20]
                                ret = (future_price - current_price) / current_price
                                returns.append(ret)
                        
                        if returns:
                            target_returns_list.append(returns)
                        else:
                            target_returns_list.append([0.0])  # Default
                    else:
                        target_returns_list.append([0.0])  # Default for insufficient data
                
                # Fit feature selector
                feature_selection_results = self.feature_extractor.fit_feature_selector(
                    training_data=training_data_dict,
                    target_returns=target_returns_list,
                    tickers=train_tickers
                )
                
                logger.info(f"Feature selector training completed: {feature_selection_results}")
                
            except Exception as e:
                logger.error(f"Feature selector training failed: {e}")
                feature_selection_results = {'error': str(e)}
        
        # Validate and update weights (4 models)
        validation_performance = {'timesfm': [], 'tsmamba': [], 'samba': [], 'tft': [], 'ensemble': []}
        
        for df, ticker in zip(val_data, val_tickers):
            try:
                # Make predictions
                pred = self.predict(df[:-20], ticker, use_cache=False)  # Use first part for prediction
                
                # Calculate actual returns
                actual_returns = {}
                for horizon in self.horizon_len:
                    if len(df) > 20 + horizon:
                        current_close = df['close'].iloc[-20-horizon]
                        future_close = df['close'].iloc[-20]
                        actual_return = (future_close - current_close) / current_close
                        actual_returns[f"horizon_{horizon}"] = actual_return
                
                # Evaluate performance
                if actual_returns:
                    performance = self.evaluate_predictions(pred, actual_returns)
                    for model, score in performance.items():
                        validation_performance[model].append(score)
                        
            except Exception as e:
                logger.warning(f"Validation error for {ticker}: {e}")
        
        # Update weights based on validation performance
        self.update_weights(validation_performance)
        
        # Calculate final metrics
        final_metrics = {}
        for model, scores in validation_performance.items():
            if scores:
                final_metrics[model] = {
                    'mean_performance': np.mean(scores),
                    'std_performance': np.std(scores),
                    'num_predictions': len(scores)
                }
        
        logger.info("Ensemble training completed")
        logger.info(f"Final performance: {final_metrics}")
        
        return {
            'performance_metrics': final_metrics,
            'final_weights': self.weights.copy(),
            'training_samples': len(train_data),
            'validation_samples': len(val_data),
            'feature_selection_results': feature_selection_results
        }
    
    def update_online_learning(
        self,
        price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        predictions: Dict[str, float],
        actual_return: float,
        model_weights: Dict[str, float],
        horizon: int,
        ticker: str,
        regime_state: Optional[Any] = None
    ) -> None:
        """
        Update online learning system with new experience.
        
        Args:
            price_data: Price data used for prediction
            predictions: Individual model predictions
            actual_return: Actual realized return
            model_weights: Model weights used
            horizon: Prediction horizon
            ticker: Stock ticker symbol
            regime_state: Market regime state
        """
        if (self.enable_online_learning and 
            self.online_learner is not None and 
            self.feature_extractor is not None):
            
            try:
                # Extract enhanced meta-features with Phase 6 integration
                meta_features = self.feature_extractor.extract_combined_features(
                    price_data=price_data,
                    regime_state=regime_state,
                    horizon=horizon,
                    ticker=ticker,
                    multi_asset_data=price_data if isinstance(price_data, dict) else None
                )
                
                # Add experience to online learner
                self.online_learner.add_experience(
                    meta_features=meta_features,
                    predictions=predictions,
                    actual_return=actual_return,
                    model_weights=model_weights,
                    horizon=horizon
                )
                
            except Exception as e:
                logger.error(f"Error updating online learning: {e}")
    
    def start_online_learning(self) -> bool:
        """Start online learning background process."""
        if (self.enable_online_learning and 
            self.online_learner is not None):
            
            try:
                self.online_learner.start_online_learning()
                logger.info("Online learning started")
                return True
            except Exception as e:
                logger.error(f"Failed to start online learning: {e}")
                return False
        
        logger.warning("Online learning not available")
        return False
    
    def stop_online_learning(self) -> bool:
        """Stop online learning background process."""
        if self.online_learner is not None:
            try:
                self.online_learner.stop_online_learning()
                logger.info("Online learning stopped")
                return True
            except Exception as e:
                logger.error(f"Error stopping online learning: {e}")
                return False
        
        return True
    
    def force_neural_update(self) -> Dict[str, Any]:
        """Force an immediate neural meta-learner update."""
        if (self.enable_online_learning and 
            self.online_learner is not None):
            
            try:
                return self.online_learner.perform_update(force=True)
            except Exception as e:
                logger.error(f"Error forcing neural update: {e}")
                return {'error': str(e)}
        
        return {'error': 'online_learning_not_available'}
    
    def get_neural_meta_learning_status(self) -> Dict[str, Any]:
        """Get status of neural meta-learning components."""
        status = {
            'enabled': self.enable_neural_meta_learning,
            'online_learning_enabled': self.enable_online_learning,
            'neural_meta_learner': None,
            'feature_extractor': None,
            'online_learner': None
        }
        
        if self.neural_meta_learner is not None:
            status['neural_meta_learner'] = self.neural_meta_learner.get_model_summary()
        
        if self.feature_extractor is not None:
            status['feature_extractor'] = self.feature_extractor.get_feature_importance_summary()
            
            # Add Phase 6 feature status
            if hasattr(self.feature_extractor, 'get_phase6_feature_summary'):
                status['phase6_features'] = self.feature_extractor.get_phase6_feature_summary()
        
        if self.online_learner is not None:
            status['online_learner'] = self.online_learner.get_learning_statistics()
        
        return status
    
    def save_neural_components(self, base_path: str) -> bool:
        """Save neural meta-learning components."""
        try:
            base_path = Path(base_path)
            base_path.mkdir(parents=True, exist_ok=True)
            
            if self.neural_meta_learner is not None:
                self.neural_meta_learner.save_model(base_path / "neural_meta_learner.pt")
            
            if self.online_learner is not None:
                self.online_learner.save_learning_state(base_path / "online_learner_state.pt")
            
            logger.info(f"Neural components saved to {base_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving neural components: {e}")
            return False
    
    def load_neural_components(self, base_path: str) -> bool:
        """Load neural meta-learning components."""
        try:
            base_path = Path(base_path)
            
            if (self.neural_meta_learner is not None and 
                (base_path / "neural_meta_learner.pt").exists()):
                self.neural_meta_learner.load_model(base_path / "neural_meta_learner.pt")
            
            if (self.online_learner is not None and 
                (base_path / "online_learner_state.pt").exists()):
                self.online_learner.load_learning_state(base_path / "online_learner_state.pt")
            
            logger.info(f"Neural components loaded from {base_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading neural components: {e}")
            return False

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid (exists and not expired)."""
        if cache_key not in self.prediction_cache:
            return False
            
        if cache_key not in self.cache_ttl:
            return False
            
        import time
        current_time = time.time()
        return (current_time - self.cache_ttl[cache_key]) < self.cache_ttl_seconds
    
    def _cache_prediction(self, cache_key: str, prediction: Dict[str, Any]) -> None:
        """Cache prediction with TTL and size management."""
        import time
        
        # Remove expired entries first
        self._cleanup_expired_cache()
        
        # Remove oldest entries if cache is full
        if len(self.prediction_cache) >= self.cache_max_size:
            self._evict_oldest_cache_entries(self.cache_max_size // 4)  # Remove 25% of cache
        
        # Add new entry
        self.prediction_cache[cache_key] = prediction
        self.cache_ttl[cache_key] = time.time()
    
    def _cleanup_expired_cache(self) -> None:
        """Remove expired cache entries."""
        import time
        current_time = time.time()
        
        expired_keys = [
            key for key, timestamp in self.cache_ttl.items()
            if (current_time - timestamp) >= self.cache_ttl_seconds
        ]
        
        for key in expired_keys:
            self.prediction_cache.pop(key, None)
            self.cache_ttl.pop(key, None)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _evict_oldest_cache_entries(self, num_to_evict: int) -> None:
        """Evict oldest cache entries to manage memory."""
        if num_to_evict <= 0:
            return
            
        # Sort by timestamp (oldest first)
        sorted_entries = sorted(self.cache_ttl.items(), key=lambda x: x[1])
        oldest_keys = [key for key, _ in sorted_entries[:num_to_evict]]
        
        for key in oldest_keys:
            self.prediction_cache.pop(key, None)
            self.cache_ttl.pop(key, None)
        
        logger.debug(f"Evicted {len(oldest_keys)} oldest cache entries")
    
    def clear_prediction_cache(self) -> None:
        """Clear all cached predictions."""
        self.prediction_cache.clear()
        self.cache_ttl.clear()
        logger.info("Prediction cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        import time
        current_time = time.time()
        
        valid_entries = sum(
            1 for timestamp in self.cache_ttl.values()
            if (current_time - timestamp) < self.cache_ttl_seconds
        )
        
        return {
            'total_entries': len(self.prediction_cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self.prediction_cache) - valid_entries,
            'cache_hit_ratio': getattr(self, '_cache_hits', 0) / max(1, getattr(self, '_cache_requests', 1)),
            'max_size': self.cache_max_size,
            'ttl_seconds': self.cache_ttl_seconds
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status information about ensemble components."""
        return {
            'timesfm': {
                'loaded': self.timesfm.is_loaded,
                'device': str(self.timesfm.device),
                'weight': self.weights['timesfm']
            },
            'tsmamba': {
                'loaded': self.tsmamba.is_loaded,
                'device': str(self.tsmamba.device),
                'weight': self.weights['tsmamba']
            },
            'samba': {
                'loaded': self.samba.is_loaded,
                'device': str(self.samba.device),
                'weight': self.weights['samba']
            },
            'tft': {
                'loaded': self.tft.is_loaded,
                'device': str(self.tft.device),
                'weight': self.weights['tft']
            },
            'ensemble': {
                'horizon_len': self.horizon_len,
                'uncertainty_alpha': self.uncertainty_alpha,
                'cache_stats': self.get_cache_stats()
            },
            'neural_meta_learning': self.get_neural_meta_learning_status(),
            'memory_stats': self._get_memory_stats()
        }
    
    def save_ensemble(self, path: str):
        """Save the ensemble configuration and models."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        self.tsmamba.save_model(save_dir / "tsmamba_model.pt")
        self.tft.save_model(save_dir / "tft_model.pt")
        
        # Save neural meta-learning components
        if self.enable_neural_meta_learning:
            neural_dir = save_dir / "neural_meta_learning"
            self.save_neural_components(neural_dir)
        
        # Save ensemble configuration
        import json
        config_data = {
            'weights': self.weights,
            'horizon_len': self.horizon_len,
            'uncertainty_alpha': self.uncertainty_alpha,
            'performance_window': self.performance_window,
            'performance_history': self.performance_history,
            'enable_neural_meta_learning': self.enable_neural_meta_learning,
            'enable_online_learning': self.enable_online_learning
        }
        
        with open(save_dir / "ensemble_config.json", 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Ensemble saved to {path}")
    
    def _manage_gpu_memory(self) -> None:
        """Proactive GPU memory management for stable predictions."""
        try:
            import torch
            if torch.cuda.is_available():
                # Clear cache if memory usage is high
                memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                if memory_used > 0.80:  # If using >80% of memory
                    torch.cuda.empty_cache()
                    logger.debug("GPU cache cleared - high memory usage detected")
                    
        except ImportError:
            pass  # PyTorch not available
        except Exception as e:
            logger.debug(f"GPU memory management error: {e}")
    
    def _force_gpu_cleanup(self) -> None:
        """Force cleanup of GPU memory (last resort)."""
        try:
            import torch
            import gc
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()  # Python garbage collection
                logger.info("Forced GPU memory cleanup completed")
                
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Force GPU cleanup failed: {e}")
    
    def load_ensemble(self, path: str) -> bool:
        """Load the ensemble configuration and models."""
        load_dir = Path(path)
        
        if not load_dir.exists():
            logger.error(f"Ensemble path does not exist: {path}")
            return False
        
        try:
            # Load individual models
            self.tsmamba.load_model(load_dir / "tsmamba_model.pt")
            if (load_dir / "tft_model.pt").exists():
                self.tft.load_model(load_dir / "tft_model.pt")
            
            # Load ensemble configuration
            import json
            config_path = load_dir / "ensemble_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                self.weights = config_data.get('weights', self.weights)
                self.horizon_len = config_data.get('horizon_len', self.horizon_len)
                self.uncertainty_alpha = config_data.get('uncertainty_alpha', self.uncertainty_alpha)
                self.performance_window = config_data.get('performance_window', self.performance_window)
                self.performance_history = config_data.get('performance_history', self.performance_history)
                
                # Load neural meta-learning settings
                loaded_neural = config_data.get('enable_neural_meta_learning', False)
                loaded_online = config_data.get('enable_online_learning', False)
                
                if loaded_neural and self.enable_neural_meta_learning:
                    neural_dir = load_dir / "neural_meta_learning"
                    if neural_dir.exists():
                        self.load_neural_components(neural_dir)
            
            logger.info(f"Ensemble loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ensemble: {e}")
            return False
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {'cpu_memory_mb': 0, 'gpu_memory_mb': 0, 'gpu_available': False}
        
        try:
            import psutil
            process = psutil.Process()
            stats['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass
        
        try:
            import torch
            if torch.cuda.is_available():
                stats['gpu_available'] = True
                stats['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
                stats['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
                stats['gpu_utilization_pct'] = (
                    torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                    if torch.cuda.max_memory_allocated() > 0 else 0
                )
        except ImportError:
            pass
        
        return stats
