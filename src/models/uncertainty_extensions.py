"""Enhanced uncertainty quantification extensions for Phase 3 implementation."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any, Callable
from datetime import datetime
import warnings
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

try:
    from mapie.regression import MapieRegressor
    from mapie.time_series_regression import MapieTimeSeriesRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False
    logger.warning("MAPIE not available. Running in mock mode for uncertainty quantification.")

from .uncertainty import ConformalPredictor, TradingUncertaintyFilter
from .regime_detector import HiddenMarkovRegimeDetector, MarketRegime, RegimeState
from ..config.settings import config


class RegimeAdaptiveMapie(ConformalPredictor):
    """MAPIE with regime-adaptive alpha parameters for improved uncertainty quantification."""
    
    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        regime_detector: Optional[HiddenMarkovRegimeDetector] = None,
        regime_alpha_mapping: Optional[Dict[str, float]] = None,
        alpha_adaptation_rate: float = 0.05,
        coverage_history_window: int = 20,
        **kwargs
    ):
        """Initialize regime-adaptive MAPIE.
        
        Args:
            base_estimator: Base ML model
            regime_detector: Regime detection system
            regime_alpha_mapping: Custom alpha values for each regime
            alpha_adaptation_rate: Rate of alpha adaptation based on coverage
            coverage_history_window: Window for tracking coverage performance
        """
        super().__init__(base_estimator=base_estimator, **kwargs)
        
        self.regime_detector = regime_detector or HiddenMarkovRegimeDetector()
        self.alpha_adaptation_rate = alpha_adaptation_rate
        self.coverage_history_window = coverage_history_window
        
        # Regime-specific alpha values (research-validated)
        self.regime_alpha_mapping = regime_alpha_mapping or {
            MarketRegime.BULL_TREND.value: 0.20,      # 80% intervals in bull markets (more confident)
            MarketRegime.BEAR_TREND.value: 0.35,      # 65% intervals in bear markets (less confident)
            MarketRegime.HIGH_VOLATILITY.value: 0.40, # 60% intervals in volatile periods
            MarketRegime.SIDEWAYS.value: 0.25,        # 75% intervals in sideways markets
            MarketRegime.TRANSITION.value: 0.30       # 70% intervals during transitions
        }
        
        # Coverage tracking by regime
        self.regime_coverage_history = {regime: [] for regime in self.regime_alpha_mapping.keys()}
        self.current_regime_alpha = self.alpha
        
        logger.info(f"RegimeAdaptiveMapie initialized with regime-specific alphas: {self.regime_alpha_mapping}")
    
    def get_regime_adaptive_alpha(
        self,
        market_data: Dict[str, pd.DataFrame],
        current_regime: Optional[RegimeState] = None
    ) -> float:
        """Get regime-adaptive alpha parameter.
        
        Args:
            market_data: Market data for regime detection
            current_regime: Pre-detected regime (optional)
            
        Returns:
            Adapted alpha value
        """
        try:
            # Detect current regime if not provided
            if current_regime is None:
                current_regime = self.regime_detector.detect_regime(market_data)
            
            # Get base alpha for current regime
            base_alpha = self.regime_alpha_mapping.get(
                current_regime.regime.value,
                self.alpha  # Fallback to default
            )
            
            # Apply confidence-based adjustment
            confidence_factor = current_regime.confidence
            confidence_adjustment = (confidence_factor - 0.5) * 0.1  # ±5% max adjustment
            
            # Get coverage-based adaptation
            coverage_adjustment = self._get_coverage_based_adjustment(current_regime.regime.value)
            
            # Combine adjustments
            adapted_alpha = base_alpha - confidence_adjustment + coverage_adjustment
            
            # Ensure alpha stays within reasonable bounds
            adapted_alpha = np.clip(adapted_alpha, 0.05, 0.45)
            
            self.current_regime_alpha = adapted_alpha
            
            logger.debug(f"Adapted alpha for {current_regime.regime.value}: {adapted_alpha:.3f} "
                        f"(base: {base_alpha:.3f}, confidence_adj: {confidence_adjustment:.3f}, "
                        f"coverage_adj: {coverage_adjustment:.3f})")
            
            return adapted_alpha
            
        except Exception as e:
            logger.error(f"Error in regime adaptive alpha calculation: {e}")
            return self.alpha
    
    def _get_coverage_based_adjustment(self, regime_name: str) -> float:
        """Calculate alpha adjustment based on recent coverage performance."""
        try:
            regime_history = self.regime_coverage_history.get(regime_name, [])
            
            if len(regime_history) < 3:
                return 0.0  # Not enough history
            
            # Calculate recent coverage statistics
            recent_coverage = [entry['empirical_coverage'] for entry in regime_history[-self.coverage_history_window:]]
            avg_coverage = np.mean(recent_coverage)
            target_coverage = 1 - self.regime_alpha_mapping[regime_name]
            
            # Coverage error (positive means over-covering, negative means under-covering)
            coverage_error = avg_coverage - target_coverage
            
            # Adjust alpha to correct coverage (increase alpha if over-covering, decrease if under-covering)
            adjustment = coverage_error * self.alpha_adaptation_rate
            
            return adjustment
            
        except Exception as e:
            logger.warning(f"Error calculating coverage adjustment: {e}")
            return 0.0
    
    def predict_with_regime_adaptation(
        self,
        X: np.ndarray,
        market_data: Dict[str, pd.DataFrame],
        current_regime: Optional[RegimeState] = None
    ) -> Tuple[np.ndarray, np.ndarray, RegimeState]:
        """Generate regime-adapted predictions with uncertainty intervals.
        
        Args:
            X: Input features
            market_data: Market data for regime detection
            current_regime: Pre-detected regime (optional)
            
        Returns:
            Tuple of (predictions, intervals, regime_state)
        """
        try:
            # Get regime-adaptive alpha
            adaptive_alpha = self.get_regime_adaptive_alpha(market_data, current_regime)
            
            # Get current regime for return
            if current_regime is None:
                current_regime = self.regime_detector.detect_regime(market_data)
            
            # Make predictions with adapted alpha
            predictions, intervals = self.predict(X, alpha=adaptive_alpha)
            
            logger.info(f"Regime-adaptive prediction: {current_regime.regime.value} "
                       f"(alpha: {adaptive_alpha:.3f}, confidence: {current_regime.confidence:.3f})")
            
            return predictions, intervals, current_regime
            
        except Exception as e:
            logger.error(f"Error in regime-adaptive prediction: {e}")
            # Fallback to standard prediction
            predictions, intervals = self.predict(X)
            fallback_regime = self.regime_detector._fallback_regime()
            return predictions, intervals, fallback_regime
    
    def validate_coverage(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray,
        intervals: np.ndarray,
        regime_state: RegimeState,
        store_history: bool = True
    ) -> Dict[str, Any]:
        """Enhanced coverage validation with regime tracking.
        
        Args:
            y_true: True values
            predictions: Point predictions
            intervals: Prediction intervals
            regime_state: Current market regime
            store_history: Whether to store coverage history
            
        Returns:
            Comprehensive coverage statistics
        """
        try:
            # Calculate base coverage statistics
            coverage_stats = self.calculate_coverage(
                y_true, predictions, intervals, self.current_regime_alpha
            )
            
            # Add regime-specific information
            regime_info = {
                'regime': regime_state.regime.value,
                'regime_confidence': regime_state.confidence,
                'regime_duration': regime_state.duration,
                'adapted_alpha': self.current_regime_alpha,
                'base_alpha': self.regime_alpha_mapping.get(regime_state.regime.value, self.alpha)
            }
            
            coverage_stats.update(regime_info)
            
            # Calculate regime-specific metrics
            target_coverage = 1 - self.current_regime_alpha
            coverage_stats['regime_target_coverage'] = target_coverage
            coverage_stats['regime_coverage_error'] = abs(coverage_stats['empirical_coverage'] - target_coverage)
            
            # Calculate adaptive efficiency
            coverage_stats['adaptive_efficiency'] = self._calculate_adaptive_efficiency(
                coverage_stats['empirical_coverage'],
                target_coverage,
                coverage_stats['mean_interval_width']
            )
            
            # Store in regime-specific history
            if store_history:
                regime_name = regime_state.regime.value
                if regime_name in self.regime_coverage_history:
                    self.regime_coverage_history[regime_name].append(coverage_stats)
                    
                    # Keep only recent history
                    if len(self.regime_coverage_history[regime_name]) > 50:
                        self.regime_coverage_history[regime_name] = self.regime_coverage_history[regime_name][-50:]
            
            logger.info(f"Coverage validation for {regime_state.regime.value}: "
                       f"{coverage_stats['empirical_coverage']:.3f} (target: {target_coverage:.3f}, "
                       f"error: {coverage_stats['regime_coverage_error']:.3f})")
            
            return coverage_stats
            
        except Exception as e:
            logger.error(f"Error in coverage validation: {e}")
            return {'error': str(e)}
    
    def _calculate_adaptive_efficiency(self, empirical_coverage: float, target_coverage: float, mean_width: float) -> float:
        """Calculate efficiency metric for adaptive coverage."""
        try:
            # Penalize both coverage errors and wide intervals
            coverage_penalty = abs(empirical_coverage - target_coverage) * 2  # Double penalty for coverage errors
            width_penalty = mean_width * 0.5  # Penalty for wide intervals
            
            # Efficiency score (higher is better)
            efficiency = max(0.0, 1.0 - coverage_penalty - width_penalty)
            
            return efficiency
            
        except:
            return 0.0
    
    def get_regime_coverage_summary(self) -> Dict[str, Any]:
        """Get comprehensive coverage summary by regime."""
        summary = {
            'overall_summary': self.get_uncertainty_summary(),
            'regime_summaries': {},
            'adaptive_performance': {}
        }
        
        # Per-regime summaries
        for regime_name, history in self.regime_coverage_history.items():
            if not history:
                continue
                
            recent_history = history[-10:]  # Last 10 evaluations
            
            regime_summary = {
                'n_evaluations': len(history),
                'base_alpha': self.regime_alpha_mapping.get(regime_name, self.alpha),
                'recent_avg_coverage': np.mean([h['empirical_coverage'] for h in recent_history]),
                'recent_avg_width': np.mean([h['mean_interval_width'] for h in recent_history]),
                'recent_avg_efficiency': np.mean([h.get('adaptive_efficiency', 0) for h in recent_history]),
                'coverage_stability': np.std([h['empirical_coverage'] for h in recent_history])
            }
            
            # Check calibration quality
            target_coverage = 1 - regime_summary['base_alpha']
            coverage_error = abs(regime_summary['recent_avg_coverage'] - target_coverage)
            regime_summary['is_well_calibrated'] = coverage_error < 0.05
            regime_summary['calibration_error'] = coverage_error
            
            summary['regime_summaries'][regime_name] = regime_summary
        
        # Overall adaptive performance
        if any(self.regime_coverage_history.values()):
            all_recent = []
            for history in self.regime_coverage_history.values():
                all_recent.extend(history[-5:])  # Last 5 from each regime
            
            if all_recent:
                summary['adaptive_performance'] = {
                    'overall_avg_efficiency': np.mean([h.get('adaptive_efficiency', 0) for h in all_recent]),
                    'adaptation_consistency': np.std([h.get('adaptive_efficiency', 0) for h in all_recent]),
                    'total_evaluations': len(all_recent)
                }
        
        return summary


class MonteCarloDropoutUncertainty(nn.Module):
    """Monte Carlo Dropout for epistemic uncertainty estimation in neural networks."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        output_dim: int = 1,
        dropout_rate: float = 0.2,
        n_samples: int = 100,
        activation: str = 'relu'
    ):
        """Initialize Monte Carlo Dropout network.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output dimensions
            dropout_rate: Dropout probability
            n_samples: Number of MC samples for uncertainty estimation
            activation: Activation function ('relu', 'tanh', 'gelu')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no dropout on final layer for better stability)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"MonteCarloDropout network initialized: {input_dim} -> {hidden_dims} -> {output_dim}")
        logger.info(f"Dropout rate: {dropout_rate}, MC samples: {n_samples}")
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU()
        }
        return activations.get(activation, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        return self.network(x)
    
    def predict_with_uncertainty(
        self,
        X: Union[np.ndarray, torch.Tensor],
        return_samples: bool = False
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Generate predictions with epistemic uncertainty via Monte Carlo Dropout.
        
        Args:
            X: Input features
            return_samples: Whether to return individual MC samples
            
        Returns:
            Dictionary with predictions, uncertainties, and optionally samples
        """
        self.train()  # Enable dropout for uncertainty estimation
        
        # Convert to tensor if needed
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X)
            return_numpy = True
        else:
            X_tensor = X
            return_numpy = False
        
        # Generate MC samples
        samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                sample_pred = self.forward(X_tensor)
                samples.append(sample_pred)
        
        # Stack samples
        mc_samples = torch.stack(samples, dim=0)  # Shape: (n_samples, batch_size, output_dim)
        
        # Calculate statistics
        mean_pred = torch.mean(mc_samples, dim=0)
        epistemic_uncertainty = torch.std(mc_samples, dim=0)
        
        # Calculate prediction intervals (95% by default)
        lower_quantile = torch.quantile(mc_samples, 0.025, dim=0)
        upper_quantile = torch.quantile(mc_samples, 0.975, dim=0)
        
        # Prepare results
        results = {
            'predictions': mean_pred.numpy() if return_numpy else mean_pred,
            'epistemic_uncertainty': epistemic_uncertainty.numpy() if return_numpy else epistemic_uncertainty,
            'prediction_intervals': torch.stack([lower_quantile, upper_quantile], dim=-1).numpy() if return_numpy 
                                  else torch.stack([lower_quantile, upper_quantile], dim=-1),
            'confidence': (1.0 - epistemic_uncertainty).numpy() if return_numpy else (1.0 - epistemic_uncertainty)
        }
        
        if return_samples:
            results['mc_samples'] = mc_samples.numpy() if return_numpy else mc_samples
        
        self.eval()  # Return to eval mode
        
        return results
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train the Monte Carlo Dropout network.
        
        Args:
            X: Training features
            y: Training targets
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            validation_split: Fraction of data for validation
            early_stopping_patience: Early stopping patience
            verbose: Whether to print training progress
            
        Returns:
            Training history
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Split data
        n_val = int(len(X) * validation_split)
        indices = torch.randperm(len(X))
        
        X_train = X_tensor[indices[n_val:]]
        y_train = y_tensor[indices[n_val:]]
        X_val = X_tensor[indices[:n_val]]
        y_val = y_tensor[indices[:n_val]]
        
        # Setup training
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_uncertainty': []}
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_losses = []
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation phase
            self.eval()
            with torch.no_grad():
                val_outputs = self.forward(X_val)
                val_loss = criterion(val_outputs.squeeze(), y_val).item()
                
                # Calculate validation uncertainty
                val_results = self.predict_with_uncertainty(X_val.numpy())
                avg_uncertainty = np.mean(val_results['epistemic_uncertainty'])
            
            # Update history
            train_loss = np.mean(train_losses)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_uncertainty'].append(avg_uncertainty)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Progress logging
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}, "
                           f"val_loss={val_loss:.6f}, val_uncertainty={avg_uncertainty:.6f}")
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        return history


class DeepEnsembleUncertainty:
    """Deep Ensemble for aleatoric and epistemic uncertainty estimation."""
    
    def __init__(
        self,
        input_dim: int,
        n_models: int = 5,
        hidden_dims: List[int] = [256, 128, 64],
        output_dim: int = 2,  # Mean and variance
        model_diversity_reg: float = 0.01,
        ensemble_method: str = 'mean_var'
    ):
        """Initialize Deep Ensemble.
        
        Args:
            input_dim: Number of input features
            n_models: Number of ensemble members
            hidden_dims: Hidden layer dimensions
            output_dim: 2 for mean and variance prediction
            model_diversity_reg: Regularization for model diversity
            ensemble_method: 'mean_var' or 'quantile'
        """
        self.input_dim = input_dim
        self.n_models = n_models
        self.output_dim = output_dim
        self.model_diversity_reg = model_diversity_reg
        self.ensemble_method = ensemble_method
        
        # Create ensemble members
        self.models = []
        for i in range(n_models):
            model = self._create_ensemble_member(input_dim, hidden_dims, output_dim, i)
            self.models.append(model)
        
        self.is_fitted = False
        
        logger.info(f"DeepEnsemble initialized with {n_models} models, diversity_reg={model_diversity_reg}")
    
    def _create_ensemble_member(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        model_id: int
    ) -> nn.Module:
        """Create a single ensemble member with architectural diversity."""
        
        class EnsembleMember(nn.Module):
            def __init__(self, input_dim, hidden_dims, output_dim, model_id):
                super().__init__()
                self.model_id = model_id
                
                # Add architectural diversity
                diversified_hidden = hidden_dims.copy()
                if model_id % 2 == 1:  # Vary depth for odd models
                    diversified_hidden = diversified_hidden + [diversified_hidden[-1] // 2]
                
                # Vary width for different models
                width_multiplier = 0.8 + 0.4 * (model_id / max(1, len(hidden_dims) - 1))
                diversified_hidden = [int(dim * width_multiplier) for dim in diversified_hidden]
                
                # Build network
                layers = []
                prev_dim = input_dim
                
                for i, hidden_dim in enumerate(diversified_hidden):
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU() if model_id % 3 != 0 else nn.GELU(),  # Activation diversity
                        nn.Dropout(0.1 + 0.1 * (model_id % 3) / 2)      # Dropout diversity
                    ])
                    prev_dim = hidden_dim
                
                # Output layer for mean and variance
                layers.append(nn.Linear(prev_dim, output_dim))
                
                self.network = nn.Sequential(*layers)
                
                # Initialize with different strategies for diversity
                self._initialize_weights(model_id)
            
            def _initialize_weights(self, model_id):
                """Initialize with diversity."""
                init_methods = [
                    lambda w: nn.init.kaiming_normal_(w, mode='fan_out'),
                    lambda w: nn.init.xavier_uniform_(w),
                    lambda w: nn.init.kaiming_uniform_(w, mode='fan_in')
                ]
                
                init_method = init_methods[model_id % len(init_methods)]
                
                for module in self.network:
                    if isinstance(module, nn.Linear):
                        init_method(module.weight)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
            
            def forward(self, x):
                output = self.network(x)
                if self.training and output.size(-1) == 2:
                    # During training, output mean and log variance
                    mean, log_var = output.chunk(2, dim=-1)
                    # Ensure log_var is not too negative (for numerical stability)
                    log_var = torch.clamp(log_var, min=-7, max=7)
                    return torch.cat([mean, log_var], dim=-1)
                return output
        
        return EnsembleMember(input_dim, hidden_dims, output_dim, model_id)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train the deep ensemble."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1) if y.ndim == 1 else torch.FloatTensor(y)
        
        # Split data
        n_val = int(len(X) * validation_split)
        indices = torch.randperm(len(X))
        
        X_train = X_tensor[indices[n_val:]]
        y_train = y_tensor[indices[n_val:]]
        X_val = X_tensor[indices[:n_val]]
        y_val = y_tensor[indices[:n_val]]
        
        # Train each model
        history = {'train_losses': [], 'val_losses': [], 'diversity_scores': []}
        
        for i, model in enumerate(self.models):
            if verbose:
                logger.info(f"Training ensemble member {i+1}/{self.n_models}")
            
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            model_train_losses = []
            model_val_losses = []
            
            for epoch in range(epochs):
                # Training
                model.train()
                epoch_train_losses = []
                
                for j in range(0, len(X_train), batch_size):
                    batch_X = X_train[j:j+batch_size]
                    batch_y = y_train[j:j+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    
                    if self.ensemble_method == 'mean_var':
                        # Heteroscedastic loss (predicting mean and variance)
                        mean_pred, log_var = outputs.chunk(2, dim=-1)
                        var = torch.exp(log_var)
                        
                        # Negative log likelihood loss
                        loss = 0.5 * (torch.log(2 * np.pi * var) + (batch_y - mean_pred) ** 2 / var)
                        loss = loss.mean()
                        
                        # Add diversity regularization
                        if self.model_diversity_reg > 0 and i > 0:
                            diversity_loss = self._calculate_diversity_loss(model, batch_X, i)
                            loss = loss + self.model_diversity_reg * diversity_loss
                    else:
                        # Standard MSE loss
                        loss = F.mse_loss(outputs, batch_y)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_train_losses.append(loss.item())
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    if self.ensemble_method == 'mean_var':
                        val_mean, val_log_var = val_outputs.chunk(2, dim=-1)
                        val_var = torch.exp(val_log_var)
                        val_loss = 0.5 * (torch.log(2 * np.pi * val_var) + (y_val - val_mean) ** 2 / val_var)
                        val_loss = val_loss.mean().item()
                    else:
                        val_loss = F.mse_loss(val_outputs, y_val).item()
                
                model_train_losses.append(np.mean(epoch_train_losses))
                model_val_losses.append(val_loss)
                
                if verbose and epoch % 20 == 0:
                    logger.debug(f"Model {i+1} Epoch {epoch}: train_loss={model_train_losses[-1]:.6f}, val_loss={val_loss:.6f}")
            
            history['train_losses'].append(model_train_losses)
            history['val_losses'].append(model_val_losses)
        
        self.is_fitted = True
        
        # Calculate ensemble diversity
        if verbose:
            diversity = self._calculate_ensemble_diversity(X_val)
            history['diversity_scores'] = diversity
            logger.info(f"Ensemble training completed. Final diversity score: {np.mean(diversity):.4f}")
        
        return history
    
    def _calculate_diversity_loss(self, current_model, X, model_idx):
        """Calculate diversity loss to encourage different models to make different predictions."""
        if model_idx == 0:
            return torch.tensor(0.0)
        
        current_pred = current_model(X)
        
        diversity_losses = []
        for j in range(model_idx):
            other_model = self.models[j]
            other_model.eval()
            with torch.no_grad():
                other_pred = other_model(X)
            
            # Negative correlation loss (encourage different predictions)
            correlation = F.cosine_similarity(current_pred.flatten(), other_pred.flatten(), dim=0)
            diversity_loss = correlation ** 2  # Penalize high correlation
            diversity_losses.append(diversity_loss)
        
        return torch.stack(diversity_losses).mean()
    
    def _calculate_ensemble_diversity(self, X):
        """Calculate diversity metrics for the ensemble."""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X)
                if self.ensemble_method == 'mean_var':
                    pred_mean, _ = pred.chunk(2, dim=-1)
                    predictions.append(pred_mean.numpy())
                else:
                    predictions.append(pred.numpy())
        
        predictions = np.array(predictions)  # Shape: (n_models, n_samples, output_dim)
        
        # Calculate pairwise correlations
        diversity_scores = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                corr = np.corrcoef(predictions[i].flatten(), predictions[j].flatten())[0, 1]
                diversity_scores.append(1 - abs(corr))  # Higher diversity = lower correlation
        
        return diversity_scores
    
    def predict_with_uncertainty(
        self,
        X: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """Generate predictions with both aleatoric and epistemic uncertainty."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Convert to tensor if needed
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X)
            return_numpy = True
        else:
            X_tensor = X
            return_numpy = False
        
        # Collect predictions from all models
        predictions = []
        aleatoric_vars = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(X_tensor)
                
                if self.ensemble_method == 'mean_var':
                    mean_pred, log_var = output.chunk(2, dim=-1)
                    aleatoric_var = torch.exp(log_var)
                    predictions.append(mean_pred)
                    aleatoric_vars.append(aleatoric_var)
                else:
                    predictions.append(output)
                    aleatoric_vars.append(torch.zeros_like(output))  # No aleatoric uncertainty
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # Shape: (n_models, batch_size, output_dim)
        aleatoric_vars = torch.stack(aleatoric_vars, dim=0)
        
        # Calculate ensemble statistics
        ensemble_mean = torch.mean(predictions, dim=0)
        epistemic_uncertainty = torch.std(predictions, dim=0)  # Disagreement between models
        aleatoric_uncertainty = torch.sqrt(torch.mean(aleatoric_vars, dim=0))  # Average aleatoric uncertainty
        
        # Total uncertainty (combining both sources)
        total_uncertainty = torch.sqrt(epistemic_uncertainty ** 2 + aleatoric_uncertainty ** 2)
        
        # Prediction intervals based on total uncertainty
        z_score = 1.96  # For 95% confidence
        lower_bound = ensemble_mean - z_score * total_uncertainty
        upper_bound = ensemble_mean + z_score * total_uncertainty
        
        # Prepare results
        results = {
            'predictions': ensemble_mean.numpy() if return_numpy else ensemble_mean,
            'epistemic_uncertainty': epistemic_uncertainty.numpy() if return_numpy else epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty.numpy() if return_numpy else aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty.numpy() if return_numpy else total_uncertainty,
            'prediction_intervals': torch.stack([lower_bound, upper_bound], dim=-1).numpy() if return_numpy 
                                  else torch.stack([lower_bound, upper_bound], dim=-1),
            'confidence': (1.0 - total_uncertainty).numpy() if return_numpy else (1.0 - total_uncertainty)
        }
        
        return results


class QuantileRegressor(nn.Module):
    """Neural network for quantile regression with multiple quantiles."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        activation: str = 'relu'
    ):
        """Initialize Quantile Regressor.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            quantiles: List of quantiles to predict
            activation: Activation function
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        
        # Build shared layers
        shared_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims[:-1]:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Quantile-specific heads
        self.quantile_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                self._get_activation(activation),
                nn.Dropout(0.1),
                nn.Linear(hidden_dims[-1], 1)
            )
            for _ in quantiles
        ])
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"QuantileRegressor initialized: {input_dim} -> {hidden_dims} -> {self.n_quantiles} quantiles")
        logger.info(f"Target quantiles: {quantiles}")
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU()
        }
        return activations.get(activation, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        # Shared features
        shared_features = self.shared_network(x)
        
        # Quantile predictions
        quantile_outputs = []
        for head in self.quantile_heads:
            quantile_outputs.append(head(shared_features))
        
        # Stack quantile predictions
        outputs = torch.cat(quantile_outputs, dim=-1)
        
        # Ensure quantile ordering (monotonicity constraint)
        if self.training:
            outputs = torch.sort(outputs, dim=-1)[0]
        
        return outputs
    
    def quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate quantile loss."""
        batch_size = predictions.size(0)
        total_loss = 0.0
        
        for i, quantile in enumerate(self.quantiles):
            pred_quantile = predictions[:, i]
            residuals = targets.squeeze() - pred_quantile
            
            # Quantile loss (pinball loss)
            loss = torch.where(
                residuals >= 0,
                quantile * residuals,
                (quantile - 1.0) * residuals
            )
            
            total_loss += loss.mean()
        
        return total_loss / self.n_quantiles
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train the quantile regressor."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Split data
        n_val = int(len(X) * validation_split)
        indices = torch.randperm(len(X))
        
        X_train = X_tensor[indices[n_val:]]
        y_train = y_tensor[indices[n_val:]]
        X_val = X_tensor[indices[:n_val]]
        y_val = y_tensor[indices[:n_val]]
        
        # Setup training
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_coverage': []}
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_losses = []
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = self.quantile_loss(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation phase
            self.eval()
            with torch.no_grad():
                val_outputs = self.forward(X_val)
                val_loss = self.quantile_loss(val_outputs, y_val).item()
                
                # Calculate coverage for middle quantiles (25%-75%)
                if len(self.quantiles) >= 4:
                    q25_idx = 1  # Assuming quantiles include 0.25
                    q75_idx = 3  # Assuming quantiles include 0.75
                    
                    lower_bound = val_outputs[:, q25_idx]
                    upper_bound = val_outputs[:, q75_idx]
                    
                    coverage = torch.mean(
                        ((y_val.squeeze() >= lower_bound) & (y_val.squeeze() <= upper_bound)).float()
                    ).item()
                else:
                    coverage = 0.0
            
            # Update history
            train_loss = np.mean(train_losses)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_coverage'].append(coverage)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Progress logging
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}, "
                           f"val_loss={val_loss:.6f}, val_coverage={coverage:.3f}")
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        return history
    
    def predict_with_intervals(
        self,
        X: Union[np.ndarray, torch.Tensor],
        confidence_level: float = 0.8
    ) -> Dict[str, np.ndarray]:
        """Generate predictions with confidence intervals from quantiles."""
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X)
            return_numpy = True
        else:
            X_tensor = X
            return_numpy = False
        
        self.eval()
        with torch.no_grad():
            quantile_preds = self.forward(X_tensor)
        
        # Find median prediction (or closest to 0.5)
        median_idx = min(range(len(self.quantiles)), key=lambda i: abs(self.quantiles[i] - 0.5))
        predictions = quantile_preds[:, median_idx]
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        # Find closest quantile indices
        lower_idx = min(range(len(self.quantiles)), key=lambda i: abs(self.quantiles[i] - lower_quantile))
        upper_idx = min(range(len(self.quantiles)), key=lambda i: abs(self.quantiles[i] - upper_quantile))
        
        lower_bounds = quantile_preds[:, lower_idx]
        upper_bounds = quantile_preds[:, upper_idx]
        
        # Calculate uncertainty metrics
        interval_widths = upper_bounds - lower_bounds
        uncertainty = interval_widths / 2  # Half-width as uncertainty measure
        
        results = {
            'predictions': predictions.numpy() if return_numpy else predictions,
            'quantile_predictions': quantile_preds.numpy() if return_numpy else quantile_preds,
            'prediction_intervals': torch.stack([lower_bounds, upper_bounds], dim=-1).numpy() if return_numpy
                                  else torch.stack([lower_bounds, upper_bounds], dim=-1),
            'uncertainty': uncertainty.numpy() if return_numpy else uncertainty,
            'confidence': (1.0 - uncertainty / torch.max(uncertainty)).numpy() if return_numpy 
                         else (1.0 - uncertainty / torch.max(uncertainty))
        }
        
        return results


class UncertaintyFusionNetwork(nn.Module):
    """Neural network for fusing multiple uncertainty estimation methods."""
    
    def __init__(
        self,
        input_dim: int,
        n_uncertainty_methods: int = 4,  # MAPIE, MC Dropout, Deep Ensemble, Quantile
        hidden_dims: List[int] = [128, 64, 32],
        fusion_strategy: str = 'learned_weights'
    ):
        """Initialize Uncertainty Fusion Network.
        
        Args:
            input_dim: Number of input features
            n_uncertainty_methods: Number of uncertainty methods to fuse
            hidden_dims: Hidden layer dimensions for fusion network
            fusion_strategy: 'learned_weights', 'attention', or 'simple_average'
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.n_methods = n_uncertainty_methods
        self.fusion_strategy = fusion_strategy
        
        if fusion_strategy == 'learned_weights':
            # Simple MLP for learning fusion weights
            layers = []
            prev_dim = input_dim + n_uncertainty_methods * 2  # Features + predictions + uncertainties
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                prev_dim = hidden_dim
            
            # Output fusion weights (softmax normalized)
            layers.append(nn.Linear(prev_dim, n_uncertainty_methods))
            self.fusion_network = nn.Sequential(*layers)
            
        elif fusion_strategy == 'attention':
            # Attention-based fusion
            self.attention_dim = 64
            self.query_net = nn.Linear(input_dim, self.attention_dim)
            self.key_net = nn.Linear(n_uncertainty_methods * 2, self.attention_dim)
            self.value_net = nn.Linear(n_uncertainty_methods * 2, self.attention_dim)
            self.output_net = nn.Linear(self.attention_dim, n_uncertainty_methods)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"UncertaintyFusionNetwork initialized with {fusion_strategy} strategy")
        logger.info(f"Fusing {n_uncertainty_methods} uncertainty methods")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        features: torch.Tensor,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass to compute fusion weights.
        
        Args:
            features: Input features (batch_size, input_dim)
            predictions: Predictions from different methods (batch_size, n_methods)
            uncertainties: Uncertainties from different methods (batch_size, n_methods)
            
        Returns:
            Fusion weights (batch_size, n_methods)
        """
        if self.fusion_strategy == 'simple_average':
            # Equal weights for all methods
            batch_size = features.size(0)
            return torch.ones(batch_size, self.n_methods) / self.n_methods
        
        elif self.fusion_strategy == 'learned_weights':
            # Concatenate all inputs
            fusion_input = torch.cat([features, predictions, uncertainties], dim=-1)
            weights = self.fusion_network(fusion_input)
            return F.softmax(weights, dim=-1)
        
        elif self.fusion_strategy == 'attention':
            # Attention mechanism
            queries = self.query_net(features)  # (batch_size, attention_dim)
            
            # Combine predictions and uncertainties for keys/values
            pred_unc = torch.cat([predictions, uncertainties], dim=-1)
            keys = self.key_net(pred_unc)     # (batch_size, attention_dim)
            values = self.value_net(pred_unc) # (batch_size, attention_dim)
            
            # Compute attention scores
            attention_scores = torch.matmul(queries.unsqueeze(1), keys.unsqueeze(-1)).squeeze(-1)
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # Apply attention to values
            attended = attention_weights.unsqueeze(-1) * values.unsqueeze(1)
            attended = attended.sum(dim=1)
            
            # Output fusion weights
            weights = self.output_net(attended)
            return F.softmax(weights, dim=-1)
    
    def fuse_predictions(
        self,
        features: np.ndarray,
        method_predictions: List[np.ndarray],
        method_uncertainties: List[np.ndarray],
        method_intervals: List[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Fuse predictions from multiple uncertainty methods.
        
        Args:
            features: Input features
            method_predictions: List of predictions from each method
            method_uncertainties: List of uncertainties from each method
            method_intervals: List of prediction intervals from each method
            
        Returns:
            Fused prediction results
        """
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        predictions_tensor = torch.FloatTensor(np.column_stack(method_predictions))
        uncertainties_tensor = torch.FloatTensor(np.column_stack(method_uncertainties))
        
        self.eval()
        with torch.no_grad():
            # Get fusion weights
            fusion_weights = self.forward(features_tensor, predictions_tensor, uncertainties_tensor)
            
            # Weighted combination of predictions
            weighted_predictions = torch.sum(fusion_weights * predictions_tensor, dim=-1)
            
            # Uncertainty fusion (weighted combination)
            weighted_uncertainties = torch.sum(fusion_weights * uncertainties_tensor, dim=-1)
            
            # Additional uncertainty from disagreement between methods
            pred_disagreement = torch.std(predictions_tensor, dim=-1)
            total_uncertainty = torch.sqrt(weighted_uncertainties ** 2 + pred_disagreement ** 2)
            
            # Prediction intervals from uncertainty
            z_score = 1.96  # 95% confidence
            fused_lower = weighted_predictions - z_score * total_uncertainty
            fused_upper = weighted_predictions + z_score * total_uncertainty
        
        # Method reliability scores (based on inverse uncertainty)
        reliability_scores = 1.0 / (method_uncertainties + [1e-8] * len(method_uncertainties))
        normalized_reliability = reliability_scores / np.sum(reliability_scores, axis=0)
        
        results = {
            'fused_predictions': weighted_predictions.numpy(),
            'fused_uncertainty': total_uncertainty.numpy(),
            'fused_intervals': torch.stack([fused_lower, fused_upper], dim=-1).numpy(),
            'fusion_weights': fusion_weights.numpy(),
            'method_reliability': normalized_reliability,
            'confidence': (1.0 - total_uncertainty).numpy(),
            'method_agreement': 1.0 - pred_disagreement.numpy()  # High agreement = low disagreement
        }
        
        return results
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method_predictions: List[np.ndarray],
        method_uncertainties: List[np.ndarray],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train the uncertainty fusion network.
        
        Args:
            X: Input features
            y: True targets
            method_predictions: Predictions from each uncertainty method
            method_uncertainties: Uncertainties from each method
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Validation split ratio
            verbose: Verbose training
            
        Returns:
            Training history
        """
        # Prepare data
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        pred_tensor = torch.FloatTensor(np.column_stack(method_predictions))
        unc_tensor = torch.FloatTensor(np.column_stack(method_uncertainties))
        
        # Split data
        n_val = int(len(X) * validation_split)
        indices = torch.randperm(len(X))
        
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        
        X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
        y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
        pred_train, pred_val = pred_tensor[train_idx], pred_tensor[val_idx]
        unc_train, unc_val = unc_tensor[train_idx], unc_tensor[val_idx]
        
        # Setup training
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'fusion_entropy': []}
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_losses = []
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                batch_pred = pred_train[i:i+batch_size]
                batch_unc = unc_train[i:i+batch_size]
                
                optimizer.zero_grad()
                
                # Get fusion weights
                fusion_weights = self.forward(batch_X, batch_pred, batch_unc)
                
                # Fused predictions
                fused_pred = torch.sum(fusion_weights * batch_pred, dim=-1)
                
                # Loss (prediction accuracy + regularization)
                pred_loss = criterion(fused_pred, batch_y)
                
                # Regularization: encourage diverse weights (entropy regularization)
                weight_entropy = -torch.sum(fusion_weights * torch.log(fusion_weights + 1e-8), dim=-1).mean()
                reg_loss = 0.01 * (1.0 - weight_entropy)  # Penalize low entropy (uniform weights)
                
                total_loss = pred_loss + reg_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_losses.append(total_loss.item())
            
            # Validation phase
            self.eval()
            with torch.no_grad():
                val_weights = self.forward(X_val, pred_val, unc_val)
                val_fused = torch.sum(val_weights * pred_val, dim=-1)
                val_loss = criterion(val_fused, y_val).item()
                
                # Calculate fusion entropy (diversity measure)
                fusion_entropy = -torch.sum(val_weights * torch.log(val_weights + 1e-8), dim=-1).mean().item()
            
            # Update history
            history['train_loss'].append(np.mean(train_losses))
            history['val_loss'].append(val_loss)
            history['fusion_entropy'].append(fusion_entropy)
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={np.mean(train_losses):.6f}, "
                           f"val_loss={val_loss:.6f}, entropy={fusion_entropy:.3f}")
        
        logger.info(f"Fusion network training completed.")
        return history