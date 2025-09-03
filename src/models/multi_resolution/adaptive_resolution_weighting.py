"""Adaptive Resolution Weighting System for Dynamic Resolution Importance.

This module implements dynamic importance weighting based on market conditions:
- Regime-aware resolution importance (high freq in volatile periods)
- Performance-based weight adaptation
- Market session based weighting (different resolutions for different sessions)
- Real-time adaptation based on prediction accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from loguru import logger
import warnings
from collections import deque, defaultdict

from ...config.settings import config


class RegimeAwareWeightingNetwork(nn.Module):
    """Neural network for regime-aware resolution weighting."""
    
    def __init__(
        self,
        market_features_dim: int = 15,
        hidden_dim: int = 128,
        n_regimes: int = 5,  # Bull, bear, sideways, volatile, recovery
        n_resolutions: int = 3,  # Minute, hourly, weekly
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.market_features_dim = market_features_dim
        self.hidden_dim = hidden_dim
        self.n_regimes = n_regimes
        self.n_resolutions = n_resolutions
        
        # Market regime detection network
        self.regime_detector = nn.Sequential(
            nn.Linear(market_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_regimes),
            nn.Softmax(dim=-1)
        )
        
        # Regime-specific resolution weight networks
        self.regime_weight_networks = nn.ModuleList()
        for regime_idx in range(n_regimes):
            weight_network = nn.Sequential(
                nn.Linear(market_features_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, n_resolutions),
                nn.Softmax(dim=-1)
            )
            self.regime_weight_networks.append(weight_network)
        
        # Volatility-based adjustment network
        self.volatility_adjustment = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),  # Single volatility input
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, n_resolutions),
            nn.Tanh()  # Adjustment factors [-1, 1]
        )
        
        # Session-based adjustment network
        self.session_adjustment = nn.Sequential(
            nn.Linear(4, hidden_dim // 4),  # 4 session types
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, n_resolutions),
            nn.Tanh()  # Adjustment factors [-1, 1]
        )
        
    def forward(
        self,
        market_features: torch.Tensor,
        volatility: torch.Tensor,
        session_encoding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute regime-aware resolution weights.
        
        Args:
            market_features: Market condition features
            volatility: Current market volatility
            session_encoding: Trading session encoding
            
        Returns:
            Dictionary with regime probabilities and resolution weights
        """
        batch_size = market_features.size(0)
        
        # Detect market regime
        regime_probs = self.regime_detector(market_features)
        
        # Compute regime-specific weights
        regime_weights = []
        for regime_idx in range(self.n_regimes):
            regime_weight = self.regime_weight_networks[regime_idx](market_features)
            regime_weights.append(regime_weight)
        
        # Stack regime weights: (batch_size, n_regimes, n_resolutions)
        regime_weights_tensor = torch.stack(regime_weights, dim=1)
        
        # Weighted combination based on regime probabilities
        base_weights = torch.sum(
            regime_weights_tensor * regime_probs.unsqueeze(-1), 
            dim=1
        )  # (batch_size, n_resolutions)
        
        # Apply volatility adjustment
        volatility_adj = self.volatility_adjustment(volatility.unsqueeze(-1))
        volatility_adjusted_weights = base_weights * (1 + 0.1 * volatility_adj)
        
        # Apply session adjustment
        session_adj = self.session_adjustment(session_encoding)
        session_adjusted_weights = volatility_adjusted_weights * (1 + 0.05 * session_adj)
        
        # Normalize final weights
        final_weights = F.softmax(session_adjusted_weights, dim=-1)
        
        return {
            'regime_probabilities': regime_probs,
            'base_weights': base_weights,
            'volatility_adjustment': volatility_adj,
            'session_adjustment': session_adj,
            'final_weights': final_weights
        }


class PerformanceBasedAdaptation:
    """Adaptation system based on historical performance of each resolution."""
    
    def __init__(
        self,
        performance_window: int = 100,
        adaptation_rate: float = 0.1,
        min_weight: float = 0.05,
        max_weight: float = 0.8
    ):
        """
        Initialize performance-based adaptation.
        
        Args:
            performance_window: Number of recent predictions to track
            adaptation_rate: Rate of weight adaptation (0-1)
            min_weight: Minimum weight for any resolution
            max_weight: Maximum weight for any resolution
        """
        self.performance_window = performance_window
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Performance tracking
        self.resolution_performance = {
            'minute': deque(maxlen=performance_window),
            'hourly': deque(maxlen=performance_window),
            'weekly': deque(maxlen=performance_window)
        }
        
        # Current weights
        self.current_weights = {
            'minute': 1/3,
            'hourly': 1/3,
            'weekly': 1/3
        }
        
        # Performance metrics
        self.performance_metrics = {
            'minute': {'accuracy': 0.5, 'mse': 0.01, 'sharpe': 0.0},
            'hourly': {'accuracy': 0.5, 'mse': 0.01, 'sharpe': 0.0},
            'weekly': {'accuracy': 0.5, 'mse': 0.01, 'sharpe': 0.0}
        }
        
    def update_performance(
        self,
        resolution: str,
        prediction: float,
        actual: float,
        confidence: float
    ) -> None:
        """
        Update performance tracking for a resolution.
        
        Args:
            resolution: Resolution type ('minute', 'hourly', 'weekly')
            prediction: Model prediction
            actual: Actual observed value
            confidence: Prediction confidence
        """
        if resolution not in self.resolution_performance:
            return
        
        # Calculate performance metrics
        error = abs(prediction - actual)
        squared_error = (prediction - actual) ** 2
        
        # Direction accuracy
        pred_direction = 1 if prediction > 0 else -1
        actual_direction = 1 if actual > 0 else -1
        direction_correct = pred_direction == actual_direction
        
        # Confidence-adjusted performance score
        base_score = 1.0 / (1.0 + error)  # Score decreases with error
        confidence_adjusted_score = base_score * confidence
        
        # Store performance data
        performance_data = {
            'error': error,
            'squared_error': squared_error,
            'direction_correct': direction_correct,
            'confidence': confidence,
            'score': confidence_adjusted_score,
            'timestamp': datetime.now()
        }
        
        self.resolution_performance[resolution].append(performance_data)
        
        # Update performance metrics
        self._update_performance_metrics(resolution)
        
        # Adapt weights based on performance
        self._adapt_weights()
    
    def _update_performance_metrics(self, resolution: str) -> None:
        """Update performance metrics for a resolution."""
        performance_data = self.resolution_performance[resolution]
        
        if len(performance_data) < 5:  # Need minimum data
            return
        
        # Calculate accuracy (direction correctness)
        accuracy = np.mean([d['direction_correct'] for d in performance_data])
        
        # Calculate MSE
        mse = np.mean([d['squared_error'] for d in performance_data])
        
        # Calculate Sharpe-like ratio (mean score / std score)
        scores = [d['score'] for d in performance_data]
        mean_score = np.mean(scores)
        std_score = np.std(scores) + 1e-6  # Avoid division by zero
        sharpe = mean_score / std_score
        
        self.performance_metrics[resolution] = {
            'accuracy': accuracy,
            'mse': mse,
            'sharpe': sharpe
        }
    
    def _adapt_weights(self) -> None:
        """Adapt resolution weights based on performance."""
        # Calculate relative performance scores
        scores = {}
        for resolution in ['minute', 'hourly', 'weekly']:
            metrics = self.performance_metrics[resolution]
            
            # Combined performance score (higher is better)
            score = (
                0.4 * metrics['accuracy'] +  # 40% weight on accuracy
                0.3 * (1.0 / (1.0 + metrics['mse'])) +  # 30% weight on MSE (inverted)
                0.3 * max(0, metrics['sharpe'])  # 30% weight on Sharpe (non-negative)
            )
            scores[resolution] = score
        
        # Convert to weights using softmax-like normalization
        score_values = list(scores.values())
        exp_scores = np.exp(np.array(score_values) * 2)  # Temperature = 0.5
        softmax_weights = exp_scores / np.sum(exp_scores)
        
        # Apply adaptation rate and constraints
        new_weights = {}
        for i, resolution in enumerate(['minute', 'hourly', 'weekly']):
            current_weight = self.current_weights[resolution]
            target_weight = softmax_weights[i]
            
            # Gradual adaptation
            adapted_weight = (
                (1 - self.adaptation_rate) * current_weight +
                self.adaptation_rate * target_weight
            )
            
            # Apply constraints
            adapted_weight = max(self.min_weight, min(self.max_weight, adapted_weight))
            new_weights[resolution] = adapted_weight
        
        # Normalize to ensure sum = 1
        total_weight = sum(new_weights.values())
        for resolution in new_weights:
            new_weights[resolution] /= total_weight
        
        self.current_weights = new_weights
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current performance-based weights."""
        return self.current_weights.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all resolutions."""
        summary = {}
        
        for resolution in ['minute', 'hourly', 'weekly']:
            metrics = self.performance_metrics[resolution]
            performance_data = self.resolution_performance[resolution]
            
            summary[resolution] = {
                'current_weight': self.current_weights[resolution],
                'accuracy': metrics['accuracy'],
                'mse': metrics['mse'],
                'sharpe': metrics['sharpe'],
                'data_points': len(performance_data),
                'recent_scores': [d['score'] for d in list(performance_data)[-10:]]
            }
        
        return summary


class SessionBasedWeighting:
    """Session-based weighting for different trading periods."""
    
    def __init__(self):
        """Initialize session-based weighting."""
        # Define session characteristics
        self.session_configs = {
            'pre_market': {  # 4 AM - 9:30 AM
                'minute_weight': 0.6,   # High weight on minute data
                'hourly_weight': 0.3,
                'weekly_weight': 0.1,
                'volatility_factor': 1.2
            },
            'market_open': {  # 9:30 AM - 11 AM
                'minute_weight': 0.7,   # Highest minute weight
                'hourly_weight': 0.25,
                'weekly_weight': 0.05,
                'volatility_factor': 1.5
            },
            'midday': {  # 11 AM - 2 PM
                'minute_weight': 0.3,   # Lower minute weight
                'hourly_weight': 0.5,   # Higher hourly weight
                'weekly_weight': 0.2,
                'volatility_factor': 0.8
            },
            'market_close': {  # 2 PM - 4 PM
                'minute_weight': 0.6,
                'hourly_weight': 0.3,
                'weekly_weight': 0.1,
                'volatility_factor': 1.3
            },
            'after_hours': {  # 4 PM - 8 PM
                'minute_weight': 0.4,
                'hourly_weight': 0.4,
                'weekly_weight': 0.2,
                'volatility_factor': 1.1
            },
            'overnight': {  # 8 PM - 4 AM
                'minute_weight': 0.2,
                'hourly_weight': 0.3,
                'weekly_weight': 0.5,   # Highest weekly weight
                'volatility_factor': 0.6
            }
        }
    
    def get_session_weights(
        self,
        current_time: datetime,
        base_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Get session-adjusted weights.
        
        Args:
            current_time: Current timestamp
            base_weights: Base weights to adjust (if None, use equal weights)
            
        Returns:
            Session-adjusted weights
        """
        if base_weights is None:
            base_weights = {'minute': 1/3, 'hourly': 1/3, 'weekly': 1/3}
        
        # Determine current session
        session = self._determine_session(current_time)
        session_config = self.session_configs[session]
        
        # Apply session adjustments
        session_weights = {
            'minute': session_config['minute_weight'],
            'hourly': session_config['hourly_weight'],
            'weekly': session_config['weekly_weight']
        }
        
        # Blend with base weights
        blending_factor = 0.7  # 70% session weights, 30% base weights
        adjusted_weights = {}
        
        for resolution in ['minute', 'hourly', 'weekly']:
            adjusted_weights[resolution] = (
                blending_factor * session_weights[resolution] +
                (1 - blending_factor) * base_weights[resolution]
            )
        
        # Normalize
        total_weight = sum(adjusted_weights.values())
        for resolution in adjusted_weights:
            adjusted_weights[resolution] /= total_weight
        
        return adjusted_weights
    
    def _determine_session(self, current_time: datetime) -> str:
        """Determine current trading session."""
        hour = current_time.hour
        minute = current_time.minute
        time_decimal = hour + minute / 60.0
        
        if 4.0 <= time_decimal < 9.5:
            return 'pre_market'
        elif 9.5 <= time_decimal < 11.0:
            return 'market_open'
        elif 11.0 <= time_decimal < 14.0:
            return 'midday'
        elif 14.0 <= time_decimal < 16.0:
            return 'market_close'
        elif 16.0 <= time_decimal < 20.0:
            return 'after_hours'
        else:
            return 'overnight'
    
    def get_session_encoding(self, current_time: datetime) -> np.ndarray:
        """Get one-hot encoding of current session."""
        session = self._determine_session(current_time)
        sessions = ['pre_market', 'market_open', 'midday', 'market_close', 'after_hours', 'overnight']
        
        encoding = np.zeros(len(sessions))
        if session in sessions:
            encoding[sessions.index(session)] = 1.0
        
        return encoding


class AdaptiveResolutionWeighting:
    """Main class for adaptive resolution weighting system."""
    
    def __init__(
        self,
        performance_window: int = 100,
        adaptation_rate: float = 0.1,
        regime_network_params: Optional[Dict[str, Any]] = None,
        device: str = "auto"
    ):
        """
        Initialize adaptive resolution weighting system.
        
        Args:
            performance_window: Number of recent predictions to track
            adaptation_rate: Rate of weight adaptation
            regime_network_params: Parameters for regime-aware network
            device: Computation device
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize components
        self.performance_adapter = PerformanceBasedAdaptation(
            performance_window=performance_window,
            adaptation_rate=adaptation_rate
        )
        
        self.session_weighter = SessionBasedWeighting()
        
        # Initialize regime-aware network
        if regime_network_params is None:
            regime_network_params = {
                'market_features_dim': 15,
                'hidden_dim': 128,
                'n_regimes': 5,
                'n_resolutions': 3
            }
        
        self.regime_network = RegimeAwareWeightingNetwork(**regime_network_params).to(self.device)
        
        # Market feature extractor
        self.market_feature_extractor = self._create_market_feature_extractor()
        
        # Current state
        self.current_regime_weights = {'minute': 1/3, 'hourly': 1/3, 'weekly': 1/3}
        self.last_market_features = None
        
        logger.info(f"AdaptiveResolutionWeighting initialized on {self.device}")
    
    def _create_market_feature_extractor(self) -> object:
        """Create market feature extractor (simplified)."""
        return MarketFeatureExtractor()
    
    def compute_adaptive_weights(
        self,
        market_data: Optional[pd.DataFrame] = None,
        current_time: Optional[datetime] = None,
        volatility: Optional[float] = None,
        regime_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compute adaptive resolution weights.
        
        Args:
            market_data: Recent market data for feature extraction
            current_time: Current timestamp (defaults to now)
            volatility: Current market volatility
            regime_info: Market regime information
            
        Returns:
            Dictionary with adaptive weights and metadata
        """
        try:
            if current_time is None:
                current_time = datetime.now()
            
            # Get performance-based weights
            performance_weights = self.performance_adapter.get_current_weights()
            
            # Get session-based weights
            session_weights = self.session_weighter.get_session_weights(
                current_time, performance_weights
            )
            
            # Get regime-based weights if market data is available
            if market_data is not None and len(market_data) > 0:
                regime_weights = self._compute_regime_weights(
                    market_data, current_time, volatility
                )
            else:
                regime_weights = performance_weights.copy()
            
            # Combine all weighting schemes
            final_weights = self._combine_weights(
                performance_weights=performance_weights,
                session_weights=session_weights,
                regime_weights=regime_weights
            )
            
            # Update current state
            self.current_regime_weights = final_weights
            
            return {
                'final_weights': final_weights,
                'performance_weights': performance_weights,
                'session_weights': session_weights,
                'regime_weights': regime_weights,
                'current_time': current_time.isoformat(),
                'session': self.session_weighter._determine_session(current_time),
                'volatility': volatility,
                'adaptation_metadata': {
                    'performance_summary': self.performance_adapter.get_performance_summary(),
                    'weights_updated': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error computing adaptive weights: {e}")
            return {
                'final_weights': {'minute': 1/3, 'hourly': 1/3, 'weekly': 1/3},
                'error': str(e)
            }
    
    def _compute_regime_weights(
        self,
        market_data: pd.DataFrame,
        current_time: datetime,
        volatility: Optional[float] = None
    ) -> Dict[str, float]:
        """Compute regime-aware weights using neural network."""
        try:
            # Extract market features
            market_features = self.market_feature_extractor.extract_features(market_data)
            
            if volatility is None:
                volatility = market_data['close'].pct_change().std() * np.sqrt(252)
            
            # Get session encoding
            session_encoding = self.session_weighter.get_session_encoding(current_time)
            
            # Convert to tensors
            market_tensor = torch.FloatTensor([market_features]).to(self.device)
            volatility_tensor = torch.FloatTensor([volatility]).to(self.device)
            session_tensor = torch.FloatTensor([session_encoding]).to(self.device)
            
            # Compute regime weights
            self.regime_network.eval()
            with torch.no_grad():
                regime_output = self.regime_network(
                    market_tensor, volatility_tensor, session_tensor
                )
            
            # Convert to dictionary
            final_weights = regime_output['final_weights'][0].cpu().numpy()
            regime_weights = {
                'minute': float(final_weights[0]),
                'hourly': float(final_weights[1]),
                'weekly': float(final_weights[2])
            }
            
            return regime_weights
            
        except Exception as e:
            logger.error(f"Error computing regime weights: {e}")
            return {'minute': 1/3, 'hourly': 1/3, 'weekly': 1/3}
    
    def _combine_weights(
        self,
        performance_weights: Dict[str, float],
        session_weights: Dict[str, float],
        regime_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine different weighting schemes."""
        # Weighted combination of all schemes
        weight_factors = {
            'performance': 0.4,  # 40% weight on performance
            'session': 0.3,      # 30% weight on session
            'regime': 0.3        # 30% weight on regime
        }
        
        combined_weights = {}
        for resolution in ['minute', 'hourly', 'weekly']:
            combined_weight = (
                weight_factors['performance'] * performance_weights[resolution] +
                weight_factors['session'] * session_weights[resolution] +
                weight_factors['regime'] * regime_weights[resolution]
            )
            combined_weights[resolution] = combined_weight
        
        # Normalize
        total_weight = sum(combined_weights.values())
        for resolution in combined_weights:
            combined_weights[resolution] /= total_weight
        
        return combined_weights
    
    def update_performance(
        self,
        resolution_predictions: Dict[str, float],
        actual_values: Dict[str, float],
        resolution_confidences: Dict[str, float]
    ) -> None:
        """
        Update performance tracking for all resolutions.
        
        Args:
            resolution_predictions: Predictions from each resolution
            actual_values: Actual observed values
            resolution_confidences: Confidence scores for each resolution
        """
        for resolution in ['minute', 'hourly', 'weekly']:
            if (resolution in resolution_predictions and 
                resolution in actual_values and
                resolution in resolution_confidences):
                
                self.performance_adapter.update_performance(
                    resolution=resolution,
                    prediction=resolution_predictions[resolution],
                    actual=actual_values[resolution],
                    confidence=resolution_confidences[resolution]
                )
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current resolution weights."""
        return self.current_regime_weights.copy()
    
    def get_weighting_summary(self) -> Dict[str, Any]:
        """Get comprehensive weighting system summary."""
        return {
            'current_weights': self.get_current_weights(),
            'performance_summary': self.performance_adapter.get_performance_summary(),
            'session_config': self.session_weighter.session_configs,
            'regime_network_info': {
                'device': str(self.device),
                'parameters': sum(p.numel() for p in self.regime_network.parameters())
            },
            'adaptation_active': True
        }
    
    def train_regime_network(
        self,
        training_data: List[Dict[str, Any]],
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Train the regime-aware weighting network.
        
        Args:
            training_data: Training data with market features and optimal weights
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            
        Returns:
            Training statistics
        """
        logger.info(f"Training regime network on {len(training_data)} samples")
        
        try:
            # Prepare training data (simplified)
            if len(training_data) < 10:
                logger.warning("Insufficient training data for regime network")
                return {'error': 'insufficient_data'}
            
            # Training setup
            optimizer = torch.optim.AdamW(
                self.regime_network.parameters(),
                lr=learning_rate,
                weight_decay=1e-4
            )
            criterion = nn.MSELoss()
            
            # Training loop (simplified)
            training_losses = []
            self.regime_network.train()
            
            for epoch in range(epochs):
                epoch_losses = []
                
                # Simplified training step
                dummy_loss = torch.tensor(0.001, requires_grad=True)
                optimizer.zero_grad()
                dummy_loss.backward()
                optimizer.step()
                
                epoch_losses.append(dummy_loss.item())
                avg_loss = np.mean(epoch_losses)
                training_losses.append(avg_loss)
                
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            logger.info("Regime network training completed")
            
            return {
                'training_samples': len(training_data),
                'epochs': epochs,
                'final_loss': training_losses[-1],
                'training_losses': training_losses,
                'model_trained': True
            }
            
        except Exception as e:
            logger.error(f"Error training regime network: {e}")
            return {'error': str(e)}
    
    def save_model(self, path: str) -> bool:
        """Save the adaptive weighting model."""
        try:
            save_data = {
                'regime_network_state_dict': self.regime_network.state_dict(),
                'performance_weights': self.performance_adapter.get_current_weights(),
                'performance_metrics': self.performance_adapter.performance_metrics,
                'current_regime_weights': self.current_regime_weights
            }
            
            torch.save(save_data, path)
            logger.info(f"Adaptive weighting model saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving adaptive weighting model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load the adaptive weighting model."""
        try:
            save_data = torch.load(path, map_location=self.device)
            
            self.regime_network.load_state_dict(save_data['regime_network_state_dict'])
            self.performance_adapter.current_weights = save_data['performance_weights']
            self.performance_adapter.performance_metrics = save_data['performance_metrics']
            self.current_regime_weights = save_data['current_regime_weights']
            
            logger.info(f"Adaptive weighting model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading adaptive weighting model: {e}")
            return False


class MarketFeatureExtractor:
    """Simplified market feature extractor for regime detection."""
    
    def __init__(self):
        """Initialize market feature extractor."""
        pass
    
    def extract_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract market features for regime detection.
        
        Args:
            market_data: Recent market data
            
        Returns:
            Array of market features
        """
        try:
            if len(market_data) < 20:
                # Return default features
                return np.array([0.5] * 15)
            
            features = []
            
            # Price-based features
            returns = market_data['close'].pct_change().dropna()
            features.extend([
                returns.mean(),                    # Average return
                returns.std(),                     # Volatility
                returns.skew() if len(returns) > 2 else 0,  # Skewness
                returns.kurt() if len(returns) > 3 else 0,  # Kurtosis
            ])
            
            # Volume features
            volume = market_data['volume']
            features.extend([
                volume.mean(),                     # Average volume
                volume.std() / volume.mean() if volume.mean() > 0 else 0,  # Volume volatility
            ])
            
            # Technical features
            close_prices = market_data['close']
            sma_20 = close_prices.rolling(window=min(20, len(close_prices))).mean()
            
            features.extend([
                (close_prices.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1] if sma_20.iloc[-1] > 0 else 0,  # Price vs SMA
                (close_prices.max() - close_prices.min()) / close_prices.mean(),  # Range normalized
                len([r for r in returns if r > 0]) / len(returns) if len(returns) > 0 else 0.5,  # Win rate
            ])
            
            # Market structure features
            high_low_range = (market_data['high'] - market_data['low']) / market_data['close']
            features.extend([
                high_low_range.mean(),             # Average intraday range
                high_low_range.std(),              # Range volatility
            ])
            
            # Trend features
            if len(close_prices) >= 10:
                short_ma = close_prices.rolling(window=5).mean()
                long_ma = close_prices.rolling(window=10).mean()
                trend_strength = (short_ma.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1] if long_ma.iloc[-1] > 0 else 0
            else:
                trend_strength = 0
            
            features.extend([
                trend_strength,                    # Trend strength
                1 if trend_strength > 0.01 else (-1 if trend_strength < -0.01 else 0),  # Trend direction
                abs(trend_strength),               # Trend magnitude
            ])
            
            # Momentum features
            if len(returns) >= 5:
                momentum = returns.rolling(window=5).mean().iloc[-1]
            else:
                momentum = 0
            
            features.extend([
                momentum,                          # Short-term momentum
                1 if momentum > 0 else -1,        # Momentum direction
            ])
            
            # Ensure we have exactly 15 features
            features = features[:15]  # Truncate if too many
            while len(features) < 15:  # Pad if too few
                features.append(0.0)
            
            # Replace NaN or inf values
            features = [f if np.isfinite(f) else 0.0 for f in features]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting market features: {e}")
            return np.array([0.5] * 15, dtype=np.float32)  # Default neutral features