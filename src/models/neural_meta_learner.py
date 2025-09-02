"""Neural Meta-Learning System for Dynamic Ensemble Weight Generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import warnings
from loguru import logger
from pathlib import Path

from ..config.settings import config

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for meta-feature processing."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Apply layer norm first (pre-norm transformer)
        x_norm = self.layer_norm(x)
        
        # Linear transformations and split into heads
        Q = self.w_q(x_norm).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x_norm).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x_norm).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output linear layer
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(attended)
        
        # Residual connection
        return x + output

class HorizonSpecificNetwork(nn.Module):
    """Horizon-specific neural network for weight generation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 3,  # 3 models: TimesFM, TSMamba, SAMBA
        horizon: int = 1,
        dropout: float = 0.15
    ):
        super().__init__()
        self.horizon = horizon
        
        # Adjust network depth based on horizon (longer horizons need more complexity)
        if horizon == 1:  # Daily - simpler patterns
            layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim)
            ]
        elif horizon <= 5:  # Weekly - medium complexity
            layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim)
            ]
        else:  # Monthly - complex patterns
            layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim)
            ]
        
        self.network = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=-1)
        
        # Initialize weights for stable training
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        # Apply softmax to ensure weights sum to 1
        weights = self.softmax(logits)
        return weights

class ConfidenceCalibrationNetwork(nn.Module):
    """Neural network for calibrating prediction confidence scores."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class NeuralMetaLearner(nn.Module):
    """Neural meta-learning system for dynamic ensemble weight generation."""
    
    def __init__(
        self,
        meta_feature_dim: int = 32,
        hidden_dim: int = 128,
        horizons: List[int] = [1, 5, 20],
        n_models: int = 3,
        attention_heads: int = 8,
        dropout: float = 0.15,
        device: str = "auto"
    ):
        super().__init__()
        
        self.horizons = horizons
        self.n_models = n_models
        self.meta_feature_dim = meta_feature_dim
        
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Meta-feature processing with attention
        self.feature_attention = MultiHeadAttention(
            d_model=meta_feature_dim,
            n_heads=attention_heads,
            dropout=dropout
        )
        
        # Feature embedding
        self.feature_embedding = nn.Sequential(
            nn.Linear(meta_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # Horizon-specific weight generators
        self.horizon_networks = nn.ModuleDict({
            f'horizon_{h}': HorizonSpecificNetwork(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=n_models,
                horizon=h,
                dropout=dropout
            )
            for h in horizons
        })
        
        # Confidence calibration network
        self.confidence_calibrator = ConfidenceCalibrationNetwork(
            input_dim=hidden_dim + n_models,  # features + raw ensemble prediction
            hidden_dim=hidden_dim // 2
        )
        
        # Performance tracking
        self.performance_history = {}
        self.training_history = []
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"NeuralMetaLearner initialized with {sum(p.numel() for p in self.parameters())} parameters on {self.device}")
    
    def forward(
        self,
        meta_features: torch.Tensor,
        horizon: int,
        ensemble_prediction: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to generate dynamic weights and calibrated confidence.
        
        Args:
            meta_features: Tensor of shape (batch_size, seq_len, feature_dim)
            horizon: Prediction horizon
            ensemble_prediction: Raw ensemble prediction for confidence calibration
            
        Returns:
            Dictionary with dynamic weights and calibrated confidence
        """
        # Ensure input is on correct device
        meta_features = meta_features.to(self.device)
        
        # Apply attention to meta-features
        attended_features = self.feature_attention(meta_features)
        
        # Pool features (mean over sequence dimension)
        pooled_features = torch.mean(attended_features, dim=1)  # (batch_size, feature_dim)
        
        # Embed features
        embedded_features = self.feature_embedding(pooled_features)
        
        # Generate horizon-specific weights
        horizon_key = f'horizon_{horizon}'
        if horizon_key not in self.horizon_networks:
            # Fallback to closest horizon
            available_horizons = [int(h.split('_')[1]) for h in self.horizon_networks.keys()]
            closest_horizon = min(available_horizons, key=lambda x: abs(x - horizon))
            horizon_key = f'horizon_{closest_horizon}'
            logger.warning(f"Horizon {horizon} not found, using closest: {closest_horizon}")
        
        dynamic_weights = self.horizon_networks[horizon_key](embedded_features)
        
        results = {'dynamic_weights': dynamic_weights}
        
        # Calibrate confidence if ensemble prediction provided
        if ensemble_prediction is not None:
            ensemble_prediction = ensemble_prediction.to(self.device)
            # Concatenate features with prediction for confidence calibration
            confidence_input = torch.cat([embedded_features, ensemble_prediction], dim=-1)
            calibrated_confidence = self.confidence_calibrator(confidence_input)
            results['calibrated_confidence'] = calibrated_confidence
        
        return results
    
    def update_performance(
        self,
        actual_returns: torch.Tensor,
        predicted_returns: torch.Tensor,
        model_weights: torch.Tensor,
        horizon: int,
        meta_features: torch.Tensor
    ) -> None:
        """Update performance history for online learning."""
        with torch.no_grad():
            # Calculate individual model performance
            mse = F.mse_loss(predicted_returns, actual_returns, reduction='none')
            mae = F.l1_loss(predicted_returns, actual_returns, reduction='none')
            
            # Performance score (inverse of error)
            performance_score = 1.0 / (1.0 + mse.mean().item())
            
            # Store performance data
            performance_data = {
                'timestamp': datetime.now(),
                'horizon': horizon,
                'performance_score': performance_score,
                'mse': mse.mean().item(),
                'mae': mae.mean().item(),
                'weights': model_weights.cpu().numpy() if isinstance(model_weights, torch.Tensor) else model_weights
            }
            
            horizon_key = f'horizon_{horizon}'
            if horizon_key not in self.performance_history:
                self.performance_history[horizon_key] = []
            
            self.performance_history[horizon_key].append(performance_data)
            
            # Keep only recent history (last 100 updates)
            if len(self.performance_history[horizon_key]) > 100:
                self.performance_history[horizon_key] = self.performance_history[horizon_key][-100:]
    
    def get_optimal_weights(
        self,
        meta_features: np.ndarray,
        horizon: int,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Generate optimal weights for given market conditions.
        
        Args:
            meta_features: Meta-features as numpy array
            horizon: Prediction horizon
            return_confidence: Whether to return confidence calibration
            
        Returns:
            Dictionary with weights and optional confidence
        """
        self.eval()
        
        with torch.no_grad():
            # Convert to tensor and add batch/sequence dimensions if needed
            if len(meta_features.shape) == 1:
                # Single sample - add batch and sequence dims
                features_tensor = torch.FloatTensor(meta_features).unsqueeze(0).unsqueeze(0)
            elif len(meta_features.shape) == 2:
                # Batch of samples - add sequence dim
                features_tensor = torch.FloatTensor(meta_features).unsqueeze(1)
            else:
                features_tensor = torch.FloatTensor(meta_features)
            
            # Forward pass
            results = self.forward(features_tensor, horizon)
            
            # Extract weights
            weights = results['dynamic_weights'].cpu().numpy()
            if weights.shape[0] == 1:  # Single sample
                weights = weights[0]
            
            # Convert to model name mapping (4 models: TimesFM, TSMamba, SAMBA, TFT)
            model_names = ['timesfm', 'tsmamba', 'samba', 'tft']
            weight_dict = {name: float(weight) for name, weight in zip(model_names, weights)}
            
            output = {
                'weights': weight_dict,
                'horizon': horizon,
                'timestamp': datetime.now().isoformat()
            }
            
            if return_confidence and 'calibrated_confidence' in results:
                confidence = results['calibrated_confidence'].cpu().numpy()
                if confidence.shape[0] == 1:
                    confidence = float(confidence[0])
                output['calibrated_confidence'] = confidence
            
            return output
    
    def train_step(
        self,
        meta_features: torch.Tensor,
        target_performance: torch.Tensor,
        horizon: int,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Single training step for online learning.
        
        Args:
            meta_features: Input meta-features
            target_performance: Target performance scores
            horizon: Prediction horizon
            optimizer: Optimizer instance
            
        Returns:
            Training loss
        """
        self.train()
        optimizer.zero_grad()
        
        # Forward pass
        results = self.forward(meta_features, horizon)
        weights = results['dynamic_weights']
        
        # Loss: maximize performance weighted by dynamic weights
        # Convert target_performance to weights importance
        performance_loss = -torch.mean(target_performance * torch.sum(weights, dim=-1))
        
        # Regularization: prefer balanced weights (entropy regularization)
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
        entropy_loss = -0.01 * torch.mean(entropy)  # Encourage diversity
        
        total_loss = performance_loss + entropy_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Store training history
        self.training_history.append({
            'timestamp': datetime.now(),
            'loss': total_loss.item(),
            'performance_loss': performance_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'horizon': horizon
        })
        
        return total_loss.item()
    
    def save_model(self, filepath: str) -> None:
        """Save the neural meta-learner model."""
        save_data = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'meta_feature_dim': self.meta_feature_dim,
                'horizons': self.horizons,
                'n_models': self.n_models
            },
            'performance_history': self.performance_history,
            'training_history': self.training_history[-100:],  # Keep recent history
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(save_data, filepath)
        logger.info(f"NeuralMetaLearner saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a saved neural meta-learner model."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        save_data = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.load_state_dict(save_data['model_state_dict'])
        
        # Load performance history
        self.performance_history = save_data.get('performance_history', {})
        self.training_history = save_data.get('training_history', [])
        
        logger.info(f"NeuralMetaLearner loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of model state and performance."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        recent_performance = {}
        for horizon_key, history in self.performance_history.items():
            if history:
                recent_perf = np.mean([h['performance_score'] for h in history[-10:]])
                recent_performance[horizon_key] = recent_perf
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'horizons': self.horizons,
            'recent_performance': recent_performance,
            'training_steps': len(self.training_history),
            'last_update': self.training_history[-1]['timestamp'] if self.training_history else None
        }

