"""
Optimized Neural Meta-Learning System with Phase 7 Enhancements.
Incorporates hyperparameter optimization, computational efficiency improvements,
and enhanced architecture for better performance with Phase 6 features.
"""

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
import math

from .neural_hyperparams import (
    NeuralArchitectureConfig, 
    TrainingConfig, 
    OnlineLearningConfig,
    ComputationalConfig,
    PRODUCTION_CONFIG
)
from ..config.settings import config


class OptimizedMultiHeadAttention(nn.Module):
    """Optimized multi-head attention with computational efficiency improvements."""
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int = 8, 
        dropout: float = 0.1,
        use_flash_attention: bool = False
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_flash_attention = use_flash_attention
        
        # Fused QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Optimized weight initialization."""
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1.0 / math.sqrt(3))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Pre-layer norm for training stability
        x_norm = self.layer_norm(x)
        
        # Fused QKV computation
        qkv = self.qkv_proj(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Efficient attention computation
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized attention (PyTorch 2.0+)
            attn_output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            # Standard attention computation with optimizations
            scale = 1.0 / math.sqrt(self.d_k)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.out_proj(attn_output)
        
        # Residual connection
        return x + output


class AdaptiveHorizonNetwork(nn.Module):
    """Adaptive neural network that adjusts architecture based on horizon."""
    
    def __init__(
        self,
        input_dim: int,
        config: NeuralArchitectureConfig,
        horizon: int
    ):
        super().__init__()
        self.horizon = horizon
        
        # Get horizon-specific configuration
        horizon_config = config.horizon_configs.get(horizon, config.horizon_configs[1])
        hidden_dim = int(config.hidden_dim * horizon_config['hidden_multiplier'])
        
        # Build adaptive architecture
        layers = []
        current_dim = input_dim
        
        for i in range(horizon_config['layers']):
            if i == 0:
                # Input layer
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU() if horizon_config['activation'] == 'ReLU' else nn.GELU(),
                    nn.Dropout(horizon_config['dropout']),
                    nn.LayerNorm(hidden_dim)  # Add layer norm for stability
                ])
                current_dim = hidden_dim
            elif i == horizon_config['layers'] - 1:
                # Output layer
                layers.append(nn.Linear(current_dim, config.n_models))
            else:
                # Hidden layers with residual connections for deeper networks
                if i > 1 and current_dim == hidden_dim:
                    # Add residual block
                    layers.append(ResidualBlock(hidden_dim, horizon_config['dropout']))
                else:
                    layers.extend([
                        nn.Linear(current_dim, hidden_dim),
                        nn.ReLU() if horizon_config['activation'] == 'ReLU' else nn.GELU(),
                        nn.Dropout(horizon_config['dropout']),
                        nn.LayerNorm(hidden_dim)
                    ])
                    current_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        
        # Temperature parameter for calibrated softmax
        self.temperature = nn.Parameter(torch.ones(1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        # Temperature-scaled softmax for better calibration
        weights = F.softmax(logits / torch.clamp(self.temperature, min=0.1, max=10.0), dim=-1)
        return weights


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


class EnhancedConfidenceCalibrator(nn.Module):
    """Enhanced confidence calibration with uncertainty estimation."""
    
    def __init__(self, config: NeuralArchitectureConfig):
        super().__init__()
        
        input_dim = config.hidden_dim + config.n_models + 3  # +3 for ensemble stats
        
        # Main calibration network
        self.calibration_net = nn.Sequential(
            nn.Linear(input_dim, config.confidence_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.confidence_dropout),
            nn.Linear(config.confidence_hidden_dim, config.confidence_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.confidence_dropout),
            nn.Linear(config.confidence_hidden_dim // 2, 2)  # mean and log_var for uncertainty
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        features: torch.Tensor, 
        ensemble_weights: torch.Tensor,
        ensemble_stats: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with uncertainty quantification.
        
        Args:
            features: Processed meta-features
            ensemble_weights: Current ensemble weights
            ensemble_stats: Ensemble statistics (mean, std, disagreement)
            
        Returns:
            Dictionary with calibrated confidence and uncertainty
        """
        # Combine inputs
        combined_input = torch.cat([features, ensemble_weights, ensemble_stats], dim=-1)
        
        # Get mean and log variance
        output = self.calibration_net(combined_input)
        mean, log_var = output.chunk(2, dim=-1)
        
        # Apply sigmoid to mean for confidence
        confidence = torch.sigmoid(mean)
        
        # Compute uncertainty (epistemic)
        uncertainty = torch.exp(log_var * 0.5)  # Standard deviation
        
        return {
            'confidence': confidence,
            'uncertainty': uncertainty,
            'log_var': log_var
        }


class OptimizedNeuralMetaLearner(nn.Module):
    """Optimized neural meta-learning system with Phase 7 enhancements."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: str = "auto"
    ):
        super().__init__()
        
        # Use provided config or default production config
        if config is None:
            config = PRODUCTION_CONFIG
        
        self.arch_config = config['architecture']
        self.train_config = config['training']
        self.online_config = config['online_learning']
        self.compute_config = config['computational']
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Enhanced feature processing
        self.feature_attention = OptimizedMultiHeadAttention(
            d_model=self.arch_config.meta_feature_dim,
            n_heads=self.arch_config.attention_heads,
            dropout=self.arch_config.attention_dropout,
            use_flash_attention=torch.cuda.is_available()
        )
        
        # Feature embedding with skip connection
        self.feature_embedding = nn.Sequential(
            nn.Linear(self.arch_config.meta_feature_dim, self.arch_config.hidden_dim),
            nn.GELU(),  # GELU often works better than ReLU for transformers
            nn.Dropout(self.train_config.base_learning_rate * 100),  # Adaptive dropout
            nn.LayerNorm(self.arch_config.hidden_dim)
        )
        
        # Optional feature selection integration
        if self.arch_config.enable_feature_selection:
            self.feature_selector = nn.Sequential(
                nn.Linear(self.arch_config.meta_feature_dim, self.arch_config.max_features),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            self.feature_selector = None
        
        # Adaptive horizon networks
        self.horizon_networks = nn.ModuleDict({
            f'horizon_{h}': AdaptiveHorizonNetwork(
                input_dim=self.arch_config.hidden_dim,
                config=self.arch_config,
                horizon=h
            )
            for h in [1, 5, 20]  # Standard horizons
        })
        
        # Enhanced confidence calibration
        self.confidence_calibrator = EnhancedConfidenceCalibrator(self.arch_config)
        
        # Performance tracking and online learning support
        self.register_buffer('performance_history', torch.zeros(100))  # Rolling history
        self.register_buffer('adaptation_count', torch.zeros(1))
        
        # Model complexity tracking
        self.param_count = sum(p.numel() for p in self.parameters())
        
        # Move to device
        self.to(self.device)
        
        # Compile model if requested (PyTorch 2.0+)
        if self.compute_config.jit_compile and hasattr(torch, 'compile'):
            try:
                self = torch.compile(self, mode='default')
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
        
        logger.info(f"OptimizedNeuralMetaLearner initialized with {self.param_count:,} parameters on {self.device}")
    
    def forward(
        self,
        meta_features: torch.Tensor,
        horizon: int,
        ensemble_predictions: Optional[torch.Tensor] = None,
        return_uncertainty: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with uncertainty quantification.
        
        Args:
            meta_features: Input features [batch_size, feature_dim]
            horizon: Prediction horizon
            ensemble_predictions: Current ensemble predictions for calibration
            return_uncertainty: Whether to compute uncertainty estimates
            
        Returns:
            Dictionary with weights, confidence, and uncertainty estimates
        """
        batch_size = meta_features.size(0)
        meta_features = meta_features.to(self.device)
        
        # Ensure 3D input for attention (add sequence dimension if needed)
        if meta_features.dim() == 2:
            meta_features = meta_features.unsqueeze(1)  # [batch, 1, features]
        
        # Optional feature selection
        if self.feature_selector is not None:
            # Apply feature selection (only use first max_features)
            selected_features = self.feature_selector(meta_features.squeeze(1))
            selected_features = selected_features.unsqueeze(1)
        else:
            selected_features = meta_features
        
        # Enhanced feature processing with attention
        attended_features = self.feature_attention(selected_features)
        
        # Feature embedding
        embedded_features = self.feature_embedding(attended_features.squeeze(1))
        
        # Generate horizon-specific weights
        horizon_key = f'horizon_{horizon}'
        if horizon_key not in self.horizon_networks:
            # Use closest available horizon
            available_horizons = [int(k.split('_')[1]) for k in self.horizon_networks.keys()]
            closest_horizon = min(available_horizons, key=lambda x: abs(x - horizon))
            horizon_key = f'horizon_{closest_horizon}'
            logger.debug(f"Using {horizon_key} for horizon {horizon}")
        
        ensemble_weights = self.horizon_networks[horizon_key](embedded_features)
        
        # Calculate ensemble statistics for confidence calibration
        if ensemble_predictions is not None:
            ensemble_mean = torch.mean(ensemble_predictions, dim=-1, keepdim=True)
            ensemble_std = torch.std(ensemble_predictions, dim=-1, keepdim=True)
            ensemble_disagreement = torch.max(ensemble_predictions, dim=-1)[0] - torch.min(ensemble_predictions, dim=-1)[0]
            ensemble_stats = torch.cat([
                ensemble_mean, 
                ensemble_std, 
                ensemble_disagreement.unsqueeze(-1)
            ], dim=-1)
        else:
            # Default statistics when ensemble predictions not available
            ensemble_stats = torch.zeros(batch_size, 3, device=self.device)
        
        results = {'weights': ensemble_weights}
        
        # Enhanced confidence calibration with uncertainty
        if return_uncertainty:
            confidence_results = self.confidence_calibrator(
                embedded_features, 
                ensemble_weights, 
                ensemble_stats
            )
            results.update(confidence_results)
        
        return results
    
    def get_optimal_weights(
        self,
        meta_features: np.ndarray,
        horizon: int,
        return_confidence: bool = True,
        ensemble_predictions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Get optimal ensemble weights with enhanced output.
        
        Args:
            meta_features: Meta-features array
            horizon: Prediction horizon
            return_confidence: Whether to return confidence estimates
            ensemble_predictions: Current ensemble predictions
            
        Returns:
            Dictionary with weights and confidence information
        """
        self.eval()
        
        with torch.no_grad():
            # Convert inputs
            features_tensor = torch.FloatTensor(meta_features).to(self.device)
            if features_tensor.dim() == 1:
                features_tensor = features_tensor.unsqueeze(0)
            
            ensemble_tensor = None
            if ensemble_predictions is not None:
                ensemble_tensor = torch.FloatTensor(ensemble_predictions).to(self.device)
                if ensemble_tensor.dim() == 1:
                    ensemble_tensor = ensemble_tensor.unsqueeze(0)
            
            # Forward pass
            results = self.forward(
                features_tensor,
                horizon=horizon,
                ensemble_predictions=ensemble_tensor,
                return_uncertainty=return_confidence
            )
            
            # Convert back to numpy and create output
            weights = results['weights'].cpu().numpy().flatten()
            
            # Map to model names (assuming 5 models: TimesFM, TSMamba, SAMBA, TFT, Multi-resolution)
            model_names = ['timesfm', 'tsmamba', 'samba', 'tft', 'multi_resolution']
            if len(weights) < len(model_names):
                # Pad with zeros if needed
                padded_weights = np.zeros(len(model_names))
                padded_weights[:len(weights)] = weights
                weights = padded_weights
            
            weight_dict = {name: float(weight) for name, weight in zip(model_names, weights)}
            
            output = {'weights': weight_dict}
            
            if return_confidence and 'confidence' in results:
                confidence = results['confidence'].cpu().numpy().flatten()[0]
                uncertainty = results.get('uncertainty', torch.zeros(1)).cpu().numpy().flatten()[0]
                
                output.update({
                    'calibrated_confidence': float(confidence),
                    'epistemic_uncertainty': float(uncertainty),
                    'horizon': horizon,
                    'model_type': 'optimized_neural_meta_learner'
                })
        
        return output
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        return {
            'model_type': 'OptimizedNeuralMetaLearner',
            'total_parameters': self.param_count,
            'architecture': {
                'meta_feature_dim': self.arch_config.meta_feature_dim,
                'hidden_dim': self.arch_config.hidden_dim,
                'attention_heads': self.arch_config.attention_heads,
                'n_models': self.arch_config.n_models
            },
            'device': str(self.device),
            'compiled': hasattr(self, '_orig_mod'),  # Indicates torch.compile was used
            'memory_usage_mb': self.param_count * 4 / (1024 * 1024),
            'feature_selection_enabled': self.arch_config.enable_feature_selection,
            'adaptation_count': int(self.adaptation_count.item()) if hasattr(self, 'adaptation_count') else 0
        }
    
    def save_model(self, path: str) -> None:
        """Save optimized model state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model_state_dict': self.state_dict(),
            'config': {
                'architecture': self.arch_config.__dict__,
                'training': self.train_config.__dict__,
                'online_learning': self.online_config.__dict__,
                'computational': self.compute_config.__dict__
            },
            'param_count': self.param_count,
            'device': str(self.device),
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(state, path)
        logger.info(f"Optimized model saved to {path}")
    
    def load_model(self, path: str) -> bool:
        """Load optimized model state."""
        try:
            path = Path(path)
            if not path.exists():
                logger.error(f"Model file not found: {path}")
                return False
            
            state = torch.load(path, map_location=self.device)
            self.load_state_dict(state['model_state_dict'])
            
            logger.info(f"Optimized model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False