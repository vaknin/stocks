"""
Optimized Hyperparameter Configurations for Neural Meta-Learning System.
Based on Phase 7 optimization and research best practices.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class NeuralArchitectureConfig:
    """Configuration for neural meta-learner architecture optimization."""
    
    # Meta-feature processing
    meta_feature_dim: int = 139  # Enhanced with Phase 6 features (23 regime + 14 performance + 32 multi-resolution + 75 phase6)
    
    # Attention mechanism
    attention_heads: int = 8  # Optimal for meta-feature dimension
    attention_dropout: float = 0.1  # Reduced from 0.15 for better learning
    
    # Hidden dimensions - optimized for feature count
    hidden_dim: int = 256  # Increased for better capacity with Phase 6 features
    
    # Horizon-specific network architecture
    horizon_configs: Dict[int, Dict[str, Any]] = field(default_factory=lambda: {
        1: {  # Daily predictions
            'layers': 3,
            'hidden_multiplier': 1.0,
            'dropout': 0.1,
            'activation': 'ReLU'
        },
        5: {  # Weekly predictions
            'layers': 4,
            'hidden_multiplier': 1.2,
            'dropout': 0.12,
            'activation': 'ReLU'
        },
        20: {  # Monthly predictions
            'layers': 5,
            'hidden_multiplier': 1.5,
            'dropout': 0.15,
            'activation': 'ReLU'
        }
    })
    
    # Model count (4 base models + multi-resolution)
    n_models: int = 5  # TimesFM, TSMamba, SAMBA, TFT, Multi-resolution
    
    # Confidence calibration
    confidence_hidden_dim: int = 128
    confidence_dropout: float = 0.1
    
    # Feature selection integration
    enable_feature_selection: bool = True
    max_features: int = 75  # Optimal feature count after selection
    
    # Regularization
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05


@dataclass
class TrainingConfig:
    """Optimized training configuration for neural meta-learner."""
    
    # Learning rates - optimized schedule
    base_learning_rate: float = 3e-4
    meta_learning_rate: float = 1e-4  # Lower for meta-parameters
    confidence_learning_rate: float = 5e-4  # Higher for confidence calibration
    
    # Optimizer settings
    optimizer: str = 'AdamW'  # Better for transformers
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Learning rate scheduling
    scheduler: str = 'CosineAnnealingWarmRestarts'
    warmup_steps: int = 500
    t_0: int = 1000  # Cosine annealing period
    t_mult: int = 2
    eta_min: float = 1e-6
    
    # Training parameters
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 15
    gradient_clip_norm: float = 1.0
    
    # Validation
    validation_split: float = 0.2
    validation_frequency: int = 10  # Every 10 batches
    
    # Data augmentation for robustness
    feature_noise_std: float = 0.01
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 0.0  # Disabled for time series


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning optimization."""
    
    # Online adaptation rates
    online_learning_rate: float = 1e-5  # Very conservative for stability
    momentum: float = 0.95
    
    # Experience replay
    replay_buffer_size: int = 10000
    replay_batch_size: int = 16
    replay_frequency: int = 5  # Every 5 new experiences
    
    # Performance weighting
    performance_window: int = 100
    performance_decay: float = 0.95
    
    # Adaptation thresholds
    min_experiences: int = 50  # Minimum before adaptation
    adaptation_frequency_minutes: int = 30  # Every 30 minutes
    
    # Drift detection
    drift_detection_window: int = 200
    drift_threshold: float = 0.15  # 15% performance drop triggers retraining


@dataclass 
class ComputationalConfig:
    """Computational efficiency optimization configuration."""
    
    # Memory management
    gradient_accumulation_steps: int = 4  # For large effective batch size
    mixed_precision: bool = True  # FP16 training
    
    # Feature caching
    feature_cache_size: int = 1000
    feature_cache_ttl_seconds: int = 300  # 5 minutes
    phase6_cache_ttl_seconds: int = 600  # 10 minutes for expensive features
    
    # Parallel processing
    num_workers: int = 4
    pin_memory: bool = True
    
    # Model checkpointing
    checkpoint_frequency: int = 500  # Every 500 steps
    keep_n_checkpoints: int = 3
    
    # Inference optimization
    jit_compile: bool = True  # TorchScript compilation
    tensorrt_optimization: bool = False  # Enable on production
    
    # GPU optimization
    cudnn_benchmark: bool = True
    deterministic: bool = False  # Set True for reproducibility


class HyperparameterOptimizer:
    """Hyperparameter optimization utilities."""
    
    @staticmethod
    def get_architecture_search_space() -> Dict[str, List[Any]]:
        """Get architecture search space for optimization."""
        return {
            'hidden_dim': [128, 256, 384, 512],
            'attention_heads': [4, 8, 12, 16],
            'attention_dropout': [0.05, 0.1, 0.15, 0.2],
            'n_horizon_layers': {
                1: [2, 3, 4],
                5: [3, 4, 5], 
                20: [4, 5, 6]
            },
            'confidence_hidden_dim': [64, 128, 192, 256]
        }
    
    @staticmethod
    def get_training_search_space() -> Dict[str, List[Any]]:
        """Get training hyperparameter search space."""
        return {
            'base_learning_rate': [1e-4, 3e-4, 5e-4, 1e-3],
            'batch_size': [16, 32, 48, 64],
            'weight_decay': [1e-5, 1e-4, 1e-3],
            'gradient_clip_norm': [0.5, 1.0, 2.0],
            'label_smoothing': [0.0, 0.05, 0.1],
            'warmup_steps': [200, 500, 1000]
        }
    
    @staticmethod
    def create_optimized_config(
        regime_type: str = 'balanced',
        computational_budget: str = 'high'
    ) -> Dict[str, Any]:
        """
        Create optimized configuration based on market regime and computational budget.
        
        Args:
            regime_type: 'bull', 'bear', 'volatile', 'sideways', 'balanced'
            computational_budget: 'low', 'medium', 'high'
            
        Returns:
            Optimized configuration dictionary
        """
        base_arch = NeuralArchitectureConfig()
        base_train = TrainingConfig()
        base_online = OnlineLearningConfig()
        base_compute = ComputationalConfig()
        
        # Regime-specific optimizations
        if regime_type == 'volatile':
            # Higher capacity for volatile markets
            base_arch.hidden_dim = 384
            base_arch.attention_heads = 12
            base_train.base_learning_rate = 2e-4  # More conservative
            base_online.adaptation_frequency_minutes = 15  # More frequent adaptation
            
        elif regime_type == 'bull':
            # Optimize for trend following
            base_arch.horizon_configs[20]['layers'] = 4  # Simpler monthly patterns in bull
            base_train.base_learning_rate = 5e-4
            
        elif regime_type == 'bear':
            # Optimize for defensive patterns
            base_arch.attention_dropout = 0.15  # Higher regularization
            base_train.early_stopping_patience = 20  # More patience
            base_online.performance_decay = 0.9  # Faster adaptation to new patterns
            
        # Computational budget adjustments
        if computational_budget == 'low':
            base_arch.hidden_dim = 128
            base_arch.attention_heads = 4
            base_train.batch_size = 16
            base_compute.mixed_precision = True
            base_compute.gradient_accumulation_steps = 8
            
        elif computational_budget == 'high':
            base_arch.hidden_dim = 512
            base_arch.attention_heads = 16
            base_train.batch_size = 64
            base_compute.tensorrt_optimization = True
            base_compute.jit_compile = True
        
        return {
            'architecture': base_arch,
            'training': base_train,
            'online_learning': base_online,
            'computational': base_compute
        }
    
    @staticmethod
    def estimate_model_complexity(config: NeuralArchitectureConfig) -> Dict[str, int]:
        """Estimate model complexity from configuration."""
        # Feature processing parameters
        attention_params = config.meta_feature_dim * config.hidden_dim * 4  # Q, K, V, O
        embedding_params = config.meta_feature_dim * config.hidden_dim
        
        # Horizon-specific network parameters
        horizon_params = 0
        for horizon, horizon_config in config.horizon_configs.items():
            layers = horizon_config['layers']
            hidden_size = int(config.hidden_dim * horizon_config['hidden_multiplier'])
            
            # First layer
            horizon_params += config.hidden_dim * hidden_size
            
            # Hidden layers
            for i in range(layers - 2):
                horizon_params += hidden_size * hidden_size
            
            # Output layer
            horizon_params += hidden_size * config.n_models
        
        # Confidence calibration parameters
        confidence_input_dim = config.hidden_dim + config.n_models
        confidence_params = (
            confidence_input_dim * config.confidence_hidden_dim +
            config.confidence_hidden_dim * (config.confidence_hidden_dim // 2) +
            (config.confidence_hidden_dim // 2) * 1
        )
        
        total_params = attention_params + embedding_params + horizon_params + confidence_params
        
        return {
            'total_parameters': total_params,
            'attention_parameters': attention_params,
            'horizon_parameters': horizon_params,
            'confidence_parameters': confidence_params,
            'memory_estimate_mb': total_params * 4 / (1024 * 1024),  # Float32
            'flops_per_forward': total_params * 2  # Rough estimate
        }


# Default optimized configurations
PRODUCTION_CONFIG = HyperparameterOptimizer.create_optimized_config('balanced', 'high')
DEVELOPMENT_CONFIG = HyperparameterOptimizer.create_optimized_config('balanced', 'medium')
RESOURCE_CONSTRAINED_CONFIG = HyperparameterOptimizer.create_optimized_config('balanced', 'low')

# Regime-specific configs
VOLATILE_MARKET_CONFIG = HyperparameterOptimizer.create_optimized_config('volatile', 'high')
BULL_MARKET_CONFIG = HyperparameterOptimizer.create_optimized_config('bull', 'high')
BEAR_MARKET_CONFIG = HyperparameterOptimizer.create_optimized_config('bear', 'high')