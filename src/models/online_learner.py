"""Online learning system for continuous neural meta-learner adaptation."""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import warnings
from loguru import logger
from collections import deque
import threading
import time
import asyncio

from .neural_meta_learner import NeuralMetaLearner
from ..features.meta_feature_extractor import MetaFeatureExtractor
from ..config.settings import config

class OnlineLearningSystem:
    """Online learning system for continuous neural meta-learner adaptation."""
    
    def __init__(
        self,
        neural_meta_learner: NeuralMetaLearner,
        feature_extractor: MetaFeatureExtractor,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        update_frequency_minutes: int = 60,
        performance_window: int = 100,
        min_samples_for_update: int = 10,
        max_learning_iterations: int = 5,
        device: str = "auto"
    ):
        """
        Initialize online learning system.
        
        Args:
            neural_meta_learner: The neural meta-learner to adapt
            feature_extractor: Meta-feature extractor instance
            learning_rate: Learning rate for online updates
            batch_size: Batch size for training updates
            update_frequency_minutes: How often to perform updates
            performance_window: Window for tracking performance
            min_samples_for_update: Minimum samples needed for update
            max_learning_iterations: Maximum iterations per update
            device: Computing device
        """
        self.meta_learner = neural_meta_learner
        self.feature_extractor = feature_extractor
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_frequency_minutes = update_frequency_minutes
        self.performance_window = performance_window
        self.min_samples_for_update = min_samples_for_update
        self.max_learning_iterations = max_learning_iterations
        
        # Device configuration
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Performance tracking
        self.performance_buffer = {
            'meta_features': deque(maxlen=performance_window),
            'predictions': deque(maxlen=performance_window),
            'actual_returns': deque(maxlen=performance_window),
            'model_weights': deque(maxlen=performance_window),
            'timestamps': deque(maxlen=performance_window),
            'horizons': deque(maxlen=performance_window)
        }
        
        # Learning statistics
        self.learning_stats = {
            'total_updates': 0,
            'last_update_time': None,
            'recent_losses': deque(maxlen=50),
            'performance_improvements': deque(maxlen=20),
            'learning_rates': deque(maxlen=50)
        }
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.meta_learner.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.8,
            patience=5,
            min_lr=1e-6
        )
        
        # Online learning state
        self.is_running = False
        self.update_thread = None
        self.stop_event = threading.Event()
        
        # Adaptive learning parameters
        self.adaptive_params = {
            'performance_threshold': 0.6,  # Minimum performance for weight updates
            'stability_threshold': 0.05,   # Maximum loss variance for stable learning
            'convergence_patience': 10,    # Iterations to wait for convergence
            'exploration_rate': 0.1        # Rate of random exploration
        }
        
        logger.info(f"OnlineLearningSystem initialized with lr={learning_rate}, update_freq={update_frequency_minutes}min")
    
    def add_experience(
        self,
        meta_features: np.ndarray,
        predictions: Dict[str, float],
        actual_return: float,
        model_weights: Dict[str, float],
        horizon: int
    ) -> None:
        """
        Add new experience for online learning.
        
        Args:
            meta_features: Extracted meta-features
            predictions: Individual model predictions
            actual_return: Actual realized return
            model_weights: Used model weights
            horizon: Prediction horizon
        """
        try:
            # Store experience in buffer
            self.performance_buffer['meta_features'].append(meta_features.copy())
            self.performance_buffer['predictions'].append(predictions.copy())
            self.performance_buffer['actual_returns'].append(actual_return)
            self.performance_buffer['model_weights'].append(model_weights.copy())
            self.performance_buffer['horizons'].append(horizon)
            self.performance_buffer['timestamps'].append(datetime.now())
            
            # Update meta-learner's performance tracking
            pred_array = np.array([predictions.get(model, 0.0) for model in ['timesfm', 'tsmamba', 'samba']])
            weight_array = np.array([model_weights.get(model, 0.33) for model in ['timesfm', 'tsmamba', 'samba']])
            
            self.meta_learner.update_performance(
                actual_returns=torch.tensor([actual_return], dtype=torch.float32),
                predicted_returns=torch.tensor([np.sum(pred_array * weight_array)], dtype=torch.float32),
                model_weights=torch.tensor(weight_array, dtype=torch.float32),
                horizon=horizon,
                meta_features=torch.tensor(meta_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
            
            logger.debug(f"Added experience: horizon={horizon}, actual={actual_return:.4f}, pred={np.sum(pred_array * weight_array):.4f}")
            
        except Exception as e:
            logger.error(f"Error adding experience: {e}")
    
    def should_update(self) -> bool:
        """Check if model should be updated based on accumulated experience."""
        # Check minimum samples
        if len(self.performance_buffer['actual_returns']) < self.min_samples_for_update:
            return False
        
        # Check time since last update
        if self.learning_stats['last_update_time']:
            time_since_update = datetime.now() - self.learning_stats['last_update_time']
            if time_since_update < timedelta(minutes=self.update_frequency_minutes):
                return False
        
        # Check performance variance (update if performance is unstable)
        if len(self.performance_buffer['actual_returns']) >= 20:
            recent_performance = self._calculate_recent_performance()
            performance_variance = np.var(recent_performance)
            
            if performance_variance > self.adaptive_params['stability_threshold']:
                logger.info(f"High performance variance detected: {performance_variance:.4f}, triggering update")
                return True
        
        return True
    
    def perform_update(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform online learning update.
        
        Args:
            force: Force update regardless of conditions
            
        Returns:
            Update statistics
        """
        if not force and not self.should_update():
            return {'skipped': True, 'reason': 'conditions_not_met'}
        
        try:
            logger.info("Starting online learning update...")
            
            # Prepare training data
            training_data = self._prepare_training_data()
            if not training_data:
                return {'error': 'insufficient_training_data'}
            
            # Perform training iterations
            update_results = self._train_update(training_data)
            
            # Update learning statistics
            self.learning_stats['total_updates'] += 1
            self.learning_stats['last_update_time'] = datetime.now()
            
            if 'loss' in update_results:
                self.learning_stats['recent_losses'].append(update_results['loss'])
                self.lr_scheduler.step(update_results['loss'])
            
            # Calculate performance improvement
            performance_improvement = self._calculate_performance_improvement()
            self.learning_stats['performance_improvements'].append(performance_improvement)
            
            # Adaptive parameter adjustment
            self._adapt_learning_parameters(update_results)
            
            logger.info(f"Online learning update completed. Loss: {update_results.get('loss', 'N/A'):.6f}, "
                       f"Performance improvement: {performance_improvement:.4f}")
            
            return {
                'success': True,
                'loss': update_results.get('loss', 0),
                'performance_improvement': performance_improvement,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'samples_used': len(training_data['meta_features'])
            }
            
        except Exception as e:
            logger.error(f"Error during online learning update: {e}")
            return {'error': str(e)}
    
    def _prepare_training_data(self) -> Optional[Dict[str, torch.Tensor]]:
        """Prepare training data from experience buffer."""
        if len(self.performance_buffer['actual_returns']) < self.min_samples_for_update:
            return None
        
        try:
            # Extract recent experiences
            n_samples = min(len(self.performance_buffer['actual_returns']), self.batch_size * 2)
            
            meta_features = list(self.performance_buffer['meta_features'])[-n_samples:]
            actual_returns = list(self.performance_buffer['actual_returns'])[-n_samples:]
            predictions = list(self.performance_buffer['predictions'])[-n_samples:]
            model_weights = list(self.performance_buffer['model_weights'])[-n_samples:]
            horizons = list(self.performance_buffer['horizons'])[-n_samples:]
            
            # Convert to tensors
            features_array = np.array(meta_features)
            features_tensor = torch.tensor(features_array, dtype=torch.float32)
            
            # Add sequence dimension (batch_size, seq_len=1, feature_dim)
            if len(features_tensor.shape) == 2:
                features_tensor = features_tensor.unsqueeze(1)
            
            # Calculate performance targets
            performance_targets = []
            for i, (pred_dict, actual, weights_dict) in enumerate(zip(predictions, actual_returns, model_weights)):
                # Calculate ensemble prediction error
                model_preds = np.array([pred_dict.get(model, 0.0) for model in ['timesfm', 'tsmamba', 'samba']])
                model_ws = np.array([weights_dict.get(model, 0.33) for model in ['timesfm', 'tsmamba', 'samba']])
                ensemble_pred = np.sum(model_preds * model_ws)
                
                # Performance score (inverse of absolute error)
                error = abs(ensemble_pred - actual)
                performance = 1.0 / (1.0 + error)
                performance_targets.append(performance)
            
            performance_tensor = torch.tensor(performance_targets, dtype=torch.float32)
            
            return {
                'meta_features': features_tensor,
                'performance_targets': performance_tensor,
                'horizons': horizons,
                'n_samples': n_samples
            }
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None
    
    def _train_update(self, training_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Perform training update iterations."""
        self.meta_learner.train()
        
        meta_features = training_data['meta_features'].to(self.device)
        performance_targets = training_data['performance_targets'].to(self.device)
        horizons = training_data['horizons']
        
        total_loss = 0.0
        iterations = 0
        
        # Group by horizon for specialized training
        horizon_groups = {}
        for i, horizon in enumerate(horizons):
            if horizon not in horizon_groups:
                horizon_groups[horizon] = []
            horizon_groups[horizon].append(i)
        
        # Train on each horizon group
        for horizon, indices in horizon_groups.items():
            if len(indices) < 3:  # Skip if too few samples
                continue
            
            horizon_features = meta_features[indices]
            horizon_targets = performance_targets[indices]
            
            # Mini-batch training
            n_samples = len(horizon_features)
            batch_size = min(self.batch_size, n_samples)
            
            for iteration in range(min(self.max_learning_iterations, n_samples // batch_size + 1)):
                # Random batch sampling
                batch_indices = torch.randperm(n_samples)[:batch_size]
                batch_features = horizon_features[batch_indices]
                batch_targets = horizon_targets[batch_indices]
                
                # Forward pass
                results = self.meta_learner.forward(batch_features, horizon)
                dynamic_weights = results['dynamic_weights']
                
                # Loss: maximize performance weighted by dynamic weights
                # Performance loss
                performance_loss = -torch.mean(batch_targets * torch.sum(dynamic_weights, dim=-1))
                
                # Regularization losses
                entropy_reg = -0.01 * torch.mean(torch.sum(dynamic_weights * torch.log(dynamic_weights + 1e-8), dim=-1))
                weight_reg = 0.001 * torch.mean(torch.sum(dynamic_weights ** 2, dim=-1))
                
                # Total loss
                loss = performance_loss + entropy_reg + weight_reg
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.meta_learner.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                iterations += 1
                
                # Early stopping if loss is very small
                if loss.item() < 1e-6:
                    break
        
        avg_loss = total_loss / max(iterations, 1)
        
        self.meta_learner.eval()
        
        return {
            'loss': avg_loss,
            'iterations': iterations,
            'horizons_trained': list(horizon_groups.keys())
        }
    
    def _calculate_recent_performance(self) -> List[float]:
        """Calculate recent performance scores."""
        if len(self.performance_buffer['actual_returns']) < 10:
            return [0.5] * 10
        
        recent_predictions = list(self.performance_buffer['predictions'])[-20:]
        recent_actuals = list(self.performance_buffer['actual_returns'])[-20:]
        recent_weights = list(self.performance_buffer['model_weights'])[-20:]
        
        performance_scores = []
        for pred_dict, actual, weights_dict in zip(recent_predictions, recent_actuals, recent_weights):
            model_preds = np.array([pred_dict.get(model, 0.0) for model in ['timesfm', 'tsmamba', 'samba']])
            model_ws = np.array([weights_dict.get(model, 0.33) for model in ['timesfm', 'tsmamba', 'samba']])
            ensemble_pred = np.sum(model_preds * model_ws)
            
            error = abs(ensemble_pred - actual)
            performance = 1.0 / (1.0 + error)
            performance_scores.append(performance)
        
        return performance_scores
    
    def _calculate_performance_improvement(self) -> float:
        """Calculate performance improvement from recent updates."""
        if len(self.learning_stats['performance_improvements']) < 2:
            return 0.0
        
        recent_performance = self._calculate_recent_performance()
        if len(recent_performance) < 10:
            return 0.0
        
        current_perf = np.mean(recent_performance[-5:])
        baseline_perf = np.mean(recent_performance[-10:-5]) if len(recent_performance) >= 10 else current_perf
        
        return current_perf - baseline_perf
    
    def _adapt_learning_parameters(self, update_results: Dict[str, Any]) -> None:
        """Adapt learning parameters based on recent performance."""
        try:
            # Adapt based on loss trends
            if len(self.learning_stats['recent_losses']) >= 5:
                recent_losses = list(self.learning_stats['recent_losses'])[-5:]
                loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
                
                # If loss is increasing, reduce exploration
                if loss_trend > 0:
                    self.adaptive_params['exploration_rate'] *= 0.9
                else:
                    self.adaptive_params['exploration_rate'] = min(0.2, self.adaptive_params['exploration_rate'] * 1.1)
            
            # Adapt update frequency based on performance stability
            if len(self.learning_stats['performance_improvements']) >= 5:
                performance_improvements = list(self.learning_stats['performance_improvements'])[-5:]
                improvement_variance = np.var(performance_improvements)
                
                if improvement_variance > 0.01:  # High variance - update more frequently
                    self.update_frequency_minutes = max(30, self.update_frequency_minutes - 10)
                else:  # Stable - update less frequently
                    self.update_frequency_minutes = min(180, self.update_frequency_minutes + 15)
            
        except Exception as e:
            logger.debug(f"Error adapting learning parameters: {e}")
    
    def start_online_learning(self) -> None:
        """Start online learning in background thread."""
        if self.is_running:
            logger.warning("Online learning already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        def learning_loop():
            logger.info("Online learning loop started")
            
            while not self.stop_event.is_set():
                try:
                    # Wait for update frequency or stop event
                    if self.stop_event.wait(timeout=self.update_frequency_minutes * 60):
                        break  # Stop event was set
                    
                    # Perform update if conditions are met
                    if self.should_update():
                        update_result = self.perform_update()
                        
                        if 'error' in update_result:
                            logger.error(f"Online learning update failed: {update_result['error']}")
                        else:
                            logger.info(f"Online learning update completed: {update_result}")
                    
                except Exception as e:
                    logger.error(f"Error in online learning loop: {e}")
                    time.sleep(60)  # Wait before retry
            
            logger.info("Online learning loop stopped")
        
        self.update_thread = threading.Thread(target=learning_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("Online learning started in background")
    
    def stop_online_learning(self) -> None:
        """Stop online learning background thread."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=30)
        
        logger.info("Online learning stopped")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        return {
            'is_running': self.is_running,
            'total_updates': self.learning_stats['total_updates'],
            'last_update': self.learning_stats['last_update_time'].isoformat() if self.learning_stats['last_update_time'] else None,
            'buffer_size': len(self.performance_buffer['actual_returns']),
            'recent_losses': list(self.learning_stats['recent_losses'])[-10:],
            'recent_performance_improvements': list(self.learning_stats['performance_improvements'])[-10:],
            'current_learning_rate': self.optimizer.param_groups[0]['lr'],
            'adaptive_params': self.adaptive_params.copy(),
            'update_frequency_minutes': self.update_frequency_minutes,
            'avg_recent_performance': np.mean(self._calculate_recent_performance()) if len(self.performance_buffer['actual_returns']) >= 10 else None
        }
    
    def save_learning_state(self, filepath: str) -> None:
        """Save online learning state."""
        state_data = {
            'learning_stats': dict(self.learning_stats),
            'adaptive_params': self.adaptive_params.copy(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'buffer_data': {
                'meta_features': list(self.performance_buffer['meta_features']),
                'predictions': list(self.performance_buffer['predictions']),
                'actual_returns': list(self.performance_buffer['actual_returns']),
                'model_weights': list(self.performance_buffer['model_weights']),
                'timestamps': [t.isoformat() for t in self.performance_buffer['timestamps']],
                'horizons': list(self.performance_buffer['horizons'])
            },
            'update_frequency_minutes': self.update_frequency_minutes,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(state_data, filepath)
        logger.info(f"Online learning state saved to {filepath}")
    
    def load_learning_state(self, filepath: str) -> None:
        """Load online learning state."""
        try:
            state_data = torch.load(filepath, map_location=self.device)
            
            # Restore statistics
            self.learning_stats.update(state_data['learning_stats'])
            self.adaptive_params.update(state_data['adaptive_params'])
            self.update_frequency_minutes = state_data.get('update_frequency_minutes', self.update_frequency_minutes)
            
            # Restore optimizer state
            self.optimizer.load_state_dict(state_data['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(state_data['lr_scheduler_state_dict'])
            
            # Restore buffer data
            buffer_data = state_data['buffer_data']
            for i in range(len(buffer_data['actual_returns'])):
                self.performance_buffer['meta_features'].append(np.array(buffer_data['meta_features'][i]))
                self.performance_buffer['predictions'].append(buffer_data['predictions'][i])
                self.performance_buffer['actual_returns'].append(buffer_data['actual_returns'][i])
                self.performance_buffer['model_weights'].append(buffer_data['model_weights'][i])
                self.performance_buffer['timestamps'].append(datetime.fromisoformat(buffer_data['timestamps'][i]))
                self.performance_buffer['horizons'].append(buffer_data['horizons'][i])
            
            logger.info(f"Online learning state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading online learning state: {e}")
            raise