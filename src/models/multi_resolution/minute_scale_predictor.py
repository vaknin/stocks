"""Minute-Scale Predictor for High-Frequency Pattern Recognition.

This module implements a lightweight temporal CNN optimized for minute-level trading patterns.
Designed for 1-5 minute prediction horizons with focus on:
- Short-term momentum patterns
- Volume spike detection  
- Intraday technical patterns
- Microstructure approximations
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

from ...config.settings import config


class LightweightTemporalCNN(nn.Module):
    """Lightweight temporal CNN for fast minute-level pattern recognition."""
    
    def __init__(
        self,
        input_dim: int = 12,  # OHLCV + derived features
        hidden_dim: int = 64,
        n_kernels: List[int] = [3, 5, 8],  # Multiple kernel sizes for different patterns
        dropout: float = 0.1,
        output_horizons: List[int] = [1, 5]  # 1min, 5min predictions
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_kernels = n_kernels
        self.output_horizons = output_horizons
        
        # Multi-scale convolutional layers
        self.conv_layers = nn.ModuleList()
        for kernel_size in n_kernels:
            conv_block = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim // len(n_kernels), kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim // len(n_kernels)),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(conv_block)
        
        # Attention mechanism for important time steps
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Horizon-specific prediction heads
        self.prediction_heads = nn.ModuleDict()
        for horizon in output_horizons:
            self.prediction_heads[f'horizon_{horizon}'] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 3)  # [return, confidence, volatility]
            )
        
        # Uncertainty quantification head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for minute-scale prediction.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary with predictions for each horizon
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Transpose for conv1d: (batch_size, input_dim, seq_len)
        x_conv = x.transpose(1, 2)
        
        # Multi-scale convolution
        conv_outputs = []
        min_len = None
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(x_conv)  # (batch_size, hidden_dim//n_kernels, seq_len')
            # Track minimum temporal length across branches to ensure concatenation safety
            seq_len_out = conv_out.shape[-1]
            min_len = seq_len_out if min_len is None else min(min_len, seq_len_out)
            conv_outputs.append(conv_out)

        # If different kernel sizes produced different sequence lengths (e.g., even kernels),
        # center/right-crop to the shortest length before concatenation.
        if any(out.shape[-1] != min_len for out in conv_outputs):
            conv_outputs = [out[..., :min_len] for out in conv_outputs]
        
        # Concatenate multi-scale features (batch_size, hidden_dim, seq_len_common)
        x_conv = torch.cat(conv_outputs, dim=1)
        
        # Transpose back for attention: (batch_size, seq_len, hidden_dim)
        x_conv = x_conv.transpose(1, 2)
        
        # Self-attention to focus on important time steps
        attended, attention_weights = self.attention(x_conv, x_conv, x_conv)
        
        # Feature fusion
        x_fused = self.fusion_layer(attended)
        
        # Global average pooling across time
        x_pooled = torch.mean(x_fused, dim=1)  # (batch_size, hidden_dim)
        
        # Generate predictions for each horizon
        outputs = {}
        for horizon in self.output_horizons:
            head_output = self.prediction_heads[f'horizon_{horizon}'](x_pooled)
            outputs[f'horizon_{horizon}'] = {
                'prediction': head_output[:, 0],      # Return prediction
                'confidence': torch.sigmoid(head_output[:, 1]),  # Confidence [0,1]
                'volatility': F.softplus(head_output[:, 2])      # Volatility estimate
            }
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_head(x_pooled).squeeze(-1)
        
        # Add uncertainty to all outputs
        for horizon_key in outputs.keys():
            outputs[horizon_key]['uncertainty'] = uncertainty
            outputs[horizon_key]['attention_weights'] = attention_weights
        
        return outputs


class MinuteScaleFeatureExtractor:
    """Extract high-frequency features from minute-level OHLCV data."""
    
    def __init__(self, lookback_window: int = 30):
        """
        Initialize feature extractor.
        
        Args:
            lookback_window: Number of minutes to look back for feature computation
        """
        self.lookback_window = lookback_window
        
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract minute-level features from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data at minute resolution
            
        Returns:
            Feature array of shape (n_samples, n_features)
        """
        if len(df) < self.lookback_window:
            raise ValueError(f"Need at least {self.lookback_window} data points")
        
        features = []
        
        # Basic price features
        features.extend([
            df['open'].values,
            df['high'].values,
            df['low'].values, 
            df['close'].values,
            df['volume'].values
        ])
        
        # Price-based features
        returns = df['close'].pct_change().fillna(0).values
        log_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0).values
        
        features.extend([
            returns,
            log_returns
        ])
        
        # Technical indicators (minute-level)
        features.extend([
            self._calculate_rsi(df['close'], period=14),
            self._calculate_vwap(df),
            self._calculate_momentum(df['close'], period=5),
            self._calculate_volume_ratio(df),
            self._calculate_price_range(df),
            self._calculate_intraday_position(df)
        ])
        
        # Stack features and handle the lookback window
        feature_array = np.stack(features, axis=1)  # (n_samples, n_features)
        
        # Create rolling windows
        windowed_features = []
        for i in range(self.lookback_window, len(feature_array)):
            window = feature_array[i-self.lookback_window:i]
            windowed_features.append(window)
        
        if not windowed_features:
            raise ValueError("Not enough data for windowing")
            
        return np.array(windowed_features)  # (n_windows, lookback_window, n_features)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values
    
    def _calculate_vwap(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate Volume Weighted Average Price."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        return vwap.fillna(method='bfill').values
    
    def _calculate_momentum(self, prices: pd.Series, period: int = 5) -> np.ndarray:
        """Calculate price momentum."""
        momentum = prices.pct_change(periods=period)
        return momentum.fillna(0).values
    
    def _calculate_volume_ratio(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate volume ratio vs average."""
        avg_volume = df['volume'].rolling(window=20).mean()
        volume_ratio = df['volume'] / avg_volume
        return volume_ratio.fillna(1).values
    
    def _calculate_price_range(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate high-low price range normalized by close."""
        price_range = (df['high'] - df['low']) / df['close']
        return price_range.fillna(0).values
    
    def _calculate_intraday_position(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate position within daily range."""
        daily_high = df['high'].rolling(window=390).max()  # ~6.5 hours of minutes
        daily_low = df['low'].rolling(window=390).min()
        
        intraday_position = (df['close'] - daily_low) / (daily_high - daily_low)
        return intraday_position.fillna(0.5).values


class MinuteScalePredictor:
    """Main predictor class for minute-scale pattern recognition."""
    
    def __init__(
        self,
        lookback_window: int = 30,
        hidden_dim: int = 64,
        prediction_horizons: List[int] = [1, 5],  # 1min, 5min
        device: str = "auto"
    ):
        """
        Initialize minute-scale predictor.
        
        Args:
            lookback_window: Minutes of history to use for prediction
            hidden_dim: Hidden dimension for neural network
            prediction_horizons: List of prediction horizons in minutes
            device: Computation device ('auto', 'cpu', 'cuda')
        """
        self.lookback_window = lookback_window
        self.prediction_horizons = prediction_horizons
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize components
        self.feature_extractor = MinuteScaleFeatureExtractor(lookback_window)
        
        # Initialize model (will be created after first data sample)
        self.model = None
        self.is_trained = False
        
        logger.info(f"MinuteScalePredictor initialized on {self.device}")
    
    def _initialize_model(self, input_dim: int):
        """Initialize the neural network model."""
        self.model = LightweightTemporalCNN(
            input_dim=input_dim,
            hidden_dim=64,
            n_kernels=[3, 5, 8],
            dropout=0.1,
            output_horizons=self.prediction_horizons
        ).to(self.device)
        
        logger.info(f"Initialized LightweightTemporalCNN with input_dim={input_dim}")
    
    def predict(
        self,
        df: pd.DataFrame,
        ticker: str,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Generate minute-scale predictions.
        
        Args:
            df: DataFrame with minute-level OHLCV data
            ticker: Stock ticker symbol
            return_confidence: Whether to return confidence estimates
            
        Returns:
            Dictionary with predictions for each horizon
        """
        try:
            # Check minimum data requirement
            if len(df) < self.lookback_window + 10:
                logger.warning(f"Insufficient minute data for {ticker}: {len(df)} points")
                return self._fallback_prediction(ticker)
            
            # Extract features
            features = self.feature_extractor.extract_features(df)
            
            # Initialize model if needed
            if self.model is None:
                input_dim = features.shape[-1]
                self._initialize_model(input_dim)
            
            # Use only the latest window for prediction
            latest_features = features[-1:]  # (1, lookback_window, n_features)
            
            # Convert to tensor
            x = torch.FloatTensor(latest_features).to(self.device)
            
            # Generate prediction
            self.model.eval()
            with torch.no_grad():
                model_outputs = self.model(x)
            
            # Process outputs
            predictions = {}
            for horizon in self.prediction_horizons:
                horizon_key = f"horizon_{horizon}"
                model_out = model_outputs[horizon_key]
                
                # Extract predictions and convert to numpy
                prediction = float(model_out['prediction'][0].cpu().numpy())
                confidence = float(model_out['confidence'][0].cpu().numpy()) if return_confidence else 0.7
                volatility = float(model_out['volatility'][0].cpu().numpy())
                uncertainty = float(model_out['uncertainty'][0].cpu().numpy())
                
                # Create prediction intervals using uncertainty
                interval_width = uncertainty * 2.0  # 2-sigma intervals
                lower_bound = prediction - interval_width
                upper_bound = prediction + interval_width
                
                predictions[horizon_key] = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'prediction_interval': [lower_bound, upper_bound],
                    'volatility_forecast': volatility,
                    'uncertainty': uncertainty,
                    'model_type': 'minute_scale_cnn'
                }
            
            # Add metadata
            predictions['minute_scale_meta'] = {
                'ticker': ticker,
                'data_points': len(df),
                'lookback_window': self.lookback_window,
                'features_used': features.shape[-1],
                'timestamp': datetime.now().isoformat()
            }
            
            logger.debug(f"Generated minute-scale predictions for {ticker}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error in minute-scale prediction for {ticker}: {e}")
            return self._fallback_prediction(ticker)
    
    def _fallback_prediction(self, ticker: str) -> Dict[str, Any]:
        """Generate fallback predictions when model fails."""
        predictions = {}
        
        for horizon in self.prediction_horizons:
            horizon_key = f"horizon_{horizon}"
            
            # Simple random walk prediction
            prediction = np.random.normal(0, 0.001)  # Small random movement
            prediction = np.clip(prediction, -0.02, 0.02)  # Clip to ±2%
            
            predictions[horizon_key] = {
                'prediction': float(prediction),
                'confidence': 0.5,
                'prediction_interval': [float(prediction - 0.01), float(prediction + 0.01)],
                'volatility_forecast': 0.01,
                'uncertainty': 0.01,
                'model_type': 'minute_scale_fallback'
            }
        
        predictions['minute_scale_meta'] = {
            'ticker': ticker,
            'fallback_used': True,
            'timestamp': datetime.now().isoformat()
        }
        
        return predictions
    
    def train_model(
        self,
        training_data: List[pd.DataFrame],
        tickers: List[str],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Train the minute-scale prediction model.
        
        Args:
            training_data: List of DataFrames with minute-level OHLCV data
            tickers: Corresponding ticker symbols
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            
        Returns:
            Training statistics and performance metrics
        """
        logger.info(f"Training minute-scale model on {len(training_data)} datasets")
        
        try:
            # Prepare training data
            X_train, y_train = self._prepare_training_data(training_data, tickers)
            
            if len(X_train) < 10:
                logger.warning("Insufficient training data for minute-scale model")
                return {'error': 'insufficient_data'}
            
            # Initialize model if needed
            if self.model is None:
                input_dim = X_train.shape[-1]
                self._initialize_model(input_dim)
            
            # Training setup
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            # Training loop
            training_losses = []
            self.model.train()
            
            for epoch in range(epochs):
                epoch_losses = []
                
                # Create batches
                indices = torch.randperm(len(X_train))
                for i in range(0, len(X_train), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    X_batch = X_train[batch_indices]
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(X_batch)
                    
                    # Calculate loss for each horizon
                    total_loss = 0
                    for horizon in self.prediction_horizons:
                        horizon_key = f"horizon_{horizon}"
                        y_batch = y_train[horizon_key][batch_indices]
                        
                        # Prediction loss
                        pred_loss = criterion(outputs[horizon_key]['prediction'], y_batch)
                        total_loss += pred_loss
                    
                    # Backward pass
                    total_loss.backward()
                    optimizer.step()
                    
                    epoch_losses.append(total_loss.item())
                
                avg_loss = np.mean(epoch_losses)
                training_losses.append(avg_loss)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            self.is_trained = True
            logger.info("Minute-scale model training completed")
            
            return {
                'training_samples': len(X_train),
                'epochs': epochs,
                'final_loss': training_losses[-1],
                'training_losses': training_losses,
                'model_trained': True
            }
            
        except Exception as e:
            logger.error(f"Error training minute-scale model: {e}")
            return {'error': str(e)}
    
    def _prepare_training_data(
        self,
        training_data: List[pd.DataFrame],
        tickers: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare training data from historical minute-level data."""
        X_all = []
        y_all = {f"horizon_{h}": [] for h in self.prediction_horizons}
        
        for df, ticker in zip(training_data, tickers):
            try:
                if len(df) < self.lookback_window + max(self.prediction_horizons) + 10:
                    continue
                
                # Extract features
                features = self.feature_extractor.extract_features(df)
                
                # Create targets for each horizon
                targets = {}
                for horizon in self.prediction_horizons:
                    # Calculate forward returns
                    forward_returns = df['close'].pct_change(periods=horizon).shift(-horizon)
                    
                    # Align with features (features start from lookback_window index)
                    valid_targets = forward_returns.iloc[self.lookback_window:self.lookback_window + len(features)]
                    valid_targets = valid_targets.dropna()
                    
                    if len(valid_targets) != len(features):
                        # Adjust features to match targets
                        min_len = min(len(features), len(valid_targets))
                        features = features[:min_len]
                        valid_targets = valid_targets.iloc[:min_len]
                    
                    targets[f"horizon_{horizon}"] = valid_targets.values
                
                # Add to training set
                if len(features) > 0 and all(len(t) > 0 for t in targets.values()):
                    X_all.append(features)
                    for horizon in self.prediction_horizons:
                        y_all[f"horizon_{horizon}"].append(targets[f"horizon_{horizon}"])
                
            except Exception as e:
                logger.debug(f"Error processing training data for {ticker}: {e}")
                continue
        
        if not X_all:
            raise ValueError("No valid training data found")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(np.concatenate(X_all, axis=0)).to(self.device)
        y_tensors = {}
        for horizon in self.prediction_horizons:
            horizon_key = f"horizon_{horizon}"
            y_tensors[horizon_key] = torch.FloatTensor(np.concatenate(y_all[horizon_key])).to(self.device)
        
        return X_tensor, y_tensors
    
    def save_model(self, path: str) -> bool:
        """Save the trained model."""
        try:
            if self.model is None:
                logger.warning("No model to save")
                return False
                
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'lookback_window': self.lookback_window,
                'prediction_horizons': self.prediction_horizons,
                'is_trained': self.is_trained
            }, path)
            
            logger.info(f"Minute-scale model saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving minute-scale model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load a trained model."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.lookback_window = checkpoint['lookback_window']
            self.prediction_horizons = checkpoint['prediction_horizons']
            self.is_trained = checkpoint['is_trained']
            
            # Initialize model structure
            if self.model is None:
                # Dummy input to get dimensions
                dummy_features = np.random.randn(1, self.lookback_window, 12)
                input_dim = dummy_features.shape[-1]
                self._initialize_model(input_dim)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"Minute-scale model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading minute-scale model: {e}")
            return False
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'model_type': 'MinuteScalePredictor',
            'lookback_window': self.lookback_window,
            'prediction_horizons': self.prediction_horizons,
            'device': str(self.device),
            'is_loaded': self.is_loaded,
            'is_trained': self.is_trained,
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }
