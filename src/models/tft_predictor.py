"""Temporal Fusion Transformer (TFT) predictor for financial time series forecasting."""

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

from ..config.settings import config


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for automatic feature importance learning."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Context vector network
        self.context_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Variable selection weights
        self.selection_weights = nn.Linear(input_dim, input_dim)
        
        # Non-linear transformation
        self.transform_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for variable selection.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Tuple of (selected_features, selection_weights)
        """
        # Flatten for context computation
        batch_size, seq_len, input_dim = x.shape
        x_flat = x.view(-1, input_dim)  # (batch_size * seq_len, input_dim)
        
        # Compute context vector
        context = self.context_network(x_flat)  # (batch_size * seq_len, input_dim)
        
        # Variable selection weights (softmax over features)
        selection_logits = self.selection_weights(context)  # (batch_size * seq_len, input_dim)
        selection_weights = F.softmax(selection_logits, dim=-1)  # Softmax over feature dimension
        
        # Apply selection weights
        selected_features = x_flat * selection_weights  # Element-wise multiplication
        
        # Transform selected features
        transformed_features = self.transform_network(selected_features)  # (batch_size * seq_len, hidden_dim)
        
        # Reshape back to sequential format
        transformed_features = transformed_features.view(batch_size, seq_len, -1)
        selection_weights = selection_weights.view(batch_size, seq_len, input_dim)
        
        return transformed_features, selection_weights


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) for non-linear processing with gating."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_context: bool = False,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        
        output_dim = output_dim or input_dim
        self.output_dim = output_dim
        self.use_context = use_context
        
        # Primary network
        self.primary_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )
        
        # Context network (if context is used)
        if use_context and context_dim is not None:
            self.context_network = nn.Linear(context_dim, output_dim, bias=False)
        
        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
        # Skip connection projection (if dimensions don't match)
        if input_dim != output_dim:
            self.skip_projection = nn.Linear(input_dim, output_dim)
        else:
            self.skip_projection = None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Gated Residual Network.
        
        Args:
            x: Input tensor
            context: Optional context tensor
            
        Returns:
            Output tensor after gated residual processing
        """
        # Primary processing
        primary_output = self.primary_network(x)
        
        # Add context if provided
        if self.use_context and context is not None:
            context_output = self.context_network(context)
            primary_output = primary_output + context_output
        
        # Gating
        gate = self.gate_network(x)
        gated_output = gate * primary_output
        
        # Skip connection
        if self.skip_projection is not None:
            skip_output = self.skip_projection(x)
        else:
            skip_output = x
        
        # Residual connection and layer norm
        output = self.layer_norm(gated_output + skip_output)
        
        return output


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-Head Attention with interpretability for temporal pattern recognition."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear transformations
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Attention weights storage for interpretability
        self.attention_weights = None
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor after attention
        """
        batch_size, seq_len, _ = x.size()
        
        # Apply layer norm (pre-norm)
        x_norm = self.layer_norm(x)
        
        # Linear transformations and reshape for multi-head
        Q = self.w_q(x_norm).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x_norm).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x_norm).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attention_weights.detach()  # Store for interpretability
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(attended)
        
        # Residual connection
        return x + output
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get the last computed attention weights for interpretability."""
        return self.attention_weights


class QuantileRegressionHead(nn.Module):
    """Quantile regression head for native uncertainty estimation."""
    
    def __init__(
        self,
        input_dim: int,
        horizon_len: int,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.quantiles = quantiles
        self.horizon_len = horizon_len
        self.n_quantiles = len(quantiles)
        
        # Quantile-specific networks
        self.quantile_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, horizon_len)
            )
            for _ in range(self.n_quantiles)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for quantile regression.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Quantile predictions of shape (batch_size, horizon_len, n_quantiles)
        """
        batch_size = x.size(0)
        
        # Compute predictions for each quantile
        quantile_outputs = []
        for quantile_net in self.quantile_networks:
            quantile_pred = quantile_net(x)  # (batch_size, horizon_len)
            quantile_outputs.append(quantile_pred)
        
        # Stack quantiles: (batch_size, horizon_len, n_quantiles)
        quantile_predictions = torch.stack(quantile_outputs, dim=-1)
        
        return quantile_predictions


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for financial time series forecasting.
    
    Based on the paper: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
    by Lim et al. (2021)
    """
    
    def __init__(
        self,
        input_dim: int = 20,  # Number of input features
        hidden_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        horizon_len: Union[int, List[int]] = [1, 5, 20],
        dropout: float = 0.1,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.horizon_len = horizon_len if isinstance(horizon_len, list) else [horizon_len]
        self.quantiles = quantiles
        
        # Variable Selection Network
        self.variable_selection = VariableSelectionNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Sequence processing with GRN
        self.sequence_processing = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(2)  # Two layers for sequence processing
        ])
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            InterpretableMultiHeadAttention(
                d_model=hidden_dim,
                n_heads=n_heads,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Position-wise feed-forward networks
        self.feed_forward = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim * 4,
                output_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Output processing
        self.output_processing = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Quantile regression heads for each horizon
        self.quantile_heads = nn.ModuleDict({
            f'horizon_{h}': QuantileRegressionHead(
                input_dim=hidden_dim,
                horizon_len=1,  # Each head predicts one step
                quantiles=quantiles,
                hidden_dim=hidden_dim // 2
            )
            for h in self.horizon_len
        })
        
        # Attention weights for interpretability
        self.attention_interpretability = {}
        
        logger.info(f"TFT initialized: {sum(p.numel() for p in self.parameters())} parameters")
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Temporal Fusion Transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with quantile predictions for each horizon
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Variable Selection Network
        selected_features, selection_weights = self.variable_selection(x)  # (batch_size, seq_len, hidden_dim)
        
        # Sequence processing with GRNs
        processed_features = selected_features
        for grn in self.sequence_processing:
            # Apply GRN to each timestep
            processed_features = processed_features.view(-1, self.hidden_dim)
            processed_features = grn(processed_features)
            processed_features = processed_features.view(batch_size, seq_len, self.hidden_dim)
        
        # Multi-head attention layers
        attention_output = processed_features
        attention_weights_list = []
        
        for i, (attention_layer, ff_layer) in enumerate(zip(self.attention_layers, self.feed_forward)):
            # Multi-head attention
            attention_output = attention_layer(attention_output)
            
            # Store attention weights for interpretability
            if return_attention:
                attention_weights_list.append(attention_layer.get_attention_weights())
            
            # Feed-forward network (applied to each timestep)
            ff_input = attention_output.view(-1, self.hidden_dim)
            ff_output = ff_layer(ff_input)
            attention_output = ff_output.view(batch_size, seq_len, self.hidden_dim)
        
        # Final output processing (use last timestep for prediction)
        final_features = attention_output[:, -1, :]  # (batch_size, hidden_dim)
        output_features = self.output_processing(final_features)
        
        # Generate quantile predictions for each horizon
        predictions = {}
        for horizon in self.horizon_len:
            horizon_key = f'horizon_{horizon}'
            quantile_head = self.quantile_heads[horizon_key]
            
            # Get quantile predictions: (batch_size, 1, n_quantiles)
            quantile_preds = quantile_head(output_features)
            
            # Extract median (0.5 quantile) as main prediction
            median_idx = len(self.quantiles) // 2  # Index of median quantile
            prediction = quantile_preds[:, 0, median_idx]  # (batch_size,)
            
            # Extract prediction intervals (0.1 and 0.9 quantiles)
            lower_idx = 0  # 0.1 quantile
            upper_idx = -1  # 0.9 quantile
            lower_bound = quantile_preds[:, 0, lower_idx]  # (batch_size,)
            upper_bound = quantile_preds[:, 0, upper_idx]  # (batch_size,)
            
            predictions[horizon_key] = {
                'prediction': prediction,
                'quantile_predictions': quantile_preds[:, 0, :],  # All quantiles
                'prediction_interval': torch.stack([lower_bound, upper_bound], dim=1),  # (batch_size, 2)
                'confidence': torch.ones_like(prediction) * 0.8  # Default TFT confidence
            }
        
        # Store attention weights for interpretability
        if return_attention:
            self.attention_interpretability = {
                'variable_selection_weights': selection_weights,
                'attention_weights': attention_weights_list
            }
        
        return predictions
    
    def get_attention_interpretability(self) -> Dict[str, Any]:
        """Get attention weights and variable selection for interpretability."""
        return self.attention_interpretability
    
    def quantile_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        quantiles: List[float]
    ) -> torch.Tensor:
        """
        Compute quantile regression loss.
        
        Args:
            predictions: Quantile predictions of shape (batch_size, n_quantiles)
            targets: Target values of shape (batch_size,)
            quantiles: List of quantile levels
            
        Returns:
            Quantile loss tensor
        """
        batch_size, n_quantiles = predictions.shape
        quantile_tensor = torch.tensor(quantiles, device=predictions.device, dtype=predictions.dtype)
        
        # Expand dimensions for broadcasting
        targets = targets.unsqueeze(1).expand(-1, n_quantiles)  # (batch_size, n_quantiles)
        quantile_tensor = quantile_tensor.unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_quantiles)
        
        # Compute quantile loss
        errors = targets - predictions
        loss = torch.where(
            errors >= 0,
            quantile_tensor * errors,
            (quantile_tensor - 1) * errors
        )
        
        return loss.mean()


class TFTPredictor:
    """TFT Predictor wrapper for financial time series forecasting."""
    
    def __init__(
        self,
        horizon_len: Union[int, List[int]] = [1, 5, 20],
        input_dim: int = 20,
        hidden_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        device: str = "auto",
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ):
        """
        Initialize TFT predictor.
        
        Args:
            horizon_len: Prediction horizons
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
            device: Device for computation
            quantiles: Quantile levels for uncertainty estimation
        """
        
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.horizon_len = horizon_len if isinstance(horizon_len, list) else [horizon_len]
        self.input_dim = input_dim
        self.quantiles = quantiles
        self.is_loaded = False
        
        # Initialize TFT model
        self.model = TemporalFusionTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            horizon_len=self.horizon_len,
            dropout=dropout,
            quantiles=quantiles
        ).to(self.device)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.training_history = []
        
        self.is_loaded = True
        logger.info(f"TFT predictor initialized on {self.device}")
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features from OHLCV data for TFT input.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Feature array
        """
        try:
            features = []
            
            # Basic price features
            close_prices = df['close'].values
            if len(close_prices) < 20:
                # Insufficient data - return default features
                return np.zeros((max(20, len(close_prices)), self.input_dim))
            
            # Price-based features
            returns = np.diff(np.log(close_prices + 1e-8))
            returns = np.concatenate([[0], returns])  # Add zero for first value
            
            # Price normalization (z-score over lookback window)
            price_mean = np.mean(close_prices[-60:])
            price_std = np.std(close_prices[-60:]) + 1e-8
            normalized_prices = (close_prices - price_mean) / price_std
            
            # Technical indicators
            sma_5 = pd.Series(close_prices).rolling(5).mean().fillna(method='bfill').values
            sma_10 = pd.Series(close_prices).rolling(10).mean().fillna(method='bfill').values
            sma_20 = pd.Series(close_prices).rolling(20).mean().fillna(method='bfill').values
            
            # Price position relative to moving averages
            price_vs_sma5 = (close_prices - sma_5) / sma_5
            price_vs_sma10 = (close_prices - sma_10) / sma_10
            price_vs_sma20 = (close_prices - sma_20) / sma_20
            
            # Volatility features
            volatility = pd.Series(returns).rolling(10).std().fillna(method='bfill').values
            volatility_ma = pd.Series(volatility).rolling(5).mean().fillna(method='bfill').values
            
            # Volume features (if available)
            if 'volume' in df.columns:
                volume = df['volume'].values
                volume_ma = pd.Series(volume).rolling(10).mean().fillna(method='bfill').values
                volume_ratio = volume / (volume_ma + 1e-8)
                volume_normalized = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
            else:
                volume_ratio = np.ones_like(close_prices)
                volume_normalized = np.zeros_like(close_prices)
            
            # High-low range features (if available)
            if 'high' in df.columns and 'low' in df.columns:
                hl_range = (df['high'] - df['low']) / df['close']
                hl_range = hl_range.fillna(0.02).values  # Default 2% range
                hl_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
                hl_position = hl_position.fillna(0.5).values
            else:
                hl_range = np.full_like(close_prices, 0.02)
                hl_position = np.full_like(close_prices, 0.5)
            
            # Momentum features
            momentum_3 = np.concatenate([np.zeros(3), (close_prices[3:] - close_prices[:-3]) / close_prices[:-3]])
            momentum_5 = np.concatenate([np.zeros(5), (close_prices[5:] - close_prices[:-5]) / close_prices[:-5]])
            momentum_10 = np.concatenate([np.zeros(10), (close_prices[10:] - close_prices[:-10]) / close_prices[:-10]])
            
            # RSI approximation
            gains = np.where(returns > 0, returns, 0)
            losses = np.where(returns < 0, -returns, 0)
            avg_gains = pd.Series(gains).rolling(14).mean().fillna(method='bfill').values
            avg_losses = pd.Series(losses).rolling(14).mean().fillna(method='bfill').values
            rs = avg_gains / (avg_losses + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            rsi_normalized = (rsi - 50) / 50  # Normalize to [-1, 1]
            
            # Time-based features
            timestamps = pd.to_datetime(df.index) if hasattr(df.index, 'to_pydatetime') else pd.date_range(start='2020-01-01', periods=len(df), freq='D')
            hour_of_day = np.sin(2 * np.pi * timestamps.hour / 24)
            day_of_week = np.sin(2 * np.pi * timestamps.dayofweek / 7)
            day_of_month = np.sin(2 * np.pi * timestamps.day / 31)
            
            # Combine all features
            feature_matrix = np.column_stack([
                normalized_prices,      # 1: Normalized prices
                returns,                # 2: Returns
                price_vs_sma5,         # 3: Price vs SMA5
                price_vs_sma10,        # 4: Price vs SMA10
                price_vs_sma20,        # 5: Price vs SMA20
                volatility,            # 6: Volatility
                volatility_ma,         # 7: Volatility MA
                volume_ratio,          # 8: Volume ratio
                volume_normalized,     # 9: Volume normalized
                hl_range,              # 10: High-low range
                hl_position,           # 11: High-low position
                momentum_3,            # 12: 3-period momentum
                momentum_5,            # 13: 5-period momentum
                momentum_10,           # 14: 10-period momentum
                rsi_normalized,        # 15: RSI normalized
                hour_of_day,           # 16: Hour of day (cyclical)
                day_of_week,           # 17: Day of week (cyclical)
                day_of_month,          # 18: Day of month (cyclical)
                np.ones_like(close_prices) * 0.1,  # 19: Constant feature
                np.arange(len(close_prices)) / len(close_prices)  # 20: Time trend
            ])
            
            # Handle NaN values and ensure finite values
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
            feature_matrix = np.clip(feature_matrix, -5.0, 5.0)
            
            return feature_matrix.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error preparing TFT features: {e}")
            # Return default feature matrix
            return np.zeros((max(20, len(df)), self.input_dim), dtype=np.float32)
    
    def predict(
        self,
        df: pd.DataFrame,
        ticker: str,
        return_confidence: bool = True,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Generate TFT predictions.
        
        Args:
            df: Historical OHLCV data
            ticker: Stock ticker symbol
            return_confidence: Whether to return confidence scores
            return_attention: Whether to return attention weights
            
        Returns:
            Prediction results for each horizon
        """
        try:
            if not self.is_loaded:
                logger.error("TFT model not loaded")
                return self._fallback_prediction()
            
            if len(df) < 20:
                logger.warning(f"Insufficient data for TFT prediction: {len(df)} < 20")
                return self._fallback_prediction()
            
            # Prepare features
            features = self._prepare_features(df)
            
            # Use last 60 timesteps for prediction (if available)
            sequence_length = min(60, len(features))
            input_sequence = features[-sequence_length:]
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)  # (1, seq_len, input_dim)
            
            # Model prediction
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(input_tensor, return_attention=return_attention)
            
            # Process predictions
            results = {}
            for horizon in self.horizon_len:
                horizon_key = f"horizon_{horizon}"
                
                if horizon_key in predictions:
                    pred_data = predictions[horizon_key]
                    
                    # Extract main prediction
                    prediction = float(pred_data['prediction'].cpu().item())
                    
                    # Extract prediction intervals
                    pred_intervals = pred_data['prediction_interval'].cpu().numpy()[0]  # (2,)
                    lower_bound = float(pred_intervals[0])
                    upper_bound = float(pred_intervals[1])
                    
                    # Calculate confidence from interval width (narrower = more confident)
                    interval_width = upper_bound - lower_bound
                    confidence = max(0.5, min(0.95, 1.0 / (1.0 + interval_width)))
                    
                    results[horizon_key] = {
                        'prediction': prediction,
                        'confidence': confidence,
                        'prediction_interval': [lower_bound, upper_bound],
                        'quantile_predictions': pred_data['quantile_predictions'].cpu().numpy()[0].tolist(),
                        'model_source': 'tft'
                    }
                else:
                    # Fallback for missing horizon
                    results[horizon_key] = self._fallback_single_prediction(horizon)
            
            # Add attention interpretability if requested
            if return_attention:
                interpretability = self.model.get_attention_interpretability()
                results['attention_interpretability'] = {
                    'variable_selection_weights': interpretability.get('variable_selection_weights'),
                    'attention_patterns': interpretability.get('attention_weights')
                }
            
            logger.debug(f"TFT prediction generated for {ticker}")
            return results
            
        except Exception as e:
            logger.error(f"TFT prediction error for {ticker}: {e}")
            return self._fallback_prediction()
    
    def _fallback_prediction(self) -> Dict[str, Any]:
        """Generate fallback predictions when TFT fails."""
        results = {}
        for horizon in self.horizon_len:
            results[f"horizon_{horizon}"] = self._fallback_single_prediction(horizon)
        return results
    
    def _fallback_single_prediction(self, horizon: int) -> Dict[str, Any]:
        """Generate fallback prediction for single horizon."""
        # Generate reasonable fallback prediction
        prediction = np.random.normal(0.001, 0.01 * np.sqrt(horizon))  # Scale volatility by horizon
        prediction = np.clip(prediction, -0.1, 0.1)  # ±10% bounds
        
        return {
            'prediction': float(prediction),
            'confidence': 0.6,  # Moderate confidence for fallback
            'prediction_interval': [float(prediction - 0.02), float(prediction + 0.02)],
            'quantile_predictions': [float(prediction - 0.03), float(prediction), float(prediction + 0.03)],
            'model_source': 'tft_fallback'
        }
    
    def train_model(
        self,
        training_data: List[pd.DataFrame],
        tickers: List[str],
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the TFT model on historical data.
        
        Args:
            training_data: List of historical OHLCV DataFrames
            tickers: Corresponding ticker symbols
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            validation_split: Validation data fraction
            
        Returns:
            Training results and metrics
        """
        logger.info(f"Training TFT on {len(training_data)} datasets for {epochs} epochs")
        
        try:
            # Prepare training data
            X_train, y_train = self._prepare_training_data(training_data, tickers, validation_split)
            
            if len(X_train) < batch_size:
                logger.warning(f"Insufficient training data: {len(X_train)} samples")
                return {'error': 'insufficient_training_data'}
            
            # Initialize training components
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
            
            # Training loop
            self.model.train()
            training_losses = []
            
            for epoch in range(epochs):
                epoch_losses = []
                
                # Shuffle data
                indices = np.random.permutation(len(X_train))
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]
                
                # Batch training
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_shuffled[i:i+batch_size]
                    batch_y = y_shuffled[i:i+batch_size]
                    
                    # Convert to tensors
                    batch_X_tensor = torch.FloatTensor(batch_X).to(self.device)
                    batch_y_tensor = torch.FloatTensor(batch_y).to(self.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    predictions = self.model(batch_X_tensor)
                    
                    # Calculate loss (focus on horizon_1 for training)
                    if 'horizon_1' in predictions:
                        quantile_preds = predictions['horizon_1']['quantile_predictions']  # (batch_size, n_quantiles)
                        loss = self.model.quantile_loss(quantile_preds, batch_y_tensor, self.quantiles)
                    else:
                        # Fallback loss
                        loss = torch.tensor(0.0, requires_grad=True, device=self.device)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    epoch_losses.append(loss.item())
                
                # Epoch statistics
                avg_loss = np.mean(epoch_losses)
                training_losses.append(avg_loss)
                
                # Learning rate scheduling
                self.scheduler.step(avg_loss)
                
                # Logging
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
            
            # Store training history
            self.training_history = training_losses
            
            logger.info("TFT training completed")
            return {
                'training_loss': training_losses,
                'final_loss': training_losses[-1] if training_losses else 0.0,
                'epochs_trained': epochs,
                'samples_trained': len(X_train)
            }
            
        except Exception as e:
            logger.error(f"TFT training error: {e}")
            return {'error': str(e)}
    
    def _prepare_training_data(
        self,
        training_data: List[pd.DataFrame],
        tickers: List[str],
        validation_split: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for TFT."""
        X_data = []
        y_data = []
        
        sequence_length = 60
        
        for df, ticker in zip(training_data, tickers):
            try:
                if len(df) < sequence_length + 20:  # Need enough data for sequences and targets
                    continue
                
                # Prepare features
                features = self._prepare_features(df)
                
                # Create sequences and targets
                for i in range(sequence_length, len(features) - 5):  # Leave room for targets
                    # Input sequence
                    sequence = features[i-sequence_length:i]
                    
                    # Target (next day return)
                    current_price = df['close'].iloc[i]
                    future_price = df['close'].iloc[i + 1]
                    target_return = (future_price - current_price) / current_price
                    
                    X_data.append(sequence)
                    y_data.append(target_return)
                    
            except Exception as e:
                logger.debug(f"Error preparing training data for {ticker}: {e}")
                continue
        
        if len(X_data) == 0:
            logger.error("No valid training data prepared")
            return np.array([]), np.array([])
        
        X_data = np.array(X_data, dtype=np.float32)
        y_data = np.array(y_data, dtype=np.float32)
        
        # Train/validation split
        split_idx = int(len(X_data) * (1 - validation_split))
        X_train = X_data[:split_idx]
        y_train = y_data[:split_idx]
        
        logger.info(f"Prepared {len(X_train)} training samples")
        
        return X_train, y_train
    
    def save_model(self, filepath: str) -> None:
        """Save TFT model."""
        try:
            save_data = {
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'horizon_len': self.horizon_len,
                    'input_dim': self.input_dim,
                    'quantiles': self.quantiles
                },
                'training_history': self.training_history,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(save_data, filepath)
            logger.info(f"TFT model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving TFT model: {e}")
    
    def load_model(self, filepath: str) -> bool:
        """Load TFT model."""
        try:
            if not Path(filepath).exists():
                logger.warning(f"TFT model file not found: {filepath}")
                return False
            
            save_data = torch.load(filepath, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(save_data['model_state_dict'])
            self.training_history = save_data.get('training_history', [])
            
            logger.info(f"TFT model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading TFT model: {e}")
            return False
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get TFT model summary."""
        return {
            'model_type': 'TemporalFusionTransformer',
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'horizons': self.horizon_len,
            'device': str(self.device),
            'is_loaded': self.is_loaded,
            'training_epochs': len(self.training_history),
            'quantiles': self.quantiles
        }