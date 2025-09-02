"""Hourly Predictor for Medium-Term Temporal Pattern Recognition.

This module implements multi-scale temporal convolution networks optimized for hourly trading patterns.
Designed for 1-4 hour prediction horizons with focus on:
- Intraday trend patterns
- Market session analysis (pre-market, regular, after-hours)  
- Cross-asset correlations at hourly resolution
- Economic calendar event impact modeling
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


class MultiScaleTemporalCNN(nn.Module):
    """Multi-scale temporal convolution network for hourly pattern recognition."""
    
    def __init__(
        self,
        input_dim: int = 16,  # Extended features for hourly data
        hidden_dim: int = 128,
        n_scales: int = 3,    # Different temporal scales
        dropout: float = 0.15,
        output_horizons: List[int] = [1, 2, 4]  # 1h, 2h, 4h predictions
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.output_horizons = output_horizons
        
        # Multi-scale convolution blocks
        # Scale 1: Short-term patterns (kernel=3)  
        # Scale 2: Medium-term patterns (kernel=6)
        # Scale 3: Long-term patterns (kernel=12)
        self.conv_blocks = nn.ModuleList()
        kernel_sizes = [3, 6, 12]
        
        for i, kernel_size in enumerate(kernel_sizes):
            conv_block = nn.Sequential(
                nn.Conv1d(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size//2,
                    dilation=1
                ),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                # Second conv layer with dilation
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=(kernel_size//2) * 2,  # Account for dilation
                    dilation=2
                ),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.conv_blocks.append(conv_block)
        
        # Market session attention (pre-market, regular, after-hours)
        self.session_attention = MarketSessionAttention(
            hidden_dim=hidden_dim,
            n_sessions=3,  # Pre-market, regular, after-hours
            dropout=dropout
        )
        
        # Cross-temporal attention for integrating different scales
        self.cross_temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature integration layer
        self.integration_layer = nn.Sequential(
            nn.Linear(hidden_dim * n_scales, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Horizon-specific prediction heads with different complexities
        self.prediction_heads = nn.ModuleDict()
        for horizon in output_horizons:
            # Longer horizons get more complex heads
            head_complexity = min(horizon, 4)  # Cap complexity
            head_layers = []
            
            current_dim = hidden_dim
            for i in range(head_complexity):
                next_dim = max(hidden_dim // (2 ** i), 32)
                head_layers.extend([
                    nn.Linear(current_dim, next_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                current_dim = next_dim
            
            # Final prediction layer
            head_layers.append(nn.Linear(current_dim, 4))  # [return, confidence, volatility, momentum]
            
            self.prediction_heads[f'horizon_{horizon}'] = nn.Sequential(*head_layers)
        
        # Economic event impact head
        self.event_impact_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Event impact can be positive or negative
        )
        
    def forward(self, x: torch.Tensor, session_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for hourly prediction.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            session_mask: Optional market session mask
            
        Returns:
            Dictionary with predictions for each horizon
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Transpose for conv1d: (batch_size, input_dim, seq_len)
        x_conv = x.transpose(1, 2)
        
        # Multi-scale convolution processing
        scale_features = []
        current_input = x_conv
        
        for conv_block in self.conv_blocks:
            scale_output = conv_block(current_input)  # (batch_size, hidden_dim, seq_len)
            scale_features.append(scale_output)
            current_input = scale_output  # Chain the scales
        
        # Convert back to sequence format for attention
        scale_sequences = []
        for scale_feat in scale_features:
            scale_seq = scale_feat.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
            scale_sequences.append(scale_seq)
        
        # Apply market session attention to each scale
        if session_mask is not None:
            attended_scales = []
            for scale_seq in scale_sequences:
                attended = self.session_attention(scale_seq, session_mask)
                attended_scales.append(attended)
            scale_sequences = attended_scales
        
        # Cross-temporal attention between scales
        # Use the finest scale as query, others as key/value
        query = scale_sequences[0]  # Finest scale (shortest patterns)
        key_value = torch.cat(scale_sequences[1:], dim=1)  # Coarser scales
        
        if key_value.size(1) > 0:  # Check if we have coarser scales
            attended_output, _ = self.cross_temporal_attention(query, key_value, key_value)
        else:
            attended_output = query
        
        # Integrate multi-scale features
        concatenated_features = torch.cat([seq.mean(dim=1) for seq in scale_sequences], dim=1)
        integrated_features = self.integration_layer(concatenated_features)
        
        # Global temporal pooling with attention weights
        temporal_features = attended_output.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Combine integrated and temporal features
        combined_features = integrated_features + temporal_features
        
        # Generate predictions for each horizon
        outputs = {}
        for horizon in self.output_horizons:
            head_output = self.prediction_heads[f'horizon_{horizon}'](combined_features)
            
            outputs[f'horizon_{horizon}'] = {
                'prediction': head_output[:, 0],                    # Return prediction
                'confidence': torch.sigmoid(head_output[:, 1]),    # Confidence [0,1]
                'volatility': F.softplus(head_output[:, 2]),       # Volatility estimate
                'momentum': torch.tanh(head_output[:, 3])          # Momentum [-1,1]
            }
        
        # Economic event impact
        event_impact = self.event_impact_head(combined_features).squeeze(-1)
        
        # Add event impact to all outputs
        for horizon_key in outputs.keys():
            outputs[horizon_key]['event_impact'] = event_impact
            outputs[horizon_key]['scale_features'] = scale_sequences  # For analysis
        
        return outputs


class MarketSessionAttention(nn.Module):
    """Attention mechanism for different market sessions."""
    
    def __init__(self, hidden_dim: int, n_sessions: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_sessions = n_sessions
        
        # Session-specific attention weights
        self.session_weights = nn.Parameter(torch.randn(n_sessions, hidden_dim))
        
        # Attention computation
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_sessions)
        )
        
    def forward(self, x: torch.Tensor, session_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply session-aware attention.
        
        Args:
            x: Input features (batch_size, seq_len, hidden_dim)
            session_mask: Session indicators (batch_size, seq_len, n_sessions)
            
        Returns:
            Session-attended features
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Compute attention weights for each session
        attention_logits = self.attention_layer(x)  # (batch_size, seq_len, n_sessions)
        attention_weights = F.softmax(attention_logits, dim=-1)
        
        # Apply session mask
        masked_attention = attention_weights * session_mask
        normalized_attention = masked_attention / (masked_attention.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Weight features by session attention
        session_weighted_features = torch.zeros_like(x)
        for session_id in range(self.n_sessions):
            session_weight = self.session_weights[session_id]  # (hidden_dim,)
            session_attention = normalized_attention[:, :, session_id:session_id+1]  # (batch_size, seq_len, 1)
            
            session_features = x * session_weight.unsqueeze(0).unsqueeze(0)  # Broadcast
            weighted_features = session_features * session_attention
            session_weighted_features += weighted_features
        
        return session_weighted_features


class HourlyFeatureExtractor:
    """Extract hourly-resolution features from OHLCV data."""
    
    def __init__(self, lookback_hours: int = 48):  # 2 days of hourly data
        """
        Initialize feature extractor.
        
        Args:
            lookback_hours: Number of hours to look back for feature computation
        """
        self.lookback_hours = lookback_hours
        
    def extract_features(
        self, 
        df: pd.DataFrame,
        session_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract hourly features from OHLCV data.
        
        Args:
            df: DataFrame with hourly OHLCV data
            session_info: Optional market session information
            
        Returns:
            Tuple of (feature_array, session_mask)
        """
        if len(df) < self.lookback_hours:
            raise ValueError(f"Need at least {self.lookback_hours} data points")
        
        features = []
        
        # Basic OHLCV features
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
        volatility = df['close'].rolling(window=6).std().fillna(0).values  # 6-hour rolling volatility
        
        features.extend([returns, log_returns, volatility])
        
        # Hourly technical indicators
        features.extend([
            self._calculate_hourly_rsi(df['close'], period=14),
            self._calculate_hourly_macd(df['close']),
            self._calculate_hourly_bollinger_position(df['close']),
            self._calculate_hourly_volume_profile(df),
            self._calculate_cross_session_momentum(df),
            self._calculate_hourly_trend_strength(df['close']),
            self._calculate_gap_analysis(df),
            self._calculate_session_relative_performance(df)
        ])
        
        # Stack features
        feature_array = np.stack(features, axis=1)  # (n_samples, n_features)
        
        # Create rolling windows
        windowed_features = []
        for i in range(self.lookback_hours, len(feature_array)):
            window = feature_array[i-self.lookback_hours:i]
            windowed_features.append(window)
        
        if not windowed_features:
            raise ValueError("Not enough data for windowing")
        
        # Create session mask
        session_mask = self._create_session_mask(df, len(windowed_features))
        
        return np.array(windowed_features), session_mask
    
    def _calculate_hourly_rsi(self, prices: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate hourly RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values
    
    def _calculate_hourly_macd(self, prices: pd.Series) -> np.ndarray:
        """Calculate MACD for hourly data."""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        return macd.fillna(0).values
    
    def _calculate_hourly_bollinger_position(self, prices: pd.Series, period: int = 20) -> np.ndarray:
        """Calculate position within Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        return bb_position.fillna(0.5).values
    
    def _calculate_hourly_volume_profile(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate volume profile relative to hourly average."""
        hourly_avg_volume = df['volume'].rolling(window=24).mean()
        volume_profile = df['volume'] / hourly_avg_volume
        return volume_profile.fillna(1).values
    
    def _calculate_cross_session_momentum(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate momentum across different trading sessions."""
        # Compare current hour to same hour previous day
        daily_momentum = df['close'].pct_change(periods=24).fillna(0)
        return daily_momentum.values
    
    def _calculate_hourly_trend_strength(self, prices: pd.Series, period: int = 12) -> np.ndarray:
        """Calculate trend strength over hourly periods."""
        # Linear regression slope over rolling window
        trend_strength = []
        for i in range(len(prices)):
            start_idx = max(0, i - period + 1)
            window_prices = prices.iloc[start_idx:i+1]
            
            if len(window_prices) < 3:
                trend_strength.append(0)
                continue
            
            x = np.arange(len(window_prices))
            y = window_prices.values
            
            # Calculate slope
            slope = np.polyfit(x, y, 1)[0]
            normalized_slope = slope / window_prices.mean() if window_prices.mean() != 0 else 0
            trend_strength.append(normalized_slope)
        
        return np.array(trend_strength)
    
    def _calculate_gap_analysis(self, df: pd.DataFrame) -> np.ndarray:
        """Analyze price gaps between sessions."""
        # Gap from previous close to current open
        gaps = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        return gaps.fillna(0).values
    
    def _calculate_session_relative_performance(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate performance relative to session start."""
        # Performance from session start (assuming sessions start every 8 hours)
        session_starts = df.index[::8] if len(df) >= 8 else df.index[:1]
        relative_performance = []
        
        current_session_start_price = df['open'].iloc[0]
        for i, (idx, row) in enumerate(df.iterrows()):
            # Update session start price every 8 hours
            if i % 8 == 0 and i > 0:
                current_session_start_price = row['open']
            
            rel_perf = (row['close'] - current_session_start_price) / current_session_start_price
            relative_performance.append(rel_perf)
        
        return np.array(relative_performance)
    
    def _create_session_mask(self, df: pd.DataFrame, n_windows: int) -> np.ndarray:
        """Create market session mask (pre-market, regular, after-hours)."""
        # Simple session classification based on hour of day
        # This is a simplified version - real implementation would use market calendars
        
        session_mask = np.zeros((n_windows, self.lookback_hours, 3))  # 3 sessions
        
        for i in range(n_windows):
            start_idx = self.lookback_hours + i - self.lookback_hours
            window_times = df.index[start_idx:start_idx + self.lookback_hours]
            
            for j, timestamp in enumerate(window_times):
                hour = timestamp.hour if hasattr(timestamp, 'hour') else 9  # Default to regular hours
                
                if 4 <= hour < 9:  # Pre-market (4 AM - 9 AM)
                    session_mask[i, j, 0] = 1
                elif 9 <= hour < 16:  # Regular hours (9 AM - 4 PM)  
                    session_mask[i, j, 1] = 1
                else:  # After-hours (4 PM - 4 AM)
                    session_mask[i, j, 2] = 1
        
        return session_mask


class HourlyPredictor:
    """Main predictor class for hourly temporal pattern recognition."""
    
    def __init__(
        self,
        lookback_hours: int = 48,  # 2 days
        hidden_dim: int = 128,
        prediction_horizons: List[int] = [1, 2, 4],  # 1h, 2h, 4h
        device: str = "auto"
    ):
        """
        Initialize hourly predictor.
        
        Args:
            lookback_hours: Hours of history to use for prediction
            hidden_dim: Hidden dimension for neural network
            prediction_horizons: List of prediction horizons in hours
            device: Computation device ('auto', 'cpu', 'cuda')
        """
        self.lookback_hours = lookback_hours
        self.prediction_horizons = prediction_horizons
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize components
        self.feature_extractor = HourlyFeatureExtractor(lookback_hours)
        
        # Initialize model (will be created after first data sample)
        self.model = None
        self.is_trained = False
        
        logger.info(f"HourlyPredictor initialized on {self.device}")
    
    def _initialize_model(self, input_dim: int):
        """Initialize the neural network model."""
        self.model = MultiScaleTemporalCNN(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim if hasattr(self, 'hidden_dim') else 128,
            n_scales=3,
            dropout=0.15,
            output_horizons=self.prediction_horizons
        ).to(self.device)
        
        logger.info(f"Initialized MultiScaleTemporalCNN with input_dim={input_dim}")
    
    def predict(
        self,
        df: pd.DataFrame,
        ticker: str,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Generate hourly predictions.
        
        Args:
            df: DataFrame with hourly OHLCV data
            ticker: Stock ticker symbol
            return_confidence: Whether to return confidence estimates
            
        Returns:
            Dictionary with predictions for each horizon
        """
        try:
            # Check minimum data requirement
            if len(df) < self.lookback_hours + 10:
                logger.warning(f"Insufficient hourly data for {ticker}: {len(df)} points")
                return self._fallback_prediction(ticker)
            
            # Extract features and session mask
            features, session_mask = self.feature_extractor.extract_features(df)
            
            # Initialize model if needed
            if self.model is None:
                input_dim = features.shape[-1]
                self._initialize_model(input_dim)
            
            # Use only the latest window for prediction
            latest_features = features[-1:]  # (1, lookback_hours, n_features)
            latest_session_mask = session_mask[-1:]  # (1, lookback_hours, n_sessions)
            
            # Convert to tensors
            x = torch.FloatTensor(latest_features).to(self.device)
            session_tensor = torch.FloatTensor(latest_session_mask).to(self.device)
            
            # Generate prediction
            self.model.eval()
            with torch.no_grad():
                model_outputs = self.model(x, session_tensor)
            
            # Process outputs
            predictions = {}
            for horizon in self.prediction_horizons:
                horizon_key = f"horizon_{horizon}"
                model_out = model_outputs[horizon_key]
                
                # Extract predictions and convert to numpy
                prediction = float(model_out['prediction'][0].cpu().numpy())
                confidence = float(model_out['confidence'][0].cpu().numpy()) if return_confidence else 0.7
                volatility = float(model_out['volatility'][0].cpu().numpy())
                momentum = float(model_out['momentum'][0].cpu().numpy())
                event_impact = float(model_out['event_impact'][0].cpu().numpy())
                
                # Create prediction intervals using volatility
                interval_width = volatility * 1.96  # 95% confidence intervals
                lower_bound = prediction - interval_width
                upper_bound = prediction + interval_width
                
                predictions[horizon_key] = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'prediction_interval': [lower_bound, upper_bound],
                    'volatility_forecast': volatility,
                    'momentum_indicator': momentum,
                    'event_impact_estimate': event_impact,
                    'model_type': 'hourly_temporal_cnn'
                }
            
            # Add metadata
            predictions['hourly_meta'] = {
                'ticker': ticker,
                'data_points': len(df),
                'lookback_hours': self.lookback_hours,
                'features_used': features.shape[-1],
                'session_aware': True,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.debug(f"Generated hourly predictions for {ticker}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error in hourly prediction for {ticker}: {e}")
            return self._fallback_prediction(ticker)
    
    def _fallback_prediction(self, ticker: str) -> Dict[str, Any]:
        """Generate fallback predictions when model fails."""
        predictions = {}
        
        for horizon in self.prediction_horizons:
            horizon_key = f"horizon_{horizon}"
            
            # Simple trend-following prediction based on horizon
            base_return = 0.001 * horizon  # Slight positive bias for longer horizons
            prediction = np.random.normal(base_return, 0.005)  
            prediction = np.clip(prediction, -0.05, 0.05)  # Clip to ±5%
            
            predictions[horizon_key] = {
                'prediction': float(prediction),
                'confidence': 0.5,
                'prediction_interval': [float(prediction - 0.02), float(prediction + 0.02)],
                'volatility_forecast': 0.02,
                'momentum_indicator': 0.0,
                'event_impact_estimate': 0.0,
                'model_type': 'hourly_fallback'
            }
        
        predictions['hourly_meta'] = {
            'ticker': ticker,
            'fallback_used': True,
            'timestamp': datetime.now().isoformat()
        }
        
        return predictions
    
    def train_model(
        self,
        training_data: List[pd.DataFrame],
        tickers: List[str],
        epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Train the hourly prediction model.
        
        Args:
            training_data: List of DataFrames with hourly OHLCV data
            tickers: Corresponding ticker symbols
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            
        Returns:
            Training statistics and performance metrics
        """
        logger.info(f"Training hourly model on {len(training_data)} datasets")
        
        try:
            # Prepare training data
            X_train, y_train, session_masks = self._prepare_training_data(training_data, tickers)
            
            if len(X_train) < 5:
                logger.warning("Insufficient training data for hourly model")
                return {'error': 'insufficient_data'}
            
            # Initialize model if needed
            if self.model is None:
                input_dim = X_train.shape[-1]
                self._initialize_model(input_dim)
            
            # Training setup
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
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
                    session_batch = session_masks[batch_indices]
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(X_batch, session_batch)
                    
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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_losses.append(total_loss.item())
                
                avg_loss = np.mean(epoch_losses)
                training_losses.append(avg_loss)
                scheduler.step(avg_loss)
                
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            self.is_trained = True
            logger.info("Hourly model training completed")
            
            return {
                'training_samples': len(X_train),
                'epochs': epochs,
                'final_loss': training_losses[-1],
                'training_losses': training_losses,
                'model_trained': True
            }
            
        except Exception as e:
            logger.error(f"Error training hourly model: {e}")
            return {'error': str(e)}
    
    def _prepare_training_data(
        self,
        training_data: List[pd.DataFrame],
        tickers: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare training data from historical hourly data."""
        X_all = []
        y_all = {f"horizon_{h}": [] for h in self.prediction_horizons}
        session_masks_all = []
        
        for df, ticker in zip(training_data, tickers):
            try:
                min_required = self.lookback_hours + max(self.prediction_horizons) + 10
                if len(df) < min_required:
                    continue
                
                # Extract features and session masks
                features, session_masks = self.feature_extractor.extract_features(df)
                
                # Create targets for each horizon
                targets = {}
                for horizon in self.prediction_horizons:
                    # Calculate forward returns
                    forward_returns = df['close'].pct_change(periods=horizon).shift(-horizon)
                    
                    # Align with features
                    valid_targets = forward_returns.iloc[self.lookback_hours:self.lookback_hours + len(features)]
                    valid_targets = valid_targets.dropna()
                    
                    if len(valid_targets) != len(features):
                        min_len = min(len(features), len(valid_targets))
                        features = features[:min_len]
                        session_masks = session_masks[:min_len]
                        valid_targets = valid_targets.iloc[:min_len]
                    
                    targets[f"horizon_{horizon}"] = valid_targets.values
                
                # Add to training set
                if len(features) > 0 and all(len(t) > 0 for t in targets.values()):
                    X_all.append(features)
                    session_masks_all.append(session_masks)
                    for horizon in self.prediction_horizons:
                        y_all[f"horizon_{horizon}"].append(targets[f"horizon_{horizon}"])
                
            except Exception as e:
                logger.debug(f"Error processing hourly training data for {ticker}: {e}")
                continue
        
        if not X_all:
            raise ValueError("No valid training data found")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(np.concatenate(X_all, axis=0)).to(self.device)
        session_tensor = torch.FloatTensor(np.concatenate(session_masks_all, axis=0)).to(self.device)
        y_tensors = {}
        for horizon in self.prediction_horizons:
            horizon_key = f"horizon_{horizon}"
            y_tensors[horizon_key] = torch.FloatTensor(np.concatenate(y_all[horizon_key])).to(self.device)
        
        return X_tensor, y_tensors, session_tensor
    
    def save_model(self, path: str) -> bool:
        """Save the trained model."""
        try:
            if self.model is None:
                logger.warning("No model to save")
                return False
                
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'lookback_hours': self.lookback_hours,
                'prediction_horizons': self.prediction_horizons,
                'is_trained': self.is_trained
            }, path)
            
            logger.info(f"Hourly model saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving hourly model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load a trained model."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.lookback_hours = checkpoint['lookback_hours']
            self.prediction_horizons = checkpoint['prediction_horizons']
            self.is_trained = checkpoint['is_trained']
            
            # Initialize model structure
            if self.model is None:
                # Dummy input to get dimensions
                dummy_features = np.random.randn(1, self.lookback_hours, 16)
                input_dim = dummy_features.shape[-1]
                self._initialize_model(input_dim)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"Hourly model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading hourly model: {e}")
            return False
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'model_type': 'HourlyPredictor',
            'lookback_hours': self.lookback_hours,
            'prediction_horizons': self.prediction_horizons,
            'device': str(self.device),
            'is_loaded': self.is_loaded,
            'is_trained': self.is_trained,
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            'session_aware': True,
            'multi_scale_convolution': True
        }