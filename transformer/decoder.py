"""
HiDiNet: Hierarchical Diffusion Network for Healthcare Time Series Modeling
Transformer Decoder Component

This module implements the transformer decoder component of the HiDiNet architecture.
The decoder generates survival predictions (hazard rates and survival probabilities)
from encoded health trajectory representations using an autoregressive approach.

Key Features:
- Autoregressive survival prediction with memory efficiency
- Incorporates baseline health state and previous predictions
- Uses transformer decoder architecture with cross-attention to encoded trajectories
- Generates hazard rates with numerical stability guarantees
- Computes survival probabilities with proper clamping

Architecture:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Baseline State  │    │ Previous Preds   │    │ Current Time    │
│ [B, 29]         │    │ [B, 1, 2]        │    │ [B, 1, 1]       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        v                       v                       v
┌─────────────────────────────────────────────────────────────────┐
│                    Output Embedding                             │
│            (Combines all inputs -> [B, 1, d_model])            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Encoded Memory  │ -> │ Transformer      │ -> │ Hazard Pred.    │
│ [B, 50, d_model]│    │ Decoder Layer    │    │ [B, 1, 1]       │
└─────────────────┘    └──────────────────┘    └─────────────────┘

The decoder processes one timestep at a time in an autoregressive manner,
using previous predictions as input for future predictions. This enables
stable long-term survival forecasting.

Authors: [Your names here]
Paper: [Paper title and venue]
"""

import torch
import torch.nn as nn
import math

class Decoder(nn.Module):
    """
    Transformer Decoder for Autoregressive Survival Prediction
    
    This decoder generates survival predictions from encoded health trajectories
    using an autoregressive approach. It processes one timestep at a time,
    incorporating previous predictions to maintain temporal consistency.
    
    The decoder is designed to handle:
    - Sequential survival prediction over long time horizons
    - Integration of baseline health state and temporal information
    - Memory-efficient processing (single timestep vs. full sequence)
    - Numerical stability in hazard rate and survival probability computation
    
    Architecture Details:
    - Input: Encoded trajectories [B, 50, d_model] from encoder
    - Output Embedding: Combines baseline state, time, and previous predictions
    - Transformer Decoder: 4 layers with 8 attention heads each
    - Hazard Prediction: Neural network with ELU activation
    - Output: Survival probabilities and log hazard rates
    
    Key Innovation: Memory-efficient autoregressive processing that handles
    each timestep independently while maintaining temporal dependencies.
    
    Args:
        device (torch.device): Computing device (CPU/GPU)
        d_model (int): Model dimension (typically 512)
        dropout (float): Dropout rate for regularization
        mean_T (float): Mean age for time normalization
        std_T (float): Standard deviation of ages for normalization
    """
    
    def __init__(self, device, d_model, dropout, mean_T, std_T):
        super(Decoder, self).__init__()

        # ==================== CONFIGURATION ====================
        
        self.device = device
        self.d_model = d_model      # Transformer model dimension
        self.dropout = dropout      # Dropout rate for regularization
        self.mean_T = mean_T       # Mean age for time normalization
        self.std_T = std_T         # Standard deviation of ages for normalization

        # ==================== COMPONENT INITIALIZATION ====================
        
        # Output embedding module - combines baseline state, time, and previous predictions
        self.output_embedding = self.OutputEmbedding(d_model, dropout, mean_T, std_T).to(device)
        
        # Positional encoding for temporal order (reused from encoder)
        self.positional_encoding = self.PositionalEncoding(d_model, dropout).to(device)

        # ==================== TRANSFORMER DECODER ====================
        
        # Standard transformer decoder layers with cross-attention
        # Uses GELU activation for better performance on healthcare data
        layers = nn.TransformerDecoderLayer(
            d_model=d_model,                    # Model dimension
            nhead=8,                           # Number of attention heads (standard)
            dim_feedforward=d_model*4,         # Feed-forward network size (4x expansion)
            dropout=dropout,                   # Dropout for regularization
            batch_first=True,                  # Input shape: [batch, seq_len, features]
            activation='gelu'                  # GELU activation (better than ReLU for this task)
        )

        # Stack multiple decoder layers for deeper representations
        # 4 layers balances capacity with computational efficiency
        self.transformer_decoder = nn.TransformerDecoder(
            layers,
            num_layers=4
        ).to(device)

        # ==================== SURVIVAL PREDICTION HEAD ====================
        
        # Neural network for hazard rate prediction
        # Uses ELU activation for smooth gradients and numerical stability
        intermediate_size = max(d_model - 15, 16)  # Adaptive intermediate size
        
        self.hazard_predictor = nn.Sequential(
            nn.Linear(d_model, intermediate_size),
            nn.ELU(),                          # ELU for smooth activation
            nn.Dropout(dropout),               # Regularization
            nn.Linear(intermediate_size, 1)    # Single output: log hazard rate
        ).to(device)
        
        # Initialize weights with standard transformer initialization
        self._init_weights()

    def _init_weights(self):
        """
        Initialize model weights using standard transformer initialization.
        
        This uses Xavier uniform initialization for linear layers, which is
        standard for transformer models and provides good training stability.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    # Initialize biases to zero
                    nn.init.zeros_(module.bias)

    def generate_causal_mask(self, seq_len):
        """
        Generate causal mask for autoregressive decoding.
        
        This mask prevents the model from attending to future positions,
        ensuring that predictions at time t only depend on information
        from times 0 to t-1.
        
        Args:
            seq_len (int): Length of the sequence
            
        Returns:
            torch.Tensor: [seq_len, seq_len] causal mask 
                         (True where attention should be masked)
        """
        # Create upper triangular mask (mask future positions)
        # diagonal=1 means the diagonal itself is not masked (can attend to current position)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask

    def forward(self, encoded_vector, times, x0):
        """
        Memory-efficient autoregressive decoder for survival prediction.
        
        This method processes each timestep independently in an autoregressive manner,
        using previous predictions as input features for future predictions. This
        approach is memory-efficient and enables stable long-term forecasting.
        
        Processing Pipeline (for each timestep t):
        1. Create output embedding from baseline state, current time, and previous predictions
        2. Add positional encoding for current timestep
        3. Process through transformer decoder with cross-attention to encoded memory
        4. Predict hazard rate with numerical stability
        5. Compute survival probability from hazard rate
        6. Use current predictions as input for next timestep
        
        Args:
            encoded_vector (torch.Tensor): [B, 50, d_model] encoded health trajectories
                                          from the transformer encoder
            times (torch.Tensor): [B, 50, 1] time points (ages) with extra dimension
            x0 (torch.Tensor): [B, 29] baseline health state at initial timepoint
        
        Returns:
            tuple: (pred_S, log_Gamma) where:
                - pred_S: [B, 50] survival probabilities at each timestep
                - log_Gamma: [B, 50] log hazard rates at each timestep
        """
        
        # ==================== INPUT VALIDATION ====================
        
        # Validate input tensor shapes and dimensions
        assert encoded_vector.shape[1] == 50, f"Expected seq_len=50, got {encoded_vector.shape[1]}"
        assert times.shape[1] == 50, f"Expected seq_len=50, got {times.shape[1]}"
        assert times.shape[2] == 1, f"Expected times last dim=1, got {times.shape[2]}"
        assert x0.shape[1] == 29, f"Expected 29 features, got {x0.shape[1]}"
        assert encoded_vector.shape[0] == times.shape[0] == x0.shape[0], "Batch sizes must match"
        
        batch_size, seq_len, _ = encoded_vector.shape
        device = encoded_vector.device
        
        # ==================== OUTPUT STORAGE INITIALIZATION ====================
        
        # Pre-allocate lists for collecting predictions across timesteps
        all_logGamma = []       # Log hazard rates for each timestep
        all_pred_S = []         # Survival probabilities for each timestep
        
        # ==================== AUTOREGRESSIVE PREDICTION LOOP ====================
        
        # Process each timestep sequentially in autoregressive manner
        for t in range(seq_len):
            
            # Extract current timepoint (maintaining batch and sequence dimensions)
            current_time = times[:, t:t+1]  # [B, 1, 1] - keep all dimensions
            assert current_time.shape == (batch_size, 1, 1), f"Current time shape mismatch: {current_time.shape}"

            # ==================== PREVIOUS PREDICTION HANDLING ====================
            
            if t == 0:
                # First timestep: no previous predictions available
                prev_predictions = None
            else:
                # Subsequent timesteps: use previous predictions as input features
                assert len(all_logGamma) > 0, "No previous logGamma available"
                assert len(all_pred_S) > 0, "No previous pred_S available"
                
                # Combine previous hazard rate and survival probability
                prev_predictions = torch.cat([
                    all_logGamma[-1].unsqueeze(1),  # [B, 1, 1] - previous log hazard
                    all_pred_S[-1].unsqueeze(1),    # [B, 1, 1] - previous survival prob
                ], dim=-1)  # [B, 1, 2]
                assert prev_predictions.shape == (batch_size, 1, 2), f"Previous predictions shape mismatch: {prev_predictions.shape}"
            
            # ==================== OUTPUT EMBEDDING GENERATION ====================
            
            # Create output embedding for current timestep
            # This combines baseline state, current time, and previous predictions
            current_embedding = self.output_embedding(
                x0,                 # [B, 29] - baseline health state
                current_time,       # [B, 1, 1] - current time
                prev_predictions    # [B, 1, 2] or None - previous predictions
            )  # -> [B, 1, d_model]
            assert current_embedding.shape == (batch_size, 1, self.d_model), f"current embedding shape mismatch: {current_embedding.shape}"
        
            # ==================== POSITIONAL ENCODING ====================
            
            # Add positional encoding for current timestep position
            # This helps the model understand its position in the sequence
            current_embedding_with_pos = current_embedding + self.positional_encoding.pe[t:t+1, :].unsqueeze(0)
            current_embedding_with_pos = self.positional_encoding.dropout(current_embedding_with_pos)
            assert current_embedding_with_pos.shape == (batch_size, 1, self.d_model), f"Positional encoding shape mismatch: {current_embedding_with_pos.shape}"

            # ==================== TRANSFORMER DECODER PROCESSING ====================
            
            # Process single timestep through transformer decoder
            # Memory efficient: only processes [B, 1, d_model] instead of [B, t+1, d_model]
            decoded = self.transformer_decoder(
                tgt=current_embedding_with_pos,    # [B, 1, d_model] - current timestep query
                memory=encoded_vector,             # [B, 50, d_model] - full health trajectory context
                tgt_mask=None,                     # No mask needed for single timestep
            )  # -> [B, 1, d_model]
            assert decoded.shape == (batch_size, 1, self.d_model), f"Decoded shape mismatch: {decoded.shape}"

            # ==================== HAZARD RATE PREDICTION ====================
            
            # Predict raw hazard rate from decoded representation
            raw_logGamma = self.hazard_predictor(decoded).squeeze(-1)  # [B, 1]
            assert raw_logGamma.shape == (batch_size, 1), f"raw logGamma shape mismatch: {raw_logGamma.shape}"
            
            # Apply numerical stability constraints to prevent NaN in exp(-exp(x))
            # Clamp to reasonable range to avoid numerical overflow/underflow
            current_logGamma = torch.clamp(raw_logGamma, min=-10.0, max=10.0)
            assert current_logGamma.shape == (batch_size, 1), f"logGamma shape mismatch: {current_logGamma.shape}"
            
            # ==================== SURVIVAL PROBABILITY COMPUTATION ====================
            
            # Compute survival probability with numerical stability
            # S(t) = exp(-Λ(t)) where Λ(t) is cumulative hazard
            # For discrete time: S(t) = exp(-h(t)) where h(t) is hazard rate
            
            # Convert log hazard to hazard rate with clamping
            hazard_rate = torch.exp(current_logGamma)
            hazard_rate = torch.clamp(hazard_rate, min=1e-8, max=50.0)  # Prevent extreme values
            
            # Compute survival probability
            current_pred_S = torch.exp(-hazard_rate)
            current_pred_S = torch.clamp(current_pred_S, min=1e-8, max=1.0 - 1e-8)  # Keep in valid probability range
            assert current_pred_S.shape == (batch_size, 1), f"pred_S shape mismatch: {current_pred_S.shape}"

            # ==================== RESULT STORAGE ====================
            
            # Store predictions for current timestep
            all_logGamma.append(current_logGamma)
            all_pred_S.append(current_pred_S)

        # ==================== OUTPUT PREPARATION ====================
        
        # Stack all predictions across timesteps
        log_Gamma = torch.cat(all_logGamma, dim=1)  # [B, seq_len]
        pred_S = torch.cat(all_pred_S, dim=1)       # [B, seq_len]
        
        # Final shape validation
        assert log_Gamma.shape == (batch_size, seq_len), f"Final logGamma shape mismatch: {log_Gamma.shape}"
        assert pred_S.shape == (batch_size, seq_len), f"Final pred_S shape mismatch: {pred_S.shape}"
        
        return pred_S, log_Gamma

    class OutputEmbedding(nn.Module):
        """
        Output Embedding Module for Autoregressive Decoder
        
        This module creates embeddings for the decoder by combining multiple sources
        of information: baseline health state, current time, and previous predictions.
        It's designed to provide rich contextual information for survival prediction.
        
        Components:
        - Baseline State: Patient's initial health measurements
        - Current Time: Normalized age at current timestep
        - Previous Predictions: Hazard rate and survival probability from previous timestep
        
        The combination enables the model to make informed predictions that consider
        both the patient's health trajectory and recent survival estimates.
        
        Args:
            d_model (int): Model dimension for output embeddings
            dropout (float): Dropout rate for regularization
            mean_T (float): Mean age for time normalization
            std_T (float): Standard deviation of ages for normalization
        """
        
        def __init__(self, d_model, dropout, mean_T, std_T):
            super().__init__()
            
            # ==================== CONFIGURATION ====================
            
            self.d_model = d_model
            self.dropout = nn.Dropout(dropout)
            self.mean_T = mean_T       # For time normalization
            self.std_T = std_T         # For time normalization
            
            # ==================== PROJECTION LAYERS ====================
            
            # Project baseline health state to model dimension
            # 29 health variables -> d_model dimensions
            self.baseline_projection = nn.Linear(29, d_model)
            
            # Project normalized time to model dimension
            # 1 time variable -> d_model dimensions
            self.time_projection = nn.Linear(1, d_model)
            
            # Project previous predictions to model dimension
            # 2 previous predictions (log hazard + survival prob) -> d_model dimensions
            self.prev_pred_projection = nn.Linear(2, d_model)
            
            # Combine all projections into final embedding
            # 3 * d_model -> d_model (baseline + time + previous predictions)
            self.combine = nn.Linear(d_model * 3, d_model)

        def forward(self, x0, current_time, prev_predictions=None):
            """
            Create output embedding from multiple information sources.
            
            This method combines baseline health state, current time, and previous
            predictions into a rich embedding that provides context for survival
            prediction at the current timestep.
            
            Args:
                x0 (torch.Tensor): [B, 29] baseline health state
                current_time (torch.Tensor): [B, 1, 1] current timepoint  
                prev_predictions (torch.Tensor or None): [B, 1, 2] previous predictions
                                                        (log hazard + survival prob)
            
            Returns:
                torch.Tensor: [B, 1, d_model] output embedding for decoder input
            """
            
            # ==================== INPUT VALIDATION ====================
            
            device = x0.device
            batch_size = x0.shape[0]
            
            # Validate input tensor shapes
            assert x0.shape == (batch_size, 29), f"x0 shape mismatch: {x0.shape}"
            assert current_time.shape == (batch_size, 1, 1), f"current_time shape mismatch: {current_time.shape}"
            if prev_predictions is not None:
                assert prev_predictions.shape == (batch_size, 1, 2), f"prev_predictions shape mismatch: {prev_predictions.shape}"
            
            # ==================== BASELINE STATE PROJECTION ====================
            
            # Project baseline health state to model dimension
            baseline = self.baseline_projection(x0)  # [B, d_model]
            assert baseline.shape == (batch_size, self.d_model), f"baseline projection shape mismatch: {baseline.shape}"
            
            # ==================== TIME PROJECTION ====================
            
            # Normalize time using dataset statistics
            # This ensures time values are in a reasonable range for the model
            normalized_time = (current_time - self.mean_T) / self.std_T
            
            # Project normalized time to model dimension
            time_emb = self.time_projection(normalized_time.squeeze(-1))  # [B, d_model]
            assert time_emb.shape == (batch_size, self.d_model), f"time projection shape mismatch: {time_emb.shape}"

            # ==================== PREVIOUS PREDICTIONS PROJECTION ====================
            
            # Project previous predictions if available, otherwise use zeros
            if prev_predictions is not None:
                # Project previous hazard rate and survival probability
                prev_emb = self.prev_pred_projection(prev_predictions.squeeze(1))  # [B, d_model]
            else:
                # First timestep: no previous predictions available
                prev_emb = torch.zeros_like(baseline)  # [B, d_model]
            assert prev_emb.shape == (batch_size, self.d_model), f"prev_emb shape mismatch: {prev_emb.shape}"
            
            # ==================== COMBINATION AND OUTPUT ====================
            
            # Concatenate all projections
            combined = torch.cat([
                baseline,                    # [B, d_model] - patient baseline
                time_emb,                    # [B, d_model] - current time
                prev_emb                     # [B, d_model] - previous predictions
            ], dim=-1)  # [B, d_model*3]
            assert combined.shape == (batch_size, self.d_model * 3), f"combined shape mismatch: {combined.shape}"
            
            # Project combined features to final embedding dimension
            output = self.combine(combined)  # [B, d_model]
            assert output.shape == (batch_size, self.d_model), f"output shape mismatch: {output.shape}"
            
            # Add sequence dimension for transformer compatibility
            output = output.unsqueeze(1)  # [B, 1, d_model]
            assert output.shape == (batch_size, 1, self.d_model), f"final output shape mismatch: {output.shape}"
            
            # Apply dropout for regularization
            return self.dropout(output)

    class PositionalEncoding(nn.Module):
        """
        Sinusoidal Positional Encoding for Transformer Decoder
        
        This class implements the same positional encoding as used in the encoder.
        It provides position information to help the transformer understand the
        temporal order of the sequence.
        
        The encoding uses sine and cosine functions of different frequencies to
        create unique position representations that the model can learn to use
        for temporal reasoning.
        
        Args:
            d_model (int): Model dimension
            dropout (float): Dropout rate applied after adding positional encoding
            max_len (int): Maximum sequence length supported (default: 5000)
        """
        
        def __init__(self, d_model, dropout, max_len=5000):
            super().__init__()
            
            # Dropout layer for regularization
            self.dropout = nn.Dropout(dropout)
            
            # ==================== POSITIONAL ENCODING COMPUTATION ====================
            
            # Create positional encoding matrix [max_len, d_model]
            pe = torch.zeros(max_len, d_model)
            
            # Create position indices [max_len, 1]
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            
            # Create frequency terms for sine/cosine functions
            # Uses exponentially decreasing frequencies
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                                (-math.log(10000.0) / d_model))
            
            # Apply sine to even indices (0, 2, 4, ...)
            pe[:, 0::2] = torch.sin(position * div_term)
            
            # Apply cosine to odd indices (1, 3, 5, ...)
            # Handle case where d_model is odd
            if d_model % 2 == 0:
                pe[:, 1::2] = torch.cos(position * div_term)
            else:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            
            # Register as buffer (moves with model to different devices)
            self.register_buffer('pe', pe)

        def forward(self, x):
            """
            Add positional encodings to input embeddings.
            
            Args:
                x (torch.Tensor): [B, seq_len, d_model] input embeddings
                
            Returns:
                torch.Tensor: [B, seq_len, d_model] input with positional encodings added
            """
            
            # Get sequence length
            seq_len = x.size(1)
            
            # Add positional encoding (broadcast across batch dimension)
            x = x + self.pe[:seq_len, :].unsqueeze(0)
            
            # Apply dropout for regularization
            return self.dropout(x)