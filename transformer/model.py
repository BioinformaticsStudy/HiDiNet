"""
HiDiNet: Hierarchical Diffusion Network for Healthcare Time Series Modeling
Main Model Architecture

This module implements the core HiDiNet model, which combines multiple deep learning
components for healthcare time series modeling:

1. **Variational Autoencoder (VAE) with Normalizing Flows**: For handling missing data
   and learning latent representations of patient states

2. **Stochastic Differential Equations (SDE)**: For modeling disease dynamics and 
   temporal evolution of health variables

3. **Transformer Architecture**: For capturing long-range dependencies and complex
   temporal patterns in health trajectories

4. **Memory Networks**: For maintaining patient-specific context and disease progression

5. **Survival Analysis Components**: For predicting mortality risk and survival probabilities

The model is designed to handle:
- Irregular time series with missing observations
- Multiple health variables with complex interactions
- Environmental factors and medication effects
- Survival analysis with censoring
- Long-term trajectory prediction with uncertainty quantification

Key Innovation: The model uses learned interaction matrices to capture causal relationships
between health variables, enabling interpretable disease progression modeling.

Authors: [Your names here]
Paper: [Paper title and venue]
"""

import torch
import torch.nn as nn

# Import specialized model components
from DJIN_Model.dynamics import SDEModel
from DJIN_Model.memory_model import Memory
from DJIN_Model.vae_flow import VAEImputeFlows
from DJIN_Model.solver import SolveSDE

# Import transformer components
from transformer.encoder import Encoder
from transformer.decoder import Decoder

class Model(nn.Module):
    """
    HiDiNet: Hierarchical Diffusion Network for Healthcare Time Series Modeling
    
    This is the main model class that orchestrates all components of the HiDiNet architecture.
    The model combines multiple deep learning techniques to handle the complexity of
    healthcare time series data.
    
    Architecture Overview:
    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │   Input Data    │ -> │  VAE + Flows     │ -> │  SDE Dynamics   │
    │ (Health, Meds,  │    │  (Imputation)    │    │  (Trajectories) │
    │  Environment)   │    └──────────────────┘    └─────────────────┘
    └─────────────────┘                                      │
                                                             v
    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │ Survival Pred.  │ <- │   Transformer    │ <- │ Visit Matrix    │
    │ (Hazard, Surv.) │    │ (Encoder+Decoder)│    │ (X, t combined) │
    └─────────────────┘    └──────────────────┘    └─────────────────┘
    
    Args:
        device (torch.device): Computing device (CPU/GPU)
        N (int): Number of health variables/features
        gamma_size (int): Size of gamma parameters for survival modeling
        z_size (int): Latent space dimension for VAE
        decoder_size (int): Hidden layer size for VAE decoder
        Nflows (int): Number of normalizing flow layers
        flow_hidden (int): Hidden dimension in flow layers
        f_nn_size (int): Neural network size in flow transformations
        mean_T (float): Mean age for time normalization
        std_T (float): Standard deviation of ages for normalization
        d_model (int): Transformer model dimension
        dropout (float): Dropout rate for transformer layers
        dt (float): Time step for SDE integration
        length (float): Prediction horizon in years
    """
    
    def __init__(self, device, N, gamma_size, z_size, decoder_size, Nflows, flow_hidden, f_nn_size, mean_T, std_T, d_model, dropout, dt=0.5, length=25):
        super(Model, self).__init__()

        # ==================== MODEL CONFIGURATION ====================
        
        # Core model parameters
        self.N = N                          # Number of health variables (e.g., 29 for ELSA dataset)
        self.gamma_size = gamma_size        # Size of gamma parameters for survival modeling
        self.mean_T = mean_T               # Mean age for time normalization
        self.std_T = std_T                 # Standard deviation of ages for normalization
        self.z_size = z_size               # Latent space dimension for VAE
        self.device = device               # Computing device
        self.d_model = d_model             # Transformer model dimension
        self.dropout = dropout             # Dropout rate for regularization

        # ==================== LEARNABLE PARAMETERS ====================
        
        # Interaction matrix parameters (W) - Core innovation of HiDiNet
        # These parameters define how health variables influence each other
        # W[i,j,k] represents the influence of variable j on the dynamics of variable i,
        # modulated by the current state of variable k
        self.register_parameter(
            name='mean', 
            param=nn.Parameter(0.03 * torch.randn(N, N, N))
        )
        self.register_parameter(
            name='logscale', 
            param=nn.Parameter(torch.log(0.03 * torch.ones(N, N, N)))
        )

        # Observation noise parameters (sigma_y) - Gamma distribution parameters
        # These model the uncertainty in health measurements
        # alpha and beta parameters for Gamma distribution: Gamma(alpha, beta)
        self.register_parameter(
            name='logalpha', 
            param=nn.Parameter(torch.log(10.0 * torch.ones(N)))
        )
        self.register_parameter(
            name='logbeta', 
            param=nn.Parameter(torch.log(100.0 * torch.ones(N)))
        )
        
        # ==================== MODEL COMPONENTS INITIALIZATION ====================
        
        # VAE with Normalizing Flows for missing data imputation
        # Handles irregular observations and provides uncertainty estimates
        self.impute = VAEImputeFlows(N, z_size, decoder_size, Nflows, flow_hidden, device).to(device)
        
        # Memory network for initial state processing
        # Combines patient demographics, medications, and initial health state
        # Input size: 10 (medications) + 26 (environmental/demographic variables)
        self.memory0 = Memory(N, 10 + 26, self.gamma_size).to(device)
        
        # SDE dynamics model for temporal evolution
        # Models how health variables change over time under various influences
        self.dynamics = SDEModel(N, device, 10 + 26, self.gamma_size, f_nn_size, mean_T, std_T).to(device)
        
        # SDE solver for numerical integration
        # Solves the stochastic differential equations to generate trajectories
        self.solver = SolveSDE(N, device, dt=dt, length=length).to(device)

        # ==================== TRANSFORMER COMPONENTS ====================
        
        # Transformer encoder - processes health trajectory sequences
        # Captures long-range dependencies and complex temporal patterns
        self.encoder = Encoder(device, d_model, dropout).to(device)
        
        # Transformer decoder - generates survival predictions
        # Takes encoded trajectories and produces hazard rates and survival probabilities
        self.decoder = Decoder(device, d_model, dropout, mean_T, std_T).to(device)
    
    def forward(self, data, sigma_y, test=False):
        """
        Forward pass through the HiDiNet model.
        
        This method orchestrates the entire prediction pipeline:
        1. Data preprocessing and missing value imputation
        2. Initial state reconstruction using VAE
        3. Health trajectory generation using SDE dynamics
        4. Transformer-based survival prediction
        
        Args:
            data (dict): Batch of patient data containing:
                - 'Y': Health measurements [B, T, N]
                - 'times': Time points [B, T]
                - 'mask': Observation mask [B, T, N]
                - 'mask0': Initial observation mask [B, N]
                - 'survival_mask': Survival data mask [B, T]
                - 'dead_mask': Death event mask [B, T]
                - 'after_dead_mask': Post-death mask [B, T]
                - 'censored': Censoring indicators [B, T]
                - 'env': Environmental/demographic variables [B, 26]
                - 'med': Medication data [B, T, 10]
                - 'weights': Sample importance weights [B]
                - 'missing': Population mean for missing values [B, N]
                - 'pop std': Population standard deviation [B, N]
            
            sigma_y (torch.Tensor): Sampled observation noise [B, T]
            test (bool): Whether in test mode (affects imputation strategy)
            
        Returns:
            dict: Model outputs containing:
                - 'pred_X': Predicted health trajectories [B, T, N]
                - 't': Time points [B, T]
                - 'pred_S': Survival probabilities [B, T]
                - 'pred_logGamma': Log hazard rates [B, T]
                - And many other intermediate outputs for loss computation
        """
        
        # ==================== DATA EXTRACTION AND PREPROCESSING ====================
        
        # Extract data tensors and move to device
        y = data['Y'].to(self.device)                           # Health measurements [B, T, N]
        times = data['times'].to(self.device)                   # Time points [B, T]
        mask = data['mask'].to(self.device)                     # Observation mask [B, T, N]
        mask0 = data['mask0'].to(self.device)                   # Initial observation mask [B, N]
        survival_mask = data['survival_mask'].to(self.device)   # Survival data mask [B, T]
        dead_mask = data['dead_mask'].to(self.device)           # Death event mask [B, T]
        after_dead_mask = data['after_dead_mask'].to(self.device)  # Post-death mask [B, T]
        censored = data['censored'].to(self.device)             # Censoring indicators [B, T]
        env = data['env'].to(self.device)                       # Environmental variables [B, 26]
        med = data['med'].to(self.device)                       # Medication data [B, T, 10]
        sample_weights = data['weights'].to(self.device)        # Importance sampling weights [B]
        predict_missing = data['missing'].to(self.device)       # Population mean for imputation [B, N]
        pop_std = data['pop std'].to(self.device)              # Population std for noise [B, N]

        batch_size = y.shape[0]
        
        # ==================== INITIAL STATE PROCESSING ====================
        
        # Extract initial timepoint data
        y0_ = y[:, 0, :]                                        # Initial health measurements [B, N]
        t0 = times[:, 0]                                        # Initial time [B]
        med0 = med[:, 0, :]                                     # Initial medications [B, 10]
        trans_t0 = (t0.unsqueeze(-1) - self.mean_T) / self.std_T  # Normalized time [B, 1]

        # Note: Environmental and medication data structure
        # env.shape: [B, 26] - includes 14 demographic values + 12 masks
        # med.shape: [B, T, 10] - includes 5 medication values + 5 masks
        
        # ==================== MISSING VALUE IMPUTATION ====================
        
        # Handle missing values at initial timepoint
        # Strategy differs between training and testing
        if test:
            # Test mode: Use observed values where available, impute with population mean + noise elsewhere
            y0 = mask[:, 0, :] * y0_ + (1 - mask[:, 0, :]) * (predict_missing + pop_std * torch.randn_like(y0_))
        else:
            # Training mode: Use mask0 for more aggressive corruption-based imputation
            y0 = mask0 * y0_ + (1 - mask0) * (predict_missing + pop_std * torch.randn_like(mask0))

        # ==================== VAE-BASED IMPUTATION ====================
        
        # Apply VAE with normalizing flows for sophisticated imputation
        # This captures complex dependencies between health variables
        if test:
            sample0, z_sample, mu0, logvar0, prior_entropy, log_det = self.impute(
                trans_t0, y0, mask[:, 0, :], env, med0
            )
        else:
            sample0, z_sample, mu0, logvar0, prior_entropy, log_det = self.impute(
                trans_t0, y0, mask0, env, med0
            )

        # Generate reconstructed initial state from VAE decoder
        # Combines latent representation with contextual information
        recon_mean_x0 = self.impute.decoder(torch.cat((z_sample, trans_t0, env, med0), dim=-1))
        
        # Create final initial state by combining observed and imputed values
        # Observed values take precedence where available
        x0 = mask[:, 0, :] * y0_ + (1 - mask[:, 0, :]) * recon_mean_x0
        
        # ==================== CONTEXT AND MEMORY INITIALIZATION ====================
        
        # Prepare context vector for dynamics model
        # Combines environmental factors and initial medications
        context = torch.cat((env, med0), dim=-1)  # [B, 36]
        
        # Initialize memory state using memory network
        # This processes patient context to initialize SDE dynamics
        h = self.memory0(trans_t0, x0, torch.cat((env, med0), dim=-1))
        
        # Split memory state for LSTM-style hidden states
        h1 = h[:, :self.gamma_size]                            # First part of hidden state
        h2 = h[:, self.gamma_size:]                            # Second part of hidden state
        h = (h1.unsqueeze(0).contiguous(), h2.unsqueeze(0).contiguous())  # LSTM format
        
        # ==================== HEALTH TRAJECTORY GENERATION ====================
        
        # Solve SDE to generate health trajectories over time
        # This is the core dynamics modeling component
        t, pred_X, _, _, pred_sigma_X, drifts = self.solver._solve(
            self.dynamics,      # SDE dynamics model
            x0,                # Initial state [B, N]
            t0,                # Initial time [B]
            batch_size,        # Batch size
            context,           # Context vector [B, 36]
            h,                 # Memory hidden states
            self.mean          # Interaction matrix mean parameters
        )
        
        # ==================== TRANSFORMER INPUT PREPARATION ====================
        
        # Create visit matrix for transformer input
        # Combines health trajectories with time information
        times_expanded = t.unsqueeze(-1)                        # Add time dimension [B, T, 1]
        visit_matrix = torch.cat((pred_X, times_expanded), dim=2)  # [B, T, N+1]
        
        # Validate tensor shapes for debugging
        batch_size = pred_X.shape[0]
        assert pred_X.shape == (batch_size, 50, 29), f"pred_X shape mismatch: {pred_X.shape}"
        assert times_expanded.shape == (batch_size, 50, 1), f"times shape mismatch: {times_expanded.shape}"
        assert visit_matrix.shape == (batch_size, 50, 30), f"visit_matrix shape mismatch: {visit_matrix.shape}"

        # ==================== TRANSFORMER PROCESSING ====================
        
        # Pass visit matrix through transformer encoder
        # Captures complex temporal dependencies in health trajectories
        encoded_vector = self.encoder(visit_matrix, times_expanded)
        assert encoded_vector.shape == (batch_size, 50, self.d_model), f"encoded_vector shape mismatch: {encoded_vector.shape}"

        # ==================== SURVIVAL PREDICTION ====================
        
        # Generate survival predictions using transformer decoder
        # Takes encoded trajectories and produces hazard rates and survival probabilities
        pred_S, pred_logGamma = self.decoder(encoded_vector, times_expanded, x0)

        # ==================== OUTPUT PREPARATION ====================
        
        # Return comprehensive output dictionary for loss computation and analysis
        return {
            # Core predictions
            'pred_X': pred_X,                           # Predicted health trajectories [B, T, N]
            't': t,                                     # Time points [B, T]
            'times': t,                                 # Time points (duplicate for compatibility)
            'pred_S': pred_S,                          # Survival probabilities [B, T]
            'pred_logGamma': pred_logGamma,            # Log hazard rates [B, T]
            
            # SDE-related outputs
            'sigma_X': pred_sigma_X,                   # Prediction uncertainties [B, T, N]
            'pred_sigma_X': pred_sigma_X,              # Prediction uncertainties (duplicate)
            'drifts': drifts,                          # SDE drift terms [B, T, N]
            
            # VAE-related outputs
            'z_sample': z_sample,                      # VAE latent samples [B, z_size]
            'prior_entropy': prior_entropy,            # Flow prior entropy [B]
            'log_det': log_det,                        # Flow log determinant [B]
            'recon_mean_x0': recon_mean_x0,           # Reconstructed initial state [B, N]
            
            # Context and intermediate representations
            'context': context,                        # Context vector [B, 36]
            'initial_state': x0,                       # Final initial state [B, N]
            'visit_matrix': visit_matrix,              # Transformer input [B, T, N+1]
            'encoded_vector': encoded_vector,          # Transformer encoding [B, T, d_model]
            
            # Input data (passed through for loss computation)
            'sigma_y': sigma_y,                        # Observation noise [B, T]
            'sample_weights': sample_weights,          # Importance weights [B]
            'survival_mask': survival_mask,            # Survival mask [B, T]
            'dead_mask': dead_mask,                    # Death mask [B, T]
            'after_dead_mask': after_dead_mask,        # Post-death mask [B, T]
            'censored': censored,                      # Censoring indicators [B, T]
            'mask': mask,                              # Observation mask [B, T, N]
            'mask0': mask0,                            # Initial observation mask [B, N]
            'y': y,                                    # Original health measurements [B, T, N]
            'med': med,                                # Medication data [B, T, 10]
        }