"""
HiDiNet: Hierarchical Diffusion Network for Healthcare Time Series Modeling
Prediction Script for Trained Transformer Model

This script performs inference using a trained HiDiNet model to generate:
- Future health trajectory predictions with uncertainty quantification
- Survival probability estimates over time
- Long-term disease progression forecasts

The prediction process involves:
1. Loading a trained model from a specific epoch
2. Running multiple Monte Carlo simulations for uncertainty estimation
3. Generating trajectories for specified time horizons
4. Computing survival statistics and confidence intervals
5. Saving results for downstream analysis

Key Features:
- Handles irregular time series with missing data
- Incorporates medication effects in predictions
- Provides uncertainty quantification through ensemble predictions
- Supports both ELSA and sample datasets

Authors: [Your names here]
Paper: [Paper title and venue]
"""

import argparse
import torch
from torch.nn import functional as F
import numpy as np
from scipy.stats import sem
from pandas import read_csv
from torch.utils import data
import os
import sys

# Add project root to Python path for imports
file = os.path.realpath(__file__)
package_root_directory = os.path.dirname(os.path.dirname(file))
sys.path.append(package_root_directory)

# Model and utility imports
from transformer.model import Model
from Utils.record import record
from DataLoader.dataset import Dataset
from DataLoader.collate import custom_collate

# ==================== COMMAND LINE ARGUMENTS ====================

parser = argparse.ArgumentParser('HiDiNet Prediction - Generate health trajectories from trained model')

# Model identification parameters
parser.add_argument('--job_id', type=int, required=True, help='Job ID of the trained model to load')
parser.add_argument('--epoch', type=int, required=True, help='Specific epoch/checkpoint to load for inference')

# Output configuration
parser.add_argument('--output_dir', type=str, default=None, help='Output directory for saving prediction results')

# Model architecture parameters (must match training configuration)
parser.add_argument('--gamma_size', type=int, default=25, help='Size of gamma parameters for survival modeling')
parser.add_argument('--z_size', type=int, default=20, help='Latent space dimension for VAE')
parser.add_argument('--decoder_size', type=int, default=65, help='Decoder hidden layer size')
parser.add_argument('--Nflows', type=int, default=3, help='Number of normalizing flow layers')
parser.add_argument('--flow_hidden', type=int, default=24, help='Hidden dimension in flow layers')
parser.add_argument('--f_nn_size', type=int, default=12, help='Neural network size in flow transformations')
parser.add_argument('--W_prior_scale', type=float, default=0.1, help='Scale parameter for Laplace prior on interaction weights')

# Dataset configuration
parser.add_argument('--dataset', type=str, choices=['elsa', 'sample'], default='elsa', 
                   help='Dataset that was used to train the model (determines file paths and normalization)')

args = parser.parse_args()

# ==================== CONFIGURATION AND SETUP ====================

# Determine file postfix based on dataset
postfix = '_sample' if args.dataset == 'sample' else ''
dir = os.path.dirname(os.path.realpath(__file__))

# Set number of CPU threads for PyTorch operations
torch.set_num_threads(6)

# Device configuration - use GPU if available for faster inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== MODEL AND PREDICTION PARAMETERS ====================

# Dataset configuration
N = 29  # Number of health variables/features in the dataset

# Monte Carlo simulation parameters
sims = 250  # Number of simulation runs for uncertainty quantification

# Time integration parameters
dt = 0.5  # Time step for SDE integration (must match training)
length = 25  # Prediction horizon in years

# Model architecture parameters (must match training configuration)
d_model = 512  # Transformer model dimension
dropout = 0.1  # Dropout rate (not used during inference)

print(f"Prediction configuration:")
print(f"  - Prediction horizon: {length} years")
print(f"  - Time step: {dt} years")
print(f"  - Monte Carlo simulations: {sims}")
print(f"  - Total time points: {int(length/dt)}")

# ==================== DATA NORMALIZATION SETUP ====================

# Load population statistics for data normalization
# These must match the statistics used during training
print("Loading population normalization statistics...")
pop_avg = np.load(dir + f'/../Data/Population_averages{postfix}.npy')
pop_avg_env = np.load(dir + f'/../Data/Population_averages_env{postfix}.npy')
pop_std = np.load(dir + f'/../Data/Population_std{postfix}.npy')
pop_std_env = np.load(dir + f'/../Data/Population_std_env{postfix}.npy')

# Convert to PyTorch tensors for GPU computation
pop_avg_ = torch.from_numpy(pop_avg[..., 1:]).float()  # Drop age bin centers
pop_avg_env = torch.from_numpy(pop_avg_env).float()    # Environmental variables
pop_std = torch.from_numpy(pop_std[..., 1:]).float()   # Standard deviations

# Age bin centers for reference (not used in computation)
pop_avg_bins = np.arange(40, 105, 3)[:-2]

# ==================== TEST DATASET LOADING ====================

# Load test dataset for prediction
test_name = dir + f'/../Data/test.csv'
test_set = Dataset(test_name, N, pop=False, min_count=10)
num_test = 400  # Batch size for prediction (can process multiple patients at once)

print(f"Loaded test dataset: {len(test_set)} patients")

def custom_collate_extended(x, pop_avg, pop_avg_env, pop_std, corruption):
    """
    Extended collate function that handles medication data for long-term predictions.
    
    This function extends the standard collate function to handle cases where
    we need to predict beyond the available medication data by repeating
    the last known medication state.
    
    Args:
        x: Batch of patient data
        pop_avg: Population averages for normalization
        pop_avg_env: Environmental variable averages
        pop_std: Population standard deviations
        corruption: Data corruption rate (set to 1.0 for no corruption during inference)
        
    Returns:
        dict: Extended batch with properly sized medication data
    """
    # Call the standard custom_collate function
    batch = custom_collate(x, pop_avg, pop_avg_env, pop_std, corruption)
    
    # Handle medication data extension for long-term predictions
    # Original med shape: [B, T_original, 10] where 10 is number of medication types
    # Model will double the time dimension with repeat_interleave(2)
    # So we need: [B, T_extended/2, 10] where T_extended = length/dt
    B = batch['med'].shape[0]
    T_original = batch['med'].shape[1]
    T_needed = int(length/dt/2)  # This should be 25 (will become 50 after repeat_interleave)
    
    if T_needed > T_original:
        # Extend medication data by repeating the last known medication state
        # This assumes patients continue their last prescribed medications
        last_med = batch['med'][:, -1:, :].repeat(1, T_needed - T_original, 1)
        batch['med'] = torch.cat([batch['med'], last_med], dim=1)
        print(f"Extended medication data from {T_original} to {T_needed} time points")
    elif T_needed < T_original:
        # Truncate if we have more medication data than needed
        batch['med'] = batch['med'][:, :T_needed, :]
        print(f"Truncated medication data from {T_original} to {T_needed} time points")
    
    return batch

# Create data loader with extended collate function
test_generator = data.DataLoader(
    test_set, 
    batch_size=num_test, 
    shuffle=False,  # Keep deterministic order for reproducible results
    collate_fn=lambda x: custom_collate_extended(x, pop_avg_, pop_avg_env, pop_std, 1.0)  # No corruption
)

# Get age normalization statistics from test set
mean_T = test_set.mean_T  # Mean age for normalization
std_T = test_set.std_T    # Standard deviation of ages

# ==================== MODEL LOADING ====================

print(f"Loading trained model: job_id={args.job_id}, epoch={args.epoch}")

# Initialize model with same architecture as training
model = Model(
    device, N, args.gamma_size, args.z_size, args.decoder_size, 
    args.Nflows, args.flow_hidden, args.f_nn_size, 
    mean_T, std_T, d_model, dropout, 
    dt=dt, length=length  # Specify prediction parameters
).to(device)

# Load trained model weights
model_path = dir + '/Parameters_elsa/train%d_Model_LAMP_epoch%d.params' % (args.job_id, args.epoch)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.eval()  # Set to evaluation mode

print(f"Model loaded successfully from: {model_path}")

# ==================== RESULT STORAGE INITIALIZATION ====================

# Initialize arrays to store prediction results
# Shape: [num_patients, max_time_points, features]
num_patients = len(test_set)
max_time_points = 100  # Maximum number of yearly time points to store

# Mean trajectories: [patients, time_points, features+1] (+1 for time column)
mean_results = np.zeros((num_patients, max_time_points, N+1)) * np.nan

# Standard deviation trajectories: [patients, time_points, features+1]
std_results = np.zeros((num_patients, max_time_points, N+1)) * np.nan

# Survival trajectories: [patients, time_points, 3] (time, mean_survival, std_survival)
S_results = np.zeros((num_patients, max_time_points, 3)) * np.nan

print(f"Initialized result arrays for {num_patients} patients")

# ==================== PREDICTION LOOP ====================

print("Starting prediction process...")

with torch.no_grad():  # Disable gradients for faster inference
    
    # Sample observation noise from learned posterior distribution
    # This captures the model's uncertainty about measurement noise
    sigma_posterior = torch.distributions.gamma.Gamma(model.logalpha.exp(), model.logbeta.exp())
    
    start = 0  # Index to track patient position in results arrays
    
    for batch_idx, data in enumerate(test_generator):
        print(f"Processing batch {batch_idx + 1}/{len(test_generator)}")
        
        # Move data to device
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        
        batch_size = data['Y'].shape[0]
        
        # Initialize arrays for Monte Carlo simulations
        # Shape: [simulations, patients, time_points, features]
        X = torch.zeros(sims, batch_size, int(length/dt), N).to(device)           # Mean predictions
        X_std = torch.zeros(sims, batch_size, int(length/dt), N).to(device)      # Noisy predictions
        S = torch.zeros(sims, batch_size, int(length/dt)).to(device)             # Survival probabilities
        alive = torch.ones(sims, batch_size, int(length/dt)).to(device)          # Alive indicators
        
        # ==================== MONTE CARLO SIMULATION LOOP ====================
        
        for s in range(sims):
            if s % 50 == 0:  # Progress reporting
                print(f'  Running simulation {s+1}/{sims}')
            
            # Sample noise parameters for this simulation
            sigma_y = sigma_posterior.sample((batch_size, int(length/dt)))
            
            # Forward pass through the model
            outputs = model(data, sigma_y, test=True)

            # Unpack model outputs
            pred_X = outputs['pred_X']                    # Predicted health trajectories
            t = outputs['t']                              # Time points
            sigma_X = outputs['pred_sigma_X']             # Prediction uncertainties
            drifts = outputs['drifts']                    # SDE drift terms
            context = outputs['context']                  # Transformer context
            z_sample = outputs['z_sample']                # VAE latent samples
            prior_entropy = outputs['prior_entropy']      # Flow prior entropy
            log_det = outputs['log_det']                  # Flow log determinant
            pred_S = outputs['pred_S']                    # Survival probabilities
            pred_logGamma = outputs['pred_logGamma']      # Hazard function parameters
            recon_mean_x0 = outputs['recon_mean_x0']     # Reconstructed initial states
            sample_weights = outputs['sample_weights']    # Importance sampling weights
            survival_mask = outputs['survival_mask']      # Survival data mask
            dead_mask = outputs['dead_mask']              # Death event mask
            after_dead_mask = outputs['after_dead_mask']  # Post-death mask
            censored = outputs['censored']                # Censoring indicators
            mask = outputs['mask']                        # Observation mask
            mask0 = outputs['mask0']                      # Initial observation mask
            y = outputs['y']                              # Observed values
            med = outputs['med']                          # Medication data
            pred_sigma_X = outputs['pred_sigma_X']        # Predicted uncertainties
            
            # Store simulation results
            X[s] = pred_X  # Mean trajectory predictions
            X_std[s] = pred_X + sigma_y * torch.randn_like(pred_X)  # Add observation noise
            
            # ==================== SURVIVAL PREDICTION HANDLING ====================
            
            # Handle survival predictions - they may have different time dimensions
            # due to the model's internal time handling
            if pred_S.shape[1] != int(length/dt):
                if pred_S.shape[1] == int(length/dt/2):  # If it's exactly half the expected length
                    # Repeat each time point to match expected length
                    S[s] = pred_S.repeat_interleave(2, dim=1)
                    # Generate death events using Bernoulli sampling from hazard rates
                    hazard_rates = pred_logGamma.repeat_interleave(2, dim=1).exp()
                    death_probs = torch.exp(-1 * hazard_rates[:, :-1] * dt)
                    alive[s, :, 1:] = torch.cumprod(torch.bernoulli(death_probs), dim=1)
                else:
                    # Handle other mismatched dimensions
                    print(f"Warning: pred_S has shape {pred_S.shape}, expected time dimension {int(length/dt)}")
                    # Use available predictions and pad/truncate as needed
                    min_time = min(pred_S.shape[1], int(length/dt))
                    S[s, :, :min_time] = pred_S[:, :min_time]
                    if pred_S.shape[1] > 1:
                        hazard_rates = pred_logGamma.exp()
                        death_probs = torch.exp(-1 * hazard_rates[:, :-1] * dt)
                        alive[s, :, 1:min_time] = torch.cumprod(torch.bernoulli(death_probs), dim=1)
            else:
                # Dimensions match - use predictions directly
                S[s] = pred_S
                hazard_rates = pred_logGamma.exp()
                death_probs = torch.exp(-1 * hazard_rates[:, :-1] * dt)
                alive[s, :, 1:] = torch.cumprod(torch.bernoulli(death_probs), dim=1)
            
        # ==================== TRAJECTORY RECORDING ====================
        
        # Extract predictions at yearly intervals for comparison with real data
        t0 = t[:, 0]  # Initial time for each patient
        
        # Define recording times (yearly from initial age to 110)
        record_times = [torch.from_numpy(np.arange(t0[b].cpu(), 110, 1)).to(device) 
                       for b in range(batch_size)]
        
        # Record trajectories at specified time points
        X_record, S_record = record(t, X, S, record_times, dt)
        X_std_record, alive_record = record(t, X_std, alive, record_times, dt)
        
        t0 = t0.cpu()  # Move to CPU for numpy operations
        
        # ==================== TRAJECTORY AGGREGATION ====================
        
        # Aggregate trajectories across simulations, weighted by survival status
        X_sum = []          # Sum of alive-weighted trajectories
        X_sum_std = []      # Sum of alive-weighted noisy trajectories
        X_sum2 = []         # Sum of squares for variance calculation
        X_count = []        # Count of alive patients at each time point
        
        for b in range(batch_size):
            # Weight trajectories by alive status and sum across simulations
            X_sum.append(torch.sum(X_record[b].permute(2,0,1) * alive_record[b], dim=1).cpu())
            X_sum_std.append(torch.sum(X_std_record[b].permute(2,0,1) * alive_record[b], dim=1).cpu())
            X_sum2.append(torch.sum(X_std_record[b].pow(2).permute(2,0,1) * alive_record[b], dim=1).cpu())
            X_count.append(torch.sum(alive_record[b], dim=0).cpu())

        # ==================== RESULT STORAGE ====================
        
        # Store results for each patient in the batch
        for b in range(batch_size):
            patient_idx = start + b
            time_points = np.arange(t0[b], 110, 1)
            n_times = len(time_points)
            
            # Store time points in first column
            mean_results[patient_idx, :n_times, 0] = time_points
            std_results[patient_idx, :n_times, 0] = time_points
            S_results[patient_idx, :n_times, 0] = time_points
            
            # Store mean trajectories (alive-weighted averages)
            if X_count[b].max() > 0:  # Ensure we have valid counts
                mean_trajectories = (X_sum[b] / X_count[b]).permute(1, 0).numpy()
                mean_results[patient_idx, :mean_trajectories.shape[0], 1:] = mean_trajectories
                
                # Store standard deviations using sample variance formula
                variance = (X_sum2[b] / X_count[b] - (X_sum_std[b] / X_count[b]).pow(2))
                std_trajectories = torch.sqrt(torch.clamp(variance, min=0)).permute(1, 0).numpy()
                std_results[patient_idx, :std_trajectories.shape[0], 1:] = std_trajectories
            
            # Store survival statistics
            S_results[patient_idx, :n_times, 1] = torch.mean(S_record[b], dim=0).cpu().numpy()  # Mean survival
            S_results[patient_idx, :n_times, 2] = torch.std(S_record[b], dim=0).cpu().numpy()   # Std survival
        
        start += batch_size
        print(f"  Completed batch {batch_idx + 1}, processed {start}/{num_patients} patients")

print("Prediction process completed!")

# ==================== RESULT SAVING ====================

print("Saving prediction results...")

if args.output_dir is not None:
    # Save to specified output directory
    analysis_dir = os.path.join(args.output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    mean_path = os.path.join(analysis_dir, f"Mean_trajectories_job_id{args.job_id}_epoch{args.epoch}_LAMP.npy")
    std_path = os.path.join(analysis_dir, f"Std_trajectories_job_id{args.job_id}_epoch{args.epoch}_LAMP.npy")
    survival_path = os.path.join(analysis_dir, f"Survival_trajectories_job_id{args.job_id}_epoch{args.epoch}_LAMP.npy")
    
    np.save(mean_path, mean_results)
    np.save(std_path, std_results)
    np.save(survival_path, S_results)
    
    print(f"Results saved to {analysis_dir}")
else:
    # Save to default directory structure
    analysis_dir = dir + '/Analysis_Data_elsa'
    os.makedirs(analysis_dir, exist_ok=True)
    
    mean_path = f'{analysis_dir}/Mean_trajectories_job_id{args.job_id}_epoch{args.epoch}_LAMP.npy'
    std_path = f'{analysis_dir}/Std_trajectories_job_id{args.job_id}_epoch{args.epoch}_LAMP.npy'
    survival_path = f'{analysis_dir}/Survival_trajectories_job_id{args.job_id}_epoch{args.epoch}_LAMP.npy'
    
    np.save(mean_path, mean_results)
    np.save(std_path, std_results)
    np.save(survival_path, S_results)
    
    print(f"Results saved to {analysis_dir}")

print("Prediction script completed successfully!")
print(f"Generated predictions for {num_patients} patients over {length} years")
print(f"Results include:")
print(f"  - Mean health trajectories with {N} features")
print(f"  - Uncertainty estimates from {sims} Monte Carlo simulations")
print(f"  - Survival probability predictions with confidence intervals")
