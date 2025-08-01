"""
HiDiNet: Hierarchical Diffusion Network for Healthcare Time Series Modeling
Training Script for Transformer-based Model

This script implements the training loop for the HiDiNet model, which combines:
- Transformer architecture for temporal modeling
- Variational Autoencoders (VAE) for latent representation learning
- Stochastic Differential Equations (SDE) for modeling disease dynamics
- Normalizing flows for flexible posterior approximation

The model is designed for healthcare time series data with irregular sampling,
missing values, and survival analysis components.

Authors: [Your names here]
Paper: [Paper title and venue]
"""

import os
import argparse
import sys
import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster training on A100 GPUs
import torch.nn as nn
from torch.utils import data
import numpy as np

# Add project root to Python path for imports
file = os.path.realpath(__file__)
package_root_directory = os.path.dirname(os.path.dirname(file))
sys.path.append(package_root_directory)

# Model and data loading imports
from transformer.model import Model
from DataLoader.dataset import Dataset, deficits, medications, background
from DataLoader.collate import custom_collate

# Loss functions for different model components
from DJIN_Model.loss import loss, sde_KL_loss

# Learning rate schedulers for KL divergence terms
from Utils.schedules import LinearScheduler, ZeroLinearScheduler

from collections import defaultdict
import math, time
import wandb  # Weights & Biases for experiment tracking

# ==================== COMMAND LINE ARGUMENTS ====================
parser = argparse.ArgumentParser('Train HiDiNet Transformer Model')

# Training configuration
parser.add_argument('--job_id', type=int, help='Unique job identifier for this training run')
parser.add_argument('--batch_size', type=int, default=800, help='Training batch size')
parser.add_argument('--niters', type=int, default=1500, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate for AdamW optimizer')

# Model architecture parameters
parser.add_argument('--d_model', type=int, default=512, help='Transformer model dimension')
parser.add_argument('--gamma_size', type=int, default=25, help='Size of gamma parameters for survival modeling')
parser.add_argument('--z_size', type=int, default=20, help='Latent space dimension for VAE')
parser.add_argument('--decoder_size', type=int, default=65, help='Decoder hidden layer size')

# Normalizing flow parameters
parser.add_argument('--Nflows', type=int, default=3, help='Number of normalizing flow layers')
parser.add_argument('--flow_hidden', type=int, default=24, help='Hidden dimension in flow layers')
parser.add_argument('--f_nn_size', type=int, default=12, help='Neural network size in flow transformations')

# Training hyperparameters
parser.add_argument('--output_dir', type=str, default=None, help='Output directory for saving models and results')
parser.add_argument('--corruption', type=float, default=0.9, help='Data corruption rate for denoising training')
parser.add_argument('--W_prior_scale', type=float, default=0.05, help='Scale parameter for Laplace prior on interaction weights')

# Resume training options
parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
parser.add_argument('--resume_epoch', type=int, default=None, help='Specific epoch to resume from (if not provided, finds latest)')

args = parser.parse_args()

# ==================== TRAINING UTILITIES ====================

def init_epoch_stats():
    """
    Initialize containers for tracking epoch-level statistics.
    
    Returns:
        tuple: (stats_dict, sample_count) where stats_dict contains running sums
               and sample_count tracks total samples processed
    """
    return defaultdict(float), 0

def update_stats(stats, n, recon, kl, sde_raw, beta_dyn):
    """
    Update running statistics for the current epoch.
    
    Args:
        stats (dict): Dictionary containing running sums
        n (int): Number of samples in current batch
        recon (torch.Tensor): Reconstruction loss for current batch
        kl (torch.Tensor): KL divergence loss for current batch  
        sde_raw (torch.Tensor): Raw SDE loss (before beta weighting)
        beta_dyn (float): Current beta weight for dynamics loss
    """
    stats['recon'] += recon.item() * n
    stats['kl'] += kl.item() * n
    stats['sde_raw'] += sde_raw.item() * n
    stats['beta_dyn'] = beta_dyn  # Same for the whole epoch

def epoch_report(epoch, stats, count, beta_kl):
    """
    Print formatted epoch summary with all loss components.
    
    Args:
        epoch (int): Current epoch number
        stats (dict): Accumulated statistics for the epoch
        count (int): Total number of samples processed
        beta_kl (float): KL divergence weight (currently always 1.0)
    """
    recon = stats['recon'] / count
    kl = stats['kl'] / count
    sde_w = stats['sde_raw'] / count * stats['beta_dyn']  # Weighted SDE loss
    total = recon + kl + sde_w
    
    print(f"Epoch {epoch:4d} │ "
          f"recon {recon:8.1f} │ "
          f"KL {kl:7.1e} │ "
          f"SDE_w {sde_w:7.1e} "
          f"(β_dyn={stats['beta_dyn']:.3f}) │ "
          f"β_KL={beta_kl:.3f}")

def save_checkpoint(epoch, model, optimizer, scheduler, scaler, kl_schedulers, params_folder, job_id):
    """
    Save complete training checkpoint including all optimizer states and random seeds.
    This enables exact resumption of training from any epoch.
    
    Args:
        epoch (int): Current epoch number
        model (nn.Module): The HiDiNet model
        optimizer (torch.optim.Optimizer): AdamW optimizer
        scheduler: Learning rate scheduler
        scaler (GradScaler): Mixed precision gradient scaler
        kl_schedulers (list): List of KL divergence schedulers
        params_folder (str): Directory to save checkpoint
        job_id (int): Unique job identifier
        
    Returns:
        str: Path to saved checkpoint file
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        
        # Save KL scheduler states if they support it
        'kl_scheduler_dynamics_state': kl_schedulers[0].state_dict() if hasattr(kl_schedulers[0], 'state_dict') else None,
        'kl_scheduler_vae_state': kl_schedulers[1].state_dict() if hasattr(kl_schedulers[1], 'state_dict') else None,
        'kl_scheduler_network_state': kl_schedulers[2].state_dict() if hasattr(kl_schedulers[2], 'state_dict') else None,
        
        # Save random states for reproducibility
        'random_state': torch.get_rng_state(),
        'numpy_random_state': np.random.get_state(),
    }
    
    checkpoint_path = f'{params_folder}checkpoint_job{job_id}_epoch{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(params_folder, job_id, epoch=None):
    """
    Load training checkpoint, either from specific epoch or latest available.
    
    Args:
        params_folder (str): Directory containing checkpoints
        job_id (int): Job identifier to match
        epoch (int, optional): Specific epoch to load. If None, loads latest.
        
    Returns:
        dict or None: Loaded checkpoint dictionary, or None if not found
    """
    if epoch is not None:
        # Load specific epoch checkpoint
        checkpoint_path = f'{params_folder}checkpoint_job{job_id}_epoch{epoch}.pth'
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return None
    else:
        # Find and load latest checkpoint
        checkpoint_files = [f for f in os.listdir(params_folder) 
                          if f.startswith(f'checkpoint_job{job_id}_') and f.endswith('.pth')]
        if not checkpoint_files:
            print(f"No checkpoints found for job {job_id}")
            return None
        
        # Extract epoch numbers and find the latest
        epochs = []
        for f in checkpoint_files:
            try:
                epoch_num = int(f.split('_epoch')[1].split('.pth')[0])
                epochs.append(epoch_num)
            except:
                continue
        
        if not epochs:
            print("No valid checkpoint epochs found")
            return None
            
        latest_epoch = max(epochs)
        checkpoint_path = f'{params_folder}checkpoint_job{job_id}_epoch{latest_epoch}.pth'
    
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def find_latest_model_epoch(params_folder, job_id):
    """
    Find the latest saved model file (legacy .params format) if no checkpoint exists.
    This provides backward compatibility with older training runs.
    
    Args:
        params_folder (str): Directory containing model files
        job_id (int): Job identifier to match
        
    Returns:
        int or None: Latest epoch number found, or None if no models exist
    """
    model_files = [f for f in os.listdir(params_folder) 
                  if f.startswith(f'train{job_id}_Model_LAMP_epoch') and f.endswith('.params')]
    if not model_files:
        return None
    
    epochs = []
    for f in model_files:
        try:
            epoch_num = int(f.split('_epoch')[1].split('.params')[0])
            epochs.append(epoch_num)
        except:
            continue
    
    return max(epochs) if epochs else None

# ==================== SETUP AND INITIALIZATION ====================

# Device configuration - use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Get current directory and set constants
dir = os.path.dirname(os.path.realpath(__file__))
N = 29  # Number of health variables/features in the dataset
batch_size = args.batch_size
d_model = args.d_model
dropout = 0.1  # Dropout rate for transformer layers
test_after = 1  # Validate every N epochs
test_average = 1  # Number of validation runs to average
dt = 0.5  # Time step for SDE integration

# ==================== OUTPUT DIRECTORY SETUP ====================

# Create output directories for saving models and results
if args.output_dir is not None:
    params_folder = os.path.join(args.output_dir, 'params/')
    output_folder = os.path.join(args.output_dir, 'output/')
    os.makedirs(params_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
else:
    # Default directories relative to script location
    params_folder = dir + '/Parameters_elsa/'
    output_folder = dir + '/Output_elsa/'
    os.makedirs(params_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

# ==================== RESUME TRAINING LOGIC ====================

# Initialize training state
start_epoch = 0
checkpoint = None

if args.resume:
    print("Resume flag detected, looking for checkpoints...")
    checkpoint = load_checkpoint(params_folder, args.job_id, args.resume_epoch)
    
    if checkpoint is None:
        # No checkpoint found, try to find latest model file for partial resume
        latest_epoch = find_latest_model_epoch(params_folder, args.job_id)
        if latest_epoch is not None:
            print(f"No checkpoint found, but found model file at epoch {latest_epoch}")
            print("Will load model weights and continue training (optimizer/scheduler will reset)")
            start_epoch = latest_epoch + 1
        else:
            print("No checkpoints or model files found, starting from scratch")
    else:
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded successfully, resuming from epoch {start_epoch}")

# ==================== OUTPUT FILE INITIALIZATION ====================

# Initialize loss tracking file
loss_file = '%svalidation%d.loss' % (output_folder, args.job_id)
if not args.resume or checkpoint is None:
    open(loss_file, 'w')  # Create new file only if not resuming

# Save hyperparameters for reproducibility
hyperparameters_file = f'{output_folder}train{args.job_id}.hyperparams'
if not args.resume or checkpoint is None:
    with open(hyperparameters_file, 'w') as hf:
        hf.writelines(f'batch_size, {args.batch_size}\n')
        hf.writelines(f'niters, {args.niters}\n')
        hf.writelines(f'learning_rate, {args.learning_rate:.3e}\n')
        hf.writelines(f'corruption, {args.corruption:.3f}\n')
        hf.writelines(f'W_prior_scale, {args.W_prior_scale:.4f}\n')

# ==================== DATA NORMALIZATION SETUP ====================

# Load population statistics for data normalization
# These are computed from the entire dataset and used to normalize individual patient data
pop_avg = np.load(dir + '/../Data/Population_averages.npy')
pop_avg_env = np.load(dir + '/../Data/Population_averages_env.npy')  # Environmental variables
pop_std = np.load(dir + '/../Data/Population_std.npy')
pop_std_env = np.load(dir + '/../Data/Population_std_env.npy')  # Environmental variables

# Convert to PyTorch tensors and prepare for GPU usage
pop_avg = torch.from_numpy(pop_avg[..., 1:]).float()  # Drop age bin centers, keep health variables
pop_std = torch.from_numpy(pop_std[..., 1:]).float()  # Drop age bin centers, keep health variables  
pop_avg_env = torch.from_numpy(pop_avg_env).float()  # Environmental variables (already in correct format)

# ==================== DATASET AND DATALOADER SETUP ====================

# Initialize training dataset
train_name = dir + '/../Data/train.csv'
train_set = Dataset(train_name, N, pop=False, min_count=6)

# Training data loader with custom collation function
# custom_collate handles missing data imputation and corruption mask creation
training_generator = data.DataLoader(
    train_set,
    batch_size=batch_size, 
    shuffle=True, 
    drop_last=True, 
    num_workers=16, 
    pin_memory=True, 
    persistent_workers=False,  # Avoid memory issues with long training
    prefetch_factor=8,
    collate_fn=lambda x: custom_collate(x, pop_avg, pop_avg_env, pop_std, args.corruption)
)

# Initialize validation dataset  
valid_name = dir + '/../Data/valid.csv'
valid_set = Dataset(valid_name, N, pop=False, min_count=6)

# Validation data loader (no corruption for clean evaluation)
valid_generator = data.DataLoader(
    valid_set,
    batch_size=50,
    shuffle=False,
    drop_last=False, 
    num_workers=16, 
    pin_memory=True, 
    persistent_workers=False,
    prefetch_factor=8,
    collate_fn=lambda x: custom_collate(x, pop_avg, pop_avg_env, pop_std, 1.0)  # No corruption
)

print('Data loaded: %d training examples and %d validation examples' % (train_set.__len__(), valid_set.__len__()))

# Get age normalization statistics from training data
mean_of_all_ages = train_set.mean_T  # Mean age for normalization
std_of_all_ages = train_set.std_T    # Standard deviation of ages

# ==================== MODEL AND OPTIMIZER SETUP ====================

# Initialize the HiDiNet model with all specified parameters
model = Model(
    device, N, args.gamma_size, args.z_size, args.decoder_size, 
    args.Nflows, args.flow_hidden, args.f_nn_size, 
    mean_of_all_ages, std_of_all_ages, d_model, dropout
).to(device)

# AdamW optimizer with weight decay and fused implementation for efficiency
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01, fused=True)

# Mixed precision training setup for faster training and reduced memory usage
scaler = GradScaler("cuda", growth_factor=2.0, backoff_factor=0.5, growth_interval=1000)

# Learning rate scheduler - reduces LR when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, threshold=0.01, 
    threshold_mode='rel', patience=10, min_lr=1e-5
)

# ==================== KL DIVERGENCE SCHEDULERS ====================

# These schedulers gradually increase the weight of different KL terms during training
# This helps with training stability and prevents posterior collapse

# Dynamics KL scheduler - controls weight of SDE dynamics regularization
kl_scheduler_dynamics = LinearScheduler(300)  # Ramps up over 300 epochs

# VAE KL scheduler - controls weight of VAE latent space regularization  
kl_scheduler_vae = LinearScheduler(500)  # Ramps up over 500 epochs

# Network KL scheduler - controls weight of interaction network regularization
kl_scheduler_network = ZeroLinearScheduler(300, 500)  # Zero for 300 epochs, then ramps up

# ==================== MODEL LOADING FOR RESUME ====================

if args.resume:
    if checkpoint is not None:
        # Full checkpoint restore - loads all training state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore KL schedulers if available
        if checkpoint.get('kl_scheduler_dynamics_state') is not None:
            if hasattr(kl_scheduler_dynamics, 'load_state_dict'):
                kl_scheduler_dynamics.load_state_dict(checkpoint['kl_scheduler_dynamics_state'])
        
        # Restore random states for exact reproducibility
        if 'random_state' in checkpoint:
            torch.set_rng_state(checkpoint['random_state'])
        if 'numpy_random_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_random_state'])
            
        print(f"Full checkpoint restored from epoch {checkpoint['epoch']}")
        
    elif start_epoch > 0:
        # Partial restore - only model weights available
        model_path = f'{params_folder}train{args.job_id}_Model_LAMP_epoch{start_epoch-1}.params'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model weights loaded from epoch {start_epoch-1}")
            print("Optimizer and scheduler states reset (starting fresh)")

# Print model parameter count
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('Model has %d parameters' % params)

# ==================== INTERACTION MATRIX MASK ====================

# Create mask to prevent self-interactions in the causal interaction matrix
# This ensures W[i,j,k] = 0 when any two indices are the same
matrix_mask = torch.ones(N, N, N)
for i in range(N):
    matrix_mask[i, :, :] *= (~torch.eye(N, dtype=bool)).type(torch.DoubleTensor)
    matrix_mask[:, i, :] *= (~torch.eye(N, dtype=bool)).type(torch.DoubleTensor)
    matrix_mask[:, :, i] *= (~torch.eye(N, dtype=bool)).type(torch.DoubleTensor)
matrix_mask = matrix_mask.to(device)

# ==================== SCHEDULER ADVANCEMENT FOR RESUME ====================

# Advance KL schedulers to correct position if resuming training
if args.resume and start_epoch > 0:
    for _ in range(start_epoch):
        kl_scheduler_dynamics.step()
        kl_scheduler_vae.step()
        kl_scheduler_network.step()

# ==================== PRIOR DISTRIBUTIONS ====================

# Define prior distributions for Bayesian inference

# Prior for observation noise variance (Gamma distribution)
sigma_prior = torch.distributions.gamma.Gamma(torch.tensor(1.0).to(device), torch.tensor(25000.0).to(device))

# Prior for interaction weights (Laplace distribution for sparsity)
W_prior = torch.distributions.laplace.Laplace(torch.tensor(0.0).to(device), torch.tensor(args.W_prior_scale).to(device))

# Prior for VAE latent variables (Standard normal)
vae_prior = torch.distributions.normal.Normal(torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))

# ==================== WANDB EXPERIMENT TRACKING SETUP ====================

# Create descriptive run name for experiment tracking
run_name = f"transformer-lr{args.learning_rate}-d{d_model}-job{args.job_id}"
if args.resume and start_epoch > 0:
    run_name += f"-resume{start_epoch}"

# Initialize Weights & Biases logging
run = wandb.init(
    entity="aashishd04-trinity-university",
    project="lamp-model-transformer",
    name=run_name,
    config={
        # Training hyperparameters
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "niters": args.niters,
        "weight_decay": 0.01,
        "resume": args.resume,
        "start_epoch": start_epoch,
        
        # Model architecture
        "d_model": d_model,
        "dropout": dropout,
        "optimizer": "AdamW",
        "architecture": "transformer",
        
        # Scheduler settings
        "scheduler_patience": 10,
        "scheduler_factor": 0.5,
        "scheduler_threshold": 0.01,
        
        # Model specific parameters
        "gamma_size": args.gamma_size,
        "z_size": args.z_size,
        "corruption": args.corruption,
        "W_prior_scale": args.W_prior_scale,
        
        # Data and model info
        "N_features": N,
        "total_params": params,
    }
)

# ==================== TRAINING LOOP INITIALIZATION ====================

niters = args.niters
batch_counter = start_epoch * len(training_generator)  # Adjust batch counter for resuming
from datetime import datetime

print(f"Starting training from epoch {start_epoch} to {niters}")

# ==================== MAIN TRAINING LOOP ====================

for epoch in range(start_epoch, niters):
    print(f"start time of epoch{epoch}: {datetime.now()}")
    print(f"\n------------------------- Epoch: {epoch} -------------------------")
    
    # Get current KL divergence weights from schedulers
    beta_dynamics = kl_scheduler_dynamics()  # Weight for SDE dynamics loss
    beta_network = kl_scheduler_network()    # Weight for interaction network regularization
    beta_vae = kl_scheduler_vae()           # Weight for VAE latent regularization
    beta_kl = 1.0                           # Weight for other KL terms (kept constant)

    # Initialize epoch statistics tracking
    epoch_stats, samples_seen = init_epoch_stats()

    # ==================== TRAINING BATCH LOOP ====================
    
    for data in training_generator:
        optimizer.zero_grad()
        
        # Move all data tensors to GPU
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)

        # ==================== PARAMETER SAMPLING ====================
        
        # Sample interaction weights from learned posterior
        W_posterior = torch.distributions.laplace.Laplace(model.mean.to(device), model.logscale.exp().to(device))
        
        # Sample observation noise variances from learned posterior
        sigma_posterior = torch.distributions.gamma.Gamma(model.logalpha.exp().to(device), model.logbeta.exp().to(device))
        
        # Draw samples for current batch
        W = W_posterior.rsample((data['Y'].shape[0],)).to(device)
        W_mean = model.mean
        sigma_y = sigma_posterior.rsample((data['Y'].shape[0], data['Y'].shape[1])).to(device) + 1e-6

        # ==================== FORWARD PASS ====================
        
        with autocast("cuda", dtype=torch.bfloat16):  # Mixed precision for efficiency
            # Forward pass through the model
            outputs = model(data, sigma_y)

            # Unpack all model outputs
            pred_X = outputs['pred_X']                    # Predicted health trajectories
            t = outputs['t']                              # Time points
            sigma_X = outputs['pred_sigma_X']             # Predicted trajectory uncertainties
            drifts = outputs['drifts']                    # SDE drift terms
            context = outputs['context']                  # Transformer context representations
            z_sample = outputs['z_sample']                # VAE latent samples
            prior_entropy = outputs['prior_entropy']      # Flow prior entropy
            log_det = outputs['log_det']                  # Flow log determinant
            pred_S = outputs['pred_S']                    # Survival probabilities
            pred_logGamma = outputs['pred_logGamma']      # Hazard function parameters
            recon_mean_x0 = outputs['recon_mean_x0']     # Reconstructed initial states
            sample_weights = outputs['sample_weights']    # Importance sampling weights
            survival_mask = outputs['survival_mask']      # Mask for survival data
            dead_mask = outputs['dead_mask']              # Mask for death events
            after_dead_mask = outputs['after_dead_mask']  # Mask for post-death times
            censored = outputs['censored']                # Censoring indicators
            mask = outputs['mask']                        # General observation mask
            mask0 = outputs['mask0']                      # Initial observation mask
            y = outputs['y']                              # Observed values
            med = outputs['med']                          # Medication indicators
            pred_sigma_X = outputs['pred_sigma_X']        # Predicted uncertainties

            log_S = torch.log(pred_S)  # Log survival probabilities

            # ==================== KL DIVERGENCE CALCULATION ====================
            
            # Calculate KL divergence terms for Bayesian inference
            # This encourages the learned posteriors to stay close to their priors
            
            # KL divergence for interaction weights W
            kl_term = beta_network * torch.sum(matrix_mask * (
                torch.sum(sample_weights * (W_posterior.log_prob(W).permute(1,2,3,0)), dim=-1) - 
                torch.sum(sample_weights * (W_prior.log_prob(W).permute(1,2,3,0)), dim=-1)
            )) + \
            torch.sum(torch.sum(sample_weights * ((mask * sigma_posterior.log_prob(sigma_y)).permute(1,2,0)), dim=(1,2)) - 
                    torch.sum(sample_weights * ((mask * sigma_prior.log_prob(sigma_y)).permute(1,2,0)), dim=(1,2))
            ) - \
            beta_vae * torch.sum(sample_weights * vae_prior.log_prob(z_sample).permute(1,0)) - \
            torch.sum(sample_weights * (prior_entropy.permute(1,0))) - \
            torch.sum(sample_weights * log_det)

            # ==================== LOSS CALCULATION ====================
            
            # Reconstruction loss - measures how well the model predicts observations
            recon_loss = loss(pred_X[:,::2], recon_mean_x0, 
                            pred_logGamma[:,::2], log_S[:,::2],
                            survival_mask, dead_mask, after_dead_mask,
                            t, y, censored, mask,
                            sigma_y[:,1:], sigma_y[:,0], sample_weights)
            
            # SDE loss - regularizes the learned dynamics to be physically plausible
            sde_loss = beta_dynamics * sde_KL_loss(pred_X, t, context,
                                                dead_mask, drifts,
                                                model.dynamics.prior_drift, pred_sigma_X,
                                                dt, mean_of_all_ages, std_of_all_ages,
                                                sample_weights, med,
                                                W * matrix_mask,
                                                W_mean * matrix_mask)
            
            # Total loss combines all components
            total_loss = recon_loss + sde_loss + kl_term
        
        # ==================== BACKWARD PASS AND OPTIMIZATION ====================
        
        # Backward pass with gradient scaling for mixed precision
        scaler.scale(total_loss).backward()
        
        # Check for NaN gradients periodically
        if batch_counter % 10 == 0:
            for name, p in model.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    raise RuntimeError(f"NaN gradient in {name}")
        
        # Unscale gradients for clipping
        scaler.unscale_(optimizer)
        
        # Gradient clipping for training stability
        if batch_counter % 2 == 0: 
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step with gradient scaling
        scaler.step(optimizer)
        scaler.update()

        batch_counter += 1
        
        # ==================== LOGGING ====================
        
        # Log training metrics to Weights & Biases
        if batch_counter % 10 == 0:  # Log every 10 batches to avoid overhead
            wandb.log({
                "train/recon_loss": recon_loss.item(),
                "train/sde_loss": sde_loss.item(), 
                "train/kl_loss": kl_term.item(),
                "train/total_loss": total_loss.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/epoch": epoch,
                "train/beta_dynamics": beta_dynamics,
                "train/beta_network": beta_network, 
                "train/beta_vae": beta_vae,
            })

        # Update epoch statistics
        batch_sz = data['Y'].size(0)
        update_stats(epoch_stats, batch_sz,
             recon_loss, kl_term,                    # KL term already weighted
             sde_loss / max(beta_dynamics, 1e-8),    # Recover raw SDE loss
             beta_dynamics)
        samples_seen += batch_sz
    
    # Print epoch summary
    epoch_report(epoch, epoch_stats, samples_seen, beta_kl)
    
    # Log epoch-level training metrics
    epoch_recon = epoch_stats['recon'] / samples_seen
    epoch_kl = epoch_stats['kl'] / samples_seen  
    epoch_sde_weighted = epoch_stats['sde_raw'] / samples_seen * epoch_stats['beta_dyn']
    epoch_total = epoch_recon + epoch_kl + epoch_sde_weighted
    
    wandb.log({
        "epoch_train/recon_loss": epoch_recon,
        "epoch_train/kl_loss": epoch_kl,
        "epoch_train/sde_loss_weighted": epoch_sde_weighted,
        "epoch_train/total_loss": epoch_total,
        "epoch_train/epoch": epoch,
    })

    # ==================== VALIDATION ====================
    
    if epoch % test_after == 0:
        model.eval()  # Switch to evaluation mode
        with torch.no_grad():  # Disable gradients for efficiency
            total_val_loss = 0.
            recon_val_loss = 0.
            kl_val_loss = 0.
            sde_val_loss = 0.
            
            # Run validation multiple times and average (if test_average > 1)
            for i in range(test_average):
                for data in valid_generator:
                    # Move validation data to GPU
                    for key in data:
                        if isinstance(data[key], torch.Tensor):
                            data[key] = data[key].to(device)
                    
                    # Sample parameters for validation (same process as training)
                    W_posterior = torch.distributions.laplace.Laplace(model.mean.to(device), model.logscale.exp().to(device))
                    sigma_posterior = torch.distributions.gamma.Gamma(model.logalpha.exp().to(device), model.logbeta.exp().to(device))
                    W = W_posterior.rsample((data['Y'].shape[0],)).to(device)
                    W_mean = model.mean
                    sigma_y = sigma_posterior.rsample((data['Y'].shape[0],data['Y'].shape[1])).to(device) + 1e-6
                    
                    with autocast("cuda", dtype=torch.bfloat16):
                        # Forward pass in test mode
                        valoutputs = model(data, sigma_y, test=True)
                        
                        # Unpack validation outputs (same structure as training)
                        val_pred_X = valoutputs['pred_X']
                        val_t = valoutputs['t']
                        val_sigma_X = valoutputs['pred_sigma_X'] 
                        val_drifts = valoutputs['drifts']
                        val_context = valoutputs['context']
                        val_z_sample = valoutputs['z_sample']
                        val_prior_entropy = valoutputs['prior_entropy']
                        val_log_det = valoutputs['log_det']
                        val_pred_S = valoutputs['pred_S']
                        val_pred_logGamma = valoutputs['pred_logGamma']
                        val_recon_mean_x0 = valoutputs['recon_mean_x0']
                        val_sample_weights = valoutputs['sample_weights']
                        val_survival_mask = valoutputs['survival_mask']
                        val_dead_mask = valoutputs['dead_mask']
                        val_after_dead_mask = valoutputs['after_dead_mask']
                        val_censored = valoutputs['censored']
                        val_mask = valoutputs['mask']
                        val_mask0 = valoutputs['mask0']
                        val_y = valoutputs['y']
                        val_med = valoutputs['med']
                        val_pred_sigma_X = valoutputs['pred_sigma_X']

                        val_log_S = torch.log(val_pred_S)
                        s_min = val_pred_S.min().item()
                        s_max = val_pred_S.max().item()

                        summed_weights = torch.sum(val_sample_weights)

                        # Calculate validation KL term (same as training)
                        kl_term = torch.sum(matrix_mask * (
                            torch.sum(val_sample_weights * (W_posterior.log_prob(W).permute(1,2,3,0)), dim=-1) + 
                            torch.sum(val_sample_weights * (W_prior.log_prob(W).permute(1,2,3,0)), dim=-1)
                        )) + \
                        torch.sum(torch.sum(val_sample_weights * ((val_mask * sigma_posterior.log_prob(sigma_y)).permute(1,2,0)), dim=(1,2)) - 
                                torch.sum(val_sample_weights * ((val_mask * sigma_prior.log_prob(sigma_y)).permute(1,2,0)), dim=(1,2))
                        ) - \
                        torch.sum(val_sample_weights * vae_prior.log_prob(val_z_sample).permute(1,0)) - \
                        torch.sum(val_sample_weights * (val_prior_entropy.permute(1,0))) - \
                        torch.sum(val_sample_weights * val_log_det)
                        
                        # Calculate validation losses
                        recon_l = loss(val_pred_X[:,::2], val_recon_mean_x0,
                                    val_pred_logGamma[:,::2], val_log_S[:,::2],
                                    val_survival_mask, val_dead_mask, val_after_dead_mask,
                                    val_t, val_y, val_censored, val_mask,
                                    sigma_y[:,1:], sigma_y[:,0], val_sample_weights)
                        
                        sde_l = sde_KL_loss(val_pred_X, val_t, val_context,
                                        val_dead_mask, val_drifts,
                                        model.dynamics.prior_drift, val_pred_sigma_X,
                                        dt, mean_of_all_ages, std_of_all_ages,
                                        val_sample_weights, val_med,
                                        W * matrix_mask,
                                        W_mean * matrix_mask)
                        
                        # Accumulate validation losses
                        kl_val_loss += kl_term
                        total_val_loss += sde_l + recon_l + kl_term
                        recon_val_loss += recon_l
                        sde_val_loss += sde_l
            
            # Average validation losses over all runs
            avg_recon = recon_val_loss / test_average
            avg_total = total_val_loss / test_average
            avg_kl = kl_val_loss / test_average
            avg_sde = sde_val_loss / test_average
            
            # Save validation results to file
            with open(loss_file, 'a') as lf:
                lf.writelines(f'{epoch}, {avg_recon.cpu().numpy():.3f}, {avg_total.cpu().numpy():.3f}\n')
            
            # Log validation metrics to Weights & Biases
            wandb.log({
                "val/recon_loss": avg_recon.item(),
                "val/sde_loss": avg_sde.item(), 
                "val/kl_loss": avg_kl.item(),
                "val/total_loss": avg_total.item(),
                "val/epoch": epoch,
            })
            
            # Log additional validation statistics
            val_pred_S_stats = {
                "val/pred_S_mean": val_pred_S.mean().item(),
                "val/pred_S_std": val_pred_S.std().item(),
                "val/pred_S_min": val_pred_S.min().item(),
                "val/pred_S_max": val_pred_S.max().item(),
            }

            val_logGamma_stats = {
                "val/logGamma_mean": val_pred_logGamma.mean().item(),
                "val/logGamma_std": val_pred_logGamma.std().item(), 
                "val/logGamma_min": val_pred_logGamma.min().item(),
                "val/logGamma_max": val_pred_logGamma.max().item(),
            }

            wandb.log({**val_pred_S_stats, **val_logGamma_stats})
            
        model.train()  # Switch back to training mode
        scheduler.step(avg_total)  # Update learning rate based on validation loss
    
    # ==================== MODEL SAVING ====================
    
    # Save model and checkpoint periodically
    if epoch % 20 == 0:
        # Save regular model weights for backward compatibility
        torch.save(model.state_dict(), '%strain%d_Model_LAMP_epoch%d.params'%(params_folder, args.job_id, epoch))
        wandb.save(f'{params_folder}train{args.job_id}_Model_LAMP_epoch{epoch}.params')
        
        # Save full checkpoint for resuming training
        save_checkpoint(epoch, model, optimizer, scheduler, scaler, 
                       [kl_scheduler_dynamics, kl_scheduler_vae, kl_scheduler_network], 
                       params_folder, args.job_id)
    
    # ==================== SCHEDULER UPDATES ====================
    
    # Step all KL divergence schedulers
    kl_scheduler_dynamics.step()
    kl_scheduler_network.step()
    kl_scheduler_vae.step()
    
    print(f"end of epoch {epoch}: {datetime.now()}")

# ==================== FINAL MODEL SAVING ====================

# Save final model and checkpoint after training completion
torch.save(model.state_dict(), '%strain%d_Model_LAMP_epoch%d.params'%(params_folder, args.job_id, epoch))
save_checkpoint(epoch, model, optimizer, scheduler, scaler, 
               [kl_scheduler_dynamics, kl_scheduler_vae, kl_scheduler_network], 
               params_folder, args.job_id)
