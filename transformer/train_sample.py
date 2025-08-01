import os
import argparse
import sys
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np

file = os.path.realpath(__file__)
package_root_directory = os.path.dirname(os.path.dirname(file))
sys.path.append(package_root_directory)

from transformer.model import Model

from DataLoader.dataset import Dataset, deficits, medications, background
from DataLoader.collate import custom_collate

from DJIN_Model.loss import loss, sde_KL_loss

from Utils.schedules import LinearScheduler, ZeroLinearScheduler

from collections import defaultdict
import math, time


parser = argparse.ArgumentParser('Train')
parser.add_argument('--job_id', type=int)
parser.add_argument('--batch_size', type=int, default = 16)
parser.add_argument('--niters', type=int, default = 2000)
parser.add_argument('--learning_rate', type=float, default = 1e-2)
parser.add_argument('--corruption', type=float, default = 0.9)
parser.add_argument('--gamma_size', type=int, default = 25)
parser.add_argument('--z_size', type=int, default = 20)
parser.add_argument('--decoder_size', type=int, default = 65)
parser.add_argument('--Nflows', type=int, default = 3)
parser.add_argument('--flow_hidden', type=int, default = 24)
parser.add_argument('--f_nn_size', type=int, default = 12)
parser.add_argument('--W_prior_scale', type=float, default = 0.05)
args = parser.parse_args()

#compact epoch logging helpers
def init_epoch_stats():
    """Container for running sums."""
    return defaultdict(float), 0              # stats, samples_seen

def update_stats(stats, n, recon, kl, sde_raw, beta_dyn):
    stats['recon']      += recon.item()  * n
    stats['kl']         += kl.item()     * n
    stats['sde_raw']    += sde_raw.item()* n
    stats['beta_dyn']    = beta_dyn      # same for the whole epoch

def epoch_report(epoch, stats, count, beta_kl):
    recon   = stats['recon']   / count
    kl      = stats['kl']      / count
    sde_w   = stats['sde_raw'] / count * stats['beta_dyn']
    total   = recon + kl + sde_w
    print(f"Epoch {epoch:4d} │ "
          f"recon {recon:8.1f} │ "
          f"KL {kl:7.1e} │ "
          f"SDE_w {sde_w:7.1e} "
          f"(β_dyn={stats['beta_dyn']:.3f}) │ "
          f"β_KL={beta_kl:.3f}")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
dir = os.path.dirname(os.path.realpath(__file__))
N = 29
batch_size = args.batch_size
d_model = 256
dropout = 0.1
test_after = 10
test_average = 5
dt = 0.5

#set up output folders
params_folder = dir + '/Parameters/'
output_folder = dir + '/Output/'
os.makedirs(params_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

#file for loss outputs
loss_file = '%svalidation%d.loss'%(output_folder, args.job_id)
open(loss_file, 'w')

# Save hyperparameters
hyperparameters_file = f'{output_folder}train{args.job_id}.hyperparams'
with open(hyperparameters_file, 'w') as hf:
    hf.writelines(f'batch_size, {args.batch_size}\n')
    hf.writelines(f'niters, {args.niters}\n')
    hf.writelines(f'learning_rate, {args.learning_rate:.3e}\n')
    hf.writelines(f'corruption, {args.corruption:.3f}\n')
    hf.writelines(f'W_prior_scale, {args.W_prior_scale:.4f}\n')

#loading population averages
sample_pop_avg = np.load(dir + '/../Data/Population_averages_sample.npy')
sample_pop_avg_env = np.load(dir + '/../Data/Population_averages_env_sample.npy') #env variables
sample_pop_std = np.load(dir + '/../Data/Population_std_sample.npy')
sample_pop_std_env = np.load(dir + '/../Data/Population_std_env_sample.npy') # env variables

#have to convert to tensors
sample_pop_avg = torch.from_numpy(sample_pop_avg[...,1:]).float() # drops the age bin centers, columns 1 to N are population averages for health variable
sample_pop_std = torch.from_numpy(sample_pop_std[...,1:]).float() # ^ for std
sample_pop_avg_env = torch.from_numpy(sample_pop_avg_env).float() # used for environmental variables, nothing being dropped because it is already in correct format for env vars

#loading training dataset
sample_train_name = dir + '/../Data/train_sample.csv'
sample_train_set = Dataset(sample_train_name, N, pop=False, min_count=6) #creates dataset and returns pytorch dataset object returns tuple of 11 elements. Designed to use with dataloader
#custom_collate - takes individual patient records and puts them together in batchlike, creates corruption mask and for each patient, uses pop averages to fill in missing values
#end result is batch of data that is ready for model to learn from
sample_training_generator = data.DataLoader(sample_train_set,
                                            batch_size=batch_size, 
                                            shuffle=True, drop_last=True, num_workers=16, pin_memory=True, persistent_workers=False,  # Changed to False
                                            prefetch_factor=8,
                                            collate_fn= lambda x: custom_collate(x, sample_pop_avg, sample_pop_avg_env, sample_pop_std, args.corruption)) #loads individual samples from dataset, custom_collate combines them into batches. custom collate returns dict with measurements

#loading validation dataset
sample_valid_name = dir + '/../Data/valid_sample.csv'
sample_valid_set = Dataset(sample_valid_name, N, pop=False, min_count=6)
sample_valid_generator = data.DataLoader(sample_valid_set,
                                        batch_size=50,
                                        shuffle=False,drop_last=False, num_workers=16, pin_memory=True, persistent_workers=False,  # Changed to False
                                        prefetch_factor=8,
                                        collate_fn=lambda x: custom_collate(x, sample_pop_avg, sample_pop_avg_env, sample_pop_std, 1.0)
                                        )


#data loaded into batches
print('Data loaded: %d training examples and %d validation examples'%(sample_train_set.__len__(), sample_valid_set.__len__()))

mean_of_all_ages = sample_train_set.mean_T # mean of all ages, used to normalize age values in dataset to have standard scale
std_of_all_ages = sample_train_set.std_T # std of all ages, used to normalize age values in dataset. makes age comparable across patients at diff time points

# creating model to be trained
model = Model(device, N, args.gamma_size, args.z_size, args.decoder_size, args.Nflows, args.flow_hidden, args.f_nn_size, mean_of_all_ages, std_of_all_ages, d_model, dropout).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.5, threshold = 1000, threshold_mode ='abs', patience = 4, min_lr = 1e-5)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('Model has %d parameters'%params)

# 0 at any place where there is the same variable twice
matrix_mask = torch.ones(N,N,N)
for i in range(N):
    matrix_mask[i,:,:] *= (~torch.eye(N,dtype=bool)).type(torch.DoubleTensor)
    matrix_mask[:,i,:] *= (~torch.eye(N,dtype=bool)).type(torch.DoubleTensor)
    matrix_mask[:,:,i] *= (~torch.eye(N,dtype=bool)).type(torch.DoubleTensor)
matrix_mask = matrix_mask.to(device)

kl_scheduler_dynamics = LinearScheduler(300)
kl_scheduler_vae = LinearScheduler(500)
kl_scheduler_network = ZeroLinearScheduler(300, 500)

# priors
sigma_prior = torch.distributions.gamma.Gamma(torch.tensor(1.0).to(device), torch.tensor(25000.0).to(device))
W_prior = torch.distributions.laplace.Laplace(torch.tensor(0.0).to(device), torch.tensor(args.W_prior_scale).to(device))
vae_prior = torch.distributions.normal.Normal(torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))

niters = args.niters
#training for the specified number of epochs
for epoch in range(niters):
    print(F"\n------------------------- Epoch: {epoch} -------------------------")
    beta_dynamics = kl_scheduler_dynamics()
    beta_network = kl_scheduler_network()
    beta_vae = kl_scheduler_vae()
    beta_kl = 1.0

    epoch_stats, samples_seen = init_epoch_stats()

    for data in sample_training_generator:
        optimizer.zero_grad()
        
        # Move data to GPU
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)

        # Sample parameters
        W_posterior = torch.distributions.laplace.Laplace(model.mean.to(device), model.logscale.exp().to(device))
        sigma_posterior = torch.distributions.gamma.Gamma(model.logalpha.exp().to(device), model.logbeta.exp().to(device))
        W = W_posterior.rsample((data['Y'].shape[0],)).to(device)
        W_mean = model.mean
        sigma_y = sigma_posterior.rsample((data['Y'].shape[0],data['Y'].shape[1])).to(device) + 1e-6

        # Forward pass
        outputs = model(data, sigma_y)

        # Unpack outputs
        pred_X = outputs['pred_X']
        t = outputs['t']
        sigma_X = outputs['pred_sigma_X']
        drifts = outputs['drifts']
        context = outputs['context']
        z_sample = outputs['z_sample']
        prior_entropy = outputs['prior_entropy']
        log_det = outputs['log_det']
        pred_S = outputs['pred_S']
        pred_logGamma = outputs['pred_logGamma']
        recon_mean_x0 = outputs['recon_mean_x0']
        sample_weights = outputs['sample_weights']
        survival_mask = outputs['survival_mask']
        dead_mask = outputs['dead_mask']
        after_dead_mask = outputs['after_dead_mask']
        censored = outputs['censored']
        mask = outputs['mask']
        mask0 = outputs['mask0']
        y = outputs['y']
        med = outputs['med']
        pred_sigma_X = outputs['pred_sigma_X']

        # Calculate KL term (moved from model to training loop)
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

        # Calculate losses
        recon_loss = loss(pred_X[:,::2], recon_mean_x0, 
                         pred_logGamma[:,::2], pred_S[:,::2],
                         survival_mask, dead_mask, after_dead_mask,
                         t, y,censored, mask,
                         sigma_y[:,1:], sigma_y[:,0], sample_weights)
        
        sde_loss = beta_dynamics * sde_KL_loss(pred_X, t, context,
                                              dead_mask, drifts,
                                              model.dynamics.prior_drift, pred_sigma_X,
                                              dt, mean_of_all_ages, std_of_all_ages,
                                              sample_weights, med,
                                              W * matrix_mask,
                                              W_mean * matrix_mask)
        total_loss = recon_loss + sde_loss + kl_term
        
        # Backward pass
        with torch.autograd.set_detect_anomaly(True):
            total_loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                raise RuntimeError(f"NaN gradient in {name}")
            
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # -------- accumulate summary stats -------------
        batch_sz = data['Y'].size(0)
        update_stats(epoch_stats, batch_sz,
             recon_loss, kl_term,                    # kl_term already weighted
             sde_loss / max(beta_dynamics, 1e-8),    # recover *raw* SDE
             beta_dynamics)
        samples_seen += batch_sz
    
    epoch_report(epoch, epoch_stats, samples_seen, beta_kl)

    # Validation
    if epoch % test_after == 0:
        model.eval()
        with torch.no_grad():
            total_val_loss = 0.
            recon_val_loss = 0.
            kl_val_loss = 0.
            sde_val_loss = 0.
            
            for i in range(test_average):
                for data in sample_valid_generator:

                    # Move data to GPU
                    for key in data:
                        if isinstance(data[key], torch.Tensor):
                            data[key] = data[key].to(device)
                    
                    # Sample parameters for validation (same as training)
                    W_posterior = torch.distributions.laplace.Laplace(model.mean.to(device), model.logscale.exp().to(device))
                    sigma_posterior = torch.distributions.gamma.Gamma(model.logalpha.exp().to(device), model.logbeta.exp().to(device))
                    W = W_posterior.rsample((data['Y'].shape[0],)).to(device)
                    W_mean = model.mean
                    sigma_y = sigma_posterior.rsample((data['Y'].shape[0],data['Y'].shape[1])).to(device) + 1e-6
                    
                    valoutputs = model(data, sigma_y, test=True)
                    # Unpack outputs
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
                                 val_pred_logGamma[:,::2], val_pred_S[:,::2],
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
                    
                    kl_val_loss += kl_term
                    total_val_loss += sde_l + recon_l + kl_term
                    recon_val_loss += recon_l
                    sde_val_loss += sde_l
            
            # Average validation losses
            avg_recon = recon_val_loss / test_average
            avg_total = total_val_loss / test_average
            avg_kl = kl_val_loss / test_average
            avg_sde = sde_val_loss / test_average
            
            # Save validation results
            with open(loss_file, 'a') as lf:
                lf.writelines(f'{epoch}, {avg_recon.cpu().numpy():.3f}, {avg_total.cpu().numpy():.3f}\n')
            
        model.train()
        scheduler.step(avg_total)
    
    # Save model
    if epoch % 20 == 0:
        torch.save(model.state_dict(), '%strain%d_Model_LAMP_epoch%d_sample.params'%(params_folder, args.job_id, epoch))
    
    # Step schedulers
    kl_scheduler_dynamics.step()
    kl_scheduler_network.step()
    kl_scheduler_vae.step()

# Save final model
torch.save(model.state_dict(), '%strain%d_Model_LAMP_epoch%d_sample.params'%(params_folder, args.job_id, epoch))