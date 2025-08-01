# makes predictions with trained djin model, requires job_id and epoch to be specified
# results saved to Analysis_Data

import argparse
import torch
from torch.nn import functional as F
import numpy as np
from scipy.stats import sem
from pandas import read_csv
from torch.utils import data
import os

from Utils.record import record
from DJIN_Model.model import Model
from DataLoader.dataset import Dataset
from DataLoader.collate import custom_collate

parser = argparse.ArgumentParser('Predict')
parser.add_argument('--job_id', type=int)
parser.add_argument('--epoch', type=int)
parser.add_argument('--gamma_size', type=int, default = 25)
parser.add_argument('--z_size', type=int, default = 20)
parser.add_argument('--decoder_size', type=int, default = 65)
parser.add_argument('--Nflows', type=int, default = 3)
parser.add_argument('--flow_hidden', type=int, default = 24)
parser.add_argument('--f_nn_size', type=int, default = 12)
parser.add_argument('--W_prior_scale', type=float, default = 0.1)
parser.add_argument('--dataset',type=str,choices=['elsa','sample'],default='elsa',help='the dataset that was used to train the model; either \'elsa\' or \'sample\'')
args = parser.parse_args()

postfix = '_sample' if args.dataset == 'sample' else ''
dir = os.path.dirname(os.path.realpath(__file__))

torch.set_num_threads(6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# size of our feature vectors
N = 29

# how many simulations to run
sims = 250

# timestep
dt = 0.5

# how long you wish to predict for?
length = 50

# loading population averages
pop_avg = np.load(f'{dir}/Data/Population_averages{postfix}.npy')
pop_avg_env = np.load(f'{dir}/Data/Population_averages_env{postfix}.npy')
pop_std = np.load(f'{dir}/Data/Population_std{postfix}.npy')
pop_avg_ = torch.from_numpy(pop_avg[...,1:]).float()
pop_avg_env = torch.from_numpy(pop_avg_env).float()
pop_std = torch.from_numpy(pop_std[...,1:]).float()
pop_avg_bins = np.arange(40, 105, 3)[:-2]

# loading dataset
test_name = f'{dir}/Data/test{postfix}.csv'
test_set = Dataset(test_name, N, pop=False, min_count=10)
num_test = 400
test_generator = data.DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))

mean_T = test_set.mean_T
std_T = test_set.std_T

# creating model based on trained parameters
model = Model(device, N, args.gamma_size, args.z_size, args.decoder_size, args.Nflows, args.flow_hidden, args.f_nn_size, mean_T, std_T, dt, length).to(device)
model.load_state_dict(torch.load(dir+'/Parameters/train%d_Model_DJIN_epoch%d%s.params'%(args.job_id, args.epoch,postfix),map_location=device))
model = model.eval()

mean_results = np.zeros((test_set.__len__(), 100, N+1)) * np.nan
std_results = np.zeros((test_set.__len__(), 100, N+1)) * np.nan
S_results = np.zeros((test_set.__len__(), 100, 3)) * np.nan

with torch.no_grad():

    # posterior
    sigma_posterior = torch.distributions.gamma.Gamma(model.logalpha.exp(), model.logbeta.exp())
    
    start = 0
    for data in test_generator:

        size = data['Y'].shape[0]
        
        X = torch.zeros(sims, size, int(length/dt), N).to(device)
        X_std = torch.zeros(sims, size, int(length/dt), N).to(device)
        S = torch.zeros(sims, size, int(length/dt)).to(device)
        alive = torch.ones(sims, size, int(length/dt)).to(device)
        
        for s in range(sims):
            print(f'running sim {s} / {sims-1}')
            sigma_y = sigma_posterior.sample((data['Y'].shape[0], length*2))
            
            # the output from the model
            pred_X, t, pred_S, pred_logGamma, pred_sigma_X, context, y, times, mask, survival_mask, dead_mask, after_dead_mask, censored, sample_weights, med, env, z_sample, prior_entropy, log_det, recon_mean_x0, drifts, mask0, W = model(data, sigma_y, test=True)

            # record results for this simulation
            X[s] = pred_X
            X_std[s] = pred_X + sigma_y*torch.randn_like(pred_X)
            S[s] = pred_S.exp()
            alive[s,:,1:] = torch.cumprod(torch.bernoulli(torch.exp(-1*pred_logGamma.exp()[:,:-1]*dt)), dim=1)
            
        # extract the predictions for the years we want to compare to the data we have
        t0 = t[:,0]
        record_times = [torch.from_numpy(np.arange(t0[b].cpu(), 110, 1)).to(device) for b in range(size)]
        X_record, S_record = record(t, X, S, record_times, dt)
        X_std_record, alive_record = record(t, X_std, alive, record_times, dt)
        t0 = t0.cpu()
        
        # generating the various trajectories

        X_sum = []
        X_sum_std = []
        X_sum2 = []
        X_count = []
        for b in range(size):
            X_sum.append(torch.sum(X_record[b].permute(2,0,1)*alive_record[b], dim = 1).cpu())
            X_sum_std.append(torch.sum(X_std_record[b].permute(2,0,1)*alive_record[b], dim = 1).cpu())
            X_sum2.append(torch.sum(X_std_record[b].pow(2).permute(2,0,1)*alive_record[b], dim = 1).cpu())
            X_count.append(torch.sum(alive_record[b], dim = 0).cpu())

        for b in range(size):

            mean_results[start+b, :len(np.arange(t0[b], 110, 1)), 0] = np.arange(t0[b], 110, 1)
            std_results[start+b, :len(np.arange(t0[b], 110, 1)), 0] = np.arange(t0[b], 110, 1)
            S_results[start+b, :len(np.arange(t0[b], 110, 1)), 0] = np.arange(t0[b], 110, 1)
            
            mean_results[start+b, :X_sum[b].shape[1], 1:] = (X_sum[b]/X_count[b]).permute(1,0).numpy()
            std_results[start+b, :X_sum_std[b].shape[1], 1:] = np.sqrt((X_sum2[b]/X_count[b] - (X_sum_std[b]/X_count[b]).pow(2)).permute(1,0).numpy())
            S_results[start+b, :len(np.arange(t0[b], 110, 1)), 1] = torch.mean(S_record[b], dim = 0)
            S_results[start+b, :len(np.arange(t0[b], 110, 1)), 2] = torch.std(S_record[b], dim = 0)
        
        start += size

# saving the various trajectories to their respective files
np.save(dir+'/Analysis_Data/Mean_trajectories_job_id%d_epoch%d_DJIN%s.npy'%(args.job_id, args.epoch,postfix), mean_results)
np.save(dir+'/Analysis_Data/Std_trajectories_job_id%d_epoch%d_DJIN%s.npy'%(args.job_id, args.epoch,postfix), std_results)
np.save(dir+'/Analysis_Data/Survival_trajectories_job_id%d_epoch%d_DJIN%s.npy'%(args.job_id, args.epoch,postfix), S_results)
