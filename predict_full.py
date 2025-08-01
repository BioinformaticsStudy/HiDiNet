# creates predictions using model trained in train_full.py

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from pandas import read_csv
import os

from Alternate_models.model_full import Model
from Alternate_models.dataset_dim import Dataset
from DataLoader.collate import custom_collate
from Utils.record import record


def predict(job_id,epoch,niters,learning_rate,gamma_size,z_size,decoder_size,Nflows,flow_hidden,dataset,N):
    postfix = f'_latent{N}_sample' if dataset == 'sample' else f'_latent{N}'
    repository = os.path.dirname(os.path.realpath(__file__))
  
    torch.set_num_threads(6)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sims = 250
    dt = 0.5
    length = 50

    pop_avg = np.load(f'{repository}/Data/Population_averages{postfix}.npy')
    pop_avg_env = np.load(f'{repository}/Data/Population_averages_env{postfix}.npy')
    pop_std = np.load(f'{repository}/Data/Population_std{postfix}.npy')
    pop_avg_ = torch.from_numpy(pop_avg[...,1:]).float()
    pop_avg_env = torch.from_numpy(pop_avg_env).float()
    pop_std = torch.from_numpy(pop_std[...,1:]).float()
    pop_avg_bins = np.arange(40, 105, 3)[:-2]

    test_name = f'{repository}/Data/test{postfix}.csv'
    min_count = N // 3
    prune = min_count >= 1

    test_set = Dataset(test_name, N, pop=False, min_count=min_count,prune=prune)
    num_test = 400
    test_generator = DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))

    mean_T = test_set.mean_T
    std_T = test_set.std_T

    model = Model(device, N, gamma_size, z_size, decoder_size, Nflows, flow_hidden, mean_T, std_T, dt,length=length).to(device)
    model.load_state_dict(torch.load('%s/Parameters/train%d_Model%s_epoch%d.params'%(repository,job_id, postfix,epoch),map_location=device))
    model = model.eval()

    mean_results = np.zeros((test_set.__len__(), 100, N+1)) * np.nan
    std_results = np.zeros((test_set.__len__(), 100, N+1)) * np.nan
    S_results = np.zeros((test_set.__len__(), 100, 3)) * np.nan
    

    with torch.no_grad():
        
        sigma_posterior = torch.distributions.gamma.Gamma(model.logalpha.exp(), model.logbeta.exp())
        start = 0
        for i,data in enumerate(test_generator):
            print(f'predicting batch {i}')
            size = data['Y'].shape[0]

            X = torch.zeros(sims, size, int(length/dt), N).to(device)
            X_std = torch.zeros(sims, size, int(length/dt), N).to(device)
            S = torch.zeros(sims, size, int(length/dt)).to(device)
            alive = torch.ones(sims, size, int(length/dt)).to(device)
            for s in range(sims):
                sigma_y = sigma_posterior.sample((data['Y'].shape[0], length*2))
                

                pred_X, pred_Z, t, pred_S, pred_logGamma, pred_sigma_X, context,\
                y, times, mask, survival_mask, dead_mask, after_dead_mask, censored, \
                sample_weights, med, env, z_sample, prior_entropy, log_det, recon_mean_x0, mask0 = model(data, sigma_y, test=True)
                
                X[s] = pred_X
                X_std[s] = pred_X + sigma_y*torch.randn_like(pred_X)
                S[s] = pred_S.exp()
                alive[s,:,1:] = torch.cumprod(torch.bernoulli(torch.exp(-1*pred_logGamma.exp()[:,:-1]*dt)), dim=1)

            t0 = t[:,0]
            record_times = [torch.from_numpy(np.arange(t0[b].cpu(), 121, 1)).to(device) for b in range(size)]
            X_record, S_record = record(t, X, S, record_times, dt)
            X_std_record, alive_record = record(t, X_std, alive, record_times, dt)
            t0 = t0.cpu()

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
                mean_results[start+b, :len(np.arange(t0[b], 121, 1)), 0] = np.arange(t0[b], 121, 1)
                std_results[start+b, :len(np.arange(t0[b], 121, 1)), 0] = np.arange(t0[b], 121, 1)
                S_results[start+b, :len(np.arange(t0[b], 121, 1)), 0] = np.arange(t0[b], 121, 1)

                mean_results[start+b, :X_sum[b].shape[1], 1:] = (X_sum[b]/X_count[b]).permute(1,0).numpy()
                std_results[start+b, :X_sum_std[b].shape[1], 1:] = np.sqrt((X_sum2[b]/X_count[b] - (X_sum_std[b]/X_count[b]).pow(2)).permute(1,0).numpy())
                S_results[start+b, :len(np.arange(t0[b], 121, 1)), 1] = torch.mean(S_record[b], dim = 0)
                S_results[start+b, :len(np.arange(t0[b], 121, 1)), 2] = torch.std(S_record[b], dim = 0)

            start += size
    
    np.save('%s/Analysis_Data/Mean_trajectories_job_id%d_epoch%d%s.npy'%(repository,job_id, epoch,postfix), mean_results)
    np.save('%s/Analysis_Data/Std_trajectories_job_id%d_epoch%d%s.npy'%(repository,job_id, epoch,postfix), std_results)
    np.save('%s/Analysis_Data/Survival_trajectories_job_id%d_epoch%d%s.npy'%(repository,job_id, epoch, postfix), S_results)

if __name__=='__main__':
    parser = argparse.ArgumentParser('Predict')
    parser.add_argument('--job_id', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--niters', type=int, default = 2000)
    parser.add_argument('--learning_rate', type=float, default = 1e-2)
    parser.add_argument('--gamma_size', type=int, default = 25)
    parser.add_argument('--z_size', type=int, default = 30)
    parser.add_argument('--decoder_size', type=int, default = 65)
    parser.add_argument('--Nflows', type=int, default = 3)
    parser.add_argument('--flow_hidden', type=int, default = 24)
    parser.add_argument('--dataset',type=str, default='elsa',choices=['elsa','sample'])
    parser.add_argument('--N', type=int,default=29)
    args = parser.parse_args()

    predict(args.job_id,args.epoch,args.niters,args.learning_rate,args.gamma_size,args.z_size,args.decoder_size,args.Nflows,args.flow_hidden,args.dataset,args.N)