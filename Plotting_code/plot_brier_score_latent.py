#plots the integrated brier score for multiple latent space models depending on N
from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))  

import os
import argparse
import torch
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, binned_statistic
from Alternate_models.dataset_dim import Dataset
from torch.utils.data import DataLoader
from DataLoader.collate import custom_collate
from lifelines import KaplanMeierFitter

parser = argparse.ArgumentParser('Brier_score_latent')
parser.add_argument('--job_id',type=int)
parser.add_argument('--epoch',type=int,default=1999)
parser.add_argument('--start', type=int,default=2,help='lowest N')
parser.add_argument('--step', type=int, default=5,help='difference in N between models')
parser.add_argument('--stop', type=int,default=35,help='highest N')
parser.add_argument('--dataset', type=str, default='elsa',choices=['elsa','sample'],help='what dataset was used to train the models')
parser.add_argument('--djin_id',type=int,default=None)
parser.add_argument('--djin_epoch',type=int,default=None)
args = parser.parse_args()

djin_compare = args.djin_id != None and args.djin_epoch != None
dir = os.path.dirname(os.path.realpath(__file__))

Ns = list(np.arange(args.start,args.stop,args.step)) + [args.stop]
results = pd.DataFrame(index=Ns,columns=['IBS'])

for N in Ns:
    postfix = f'_latent{N}_sample' if args.dataset=='sample' else f'_latent{N}'

    survival = np.load(dir+'/../Analysis_Data/Survival_trajectories_job_id%d_epoch%d%s.npy'%(args.job_id,args.epoch,postfix))
    
    pop_avg = np.load(f'{dir}/../Data/Population_averages{postfix}.npy')
    pop_avg_env = np.load(f'{dir}/../Data/Population_averages_env{postfix}.npy')
    pop_std = np.load(f'{dir}/../Data/Population_std{postfix}.npy')
    pop_avg_ = torch.from_numpy(pop_avg[...,1:]).float()
    pop_avg_env = torch.from_numpy(pop_avg_env).float()
    pop_std = torch.from_numpy(pop_std[...,1:]).float()

    min_count = N // 3
    prune = min_count >= 1

    test_name = f'{dir}/../Data/test{postfix}.csv'
    test_set = Dataset(test_name, N,  pop=False, min_count=min_count, prune=prune)
    num_test = test_set.__len__()
    test_generator = DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))
    for data in test_generator:
        break

    death_ages = data['death age'].numpy()
    censored = data['censored'].numpy()
    times = data['times'].numpy()
    dead_mask = data['survival_mask'].numpy()
    ages = times[:,0]

    death_ages = np.array([death_ages[m] if death_ages[m] > 0 else times[m].max() for m in range(death_ages.size)])

    sample_weight = data['weights'].numpy()
    dead_mask = np.concatenate((dead_mask, np.zeros((dead_mask.shape[0], dead_mask.shape[1]*3))), axis = 1)
    
    survival_prob = survival[:,:,1]
    survival_ages = survival[:,:,0]

    observed = 1 - np.array(censored,dtype=int)

    kmf_G = KaplanMeierFitter()
    kmf_G.fit(death_ages, event_observed = 1 - observed, timeline = np.arange(0, 200, 1))
    G = kmf_G.survival_function_.values.flatten()

    G = np.array([G[int(np.nanmin(survival_ages[i])):][:len(survival_ages[i])] for i in range(len(censored))])

    bin_edges = np.arange(30.5, 130.5, 1)
    bin_centers = bin_edges[1:] - np.diff(bin_edges)

    BS = np.zeros(bin_centers.shape)
    BS_count = np.zeros(bin_centers.shape)

    BS_S = np.zeros(bin_centers.shape)
    BS_S_count = np.zeros(bin_centers.shape)

    for i in range(len(survival_ages)):
        # die
        if censored[i] == 0:
            ages = survival_ages[i, ~np.isnan(survival_ages[i])]
            mask = dead_mask[i, ~np.isnan(survival_ages[i])]
            prob = survival_prob[i, ~np.isnan(survival_ages[i])]
            G_i = G[i][~np.isnan(survival_ages[i])]

            ages = ages[~np.isnan(prob)]
            mask = mask[~np.isnan(prob)]
            G_i = G_i[~np.isnan(prob)]
            prob = prob[~np.isnan(prob)]
            
            
            G_alive = G_i[mask==1]
            G_dead = G_i[mask==0]
            G_dead = G_alive[-1]*np.ones(G_dead.shape)
            G_i = np.concatenate((G_alive, G_dead))
            
            
            G_i[G_i < 1e-5] = np.nan
            
            if len(ages[~np.isnan(G_i)]) != 0:
                BS += binned_statistic(ages[~np.isnan(G_i)], ((mask - prob)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = np.nansum)[0]
                BS_count += binned_statistic(ages[~np.isnan(G_i)], ((mask - prob)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = 'count')[0]

        
        else:
            ages = survival_ages[i, ~np.isnan(survival_ages[i])]
            mask = dead_mask[i, ~np.isnan(survival_ages[i])]
            prob = survival_prob[i, ~np.isnan(survival_ages[i])]
            G_i = G[i][~np.isnan(survival_ages[i])[:len(G[i])]]

            
            ages = ages[~np.isnan(prob)]
            mask = mask[~np.isnan(prob)]
            G_i = G_i[~np.isnan(prob)]
            prob = prob[~np.isnan(prob)]
            
            ages = ages[mask==1]
            prob = prob[mask==1]
            G_i = G_i[mask==1]
            mask = mask[mask==1]
            
            G_i[G_i < 1e-5] = np.nan

            if len(ages[~np.isnan(G_i)]) != 0:
                BS += sample_weight[i]*binned_statistic(ages[~np.isnan(G_i)], ((mask - prob)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = np.nansum)[0]
                BS_count += sample_weight[i]*binned_statistic(ages[~np.isnan(G_i)], ((mask-prob)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = 'count')[0]

    BS_t = (BS/BS_count)
    min_death_age = death_ages[censored==0].min()
    max_death_age = death_ages[censored==0].max()

    IBS = np.trapz(y = BS_t[ (bin_centers>=min_death_age) & (bin_centers<=max_death_age) ], x = bin_centers[ (bin_centers>=min_death_age) & (bin_centers<=max_death_age) ])/(max_death_age-min_death_age)
    results['IBS'][N] = IBS




# djin and linear
if djin_compare:
    djin_dataset = '_sample' if args.dataset=='sample' else ''
    with open(f'{dir}/../Analysis_Data/IBS_job_id{args.djin_id}_epoch{args.djin_epoch}{djin_dataset}.txt','r') as infile:
        lines = infile.readlines()[0].split(',')
        djin_IBS = float(lines[0])
        if len(lines) > 1:
            linear_IBS = float(lines[1])
        else:
            linear_IBS = None

# plotting code
results.index.name = 'N'
results.reset_index(inplace=True)    
plot = sns.scatterplot(data=results,x='N',y='IBS')
plot.set_xlabel('Model dimension')
plot.set_ylabel('Integrated Brier Score')
plt.ylim(.2,1.6)

if djin_compare:
    custom_legend = []
    labels = []
    plt.scatter(x=29,y=djin_IBS,color='r')
    custom_legend.append(plt.Line2D([], [], marker='o', color='r', linestyle='None'))
    labels.append('DJIN model')
    if linear_IBS is not None:
        plt.plot([0,args.stop],[linear_IBS,linear_IBS],linestyle='--',color='g')
        custom_legend.append(plt.Line2D([], [], color='g', linestyle='--'))
        labels.append('Elastic-net Cox model')
    plt.legend(custom_legend, labels)

fig = plot.get_figure()
postfix = '_sample' if args.dataset=='sample' else ''
fig.savefig(f'{dir}/../Plots/latent_brier_score_by_dim_job_id{args.job_id}_epoch{args.epoch}{postfix}.pdf')

