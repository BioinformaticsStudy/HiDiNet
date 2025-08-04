# plots time-dependent c-index of multiple latent space models depending on their N
# latent models should be trained with train_full_multiple.py or individually with train_full.py
from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [2]  
sys.path.append(str(package_root_directory))  

import os
import argparse
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Utils.cindex import cindex_td
from Alternate_models.dataset_dim import Dataset
from torch.utils.data import DataLoader
from DataLoader.collate import custom_collate


parser = argparse.ArgumentParser('Cindex_Latent')
parser.add_argument('--job_id',type=int)
parser.add_argument('--epoch',type=int,default=1999)
parser.add_argument('--start', type=int,default=2,help='lowest N')
parser.add_argument('--step', type=int, default=5,help='difference in N between models')
parser.add_argument('--stop', type=int,default=35,help='highest N')
parser.add_argument('--dataset', type=str, default='elsa',choices=['elsa','sample'],help='what dataset was used to train the models')
parser.add_argument('--djin_id',type=int,default=None)
parser.add_argument('--djin_epoch',type=int,default=None)
parser.add_argument('--lamp_job_id',type=int,default=None)
parser.add_argument('--lamp_epoch',type=int,default=None)
args = parser.parse_args()

djin_compare = args.djin_id != None and args.djin_epoch != None
lamp_compare = args.lamp_job_id != None and args.lamp_epoch != None
dir = os.path.dirname(os.path.realpath(__file__))

# latent
Ns = list(np.arange(args.start,args.stop,args.step)) + [args.stop]
results = pd.DataFrame(index=Ns,columns=['C-index'])

for N in Ns:
    postfix = f'_latent{N}_sample' if args.dataset=='sample' else f'_latent{N}'
    test_name = f'{dir}/../../Data/test{postfix}.csv'

    survival = np.load(dir+'/../../Analysis_Data/Survival_trajectories_job_id%d_epoch%d%s.npy'%(args.job_id,args.epoch,postfix))    

    pop_avg = np.load(f'{dir}/../../Data/Population_averages{postfix}.npy')
    pop_avg_env = np.load(f'{dir}/../../Data/Population_averages_env{postfix}.npy')
    pop_std = np.load(f'{dir}/../../Data/Population_std{postfix}.npy')
    pop_avg_ = torch.from_numpy(pop_avg[...,1:]).float()
    pop_avg_env = torch.from_numpy(pop_avg_env).float()
    pop_std = torch.from_numpy(pop_std[...,1:]).float()

    min_count = N // 3
    prune = min_count >= 1
    test_set = Dataset(test_name, N,  pop=False,prune=prune,min_count=min_count)
    num_test = test_set.__len__()
    test_generator = DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))
    for data in test_generator:
        break

    death_ages = data['death age'].numpy()
    censored = data['censored'].numpy()
    times = data['times'].numpy()
    ages = times[:,0]
    death_ages = np.array([death_ages[m] if death_ages[m] > 0 else times[m].max() for m in range(death_ages.size)])
    sample_weight = data['weights'].numpy()

    cindex = cindex_td(death_ages, survival[:,:,1], survival[:,:,0], 1 - censored)
    results['C-index'][N] = cindex

# djin and latent
if djin_compare:
    djin_set = '_sample' if args.dataset=='sample' else ''
    with open(f'{dir}/../../Analysis_Data/overall_cindex_job_id{args.djin_id}_epoch{args.djin_epoch}{djin_set}.txt','r') as infile:
        lines = infile.readlines()[0].split(',')
        djin_cindex = float(lines[0])
        if len(lines) > 1:
            linear_cindex = float(lines[1])
        else:
            linear_cindex = None

if lamp_compare:
    with open(f'{dir}/../Analysis_Data_elsa/overall_cindex_job_id{args.lamp_job_id}_epoch{args.lamp_epoch}.txt','r') as infile:
        value = infile.readlines()
        lamp_cindex = float(value[0])


#plotting 
results.index.name = 'N'
results.reset_index(inplace=True)    
plot = sns.scatterplot(data=results,x='N',y='C-index')
plot.set_xlabel('Model dimension', fontsize=14)
plot.set_ylabel('Survival C-index', fontsize=14)
plt.ylim(.5,1)
plt.xlim(0,args.stop+1)    
if djin_compare and lamp_compare:
    custom_legend = []
    labels = []
    plt.scatter(29, lamp_cindex, color='b', marker='v', zorder=5)
    plt.scatter(x=29,y=djin_cindex,color='orange')
    custom_legend.append(
        plt.Line2D([], [], marker='v', color='b',
                   markersize=8, linestyle='None'))
    custom_legend.append(plt.Line2D([], [], marker='o', color='orange', linestyle='None'))
    labels.append('HiDiNet')
    labels.append('RNN')
    if linear_cindex is not None:
        plt.plot([0,args.stop+1],[linear_cindex,linear_cindex],linestyle='--',color='g')
        custom_legend.append(plt.Line2D([], [], color='g', linestyle='--'))
        labels.append('Elastic-net Cox model')
    plt.legend(custom_legend, labels, fontsize=14)


postfix = '_sample' if args.dataset=='sample' else ''
fig = plot.get_figure()
fig.savefig(f'{dir}/../Plots/latent_cindex_by_dim_job_id{args.lamp_job_id}_epoch{args.lamp_epoch}{postfix}.pdf')

