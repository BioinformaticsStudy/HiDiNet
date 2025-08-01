import argparse
import torch
import numpy as np
from scipy.stats import sem
from torch.utils import data
import os

from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))  

from DataLoader.dataset import Dataset
from DataLoader.collate import custom_collate
from Utils.cindex import cindex_td

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cm = plt.get_cmap('Set1')


parser = argparse.ArgumentParser('Cindex')
parser.add_argument('--job_id', type=int)
parser.add_argument('--epoch', type=int)
parser.add_argument('--linear_id',type=int, default=1)
parser.add_argument('--djin_id',type=int)
parser.add_argument('--djin_epoch',type=int)
parser.add_argument('--dataset',type=str,choices=['elsa','sample'],default='elsa',help='the dataset that was used to train the model; either \'elsa\' or \'sample\'')


args = parser.parse_args()
postfix = '_sample' if args.dataset == 'sample' else ''
dir = os.path.dirname(os.path.realpath(__file__))

device = 'cpu'

N = 29
dt = 0.5
length = 50

pop_avg = np.load(f'{dir}/../Data/Population_averages{postfix}.npy')
pop_avg_env = np.load(f'{dir}/../Data/Population_averages_env{postfix}.npy')
pop_std = np.load(f'{dir}/../Data/Population_std{postfix}.npy')
pop_avg_ = torch.from_numpy(pop_avg[...,1:]).float()
pop_avg_env = torch.from_numpy(pop_avg_env).float()
pop_std = torch.from_numpy(pop_std[...,1:]).float()

test_name = f'{dir}/../Data/test{postfix}.csv'
test_set = Dataset(test_name, N,  pop=False, min_count=10)
num_test = test_set.__len__()
test_generator = data.DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))


with torch.no_grad():

    survival_mdiin = np.load(dir+'/../Analysis_Data/Survival_trajectories_job_id%d_epoch%d_DJIN%s.npy'%(args.job_id,args.epoch,postfix))
    # survival_djin = np.load('../Analysis_Data/Survival_trajectories_job_id%d_epoch%d_DJIN%s.npy'%(args.djin_id,args.djin_epoch,postfix))
    linear = np.load(f'{dir}/../Comparison_models/Predictions/Survival_trajectories_baseline_id{args.linear_id}_rfmice{postfix}.npy')
    
    start = 0
    for data in test_generator:
        break

    death_ages = data['death age'].numpy()
    censored = data['censored'].numpy()
    times = data['times'].numpy()
    ages = times[:,0]
    death_ages = np.array([death_ages[m] if death_ages[m] > 0 else times[m].max() for m in range(death_ages.size)])
    
    sample_weight = data['weights'].numpy()
    sex_index = data['env'][:,-1].long().numpy()
    

age_bins = np.arange(40, 105, 3)
bin_centers = age_bins[1:] - np.diff(age_bins)

#mdiin calculations
c_index_list_mdiin = np.ones(bin_centers.shape)*np.nan
for j in range(len(age_bins)-1):
    selected = []
    for i in range(death_ages.shape[0]):
        if age_bins[j] <= ages[i] and ages[i] < age_bins[j+1]:
            selected.append(i)
    c_index = cindex_td(death_ages[selected], survival_mdiin[selected,:,1], survival_mdiin[selected,:,0], 1 - censored[selected], weights = sample_weight)
    c_index_list_mdiin[j] = c_index

# djin calculations
# c_index_list_djin = np.ones(bin_centers.shape)*np.nan
# for j in range(len(age_bins)-1):
#     selected = []
#     for i in range(death_ages.shape[0]):
#         if age_bins[j] <= ages[i] and ages[i] < age_bins[j+1]:
#             selected.append(i)
#     c_index = cindex_td(death_ages[selected], survival_djin[selected,:,1], survival_djin[selected,:,0], 1 - censored[selected], weights = sample_weight)
#     c_index_list_djin[j] = c_index

#elastic net calculations
c_index_linear = np.ones(bin_centers.shape)*np.nan
for j in range(len(age_bins)-1):
    selected = []
    for i in range(death_ages.shape[0]):
        if age_bins[j] <= ages[i] and ages[i] < age_bins[j+1]:
            selected.append(i)
    c_index = cindex_td(death_ages[selected], linear[selected,:,1], linear[selected,:,0], 1 - censored[selected], weights = sample_weight)
    c_index_linear[j] = c_index


#### Plot C index
fig,ax = plt.subplots(figsize=(4.5,4.5))

# overall_cindex_djin = cindex_td(death_ages, survival_djin[:,:,1], survival_djin[:,:,0], 1 - censored)
# plt.plot(bin_centers, c_index_list_djin, marker = 'o',color=cm(0), markersize=8, linestyle = '', label = f'DJIN model')
# plt.plot(bin_centers, overall_cindex_djin*np.ones(bin_centers.shape), color = cm(0), linewidth = 2.5, label = '')

overall_cindex_linear = cindex_td(death_ages, linear[:,:,1], survival_mdiin[:,:,0], 1 - censored)
plt.plot(bin_centers, c_index_linear, marker = 's',color=cm(2), markersize=7, linestyle = '', label = 'Elastic-net Cox')
plt.plot(bin_centers, overall_cindex_linear*np.ones(bin_centers.shape), color = cm(2), linewidth = 2.5, label = '', linestyle = '--')

overall_cindex_mdiin = cindex_td(death_ages, survival_mdiin[:,:,1], survival_mdiin[:,:,0], 1 - censored)
plt.plot(bin_centers, c_index_list_mdiin, marker = 'o',color=cm(4), markersize=8, linestyle = '', label = f'MDiiN model')
plt.plot(bin_centers, overall_cindex_mdiin*np.ones(bin_centers.shape), color = cm(4), linewidth = 2.5, label = '')

print(overall_cindex_mdiin)

plt.ylabel('Survival C-index', fontsize = 14)
plt.xlabel('Baseline age (years)', fontsize = 14)
plt.legend(loc = 'lower right')

ax.text(-0.05, 1.05, 'a', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 16, zorder=1000000,
        fontweight='bold')

plt.ylim(0.5, 1.02)
plt.xlim(60,100)
ax.tick_params(labelsize=12)

ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))

plt.tight_layout()
plt.savefig(dir+'/../Plots/Survival_Cindex_job_id%d_epoch%d_MDiiN%s.pdf'%(args.job_id, args.epoch,postfix))

# with open(f'../Analysis_Data/overall_cindex_job_id{args.job_id}_epoch{args.epoch}{postfix}.txt','w') as outfile:
#     outfile.writelines(str(overall_cindex))
#     if not args.no_compare:
#         outfile.writelines(',' + str(overall_cindex_linear))