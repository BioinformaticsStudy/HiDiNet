import argparse
import torch
import numpy as np
from scipy.stats import sem
from pandas import read_csv

from torch.utils import data
import os


from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))  

from Utils.transformation import Transformation
from Utils.record import record

from DataLoader.dataset import Dataset
from DataLoader.collate import custom_collate

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cm = plt.get_cmap('Set1')

parser = argparse.ArgumentParser('Predict longitudinal missing %')
parser.add_argument('--job_id', type=int)
parser.add_argument('--epoch', type=int)
parser.add_argument('--years', type=int)
parser.add_argument('--gamma_size', type=int, default = 25)
parser.add_argument('--dataset',type=str,choices=['elsa','sample'],default='elsa',help='the dataset that was used to train the model; either \'elsa\' or \'sample\'')
parser.add_argument('--no_compare',action='store_true',help='whether or not to plot the comparison model')
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
pop_avg_bins = np.arange(40, 105, 3)[:-2]

test_name = f'{dir}/../Data/test{postfix}.csv'
test_set = Dataset(test_name, N, pop=False, min_count = 10)
num_test = test_set.__len__()
test_generator = data.DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))

missing_percent = read_csv(f'{dir}/../Analysis_Data/ELSA_missing_percent.csv').values[:,1]


mean_deficits = read_csv(f'{dir}/../Data/mean_deficits{postfix}.txt', index_col=0,sep=',',header=None, names = ['variable']).values[1:].flatten()
std_deficits = read_csv(f'{dir}/../Data/std_deficits{postfix}.txt', index_col=0,sep=',',header=None, names = ['variable']).values[1:].flatten()

psi = Transformation(mean_deficits[:-3], std_deficits[:-3], [6, 7, 15, 16, 23, 25, 26, 28])

missing = [[] for i in range(N)]
notmissing = [[] for i in range(N)]
exact_missing = [[] for i in range(N)]
exact_notmissing = [[] for i in range(N)]
weights_missing = [[] for i in range(N)]
weights_notmissing = [[] for i in range(N)]
if not args.no_compare:
    linear_missing = [[] for i in range(N)]

first_notmissing = [[] for i in range(N)]
first_impute = [[] for i in range(N)]
pop_missing = [[] for i in range(N)]
pop_notmissing = [[] for i in range(N)]
if not args.no_compare:
    linear_notmissing = [[] for i in range(N)]
with torch.no_grad():

    mean = np.load(dir+'/../Analysis_Data/Mean_trajectories_job_id%d_epoch%d_DJIN%s.npy'%(args.job_id,args.epoch,postfix))
    
    if not args.no_compare:
        linear = np.load(f'{dir}/../Comparison_models/Predictions/Longitudinal_predictions_baseline_id1_rfmice{postfix}.npy')
    
    start = 0
    for data in test_generator:
        break
    
    y = data['Y'].numpy()
    times = data['times'].numpy()
    mask = data['mask'].numpy()
    sample_weight = data['weights'].numpy()
    sex_index = data['env'][:,12].long().numpy()

    # transform
    mean[:,:,1:] = psi.untransform(mean[:,:,1:])
    if not args.no_compare:
        linear[:,:,1:] = psi.untransform(linear[:,:,1:])
    y = psi.untransform(y)
    y = mask*y + (1-mask)*(-1000)
    pop_avg_ = psi.untransform(pop_avg_.numpy())
    
    record_times = []
    record_y= []
    record_mask = []
    for b in range(num_test):
        observed = np.sum(mask[b,:, :], axis = -1) > 0
        record_times.append(times[b, observed].astype(int))
        record_y.append(y[b, observed, :])
        record_mask.append(mask[b, observed, :].astype(int))
    
    for b in range(num_test):
        t = 0
        for t_rec in range(len(record_times[b])):

            t_index = np.digitize(record_times[b][t_rec], pop_avg_bins, right=True)-1
            if t_index < 0:
                t_index = 0
            pop_data_t = pop_avg_[sex_index[b], t_index]
            
            while t < min(50, int(np.sum(~np.isnan(mean[b,:,1])))):
                
                if record_times[b][t_rec] == mean[b, t, 0].astype(int):
                    
                    for n in range(N):
                        
                        if record_mask[b][t_rec, n] > 0 and record_times[b][t_rec] - record_times[b][0] <= args.years and record_times[b][t_rec] - record_times[b][0] >= 1:
                            
                            # missing
                            if record_mask[b][0, n] < 1:
                                missing[n].append(mean[b, t, n+1])
                                exact_missing[n].append(record_y[b][t_rec, n])
                                weights_missing[n].append(sample_weight[b])
                                pop_missing[n].append(pop_data_t[n])
                                first_impute[n].append(mean[b, 0, n+1])
                                if not args.no_compare:
                                    linear_missing[n].append(linear[b, t, n+1])
                            else:
                                notmissing[n].append(mean[b, t, n+1])
                                exact_notmissing[n].append(record_y[b][t_rec, n])
                                weights_notmissing[n].append(sample_weight[b])
                                first_notmissing[n].append(record_y[b][0, n])
                                pop_notmissing[n].append(pop_data_t[n])
                                if not args.no_compare:
                                    linear_notmissing[n].append(linear[b, t, n+1])
                    break
                t += 1

RMSE_missing = np.zeros(N)
RMSE_notmissing = np.zeros(N)
RMSE_first_notmissing = np.zeros(N)
RMSE_first_missing = np.zeros(N)
RMSE_pop_missing = np.zeros(N)
RMSE_pop_notmissing = np.zeros(N)
if not args.no_compare:
    RMSE_linear_missing = np.zeros(N)
    RMSE_linear_notmissing = np.zeros(N)

for n in range(N):

    # missing
    weights_missing[n] = np.array(weights_missing[n])
    exact_missing[n] = np.array(exact_missing[n])
    missing[n] = np.array(missing[n])
    if not args.no_compare:
       linear_missing[n] = np.array(linear_missing[n])

    # not missing
    weights_notmissing[n] = np.array(weights_notmissing[n])
    exact_notmissing[n] = np.array(exact_notmissing[n])
    notmissing[n] = np.array(notmissing[n])
    if not args.no_compare:
        linear_notmissing[n] = np.array(linear_notmissing[n])
    
    # population and first 
    first_notmissing[n] = np.array(first_notmissing[n])
    first_impute[n] = np.array(first_impute[n])
    pop_notmissing[n] = np.array(pop_notmissing[n])
    pop_missing[n] = np.array(pop_missing[n])


    #RMSE
    RMSE_missing[n] = np.sqrt((weights_missing[n] * ((exact_missing[n] - missing[n]))**2).sum()/np.sum(weights_missing[n]))
    RMSE_notmissing[n] = np.sqrt((weights_notmissing[n] * ((exact_notmissing[n] - notmissing[n]))**2).sum()/np.sum(weights_notmissing[n]))
    if not args.no_compare:
        RMSE_linear_missing[n] = np.sqrt((weights_missing[n] * ((exact_missing[n] - linear_missing[n]))**2).sum()/np.sum(weights_missing[n]))
        RMSE_linear_notmissing[n] = np.sqrt((weights_notmissing[n] * ((exact_notmissing[n] - linear_notmissing[n]))**2).sum()/np.sum(weights_notmissing[n]))
    RMSE_first_notmissing[n] = np.sqrt((weights_notmissing[n] * ((exact_notmissing[n] - first_notmissing[n]))**2).sum()/np.sum(weights_notmissing[n]))
    RMSE_first_missing[n] = np.sqrt((weights_missing[n] * ((exact_missing[n] - first_impute[n]))**2).sum()/np.sum(weights_missing[n]))
    RMSE_pop_missing[n] = np.sqrt((weights_missing[n] * ((exact_missing[n] - pop_missing[n]))**2).sum()/np.sum(weights_missing[n]))
    RMSE_pop_notmissing[n] = np.sqrt((weights_notmissing[n] * ((exact_notmissing[n] - pop_notmissing[n]))**2).sum()/np.sum(weights_notmissing[n]))



RMSE_sort_missing = np.zeros((N,5))
RMSE_sort_missing[:,4] = RMSE_first_missing/RMSE_pop_missing
if not args.no_compare:   
    RMSE_sort_missing[:,3] = RMSE_linear_missing/RMSE_pop_missing
RMSE_sort_missing[:,2] = RMSE_pop_missing
RMSE_sort_missing[:,1] = RMSE_missing/RMSE_pop_missing
RMSE_sort_missing[:,0] = np.arange(N)
missing_index = RMSE_sort_missing[:,1].argsort()

RMSE_sort_notmissing = np.zeros((N,6))
RMSE_sort_notmissing[:,5] = RMSE_missing/RMSE_pop_missing
if not args.no_compare:
    RMSE_sort_notmissing[:,4] = RMSE_linear_notmissing/RMSE_pop_notmissing
RMSE_sort_notmissing[:,3] = RMSE_first_notmissing/RMSE_pop_notmissing
RMSE_sort_notmissing[:,2] = RMSE_pop_notmissing
RMSE_sort_notmissing[:,1] = RMSE_notmissing/RMSE_pop_notmissing
RMSE_sort_notmissing[:,0] = np.arange(N)
notmissing_index = RMSE_sort_notmissing[:,1].argsort()

##### Predict longitudinal change average
fig,ax = plt.subplots(figsize=(6.2,5))

deficits_small = ['Gait', 'Grip str dom', 'Grip str ndom','ADL score', 'IADL score', 'Chair rise', 'Leg raise','Full tandem',
                      'SRH', 'Eyesight','Hearing', 'Walking ability', 'Dias BP', 'Sys BP', 'Pulse', 'Trig', 'CRP','HDL','LDL',
                      'Gluc','IGF-1','HGB','Fib','Fer', 'Chol', 'WBC', 'MCH', 'hba1c', 'VIT-D']

ax.errorbar(missing_percent, RMSE_sort_notmissing[:,1], marker = 'o', color = cm(0),markersize = 7, linestyle = '', label = 'DJIN model', zorder= 10000000)

if not args.no_compare:
    ax.errorbar(missing_percent, RMSE_sort_notmissing[:,4], marker = 's',color = cm(2),markersize = 5, linestyle = '', label = 'Elastic net linear models', zorder= 10000)

ax.plot([0,1],[1,1], color='k', linestyle='--', zorder=-1000, linewidth = 0.75, label = 'Population mean')

ax.set_ylabel(r'Relative RMSE',fontsize = 14)
ax.set_xlabel(r'Proportion of variable missing in data',fontsize = 14)
ax.set_ylim(0.55, 1.1)


plt.legend(loc='lower left', bbox_to_anchor=(0.01, 0.75), facecolor='white', framealpha=1)

ax.text(0.02,0.94, 'Longitudinal predictions  between 1 and 6 years', horizontalalignment='left', verticalalignment='bottom',transform=ax.transAxes,color='k',fontsize = 12, zorder=1000000)


ax.yaxis.set_minor_locator(MultipleLocator(0.05))

plt.tight_layout()
plt.savefig(dir+'/../Plots/Longitudinal_RMSE_job_id%d_epoch%d_missing%s.pdf'%(args.job_id, args.epoch,postfix))
