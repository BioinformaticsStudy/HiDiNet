import argparse
import torch
import numpy as np
from scipy.stats import sem, binned_statistic, chi2
from pandas import read_csv
from torch.utils import data

from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))  
import os

from DataLoader.dataset import Dataset
from DataLoader.collate import custom_collate

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cm = plt.get_cmap('Set1')


parser = argparse.ArgumentParser('Dcalibration')
parser.add_argument('--job_id', type=int)
parser.add_argument('--epoch', type=int)
parser.add_argument('--dataset',type=str,choices=['elsa','sample'],default='elsa',help='the dataset that was used to train the model; either \'elsa\' or \'sample\'')
parser.add_argument('--linear_id',type=int,default=1)
parser.add_argument('--djin_id',type=int)
parser.add_argument('--djin_epoch',type=int)

args = parser.parse_args()
postfix = '_sample' if args.dataset == 'sample' else ''

torch.set_num_threads(4)
device = 'cpu'

dir = os.path.dirname(os.path.realpath(__file__))
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

with torch.no_grad():

    survival_mdiin = np.load(dir+'/../Analysis_Data/Survival_trajectories_job_id%d_epoch%d_DJIN%s.npy'%(args.job_id,args.epoch,postfix))
    # survival_djin = np.load(dir+'/../Analysis_Data/Survival_trajectories_job_id%d_epoch%d_DJIN%s.npy'%(args.djin_id,args.djin_epoch,postfix))
    linear=np.load(f'{dir}/../Comparison_models/Predictions/Survival_trajectories_baseline_id1_rfmice{postfix}.npy')

    start = 0
    for data in test_generator:
        break

    death_ages = data['death age'].numpy()
    censored = data['censored'].numpy()
    times = data['times'].numpy()
    dead_mask = data['dead_mask'].numpy()
    ages = times[:,0]
    death_ages = np.array([death_ages[m] if death_ages[m] > 0 else times[m].max() for m in range(death_ages.size)])
    
    sample_weight = data['weights'].numpy()
    sex_index = data['env'][:,12].long().numpy()
    

dead_mask = np.concatenate((dead_mask, np.zeros((dead_mask.shape[0], dead_mask.shape[1]*3))), axis = 1)

def calculate_DCal(survival):
    uncensored_list = survival[censored<1,:,1][(dead_mask[censored<1]*survival[censored<1,:,1])>0].flatten()
    bin_edges = np.linspace(0,1,11)
    uncen_buckets = np.zeros(10)
    cen_buckets = np.zeros(10)

    uncen_buckets += np.histogram(uncensored_list, bins=bin_edges)[0]
    for i in range(len(ages)):
        if censored[i] == 1:
            survival[i,:,1][survival[i,:,1] < 1e-10] = 1e-10
            survival[i,:,1][survival[i,:,1] > 1 ] = 1
            
            Sc = survival[i,:,1][(dead_mask[i]*survival[i,:,1]) > 0][0]
            
            bin = (np.digitize(Sc, bin_edges, right=True) - 1)
            
            cen_buckets[bin] += (1 - bin_edges[bin]/Sc)
            
            for j in range(bin-1, -1, -1):
                cen_buckets[j] += 0.1/Sc
            
    buckets = cen_buckets + uncen_buckets
    uncen_buckets /= buckets.sum()
    cen_buckets /= buckets.sum()
    error_buckets = np.sqrt(buckets)/buckets.sum()

    statistic = 10./len(ages) * np.sum( (buckets - len(ages)/10.)**2 )
    pval = 1 - chi2.cdf(statistic, 9)

    return uncen_buckets, cen_buckets, error_buckets, statistic, pval

uncen_buckets_mdiin,cen_buckets_mdiin, \
error_buckets_mdiin, statistic_mdiin, pval_mdiin = calculate_DCal(survival_mdiin)

# uncen_buckets_djin,cen_buckets_djin, \
# error_buckets_djin, statistic_djin, pval_djin = calculate_DCal(survival_djin)

linear_uncen_buckets,linear_cen_buckets, \
linear_error_buckets, linear_statistic, linear_pval = calculate_DCal(linear)


def plot_DCal(model_name,uncen_buckets,cen_buckets,error_buckets,color):
    fig, ax = plt.subplots(figsize=(4.5,4.5))


    bin_labels = ['[0.,.1)', '[.1,.2)', '[.2,.3)', '[.3,.4)',
                      '[.4,.5)', '[.5,.6)', '[.6,.7)', '[.7,.8)', '[.8,.9)',
                      '[.9,1.]']
    bin_index = np.arange(0,len(bin_labels))

    plt.barh(bin_labels,  uncen_buckets + cen_buckets, height=0.98, color = cm(color), label = 'MDiiN')
    plt.errorbar(uncen_buckets + cen_buckets, bin_index, xerr=error_buckets, color = 'k', zorder=10000000, linestyle = '')

    ax.text(.7, 0.77, r'MDiiN model', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 12, zorder=1000000)
    ax.text(.7, 0.71, r'$\chi^2 = {{%.1f}}$'%(statistic_mdiin), horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 14, zorder=1000000)
    ax.text(.7, 0.65, r'$p={{%.1f}}$'%(pval_mdiin), horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 14, zorder=1000000)

    # ax.text(.7, 0.55, r'DJIN model', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 12, zorder=1000000)
    # ax.text(.7, 0.49, r'$\chi^2 = {{%.1f}}$'%(statistic_djin), horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 14, zorder=1000000)
    # ax.text(.7, 0.43, r'$p={{%.1f}}$'%(pval_djin), horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 14, zorder=1000000)

    ax.text(.71, 0.33, r'E-net Cox', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 12, zorder=1000000)
    ax.text(.71, 0.27, r'$\chi^2 = {{%.1f}}$'%(linear_statistic), horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 14, zorder=1000000)
    ax.text(.71, 0.21, r'$p={{%.1f}}$'%(linear_pval), horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 14, zorder=1000000)


    ax.text(-0.05, 1.05, 'c', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 16, zorder=1000000,
            fontweight='bold')


    plt.plot(2*[0.1], [-0.5,9.5], linestyle = '--', color = 'k', zorder=100, linewidth = 2, label='Uniform')

    plt.legend(handlelength=0.75, handletextpad=0.6)


    plt.xlim(0, 0.15)
    plt.ylim(-0.5,9.5)
    plt.ylabel('Survival probability', fontsize=14)
    plt.xlabel('Fraction in bin', fontsize=14)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(dir+'/../Plots/D-Calibration_job_id%d_epoch%d%s%s.pdf'%(args.job_id, args.epoch,model_name,postfix))


plot_DCal('MDiiN',uncen_buckets_mdiin,cen_buckets_mdiin,error_buckets_mdiin,4)
# plot_DCal('DJIN',uncen_buckets_djin,cen_buckets_djin,error_buckets_djin,0)
plot_DCal('Cox',linear_uncen_buckets,linear_cen_buckets,linear_error_buckets,2)
