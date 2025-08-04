import argparse
import torch
import numpy as np
from scipy.stats import sem, binned_statistic, chi2
from pandas import read_csv
from torch.utils import data

from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [2]  
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
parser.add_argument('--djin_job_id', type=int, help='MDiiN job ID')
parser.add_argument('--djin_epoch', type=int, help='MDiiN epoch')
parser.add_argument('--lamp_job_id', type=int, help='LAMP job ID')
parser.add_argument('--lamp_epoch', type=int, help='LAMP epoch')
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--dataset',type=str,choices=['elsa','sample'],default='elsa',help='the dataset that was used to train the model; either \'elsa\' or \'sample\'')
parser.add_argument('--linear_id',type=int,default=1)

args = parser.parse_args()
postfix = '_sample' if args.dataset == 'sample' else ''

torch.set_num_threads(4)
device = 'cpu'

dir = os.path.dirname(os.path.realpath(__file__))
N = 29
dt = 0.5
length = 50

pop_avg = np.load(f'{dir}/../../Data/Population_averages{postfix}.npy')
pop_avg_env = np.load(f'{dir}/../../Data/Population_averages_env{postfix}.npy')
pop_std = np.load(f'{dir}/../../Data/Population_std{postfix}.npy')
pop_avg_ = torch.from_numpy(pop_avg[...,1:]).float()
pop_avg_env = torch.from_numpy(pop_avg_env).float()
pop_std = torch.from_numpy(pop_std[...,1:]).float()
pop_avg_bins = np.arange(40, 105, 3)[:-2]

test_name = f'{dir}/../../Data/test{postfix}.csv'
test_set = Dataset(test_name, N, pop=False, min_count = 10)
num_test = test_set.__len__()
test_generator = data.DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))

with torch.no_grad():

    # Load all model predictions
    survival_mdiin = np.load(dir+'/../../Analysis_Data/Survival_trajectories_job_id%d_epoch%d_DJIN%s.npy'%(args.djin_job_id,args.djin_epoch,postfix))
    survival_lamp = np.load(dir+'/../Analysis_Data_elsa/Survival_trajectories_job_id%d_epoch%d_LAMP.npy'%(args.lamp_job_id,args.lamp_epoch))
    linear = np.load(f'{dir}/../../Comparison_models/Predictions/Survival_trajectories_baseline_id{args.linear_id}_rfmice{postfix}.npy')

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

# Calculate D-Calibration for all models
print("Calculating D-Calibration for all models...")

print("  MDiiN model...")
uncen_buckets_mdiin, cen_buckets_mdiin, error_buckets_mdiin, statistic_mdiin, pval_mdiin = calculate_DCal(survival_mdiin)

print("  LAMP model...")
uncen_buckets_lamp, cen_buckets_lamp, error_buckets_lamp, statistic_lamp, pval_lamp = calculate_DCal(survival_lamp)

print("  Linear model...")
uncen_buckets_linear, cen_buckets_linear, error_buckets_linear, statistic_linear, pval_linear = calculate_DCal(linear)

print(f"\nD-Calibration Results:")
print(f"MDiiN:     χ² = {statistic_mdiin:.2f}, p = {pval_mdiin:.3f}")
print(f"HiDiNet-T: χ² = {statistic_lamp:.2f}, p = {pval_lamp:.3f}")
print(f"Linear:    χ² = {statistic_linear:.2f}, p = {pval_linear:.3f}")

print(f"OVERALL_D_CALI_LAMP: {statistic_lamp}")
print(f"OVERALL_D_CALI_LAMP_P: {pval_lamp}")


def plot_DCal_single(model_name, uncen_buckets, cen_buckets, error_buckets, statistic, pval, color, save_suffix):
    """Plot individual D-Calibration plot for a single model (matching original style)"""
    
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    bin_labels = ['[0.,.1)', '[.1,.2)', '[.2,.3)', '[.3,.4)',
                  '[.4,.5)', '[.5,.6)', '[.6,.7)', '[.7,.8)', '[.8,.9)',
                  '[.9,1.]']
    bin_index = np.arange(0, len(bin_labels))

    plt.barh(bin_labels, uncen_buckets + cen_buckets, height=0.98, color=cm(color), label=model_name)
    plt.errorbar(uncen_buckets + cen_buckets, bin_index, xerr=error_buckets, color='k', zorder=10000000, linestyle='')

    ax.text(.7, 0.77, f'{model_name}', horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes, color='k', fontsize=12, zorder=1000000)
    ax.text(.7, 0.71, r'$\chi^2 = {{%.1f}}$'%(statistic), horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes, color='k', fontsize=14, zorder=1000000)
    ax.text(.7, 0.65, r'$p={{%.3f}}$'%(pval), horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes, color='k', fontsize=14, zorder=1000000)

    ax.text(-0.05, 1.05, 'c', horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes, color='k', fontsize=16, zorder=1000000, fontweight='bold')

    plt.plot(2*[0.1], [-0.5, 9.5], linestyle='--', color='k', zorder=100, linewidth=2, label='Uniform')

    plt.legend(handlelength=0.75, handletextpad=0.6)

    plt.xlim(0, 0.15)
    plt.ylim(-0.5, 9.5)
    plt.ylabel('Survival probability', fontsize=14)
    plt.xlabel('Fraction in bin', fontsize=14)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(dir+'/../Plots/D-Calibration_%s_job_id%d_epoch%d%s.pdf'%(save_suffix, args.lamp_job_id, args.lamp_epoch, postfix))

def plot_HiDiNet_with_all_stats():
    """Plot HiDiNet-T bars with all model statistics (matching original style)"""
    
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    bin_labels = ['[0.,.1)', '[.1,.2)', '[.2,.3)', '[.3,.4)',
                  '[.4,.5)', '[.5,.6)', '[.6,.7)', '[.7,.8)', '[.8,.9)',
                  '[.9,1.]']
    bin_index = np.arange(0, len(bin_labels))

    # Show HiDiNet-T bars
    plt.barh(bin_labels, uncen_buckets_lamp + cen_buckets_lamp, height=0.98, color=cm(1), label='HiDiNet')
    plt.errorbar(uncen_buckets_lamp + cen_buckets_lamp, bin_index, xerr=error_buckets_lamp, 
                color='k', zorder=10000000, linestyle='')

    # Add statistics for ALL models (matching original layout)
    ax.text(.7, 0.77, r'HiDiNet', horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes, color='k', fontsize=12, zorder=1000000)
    ax.text(.7, 0.71, r'$\chi^2 = {%.1f}$' % statistic_lamp, horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes, color='k', fontsize=14, zorder=1000000)
    ax.text(.7, 0.65, r'$p={%.3f}$' % pval_lamp, horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes, color='k', fontsize=14, zorder=1000000)

    ax.text(.7, 0.55, r'RNN', horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes, color='k', fontsize=12, zorder=1000000)
    ax.text(.7, 0.49, r'$\chi^2 = {%.1f}$' % statistic_mdiin, horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes, color='k', fontsize=14, zorder=1000000)
    ax.text(.7, 0.43, r'$p={%.3f}$' % pval_mdiin, horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes, color='k', fontsize=14, zorder=1000000)

    ax.text(.71, 0.33, r'E-net Cox', horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes, color='k', fontsize=12, zorder=1000000)
    ax.text(.71, 0.27, r'$\chi^2 = {%.1f}$' % statistic_linear, horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes, color='k', fontsize=14, zorder=1000000)
    ax.text(.71, 0.21, r'$p={%.3f}$' % pval_linear, horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes, color='k', fontsize=14, zorder=1000000)

    ax.text(-0.05, 1.05, 'c', horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes, color='k', fontsize=16, zorder=1000000, fontweight='bold')

    plt.plot(2*[0.1], [-0.5, 9.5], linestyle='--', color='k', zorder=100, linewidth=2, label='Uniform')

    plt.legend(handlelength=0.75, handletextpad=0.6)

    plt.xlim(0, 0.15)
    plt.ylim(-0.5, 9.5)
    plt.ylabel('Survival probability', fontsize=14)
    plt.xlabel('Fraction in bin', fontsize=14)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    if args.output_dir is not None:
        plot_dir = os.path.join(args.output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
    else:
        plot_dir = os.path.join(dir, '../Plots')
        os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_dir+'/D-Calibration_HiDiNet-T_job_id%d_epoch%d%s.pdf'%(args.lamp_job_id, args.lamp_epoch, postfix))

# Create plots
print("\nGenerating plots...")

# 2. Individual HiDiNet-T plot with all model statistics (matching original style)
plot_HiDiNet_with_all_stats()

print(f"\nPlots saved to: {dir}/../Plots/")
print("- D-Calibration_HiDiNet-T_job_id%d_epoch%d%s.pdf (HiDiNet-T focus with all stats)" % (args.lamp_job_id, args.lamp_epoch, postfix))