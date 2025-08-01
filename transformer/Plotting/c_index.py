import argparse
import torch
import numpy as np
from scipy.stats import sem, ttest_rel
from torch.utils import data
import os

from pathlib import Path
import sys

file = Path(__file__). resolve()  
package_root_directory = file.parents [2]  
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
parser.add_argument('--djin_job_id', type=int)
parser.add_argument('--djin_epoch', type=int)
parser.add_argument('--lamp_job_id', type=int)
parser.add_argument('--lamp_epoch', type=int)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--linear_id',type=int, default=1)
parser.add_argument('--djin_id',type=int)
parser.add_argument('--djin_epoch',type=int)
parser.add_argument('--dataset',type=str,choices=['elsa','sample'],default='elsa',help='the dataset that was used to train the model; either \'elsa\' or \'sample\'')

def bootstrap_cindex_comparison(death_ages, mdiin_survival, lamp_survival, 
                              mdiin_times, lamp_times, censored, 
                              sample_weights=None, n_bootstrap=1000):
    
    n_patients = len(death_ages)
    mdiin_cindex_bootstrap = []
    lamp_cindex_bootstrap = []
    
    print(f"Running {n_bootstrap} bootstrap samples...")
    
    for i in range(n_bootstrap):
        
        print(f"Bootstrap sample {i}/{n_bootstrap}")
            
        # Bootstrap sampling: randomly sample patients with replacement
        bootstrap_indices = np.random.choice(n_patients, size=n_patients, replace=True)
        
        # Get bootstrap sample data
        bootstrap_death_ages = death_ages[bootstrap_indices]
        bootstrap_censored = censored[bootstrap_indices]
        bootstrap_mdiin_survival = mdiin_survival[bootstrap_indices]
        bootstrap_lamp_survival = lamp_survival[bootstrap_indices]
        bootstrap_mdiin_times = mdiin_times[bootstrap_indices] if mdiin_times.ndim > 1 else mdiin_times
        bootstrap_lamp_times = lamp_times[bootstrap_indices] if lamp_times.ndim > 1 else lamp_times
        
        if sample_weights is not None:
            bootstrap_weights = sample_weights[bootstrap_indices]
        else:
            bootstrap_weights = None
            
        # Calculate C-index for both models on bootstrap sample
        try:
            mdiin_cindex = cindex_td(bootstrap_death_ages, 
                                   bootstrap_mdiin_survival[:,:,1], 
                                   bootstrap_mdiin_survival[:,:,0], 
                                   1 - bootstrap_censored, 
                                   weights=bootstrap_weights)
            
            lamp_cindex = cindex_td(bootstrap_death_ages,
                                  bootstrap_lamp_survival[:,:,1], 
                                  bootstrap_lamp_survival[:,:,0], 
                                  1 - bootstrap_censored,
                                  weights=bootstrap_weights)
            
            mdiin_cindex_bootstrap.append(mdiin_cindex)
            lamp_cindex_bootstrap.append(lamp_cindex)
            
        except Exception as e:
            print(f"Error in bootstrap sample {i}: {e}")
            continue
    
    # Convert to numpy arrays
    mdiin_cindex_bootstrap = np.array(mdiin_cindex_bootstrap)
    lamp_cindex_bootstrap = np.array(lamp_cindex_bootstrap)
    
    # Perform paired t-test
    t_statistic, p_value = ttest_rel(lamp_cindex_bootstrap, mdiin_cindex_bootstrap)
    
    # Calculate summary statistics
    mdiin_mean = np.mean(mdiin_cindex_bootstrap)
    lamp_mean = np.mean(lamp_cindex_bootstrap)
    mdiin_std = np.std(mdiin_cindex_bootstrap)
    lamp_std = np.std(lamp_cindex_bootstrap)
    difference_mean = lamp_mean - mdiin_mean
    difference_std = np.std(lamp_cindex_bootstrap - mdiin_cindex_bootstrap)
    
    print("\n" + "="*60)
    print("BOOTSTRAP C-INDEX COMPARISON RESULTS")
    print("="*60)
    print(f"Number of successful bootstrap samples: {len(mdiin_cindex_bootstrap)}")
    print(f"\nMDiiN C-index: {mdiin_mean:.6f} ± {mdiin_std:.6f}")
    print(f"LAMP C-index:  {lamp_mean:.6f} ± {lamp_std:.6f}")
    print(f"Difference:    {difference_mean:.6f} ± {difference_std:.6f}")
    print(f"\nT-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.6f}")
    
    # Interpret results
    print(f"\n" + "-"*60)
    print("STATISTICAL INTERPRETATION:")
    if p_value < 0.001:
        print("*** HIGHLY SIGNIFICANT *** (p < 0.001)")
        print("The improvement is extremely unlikely to be due to chance.")
    elif p_value < 0.01:
        print("** VERY SIGNIFICANT ** (p < 0.01)")
        print("The improvement is very unlikely to be due to chance.")
    elif p_value < 0.05:
        print("* SIGNIFICANT * (p < 0.05)")
        print("The improvement is unlikely to be due to chance.")
    else:
        print("NOT SIGNIFICANT (p >= 0.05)")
        print("The improvement could be due to random chance.")
    
    print("-"*60)
    
    return mdiin_cindex_bootstrap, lamp_cindex_bootstrap, p_value


def plot_bootstrap_results(mdiin_cindex_bootstrap, lamp_cindex_bootstrap):
    """Plot bootstrap results"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Histograms
    ax1.hist(mdiin_cindex_bootstrap, bins=30, alpha=0.7, label='MDiiN', color='blue')
    ax1.hist(lamp_cindex_bootstrap, bins=30, alpha=0.7, label='LAMP', color='red')
    ax1.axvline(np.mean(mdiin_cindex_bootstrap), color='blue', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(lamp_cindex_bootstrap), color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('C-index')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Bootstrap C-index Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference distribution
    differences = lamp_cindex_bootstrap - mdiin_cindex_bootstrap
    ax2.hist(differences, bins=30, alpha=0.7, color='green')
    ax2.axvline(np.mean(differences), color='black', linestyle='--', linewidth=2, 
                label=f'Mean diff: {np.mean(differences):.6f}')
    ax2.axvline(0, color='red', linestyle='-', linewidth=2, label='No difference')
    ax2.set_xlabel('C-index Difference (LAMP - MDiiN)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Bootstrap Difference Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(dir+'/../Plots/Survival_Cindex_TTest_job_id%d_epoch%d_LAMP%s.pdf'%(args.lamp_job_id, args.lamp_epoch,postfix))
    
    return fig
    


args = parser.parse_args()
postfix = '_sample' if args.dataset == 'sample' else ''
dir = os.path.dirname(os.path.realpath(__file__))

device = 'cpu'

N = 29
dt = 0.5
length = 50

pop_avg = np.load(f'{dir}/../../Data/Population_averages{postfix}.npy')
pop_avg_env = np.load(f'{dir}/../../Data/Population_averages_env{postfix}.npy')
pop_std = np.load(f'{dir}/../../Data/Population_std{postfix}.npy')
pop_avg_ = torch.from_numpy(pop_avg[...,1:]).float()
pop_avg_env = torch.from_numpy(pop_avg_env).float()
pop_std = torch.from_numpy(pop_std[...,1:]).float()

test_name = f'{dir}/../../Data/test{postfix}.csv'
test_set = Dataset(test_name, N,  pop=False, min_count=10)
num_test = test_set.__len__()
test_generator = data.DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))

with torch.no_grad():
    survival_mdiin = np.load(dir+'/../../Analysis_Data/Survival_trajectories_job_id%d_epoch%d_DJIN.npy'%(args.djin_job_id,args.djin_epoch)) #will add postfix later
    # survival_djin = np.load('../Analysis_Data/Survival_trajectories_job_id%d_epoch%d_DJIN%s.npy'%(args.djin_id,args.djin_epoch,postfix))
    linear = np.load(f'{dir}/../../Comparison_models/Predictions/Survival_trajectories_baseline_id{args.linear_id}_rfmice.npy') #will add postfix later
    lamp = np.load(dir+'/../Analysis_Data_elsa/Survival_trajectories_job_id%d_epoch%d_LAMP.npy'%(args.lamp_job_id,args.lamp_epoch))  # will add postfix later


    
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

    # mdiin_bootstrap, lamp_bootstrap, p_val = bootstrap_cindex_comparison(
    # death_ages=death_ages,
    # mdiin_survival=survival_mdiin,
    # lamp_survival=lamp,
    # mdiin_times=survival_mdiin[:,:,0],
    # lamp_times=lamp[:,:,0], 
    # censored=censored,
    # sample_weights=sample_weight,
    # n_bootstrap=1000
    # )
    

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

# lamp calculations
c_index_list_lamp = np.ones(bin_centers.shape)*np.nan
for j in range(len(age_bins)-1):
    selected = []
    for i in range(death_ages.shape[0]):
        if age_bins[j] <= ages[i] and ages[i] < age_bins[j+1]:
            selected.append(i)
    c_index = cindex_td(death_ages[selected], lamp[selected,:,1], lamp[selected,:,0], 1 - censored[selected], weights = sample_weight)
    c_index_list_lamp[j] = c_index

#### Plot C index
fig,ax = plt.subplots(figsize=(4.5,4.5))

# overall_cindex_djin = cindex_td(death_ages, survival_djin[:,:,1], survival_djin[:,:,0], 1 - censored)
# plt.plot(bin_centers, c_index_list_djin, marker = 'o',color=cm(0), markersize=8, linestyle = '', label = f'DJIN model')
# plt.plot(bin_centers, overall_cindex_djin*np.ones(bin_centers.shape), color = cm(0), linewidth = 2.5, label = '')

overall_cindex_lamp = cindex_td(death_ages, lamp[:,:,1], lamp[:,:,0], 1 - censored)
plt.plot(bin_centers, c_index_list_lamp, marker = 'v',color=cm(1), markersize=8, linestyle = '', label = f'HiDiNet-T')
plt.plot(bin_centers, overall_cindex_lamp*np.ones(bin_centers.shape), color = cm(1), linewidth = 2.5, label = '')


overall_cindex_mdiin = cindex_td(death_ages, survival_mdiin[:,:,1], survival_mdiin[:,:,0], 1 - censored)
plt.plot(bin_centers, c_index_list_mdiin, marker = 'o',color=cm(4), markersize=8, linestyle = '', label = f'HiDiNet')
plt.plot(bin_centers, overall_cindex_mdiin*np.ones(bin_centers.shape), color = cm(4), linewidth = 2.5, label = '')


overall_cindex_linear = cindex_td(death_ages, linear[:,:,1], survival_mdiin[:,:,0], 1 - censored)
plt.plot(bin_centers, c_index_linear, marker = 's',color=cm(2), markersize=7, linestyle = '', label = 'Elastic-net Cox')
plt.plot(bin_centers, overall_cindex_linear*np.ones(bin_centers.shape), color = cm(2), linewidth = 2.5, label = '', linestyle = '--')


print("MDiiN overall c-index:", overall_cindex_mdiin)
print(f"OVERALL_CINDEX_LAMP: {overall_cindex_lamp}")

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
if args.output_dir is not None:
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
else:
    plot_dir = os.path.join(dir, '../Plots')
    os.makedirs(plot_dir, exist_ok=True)
plt.savefig(plot_dir +'/Survival_Cindex_job_id%d_epoch%d_LAMP%s.pdf'%(args.lamp_job_id, args.lamp_epoch,postfix))

# with open(f'{dir}/../../Analysis_Data/overall_cindex_job_id{args.djin_job_id}_epoch{args.djin_epoch_id}{postfix}.txt','w') as outfile:
#     outfile.writelines(str(overall_cindex_mdiin))
#     # If you want to also save the linear c-index, uncomment the next line:
#     outfile.writelines(',' + str(overall_cindex_linear))

# with open(f'{dir}/../Analysis_Data_elsa/overall_cindex_job_id{args.lamp_job_id}_epoch{args.lamp_epoch}{postfix}.txt','w') as outfile:
#     outfile.writelines(str(overall_cindex_lamp))