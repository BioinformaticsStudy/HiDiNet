# plots relative RMSE of each health variable

import argparse
import torch
import numpy as np
from scipy.stats import sem
import pandas as pd
import os


from torch.utils import data

from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [2]  
sys.path.append(str(package_root_directory))  

from Utils.transformation import Transformation

from DataLoader.collate import custom_collate
from DataLoader.dataset import Dataset

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cm = plt.get_cmap('Set1')

parser = argparse.ArgumentParser('Predict longitudinal')
parser.add_argument('--job_id', type=int)
parser.add_argument('--epoch', type=int)
parser.add_argument('--lamp_job_id', type=int)
parser.add_argument('--lamp_epoch', type=int)
parser.add_argument('--years', type=int,default=6)
parser.add_argument('--dataset',type=str,choices=['elsa','sample'],default='elsa',help='the dataset that was used to train the model; either \'elsa\' or \'sample\'')
parser.add_argument('--linear_id',type=int,default=1)
parser.add_argument('--djin_id',type=int)
parser.add_argument('--djin_epoch',type=int)
args = parser.parse_args()
postfix = 'sample' if args.dataset == 'sample' else ''

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
pop_avg_bins = np.arange(40, 105, 3)[:-2]

test_name = f'{dir}/../../Data/test{postfix}.csv'
test_set = Dataset(test_name, N, pop=False, min_count = 10)
num_test = test_set.__len__()
test_generator = data.DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))


mean_deficits = pd.read_csv(f'{dir}/../../Data/mean_deficits{postfix}.txt',sep=',',header=None, names = ['variable','value'])[1:N+1]
std_deficits = pd.read_csv(f'{dir}/../../Data/std_deficits{postfix}.txt',sep=',',header=None, names = ['variable','value'])[1:N+1]
mean_deficits.reset_index(inplace=True,drop=True)
std_deficits.reset_index(inplace=True,drop=True)
# get indexes of log scaled variables to be used in Transformation
log_scaled_variables = ['fer','trig','crp', 'wbc', 'mch', 'vitd', 'dheas','leg raise','full tandem']
log_scaled_indexes = []
for variable in log_scaled_variables:
    row = mean_deficits.loc[mean_deficits['variable']==variable]
    if len(row) > 0:
        index = row.index[0]
        log_scaled_indexes.append(index)
mean_deficits.drop(['variable'],axis='columns', inplace=True)
std_deficits.drop(['variable'],axis='columns', inplace=True)
mean_deficits = mean_deficits.values.flatten()
std_deficits = std_deficits.values.flatten()

psi = Transformation(mean_deficits, std_deficits, log_scaled_indexes)

missing_mdiin = [[] for i in range(N)]
notmissing_mdiin = [[] for i in range(N)]
missing_djin = [[] for i in range(N)]
notmissing_djin = [[] for i in range(N)]
missing_lamp = [[] for i in range(N)]
notmissing_lamp = [[] for i in range(N)]
exact_missing = [[] for i in range(N)]
exact_notmissing = [[] for i in range(N)]
weights_missing = [[] for i in range(N)]
weights_notmissing = [[] for i in range(N)]
linear_missing = [[] for i in range(N)]

first_notmissing = [[] for i in range(N)]
first_impute = [[] for i in range(N)]
pop_missing = [[] for i in range(N)]
pop_notmissing = [[] for i in range(N)]
linear_notmissing = [[] for i in range(N)]
collected_t = []
with torch.no_grad():
    mean_mdiin = np.load(dir+ '/../../Analysis_Data/Mean_trajectories_job_id%d_epoch%d_DJIN%s.npy'%(args.job_id,args.epoch,postfix))
    # mean_djin = np.load(dir+ '/../Analysis_Data/Mean_trajectories_job_id%d_epoch%d_DJIN%s.npy'%(args.djin_id,args.djin_epoch,postfix))
    linear = np.load(f'{dir}/../../Comparison_models/Predictions/Longitudinal_predictions_baseline_id{args.linear_id}_rfmice{postfix}.npy')
    lamp = np.load(dir + '/../Analysis_Data_elsa/Mean_trajectories_job_id%d_epoch%d_LAMP.npy'%(args.lamp_job_id,args.lamp_epoch))
    
    for data in test_generator:
        break

    y = data['Y'].numpy()
    times = data['times'].numpy()
    mask = data['mask'].numpy()
    sample_weight = data['weights'].numpy()
    num_env = 29+19-N-5
    sex_index = data['env'][:,num_env-1].long().numpy()
    # transform
    mean_mdiin[:,:,1:] = psi.untransform(mean_mdiin[:,:,1:])
    # mean_djin[:,:,1:] = psi.untransform(mean_djin[:,:,1:])
    linear[:,:,1:] = psi.untransform(linear[:,:,1:])
    lamp[:, :, 1:] = psi.untransform(lamp[:, :, 1:])

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
            
            while t < min(50, int(np.sum(~np.isnan(mean_mdiin[b,:,1])))):
                
                if record_times[b][t_rec] == mean_mdiin[b, t, 0].astype(int):
                    
                    for n in range(N):
                        
                        if record_mask[b][t_rec, n] > 0 and record_times[b][t_rec] - record_times[b][0] <= args.years and record_times[b][t_rec] - record_times[b][0] >= 1:
                            
                            # missing
                            if record_mask[b][0, n] < 1:
                                missing_mdiin[n].append(mean_mdiin[b, t, n+1])
                                # missing_djin[n].append(mean_djin[b, t, n+1])
                                missing_lamp[n].append(lamp[b, t, n+1])
                                exact_missing[n].append(record_y[b][t_rec, n])
                                weights_missing[n].append(sample_weight[b])
                                pop_missing[n].append(pop_data_t[n])
                                first_impute[n].append(mean_mdiin[b, 0, n+1])
                                linear_missing[n].append(linear[b, t, n+1])
                            else:
                                notmissing_mdiin[n].append(mean_mdiin[b, t, n+1])
                                # notmissing_djin[n].append(mean_djin[b, t, n+1])
                                notmissing_lamp[n].append(lamp[b, t, n+1])
                                exact_notmissing[n].append(record_y[b][t_rec, n])
                                weights_notmissing[n].append(sample_weight[b])
                                first_notmissing[n].append(record_y[b][0, n])
                                pop_notmissing[n].append(pop_data_t[n])
                                linear_notmissing[n].append(linear[b, t, n+1])
                    break
                t += 1

RMSE_first_notmissing = np.zeros(N)
RMSE_first_missing = np.zeros(N)
RMSE_pop_missing = np.zeros(N)
RMSE_pop_notmissing = np.zeros(N)

RMSE_missing_mdiin = np.zeros(N)
RMSE_notmissing_mdiin = np.zeros(N)

RMSE_lamp_missing = np.zeros(N)
RMSE_lamp_notmissing = np.zeros(N)

# RMSE_missing_djin = np.zeros(N)
# RMSE_notmissing_djin = np.zeros(N)

RMSE_linear_missing = np.zeros(N)
RMSE_linear_notmissing = np.zeros(N)

for n in range(N):
    # missing
    weights_missing[n] = np.array(weights_missing[n])
    exact_missing[n] = np.array(exact_missing[n])
    missing_mdiin[n] = np.array(missing_mdiin[n])
    missing_djin[n] = np.array(missing_djin[n])
    missing_lamp[n] = np.array(missing_lamp[n])
    linear_missing[n] = np.array(linear_missing[n])

    # not missing
    weights_notmissing[n] = np.array(weights_notmissing[n])
    exact_notmissing[n] = np.array(exact_notmissing[n])
    notmissing_mdiin[n] = np.array(notmissing_mdiin[n])
    notmissing_djin[n] = np.array(notmissing_djin[n])
    notmissing_lamp[n] = np.array(notmissing_lamp[n])
    linear_notmissing[n] = np.array(linear_notmissing[n])
    
    # population and first 
    first_notmissing[n] = np.array(first_notmissing[n])
    first_impute[n] = np.array(first_impute[n])
    pop_notmissing[n] = np.array(pop_notmissing[n])
    pop_missing[n] = np.array(pop_missing[n])

    #RMSE calculations
    RMSE_missing_mdiin[n] = np.sqrt((weights_missing[n] * ((exact_missing[n] - missing_mdiin[n]))**2).sum()/np.sum(weights_missing[n]))
    RMSE_notmissing_mdiin[n] = np.sqrt((weights_notmissing[n] * ((exact_notmissing[n] - notmissing_mdiin[n]))**2).sum()/np.sum(weights_notmissing[n]))

    RMSE_lamp_missing[n] = np.sqrt((weights_missing[n] * ((exact_missing[n] - missing_lamp[n]))**2).sum()/np.sum(weights_missing[n]))
    RMSE_lamp_notmissing[n] = np.sqrt((weights_notmissing[n] * ((exact_notmissing[n] - notmissing_lamp[n]))**2).sum()/np.sum(weights_notmissing[n]))

    # RMSE_missing_djin[n] = np.sqrt((weights_missing[n] * ((exact_missing[n] - missing_djin[n]))**2).sum()/np.sum(weights_missing[n]))
    # RMSE_notmissing_djin[n] = np.sqrt((weights_notmissing[n] * ((exact_notmissing[n] - notmissing_djin[n]))**2).sum()/np.sum(weights_notmissing[n]))
    
    RMSE_linear_missing[n] = np.sqrt((weights_missing[n] * ((exact_missing[n] - linear_missing[n]))**2).sum()/np.sum(weights_missing[n]))
    RMSE_linear_notmissing[n] = np.sqrt((weights_notmissing[n] * ((exact_notmissing[n] - linear_notmissing[n]))**2).sum()/np.sum(weights_notmissing[n]))
    
    RMSE_first_notmissing[n] = np.sqrt((weights_notmissing[n] * ((exact_notmissing[n] - first_notmissing[n]))**2).sum()/np.sum(weights_notmissing[n]))
    RMSE_first_missing[n] = np.sqrt((weights_missing[n] * ((exact_missing[n] - first_impute[n]))**2).sum()/np.sum(weights_missing[n]))
    
    RMSE_pop_missing[n] = np.sqrt((weights_missing[n] * ((exact_missing[n] - pop_missing[n]))**2).sum()/np.sum(weights_missing[n]))
    RMSE_pop_notmissing[n] = np.sqrt((weights_notmissing[n] * ((exact_notmissing[n] - pop_notmissing[n]))**2).sum()/np.sum(weights_notmissing[n]))



RMSE_sort_missing = np.zeros((N,6))
# RMSE_sort_missing[:,5] = RMSE_missing_djin/RMSE_pop_missing
RMSE_sort_missing[:,5] = RMSE_first_missing/RMSE_pop_missing
RMSE_sort_missing[:,4] = RMSE_linear_missing/RMSE_pop_missing
RMSE_sort_missing[:,3] = RMSE_lamp_missing/RMSE_pop_missing # <--- HiDiNet-T
RMSE_sort_missing[:,2] = RMSE_pop_missing
RMSE_sort_missing[:,1] = RMSE_missing_mdiin/RMSE_pop_missing # <--- HiDiNet
RMSE_sort_missing[:,0] = np.arange(N)
missing_index = RMSE_sort_missing[:,1].argsort()
RMSE_sort_missing = RMSE_sort_missing[missing_index]

RMSE_sort_notmissing = np.zeros((N,6))
# RMSE_sort_notmissing[:,5] = RMSE_notmissing_djin/RMSE_pop_notmissing
RMSE_sort_notmissing[:,5] = RMSE_first_notmissing/RMSE_pop_notmissing
RMSE_sort_notmissing[:,4] = RMSE_linear_notmissing/RMSE_pop_notmissing
RMSE_sort_notmissing[:,3] = RMSE_lamp_notmissing/RMSE_pop_notmissing # <--- HiDiNet-T
RMSE_sort_notmissing[:,2] = RMSE_pop_notmissing
RMSE_sort_notmissing[:,1] = RMSE_notmissing_mdiin/RMSE_pop_notmissing # <--- HiDiNet
RMSE_sort_notmissing[:,0] = np.arange(N)
notmissing_index = RMSE_sort_notmissing[:,1].argsort()
RMSE_sort_notmissing = RMSE_sort_notmissing[notmissing_index]

averageRMSE = np.mean(RMSE_sort_notmissing[:,1])
lamp_averageRMSE = np.mean(RMSE_sort_notmissing[:,3])
print(averageRMSE)
print(lamp_averageRMSE)
# if not args.no_compare:
# averageLinear = np.mean(RMSE_sort_notmissing[:,4])
missing_averageRMSE = np.mean(RMSE_sort_missing[:,1])
print('missing avg: ' + str(missing_averageRMSE))


#####MISSING
fig,ax = plt.subplots(figsize=(6.2,5))

deficits_small = np.array(['Gait speed', 'Dom Grip strength', 'Non-dom grip str', 'ADL score','IADL score', 'Chair rises','Leg raise','Full tandem stance', 'Self-rated health', 'Eyesight','Hearing', 'Walking ability', 'Diastolic blood pressure', 'Systolic blood pressure', 'Pulse', 'Triglycerides','C-reactive protein','HDL cholesterol','LDL cholesterol','Glucose','IGF-1','Hemoglobin','Fibrinogen','Ferritin', 'Total cholesterol', r'White blood cell count', 'MCH', 'Glycated hemoglobin', 'Vitamin-D'])

ax.errorbar(np.arange(N)+1, RMSE_sort_missing[:,3], marker = 'v',color = cm(1), markersize = 6, linestyle = '', label = 'HiDiNet-T', zorder= 10000000-1)
ax.errorbar(np.arange(N)+1, RMSE_sort_missing[:,1], marker = '*',color = cm(4),markersize = 6, linestyle = '', label = 'HiDiNet', zorder= 10000000)
ax.errorbar(np.arange(N)+1, RMSE_sort_missing[:,4], marker = 's',color = cm(2),markersize = 5, linestyle = '', label = 'Elastic-net linear models', zorder= 10000)
ax.errorbar(np.arange(N)+1, RMSE_sort_missing[:,5], marker = 'D',color = cm(0), markersize = 5, linestyle = '', label = 'Static model imputed baseline', zorder= 2)

ax.plot([0,N+3],[1,1], color='k', linestyle='--', zorder=-1000, linewidth = 0.75, label = 'Population mean')


ax.set_ylabel(r'Relative RMSE',fontsize = 12)
ax.set_xlim(0, N+1)
ax.set_ylim(0.3, 1.5)
ax.set_yticklabels(['0.6', '0.8', '1.0', '1.2', '1.4'])
ax.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4])



ax.set_xticklabels(np.array(deficits_small)[RMSE_sort_missing[:,0].astype(int)], rotation = 90)
ax.set_xticks(np.arange(1, N+1))
plt.legend(loc='lower right')

ax.text(0.015, 0.94, 'Predictions between 1 and 6 years for an imputed baseline', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes,color='k',fontsize = 11, zorder=1000000)

ax.text(-0.05, 1.05, 'e', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 16, zorder=1000000,
        fontweight='bold')

ax.yaxis.set_minor_locator(MultipleLocator(0.1))

plt.tight_layout()
plt.savefig(dir+'/../Plots/Longitudinal_missing_RMSE_job_id%d_epoch%d.pdf'%(args.lamp_job_id, args.lamp_epoch))




##### NOT MISSING
fig,ax = plt.subplots(figsize=(6.2,5))

deficits_small = np.array(['Gait speed', 'Dom Grip strength', 'Non-dom grip str', 'ADL score','IADL score', 'Chair rises','Leg raise','Full tandem stance', 'Self-rated health', 'Eyesight','Hearing', 'Walking ability', 'Diastolic blood pressure', 'Systolic blood pressure', 'Pulse', 'Triglycerides','C-reactive protein','HDL cholesterol','LDL cholesterol','Glucose','IGF-1','Hemoglobin','Fibrinogen','Ferritin', 'Total cholesterol', r'White blood cell count', 'MCH', 'Glycated hemoglobin', 'Vitamin-D'])

ax.errorbar(np.arange(N)+1, RMSE_sort_notmissing[:,3], marker = 'v',color = cm(1), markersize = 6, linestyle = '', label = 'HiDiNet-T', zorder= 10000000-1)
ax.errorbar(np.arange(N)+1, RMSE_sort_notmissing[:,1], marker = '*',color = cm(4),markersize = 6, linestyle = '', label = 'HiDiNet', zorder= 10000000)
ax.errorbar(np.arange(N)+1, RMSE_sort_notmissing[:,4], marker = 's',color = cm(2),markersize = 5, linestyle = '', label = 'Elastic-net linear models', zorder= 10000)
ax.errorbar(np.arange(N)+1, RMSE_sort_notmissing[:,5], marker = 'D',color = cm(0), markersize = 5, linestyle = '', label = 'Static observed baseline', zorder= 2)

ax.plot([0,N+3],[1,1], color='k', linestyle='--', zorder=-1000, linewidth = 0.75, label = 'Population mean')

ax.set_ylabel(r'Relative RMSE',fontsize = 12)
ax.set_xlim(0, N+1)
ax.set_ylim(0.3, 1.5)

ax.set_xticklabels(np.array(deficits_small)[RMSE_sort_notmissing[:,0].astype(int)], rotation = 90)
ax.set_xticks(np.arange(1, N+1))
plt.legend(loc='lower left', bbox_to_anchor=(0.01, 0.55))

ax.text(0.015,0.92, 'Predictions between 1 and 6 years for an observed baseline', horizontalalignment='left', verticalalignment='bottom',transform=ax.transAxes,color='k',fontsize = 11, zorder=1000000)

ax.text(-0.05, 1.05, 'd', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 16, zorder=1000000,
        fontweight='bold')

ax.yaxis.set_minor_locator(MultipleLocator(0.05))

plt.tight_layout()
plt.savefig(dir+'/../Plots/Longitudinal_RMSE_job_id%d_epoch%d.pdf'%(args.lamp_job_id, args.lamp_epoch))

# avrage
# if args.latentN == None:
# with open(f'{dir}/../../Analysis_Data/average_RMSE_job_id{args.job_id}_epoch{args.epoch}{postfix}.txt','w') as outfile:
#     outfile.writelines(str(averageRMSE))
#     outfile.writelines(',' + str(averageLinear))

with open(f'{dir}/../Analysis_Data_elsa/lamp_average_RMSE_job_id{args.lamp_job_id}_epoch{args.lamp_epoch}.txt','w') as outfile:
    outfile.writelines(str(lamp_averageRMSE))
