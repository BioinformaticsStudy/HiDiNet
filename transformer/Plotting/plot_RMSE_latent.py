# plots the average relative RMSE scores of latent models by N

from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [2]  
sys.path.append(str(package_root_directory))  
import os

import argparse
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Alternate_models.dataset_dim import Dataset
from torch.utils.data import DataLoader
from DataLoader.collate import custom_collate
from Utils.transformation import Transformation

parser = argparse.ArgumentParser('Cindex_Latent')
parser.add_argument('--job_id',type=int)
parser.add_argument('--epoch',type=int,default=1999)
parser.add_argument('--start', type=int,default=2,help='lowest N')
parser.add_argument('--step', type=int, default=5,help='difference in N between models')
parser.add_argument('--stop', type=int,default=35,help='highest N')
parser.add_argument('--dataset', type=str, default='elsa',choices=['elsa','sample'],help='what dataset was used to train the models')
parser.add_argument('--years', type=int, default=6)
parser.add_argument('--djin_id',type=int,default=None)
parser.add_argument('--djin_epoch',type=int,default=None)
parser.add_argument('--lamp_job_id',type=int,default=None)
parser.add_argument('--lamp_epoch',type=int,default=None)
args = parser.parse_args()

djin_compare = args.djin_id != None and args.djin_epoch != None
lamp_compare = args.lamp_job_id != None and args.lamp_epoch != None
device = 'cpu'

dt = 0.5
length = 50
dir = os.path.dirname(os.path.realpath(__file__))



# latent calculations
Ns = list(np.arange(args.start,args.stop,args.step)) + [args.stop] #+ [29]
results = pd.DataFrame(index=Ns,columns=['Mean RMSE']) 
for N in Ns:
    postfix = f'_latent{N}_sample' if args.dataset=='sample' else f'_latent{N}'

    pop_avg = np.load(f'{dir}/../../Data/Population_averages{postfix}.npy')
    pop_avg_env = np.load(f'{dir}/../../Data/Population_averages_env{postfix}.npy')
    pop_std = np.load(f'{dir}/../../Data/Population_std{postfix}.npy')
    pop_avg_ = torch.from_numpy(pop_avg[...,1:]).float()
    pop_avg_env = torch.from_numpy(pop_avg_env).float()
    pop_std = torch.from_numpy(pop_std[...,1:]).float()
    pop_avg_bins = np.arange(40, 105, 3)[:-2]

    min_count = N // 3
    prune = min_count >= 1
    
    test_name = f'{dir}/../../Data/test{postfix}.csv'
    test_set = Dataset(test_name, N, pop=False, min_count=min_count, prune=prune)
    num_test = test_set.__len__()
    test_generator = DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))

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

    # tranformation function
    psi = Transformation(mean_deficits, std_deficits, log_scaled_indexes)

    notmissing = [[] for i in range(N)]
    exact_notmissing = [[] for i in range(N)]
    weights_notmissing = [[] for i in range(N)]
    first_notmissing = [[] for i in range(N)]
    pop_notmissing = [[] for i in range(N)]
    collected_t = []

    mean = np.load(dir+'/../../Analysis_Data/Mean_trajectories_job_id%d_epoch%d%s.npy'%(args.job_id, args.epoch, postfix))
    
    for data in test_generator:
        break

    y = data['Y'].numpy()
    times = data['times'].numpy()
    mask = data['mask'].numpy()
    sample_weight = data['weights'].numpy()
    num_env = 29+19-N-5
    sex_index = data['env'][:,num_env-1].long().numpy()

    # transform
    mean[:,:,1:] = psi.untransform(mean[:,:,1:])
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
    for b in range(min(num_test, mean.shape[0])):
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
                            # not missing
                            if record_mask[b][0, n] >= 1:
                                notmissing[n].append(mean[b, t, n+1])
                                exact_notmissing[n].append(record_y[b][t_rec, n])
                                weights_notmissing[n].append(sample_weight[b])
                                first_notmissing[n].append(record_y[b][0, n])
                                pop_notmissing[n].append(pop_data_t[n])
                    break
                t += 1

    RMSE_notmissing = np.zeros(N)
    RMSE_first_notmissing = np.zeros(N)
    RMSE_pop_notmissing = np.zeros(N)

    #latent calculations
    for n in range(N):
        weights_notmissing[n] = np.array(weights_notmissing[n])
        exact_notmissing[n] = np.array(exact_notmissing[n])
        notmissing[n] = np.array(notmissing[n])
        
        # population and first 
        first_notmissing[n] = np.array(first_notmissing[n])
        pop_notmissing[n] = np.array(pop_notmissing[n])

        #RMSE calculations
        RMSE_notmissing[n] = np.sqrt((weights_notmissing[n] * ((exact_notmissing[n] - notmissing[n]))**2).sum()/np.sum(weights_notmissing[n]))
        RMSE_first_notmissing[n] = np.sqrt((weights_notmissing[n] * ((exact_notmissing[n] - first_notmissing[n]))**2).sum()/np.sum(weights_notmissing[n]))
        RMSE_pop_notmissing[n] = np.sqrt((weights_notmissing[n] * ((exact_notmissing[n] - pop_notmissing[n]))**2).sum()/np.sum(weights_notmissing[n]))


    RMSE_notmissing = RMSE_notmissing / RMSE_pop_notmissing
    averageRMSE = np.mean(RMSE_notmissing)
    print(averageRMSE)
    results['Mean RMSE'][N] = averageRMSE

# djin and linear
if djin_compare:
    djin_dataset = '_sample' if args.dataset=='sample' else ''
    with open(f'{dir}/../../Analysis_Data/average_RMSE_job_id{args.djin_id}_epoch{args.djin_epoch}{djin_dataset}.txt','r') as infile:
        lines = infile.readlines()[0].split(',')
        djin_avg = float(lines[0])
        if len(lines) > 1:
            linear_avg = float(lines[1])
        else:
            linear_avg = None

if lamp_compare:
    with open(f'{dir}/../Analysis_Data_elsa/lamp_average_RMSE_job_id{args.lamp_job_id}_epoch{args.lamp_epoch}.txt','r') as infile:
        value = infile.readlines()
        lamp_avg = float(value[0])


# plotting 
results.index.name = 'N'
results.reset_index(inplace=True)
plot = sns.scatterplot(data=results,x='N',y='Mean RMSE')

if djin_compare and lamp_compare:
    custom_legend = []
    labels = []
    plt.scatter(29, lamp_avg, color='b', marker='v', zorder=5)
    plt.scatter(x=29,y=djin_avg,color='orange')
    custom_legend.append(
        plt.Line2D([], [], marker='v', color='b',
                   markersize=8, linestyle='None'))
    custom_legend.append(plt.Line2D([], [], marker='o', color='orange', linestyle='None'))
    labels.append('HiDiNet‑T')
    labels.append('HiDiNet')
    if linear_avg is not None:
        plt.plot([0,args.stop+1],[linear_avg,linear_avg],linestyle='--',color='g')
        custom_legend.append(plt.Line2D([], [], color='g', linestyle='--'))
        labels.append('Elastic-net Cox model')
    plt.legend(custom_legend, labels, fontsize=14)

plot.set_ylim(.6,1)
plot.set_xlim(0,args.stop+1)
plot.set_xlabel('Model dimension', fontsize=14)
plot.set_ylabel('Mean Relative RMSE', fontsize=14)
fig = plot.get_figure()
postfix = '_sample' if args.dataset=='sample' else ''
fig.savefig(f'{dir}/../Plots/latent_RMSE_by_dim_job_id{args.lamp_job_id}_epoch{args.lamp_epoch}{postfix}.pdf')


