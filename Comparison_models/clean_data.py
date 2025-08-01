import torch
import numpy as np
from pandas import read_csv
import os

from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))  

from DataLoader.dataset import Dataset
from DataLoader.collate import custom_collate

def clean_test_data(longitudinal=False, data='test'):
    dir = os.path.dirname(os.path.realpath(__file__))
    if data == 'test':
        folder = dir+'/../Data/'+data+'_files/'
    else:
        folder = dir+'/../Data/'+data+'_files/'
    
    # Load all data first
    longitudinal_data = torch.load(folder + 'longitudinal.pt').numpy()
    times_data = torch.load(folder + 'times.pt').numpy()
    death_age = torch.load(folder + 'death_age.pt').numpy()
    censored = torch.load(folder + 'censored.pt').numpy()
    env = torch.load(folder + 'env.pt').numpy()
    med_time = torch.load(folder + 'med_time.pt').numpy()
    weights = torch.load(folder + 'weights.pt').numpy()
    print("\nInspecting test data ages:")
    print(f"Min age: {times_data[:,0].min()}")
    print(f"Max age: {times_data[:,0].max()}")
    print(f"Number of zero ages: {np.sum(times_data[:,0] == 0)}")
    print("---")
    # # Create mask for valid ages (not zero)
    # valid_age_mask = times_data[:,0] != 0
    
    # # Apply mask to all data arrays
    # times_data = times_data[valid_age_mask]
    # longitudinal_data = longitudinal_data[valid_age_mask]
    # death_age = death_age[valid_age_mask]
    # censored = censored[valid_age_mask]
    # env = env[valid_age_mask]
    # med_time = med_time[valid_age_mask]
    # weights = weights[valid_age_mask]
    
    # Create X as before but with filtered data
    X = np.concatenate((times_data[:,0, np.newaxis],
                       longitudinal_data[:,0,:],
                       med_time[:,0,:5],
                       env[:,:14]), axis = 1)
    
    if not longitudinal:
        return X
    else:
        y = longitudinal_data[:,1:,:]
        return X, y

def clean_test_data_predictions(args, device='cpu', N = 29):
  
    pop_avg = np.load(dir+'/../Data/Population_averages.npy')
    pop_avg_env = np.load(dir+'/../Data/Population_averages_env.npy')
    pop_std = np.load(dir+'/../Data/Population_std.npy')
    pop_avg_ = torch.from_numpy(pop_avg[...,1:]).float().to(device)
    pop_avg_env = torch.from_numpy(pop_avg_env).float().to(device)
    pop_std = torch.from_numpy(pop_std[...,1:]).float().to(device)
    pop_avg_bins = np.arange(40, 105, 3)[:-2]
    
    test_name = dir+'/../Data/test.csv'
    test_set = Dataset(test_name, N, pop=False, min_count = 10)
    num_test = test_set.__len__()
    test_generator = torch.utils.data.DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))
    
    mean_deficits = read_csv(dir+'/../Data/mean_deficits.txt', index_col=0,sep=',',header=None, names = ['variable']).values[1:].flatten()
    std_deficits = read_csv(dir+'/../Data/std_deficits.txt', index_col=0,sep=',',header=None, names = ['variable']).values[1:].flatten()


    for data in test_generator:
        break
    
    y = data['Y'].numpy()
    times = data['times'].numpy()
    mask = data['mask'].numpy()
    sample_weight = data['weights'].numpy()
    sex_index = data['env'][:,12].long().numpy()
    

    return y, times, mask, sex_index, sample_weight, pop_avg_.numpy(), pop_avg_bins, num_test
