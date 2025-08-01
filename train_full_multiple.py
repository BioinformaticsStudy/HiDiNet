# trains multiple latent models using different N

import argparse
import numpy as np
import population_averages_dim
import population_std_dim
import train_full
import predict_full
import os
from Data_Parser.split_data import create_cv



parser = argparse.ArgumentParser('Train_Multiple')
parser.add_argument('--job_id', type=int)
parser.add_argument('--batch_size', type=int, default = 1000)
parser.add_argument('--niters', type=int, default = 2000)
parser.add_argument('--learning_rate', type=float, default = 1e-2)
parser.add_argument('--corruption', type=float, default = 0.9)
parser.add_argument('--gamma_size', type=int, default = 25)
parser.add_argument('--z_size', type=int, default = 30)
parser.add_argument('--decoder_size', type=int, default = 65)
parser.add_argument('--Nflows', type=int, default = 3)
parser.add_argument('--flow_hidden', type=int, default = 24)
parser.add_argument('--dataset',type=str, default='elsa',choices=['elsa','sample'])
parser.add_argument('--start', type=int,default=2,help='N for the first model')
parser.add_argument('--step', type=int, default=5, help='difference between number of deficits in each model')
parser.add_argument('--end', type=int, default=35, help='N for the final model')
args = parser.parse_args()

Ns = list(np.arange(args.start,args.end,args.step)) + [args.end] + [29]
dir = os.path.dirname(os.path.realpath(__file__))


for N in Ns:
    print('*'*100 + f'\ntraining model with N={N}\n' + '*'*100)
    # split data
    postfix = f'_latent{N}_sample' if args.dataset=='sample' else f'_latent{N}'
    if not os.path.isdir(f'{dir}/Data/train{postfix}_files'):
        create_cv(args.dataset,N) 
        
    population_averages_dim.run(args.dataset,N)
    population_std_dim.run(args.dataset,N)
    train_full.train(args.job_id, args.batch_size, args.niters, args.learning_rate, args.corruption, args.gamma_size, args.z_size, args.decoder_size, args.Nflows, args.flow_hidden, args.dataset, N)
    predict_full.predict(args.job_id,args.niters-1,args.niters,args.learning_rate,args.gamma_size,args.z_size,args.decoder_size,args.Nflows,args.flow_hidden,args.dataset,N)
