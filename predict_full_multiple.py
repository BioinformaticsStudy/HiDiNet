import argparse
import numpy as np
import population_averages_dim
import population_std_dim
import train_full
import predict_full

# predicts multiple latent models using different N


parser = argparse.ArgumentParser('Train_Multiple')
parser.add_argument('--job_id', type=int)
parser.add_argument('--batch_size', type=int, default = 1000)
parser.add_argument('--epoch', type=int, default = 1999)
parser.add_argument('--learning_rate', type=float, default = 1e-2)
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

Ns = list(np.arange(args.start,args.end,args.step))
if args.end not in Ns: 
    Ns.append(args.end)

for N in Ns:
    print('*'*100 + f'\npredicting model with N={N}\n' + '*'*100)
    predict_full.predict(args.job_id,args.epoch,args.epoch+1,args.learning_rate,args.gamma_size,args.z_size,args.decoder_size,args.Nflows,args.flow_hidden,args.dataset,N)


