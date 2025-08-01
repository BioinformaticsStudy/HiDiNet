from ast import arg
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import os

parser = argparse.ArgumentParser('Plot_Background_Change')
parser.add_argument('--dataset', default='elsa',choices=['elsa','sample'])
args = parser.parse_args()
file = 'sample_data' if args.dataset=='sample' else 'ELSA_cleaned'
postfix = '_sample' if args.dataset=='sample' else ''

medications = ['BP med', 'anticoagulent med', 'chol med', 'hip/knee treat', 'lung/asthma med']    
background = ['longill', 'limitact', 'effort', 'smkevr', 'smknow', 'height', 'bmi', 'mobility', 'country',
              'alcohol', 'jointrep', 'fractures', 'sex', 'ethnicity']
env = medications + background
deficits = ['gait speed', 'grip dom', 'grip ndom', 'FI ADL', 'FI IADL', 'chair','leg raise', 'full tandem', 'srh', 'eye',
          'hear', 'func', 'dias', 'sys', 'pulse', 'trig',
         'crp','hdl','ldl','glucose','igf1','hgb','fib','fer', 'chol', 'wbc', 'mch', 'hba1c', 'vitd']

all = env + deficits
dir = os.path.dirname(os.path.realpath(__file__))

data = pd.read_csv(f'{dir}/../Data/{file}.csv')[['id'] + all]
changes = np.zeros(len(all))

for i, variable in enumerate(all):
    print(f'calculating variable {variable}')
    for label,group in data.groupby(['id']):
        group = pd.DataFrame(group)
        if len(group[variable].dropna().unique()) != 1:
            changes[i] += 1
        

sns.set(font_scale=.5)
plot = sns.barplot(y=all,x=changes, orient='h',color='b')
plot.set_xlabel('Number of individuals with more than one value')
fig = plot.get_figure()
fig.savefig(f'{dir}/../Plots/background_changes_distribution{postfix}.pdf')