import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import os

# generates a file containing the variables to be used by train_full sorted according to 
# the amount of individuals that have more than one different unique value for that variable
# also creates a file containing the mean and a file containing the std for each variable
# only used for the latent space model

parser = argparse.ArgumentParser('Generate_Variables')
parser.add_argument('--N', type=int, default=29, help='number of deficits to be used in the plot')
args = parser.parse_args()

variables = ['longill', 'limitact', 'effort', 'smkevr', 'smknow', 'mobility', 'country',
             'jointrep', 'fractures', 'gait speed', 'grip dom', 'grip ndom', 
              'FI ADL', 'FI IADL', 'chair','leg raise', 'full tandem', 'srh', 'eye', 'hear', 'func', 'dias', 
              'sys', 'pulse', 'trig', 'crp','hdl','ldl','glucose','igf1','hgb','fib','fer', 'chol', 'wbc', 
              'mch', 'hba1c', 'vitd']


dir = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(f'{dir}/Data/ELSA_cleaned.csv')[['id'] + variables]

# calculating how much variables change
changes = pd.DataFrame(index=variables, columns=['count','label'])
changes.fillna(0,inplace=True)
for i, variable in enumerate(variables):
    print(f'calculating variable {variable}')
    for label,group in data.groupby(['id']):
        group = pd.DataFrame(group)
        if len(group[variable].dropna().unique()) != 1:
            changes['count'][variable] += 1

changes.index.name = 'variable'
changes.reset_index(inplace=True)
changes.sort_values(by=['count'],ascending=False,inplace=True)
changes['label'][:args.N] = 'deficit'
changes['label'][args.N:] = 'background'

# plotting change
sns.set(font_scale=.5)
plot = sns.barplot(data=changes, x='count', y='variable', hue='label', orient='h', dodge=False)
plot.set_xlabel('Number of individuals with more than one value')
fig = plot.get_figure()
fig.tight_layout()
fig.savefig(f'{dir}/Plots/change_distribution{args.N}.pdf')

#creating mean_deficits and std_deficits
means = pd.Series(index=variables)
stds = pd.Series(index=variables)
for d in variables:
    means[d] = data.loc[data[d]>-100,d].mean()
    stds[d] = data.loc[data[d]>-100,d].std()
means.to_csv(f'{dir}/Data/mean_deficits_latent.txt')
stds.to_csv(f'{dir}/Data/std_deficits_latent.txt')

#creating variables.txt
with open(f'{dir}/Data/variables.txt','w') as outfile:
    for variable in changes['variable']:
        outfile.writelines(variable + ',')
    outfile.writelines('alcohol,height,bmi,ethnicity,sex')
