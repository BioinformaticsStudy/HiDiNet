import pandas as pd
import seaborn as sns
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser('plot_age_distribution')
parser.add_argument('--dataset', type=str,default='elsa', choices=['elsa','sample'],help='what dataset is read in; either \'elsa\' or \'sample\'')
parser.add_argument('--unique', action='store_true')
parser.add_argument('--binsize',type=int,default=10)
args = parser.parse_args()
postfix = '_sample' if args.dataset=='sample' else ''
unqPostfix = '_unique' if args.unique else ''

dir = os.path.dirname(os.path.realpath(__file__))
file = 'sample_data.csv' if args.dataset=='sample' else 'ELSA_cleaned.csv'
data = pd.read_csv(f'{dir}/../Data/{file}')


unqAgeData = pd.DataFrame(data.groupby('id')['age'].min())
if args.unique:
    ageData = unqAgeData
else:
    nonUnqAgeData = pd.DataFrame(data['age'])
    ageData = pd.DataFrame(columns=['age','label'])
    ageData=ageData.append(unqAgeData.assign(label='start'))
    ageData=ageData.append(nonUnqAgeData.assign(label='total'),ignore_index=True)



print(ageData)

bins = np.arange(20,120,args.binsize)
if args.unique:
    plot = sns.histplot(data=ageData, x='age', bins=bins).set_title('Starting Age Distsribution')
else:
    plot = sns.histplot(data=ageData, x='age', hue='label', bins=bins).set_title('Age Distsribution')
fig = plot.get_figure()
fig.savefig(f'{dir}/../Plots/age_distribution{unqPostfix}{postfix}.pdf')