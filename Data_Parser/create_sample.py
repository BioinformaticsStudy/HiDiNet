#creates a sample of elsa_cleaned.csv with a specified number of individuals

import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser('Sample')
parser.add_argument('--size', type=int, default=30,help='the amount of ids to be included in the sample')
args = parser.parse_args()
size = args.size
dir = os.path.dirname(os.path.realpath(__file__))

fullData = pd.read_csv(dir+'/../Data/ELSA_cleaned.csv')
sampleData = fullData.drop_duplicates(subset=['id']).sample(size)
output = pd.DataFrame(columns=fullData.columns)
for i,row in sampleData.iterrows():
    id = row['id']
    individualData = fullData.loc[fullData['id']==id]
    output = pd.concat([output, individualData], ignore_index=True)

output.to_csv(dir+'/../Data/sample_data.csv',index=False)

