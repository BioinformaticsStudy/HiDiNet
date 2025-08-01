# reads files created by data_parser.py and split_data.py and creates a file containing information about that data

import pandas as pd
import argparse
import numpy as np
import seaborn as sns
import os

parser = argparse.ArgumentParser('data info')
parser.add_argument('--dataset', type=str,default='elsa', choices=['elsa','sample'],help='what dataset is read in; either \'elsa\' or \'sample\'')
args = parser.parse_args()
dir = os.path.dirname(os.path.realpath(__file__))

def readData(infiles):
    #columns to be output
    columns = ['rows', 'unq ids'] + \
              ['wave ' + str(i) for i in range(10)] + \
              ['age <40s'] + [f'age {decade}s' for decade in range(40,110,10)] + ['age 110+'] + \
              ['start age <40s'] + [f'start age {decade}s' for decade in range(40,100,10)] + \
              ['censored', 'self reported missing', 'nurse missing']

    dataInfo = pd.DataFrame(index=[infiles[file] for file in infiles],columns=columns).fillna(0.0)
    for file in infiles:
        #counting num rows and ids
        data = pd.read_csv(file)
        row = infiles[file]
        ids = data['id']
        dataInfo.loc[row, 'rows'] = len(ids)
        dataInfo.loc[row,'unq ids'] = len(ids.unique())

        #counting censored deaths
        baseline = data.groupby('id').transform('first')
        dataInfo.loc[row,'censored'] = len(baseline.loc[baseline['death age'] < 0]) / len(baseline) 

        #counting missing data
        self_reported_variables = ['gait speed','FI ADL', 'FI IADL','srh','eye','hear','func',
                                   'longill','limitact','effort','smkevr','smknow','mobility','country',
                                   'anticoagulent med', 'chol med', 'hip/knee treat', 'lung/asthma med','BP med',
                                   'alcohol','jointrep','fractures','sex','ethnicity']
        nurse_variables = ['grip dom','grip ndom','chair','leg raise','full tandem','dias',
                           'sys','pulse','trig','crp','hdl','ldl','glucose','igf1','hgb','fib','fer',
                           'chol','wbc','mch','hba1c','vitd','height','bmi']

        total_self_reported = 0.0
        self_reported_missing = 0.0
        total_nurse = 0.0
        nurse_missing = 0.0

        for variable in self_reported_variables:
            total_self_reported += len(data[variable])
            self_reported_missing += len(data.loc[data[variable] < -100])

        nurse_waves = data.loc[data['wave']%2==0]
        for variable in nurse_variables:
            total_nurse += len(nurse_waves[variable])
            nurse_missing += len(nurse_waves.loc[nurse_waves[variable] < -100])

        
        dataInfo.loc[row,'self reported missing'] = self_reported_missing / total_self_reported
        dataInfo.loc[row,'nurse missing'] = nurse_missing / total_nurse



        for i in range(10):
            thisWave = data.loc[data['wave'] == i]
            dataInfo.loc[row,'wave ' + str(i)] = len(thisWave)
        dataInfo.loc[row,'age <40s'] = len(data.loc[data['age'] < 40])
        dataInfo.loc[row,'age 110+'] = len(data.loc[data['age'] >= 110])
        for decade in range(40,110,10):
            thisDecade = data.loc[(data['age'] >= decade) & (data['age'] < decade+10)]
            dataInfo.loc[row,f'age {decade}s'] = len(thisDecade)
        for label,group in data.groupby('id'):
            decade = group['age'].min() // 10
            decade = int(decade)
            if decade < 4: 
                decade = '<4'
            dataInfo.loc[row,f'start age {decade}0s'] += 1

    #calculate the difference between the full data and the sum of the other three rows
    differences = pd.DataFrame(index=['difference'],columns=dataInfo.columns)
    for i in range(len(dataInfo.columns)):
        col = dataInfo.columns[i]
        differences[col]['difference'] = dataInfo[col]['full_data']  - \
                                         dataInfo[col]['train_data'] - \
                                         dataInfo[col]['test_data']  - \
                                         dataInfo[col]['valid_data']
    dataInfo = dataInfo.append(differences)
    return dataInfo
    


postfix = '' if args.dataset=='elsa' else '_sample'

fullData = dir+'/../Data/sample_data.csv' if args.dataset=='sample' else dir+"/../Data/ELSA_cleaned.csv"
trainData = f"{dir}/../Data/train{postfix}.csv"
testData = f"{dir}/../Data/test{postfix}.csv"
validData = f"{dir}/../Data/valid{postfix}.csv"

#map the file name to the row name
infiles = {fullData:"full_data", trainData:"train_data", testData:"test_data", validData:"valid_data"}

data = readData(infiles)
data.to_csv(f"{dir}/../Data/data_info{postfix}.csv")
