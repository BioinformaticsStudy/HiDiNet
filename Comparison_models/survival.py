# trains and predicts survival data for linear model

import argparse
import numpy as np
import pandas as pd
import sys
import os


from clean_data import clean_test_data

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from lifelines import CoxPHFitter


parser = argparse.ArgumentParser('Predict change')
parser.add_argument('--param_id', type=int)
parser.add_argument('--alpha', type=float, default = 0.0001)
parser.add_argument('--l1_ratio', type=float, default = 0.0)
parser.add_argument('--max_depth', type=int, default = 10)
parser.add_argument('--dataset', type=str, choices=['train','train_sample'],default = 'train')
args = parser.parse_args()

deficits = ['gait speed', 'grip dom', 'grip ndom', 'FI ADL', 'FI IADL', 'chair','leg raise', 'full tandem', 'srh', 'eye',
          'hear', 'func', 'dias', 'sys', 'pulse', 'trig',
         'crp','hdl','ldl','glucose','igf1','hgb','fib','fer', 'chol', 'wbc', 'mch', 'hba1c', 'vitd']

        
medications = ['BP med', 'anticoagulent med', 'chol med', 'hip/knee treat', 'lung/asthma med']

background = ['longill', 'limitact', 'effort', 'smkevr', 'smknow', 'height', 'bmi', 'mobility', 'country',
              'alcohol', 'jointrep', 'fractures', 'sex', 'ethnicity']
    
postfix = '_sample' if args.dataset=='train_sample' else ''
if args.param_id is None:
    sys.exit('param_id must be specified')
dir = os.path.dirname(os.path.realpath(__file__))

def clean_data(data):

    data['status'] = 0
    data['weight'] = 1
    
    
    initial_index = []
    status = []
    for label, group in data.groupby('id'):
        
        # get baseline index
        initial_index.append(group.index.values[0])
        
        # fix death ages of censored
        if group['death age'].values[0] < 0:
            data.loc[data['id'] == label, 'death age'] = group['age'].max()
            data.loc[data['id'] == label, 'status'] = 0
        else:
            data.loc[data['id'] == label, 'status'] = 1

        data.loc[data['id'] == label, 'weight'] = 1./len(group)
    
    X = data[['age'] + deficits + medications + background]
    y = data[['weight','status', 'death age']].values

    return X, y, initial_index
    
train_data = pd.read_csv(f'{dir}/../Data/{args.dataset}.csv')
X_train, y_train, initial_index = clean_data(train_data)

min_values = X_train[X_train > -100].min().values
max_values = X_train.max().values
mask_train = (X_train.values > -100).astype(int)
mask_sum = mask_train[:,1:30].sum(-1)
X_train = X_train[mask_sum > 5]
y_train = y_train[mask_sum > 5]

print("Ages before imputation:")
print(f"Min age: {X_train['age'].min()}")
print(f"Max age: {X_train['age'].max()}")
print("---")
print("Number of missing ages:", np.sum(X_train['age'] == -1000))

# MICE imputation
imp = IterativeImputer(estimator = RandomForestRegressor(n_estimators=40, max_depth = args.max_depth, n_jobs=40), random_state=0, missing_values = -1000, max_iter=100, verbose=2)

print('starting imputation')
# Fit and transform without age column
imp.fit(X_train[deficits + medications + background])
X_train_imputed = imp.transform(X_train[deficits + medications + background])
print('imputation done')

# Concatenate age and imputed data
X_train_imputed = np.concatenate((X_train[['age']].values, X_train_imputed), axis=1)
X_train_imputed = np.concatenate((X_train_imputed, y_train), axis=1)
df_train = pd.DataFrame(X_train_imputed, columns = ['age'] + deficits + medications + background + ['weight','status', 'death age'])
df_train['age'] = df_train['age']/100.0
print('imputation transformation done')

cph = CoxPHFitter(penalizer = args.alpha, l1_ratio = args.l1_ratio, baseline_estimation_method='breslow')
cph.fit(df_train, duration_col = 'death age', event_col = 'status', weights_col = 'weight')

print('cph trained')

#####test
#X_test = clean_test_data(data=args.dataset)
#X_test_imputed = imp.transform(X_test)
#X_test_imputed = imp.transform(X_test[deficits + medications + background])
#X_test_imputed = np.concatenate((X_test[['age']].values, X_test_imputed), axis=1)
#####test
X_test = clean_test_data(data='test')
print("\nAges right after clean_test_data:")
print(f"Min age in X_test: {X_test[:,0].min()}")
print(f"Max age in X_test: {X_test[:,0].max()}")
print("---")
# Store age separately before imputation
age_test = X_test[:,0]  # Assuming age is the first column
print("\nAges after separation:")
print(f"Min age in age_test: {age_test.min()}")
print(f"Max age in age_test: {age_test.max()}")
print("---")
# Get all columns except age for imputation
X_test_no_age = X_test[:,1:]  

# Do imputation
X_test_imputed = imp.transform(X_test_no_age)
# Add age back
X_test_imputed = np.concatenate((age_test.reshape(-1,1), X_test_imputed), axis=1)
print("\nAges after concatenation back:")
print(f"Min age in X_test_imputed: {X_test_imputed[:,0].min()}")
print(f"Max age in X_test_imputed: {X_test_imputed[:,0].max()}")
print("---")

print('test data ready')

df_test = pd.DataFrame(X_test_imputed, columns = ['age'] + deficits + medications + background)
print("\nAges in DataFrame:")
print(f"Min age in df_test: {df_test['age'].min()}")
print(f"Max age in df_test: {df_test['age'].max()}")
print("---")

ages = df_test['age'].values*1.0
df_test['age'] = df_test['age']/100.0
print("\nFinal ages:")
print(f"Min age in ages array: {ages.min()}")
print(f"Max age in ages array: {ages.max()}")
print("---")

results = np.zeros((df_test.shape[0], 100, 2)) * np.nan
for i in range(df_test.shape[0]):
    
    unconditioned_sf = cph.predict_survival_function(df_test.iloc[i][['age'] + deficits + medications + background], np.arange(ages[i],110,1)).values[:,0]
    predicted = unconditioned_sf/unconditioned_sf[0]
    
    print(f"Person {i}:")
    print(f"Age: {ages[i]}, Array length: {len(np.arange(ages[i],110,1))}")
    print(f"Shape of results array: {results.shape}")
    print("---")
    
    results[i,:len(np.arange(ages[i],110,1)),0] = np.arange(ages[i],110,1)
    results[i,:len(np.arange(ages[i],110,1)),1] = predicted

print("Final shape before saving:", results.shape)
np.save(dir+'/Predictions/Survival_trajectories_baseline_id%d_rfmice%s.npy'%(args.param_id,postfix), results)