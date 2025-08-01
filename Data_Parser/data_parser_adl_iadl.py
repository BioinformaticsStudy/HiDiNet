# handles adl and iadl calculations for data_parser.py

import os
import pandas as pd
import numpy as np

print("preparing adl and iadl data...")
dir = os.path.dirname(os.path.realpath(__file__))
folder = dir + "/../ELSA/tab/"

files = ["wave_1_core_data_v3.tab","wave_2_core_data_v4.tab","wave_3_elsa_data_v4.tab",
         "wave_4_elsa_data_v3.tab","wave_5_elsa_data_v4.tab","wave_6_elsa_data_v2.tab",
         "wave_7_elsa_data.tab","wave_8_elsa_data_eul_v2.tab","wave_9_elsa_data_eul_v1.tab"]

adl = []
iadl = []
cond1 = []
cond2 = []
cond3 = []

#wave 1
adl.append(['heada0'+str(i) for i in range(1,10)] + ['heada10'] + ['heada11'])
iadl.append(['headb0'+str(i) for i in range(1,10)] + ['headb1'+str(i) for i in range(0,5)])
cond1.append(['hedib0%s'%i for i in range(1, 10)] + ['hedib10'])
cond2.append(['hedia0%s'%i for i in range(1, 10)] + ['hedia10'])
cond3.append(['heopt%s'%i for i in range(1,6)])

#wave 2
adl.append(['heada0'+str(i) for i in range(1,10)] + ['heada10'])
iadl.append(['headb0'+str(i) for i in range(1,10)] + ['headb1'+str(i) for i in range(0,4)])
cond1.append(['hedib0%s'%i for i in range(1, 10)] + ['hedib10'])
cond2.append(['hedia0%s'%i for i in range(1, 10)]) 
cond3.append(['heopt%s'%i for i in range(1,6)])

#wave3
adl.append(['hemob96','hemobwa','hemobsi','hemobch', 'hemobcs','hemobcl','hemobst','hemobre',
            'hemobpu','hemobli','hemobpi'])
iadl.append(['headl96','headldr','headlwa','headlba',
             'headlea','headlbe','headlwc','headlma','headlpr',
             'headlsh','headlph','headlme','headlho','headlmo'])

#wave 4
adl.append(['hemob96','hemobwa','hemobsi','hemobch', 'hemobcs','hemobcl','hemobst','hemobre',
            'hemobpu','hemobli','hemobpi'])
iadl.append(['headl96','headldr','headlwa','headlba',
             'headlea','headlbe','headlwc','headlma','headlpr',
             'headlsh','headlte','headlme','headlho','headlmo']) 

#wave 5, iadl is not available
adl.append(['headl96','headldr','headlwa','headlba','headlea','headlbe',
            'headlwc','headlma','headlda','headlpr','headlsh','headlte',
            'headlco','headlme','headlho','headlmo'])
iadl.append(['']) 

#wave 6
adl.append(['hemob96','hemobwa','hemobsi','hemobch','hemobcs','hemobcl','hemobst',
            'hemobre','hemobpu','hemobli','hemobpi'])
iadl.append(['headl96','headldr','headlwa','headlba','headlea','headlbe',
             'headlwc','headlma','headlpr','headlsh',
             'headlph','headlme','headlho','headlmo'])

#wave 7
adl.append(['hemob96','hemobwa','hemobsi','hemobch','hemobcs','hemobcl','hemobst',
            'hemobre','hemobpu','hemobli','hemobpi'])
iadl.append(['headl96','headldr','headlwa','headlba','headlea','headlbe',
             'headlwc','headlma','headlpr','headlsh',
             'headlph','headlme','headlho','headlmo'])

#wave 8
adl.append(['hemob96','hemobwa','hemobsi','hemobch','hemobcs','hemobcl','hemobst',
            'hemobre','hemobpu','hemobli','hemobpi'])
iadl.append(['headl96','headldr','headlwa','headlba','headlea','headlbe',
             'headlwc','headlma','headlpr','headlsh',
             'headlph','headlme','headlho','headlmo']) 

#wave 9
adl.append(['hemob96','hemobwa','hemobsi','hemobch','hemobcs','hemobcl','hemobst',
            'hemobre','hemobpu','hemobli','hemobpi'])
iadl.append(['headl96','headldr','headlwa','headlba','headlea','headlbe',
             'headlwc','headlma','headlpr','headlsh',
             'headlph','headlme','headlho','headlmo']) 

cols = []
cols.append(['idauniq','dhager','mmwlka','mmwlkb','dhdobyr'])   #wave 1
cols.append(['idauniq','dhager','MMWlkA','MMWlkB','dhdobyr'])   #wave 2
cols.append(['idauniq','dhager','mmwlka','mmwlkb','dhdobyr'])   #wave 3
cols.append(['idauniq','indager','mmwlka','mmwlkb','indobyr'])  #wave 4
cols.append(['idauniq','indager','indobyr','mmwlka','mmwlkb'])  #wave 5
cols.append(['idauniq','indager','MMWlkA','MMWlkB','Indobyr'])  #wave 6
cols.append(['idauniq','indager','MMWlkA','MMWlkB','Indobyr'])  #wave 7
cols.append(['idauniq','indager','mmwlka','mmwlkb','indobyr'])  #wave 8
cols.append(['idauniq','indager','mmwlka','mmwlkb','indobyr'])  #wave 9

print("reading files...")
waves = []
for i,f in enumerate(files):
    cols[i] += adl[i]
    if(i != 4):
        cols[i] += iadl[i]
    waves.append(pd.read_csv(folder+f, usecols=cols[i], delimiter='\t'))
    print(f + " read")
print("all files read")

# handling collapsed dob
waves[0]['dhdobyr'].replace(-7.0,1912.0,inplace=True)
waves[1]['dhdobyr'].replace(-7.0,1914.0,inplace=True)


#waves 1 and 2
for w in [0,1]:
    waves[w][adl[w]+iadl[w]] = waves[w][adl[w]+iadl[w]].replace([-8.0,-9.0],[np.nan]*2)
    
#waves 3-7 and 9
for w in [2,3,5,6,8]:
    waves[w][adl[w]+iadl[w]] = waves[w][adl[w]+iadl[w]].replace([-1.0,-2.0,-8.0,-9.0],[np.nan]*4)

#wave 5
waves[4][adl[4]] = waves[4][adl[4]].replace([-1.0,-2.0,-8.0,-9.0],[np.nan]*4)

#wave 8   
waves[7][adl[7]+iadl[7]] = waves[7][adl[7]+iadl[7]].replace([-1.0,-2.0,-3.0,-4.0,-8.0,-9.0],[np.nan]*6)

indexes = []
for i in range(len(waves)):
    indexes.append([x for x in waves[i].index.values])
    

N_adl = 10
N_iadl = 13
adl_deficits = {0:{}, 1:{}, 2: {}, 3: {}, 5: {}, 6: {}, 7: {}, 8:{}}
iadl_deficits = {0:{}, 1:{}, 2: {}, 3: {}, 5: {}, 6: {}, 7: {}, 8:{}}

for w in range(0,2):
    print("calculating wave " + str(w+1) + "...")
    waves[w]['ADL'] = int(-1)
    waves[w]['IADL'] = int(-1)
    waves[w]['ADL count'] = 0
    waves[w]['IADL count'] = 0

    for index in np.unique(indexes[w]):
        #adl
        if waves[w].loc[index,adl[w][0]] == 96.0: #had none of listed adl difficulties
            waves[w].loc[index,'ADL'] = 0
            waves[w].loc[index,'ADL count'] = N_adl
            adl_deficits[w][waves[w].loc[index,'idauniq']] = np.zeros(N_adl)
            
        elif np.all(waves[w].loc[index,adl[w][0]] == -1.0) or np.isnan(waves[w].loc[index,adl[w][0]]): #this individual has no available adl info
            waves[w].loc[index,'ADL'] = -1.0
            
        else:
            if w == 0: #first wave
                adl_deficits[w][waves[w].loc[index,'idauniq']] = \
                    waves[w].loc[index,adl[w]].ge(0).values.astype(int)[:-1]
            else: #second wave
                adl_deficits[w][waves[w].loc[index,'idauniq']] = \
                    waves[w].loc[index,adl[w]].ge(0).values.astype(int)
            
            waves[w].loc[index,'ADL'] = waves[w].loc[index,adl[w]].dropna().ge(0).sum() #number of adl variables this individual responded to
            waves[w].loc[index,'ADL count'] = N_adl
        
        #iadl
        if waves[w].loc[index,iadl[w][0]] == 96.0:
            waves[w].loc[index,'IADL'] = 0
            waves[w].loc[index,'IADL count'] = N_iadl
            iadl_deficits[w][waves[w].loc[index,'idauniq']] = np.zeros(N_iadl)
            
        elif np.all(waves[w].loc[index,iadl[w][0]] == -1.0) or np.isnan(waves[w].loc[index,iadl[w][0]]):
            waves[w].loc[index,'IADL'] = -1.0  
        else:
            if w == 0:
                iadl_deficits[w][waves[w].loc[index,'idauniq']] = \
                    waves[w].loc[index,iadl[w]].ge(0).values.astype(int)[:-1]
            else:
                iadl_deficits[w][waves[w].loc[index,'idauniq']] = \
                    waves[w].loc[index,iadl[w]].ge(0).values.astype(int)
            
            waves[w].loc[index,'IADL'] = waves[w].loc[index,iadl[w]].dropna().ge(0).sum()
            waves[w].loc[index,'IADL count'] = N_iadl

#waves 3,4,6,7,8,9
for w in [2,3,5,6,7,8]:
    print("calculating wave " + str(w+1) + "...")
    waves[w]['ADL'] = int(-1)
    waves[w]['IADL'] = int(-1)
    waves[w]['ADL count'] = 0
    waves[w]['IADL count'] = 0

    for index in np.unique(indexes[w]):
        #adl
        if waves[w].loc[index,adl[w][0]] == 1.0:
            waves[w].loc[index,'ADL'] = 0
            waves[w].loc[index,'ADL count'] = N_adl
            adl_deficits[w][waves[w].loc[index,'idauniq']] = np.zeros(N_adl)
            
        elif np.isnan(waves[w].loc[index,adl[w][0]]) or waves[w].loc[index,adl[w][0]] == -1.0:
            waves[w].loc[index,'ADL'] = -1
            
        else:
            adl_deficits[w][waves[w].loc[index,'idauniq']] = \
                waves[w].loc[index,adl[w]].values.astype(int)[1:]
            waves[w].loc[index,'ADL'] = waves[w].loc[index,adl[w]].dropna().sum()
            waves[w].loc[index,'ADL count'] = N_adl
        
        #iadl
        if waves[w].loc[index,iadl[w][0]] == 1.0:
            waves[w].loc[index,'IADL'] = 0
            waves[w].loc[index,'IADL count'] = N_iadl
            iadl_deficits[w][waves[w].loc[index,'idauniq']] = np.zeros(N_iadl)
            
        elif np.isnan(waves[w].loc[index,iadl[w][0]]) or waves[w].loc[index,iadl[w][0]] == -1.0:
            waves[w].loc[index,'IADL'] = -1
        else:
            iadl_deficits[w][waves[w].loc[index,'idauniq']] = \
                waves[w].loc[index,iadl[w]].values.astype(int)[1:]
            waves[w].loc[index,'IADL'] = waves[w].loc[index,iadl[w]].dropna().sum()
            waves[w].loc[index,'IADL count'] = N_iadl
print("all waves calculated")

def calc_ADL(row,w):
    if row['ADL'] >= 0:
        return row['ADL']/row['ADL count']
    else:
        return -1.0
    
def calc_IADL(row,w):
    if row['IADL'] >= 0:
        return row['IADL']/row['IADL count']
    else:
        return -1.0

#waves 1 to 8, skipping 5
for w in [0,1,2,3,5,6,7,8]:
    waves[w]['FI ADL'] = waves[w].apply(lambda row: calc_ADL(row,w) ,axis=1)
    waves[w]['FI IADL'] = waves[w].apply(lambda row: calc_IADL(row,w) ,axis=1)

print("writing to /Data...")
outputColumns = ['idauniq','ADL','IADL','FI ADL','FI IADL','ADL count','IADL count']
for i in [1,2,3,4,6,7,8,9]:
    outfile = dir+'/../Data/ELSA_Frailty_cleaned_wave'+str(i)+'.csv'
    waves[i-1][outputColumns].to_csv(outfile,index=False)
print("adl and iadl data finished")