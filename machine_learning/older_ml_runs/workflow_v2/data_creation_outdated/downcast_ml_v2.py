# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:24:56 2020

@author: weixiong001

Downcasts ml dataset datatypes to save memory

Modified from downcast_ml.py in
D:\GoogleDrive\machine_learning\my_features\ml_runs_v2
"""
import pandas as pd
from datetime import datetime

def get_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


print('Script started:', get_time())
ML_DATA = 'ml_dataset.txt'
FILE = 'feature_type.txt'
OUTPUT = 'ml_dataset_dc.txt'

data = pd.read_csv(ML_DATA, sep='\t', index_col=0)
ft_df = pd.read_csv(FILE, sep='\t', index_col=0)

cont_feat = ft_df.loc[ft_df['Feature type'] == 'continuous', :].index
cat_feat = ft_df.loc[ft_df['Feature type'] == 'categorical', :].index
all_cont_feat = [x  for x in data.columns if (x.split('_')[0] + '_') in cont_feat]
all_cat_feat = [x for x in data.columns if (x.split('_')[0] + '_') in cat_feat]
data[all_cat_feat] = data[all_cat_feat].fillna(value=0)

print('Operation started:', get_time())
temp = data[all_cat_feat].apply(pd.to_numeric, downcast='integer')
temp2 = data.drop(all_cat_feat, axis=1).apply(pd.to_numeric, downcast='float')
new_data = pd.concat([temp, temp2], axis=1)
print('Operation ended:', get_time())
"""
# Very slow ~1h 45min, from pandas doc, using [] tends to be slow
data[all_cat_feat] = temp
# Below results in upcasting, the [] in the list of columns passed to
# .loc upcasts  my data
# in theory, to_numpy() should work according to the pandas documentation
# data.loc[:, all_cat_feat] = temp.to_numpy()
"""
new_data.to_csv(OUTPUT, sep='\t')
print('Script ended:', get_time())
