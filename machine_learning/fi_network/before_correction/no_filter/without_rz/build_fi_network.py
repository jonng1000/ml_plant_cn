# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:28:29 2021

@author: weixiong001

Takes about 1 h 10 min in total
Creates mutual ranks from all features
"""

from datetime import datetime
from scipy import stats
import pandas as pd

FILE = 'feature_ranks.txt'
OUTPUT = 'mutual_ranks.txt'

def get_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

df = pd.read_csv(FILE, sep='\t', index_col=0)

'''
# Missing values as some features will not be assigned feature importance
# scores
df.isna().sum().sum()
Out[12]: 24674
'''
# 112 497 861 rows here, before cleaning
stacked = df.stack()
'''
temp = stacked.loc[[('go_GO:0000030', 'go_GO:0022414'), 
                    ('go_GO:0022414', 'go_GO:0000030'), 
                    ('go_GO:0022414', 'go_GO:0000096')], ]
sel = temp[temp.index.map(frozenset).duplicated(keep=False)]
'''
# Takes 5 min
print('Script started:', get_time())
# 90 898 948 rows after only selecting paired features
pairs = stacked[stacked.index.map(frozenset).duplicated(keep=False)]
print('Script ended:', get_time())
# Takes 1 h
print('Script started:', get_time())
pairs.index = pairs.index.map(frozenset)
# 45 449 474 rows after calculating mutual rank for each pair
# One row for each pair
gmean_pairs = pairs.groupby(pairs.index).apply(stats.gmean)
print('Script ended:', get_time())
# Runs immediately
print('Script started:', get_time())
gmean_df = gmean_pairs.to_frame().reset_index()
gmean_df[['f1', 'f2']] = pd.DataFrame(gmean_df['index'].tolist(), index=gmean_df.index)
gmean_df.drop(columns=['index'], inplace=True)
gmean_df = gmean_df[gmean_df.columns[[1,2,0]]]
gmean_df.rename(columns={0: 'MR'}, inplace=True)
print('Script ended:', get_time())
# Takes 2 min
'''
# Number of rows for each feature pair
>>> gmean_df.shape
(45449474, 3)
# Max and min mutual rank respectively
>>> gmean_df['MR'].max()
7795.60146813574
>>> gmean_df['MR'].min()
1.0
'''
gmean_df.index.name = 'id'
gmean_df.to_csv(OUTPUT, sep='\t')
