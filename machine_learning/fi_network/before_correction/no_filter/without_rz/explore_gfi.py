# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:28:29 2021

@author: weixiong001

Takes about 1 h 10 min in total
Calculates geometric mean from all feature importances
Plots histogram of this info
"""

from datetime import datetime
from scipy import stats
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

FILE = 'big_fi.txt'
OUTPUT = 'gmean_fi.txt'
FIG = 'gmean_fi_hist.png'

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

##########################################################################
# Plotting all scores
all_values = stacked.reset_index()
all_values.rename(columns={0: 'fi'}, inplace=True)

g = sns.histplot(data=all_values, x='fi', bins=16, log_scale=(False, True))
g.figure.savefig(FIG)
plt.close()
##########################################################################

# Takes 5 min
print('Script started:', get_time())
# 90 898 948 rows after only selecting paired features
pairs = stacked[stacked.index.map(frozenset).duplicated(keep=False)]
# Takes 1 h
pairs.index = pairs.index.map(frozenset)
# 45 449 474 rows after calculating mutual rank for each pair
# One row for each pair
gmean_pairs = pairs.groupby(pairs.index).apply(stats.gmean)
print('Script ended:', get_time())

gmean_df = gmean_pairs.to_frame().reset_index()
gmean_df[['f1', 'f2']] = pd.DataFrame(gmean_df['index'].tolist(), index=gmean_df.index)
gmean_df.drop(columns=['index'], inplace=True)
gmean_df = gmean_df[gmean_df.columns[[1,2,0]]]
gmean_df.rename(columns={0: 'MFI'}, inplace=True)

'''
# Number of rows for each feature pair
>>> gmean_df.shape
(45449474, 3)
'''
gmean_df.index.name = 'id'
gmean_df.to_csv(OUTPUT, sep='\t')

