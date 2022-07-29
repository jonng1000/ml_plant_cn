# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:28:29 2021

@author: weixiong001

Need to run on CN if input file takes up too much RAM
Explores mutual rank (MR) distribution  and obtain proportion 
of feature categories for high and low MR clusters
"""

import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

FILE = 'mutual_ranks.txt'
FIG = 'fi_mr_hist_ylog50b.png'
OUTPUT = 'prop_ab.txt'

# Cannot do the below due to some weird warnig
# FutureWarning: elementwise comparison failed; returning scalar instead,
# but in the future will perform
# elementwise comparison
# Something to do with a numpy and pandas clash, use the below workaround
#df = pd.read_csv(FILE, sep='\t', index_col=0)
df = pd.read_csv(FILE, sep='\t')
df.set_index(['id'], inplace=True)

g = sns.histplot(data=df, x='MR', bins=50, log_scale=(False, True))
g.figure.savefig(FIG)
plt.close()

sorted_df = df.sort_values(by=['MR'])

'''
# Exploring data
>>> len(df)
45449474
# Top 55 is too much
>>> len(df)/100*5
2 272 473.7
# Top 1%
>>> len(df)/100
454 494.74
'''

# From histogram, at around 5800, it spikes up, resulting in an uneven
# distribution
above = sorted_df.loc[sorted_df['MR'] >= 5800, :]
above_set = set(above['f1'].str.split('_').str[0]) | set(above['f2'].str.split('_').str[0])
below = sorted_df.loc[~(sorted_df['MR'] >= 5800), :]
below_set = set(below['f1'].str.split('_').str[0]) | set(below['f2'].str.split('_').str[0])
'''
# Exploring this
>>> below_set
{'ttf', 'tpm', 'gbm', 'pep', 'tmh', 'ort', 'pfa', 'go', 'con', 'cin', 'cid', 'dit', 'agi', 'phy', 'mob', 'ppi', 'spm', 'dge', 'gwa', 'ttr', 'tan', 'ntd', 'cif', 'dia', 'num', 'coe', 'sin', 'hom', 'agn', 'ptm', 'pid', 'twa'}
>>> above_set
{'ttf', 'tpm', 'gbm', 'pep', 'tmh', 'ort', 'pfa', 'go', 'con', 'cin', 'cid', 'dit', 'agi', 'phy', 'mob', 'ppi', 'spm', 'dge', 'gwa', 'ttr', 'tan', 'ntd', 'cif', 'dia', 'num', 'coe', 'sin', 'hom', 'agn', 'pid', 'ptm', 'twa'}
>>> above_set - below_set
set()
>>> below_set - above_set
set()
>>> above_set == below_set
True
'''

pref_f1_above = above['f1'].str.split('_').str[0]
pref_f2_above = above['f2'].str.split('_').str[0]
above_vc = pd.concat([pref_f1_above, pref_f2_above]).value_counts()
above_vcn = pd.concat([pref_f1_above, pref_f2_above]).value_counts(normalize=True)

pref_f1_below = below['f1'].str.split('_').str[0]
pref_f2_below = below['f2'].str.split('_').str[0]
below_vc = pd.concat([pref_f1_below, pref_f2_below]).value_counts()
below_vcn = pd.concat([pref_f1_below, pref_f2_below]).value_counts(normalize=True)

prop_ab = pd.concat([above_vcn, below_vcn], axis=1)
prop_ab.rename(columns={0: 'above_thresh', 1: 'below_thresh'}, inplace=True)
prop_ab['prop_diff'] = prop_ab['above_thresh'] - prop_ab['below_thresh']
prop_ab.index.name = 'feature_cat'
prop_ab.to_csv(OUTPUT, sep='\t')
