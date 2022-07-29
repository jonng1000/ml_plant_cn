# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:28:29 2021

@author: weixiong001

Permutation test on features to see if there's any significant
difference between two clusters of features observed from mutual rank
distribution
"""

import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

FILE = 'prop_ab.txt'
OUTPUT = 'p_values.txt'
NUM = 10000

df = pd.read_csv(FILE, sep='\t', index_col=0)

temp_df = df.copy()
comparison_lst = []
for i in range(NUM):
    temp_df['above_thresh'] = np.random.permutation(temp_df['above_thresh'])
    temp_df['below_thresh'] = np.random.permutation(temp_df['below_thresh'])
    temp_df['prop_diff'] = temp_df['above_thresh'] - temp_df['below_thresh']
    comparison = (df['prop_diff'] < temp_df['prop_diff']).copy()
    comparison_lst.append(comparison)

all_compare = pd.concat(comparison_lst, axis=1)
p_value = all_compare.sum(axis=1) / NUM
one_minus_p = 1 - p_value

p_df = pd.concat([p_value, one_minus_p], axis=1)
p_df.rename(columns={0: 'p_value_higher', 1: '1_minus_p_value_higher'}, inplace=True)
p_df.to_csv(OUTPUT, sep='\t')
