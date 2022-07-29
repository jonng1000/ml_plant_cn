# -*- coding: utf-8 -*-
"""
Created on 140920

@author: weixiong
Combine sleuth output to created one hot encoded features of genes which are up
and down regulated across all experiments
"""

import pandas as pd
import numpy as np
import os

INPUT_FOLDER = 'sleuth_output'
OUTPUT_FILE = 'dge_1HE.txt'

lst_dfs = []
for a_file in os.listdir('./' + INPUT_FOLDER):
    file_path = './' + INPUT_FOLDER + '/' + a_file
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    if len(df.index) != len(df.index.unique()):
        print('Repeated transcript names in original file')
        break
    genes = {i:i.split('.')[0] for i in df.index}
    if len(genes) != len(genes.values()):
        print('Transcripts and genes have different numbers')
        break

    df = df.rename(index=genes)
    conditions_up = [
        (df['qval'] < 0.05) & (df['b'] > 0),
    ]
    conditions_down = [
        (df['qval'] < 0.05) & (df['b'] < 0),
    ]
    choices = [1]
    col_name = a_file.split('_sm')[0]
    df[col_name + '_up'] = np.select(conditions_up, choices, default=0)
    df[col_name + '_down'] = np.select(conditions_down, choices, default=0)
    one_HE = df.loc[:, [col_name + '_up', col_name + '_down']]
    lst_dfs.append(one_HE)
    
combined_1HE = pd.concat(lst_dfs, axis=1)
combined_1HE.index.name = 'Gene'
combined_1HE.to_csv(OUTPUT_FILE, sep='\t')
