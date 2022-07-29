# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:46:11 2020

@author: weixiong001

For each DGE term, removes its correspond up or down term

Used in ml workflow
"""

import pandas as pd
from datetime import datetime
import os
import pickle
import gzip
import sys

ML_FILE = sys.argv[2]
feature = sys.argv[1]

df = pd.read_csv(ML_FILE, sep='\t', index_col=0)
    
terms_remove = []
term = feature.split('dge_')[1]  # To make this compatible w downstream code

if term.endswith('up'):
    dge_name = term.split('_up')[0]
    dge_remove = 'dge_' + dge_name + '_down'
elif term.endswith('down'):
    dge_name = term.split('_down')[0]
    dge_remove = 'dge_' + dge_name + '_up'
else:
    print('ERROR, CHECK', feature)
terms_remove.append(dge_remove)  # To make this compatible w downstream code

in_df_remove = df.columns.intersection(terms_remove)
new_df = df.drop(columns=in_df_remove)
output = feature  # To make this compatible w downstream code
# This is for pickle without compression
outfile = open(output,'wb')
pickle.dump(new_df,outfile)
outfile.close()
