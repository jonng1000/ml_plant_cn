# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:46:11 2020

@author: weixiong001

Placeholder removal script, does not actually remove any class labels,
exists to ensure my ml workflow works for 2 types of runs:
1) all non-dge and GO class labels
2) all continuous

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
    
new_df = df  # To make this compatible w downstream code
output = feature  # To make this compatible w downstream code
# This is for pickle without compression
outfile = open(output,'wb')
pickle.dump(new_df,outfile)
outfile.close()
