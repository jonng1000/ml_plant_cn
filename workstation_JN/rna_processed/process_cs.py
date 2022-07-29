# -*- coding: utf-8 -*-
"""
Created on 140920

@author: weixiong
From dge_1HE.txt, replaced blanks with NA, and floats
with integers. More presentable according to Marek. Does other
data preprocessing
"""

import pandas as pd
import numpy as np

INPUT_FILE = 'dge_1HE.txt'
OUTPUT_FILE = 'dge_1HE_edited.txt'

df = pd.read_csv(INPUT_FILE, sep='\t', index_col=0)
'''
# ~26k genes and ~400 features (~200 conditions as each is split into up and
# down regulated, so features will be twice that)
>>> df.shape
(26440, 436)
'''
'''
# Dataframe only has 1, 0 nan
# Many nans appear initially as nan cannot be compared with each other
# directly
>>> np.unique(df.to_numpy())
array([ 0.,  1., nan, ..., nan, nan, nan])
>>> np.unique(df.to_numpy()[~np.isnan(df.to_numpy())])
array([0., 1.])
'''
df = df.fillna(0)
df = df.astype(int)

new_cols = {name: 'dge_' + name for name in df.columns}
df = df.rename(columns=new_cols)

df.to_csv(OUTPUT_FILE, sep='\t')

