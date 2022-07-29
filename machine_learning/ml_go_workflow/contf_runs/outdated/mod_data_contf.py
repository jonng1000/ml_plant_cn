# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:33:57 2021

@author: weixiong001

Replaces spaces with _, to ensure my ml workflow for continuous features proceeds
correctly, as their names have spaces
"""

import pandas as pd

FILE = 'ml_dataset_dc.txt'
OUTPUT = 'ml_dataset_mod.txt'

data = pd.read_csv(FILE, sep='\t', index_col=0)

data.columns = data.columns.str.replace(' ', '_', regex=False)

data.to_csv(OUTPUT, sep='\t')
