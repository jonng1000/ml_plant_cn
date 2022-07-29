# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:46:11 2020

@author: weixiong001

Gets all parent and child terms for each GO term and then removes it
from the dataset. GO term is defined from the command line

Used in ml workflow
"""

import pandas as pd
from goatools import obo_parser
from datetime import datetime
import os
import pickle
import gzip
import sys

def get_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


GO_FILE = './go.obo'
ML_FILE = './ml_dataset_dc.txt'  # For testing './test_dataset.txt'
feature = sys.argv[1]

go = obo_parser.GODag(GO_FILE)
df = pd.read_csv(ML_FILE, sep='\t', index_col=0)
    
go_terms_remove = []
term = feature.split('_')[1]
go_term = go[term]
children = go_term.get_all_children()
parents = go_term.get_all_parents()
for x in children:
    temp = 'go_' + x
    go_terms_remove.append(temp)
for x in parents:
    temp = 'go_' + x
    go_terms_remove.append(temp)
in_df_remove = df.columns.intersection(go_terms_remove)
new_df = df.drop(columns=in_df_remove)
name = feature.replace(':', '_')
output = name
# This is for pickle without compression
outfile = open(output,'wb')
pickle.dump(new_df,outfile)
outfile.close()
