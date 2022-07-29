# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:46:11 2020

@author: weixiong001

Gets all parent and child terms for each GO term and then removes it,
together with itself, from the dataset. 
GO term is defined from the command line

Used in ml workflow, where training data only consists of all types of GO
terms from TAIR database
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
ML_FILE = sys.argv[2]
ML_TARGETS_FILE = './go_targets_dataset.txt'
feature = sys.argv[1].split('_')[1]

go = obo_parser.GODag(GO_FILE)
df = pd.read_csv(ML_FILE, sep='\t', index_col=0)
targets_df = pd.read_csv(ML_TARGETS_FILE, sep='\t', index_col=0)

go_terms_remove = []
go_term = go[feature]
children = go_term.get_all_children()
parents = go_term.get_all_parents()
for x in children:
    temp = 'go_' + x
    go_terms_remove.append(temp)
for x in parents:
    temp = 'go_' + x
    go_terms_remove.append(temp)
orig_term = 'go_' + feature
go_terms_remove.append(orig_term)
in_df_remove = df.columns.intersection(go_terms_remove)
new_df = df.drop(columns=in_df_remove)
new_df.insert(0, orig_term, targets_df.loc[:, orig_term])
# Need to do this as nan is created in the inserted column
new_df.fillna(0, inplace=True)
name = feature.replace(':', '_')
output = 'go_' + name
# This is for pickle without compression
outfile = open(output,'wb')
pickle.dump(new_df,outfile)
outfile.close()
