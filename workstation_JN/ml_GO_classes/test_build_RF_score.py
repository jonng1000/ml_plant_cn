# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:14:17 2020

@author: weixiong001

Improved version of build_RF_score. Builds features block by block, with a RF model.
Includes both original and randomly shuffled features.
"""

import pandas as pd
import sys
# Can comment out these two lines if running on workstation
# Needed to ensure below module is imported correctly
#PATH = 'D:/GoogleDrive/machine_learning/RF_PNAS/feature_extract'
#sys.path.insert(1, PATH + '/build_top_features')
import rfe_module_v2 as rfm

print('Started script',rfm.get_time())
# Input variables and constants
FEAT_FILE = sys.argv[1]
ml_data = FEAT_FILE.split('_feat')[0] + '_GO.txt'
CLASS_LABELS = 'AraCyc annotation'
# Things I need to modify
RUNS = 1 # number of runs for ml model
# Output variables
# File to save scores
scores_file = FEAT_FILE.split('_feat')[0] + '_build_s.txt'

# Working code from here on, try not to modify it
# Get top X features
data = pd.read_csv(FEAT_FILE, sep="\t", index_col=0)
data['mean'] = data.mean(axis=1)
top = data.sort_values(by='mean', ascending=False)
top.reset_index(inplace=True)
top_feat = top['features']

df = rfm.read_df(ml_data)  # Will be replaced by another df value 
features = len(df.columns) - 1  # minus one is because of labels
# Codes positive class as 1 and negative class as 0
pos = df[CLASS_LABELS].value_counts().idxmin()
neg = df[CLASS_LABELS].value_counts().idxmax()
print(df[CLASS_LABELS].value_counts())
print('positive class:', pos)
print('negative class:', neg)
df.loc[:, 'AraCyc annotation'].replace([pos, neg], [1, 0], inplace=True)
print(df[CLASS_LABELS].value_counts())

# master list to iterate through, to buid RF model
master = [(10, 101, 10), (200, 1001, 100), (2000, 9001, 1000),
          (features, features + 1, 1)
          ]
c = 0
runs_s = []

print('input file:', FEAT_FILE)
print('ml data file:', ml_data)
print('scores file:', scores_file)
