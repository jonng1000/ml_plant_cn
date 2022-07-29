# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:08:17 2020

@author: weixiong001

Run RF model many times and extract the most important features.
No random features.
Edited from rf_feature_extract.py
"""
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from datetime import datetime

# For testing
# file = 'membrane_GO.txt'
# Input variables
file = sys.argv[1]
scores_file = file.split('_GO')[0] + '_scores.txt'
feat_impt_file = file.split('_GO')[0] + '_feat.txt'
#print(sys.argv)
print('input file:', file)
print('scores file:', scores_file)
print('feat_impt_file:', feat_impt_file)
