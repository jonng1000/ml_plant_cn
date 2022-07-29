# -*- coding: utf-8 -*-
"""
Created on Tue May 4 14:37:57 2021

@author: weixiong001

Use my ml script with multiple models, on multiple targes
Customised for non DGE and GO class labels. Modified from multi_go_wrapper.py
in ~/machine_learning/ml_go_workflow/cores_test
"""
import csv
import os
import sys

# GO target as class label
TARGET = sys.argv[1]
JOB_ID = sys.argv[2]
ML_FILE = 'ml_dataset_dc.txt'

# Class label, ressaignment of variable is done to make it compatible with
# downstream code
one_target = '"' +  TARGET + '"'
# Hack just to take care of this outlier when there's a space at the end of the feature name
if one_target == '"cin_ATB2/AtbZIP53/AtbZIP44/GBF5 BS in ProDH"':
    one_target = '"cin_ATB2/AtbZIP53/AtbZIP44/GBF5 BS in ProDH "'
# Various scripts are possible
os.system('python rf_contf_v2.py ' + one_target + ' ' + ML_FILE + ' ' + JOB_ID)
