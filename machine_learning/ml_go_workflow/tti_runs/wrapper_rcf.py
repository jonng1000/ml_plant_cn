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
ML_FILE = 'ml_dataset_dc.txt'

# Class label, ressaignment of variable is done to make it compatible with
# downstream code
one_target = TARGET
# Various scripts are possible
os.system('python rem_placeholder.py ' + one_target
          + ' ' + ML_FILE)
os.system('python rf_rcf.py ' + one_target)
file_remove = one_target  # To make it compatible with downstream code
os.remove(file_remove)
        
