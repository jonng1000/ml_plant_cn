# -*- coding: utf-8 -*-
"""
Created on Tue May 4 14:37:57 2021

@author: weixiong001

Use my ml script with multiple models, on multiple targes
Customised for go terms. Modified from multi_go_wrapper.py
in ~/machine_learning/ml_go_workflow/cores_test
"""
import csv
import os
import sys

# GO target as class label
TARGET = sys.argv[1]
ML_FILE = 'go_features_interpro.txt'

# Renames the class so that it is compatible with
# downstream scripts
one_target = 'go_' + TARGET
# Various scripts are possible
os.system('python modify_GO_data.py ' + one_target
          + ' ' + ML_FILE)
os.system('python all_models_rs.py ' + one_target)
file_remove = one_target.replace(':', '_')
os.remove(file_remove)

